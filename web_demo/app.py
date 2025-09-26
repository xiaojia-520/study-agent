"""Flask web interface for the realtime transcription and QA system."""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request

# Ensure the project root is available for imports when running the app directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.asr.asr_processor import ASRProcessor
from src.asr.punc_processor import PuncProcessor
from src.asr.recorder import AudioRecorder
from src.asr.vad_processor import VADProcessor
from src.llm.rag_processor import RAGProcessor
from src.utils.file_utils import ensure_directory, find_jsonl_file

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_clock(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value)).strftime("%H:%M:%S")
    except (OSError, OverflowError, ValueError):
        return None


@dataclass
class TranscriptRow:
    """A serialisable transcript row used by the UI."""

    identifier: int
    text: str
    start: Optional[int]
    end: Optional[int]
    duration: Optional[float]
    session_id: Optional[str]

    @property
    def time_range(self) -> str:
        start_clock = _format_clock(self.start)
        end_clock = _format_clock(self.end)
        if start_clock and end_clock:
            return f"{start_clock} – {end_clock}"
        if start_clock:
            return f"{start_clock} –"
        if end_clock:
            return f"– {end_clock}"
        return "-"

    @property
    def duration_label(self) -> str:
        if self.duration is None:
            return "-"
        return f"{self.duration:.2f}s"

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.identifier,
            "text": self.text,
            "time_range": self.time_range,
            "duration": self.duration,
            "duration_label": self.duration_label,
            "session_id": self.session_id,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "TranscriptRow":
        return cls(
            identifier=int(payload.get("id", 0)),
            text=str(payload.get("text", "")),
            start=_safe_int(payload.get("start")),
            end=_safe_int(payload.get("end")),
            duration=_safe_float(payload.get("dur")),
            session_id=payload.get("session_id"),
        )


class StudyAgentWebBridge:
    """Bridge the realtime transcription and QA pipeline into a web friendly API."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        ensure_directory(str(self.project_root / "data" / "outputs" / "json"))
        ensure_directory(str(self.project_root / "data" / "logs"))

        self.lock = threading.RLock()
        self.vad_processor = VADProcessor()
        self.asr_processor = ASRProcessor()
        self.punc_processor = PuncProcessor()
        self.asr_processor.punc_processor = self.punc_processor
        self.rag_processor = RAGProcessor()
        self.embedding_manager = self.rag_processor.embedding_manager

        self.recorder: Optional[AudioRecorder] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.recording_started_at: Optional[float] = None

        self.current_lesson: Optional[str] = None
        self.last_session_id: Optional[str] = None
        self.last_log_file: Optional[Path] = None

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        with self.lock:
            return bool(self.recorder and self.recorder.is_recording)

    def _resolve_path(self, path: Optional[str]) -> Optional[Path]:
        if not path:
            return None
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate

    def start_session(self, lesson: str) -> Tuple[bool, str]:
        lesson_name = lesson.strip()
        if not lesson_name:
            return False, "课程名称不能为空。"

        with self.lock:
            if self.recorder and self.recorder.is_recording:
                if self.current_lesson == lesson_name:
                    return True, f"课程“{lesson_name}”已经在录制。"
                return False, f"正在录制课程“{self.current_lesson}”，请先停止后再启动新的课程。"

            logger.info("开始课程 %s 的录制", lesson_name)
            recorder = AudioRecorder()
            self.recorder = recorder
            self.recording_started_at = time.time()
            self.current_lesson = lesson_name
            self.last_session_id = lesson_name
            self.last_log_file = None

            # 重置批处理状态，避免历史残留影响新课程。
            try:
                self.embedding_manager.batch_buffer.clear()
                self.embedding_manager.batch_index = 1
            except Exception:  # pragma: no cover - defensive, attributes 应始终存在
                pass

            thread = threading.Thread(
                target=recorder.start_recording,
                kwargs={
                    "vad_processor": self.vad_processor,
                    "asr_processor": self.asr_processor,
                    "embedding_manager": self.embedding_manager,
                    "lesson_name": lesson_name,
                },
                name=f"RecorderThread-{lesson_name}",
                daemon=True,
            )
            self.recording_thread = thread

        try:
            thread.start()
        except Exception as exc:  # pragma: no cover - thread start failure is rare
            logger.error("无法启动录音线程: %s", exc)
            with self.lock:
                self.recorder = None
                self.recording_thread = None
                self.recording_started_at = None
                self.current_lesson = None
            return False, "启动录制失败，请检查音频设备配置。"

        return True, f"课程“{lesson_name}”开始录制。"

    def stop_session(self) -> Tuple[bool, str]:
        with self.lock:
            if not self.recorder or not self.recorder.is_recording:
                return False, "当前没有正在录制的课程。"

            lesson_name = self.current_lesson
            logger.info("停止课程 %s 的录制", lesson_name)
            self.recorder.stop_recording()
            thread = self.recording_thread
        if thread:
            thread.join(timeout=2.0)

        with self.lock:
            log_path = self._resolve_path(self.recorder.log_file if self.recorder else None)
            if log_path and log_path.exists():
                self.last_log_file = log_path
            self.recorder = None
            self.recording_thread = None
            self.recording_started_at = None
            if lesson_name:
                self.last_session_id = lesson_name
            self.current_lesson = None

        lesson_label = lesson_name or "当前课程"
        return True, f"已停止课程“{lesson_label}”的录制。"

    def get_session_id(self) -> Optional[str]:
        with self.lock:
            return self.current_lesson or self.last_session_id

    def get_active_jsonl_path(self) -> Optional[Path]:
        """Return the most recent transcript file on disk."""

        latest_path = find_jsonl_file(str(self.project_root / "data" / "outputs" / "json"))
        if not latest_path:
            return None

        candidate = Path(latest_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate

        if candidate.exists():
            with self.lock:
                self.last_log_file = candidate
            return candidate

        return None

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------
    def get_recent_segments(self, limit: int = 50) -> List[TranscriptRow]:
        jsonl_path = self.get_active_jsonl_path()
        if not jsonl_path:
            return []

        buffer: deque[dict] = deque(maxlen=limit)
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    buffer.append(payload)
        except FileNotFoundError:
            return []
        except OSError as exc:
            logger.warning("读取转录文件失败: %s", exc)
            return []

        rows = []
        for payload in buffer:
            try:
                rows.append(TranscriptRow.from_payload(payload))
            except Exception:  # pragma: no cover - 防御, 单条解析失败不影响整体
                continue

        rows.sort(key=lambda item: item.identifier)
        return rows

    def answer_question(self, question: str) -> Tuple[bool, str]:
        cleaned = question.strip()
        if not cleaned:
            return False, "请输入想要咨询的问题。"

        jsonl_path = self.get_active_jsonl_path()
        if not jsonl_path:
            return False, "当前没有可用的转录内容，请先开始录制课程。"

        session_id = self.get_session_id()
        try:
            answer = self.rag_processor.generate_response(
                cleaned,
                str(jsonl_path),
                session_id=session_id,
            )
            return True, answer
        except Exception as exc:  # pragma: no cover - 统一兜底错误
            logger.error("生成回答失败: %s", exc, exc_info=True)
            return False, "生成回答时出现异常，请稍后再试。"

    def get_status(self) -> Dict[str, object]:
        with self.lock:
            log_path: Optional[Path] = None
            if self.recorder and self.recorder.log_file:
                log_path = self._resolve_path(self.recorder.log_file)
            elif self.last_log_file:
                log_path = self.last_log_file

            return {
                "recording": bool(self.recorder and self.recorder.is_recording),
                "lesson": self.current_lesson,
                "session_id": self.current_lesson or self.last_session_id,
                "log_file": str(log_path) if log_path else None,
                "started_at": (
                    datetime.fromtimestamp(self.recording_started_at).isoformat()
                    if self.recording_started_at
                    else None
                ),
            }


BRIDGE = StudyAgentWebBridge(PROJECT_ROOT)


# ----------------------------------------------------------------------
# Flask routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    segments = [row.to_dict() for row in BRIDGE.get_recent_segments(limit=20)]
    status = BRIDGE.get_status()
    return render_template("index.html", segments=segments, status=status)


@app.get("/api/status")
def api_status():
    return jsonify({"success": True, "status": BRIDGE.get_status()})


@app.post("/api/start")
def api_start():
    payload = request.get_json(silent=True) or {}
    lesson = str(payload.get("lesson", ""))
    success, message = BRIDGE.start_session(lesson)
    status = BRIDGE.get_status()
    http_status = 200 if success else 400
    return jsonify({"success": success, "message": message, "status": status}), http_status


@app.post("/api/stop")
def api_stop():
    success, message = BRIDGE.stop_session()
    status = BRIDGE.get_status()
    http_status = 200 if success else 400
    return jsonify({"success": success, "message": message, "status": status}), http_status


@app.get("/api/transcript")
def api_transcript():
    limit = request.args.get("limit", default=50, type=int)
    segments = [row.to_dict() for row in BRIDGE.get_recent_segments(limit=limit)]
    return jsonify({"success": True, "segments": segments})


@app.post("/api/ask")
def api_ask():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", ""))
    success, answer = BRIDGE.answer_question(question)
    status_code = 200 if success else 400
    return jsonify({"success": success, "answer": answer}), status_code


if __name__ == "__main__":  # pragma: no cover - script entry point
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=8000, debug=True)