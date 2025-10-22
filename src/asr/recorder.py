import sounddevice as sd
import numpy as np
from typing import Optional, Callable
from config.settings import config
import logging
from pathlib import Path
from ..utils.file_utils import BASE_DIR, ensure_directory, get_next_id, write_jsonl
from ..utils.time_utils import get_current_time, format_time
from collections import deque
import threading
from queue import Queue, Empty, Full
from src.asr.vad_processor import VADProcessor

logger = logging.getLogger(__name__)


class AudioRecorder:
    """音频录制器"""

    def __init__(self):
        self.stream = None
        self.is_recording = False
        self.log_file: Optional[str] = None
        self.lesson_name: Optional[str] = None
        self._cb_queue = Queue(maxsize=256)
        self._overflow_warned = False

    def start_recording(self, vad_processor, asr_processor, embedding_manager, lesson_name: str):
        """开始录制和处理"""
        self.is_recording = True
        self.lesson_name = lesson_name
        if vad_processor is None:
            vad_processor = VADProcessor()

        with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype='float32',
                device=config.DEVICE,
                blocksize=512,
                callback=self._audio_callback,
        ) as stream:
            self._recording_loop(stream, vad_processor, asr_processor, embedding_manager, lesson_name)

    def _recording_loop(self, stream, vad_processor, asr_processor, embedding_manager, lesson_name: str):
        """录制循环"""

        audio_chunk = []
        speaking = False
        silence_start = get_current_time()
        speaking_start = None
        prev_chunk_stride = int(0.2 * config.SAMPLE_RATE / 512)
        prev_audio_chunk = deque(maxlen=prev_chunk_stride)

        log_dir = Path(BASE_DIR) / "data" / "outputs" / "json"
        ensure_directory(str(log_dir))
        log_file_path = log_dir / f"{format_time(get_current_time(), '%Y-%m-%d_%H-%M-%S')}.jsonl"
        self.log_file = str(log_file_path)

        while self.is_recording:
            samples = None
            # 等队列有数据；不给死等，避免无法停机
            try:
                samples = self._cb_queue.get(timeout=0.01)  # 最多等10ms
            except Empty:
                continue

            speech_dict = vad_processor.process_samples(samples)

            if speech_dict and 'start' in speech_dict:
                if not speaking:
                    speaking = True
                    audio_chunk = []
                    if silence_start is not None:
                        speaking_start = get_current_time()
                        logger.info(f"静音区间[{format_time(silence_start)[-8:]}-{format_time(speaking_start)[-8:]}]")
                        silence_start = None

            if speaking:
                audio_chunk.append(samples)
            else:
                prev_audio_chunk.append(samples)

            if speech_dict and 'end' in speech_dict:
                speaking = False
                if speaking_start is not None:
                    silence_start = get_current_time()
                    logger.info(f"说话区间[{format_time(speaking_start)[-8:]}-{format_time(silence_start)[-8:]}]")

                self._process_audio_segment(
                    prev_audio_chunk,
                    audio_chunk,
                    speaking_start,
                    silence_start,
                    asr_processor,
                    embedding_manager,
                    lesson_name,
                    self.log_file,
                )

                speaking_start = None
                audio_chunk = []

    def _process_audio_segment(self, prev_chunk, current_chunk, start_time, end_time,
                               asr_processor, embedding_manager, lesson_name, log_file):
        """处理音频片段"""

        if start_time is None or end_time is None:
            return

        if not prev_chunk and not current_chunk:
            return

        audio = np.concatenate([*list(prev_chunk), *current_chunk], axis=0)

        try:
            # 语音识别
            text = asr_processor.transcribe_audio(audio)
            logger.info(f"识别结果: {text}")

            # 准备数据
            id_val = get_next_id(log_file)
            duration = round((end_time - start_time).total_seconds(), 2)

            json_data = {
                "id": id_val,
                "session_id": lesson_name,
                "type": "speech",
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "start_str": start_time.isoformat(),
                "end_str": end_time.isoformat(),
                "text": text,
                "dur": duration,
            }

            # 写入文件
            write_jsonl(log_file, json_data)

            # 加入嵌入队列
            embedding_manager.enqueue_for_embedding(text, json_data, lesson_name, id_val)

        except Exception as e:
            logger.error(f"音频处理失败: {e}")

    def _audio_callback(self, indata, frames, ctime, status):
        if status and status.input_overflow and not self._overflow_warned:
            logger.warning("检测到缓冲区溢出（同类警告仅提示一次）")
            self._overflow_warned = True
        if indata.ndim == 2:
            indata = indata[:, 0]

        try:
            self._cb_queue.put_nowait(indata.copy())
        except Full:
            try:
                self._cb_queue.get_nowait()
                self._cb_queue.put_nowait(indata.copy())
            except Exception:
                pass

    def stop_recording(self):
        """停止录制"""
        self.is_recording = False
