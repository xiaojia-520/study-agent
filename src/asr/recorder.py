import sounddevice as sd
import numpy as np
from typing import Optional, Callable
from config.settings import config
import logging
from pathlib import Path
from ..utils.file_utils import BASE_DIR, ensure_directory, get_next_id, write_jsonl
from ..utils.time_utils import get_current_time, format_time
from collections import deque

logger = logging.getLogger(__name__)


class AudioRecorder:
    """音频录制器"""

    def __init__(self):
        self.stream = None
        self.is_recording = False
        self.json_file: Optional[str] = None
        self.lesson_name: Optional[str] = None

    def start_recording(self, vad_processor, asr_processor, embedding_manager, lesson_name: str):
        """开始录制和处理"""
        self.is_recording = True
        self.lesson_name = lesson_name

        with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype='float32',
                device=config.DEVICE,
                blocksize=512
        ) as stream:
            self._recording_loop(stream, vad_processor, asr_processor, embedding_manager, lesson_name)

    def _recording_loop(self, stream, vad_processor, asr_processor, embedding_manager, lesson_name: str):
        """录制循环"""

        audio_chunk = deque()
        prev_chunk_stride = int(0.2 * config.SAMPLE_RATE / 512)
        prev_audio_chunk = deque(maxlen=prev_chunk_stride)
        speaking = False
        silence_start = get_current_time()
        speaking_start = None

        json_dir = Path(BASE_DIR) / "data" / "outputs" / "json"
        ensure_directory(str(json_dir))
        json_file_path = json_dir / f"{format_time(get_current_time(), '%Y-%m-%d_%H-%M-%S')}.jsonl"
        self.json_file = str(json_file_path)

        while self.is_recording:
            samples, overflowed = stream.read(512)
            if overflowed:
                logger.warning("缓冲区溢出，有音频数据丢失")

            samples = samples[:, 0]  # 转换为单声道

            speech_dict = vad_processor.process_samples(samples)

            if speech_dict and 'start' in speech_dict:
                if not speaking:
                    speaking = True
                    audio_chunk.clear()
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
                    self.json_file,
                )

                speaking_start = None
                audio_chunk.clear()

    def _process_audio_segment(self, prev_chunk, current_chunk, start_time, end_time,
                               asr_processor, embedding_manager, lesson_name, json_file):
        """处理音频片段"""

        if not prev_chunk and not current_chunk:
            return

        if start_time is None or end_time is None:
            logger.warning("跳过一段：起止时间缺失")
            return

        bufs = list(prev_chunk) + list(current_chunk)
        if not bufs:
            return
        audio = np.concatenate(bufs)
        audio = np.ascontiguousarray(audio)

        try:
            # 语音识别
            text = asr_processor.transcribe_audio(audio, lesson_name)
            logger.info(f"识别成功")

            # 准备数据
            id_val = get_next_id(json_file)
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
            write_jsonl(json_file, json_data)

            # 加入嵌入队列
            embedding_manager.enqueue_for_embedding(text, json_data, lesson_name, json_file)

        except Exception as e:
            logger.error(f"音频处理失败: {e}")

    def stop_recording(self):
        """停止录制"""
        self.is_recording = False
