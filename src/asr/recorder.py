import sounddevice as sd
import numpy as np
from typing import Optional, Callable
from config.settings import config
from ..utils.time_utils import get_current_time
import logging

logger = logging.getLogger(__name__)


class AudioRecorder:
    """音频录制器"""

    def __init__(self):
        self.stream = None
        self.is_recording = False

    def start_recording(self, vad_processor, asr_processor, embedding_manager, lesson_name: str):
        """开始录制和处理"""
        self.is_recording = True

        with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype='float32',
                device=config.DEVICE
        ) as stream:
            self._recording_loop(stream, vad_processor, asr_processor, embedding_manager, lesson_name)

    def _recording_loop(self, stream, vad_processor, asr_processor, embedding_manager, lesson_name: str):
        """录制循环"""
        from ..utils.file_utils import get_next_id, write_jsonl
        from ..utils.time_utils import format_time

        audio_chunk = []
        prev_audio_chunk = []
        speaking = False
        silence_start = get_current_time()
        speaking_start = None
        prev_chunk_stride = int(0.2 * config.SAMPLE_RATE / 512)

        LOG_FILE = f"data/outputs/json/{format_time(get_current_time(), '%Y-%m-%d_%H-%M-%S')}.jsonl"

        while self.is_recording:
            samples, overflowed = stream.read(512)
            if overflowed:
                logger.warning("缓冲区溢出，有音频数据丢失")

            samples = samples[:, 0]  # 转换为单声道

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
                if len(prev_audio_chunk) > prev_chunk_stride:
                    prev_audio_chunk.pop(0)

            if speech_dict and 'end' in speech_dict:
                speaking = False
                if speaking_start is not None:
                    silence_start = get_current_time()
                    logger.info(f"说话区间[{format_time(speaking_start)[-8:]}-{format_time(silence_start)[-8:]}]")

                self._process_audio_segment(
                    prev_audio_chunk, audio_chunk, speaking_start, silence_start,
                    asr_processor, embedding_manager, lesson_name, LOG_FILE
                )

                speaking_start = None
                audio_chunk = []

    def _process_audio_segment(self, prev_chunk, current_chunk, start_time, end_time,
                               asr_processor, embedding_manager, lesson_name, log_file):
        """处理音频片段"""
        from ..utils.file_utils import get_next_id, write_jsonl

        if not prev_chunk and not current_chunk:
            return

        audio = np.concatenate(prev_chunk + current_chunk)

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

    def stop_recording(self):
        """停止录制"""
        self.is_recording = False