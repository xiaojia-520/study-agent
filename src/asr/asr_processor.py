from typing import Optional
from funasr import AutoModel
from config.settings import config
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ASRProcessor:
    """语音识别处理器"""

    def __init__(self, hotwords):
        self.model = None
        self.punc_processor: Optional[object] = None
        self.hotword_str = " ".join(hotwords) if hotwords else None
        self._initialize_model()

    def _initialize_model(self):
        """初始化ASR模型"""
        try:
            self.model = AutoModel(
                model=config.ASR_MODEL_PATH,
                model_revision="v2.0.4",
                disable_update=True,
            )
            logger.info("ASR模型加载成功")
        except Exception as e:
            logger.error(f"ASR模型加载失败: {e}")
            raise

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """转录音频数据"""
        if self.model is None:
            self._initialize_model()
        try:
            result = self.model.inference(audio_data * 32768, hotword=self.hotword_str)
            text = "".join(item["text"].replace(" ", "") for item in result)
            if self.punc_processor and text:
                try:
                    text = self.punc_processor.add_punctuation(text)
                except Exception as punc_error:
                    logger.warning("标点处理失败，将使用原始文本: %s", punc_error)
            print(f"识别结果:{text}")
            return text
        except Exception as e:
            logger.error(f"语音识别失败: {e}")

            return ""
