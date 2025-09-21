from funasr import AutoModel
from config.settings import config
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ASRProcessor:
    """语音识别处理器"""

    def __init__(self):
        self.model = None
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
            result = self.model.inference(audio_data * 32768)
            text = "".join(item["text"].replace(" ", "") for item in result)
            print(f"识别结果:{text}")
            return text
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return ""