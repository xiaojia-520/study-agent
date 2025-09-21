from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from config.settings import config
import logging

logger = logging.getLogger(__name__)


class PuncProcessor:
    """标点处理器"""

    def __init__(self):
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化标点模型"""
        try:
            self.model = AutoModel(
                model=config.PUNC_MODEL_PATH,
                model_revision="v2.0.4",
                disable_update=True,
            )
            logger.info("标点模型加载成功")
        except Exception as e:
            logger.error(f"标点模型加载失败: {e}")
            raise

    def add_punctuation(self, text: str) -> str:
        """添加标点符号"""
        if self.model is None:
            self._initialize_model()

        try:
            punc_input = " ".join(list(text))
            punc_result = self.model.generate(input=punc_input)
            final_text = rich_transcription_postprocess(punc_result[0]['text'])
            return final_text
        except Exception as e:
            logger.error(f"标点处理失败: {e}")
            return text