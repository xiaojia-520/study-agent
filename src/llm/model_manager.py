from langchain.chat_models import init_chat_model
from config.prompts import MODELS, CURRENT_MODEL
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器"""

    def __init__(self):
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        try:
            model_config = MODELS[CURRENT_MODEL]
            self.model = init_chat_model(
                model=model_config['MODEL_NAME'],
                model_provider=model_config['PROVIDER'],
                api_key=model_config['API_KEY'],
                base_url=model_config['BASE_URL']
            )
            logger.info(f"{CURRENT_MODEL} 模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def get_model(self):
        """获取模型实例"""
        if self.model is None:
            self._initialize_model()
        return self.model