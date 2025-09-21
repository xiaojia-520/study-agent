from silero_vad import load_silero_vad, VADIterator
from config.settings import config
import logging

logger = logging.getLogger(__name__)


class VADProcessor:
    """VAD处理器"""

    def __init__(self):
        self.vad = None
        self.vad_iter = None
        self._initialize_vad()

    def _initialize_vad(self):
        """初始化VAD"""
        try:
            self.vad = load_silero_vad()
            self.vad_iter = VADIterator(self.vad, threshold=0.3)
            logger.info("VAD模型加载成功")
        except Exception as e:
            logger.error(f"VAD模型加载失败: {e}")
            raise

    def process_samples(self, samples):
        """处理音频样本"""
        if self.vad_iter is None:
            self._initialize_vad()

        try:
            return self.vad_iter(samples)
        except Exception as e:
            logger.error(f"VAD处理失败: {e}")
            return None