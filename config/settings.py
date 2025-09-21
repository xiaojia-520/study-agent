import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent.absolute()


class AppConfig():
    """统一应用配置"""

    # 音频配置
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    DEVICE: int = 1
    VAD_CHUNK_SIZE: int = 200
    SILENCE_THRESHOLD: int = 0.1

    # 模型路径
    ASR_MODEL_PATH: str = (
                BASE_DIR / 'data/models/asr/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404'
                           '-pytorch')
    PUNC_MODEL_PATH: str = (BASE_DIR / 'data/models/punc/punc_ct-transformer_cn-en-common-vocab471067-large')
    EMBEDDING_MODEL_PATH = (BASE_DIR / 'data/models/embedding/bge-small-zh-v1.5')

    # Qdrant配置
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "asr"

    # 处理参数
    BATCH_SIZE: int = 20
    QUEUE_MAXSIZE: int = 500

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None

    class Config:
        env_file = ".env"
        env_prefix = "STUDY_AGENT_"


# 全局配置实例
config = AppConfig()
