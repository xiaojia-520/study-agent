import os
from pathlib import Path
from typing import Optional
from src.utils.time_utils import get_current_time, format_time

BASE_DIR = Path(__file__).parent.parent.absolute()


class AppConfig():
    """统一应用配置"""

    # 音频配置
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    DEVICE: int = 3
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
    LOG_FILE: Optional[str] = BASE_DIR / 'data' / 'logs' / f'{format_time(get_current_time())}.log'

    class Config:
        env_file = ".env"
        env_prefix = "STUDY_AGENT_"

    # 热词参数
    HOTWORDS = {
        "流体力学": "流体力学 流体 粘度 压力 静压 动压 伯努利方程 连续方程 雷诺数 层流 湍流 边界层 流线 管道流 空气动力学",
        "概率论": "概率论 数理统计 随机变量 概率分布 二项分布 正态分布 泊松分布 独立性 方差 协方差 数学期望 极大似然估计 假设检验 置信区间",
        "工程热力学": "工程热力学 热力学第一定律 热力学第二定律 熵 内能 焓 卡诺循环 等温过程 等压过程 等容过程 绝热过程 热机效率 热功转换 热量守恒",
        "马克思主义基本原理": "马克思主义 马原 历史唯物主义 辩证唯物主义 阶级斗争 生产关系 生产力 剩余价值 资本主义 社会主义 共产主义 马克思 恩格斯 历史发展规律",
        "普通物理": "普通物理 普物 普通物理B 普物B 力学 牛顿定律 万有引力 功和能量 动量守恒 振动 波动 电场 电势 磁场 电磁感应 光学 干涉 衍射",
        "工程力学": "工程力学 工程力学B 工力 工力B 静力学 动力学 受力分析 力矩 支座反力 平衡条件 杠杆原理 应力 应变 剪力 弯矩 扭转 弹性模量",


    }


# 全局配置实例
config = AppConfig()
