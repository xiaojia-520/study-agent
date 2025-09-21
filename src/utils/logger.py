import logging
import sys
from pathlib import Path
from config.settings import config


def setup_logging():
    """配置日志系统"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    handlers = []

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # 文件处理器（如果配置了日志文件）
    if config.LOG_FILE:
        log_path = Path(config.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper()),
        format=log_format,
        handlers=handlers
    )

    return logging.getLogger(__name__)