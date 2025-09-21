import datetime


def get_current_time() -> datetime.datetime:
    """获取当前时间"""
    return datetime.datetime.now()


def format_time(dt: datetime.datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化时间"""
    return dt.strftime(fmt)


def timestamp_to_datetime(timestamp: int) -> datetime.datetime:
    """时间戳转datetime"""
    return datetime.datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime.datetime) -> int:
    """datetime转时间戳"""
    return int(dt.timestamp())
