from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TranscriptBase(BaseModel):
    """转录基础模型"""
    text: str
    start_time: datetime
    end_time: datetime
    duration: float

class TranscriptItem(TranscriptBase):
    """单条转录条目"""
    id: int
    session_id: str
    inference_time: float
    type: str = "speech"

class BatchTranscript(TranscriptBase):
    """批量转录条目"""
    batch_index: int
    count: int
    combined_text: str
    type: str = "batch_speech"

class QdrantPoint(BaseModel):
    """Qdrant存储点模型"""
    id: str
    vector: list[float]
    payload: dict