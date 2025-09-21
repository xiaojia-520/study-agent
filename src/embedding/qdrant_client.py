from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import PointStruct
from config.settings import config
import logging

logger = logging.getLogger(__name__)


class QdrantManager:
    """Qdrant向量数据库管理"""

    def __init__(self):
        self.client = None
        self._initialize_client()
        self._ensure_collection()

    def _initialize_client(self):
        """初始化Qdrant客户端"""
        try:
            self.client = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT
            )
            logger.info("Qdrant客户端连接成功")
        except Exception as e:
            logger.error(f"Qdrant客户端连接失败: {e}")
            raise

    def _ensure_collection(self):
        """确保集合存在"""
        try:
            # 直接使用固定维度（bge-small-zh-v1.5的维度是512）
            DIMENSION = 512

            # 检查或创建集合
            try:
                self.client.get_collection(config.QDRANT_COLLECTION)
                logger.info(f"集合已存在: {config.QDRANT_COLLECTION}")
            except Exception:
                self.client.create_collection(
                    collection_name=config.QDRANT_COLLECTION,
                    vectors_config={
                        "text": qm.VectorParams(
                            size=DIMENSION,
                            distance=qm.Distance.COSINE
                        )
                    }
                )
                logger.info(f"创建集合: {config.QDRANT_COLLECTION}")

        except Exception as e:
            logger.error(f"集合初始化失败: {e}")
            # 如果失败，继续运行，可能在后续操作中会重新尝试

    def upsert_vector(self, point_id: str, vector: list, payload: dict):
        """插入或更新向量"""
        try:
            point = PointStruct(
                id=point_id,
                vector={"text": vector},
                payload=payload
            )
            self.client.upsert(
                collection_name=config.QDRANT_COLLECTION,
                points=[point]
            )
            logger.debug(f"向量插入成功: {point_id}")
        except Exception as e:
            logger.error(f"向量插入失败: {e}")