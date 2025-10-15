from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import config
from src.embedding.qdrant_client import QdrantManager
import threading
import queue
import logging
from uuid import uuid5, NAMESPACE_DNS
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """嵌入向量管理服务"""

    def __init__(self):
        self.embedding_model = None
        self.qdrant_manager = QdrantManager()
        self.task_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.batch_buffer = []
        self.batch_index = 1
        self._initialize_embedding_model()
        self._start_worker_thread()

    def _initialize_embedding_model(self):
        """初始化嵌入模型"""
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=str(config.EMBEDDING_MODEL_PATH),
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("嵌入模型加载成功")
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise

    def _start_worker_thread(self):
        """启动工作线程"""
        thread = threading.Thread(target=self._embedding_worker, daemon=True)
        thread.start()

    def _embedding_worker(self):
        """嵌入处理工作线程"""
        while True:
            try:
                item = self.task_queue.get()
                if item is None:
                    break

                self._process_embedding_item(item)
                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"嵌入处理错误: {e}")

    def _process_embedding_item(self, item):
        """处理单个嵌入项"""
        text, payload, session_id, id_val = item
        try:
            vector = self.embedding_model.embed_documents([text])[0]
            pid = str(uuid5(NAMESPACE_DNS, f"{session_id}-{id_val}"))
            self.qdrant_manager.upsert_vector(pid, vector, payload)
        except Exception as e:
            logger.error(f"向量化出错: {e}")

    def enqueue_for_embedding(self, text: str, payload: dict, session_id: str, id_val: int):
        """将任务加入队列"""
        try:
            # 添加到批量缓冲
            self.batch_buffer.append({
                "id": id_val,
                "text": text,
                "payload": payload,
            })

            # 如果达到批量大小，处理批量数据
            if len(self.batch_buffer) >= config.BATCH_SIZE:
                self._process_batch(session_id)

            # 同时处理单个条目
            self.task_queue.put_nowait((text, payload, session_id, id_val))

        except queue.Full:
            logger.warning("队列已满，丢弃任务")
        except Exception as e:
            logger.error(f"入队失败: {e}")

    def _process_batch(self, session_id: str):
        """处理批量数据"""
        if not self.batch_buffer:
            return

        try:
            # 合并文本
            combined_text = "".join(item["text"] for item in self.batch_buffer)

            # 计算时间范围
            starts_ts = [item["payload"].get("start") for item in self.batch_buffer]
            ends_ts = [item["payload"].get("end") for item in self.batch_buffer]

            if starts_ts and ends_ts:
                first_start_ts = int(min(starts_ts))
                last_end_ts = int(max(ends_ts))
                batch_dur = round(last_end_ts - first_start_ts, 2)
            else:
                first_start_ts = last_end_ts = None
                batch_dur = 0.0

            # 创建批量payload
            batch_payload = {
                "type": "batch_speech",
                "session_id": session_id,
                "batch_index": self.batch_index,
                "count": len(self.batch_buffer),
                "start": first_start_ts,
                "end": last_end_ts,
                "dur": batch_dur,
                "combined_text": combined_text,
            }

            # 生成批量ID并入队
            pid_batch = str(uuid5(NAMESPACE_DNS, f"{session_id}-batch-{self.batch_index}"))
            self.task_queue.put_nowait((combined_text, batch_payload, session_id, f"batch-{self.batch_index}"))

            # 清空缓冲并增加批次号
            self.batch_buffer.clear()
            self.batch_index += 1

        except Exception as e:
            logger.error(f"批量处理失败: {e}")