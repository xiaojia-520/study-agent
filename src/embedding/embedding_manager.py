from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import config
from src.embedding.qdrant_client import QdrantManager
import threading
import queue
import logging
from uuid import uuid5, NAMESPACE_DNS

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """嵌入向量管理服务"""

    def __init__(self):
        self.embedding_model = None
        self.qdrant_manager = QdrantManager()
        self.task_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.batch_buffer = []
        self.batch_index = 1
        self.worker_thread = None
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
        self.worker_thread = threading.Thread(target=self._embedding_worker, daemon=True)
        self.worker_thread.start()

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
        logger.info("嵌入")
        """处理单个嵌入项"""
        text, payload, id_val = item
        logger.info(f"id:{id_val}")
        try:
            vector = self.embedding_model.embed_documents([text])[0]
            pid = str(uuid5(NAMESPACE_DNS, f"{payload['session_id']}-{id_val}"))
            self.qdrant_manager.upsert_vector(pid, vector, payload)
        except Exception as e:
            logger.error(f"向量化出错: {e}")


    # 入口
    def enqueue_for_embedding(self, text: str, payload: dict, session_id: str, id_val: str):
        """将任务加入队列"""
        try:
            logger.info(id_val)
            # 添加到批量缓冲
            self.batch_buffer.append({
                "id": id_val,
                "text": text,
                "payload": payload,
            })

            # 如果达到批量大小，处理批量数据
            if len(self.batch_buffer) >= config.BATCH_SIZE:
                self._process_batch(session_id, id_val)

        except Exception as e:
            logger.error(f"入队失败: {e}")

    def _process_batch(self, session_id: str, id_val: str):
        """处理批量数据"""
        if not self.batch_buffer:
            return

        try:
            # 合并文本
            combined_text = "".join(item["text"] for item in self.batch_buffer)

            # 计算时间范围
            starts_ts = [item["payload"].get("start") for item in self.batch_buffer if
                         item["payload"].get("start") is not None]
            ends_ts = [item["payload"].get("end") for item in self.batch_buffer if
                       item["payload"].get("end") is not None]

            if starts_ts and ends_ts:
                first_start_ts = int(min(starts_ts))
                last_end_ts = int(max(ends_ts))
                batch_dur = round(last_end_ts - first_start_ts, 2)
            else:
                first_start_ts = last_end_ts = None
                batch_dur = 0.0

            # 创建批量payload
            batch_payload = {
                "batch_index": id_val,
                "type": "speech",
                "session_id": session_id,
                "count": len(self.batch_buffer),
                "start": first_start_ts,
                "end": last_end_ts,
                "dur": batch_dur,
                "combined_text": combined_text,
            }

            # 生成批量ID并入队
            self.task_queue.put_nowait((combined_text, batch_payload, id_val))
            logger.info("入队")
            # 清空缓冲并增加批次号
            self.batch_buffer.clear()
            self.batch_index += 1

        except Exception as e:
            logger.error(f"批量处理失败: {e}")

    def flush_batch(self):
        if not self.batch_buffer:
            return
        try:
            n_before = len(self.batch_buffer)
            session_id = self.batch_buffer[0]["payload"].get("session_id", "default")
            id_val = self.batch_buffer[0]["id"]
            self._process_batch(session_id, id_val)
            logger.info(f"Flush 尾批（条数: {n_before}）")
        except Exception as e:
            logger.error(f"Flush 失败: {e}")

    def close(self):
        """优雅关闭：flush 尾批；等待队列清空；结束工作线程"""
        try:
            # 1) flush 尾批（若有）
            self.flush_batch()

            # 2) 等待队列里的任务全部处理完（依赖 worker 每次处理后调用 task_done）
            try:
                self.task_queue.join()
            except Exception:
                pass

            # 3) 发结束信号并等待线程退出
            self.task_queue.put_nowait(None)
            if getattr(self, "worker_thread", None) is not None:
                self.worker_thread.join(timeout=10)
        except Exception as e:
            logger.error(f"关闭管理器失败: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # 退出 with 块时自动 flush + close
            self.close()
        except Exception as e:
            logger.error(f"退出时关闭 embedding_manager 失败: {e}")