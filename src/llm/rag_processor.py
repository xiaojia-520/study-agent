from langchain.prompts import ChatPromptTemplate
from config.prompts import PROMPT_TEMPLATES
from src.llm.model_manager import ModelManager
from src.embedding.embedding_manager import EmbeddingManager
from src.utils.file_utils import load_json
import re
import json
from datetime import datetime
import logging
from qdrant_client.models import Filter, FieldCondition, MatchValue



logger = logging.getLogger(__name__)


class RAGProcessor:
    """RAG处理器"""

    def __init__(self):
        self.model_manager = ModelManager()
        self.embedding_manager = EmbeddingManager()
        self.prompt_template = ChatPromptTemplate.from_messages(PROMPT_TEMPLATES["DEEPSEEK_CHAT"])

    def _read_jsonl(self, path):
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return rows

    def search_context(self, query: str, limit: int = 5, session_id: str = None):
        """搜索相关上下文"""
        try:
            query_vector = self.embedding_manager.embedding_model.embed_query(query)
            from qdrant_client import QdrantClient
            from config.settings import config
            query_filter = Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=f"{session_id}"))]
            )

            client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            results = client.search(
                collection_name=config.QDRANT_COLLECTION,
                query_vector=("text", query_vector),  # ← 指定向量名
                limit=limit,
                with_payload=True,
                query_filter=query_filter
            )

            return [result.payload for result in results]
        except Exception as e:
            logger.error(f"上下文搜索失败: {e}")
            return []

    def clean_jsonl_content(self, items):
        """清理JSONL内容"""
        FILTER_WORDS = ["啊", "嗯", "这个", "然后", "呃", "吧", "嘛", "哈"]
        alt = "|".join(re.escape(w) for w in FILTER_WORDS)
        pattern = re.compile(r"\s*(?:" + alt + r")\s*")

        cleaned_items = []
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue

            cleaned = pattern.sub(" ", text)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

            if cleaned:
                item["text"] = cleaned
                cleaned_items.append(item)

        return cleaned_items

    def jsonl_to_markdown(self, jsonl_path: str):
        """JSONL转Markdown表格"""
        items = self._read_jsonl(jsonl_path)
        cleaned_items = self.clean_jsonl_content(items)

        md_table = ["| id | 时间区间 | 文本 |", "|---|---|---|"]
        for idx, item in enumerate(cleaned_items[-10:], 1):  # 只显示最后10条
            start_ts = item.get("start")
            end_ts = item.get("end")

            if start_ts and end_ts:
                start_str = datetime.fromtimestamp(start_ts).strftime("%H:%M:%S")
                end_str = datetime.fromtimestamp(end_ts).strftime("%H:%M:%S")
                time_range = f"{start_str}–{end_str}"
            else:
                time_range = "N/A"

            text = item.get("text", "").replace("|", "\\|")
            md_table.append(f"| {idx} | {time_range} | {text} |")

        return "\n".join(md_table)

    def generate_response(self, question: str, jsonl_path: str = None, session_id: str = None):
        """生成回答"""
        try:
            # 搜索相关上下文
            context_results = self.search_context(question, limit=5, session_id=session_id)
            context_text = "\n".join([result.get("combined_text", "") for result in context_results])

            # 获取实时文本
            realtime_text = self.jsonl_to_markdown(jsonl_path) if jsonl_path else "无实时文本"

            # 构建提示词
            chain = self.prompt_template | self.model_manager.get_model()
            response = chain.invoke({
                "text": realtime_text,
                "embed_text": context_text,
                "question": question
            })

            return response.content

        except Exception as e:
            logger.error(f"回答生成失败: {e}")
            return "抱歉，生成回答时出现错误。"
