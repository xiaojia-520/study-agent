from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple

from config.prompts import PROMPT_TEMPLATES
from config.settings import config
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, QueryResponse

from src.embedding.embedding_manager import EmbeddingManager
from src.llm.model_manager import ModelManager

logger = logging.getLogger(__name__)


class RAGProcessor:
    """RAG处理器"""

    def __init__(self):
        self.model_manager = ModelManager()
        self.embedding_manager = EmbeddingManager()
        self.prompt_template = ChatPromptTemplate.from_messages(
            PROMPT_TEMPLATES["DEEPSEEK_CHAT"]
        )
        self.memories: Dict[str, ConversationBufferMemory] = {}

        # ------------------------------------------------------------------
        # Conversation helpers
        # ------------------------------------------------------------------

    def _memory_key(self, session_id: Optional[str]) -> str:
        return session_id or "__default__"

    def _get_memory(self, session_id: Optional[str]) -> ConversationBufferMemory:
        key = self._memory_key(session_id)
        if key not in self.memories:
            self.memories[key] = ConversationBufferMemory(return_messages=True)
        return self.memories[key]

    def reset_memory(self, session_id: Optional[str]) -> None:
        """重置指定会话的历史记录。"""

        key = self._memory_key(session_id)
        self.memories[key] = ConversationBufferMemory(return_messages=True)

    def _prepare_prompt(
            self, question: str, jsonl_path: Optional[str], session_id: Optional[str]
    ) -> Tuple[List[BaseMessage], ConversationBufferMemory]:
        context_results = self.search_context(question, limit=5, session_id=session_id)
        context_text = "\n".join(
            filter(None, [result.get("combined_text") for result in context_results])
        )
        logger.info("json源文本:%s", context_text)
        logger.info("jsonclear:%s", context_text)

        realtime_text = (
            self.jsonl_to_markdown(jsonl_path) if jsonl_path else "无实时文本"
        )
        logger.info("实时文本%s", realtime_text)

        memory = self._get_memory(session_id)
        history_messages: List[BaseMessage] = memory.load_memory_variables({}).get(
            "history", []
        )
        # 限制历史长度，避免上下文无限增长
        trimmed_history = history_messages[-8:]

        messages = self.prompt_template.format_messages(
            text=realtime_text,
            embed_text=context_text,
            question=question,
            history=trimmed_history,
        )

        return messages, memory

    def get_conversation_history(
            self, session_id: Optional[str], limit: int = 20
    ) -> List[Dict[str, str]]:
        memory = self._get_memory(session_id)
        history_messages = memory.load_memory_variables({}).get("history", [])
        pairs: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        for message in history_messages:
            if isinstance(message, HumanMessage):
                if current:
                    pairs.append(current)
                    current = {}
                current["question"] = message.content
            elif isinstance(message, AIMessage):
                if not current:
                    current = {}
                current["answer"] = message.content
                pairs.append(current)
                current = {}
        if current:
            pairs.append(current)
        return pairs[-limit:]

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

    def search_context(
            self, query: str, limit: int = 5, session_id: Optional[str] = None
    ) -> List[dict]:
        """搜索相关上下文"""
        try:
            query_vector = self.embedding_manager.embedding_model.embed_query(query)
            query_filter = None
            if session_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="session_id", match=MatchValue(value=f"{session_id}")
                        )
                    ]
                )
            client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            results: QueryResponse = client.query_points(
                collection_name=config.QDRANT_COLLECTION,
                query=query_vector,
                using="text",
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            )

            return [point.payload for point in results.points]
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
        for idx, item in enumerate(cleaned_items[-30:], 1):  # 只显示最后10条
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

    def generate_response(
            self,
            question: str,
            jsonl_path: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> str:
        """生成回答"""
        try:
            messages, memory = self._prepare_prompt(question, jsonl_path, session_id)
            model = self.model_manager.get_model()
            response = model.invoke(messages)
            answer = getattr(response, "content", str(response))
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            return answer

        except Exception as e:
            logger.error(f"回答生成失败: {e}")
            return "抱歉，生成回答时出现错误。"

    def generate_response_stream(
            self,
            question: str,
            jsonl_path: Optional[str] = None,
            session_id: Optional[str] = None,
    ) -> Iterator[str]:
        """生成流式回答，每次迭代返回一段文本。"""

        messages, memory = self._prepare_prompt(question, jsonl_path, session_id)
        model = self.model_manager.get_model()
        collected: List[str] = []
        try:
            for chunk in model.stream(messages):
                content = getattr(chunk, "content", None)
                if not content and hasattr(chunk, "delta"):
                    content = getattr(chunk, "delta", None)
                if not content:
                    additional = getattr(chunk, "additional_kwargs", {})
                    content = additional.get("content") if isinstance(additional, dict) else None
                if not content:
                    continue
                collected.append(content)
                yield content
        except Exception as exc:
            logger.error("流式生成回答失败: %s", exc, exc_info=True)
            raise

        answer = "".join(collected)
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)

    def get_history(
            self, session_id: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, str]]:
        """返回指定会话的问答历史。"""

        memory = self._get_memory(session_id)
        messages = memory.chat_memory.messages
        history: List[Dict[str, str]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})

        return history[-limit:]

