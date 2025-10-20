from langchain.prompts import MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")


PROMPT_TEMPLATES = {
    "DEEPSEEK_CHAT": [
        ("system", "回答用户问题"),
        MessagesPlaceholder(variable_name="history"),
        (
            "user",
            "上30段实时文本:{text}\n用户的提问:{question}\n向量库检索结果:{embed_text}",
        ),
    ]
}

AUTO_QUESTION_REPLY_PROMPT_TEMPLATES = {
    "DEEPSEEK_CHAT": [
        ("system", "回答用户问题"),
        (
            "user",
            "我在上:{lesson}课程,老师的问题是{questions}"
        )

    ]
}

MODELS = {
    "DEEPSEEK-CHAT": {
        'MODEL_NAME': 'deepseek-chat',
        'API_KEY': f'{api_key}',
        'BASE_URL': 'https://api.deepseek-chat.com',
        'PROVIDER': 'deepseek',
    }
}

CURRENT_MODEL = "DEEPSEEK-CHAT"