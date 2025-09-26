PROMPT_TEMPLATES = {
    "DEEPSEEK_CHAT": [
        ("system", """你是一个课堂实时对话助手。根据用户的问题给予回复。"""),
        ("user", "上30段实时文本:{text}\n用户的提问:{question}\n向量库检索结果:{embed_text}")
    ]
}

MODELS = {
    "DEEPSEEK-CHAT": {
        'MODEL_NAME': 'deepseek-chat',
        'API_KEY': 'sk-fe9af4ecdf384b7a8a8d99567c3fca06',
        'BASE_URL': 'https://api.deepseek-chat.com',
        'PROVIDER': 'deepseek',
    }
}

CURRENT_MODEL = "DEEPSEEK-CHAT"