PROMPT_TEMPLATES = {
    "DEEPSEEK_CHAT": [
        ("system", "回答用户问题"),
        ("user", "实时文本:{text}\n用户的提问:{question}\n向量库检索结果:{embed_text}")
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