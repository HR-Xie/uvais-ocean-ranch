import os
from langchain_openai import ChatOpenAI
from utils.config_handler import rag_conf


class BaseModelFactory:
    def __init__(self):
        model = os.getenv("LLM_MODEL_NAME", rag_conf.get("chat_model_name", "deepseek-chat"))
        base_url = os.getenv("LLM_API_BASE", rag_conf.get("chat_api_base", "https://api.deepseek.com/v1"))
        api_key = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))

        self.chat_model = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            streaming=False,
            max_retries=3,
        )


_chat_model = None

def get_chat_model():
    """Lazy singleton — avoids API key validation at import time."""
    global _chat_model
    if _chat_model is None:
        _chat_model = BaseModelFactory().chat_model
    return _chat_model
