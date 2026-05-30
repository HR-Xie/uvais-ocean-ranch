"""LLM 工厂 — ChatOpenAI 单例 + 快速 OpenAI 客户端"""

import os
import httpx
from langchain_openai import ChatOpenAI
from openai import OpenAI

_chat_model = None
_fast_client = None

# httpx async client on Windows may fail SSL verification;
# use a custom client with trusted cert bundle
def _make_http_client():
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return httpx.Client(verify=ctx)


def _make_http_async_client():
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return httpx.AsyncClient(verify=ctx)


def get_chat_model() -> ChatOpenAI:
    global _chat_model
    if _chat_model is None:
        model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
        base_url = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
        api_key = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
        extra = {"thinking": {"type": "disabled"}} if "deepseek" in model.lower() else {}
        _chat_model = ChatOpenAI(
            model=model, base_url=base_url, api_key=api_key,
            streaming=True, max_retries=3, extra_body=extra,
            http_client=_make_http_client(),
            http_async_client=_make_http_async_client(),
        )
    return _chat_model


def get_fast_llm() -> OpenAI:
    global _fast_client
    if _fast_client is None:
        api_key = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
        api_base = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
        _fast_client = OpenAI(api_key=api_key, base_url=api_base)
    return _fast_client
