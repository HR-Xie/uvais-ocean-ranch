"""RAG 查询引擎 + 对外 API

管线: Dense+BM25(RRF) → Reranker → LLM 生成
"""

import os

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.prompts import PromptTemplate
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.openai_like import OpenAILike

from rag.store import KnowledgeBase
from utils.config import milvus_conf, get_abs_path
from utils.prompts import load_rag_prompts
from utils.logging import logger


class RagEngine:

    def __init__(self, kb: KnowledgeBase):
        model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
        base_url = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
        api_key = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))

        Settings.llm = OpenAILike(
            model=model, api_base=base_url, api_key=api_key,
            max_retries=3, context_window=131072, is_chat_model=True,
        )

        index = kb.get_index()
        dense = index.as_retriever(similarity_top_k=milvus_conf["search"]["dense_top_k"])
        bm25 = BM25Retriever.from_defaults(
            nodes=kb.get_nodes(),
            similarity_top_k=milvus_conf["search"]["sparse_top_k"],
        )
        hybrid = QueryFusionRetriever(
            retrievers=[dense, bm25],
            similarity_top_k=milvus_conf["search"]["rrf_cutoff_k"],
            num_queries=1, mode="reciprocal_rerank",
        )

        reranker = SentenceTransformerRerank(
            model=get_abs_path(milvus_conf["reranker"]["model_path"]),
            top_n=milvus_conf["reranker"]["final_top_k"],
        )

        raw = load_rag_prompts()
        raw = raw.replace("{input}", "{query_str}").replace("{context}", "{context_str}")
        qa_tmpl = PromptTemplate(raw)

        self._engine = RetrieverQueryEngine.from_args(
            retriever=hybrid, node_postprocessors=[reranker], text_qa_template=qa_tmpl,
        )

        logger.info("RagEngine ready")

    def query(self, query_str: str, history: list[str] | None = None) -> str:
        logger.info(f"[query] {query_str[:60]}")
        try:
            return str(self._engine.query(query_str))
        except Exception as e:
            logger.error(f"[query] failed: {e}", exc_info=True)
            return "系统异常，暂时无法处理查询，请稍后再试。"


class RagService:

    def __init__(self):
        self._kb = KnowledgeBase()
        self._engine = RagEngine(self._kb)

    def rag_summarize(self, query: str) -> str:
        return self._engine.query(query)

    def build_knowledge_base(self) -> int:
        n = self._kb.build()
        if n > 0:
            self._engine = RagEngine(self._kb)
        return n

    def rebuild_knowledge_base(self) -> int:
        n = self._kb.rebuild()
        if n > 0:
            self._engine = RagEngine(self._kb)
        return n
