from sentence_transformers import CrossEncoder

from utils.config_handler import milvus_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class RerankerService:
    """
    BGE-Reranker Cross-Encoder 精排服务
    对混合检索召回的候选文档进行逐对语义打分，返回 Top-K
    """

    def __init__(self):
        model_path = get_abs_path(milvus_conf["reranker"]["model_path"])
        logger.info(f"正在加载 BGE-Reranker 精排模型 (路径: {model_path})...")
        self.model = CrossEncoder(
            model_path,
            max_length=512,
        )
        self.top_k = milvus_conf["reranker"]["final_top_k"]
        logger.info(f"精排服务初始化完成 (final_top_k={self.top_k})")

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        对候选文档精排

        Args:
            query: 用户查询
            candidates: 候选文档列表 [{"text": ..., "source": ..., "score": ...}, ...]

        Returns:
            精排后的 Top-K 文档列表
        """
        if not candidates:
            return []

        if len(candidates) <= self.top_k:
            logger.info(f"[精排] 候选数({len(candidates)})不足 Top-K({self.top_k})，跳过精排")
            return candidates

        logger.info(f"[精排] 正在对 {len(candidates)} 条候选文档逐对打分...")
        pairs = [(query, doc["text"]) for doc in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        result = candidates[:self.top_k]

        logger.info(f"[精排] 完成，从 {len(candidates)} 条中返回 Top-{self.top_k}")
        for i, doc in enumerate(result):
            logger.debug(f"[精排] #{i+1} | score={doc['rerank_score']:.4f} | source={doc['source']}")

        return result
