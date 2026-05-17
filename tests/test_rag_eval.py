"""
RAG 检索质量评估
===============
评估 Milvus 混合检索（稠密 BGE-M3 + 稀疏 BM25）+ BGE-Reranker 精排的召回效果。

运行前需确保:
  1. Docker 里 Milvus 已启动 (docker-compose up -d)
  2. 知识库已构建 (python -c "from rag.milvus_service import MilvusService; MilvusService().build_knowledge_base()")

运行:
  pytest tests/test_rag_eval.py -v -s
  python tests/test_rag_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ── 加载评估数据集 ─────────────────────────────────────────────────
def _load_eval_queries():
    path = Path(__file__).parent / "data" / "rag_eval_queries.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 真实检索（Milvus 混合检索 + BGE-Reranker）─────────────────────
_milvus = None
_reranker = None

def _get_retriever():
    global _milvus, _reranker
    if _milvus is None:
        from rag.milvus_service import MilvusService
        _milvus = MilvusService()
    if _reranker is None:
        from rag.reranker_service import RerankerService
        _reranker = RerankerService()
    return _milvus, _reranker

def _retrieve(query, top_k=10):
    milvus, reranker = _get_retriever()
    candidates = milvus.hybrid_search(query, top_k=top_k * 2)
    if not candidates:
        return []
    top_docs = reranker.rerank(query, candidates)[:top_k]
    return [doc["source"] for doc in top_docs]


# ── Milvus 检测 ────────────────────────────────────────────────────
def _check_kb():
    """返回 (ready: bool, entity_count: int)"""
    try:
        milvus, _ = _get_retriever()
        stats = milvus.client.get_collection_stats(milvus.collection_name)
        return True, stats["row_count"]
    except Exception:
        return False, -1


# ── 指标计算 ───────────────────────────────────────────────────────
def _compute_metrics(results, k_values=(1, 3, 5)):
    """
    results: [{"retrieved_sources": [str, ...], "expected_source": str}, ...]

    指标说明:
      Recall@K   — 正确答案出现在 Top-K 结果中的查询占比
      Precision@K — Top-K 结果中相关结果的平均占比
      MRR        — 第一个相关结果排名的倒数，全查询平均
      HitRate@K  — Top-K 至少命中一条的查询占比
    """
    n = len(results)
    if n == 0:
        return {}

    metrics = {}

    # MRR
    mrr_sum = 0.0
    for r in results:
        positions = [i for i, s in enumerate(r["retrieved_sources"]) if s == r["expected_source"]]
        mrr_sum += 1.0 / (positions[0] + 1) if positions else 0.0
    metrics["MRR"] = mrr_sum / n

    for k in k_values:
        recall = precision = hit = 0.0
        for r in results:
            top_k_sources = r["retrieved_sources"][:k]
            relevant = sum(1 for s in top_k_sources if s == r["expected_source"])
            recall += 1.0 if relevant > 0 else 0.0
            precision += relevant / k
            if relevant > 0:
                hit += 1.0
        metrics[f"Recall@{k}"] = recall / n
        metrics[f"Precision@{k}"] = precision / n
        metrics[f"HitRate@{k}"] = hit / n

    return metrics


# ── 报告 ───────────────────────────────────────────────────────────
def _print_report(metrics, n_queries, entity_count, results):
    print("\n" + "=" * 58)
    print("  RAG 检索质量评估报告")
    print("=" * 58)
    print(f"  评估查询数: {n_queries}")
    print(f"  检索方式:   Milvus 混合检索 (BGE-M3 稠密 + Sparse BM25)")
    print(f"  精排模型:   BGE-Reranker (Cross-Encoder)")
    print(f"  知识库容量: {entity_count} 条向量")
    print("-" * 58)
    print(f"  {'指标':<15s} {'得分':>6s}  {'分布':>25s}")
    print("-" * 58)

    for name in ["Recall@1", "Recall@3", "Recall@5", "Precision@3", "Precision@5", "MRR", "HitRate@1", "HitRate@3"]:
        if name in metrics:
            v = metrics[name]
            bar = "█" * int(v * 25)
            print(f"  {name:<15s} {v:6.1%}  {bar}")

    print("-" * 58)
    print("  逐查询详情:")
    for i, r in enumerate(results):
        rank = None
        if r["expected_source"] in r["retrieved_sources"]:
            rank = r["retrieved_sources"].index(r["expected_source"]) + 1
        status = f"HIT #{rank}" if rank else "MISS"
        print(f"  [{i+1:2d}] {r['query'][:32]:<34s} -> {status}")
        if not rank and len(r["retrieved_sources"]) >= 3:
            top3 = [s[:50] for s in r["retrieved_sources"][:3]]
            print(f"       expected: {r['expected_source']}")
            print(f"       got:      {top3}")
    print("=" * 58 + "\n")


# ═══════════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════════

class TestRAGEvaluation:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.queries = _load_eval_queries()
        self.kb_ready, self.entity_count = _check_kb()

    def test_retrieval_quality(self):
        """核心评估：10 条查询，输出 Recall@K / MRR / Precision@K"""
        if not self.kb_ready:
            pytest.skip("Milvus 连接失败，请先 docker-compose up -d")
        if self.entity_count == 0:
            pytest.skip(
                "知识库为空，请先构建: "
                "python -c \"from rag.milvus_service import MilvusService; "
                "MilvusService().build_knowledge_base()\""
            )

        results = []
        for item in self.queries:
            sources = _retrieve(item["query"])
            results.append({
                "query": item["query"],
                "retrieved_sources": sources,
                "expected_source": item["source_file"],
            })

        metrics = _compute_metrics(results)
        _print_report(metrics, len(results), self.entity_count, results)

        # 最低阈值：混合检索 + Reranker 至少 MRR >= 0.3
        assert metrics["MRR"] >= 0.30, f"MRR={metrics['MRR']:.2%} 低于 30% 阈值"
        assert metrics["Recall@5"] >= 0.50, f"Recall@5={metrics['Recall@5']:.2%} 低于 50% 阈值"

    def test_eval_dataset_integrity(self):
        """评估数据集本身格式正确"""
        data_dir = Path(__file__).resolve().parent.parent / "data"
        for i, q in enumerate(self.queries):
            assert "query" in q and q["query"], f"第{i}条 query 无效"
            assert "source_file" in q and q["source_file"], f"第{i}条 source_file 无效"
            assert (data_dir / q["source_file"]).exists(), f"文件不存在: {q['source_file']}"

        sources = {q["source_file"] for q in self.queries}
        assert len(sources) >= 4, f"评估数据只覆盖 {len(sources)} 个源文件，过少"

    def test_single_query_returns_results(self):
        """至少有一条查询能返回非空结果"""
        if not self.kb_ready or self.entity_count == 0:
            pytest.skip("Milvus 不可用或知识库为空")
        sources = _retrieve(self.queries[0]["query"], top_k=5)
        assert isinstance(sources, list)
        assert len(sources) > 0, "检索返回空结果"


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    queries = _load_eval_queries()
    kb_ready, n = _check_kb()

    if not kb_ready:
        print("[!] Milvus 连接失败，请先 docker-compose up -d")
        sys.exit(1)
    if n == 0:
        print("[!] 知识库为空，请先构建:")
        print("    python -c \"from rag.milvus_service import MilvusService; MilvusService().build_knowledge_base()\"")
        sys.exit(1)

    results = []
    for item in queries:
        sources = _retrieve(item["query"])
        results.append({
            "query": item["query"],
            "retrieved_sources": sources,
            "expected_source": item["source_file"],
        })

    metrics = _compute_metrics(results)
    _print_report(metrics, len(results), n, results)
