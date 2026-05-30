"""RAG 评估模块 — Ragas 4 指标 (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall)

用法:
    python rag/eval.py                 # 手标注测试集 (默认 20 题)
    python rag/eval.py --quick         # 快速测试 (5 题)
    python rag/eval.py --all           # 全部 94 题
    python rag/eval.py --generate 10   # Ragas 自动出题 (试验性)
"""

import os, json, math, argparse, time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv; load_dotenv()

import pandas as pd
from datasets import Dataset
from openai import OpenAI

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import _LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings as LangchainHFE
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from utils.config import milvus_conf, get_abs_path
from utils.logging import logger

METRIC_LABELS = {
    "faithfulness": "忠实度", "answer_relevancy": "相关性",
    "context_precision": "精确度", "context_recall": "召回率",
}


def _init_judge():
    """创建 Ragas 考官 (LLM + Embeddings)"""
    api_key = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
    api_base = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
    client = OpenAI(api_key=api_key, base_url=api_base)
    judge_llm = llm_factory(model, client=client, max_tokens=8192)

    lc_emb = LangchainHFE(model_name=get_abs_path(milvus_conf["model_path"]), model_kwargs={"device": "cpu"})
    embeddings = _LangchainEmbeddingsWrapper(lc_emb)
    return judge_llm, embeddings


def _load_queries(path: str = "data/eval/rag_eval_queries.json") -> list[dict]:
    with open(get_abs_path(path), "r", encoding="utf-8") as f:
        return [{"question": q["query"], "ground_truth": q["expected_knowledge"]} for q in json.load(f)]


def _get_engine():
    from rag.store import KnowledgeBase
    from rag.engine import RagEngine
    kb = KnowledgeBase()
    kb.get_index()
    return RagEngine(kb)._engine


class RagasEvaluator:

    def __init__(self, report_file: str = "data/eval/ragas_eval_report.csv"):
        self.report_file = get_abs_path(report_file)

    def run_curated(self, max_queries: int = 20) -> pd.DataFrame:
        print(f"\n{'='*50}\n  RAG 评估 — 手标注测试集\n{'='*50}")

        judge_llm, embeddings = _init_judge()
        queries = _load_queries()[:max_queries]
        print(f"  考题: {len(queries)} 道")

        # 系统作答
        engine = _get_engine()
        answers, contexts = [], []
        for i, q in enumerate(queries):
            print(f"  [{i + 1}/{len(queries)}] {q['question'][:60]}...")
            try:
                resp = engine.query(q["question"])
                answers.append(resp.response)
                contexts.append([n.node.get_content() for n in resp.source_nodes])
            except Exception as e:
                logger.error(f"  第 {i + 1} 题失败: {e}")
                answers.append("")
                contexts.append([])

        # Ragas 打分
        ds = Dataset.from_dict({
            "question": [q["question"] for q in queries],
            "answer": answers, "contexts": contexts,
            "ground_truth": [q["ground_truth"] for q in queries],
        })
        print(f"  打分中 (4 维 × {len(queries)} 题)...")
        t0 = time.time()
        result = evaluate(dataset=ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                          llm=judge_llm, embeddings=embeddings, raise_exceptions=False)
        print(f"  耗时 {time.time() - t0:.0f}s")

        # 打印 + 导出
        self._print(result)
        return self._export(queries, answers, result)

    def run_generate(self, test_size: int = 10) -> pd.DataFrame:
        print(f"\n{'='*50}\n  RAG 评估 — 自动出题\n{'='*50}")

        judge_llm, embeddings = _init_judge()

        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        docs = []
        data_path = get_abs_path(milvus_conf["data_path"])
        for ext in milvus_conf["allow_knowledge_file_type"]:
            docs.extend(DirectoryLoader(data_path, loader_cls=TextLoader,
                                        glob=f"**/*.{ext}", silent_errors=True).load())
        long_docs = [d for d in docs if len(d.page_content) > 1500]
        print(f"  {len(docs)} docs → {len(long_docs)} long docs")

        from ragas.testset import TestsetGenerator
        from ragas.testset.synthesizers import default_query_distribution
        gen = TestsetGenerator(llm=judge_llm, embedding_model=embeddings)
        t0 = time.time()
        testset = gen.generate_with_langchain_docs(long_docs, testset_size=test_size,
                                                   query_distribution=default_query_distribution(judge_llm))
        print(f"  生成 {test_size} 题, 耗时 {time.time() - t0:.0f}s")

        queries_df = testset.to_pandas()
        queries = [{"question": r["question"], "ground_truth": r.get("ground_truth", "")}
                   for _, r in queries_df.iterrows()]

        engine = _get_engine()
        answers, contexts = [], []
        for i, q in enumerate(queries):
            print(f"  [{i + 1}/{len(queries)}] {q['question'][:60]}...")
            try:
                resp = engine.query(q["question"])
                answers.append(resp.response)
                contexts.append([n.node.get_content() for n in resp.source_nodes])
            except Exception as e:
                answers.append("")
                contexts.append([])

        ds = Dataset.from_dict({
            "question": [q["question"] for q in queries],
            "answer": answers, "contexts": contexts,
            "ground_truth": [q["ground_truth"] for q in queries],
        })
        result = evaluate(dataset=ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                          llm=judge_llm, embeddings=embeddings, raise_exceptions=False)
        self._print(result)
        return self._export(queries, answers, result)

    def _print(self, result):
        print(f"\n  {'─' * 40}\n  评估结果\n  {'─' * 40}")
        for key, label in METRIC_LABELS.items():
            try:
                vals = [v for v in result[key] if not (isinstance(v, float) and math.isnan(v))]
            except (KeyError, TypeError):
                continue
            if vals:
                m = sum(vals) / len(vals)
                bar = "#" * int(m * 30) + "-" * (30 - int(m * 30))
                print(f"  {label}: {m:6.1%} {bar}")
        print(f"  {'─' * 40}")

    def _export(self, queries, answers, result) -> pd.DataFrame:
        df = result.to_pandas()
        df.insert(0, "ground_truth", [q["ground_truth"] for q in queries])
        df.insert(0, "question", [q["question"] for q in queries])
        df["system_answer"] = answers
        Path(self.report_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.report_file, index=False, encoding="utf-8-sig")
        print(f"  报表: {self.report_file}")
        return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--generate", type=int, nargs="?", const=10, default=None)
    p.add_argument("--report", type=str, default="data/eval/ragas_eval_report.csv")
    args = p.parse_args()

    ev = RagasEvaluator(report_file=args.report)
    if args.generate:
        ev.run_generate(test_size=args.generate)
    elif args.quick:
        ev.run_curated(max_queries=5)
    elif args.all:
        ev.run_curated(max_queries=None)
    else:
        ev.run_curated(max_queries=args.size)


if __name__ == "__main__":
    main()
