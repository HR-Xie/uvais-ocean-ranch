# RAG 深度改进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对现有 RAG 管线做三个深度模块改进：RAGAS 端到端评估、Query Rewriting + HyDE 查询增强、Self-RAG 检索反思与引用标注

**Architecture:** 三个模块独立可测试，按依赖顺序实施。模块二改造检索入口（MilvusService），模块三改造生成出口（RagSummarizeService），模块一提供评估基准验证改进效果。不改动 Agent 层和 UI 层。

**Tech Stack:** Python 3.10+, LangChain, RAGAS, BGE-M3, Milvus, BGE-Reranker, sentence-transformers

---

## File Structure

```
新增文件:
  tests/data/rag_eval_ragas.json       — RAGAS 评估数据集 (20 条, 带 ground_truth)
  tests/test_rag_eval_ragas.py         — RAGAS 端到端评估测试
  rag/query_rewriter.py                — QueryRewriter 类 (多轮改写 + HyDE)
  rag/self_rag.py                      — SelfRAG 类 (检索评估 + 引用生成)
  prompts/query_rewrite.txt            — 多轮改写 prompt
  prompts/hyde_hypothesis.txt          — HyDE 假设文档 prompt
  prompts/retrieval_eval.txt           — 检索质量评估 prompt
  prompts/self_rag_generate.txt        — 带引用生成 prompt

修改文件:
  requirements.txt                     — 新增 ragas, datasets
  config/milvus.yml                    — 新增 query_rewrite 配置节
  config/rag.yml                       — 新增 self_rag 配置节
  rag/milvus_service.py               — 新增 enhanced_search() 方法
  rag/summarize_service.py            — 集成 query rewrite + self-rag 流程
```

---

### Task 1: 创建 RAGAS 评估数据集

**Files:**
- Create: `tests/data/rag_eval_ragas.json`

- [ ] **Step 1: 写入评估数据集 JSON**

```python
# 文件: tests/data/rag_eval_ragas.json
# 内容如下:

[
  {
    "question": "浪高超过多少米禁止潜水作业",
    "ground_truth": "浪高超过2.0米时禁止任何人工潜水作业。若海况预报显示未来2小时内浪高可能突破2.0米，应提前中止作业并撤离人员。"
  },
  {
    "question": "底层流速安全阈值是多少",
    "ground_truth": "底层海流流速大于等于1.5节（约0.77米/秒）时禁止人工潜水。流速在1.0-1.5节之间时需评估风险后方可作业。"
  },
  {
    "question": "网箱巡检的标准流程是什么",
    "ground_truth": "网箱巡检标准流程包括：目视检查网衣完整性（是否有破损、生物附着）、声呐扫描锚链张力、水下摄像机记录破损点坐标和尺寸。巡检频率为每周一次例行检查，极端天气后需增加临时巡检。"
  },
  {
    "question": "海星海胆灾害怎么防治",
    "ground_truth": "海星防治：调度ROV携带柔性机械臂无损抓取海星，单个网箱海星密度超过5只/m²时启动应急清除。海胆防治：调度高压水枪清洗船清除海胆，重点清理网衣附着区域。"
  },
  {
    "question": "ROV巡检作业有什么要求",
    "ground_truth": "ROV下放速度不超过0.5m/s，回收速度不超过1m/s。巡检时ROV距网衣保持1.5-3米安全距离。作业前需检查ROV电池电量不低于80%，通信链路延迟小于200ms。"
  },
  {
    "question": "海洋气象预警分为哪些等级",
    "ground_truth": "海洋气象预警分为四个等级：蓝色预警（风速6-7级，阵风8级）、黄色预警（风速8-9级，阵风10级）、橙色预警（风速10-11级，阵风12级）、红色预警（风速≥12级或台风直接登陆）。不同等级对应不同的应急响应措施。"
  },
  {
    "question": "深水网箱养殖有哪些技术要点",
    "ground_truth": "深水网箱养殖关键技术要点：网箱底部距海底不少于5米以确保水体交换；适宜流速0.3-0.8m/s；溶解氧不低于5mg/L；水温适宜范围18-28℃；养殖密度根据品种调整，大黄鱼建议8-12kg/m³。"
  },
  {
    "question": "发现网衣破损应该怎么处理",
    "ground_truth": "发现网衣破损后：立即评估破损面积和位置。小面积破损（<0.5m²）可派遣潜水员使用专用补网工具水下修补。大面积破损（>0.5m²）需启动应急响应，先使用临时围网控制鱼群逃逸，再安排更换网衣。破损修复后需连续3天监测该区域。"
  },
  {
    "question": "潜水作业前需要检查哪些安全项目",
    "ground_truth": "潜水作业前安全检查项目：1) 气象条件确认（浪高<2.0m，底层流速<1.5节，能见度>3m）；2) 潜水装备检查（气瓶压力、呼吸调节器、潜水电脑、通讯设备）；3) 应急预案确认（减压舱就位、救援潜水员待命）；4) 作业人员健康状态确认。"
  },
  {
    "question": "台风来临前的应急处置措施",
    "ground_truth": "台风来临前应急处置：接收到橙色或红色预警后，立即启动防台预案。1) 加固网箱锚链系统，增加配重；2) 所有人员撤离海上平台，禁止一切潜水和水下作业；3) ROV等设备回收至安全区域；4) 关闭非必要电力系统；5) 通过监控系统持续监测网箱状态直至通讯中断。"
  },
  {
    "question": "水下基础设施巡检包含哪些内容",
    "ground_truth": "水下基础设施巡检内容：锚链系统（张力检测、腐蚀程度、连接件完整性）、网衣系统（破损检测、生物附着量、纲绳磨损）、浮力系统（浮筒完整性、浮力损失检测）、水下框架（变形检测、焊缝裂纹检查）。使用ROV搭载高清摄像机和声呐进行。"
  },
  {
    "question": "鱼类异常行为应该怎么判断和处理",
    "ground_truth": "鱼类异常行为判断：浮头（缺氧信号）、集群游动方向紊乱（可能受惊或病害）、摄食量骤降（环境应激或疾病）。处理流程：先检测水质参数（溶氧、温度、氨氮），排除环境因素；若水质正常，采集样本送检，根据诊断结果采取相应治疗措施。"
  },
  {
    "question": "生态灾害包括哪些类型",
    "ground_truth": "海洋牧场生态灾害类型包括：赤潮（有害藻华）、水母爆发、海星/海胆等敌害生物聚集、缺氧水团入侵、低温水团异常。每种灾害有不同的监测指标和防治方案。赤潮重点关注叶绿素a浓度和溶解氧变化。"
  },
  {
    "question": "UVAIS视觉感知系统能检测什么",
    "ground_truth": "UVAIS视觉感知系统可检测：网衣破损（通过图像分割识别异常孔洞）、鱼类数量统计与行为分析、敌害生物识别（海星、海胆、水母）、网衣生物附着程度评估、水下结构物异常检测。系统基于深度学习模型，支持实时视频流分析。"
  },
  {
    "question": "巡检报告应该包含哪些内容",
    "ground_truth": "巡检报告应包含：巡检时间与人员信息、巡检区域和设备编号、气象与海况条件记录、各项检查结果（正常/异常）、异常问题详细描述及照片/视频证据、处置建议和优先级、需要后续跟踪的项目。报告需在巡检结束后2小时内提交。"
  },
  {
    "question": "水下通讯中断了怎么办",
    "ground_truth": "水下通讯中断处理：ROV作业中通讯中断时，ROV应自动执行悬停并等待指令恢复，超时30秒后自动执行缓速上浮至水面。潜水员通讯中断时，立即执行紧急出水程序，以不高于9米/分钟的速度上升，安全停留3分钟。通讯系统需配置冗余信道（水声+缆线）。"
  },
  {
    "question": "养殖海域水质常规监测指标有哪些",
    "ground_truth": "养殖海域水质常规监测指标：水温、盐度、溶解氧、pH值、氨氮、亚硝酸盐、叶绿素a、浊度。监测频率为每日两次（早8点和晚6点）。异常天气期间加密至每4小时一次。数据自动上传至UVAIS平台，超出阈值自动告警。"
  },
  {
    "question": "敌害生物防控有哪些常用方法",
    "ground_truth": "敌害生物防控方法：物理防控（防鲨网、防鸟网、气泡幕）、生物防控（混养清洁鱼种如篮子鱼控制藻类）、机械清除（ROV抓取海星、高压水枪清除附着生物）、环境调控（调节网箱深度避开敌害聚集水层）。优先采用物理和生物方法，化学方法仅作最后手段且需评估环境影响。"
  },
  {
    "question": "海况预警达到黄色级别时应该做什么",
    "ground_truth": "黄色预警（风速8-9级，阵风10级）应对措施：停止所有水面和水下作业，只保留ROV可在抗流等级允许范围内执行紧急任务；加固网箱设施；人员集中在安全区域待命；加密气象数据监测频率至每30分钟一次；通知所有作业船只回港避风。"
  },
  {
    "question": "典型的故障处理流程是怎样的",
    "ground_truth": "典型故障处理流程：1) 发现故障（人工巡检或自动告警）；2) 故障确认与分级（一般/严重/紧急）；3) 制定处置方案（参考SOP和历史案例）；4) 执行处置并全程记录；5) 处置后验证；6) 编写故障报告并归档。整个过程需在UVAIS平台留痕。紧急故障需在15分钟内响应，4小时内给出处置方案。"
  }
]
```

- [ ] **Step 2: 验证 JSON 格式正确**

```bash
python -c "import json; data=json.load(open('tests/data/rag_eval_ragas.json','r',encoding='utf-8')); print(f'OK: {len(data)} entries')"
```
Expected: `OK: 20 entries`

- [ ] **Step 3: 提交**

```bash
git add tests/data/rag_eval_ragas.json
git commit -m "feat: add RAGAS evaluation dataset with 20 ground-truth entries"
```

---

### Task 2: 添加 RAGAS 依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 往 requirements.txt 追加 ragas 和 datasets**

在 `requirements.txt` 末尾追加两行:

```
# RAG 评估
ragas>=0.2.0
datasets>=3.0.0
```

- [ ] **Step 2: 安装依赖**

```bash
pip install ragas>=0.2.0 datasets>=3.0.0
```

- [ ] **Step 3: 验证导入成功**

```bash
python -c "from ragas import evaluate; from ragas.metrics import faithfulness, answer_relevancy; from datasets import Dataset; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: 提交**

```bash
git add requirements.txt
git commit -m "chore: add ragas and datasets dependencies"
```

---

### Task 3: 编写 RAGAS 端到端评估测试

**Files:**
- Create: `tests/test_rag_eval_ragas.py`

- [ ] **Step 1: 写入测试文件**

```python
"""
RAGAS 端到端评估
================
评估完整 RAG 管线的生成质量: Faithfulness, AnswerRelevancy,
ContextPrecision, ContextRecall.

运行前需确保:
  1. Milvus 已启动 (docker-compose up -d)
  2. 知识库已构建
  3. LLM_API_KEY 已设置 (.env)

运行:
  pytest tests/test_rag_eval_ragas.py -v -s
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def _load_ragas_queries():
    path = Path(__file__).parent / "data" / "rag_eval_ragas.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_pipeline():
    """检查 Milvus + LLM 是否就绪."""
    try:
        from rag.milvus_service import MilvusService
        from model.factory import get_chat_model
        milvus = MilvusService()
        stats = milvus.client.get_collection_stats(milvus.collection_name)
        if stats["row_count"] == 0:
            return False, "知识库为空"
        get_chat_model()
        return True, stats["row_count"]
    except Exception as e:
        return False, str(e)


def _run_rag_pipeline(query: str) -> dict:
    """运行完整 RAG 管线, 返回 answer 和 contexts."""
    from rag.summarize_service import RagSummarizeService
    service = RagSummarizeService()

    candidates = service.vector_store.hybrid_search(query)
    if not candidates:
        return {"answer": "", "contexts": []}

    top_docs = service.reranker.rerank(query, candidates)

    context_str = ""
    contexts = []
    for i, doc in enumerate(top_docs):
        context_str += f"【参考资料 {i+1}】\n来源：{doc['source']}\n内容：{doc['text']}\n{'-'*30}\n"
        contexts.append(doc["text"])

    try:
        answer = service.chain.invoke({"input": query, "context": context_str})
    except Exception:
        answer = ""

    return {"answer": answer, "contexts": contexts}


def _compute_ragas_metrics(eval_data: list) -> dict:
    """使用 RAGAS 计算 Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall."""
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from model.factory import get_chat_model

    evaluator_llm = LangchainLLMWrapper(get_chat_model())

    dataset_dict = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }
    for item in eval_data:
        dataset_dict["user_input"].append(item["question"])
        dataset_dict["response"].append(item["answer"])
        dataset_dict["retrieved_contexts"].append(item["contexts"])
        dataset_dict["reference"].append(item["ground_truth"])

    eval_dataset = EvaluationDataset.from_dict(dataset_dict)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
    )
    return result


def _print_ragas_report(metrics: dict, n_queries: int, per_query: list):
    print("\n" + "=" * 58)
    print("  RAGAS 端到端评估报告")
    print("=" * 58)
    print(f"  评估查询数: {n_queries}")
    print(f"  指标说明:")
    print(f"    Faithfulness     — 生成内容是否完全基于检索上下文 (不幻觉)")
    print(f"    AnswerRelevancy  — 回答与问题的相关程度")
    print(f"    ContextPrecision — 检索上下文中相关信息的占比")
    print(f"    ContextRecall    — 标准答案信息在上下文中的覆盖度")
    print("-" * 58)
    print(f"  {'指标':<20s} {'得分':>8s}")
    print("-" * 58)

    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "AnswerRelevancy",
        "context_precision": "ContextPrecision",
        "context_recall": "ContextRecall",
    }
    for key in metric_keys:
        if key in metrics:
            v = metrics[key]
            bar = "█" * int(v * 25)
            print(f"  {metric_labels[key]:<20s} {v:7.1%}  {bar}")

    print("-" * 58)
    print("  逐查询 Faithfulness:")
    for i, item in enumerate(per_query):
        f_val = item.get("faithfulness", None)
        f_str = f"{f_val:.2f}" if f_val is not None else "N/A"
        print(f"  [{i+1:2d}] {item['question'][:40]:<42s} Faithfulness={f_str}")
    print("=" * 58 + "\n")


class TestRAGASEvaluation:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.queries = _load_ragas_queries()
        self.ready, self.info = _check_pipeline()

    def test_ragas_end_to_end(self):
        """核心测试: 对所有查询运行完整 RAG 管线 + RAGAS 评估."""
        if not self.ready:
            pytest.skip(f"管线未就绪: {self.info}")

        eval_data = []
        for item in self.queries:
            result = _run_rag_pipeline(item["question"])
            eval_data.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": result["answer"],
                "contexts": result["contexts"],
            })

        ragas_result = _compute_ragas_metrics(eval_data)

        per_query = []
        for i, item in enumerate(eval_data):
            per_query.append({
                "question": item["question"],
                "faithfulness": ragas_result["faithfulness"][i] if "faithfulness" in ragas_result else None,
            })

        _print_ragas_report(ragas_result, len(eval_data), per_query)

        assert ragas_result.get("faithfulness", 0) >= 0.50, \
            f"Faithfulness={ragas_result.get('faithfulness', 0):.2%} 低于 50% 阈值"
        assert ragas_result.get("answer_relevancy", 0) >= 0.50, \
            f"AnswerRelevancy={ragas_result.get('answer_relevancy', 0):.2%} 低于 50% 阈值"

    def test_ragas_dataset_integrity(self):
        """验证 RAGAS 评估数据集格式完整."""
        for i, q in enumerate(self.queries):
            assert "question" in q and q["question"], f"第{i}条 question 无效"
            assert "ground_truth" in q and q["ground_truth"], f"第{i}条 ground_truth 无效"
        assert len(self.queries) >= 15, f"只有 {len(self.queries)} 条数据, 至少需要 15 条"

    def test_single_pipeline_returns_valid(self):
        """验证管线对第一条查询能返回有效回答."""
        if not self.ready:
            pytest.skip(f"管线未就绪: {self.info}")
        result = _run_rag_pipeline(self.queries[0]["question"])
        assert len(result["answer"]) > 0, "回答为空"
        assert len(result["contexts"]) > 0, "检索上下文为空"
```

- [ ] **Step 2: 运行测试确认能正确跳过或执行**

```bash
cd G:/lc/MyAgent && python -c "from tests.test_rag_eval_ragas import _load_ragas_queries; print(len(_load_ragas_queries()))"
```
Expected: `20`

- [ ] **Step 3: 提交**

```bash
git add tests/test_rag_eval_ragas.py
git commit -m "feat: add RAGAS end-to-end evaluation test"
```

---

### Task 4: 创建 Query Rewrite Prompt

**Files:**
- Create: `prompts/query_rewrite.txt`

- [ ] **Step 1: 写入提示词文件**

```
You are a query rewriting assistant for a marine ranch operations system. Your task is to rewrite the user's latest question into a standalone, self-contained query suitable for document retrieval.

## Conversation History
{history}

## Latest Question
{query}

## Rules
1. Resolve pronouns and demonstrative references (e.g., "it", "this", "that", "these") by replacing them with the explicit entities from the conversation history
2. Expand incomplete phrases and acronyms using context from the history
3. If the latest question is already standalone, return it as-is
4. Keep the rewritten query concise and focused — do NOT add unnecessary elaboration
5. Output ONLY the rewritten query text, nothing else
```

- [ ] **Step 2: 提交**

```bash
git add prompts/query_rewrite.txt
git commit -m "feat: add query rewrite prompt template"
```

---

### Task 5: 创建 HyDE Hypothesis Prompt

**Files:**
- Create: `prompts/hyde_hypothesis.txt`

- [ ] **Step 1: 写入提示词文件**

```
You are a marine ranch operations expert. Write a short, factual technical passage that answers the following question, based on standard industry practices for marine aquaculture and underwater operations.

Question: {query}

Write in the style of a technical SOP document. Keep the passage between 150 and 250 words. Focus on actionable procedures, safety thresholds, and standard protocols. Do NOT add introductions or disclaimers — write as if this is an excerpt from an operations manual.
```

- [ ] **Step 2: 提交**

```bash
git add prompts/hyde_hypothesis.txt
git commit -m "feat: add HyDE hypothesis generation prompt"
```

---

### Task 6: 实现 QueryRewriter 类

**Files:**
- Create: `rag/query_rewriter.py`

- [ ] **Step 1: 写入测试文件（TDD）**

```python
# tests/test_query_rewriter.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from rag.query_rewriter import QueryRewriter


class TestQueryRewriter:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.rewriter = QueryRewriter()

    def test_standalone_query_passes_through(self):
        """独立完整的问题应原样返回."""
        query = "深水网箱养殖技术要点是什么"
        result = self.rewriter.rewrite(query, history=[])
        assert "深水网箱" in result

    def test_pronoun_resolution(self):
        """代指应被消解."""
        history = [
            {"role": "user", "content": "ROV巡检有什么要求"},
            {"role": "assistant", "content": "ROV下放速度不超过0.5m/s，回收速度不超过1m/s。"},
        ]
        query = "它的最大作业深度是多少"
        result = self.rewriter.rewrite(query, history=history)
        assert "ROV" in result

    def test_generate_hypothesis_returns_text(self):
        """HyDE 应返回非空文本."""
        query = "海星爆发怎么处理"
        hypothesis = self.rewriter.generate_hypothesis(query)
        assert len(hypothesis) > 50
        assert "海星" in hypothesis

    def test_format_history_empty(self):
        """空历史应返回占位文本."""
        result = self.rewriter._format_history([])
        assert result == "(no previous conversation)"

    def test_format_history_with_turns(self):
        """多轮历史应正确格式化."""
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
        ]
        result = self.rewriter._format_history(history)
        assert "User: 问题1" in result
        assert "Assistant: 回答1" in result
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd G:/lc/MyAgent && pytest tests/test_query_rewriter.py -v
```
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: 实现 QueryRewriter 类**

```python
# rag/query_rewriter.py
"""Query Rewriter — 多轮改写 + HyDE 假设文档生成."""

from model.factory import get_chat_model
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def _load_prompt(filename: str) -> str:
    path = get_abs_path(f"prompts/{filename}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class QueryRewriter:
    """多轮查询改写 + HyDE 假设文档嵌入."""

    def __init__(self):
        self.llm = get_chat_model()
        self.rewrite_template = _load_prompt("query_rewrite.txt")
        self.hyde_template = _load_prompt("hyde_hypothesis.txt")

    def rewrite(self, query: str, history: list[dict] | None = None) -> str:
        """
        改写查询为独立可检索的完整问题.

        Args:
            query: 当前用户问题
            history: [{"role": "user"/"assistant", "content": "..."}, ...]

        Returns:
            改写后的查询字符串
        """
        history_str = self._format_history(history or [])
        prompt = self.rewrite_template.format(history=history_str, query=query)

        try:
            result = self.llm.invoke(prompt)
            rewritten = result.content.strip()
            logger.info(f"[QueryRewriter] '{query[:50]}...' -> '{rewritten[:80]}...'")
            return rewritten
        except Exception as e:
            logger.warning(f"[QueryRewriter] 改写失败，降级为原始查询: {e}")
            return query

    def generate_hypothesis(self, query: str) -> str:
        """
        生成假设文档用于 HyDE 检索.

        Args:
            query: 用户问题

        Returns:
            假设答案文本 (150-250 字)
        """
        prompt = self.hyde_template.format(query=query)

        try:
            result = self.llm.invoke(prompt)
            hypothesis = result.content.strip()
            logger.info(f"[HyDE] 假设文档已生成 ({len(hypothesis)} 字符)")
            return hypothesis
        except Exception as e:
            logger.warning(f"[HyDE] 生成失败: {e}")
            return query

    def _format_history(self, history: list[dict]) -> str:
        """格式化对话历史为 prompt 输入."""
        if not history:
            return "(no previous conversation)"

        lines = []
        max_turns = 5
        for turn in history[-max_turns * 2:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd G:/lc/MyAgent && pytest tests/test_query_rewriter.py -v
```
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add rag/query_rewriter.py tests/test_query_rewriter.py
git commit -m "feat: add QueryRewriter with multi-turn rewrite and HyDE"
```

---

### Task 7: 更新 Milvus 配置，增加 Query Rewrite 节

**Files:**
- Modify: `config/milvus.yml`

- [ ] **Step 1: 在 config/milvus.yml 末尾追加 query_rewrite 配置**

在 `config/milvus.yml` 末尾（`allow_knowledge_file_type` 之后）追加：

```yaml

# 8. 查询增强配置 (Query Rewriting + HyDE)
query_rewrite:
  enabled: true
  # 对话历史最大保留轮数
  max_history_turns: 5
  # HyDE 假设文档嵌入
  hyde:
    enabled: true
    # 假设文档最大长度 (字符)
    max_hypothesis_length: 300
```

- [ ] **Step 2: 验证 YAML 格式**

```bash
python -c "import yaml; d=yaml.load(open('config/milvus.yml','r',encoding='utf-8'),Loader=yaml.FullLoader); print(d['query_rewrite'])"
```
Expected: `{'enabled': True, 'max_history_turns': 5, 'hyde': {'enabled': True, 'max_hypothesis_length': 300}}`

- [ ] **Step 3: 提交**

```bash
git add config/milvus.yml
git commit -m "feat: add query_rewrite and hyde configuration to milvus.yml"
```

---

### Task 8: 在 MilvusService 中添加 enhanced_search() 方法

**Files:**
- Modify: `rag/milvus_service.py`

- [ ] **Step 1: 在 MilvusService 类中添加 enhanced_search() 方法**

在 `hybrid_search()` 方法之后（第 271 行后），`__main__` 块之前（第 273 行前），插入以下方法：

```python
    def enhanced_search(self, query: str, history: list[dict] | None = None) -> dict:
        """
        增强检索: Query Rewriting + HyDE + 混合检索 + 结果融合.

        Args:
            query: 用户查询
            history: 对话历史 [{"role": ..., "content": ...}, ...]

        Returns:
            {"candidates": [...], "rewritten_query": "...", "hypothesis": "..."}
        """
        from rag.query_rewriter import QueryRewriter

        rewriter = QueryRewriter()

        # 1. 多轮改写
        rewritten = rewriter.rewrite(query, history)
        logger.info(f"[增强检索] 改写后: '{rewritten[:80]}...'")

        # 2. HyDE 假设文档
        hypothesis = None
        if milvus_conf.get("query_rewrite", {}).get("hyde", {}).get("enabled", True):
            hypothesis = rewriter.generate_hypothesis(rewritten)
            hyde_text = hypothesis if isinstance(hypothesis, str) else ""
            max_len = milvus_conf.get("query_rewrite", {}).get("hyde", {}).get("max_hypothesis_length", 300)
            if hyde_text and len(hyde_text) > max_len:
                hypothesis = hyde_text[:max_len]

        # 3. 原始查询检索
        candidates = self.hybrid_search(rewritten)

        # 4. HyDE 检索 (多路召回)
        if hypothesis:
            hyde_candidates = self.hybrid_search(
                hypothesis,
                top_k=milvus_conf["search"]["final_top_k"]
            )
            # 去重合并 (按 text 内容去重, 优先保留原始检索结果)
            seen_texts = {c["text"] for c in candidates}
            for hc in hyde_candidates:
                if hc["text"] not in seen_texts:
                    seen_texts.add(hc["text"])
                    candidates.append(hc)

        logger.info(f"[增强检索] 最终候选数: {len(candidates)}")
        return {
            "candidates": candidates,
            "rewritten_query": rewritten,
            "hypothesis": hypothesis or "",
        }

    def hybrid_search_with_enhancement(
        self,
        query: str,
        history: list[dict] | None = None,
    ) -> list:
        """
        带可选增强的检索入口. 根据配置决定是否启用 query rewrite + hyde.

        Args:
            query: 用户查询
            history: 对话历史

        Returns:
            候选文档列表 [{"text": ..., "source": ..., "score": ...}, ...]
        """
        qr_enabled = milvus_conf.get("query_rewrite", {}).get("enabled", False)
        if qr_enabled:
            result = self.enhanced_search(query, history)
            return result["candidates"]
        return self.hybrid_search(query)
```

- [ ] **Step 2: 验证导入**

```bash
cd G:/lc/MyAgent && python -c "from rag.milvus_service import MilvusService; s=MilvusService(); print('enhanced_search' in dir(s))"
```
Expected: `True`

- [ ] **Step 3: 提交**

```bash
git add rag/milvus_service.py
git commit -m "feat: add enhanced_search with query rewrite + HyDE to MilvusService"
```

---

### Task 9: 更新 RagSummarizeService 使用增强检索

**Files:**
- Modify: `rag/summarize_service.py`

- [ ] **Step 1: 修改 rag_summarize() 方法，增加 history 参数并用 enhanced_search 替换 hybrid_search**

修改 `rag/summarize_service.py` 中的 `rag_summarize()` 方法（第 35-73 行）：

```python
    def rag_summarize(self, query: str, history: list[dict] | None = None) -> str:
        """
        对外暴露的核心调用方法
        """
        logger.info(f"[RAG 总结] 开始处理查询: '{query}'")

        # 1. 增强检索 (Query Rewriting + HyDE + 混合检索 + RRF)
        candidates = self.vector_store.hybrid_search_with_enhancement(query, history)

        if not candidates:
            logger.warning("[RAG 总结] 向量库未召回任何相关资料。")
            return "很抱歉，当前的知识库中没有查找到相关记录。"

        # 2. BGE-Reranker Cross-Encoder 精排
        top_docs = self.reranker.rerank(query, candidates)

        # 3. 格式化拼接 Context
        context_str = ""
        for i, doc in enumerate(top_docs):
            context_str += f"【参考资料 {i + 1}】\n"
            context_str += f"来源：{doc['source']}\n"
            context_str += f"内容：{doc['text']}\n"
            context_str += "-" * 30 + "\n"

        logger.debug(f"[RAG 总结] 拼接后的参考资料为:\n{context_str}")

        # 4. 提交大模型生成最终结果
        logger.info("[RAG 总结] 正在等待大模型生成总结回复...")
        try:
            final_answer = self.chain.invoke({
                "input": query,
                "context": context_str
            })
            logger.info("[RAG 总结] 回复生成成功。")
            return final_answer

        except Exception as e:
            logger.error(f"[RAG 总结] 大模型调用失败: {str(e)}", exc_info=True)
            return "系统异常，暂时无法生成总结，请稍后再试。"
```

- [ ] **Step 2: 验证工作流**

```bash
cd G:/lc/MyAgent && python -c "
from rag.summarize_service import RagSummarizeService
s = RagSummarizeService()
# 验证方法签名包含 history 参数
import inspect
sig = inspect.signature(s.rag_summarize)
print('history' in sig.parameters)
"
```
Expected: `True`

- [ ] **Step 3: 提交**

```bash
git add rag/summarize_service.py
git commit -m "feat: integrate enhanced search with query rewrite into summarize service"
```

---

### Task 10: 创建检索评估 Prompt

**Files:**
- Create: `prompts/retrieval_eval.txt`

- [ ] **Step 1: 写入提示词**

```
You are evaluating the quality of retrieved documents for a given query in a marine ranch operations context.

## Query
{query}

## Retrieved Documents
{documents}

## Task
For each document, rate its relevance to the query on a 3-point scale:
- 2 (Highly Relevant): The document directly answers the query or contains procedures/thresholds/data that directly address what was asked
- 1 (Partially Relevant): The document contains related domain information but does not directly answer the query
- 0 (Not Relevant): The document is about an unrelated topic or contains no useful information for this query

Based on the per-document scores, determine the overall retrieval quality:
- "good": 3 or more documents scored 2
- "mixed": 1-2 documents scored 2
- "poor": no documents scored 2

## Output Format
Return ONLY a JSON object with no additional text:
{{
  "scores": [score_for_doc_1, score_for_doc_2, ...],
  "overall": "good" | "mixed" | "poor",
  "brief_reason": "one sentence explaining the overall assessment"
}}
```

- [ ] **Step 2: 提交**

```bash
git add prompts/retrieval_eval.txt
git commit -m "feat: add retrieval evaluation prompt for Self-RAG"
```

---

### Task 11: 创建 Self-RAG 生成 Prompt

**Files:**
- Create: `prompts/self_rag_generate.txt`

- [ ] **Step 1: 写入提示词**

```
You are a UVAIS marine ranch expert assistant. Answer the user's question strictly based on the provided reference documents.

## Reference Documents
{documents}

## User Question
{query}

## Rules
1. ONLY use information from the provided documents — do NOT fabricate or guess
2. After every factual claim, cite the source document number in brackets: [1], [2], etc.
3. If multiple documents support the same claim, cite all of them: [1][3]
4. If the documents do not contain sufficient information to answer, state this clearly
5. Be concise and actionable — focus on procedures, thresholds, and decision criteria

## Answer (with citations)
```

- [ ] **Step 2: 提交**

```bash
git add prompts/self_rag_generate.txt
git commit -m "feat: add Self-RAG generation prompt with citation requirements"
```

---

### Task 12: 实现 SelfRAG 类

**Files:**
- Create: `rag/self_rag.py`

- [ ] **Step 1: 写入测试文件（TDD）**

```python
# tests/test_self_rag.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from rag.self_rag import SelfRAG


class TestSelfRAG:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.self_rag = SelfRAG()

    def test_evaluate_retrieval_returns_grades(self):
        """检索评估应返回 scores, overall, brief_reason."""
        query = "浪高安全阈值是多少"
        docs = [
            {"text": "浪高超过2.0米时禁止潜水作业。海况预报显示浪高可能突破2.0米应提前中止作业。", "source": "safety.txt"},
            {"text": "海水温度适宜范围18-28℃，溶解氧不低于5mg/L。", "source": "farming.txt"},
            {"text": "ROV下放速度不超过0.5m/s，回收速度不超过1m/s。", "source": "rov.txt"},
        ]
        result = self.self_rag.evaluate_retrieval(query, docs)

        assert "scores" in result
        assert "overall" in result
        assert result["overall"] in ("good", "mixed", "poor")
        assert len(result["scores"]) == len(docs)
        assert all(s in (0, 1, 2) for s in result["scores"])

    def test_generate_returns_text_with_citations(self):
        """生成应返回带引用的文本."""
        query = "海星怎么处理"
        docs = [
            {"text": "调度ROV携带柔性机械臂无损抓取海星，密度超过5只/m²时启动应急清除。", "source": "pest_control.txt"},
            {"text": "高压水枪清洗船清除海胆，重点清理网衣附着区域。", "source": "pest_control.txt"},
        ]
        answer = self.self_rag.generate(query, docs)

        assert len(answer) > 20
        # 应该包含引用标记
        assert "[" in answer

    def test_format_docs_for_eval(self):
        """文档格式化应包含 document 索引."""
        docs = [
            {"text": "内容A", "source": "file_a.txt"},
            {"text": "内容B", "source": "file_b.txt"},
        ]
        formatted = self.self_rag._format_docs_for_eval(docs)
        assert "[0]" in formatted
        assert "[1]" in formatted
        assert "内容A" in formatted

    def test_format_docs_for_generation(self):
        """生成文档格式化应包含来源信息."""
        docs = [
            {"text": "内容A", "source": "file_a.txt"},
        ]
        formatted = self.self_rag._format_docs_for_generation(docs)
        assert "[1]" in formatted
        assert "file_a.txt" in formatted
        assert "内容A" in formatted
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd G:/lc/MyAgent && pytest tests/test_self_rag.py -v
```
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: 实现 SelfRAG 类**

```python
# rag/self_rag.py
"""Self-RAG — 检索反思 + 引用标注生成."""

import json
import re

from model.factory import get_chat_model
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def _load_prompt(filename: str) -> str:
    path = get_abs_path(f"prompts/{filename}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class SelfRAG:
    """检索质量评估 + 带引用标注的答案生成."""

    def __init__(self):
        self.llm = get_chat_model()
        self.eval_template = _load_prompt("retrieval_eval.txt")
        self.generate_template = _load_prompt("self_rag_generate.txt")

    def evaluate_retrieval(self, query: str, docs: list[dict]) -> dict:
        """
        评估检索文档质量, 返回评分和等级.

        Args:
            query: 用户查询
            docs: 检索文档列表

        Returns:
            {"scores": [2,1,0,...], "overall": "good"|"mixed"|"poor", "brief_reason": "..."}
        """
        docs_text = self._format_docs_for_eval(docs)
        prompt = self.eval_template.format(query=query, documents=docs_text)

        try:
            result = self.llm.invoke(prompt)
            parsed = self._parse_eval_result(result.content, len(docs))
            logger.info(
                f"[SelfRAG] 检索评估: overall={parsed['overall']}, "
                f"scores={parsed['scores']}, reason={parsed.get('brief_reason', 'N/A')[:50]}"
            )
            return parsed
        except Exception as e:
            logger.warning(f"[SelfRAG] 检索评估失败, 默认 good: {e}")
            return {
                "scores": [1] * len(docs),
                "overall": "good",
                "brief_reason": "evaluation failed, defaulting to good",
            }

    def generate(self, query: str, docs: list[dict]) -> str:
        """
        生成带引用标注的答案.

        Args:
            query: 用户查询
            docs: 检索文档列表 (含 source 字段)

        Returns:
            带 [N] 引用标注的答案文本
        """
        docs_text = self._format_docs_for_generation(docs)
        prompt = self.generate_template.format(query=query, documents=docs_text)

        try:
            result = self.llm.invoke(prompt)
            answer = result.content.strip()
            logger.info(f"[SelfRAG] 答案已生成 ({len(answer)} 字符)")
            return answer
        except Exception as e:
            logger.error(f"[SelfRAG] 生成失败: {e}", exc_info=True)
            return "系统异常，暂时无法生成回答。"

    def _format_docs_for_eval(self, docs: list[dict]) -> str:
        lines = []
        for i, doc in enumerate(docs):
            text = doc["text"][:600]
            lines.append(f"[{i}] {text}")
        return "\n\n".join(lines)

    def _format_docs_for_generation(self, docs: list[dict]) -> str:
        lines = []
        for i, doc in enumerate(docs):
            source = doc.get("source", "unknown")
            text = doc["text"]
            lines.append(f"[{i+1}] (来源: {source})\n{text}")
        return "\n\n".join(lines)

    def _parse_eval_result(self, raw: str, doc_count: int) -> dict:
        """从 LLM 输出中提取 JSON 评估结果."""
        json_match = re.search(r'\{[^{}]*"scores"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if len(parsed.get("scores", [])) != doc_count:
                scores = parsed["scores"]
                if len(scores) < doc_count:
                    scores.extend([1] * (doc_count - len(scores)))
                else:
                    scores = scores[:doc_count]
                parsed["scores"] = scores
            if "overall" not in parsed:
                count_2 = sum(1 for s in parsed["scores"] if s == 2)
                parsed["overall"] = "good" if count_2 >= 3 else ("mixed" if count_2 >= 1 else "poor")
            return parsed

        return {
            "scores": [1] * doc_count,
            "overall": "good",
            "brief_reason": "failed to parse LLM output",
        }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd G:/lc/MyAgent && pytest tests/test_self_rag.py -v
```
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add rag/self_rag.py tests/test_self_rag.py
git commit -m "feat: add SelfRAG with retrieval evaluation and citation generation"
```

---

### Task 13: 更新 rag.yml 配置增加 self_rag 节

**Files:**
- Modify: `config/rag.yml`

- [ ] **Step 1: 在 config/rag.yml 末尾追加 self_rag 配置**

```yaml

# Self-RAG 配置
self_rag:
  enabled: true
  # 检索质量不佳时的最大重试次数
  max_retries: 1
  # 检索评估阈值
  thresholds:
    good: 3    # >= 3 篇高度相关视为 good
    mixed: 1   # 1-2 篇视为 mixed, <1 视为 poor
```

- [ ] **Step 2: 提交**

```bash
git add config/rag.yml
git commit -m "feat: add self_rag configuration to rag.yml"
```

---

### Task 14: 将 Self-RAG 集成到 RagSummarizeService

**Files:**
- Modify: `rag/summarize_service.py`

- [ ] **Step 1: 更新 RagSummarizeService 的 __init__() 和 rag_summarize()**

完整替换 `rag/summarize_service.py`：

```python
import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.milvus_service import MilvusService
from rag.reranker_service import RerankerService
from rag.self_rag import SelfRAG
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts
from utils.config_handler import rag_conf
from model.factory import get_chat_model


class RagSummarizeService:
    """
    UVAIS 海洋牧场 RAG 总结服务类
    工作流: Query Rewriting -> HyDE -> 增强检索 -> Reranker -> Self-RAG 检索评估 -> Self-RAG 引用生成
    """

    def __init__(self):
        logger.info("正在初始化 RAG 总结服务...")
        self.vector_store = MilvusService()
        self.reranker = RerankerService()
        self.self_rag = SelfRAG()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_chat_model()
        self.chain = self._init_chain()
        self.self_rag_enabled = rag_conf.get("self_rag", {}).get("enabled", False)
        logger.info(
            f"RAG 总结服务初始化完成！Self-RAG: {'启用' if self.self_rag_enabled else '未启用'}"
        )

    def _init_chain(self):
        """组装 LangChain LCEL 工作流."""
        return self.prompt_template | self.model | StrOutputParser()

    def rag_summarize(self, query: str, history: list[dict] | None = None) -> str:
        """对外暴露的核心调用方法."""
        logger.info(f"[RAG 总结] 开始处理查询: '{query}'")

        # 1. 增强检索
        candidates = self.vector_store.hybrid_search_with_enhancement(query, history)

        if not candidates:
            logger.warning("[RAG 总结] 未召回任何资料。")
            return "很抱歉，当前的知识库中没有查找到相关记录。"

        # 2. Reranker 精排
        top_docs = self.reranker.rerank(query, candidates)

        # 3. Self-RAG 检索质量评估
        if self.self_rag_enabled and len(top_docs) >= 3:
            eval_result = self.self_rag.evaluate_retrieval(query, top_docs)

            if eval_result["overall"] == "poor":
                logger.warning(
                    f"[RAG 总结] 检索质量 poor, 无法回答: {eval_result.get('brief_reason', '')}"
                )
                return (
                    "很抱歉，当前检索到的资料与您的问题匹配度较低，无法提供可靠的回答。"
                    "建议您换一种方式提问或补充更多背景信息。"
                )

            if eval_result["overall"] == "mixed":
                logger.info("[RAG 总结] 检索质量 mixed, 重试增强检索")
                retry_candidates = self.vector_store.hybrid_search_with_enhancement(
                    query, history
                )
                retry_docs = self.reranker.rerank(query, retry_candidates)
                retry_eval = self.self_rag.evaluate_retrieval(query, retry_docs)
                if retry_eval["overall"] != "poor":
                    top_docs = retry_docs

        # 4. Self-RAG 引用生成 或 标准生成
        if self.self_rag_enabled:
            try:
                final_answer = self.self_rag.generate(query, top_docs)
                logger.info("[RAG 总结] Self-RAG 回复生成成功。")
                return final_answer
            except Exception as e:
                logger.warning(f"[RAG 总结] Self-RAG 生成失败, 降级为标准生成: {e}")

        # 5. 标准 LLM 生成 (降级路径)
        context_str = ""
        for i, doc in enumerate(top_docs):
            context_str += f"【参考资料 {i + 1}】\n"
            context_str += f"来源：{doc['source']}\n"
            context_str += f"内容：{doc['text']}\n"
            context_str += "-" * 30 + "\n"

        try:
            final_answer = self.chain.invoke({"input": query, "context": context_str})
            logger.info("[RAG 总结] 标准回复生成成功。")
            return final_answer
        except Exception as e:
            logger.error(f"[RAG 总结] 大模型调用失败: {str(e)}", exc_info=True)
            return "系统异常，暂时无法生成总结，请稍后再试。"
```

- [ ] **Step 2: 验证初始化不报错**

```bash
cd G:/lc/MyAgent && python -c "from rag.summarize_service import RagSummarizeService; s=RagSummarizeService(); print('OK')"
```
Expected: `OK` (日志中应显示 Self-RAG 启用状态)

- [ ] **Step 3: 提交**

```bash
git add rag/summarize_service.py
git commit -m "feat: integrate Self-RAG retrieval evaluation and citation generation into summarize service"
```

---

### Task 15: 运行完整评估，验证基线并记录改善

**Files:**
- No files created or modified (verification only)

- [ ] **Step 1: 确认所有依赖就绪**

```bash
cd G:/lc/MyAgent && python -c "
from rag.summarize_service import RagSummarizeService
from rag.query_rewriter import QueryRewriter
from rag.self_rag import SelfRAG
print('All modules loaded OK')
"
```

- [ ] **Step 2: 运行检索层评估（已有）**

```bash
cd G:/lc/MyAgent && pytest tests/test_rag_eval.py -v -s 2>&1 | head -60
```

- [ ] **Step 3: 运行 RAGAS 端到端评估（新增）**

```bash
cd G:/lc/MyAgent && pytest tests/test_rag_eval_ragas.py -v -s
```

- [ ] **Step 4: 运行全部单元测试**

```bash
cd G:/lc/MyAgent && pytest tests/ -v
```

- [ ] **Step 5: 提交最终状态**

```bash
git add -A
git commit -m "test: full evaluation baseline with RAGAS + unit tests"
```

---

### Task 16: 面试叙述文档

**Files:**
- Create: `docs/interview-talking-points.md`

- [ ] **Step 1: 创建面试叙述参考**

```markdown
# RAG 管线面试叙述要点

## 30 秒概述
这是海洋牧场智能运维系统 UVAIS 的 RAG 管线。核心技术栈：BGE-M3 双路嵌入（稠密+稀疏）、Milvus 混合检索、RRF 融合、BGE-Reranker 精排、Self-RAG 引用生成。并在三个深度方向做了优化。

## 模块一：查询增强 (Query Rewriting + HyDE) — 2 分钟

### 为什么需要
用户提问经常有指代和省略，在多轮对话中尤其严重。直接用原始问题去检索会丢上下文。

### 怎么做的
1. 多轮改写：维护对话历史，LLM 改写为独立可检索查询
2. HyDE：生成假设答案 → 用答案的 embedding 去检索（bridge question-answer distribution gap）
3. 双路召回融合：原始 query 检索 + HyDE 检索 → 去重合并

### 设计决策
- HyDE 有效因为缩小了 question-answer 的 embedding 分布 gap
- 不适用于高度事实性问题（假设文档可能引入幻觉噪声）
- 用 RRF 做双路融合而非简单拼接，更平滑

## 模块二：Self-RAG 检索反思 — 2 分钟

### 为什么需要
检索质量不可控。Reranker 只做相对排序，不知道"这些文档是否足够回答这个问题"。

### 怎么做的
1. Retrieval Evaluator：LLM 对每篇文档打分（2/1/0），综合判断 good/mixed/poor
2. Good → 直接生成；Mixed → 重试检索；Poor → 告知无法回答
3. 生成时逐句标注引用来源 [1][2]

### 设计决策
- 用 LLM 而非 NLI 模型做评估（解释性强、灵活、减少模型依赖）
- 引用标注让回答可追溯，降低幻觉风险

## 模块三：RAGAS 评估 — 1 分钟

### 为什么需要
只测 Recall 不够，面试时面试官会问你"怎么评估生成质量"。

### 怎么做的
1. 20 条 ground truth 标注数据集
2. 四个指标：Faithfulness、AnswerRelevancy、ContextPrecision、ContextRecall
3. Faithfulness 是核心——回答是否百分百基于检索上下文

## 如果重做会改什么 — 给自己挖的"深度坑"

1. 语义分块代替固定 500 字符分块
2. 多路检索（Chroma + Milvus 做 ensemble）
3. 图数据库补充结构化知识（GraphRAG）
4. LangFuse 全链路追踪
```

- [ ] **Step 2: 提交**

```bash
git add docs/interview-talking-points.md
git commit -m "docs: add interview talking points for RAG pipeline"
```

---

## 实施顺序依赖

```
Module 1 (RAGAS) ──► 独立, 可先做
Module 2 (Query Rewrite) ──► 依赖 Module 1 的评估基础设施验证改进
Module 3 (Self-RAG) ──► 依赖 Module 2 的增强检索 + Module 1 的评估
```

建议实施顺序: Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16

每个模块完成后运行一次 `pytest tests/ -v` 确保无回归。
