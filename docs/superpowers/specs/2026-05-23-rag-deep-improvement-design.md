# RAG 深度改进设计文档

## 目标

面向 LLM 应用开发工程师面试，在 1-2 周内对现有 RAG 管线做三个深度模块改进，使每个模块都能在面试中深入讲解设计决策。

## 当前状态

项目 UVAIS-Agent 已实现：

- BGE-M3 双路嵌入（稠密 1024d + 稀疏）
- Milvus 2.6 混合检索 + RRF 融合
- BGE-Reranker Cross-Encoder 精排
- ReAct Agent + 5 个工具
- 自定义检索评估（Recall/Precision/MRR/HitRate）

## 三个改进模块

---

### 模块一：RAGAS 端到端评估体系

**问题：** 现有评估只测检索层，无法评估端到端生成质量。面试时缺少行业标准评估框架。

**方案：**

1. 在 `tests/data/` 新增 `rag_eval_ragas.json`，每个条目包含 `question`、`ground_truth`、可选的 `contexts`
2. 使用 RAGAS 库计算四类指标：
   - Faithfulness：生成内容是否完全基于检索上下文
   - Answer Relevancy：回答与问题的相关性
   - Context Precision：检索上下文中相关部分占比
   - Context Recall：标准答案所需信息在检索上下文中的覆盖度
3. 评估结果输出到 `tests/results/`，含 JSON 报告和终端可视化

**新增文件：**

- `tests/test_rag_eval_ragas.py`：RAGAS 评估测试用例
- `tests/data/rag_eval_ragas.json`：20-30 条带 ground truth 的评估数据集
- `tests/results/`：评估报告输出目录

**依赖变更：**

- `requirements.txt` 新增 `ragas>=0.2.0`、`datasets>=3.0.0`

**面试要点：**

- 为什么选 Faithfulness 和 Answer Relevancy 作为核心指标
- Ground truth 的构建方法（人工标注 + 从现有 SOP 文档中提取 FAQ 对）
- RAGAS 的 faithfulness 计算原理（将生成内容拆解为原子声明，逐条检查是否被上下文支持）

---

### 模块二：Query Rewriting + HyDE 查询增强

**问题：** 当前管线直接拿用户原始问题检索，不处理指代消解、省略补全、语义扩展。

**方案：**

1. **多轮 Query Rewriting**：维护对话历史，当检测到指代词或省略时，LLM 改写为完整、独立的检索查询
2. **HyDE（假设文档嵌入）**：对改写后的 query，先用 LLM 生成一段假设答案，再对假设答案做 embedding 检索（利用 Q-A 分布对齐）
3. **多路召回融合**：原始 query 检索结果 + HyDE 检索结果 → RRF 再融合 → Reranker

**新增文件：**

- `rag/query_rewriter.py`：Query Rewriting + HyDE 核心逻辑
- `prompts/query_rewrite.txt`：多轮改写 prompt
- `prompts/hyde_hypothesis.txt`：HyDE 假设文档生成 prompt

**修改文件：**

- `rag/milvus_service.py`：在 `search()` 中增加查询增强入口
- `config/milvus.yml`：增加 query_rewrite 和 hyde 配置项
- `rag/summarize_service.py`：透传改写后的 query 和对话历史

**配置结构：**

```yaml
query_rewrite:
  enabled: true
  max_history_turns: 3
  hyde:
    enabled: true
    max_hypothesis_length: 200
```

**面试要点：**

- HyDE 为什么有效（question 和 answer 的 embedding 分布存在 gap，假设答案 bridge 了这个 gap）
- 不适合 HyDE 的场景（高度事实性问题，假设文档可能引入幻觉噪声）
- 多轮改写的 prompt 设计思路（保留关键实体、补全省略、不过度扩展）

---

### 模块三：Self-RAG 检索反思与引用标注

**问题：** 检索结果不可控，没有质量检查机制，回答缺乏可追溯性。

**方案：**

1. **Retrieval Evaluator**：LLM 对每篇检索文档打分（相关/部分相关/不相关），综合判断质量等级
   - Good（>=3 篇相关）→ 直接生成
   - Mixed（1-2 篇相关）→ 变更检索策略重试
   - Poor（0 篇相关）→ 返回"无法回答"并记录
2. **Self-RAG 生成**：逐句引用标注，自我检查声称的 fact 是否在上下文中
3. **输出格式**：答案 + 引用来源列表 + 检索质量评分

**新增文件：**

- `rag/self_rag.py`：Self-RAG 核心逻辑
- `prompts/retrieval_eval.txt`：检索质量评估 prompt
- `prompts/self_rag_generate.txt`：带引用的生成 prompt

**修改文件：**

- `rag/summarize_service.py`：集成 Self-RAG 流程
- `config/rag.yml`：增加 self_rag 配置项

**面试要点：**

- CRAG vs Self-RAG 的区别（CRAG 侧重检索层面评估+纠正，Self-RAG 侧重生成层面反思+标注）
- 为什么用 LLM 打分而不是专门的 NLI 模型（可解释性强、灵活、减少额外模型依赖）
- 引用标注的实现方式（生成时要求 LLM 逐句标注来源编号，后处理验证）

---

## 面试叙述结构

按这个顺序讲 RAG 管线：

1. **整体架构**（30 秒）：BGE-M3 双路 → Milvus 混合检索 → RRF → Reranker → 生成
2. **查询增强**（1 分钟）：为什么需要改写 + HyDE，解决了什么问题
3. **检索反思**（1 分钟）：怎么评估检索质量，差了怎么办
4. **评估体系**（1 分钟）：用什么指标、为什么、怎么构建测试集
5. **设计决策复盘**（30 秒）：如果重新做，哪些地方会不同

## 不做的

- 语义分块（时间不够，当前 500/50 足够合理）
- LangFuse 追踪（优先级低于三个核心模块）
- Streaming 输出（Agent 模式已有流式）
- FastAPI 服务层（Streamlit 已满足演示需求）
- GraphRAG / 知识图谱（架构改动太大）
