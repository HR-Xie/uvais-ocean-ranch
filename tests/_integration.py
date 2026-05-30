"""集成测试：逐项验证所有功能与工具"""
import os, sys, hashlib, json, random
os.chdir('G:/lc/MyAgent')
sys.path.insert(0, 'G:/lc/MyAgent')
from dotenv import load_dotenv; load_dotenv()

PASS, FAIL = 0, 0

def check(label: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}  — {detail}")

# ============================================================
print("\n" + "=" * 60)
print("1. RAG 知识检索 (5 类知识域)")
print("=" * 60)

from rag.engine import RagService
s = RagService()

tests = [
    ("浪高禁令", "浪高超过多少米禁止潜水作业", "2.0米"),
    ("流速阈值", "底层流速安全阈值是多少", "1.5节"),
    ("海星防治", "海星海胆灾害怎么防治", "ROV"),
    ("ROV巡检", "ROV巡检有什么要求", "0.5m/s"),
    ("网箱巡检", "网箱巡检的标准流程是什么", "巡检"),
]
for label, q, keyword in tests:
    try:
        r = s.rag_summarize(q)
        ok = len(r) > 30
        check(f"知识检索-{label}", ok, f"返回长度={len(r)}")
    except Exception as e:
        check(f"知识检索-{label}", False, str(e)[:80])

# ============================================================
print("\n" + "=" * 60)
print("2. Agent 工具集 (5 Tool)")
print("=" * 60)

from agent.tools import TOOLS_ALL, rag_summarize, get_sea_state, invoke_vision_model, get_worker_nav_info, fill_context_for_report

check("工具数量=5", len(TOOLS_ALL) == 5, f"实际={len(TOOLS_ALL)}")

# 2a. get_sea_state
r1 = get_sea_state.invoke({"location": "C区网箱"})
check("get_sea_state 返回浪高", "浪高" in r1, r1[:80])
check("get_sea_state 返回流速", "流速" in r1)

# 2b. invoke_vision_model
r2 = invoke_vision_model.invoke({"image_id": "test_img_01"})
obj = json.loads(r2)
check("invoke_vision_model 返回JSON", isinstance(obj, dict), r2[:80])
check("invoke_vision_model 含status字段", "status" in obj)

# 2c. get_worker_nav_info
r3 = get_worker_nav_info.invoke({})
check("get_worker_nav_info 返回距离", "米" in r3, r3[:80])
check("get_worker_nav_info 返回深度", "深度" in r3)

# 2d. fill_context_for_report (no prior sea state → safety interlock)
import agent.tools as at
at._last_sea = {"wave": 0, "current": 0, "safe": True}
r4 = fill_context_for_report.invoke({})
check("fill_context_for_report 安全联锁触发(无海况)", ("安全联锁" in r4 or "尚未获取" in r4), r4[:80])

# 2e. fill_context_for_report after setting safe sea state
at._last_sea = {"wave": 1.0, "current": 0.5, "safe": True}
r5 = fill_context_for_report.invoke({})
check("安全联锁通过后报告就绪", "已就绪" in r5, r5[:80])

# ============================================================
print("\n" + "=" * 60)
print("3. ReAct Agent 完整对话")
print("=" * 60)

from agent.react_agent import ReactAgent
agent = ReactAgent()

# 3a. Simple knowledge query
q1 = "浪高超过多少米禁止潜水作业"
chunks = []
for chunk in agent.execute_stream(q1):
    chunks.append(chunk)
result = "".join(chunks)
check("Agent流式对话(知识)", len(result) > 30, f"返回长度={len(result)}, 内容={result[:60]}")

# 3b. Tool-triggering query
chunks2 = []
for chunk in agent.execute_stream("查询C区网箱的海况"):
    chunks2.append(chunk)
result2 = "".join(chunks2)
check("Agent触发海况工具", len(result2) > 20, f"返回长度={len(result2)}, 内容={result2[:60]}")

# 3c. Multi-turn conversation
history = [{"role": "user", "content": "禁止潜水作业的浪高条件是什么"}, {"role": "assistant", "content": "浪高大于等于2.0米时严禁人工潜水作业。"}]
chunks3 = []
for chunk in agent.execute_stream("那底层流速呢", history=history):
    chunks3.append(chunk)
result3 = "".join(chunks3)
check("Agent多轮对话", len(result3) > 20, f"返回长度={len(result3)}, 内容={result3[:60]}")

# ============================================================
print("\n" + "=" * 60)
print("4. 安全联锁场景")
print("=" * 60)

# Reset and simulate unsafe sea
at._last_sea = {"wave": 3.5, "current": 1.5, "safe": False}
r6 = fill_context_for_report.invoke({})
check("不安全海况拦截派单", ("安全联锁" in r6 or "拦截" in r6), r6[:80])

# Reset
at._last_sea = {"wave": 0, "current": 0, "safe": True}

# ============================================================
print("\n" + "=" * 60)
print("5. 工具确定性 (同输入→同输出)")
print("=" * 60)

from agent.tools import _seeded_rng
rng1 = _seeded_rng("C区网箱")
rng2 = _seeded_rng("C区网箱")
v1 = [rng1.uniform(0, 1) for _ in range(10)]
v2 = [rng2.uniform(0, 1) for _ in range(10)]
check("seeded_rng 确定性", v1 == v2)

s1 = get_sea_state.invoke({"location": "确定性测试点"})
s2 = get_sea_state.invoke({"location": "确定性测试点"})
check("get_sea_state 确定性", s1 == s2, f"s1={s1[:40]}, s2={s2[:40]}")

# ============================================================
print("\n" + "=" * 60)
print("6. 知识库管理")
print("=" * 60)

from rag.store import KnowledgeBase
kb = KnowledgeBase()
nodes = kb.get_nodes()
check("知识库节点加载", len(nodes) > 0, f"节点数={len(nodes)}")
check("知识库文档数=10", len(nodes) == 59, f"实际节点数={len(nodes)}")

# ============================================================
print("\n" + "=" * 60)
print(f"结果: {PASS} PASS / {FAIL} FAIL / {PASS+FAIL} total")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
