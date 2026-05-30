"""端到端验证：知识库 + RAG + Agent"""
import os, sys
os.chdir('G:/lc/MyAgent')
sys.path.insert(0, 'G:/lc/MyAgent')
from dotenv import load_dotenv; load_dotenv()

print("=" * 50)
print("  UVAIS-Agent 端到端验证")
print("=" * 50)

# 1. 知识库重建
print("\n[1/4] 知识库重建...")
from rag.store import KnowledgeBase
kb = KnowledgeBase()
n = kb.rebuild()
print(f"  OK: {n} 个节点入库")

# 2. RAG 查询
print("\n[2/4] RAG 查询测试...")
from rag.engine import RagService
rs = RagService()
result = rs.rag_summarize("海星灾害怎么防治")
print(f"  OK: {len(result)} chars")
print(f"  Preview: {result[:120]}...")

# 3. Agent 工具
print("\n[3/4] Agent 工具测试...")
from agent.tools import rag_summarize, get_sea_state, invoke_vision_model, get_worker_nav_info, fill_context_for_report
sea = get_sea_state.invoke({"location": "B区网箱"})
print(f"  get_sea_state: {sea[:60]}...")
nav = get_worker_nav_info.invoke({})
print(f"  get_worker_nav: {nav[:60]}...")
vision = invoke_vision_model.invoke({"image_id": "cam_01"})
print(f"  invoke_vision: {vision[:60]}...")

# 4. Agent 初始化
print("\n[4/4] Agent 初始化...")
from agent.react_agent import ReactAgent
agent = ReactAgent()
print("  OK: Agent 就绪")

print("\n" + "=" * 50)
print("  全部通过")
print("=" * 50)
