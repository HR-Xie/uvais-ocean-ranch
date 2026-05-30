import os, sys
os.chdir('G:/lc/MyAgent')
sys.path.insert(0, 'G:/lc/MyAgent')
from dotenv import load_dotenv; load_dotenv()

# 1. Imports
from utils.config import milvus_conf, prompts_conf, get_abs_path
from utils.llm import get_chat_model, get_fast_llm
from utils.logging import logger
from utils.prompts import load_system_prompts, load_rag_prompts, load_report_prompts
from agent.tools import TOOLS_ALL
from agent.middleware import monitor_tool, log_and_limit_tools, report_prompt_switch
from agent.react_agent import ReactAgent
from rag.store import KnowledgeBase
from rag.engine import RagEngine, RagService
print("1. All imports: PASS")

# 2. Config
assert isinstance(milvus_conf, dict) and "uri" in milvus_conf
assert isinstance(prompts_conf, dict) and "main_prompt_path" in prompts_conf
print("2. Config: PASS")

# 3. LLM singletons
llm1 = get_chat_model()
llm2 = get_chat_model()
assert llm1 is llm2
fast1 = get_fast_llm()
fast2 = get_fast_llm()
assert fast1 is fast2
print("3. LLM singletons: PASS")

# 4. Tools
assert len(TOOLS_ALL) == 5
print("4. Tools: PASS")

# 5. ReactAgent
agent = ReactAgent()
print("5. ReactAgent init: PASS")

print("\nAll smoke tests: PASS")
