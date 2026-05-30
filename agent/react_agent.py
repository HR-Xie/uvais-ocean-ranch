"""React Agent — 基于 langchain.agents.create_agent"""

from langchain.agents import create_agent
from utils.llm import get_chat_model
from utils.logging import logger
from utils.prompts import load_system_prompts
from agent.tools import TOOLS_ALL
from agent.middleware import monitor_tool, log_and_limit_tools, report_prompt_switch


class ReactAgent:

    def __init__(self):
        self._agent = create_agent(
            model=get_chat_model(),
            system_prompt=load_system_prompts(),
            tools=TOOLS_ALL,
            middleware=[monitor_tool, log_and_limit_tools, report_prompt_switch],
        )

    def execute_stream(self, query: str, history: list = None):
        messages = []
        if history:
            for msg in history:
                messages.append({"role": msg.get("role", "user"), "content": str(msg.get("content", ""))})
        messages.append({"role": "user", "content": query})

        last_ai = None
        try:
            for chunk in self._agent.stream(
                {"messages": messages},
                stream_mode="values",
                context={"report": False},
                config={"recursion_limit": 25},
            ):
                latest = chunk["messages"][-1]
                if getattr(latest, "type", "") != "ai" or getattr(latest, "tool_calls", None):
                    continue
                if latest.content:
                    last_ai = latest.content.strip()
                    yield last_ai + "\n"
        except Exception as e:
            err = type(e).__name__
            logger.error(f"Agent error: {err}: {str(e)[:200]}")
            if "Recursion" in err or "recursion" in str(e).lower():
                yield (last_ai or "**【系统提示】工单流程超限，请尝试更明确的指令。**") + "\n"
            elif "SSL" in err or "Connection" in err:
                yield "**【系统异常】大模型服务连接失败，请检查网络后重试。**\n"
            else:
                yield f"**【系统异常】{err}，请稍后重试。**\n"
