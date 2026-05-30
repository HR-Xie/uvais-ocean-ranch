"""Agent 中间件：工具监控 + 调用限流 + 动态 prompt 切换"""

from typing import Callable
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logging import logger
from utils.prompts import load_system_prompts, load_report_prompts

_MAX_TOOL_CALLS = 5

_STOP_MSG = SystemMessage(content=(
    "**【系统指令 — 最高优先级】你已经调用了过多次工具，现在必须立即停止调用任何工具，"
    "基于已获取的信息直接给出最终文字回复。禁止再使用任何工具函数，直接输出回复内容。**"
))


@wrap_tool_call
def monitor_tool(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]):
    logger.info(f"[tool] {request.tool_call['name']}({request.tool_call['args']})")
    try:
        result = handler(request)
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"[tool] {request.tool_call['name']} failed: {e}")
        raise


@before_model
def log_and_limit_tools(state, runtime: Runtime):
    messages = state.get("messages", [])
    logger.info(f"[model] {len(messages)} messages → calling LLM")

    tool_count = sum(1 for m in messages if getattr(m, "type", "") == "tool")
    if tool_count >= _MAX_TOOL_CALLS:
        already = any(hasattr(m, "content") and "必须立即停止调用任何工具" in str(m.content) for m in messages)
        if not already:
            logger.warning(f"[limit] tool calls {tool_count} >= {_MAX_TOOL_CALLS}, injecting STOP")
            return {"messages": [_STOP_MSG]}
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    if request.runtime.context.get("report", False):
        logger.info("[prompt] → report mode")
        return load_report_prompts()
    return load_system_prompts()
