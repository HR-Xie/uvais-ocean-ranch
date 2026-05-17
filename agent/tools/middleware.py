from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
# ✅ 核心修复：这里的导入名称与你的 prompt_loader.py 完美对应！
from utils.prompt_loader import load_system_prompts, load_report_prompts


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """工具执行的监控与上下文拦截中间件"""
    logger.info(f"🌊 [Tool Monitor] 准备执行工具：{request.tool_call['name']}")
    logger.debug(f"🌊 [Tool Monitor] 传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"🌊 [Tool Monitor] 工具 {request.tool_call['name']} 调用成功")

        # 🌟 核心：如果调用了生成报告的工具，修改全局运行时状态
        if request.tool_call["name"] == "fill_context_for_report":
            logger.info("🎯 [Tool Monitor] 监测到【报告生成】场景，向运行时注入 report=True 标志！")
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"❌ [Tool Monitor] 工具 {request.tool_call['name']} 调用失败，原因：{str(e)}", exc_info=True)
        raise e


@before_model
def log_before_model(
        state: AgentState,  # 整个Agent智能体中的状态记录
        runtime: Runtime,  # 记录了整个执行过程中的上下文信息
):
    """在模型执行前输出日志，用于观测推理流"""
    msg_count = len(state["messages"])
    logger.info(f"🧠 [Before Model] 大模型即将开始推理，当前上下文共 {msg_count} 条消息。")
    if msg_count > 0:
        last_msg = state["messages"][-1]
        logger.debug(
            f"🧠 [Before Model] 消息截取: {type(last_msg).__name__} | {last_msg.content.strip()[:100]}...")
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """动态切换提示词：根据上下文状态决定给大模型注入什么灵魂"""
    is_report = request.runtime.context.get("report", False)

    if is_report:
        # ✅ 如果是报告生成场景，返回报告生成提示词内容
        logger.info("🔄 [Prompt Switch] 触发形态切换 -> 注入【工单报告专属 Prompt】")
        return load_report_prompts()

    # ✅ 否则，返回默认的系统提示词内容
    logger.debug("🔄 [Prompt Switch] 维持默认形态 -> 注入【系统主决策 Prompt】")
    return load_system_prompts()