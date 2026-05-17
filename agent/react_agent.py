from langchain.agents import create_agent
from model.factory import get_chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    rag_summarize,
    get_sea_state,
    invoke_vision_model,
    get_worker_nav_info,
    fill_context_for_report,
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=get_chat_model(),
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_sea_state,
                invoke_vision_model,
                get_worker_nav_info,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str, history: list = None, role: str = "manager"):
        if history is None:
            history = []

        # 拼接角色前缀用于 prompt 中的角色识别
        role_prefix = f"[{role}_] "
        messages = history + [{"role": "user", "content": role_prefix + query}]

        input_dict = {"messages": messages}

        try:
            for chunk in self.agent.stream(
                input_dict, stream_mode="values", context={"report": False}
            ):
                latest_message = chunk["messages"][-1]
                msg_type = getattr(latest_message, "type", "")
                # 只输出 AI 消息，过滤用户消息和工具消息
                if msg_type == "ai" and latest_message.content:
                    yield latest_message.content.strip() + "\n"
        except Exception as e:
            import traceback
            from utils.logger_handler import logger
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            logger.error(f"Agent error: {error_type}: {error_msg}")
            logger.error(traceback.format_exc())
            if "SSL" in error_type or "Connection" in error_type or "SSLError" in error_type or "APIConnection" in error_type:
                yield f"**【系统异常】大模型服务连接失败，请检查网络后重试。\n\n> 错误详情: {error_type}: {error_msg}**\n"
            else:
                yield f"**【系统异常】处理请求时出错: {error_type}: {error_msg}**\n"
