"""Agent 智能体模块 — 基于 LangGraph 框架构建"""

__all__ = ["ReactAgent"]


def __getattr__(name):
    if name == "ReactAgent":
        from agent.react_agent import ReactAgent
        return ReactAgent
    raise AttributeError(f"module 'agent' has no attribute '{name}'")
