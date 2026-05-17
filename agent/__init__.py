"""
Agent 智能体模块
================

提供 UVAIS 系统的核心智能体实现，基于 LangGraph 框架构建。

核心组件:
    - ReactAgent: ReAct 模式智能体，支持「思考-行动-观察」循环
    - Tools: 水下作业专属工具函数集合
    - Middleware: 日志监控、动态提示词切换等中间件

使用示例:
    >>> from agent import ReactAgent
    >>> agent = ReactAgent()
    >>> for chunk in agent.execute_stream("查询B区海况"):
    ...     print(chunk)
"""

from agent.react_agent import ReactAgent

__all__ = ["ReactAgent"]
