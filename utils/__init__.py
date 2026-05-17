"""
工具模块
========

提供配置管理、日志、文件处理等通用功能。

核心组件:
    配置管理:
        - rag_conf: RAG 模块配置
        - chroma_conf: 向量库配置
        - prompts_conf: 提示词配置
        - agent_conf: Agent 配置

    日志:
        - logger: 全局日志实例
        - get_logger(): 获取自定义日志器

    路径工具:
        - get_abs_path(): 获取绝对路径
        - get_project_root(): 获取项目根目录

使用示例:
    >>> from utils import rag_conf, logger, get_abs_path
    >>> print(rag_conf["chat_model_name"])
    >>> logger.info("系统启动")
    >>> path = get_abs_path("config/rag.yml")
"""

from utils.config_handler import (
    rag_conf,
    milvus_conf,
    prompts_conf,
    agent_conf
)
from utils.logger_handler import logger, get_logger
from utils.path_tool import get_abs_path, get_project_root

__all__ = [
    # 配置
    "rag_conf",
    "prompts_conf",
    "agent_conf",
    # 日志
    "logger",
    "get_logger",
    # 路径
    "get_abs_path",
    "get_project_root"
]
