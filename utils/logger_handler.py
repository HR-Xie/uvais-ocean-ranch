"""
日志处理模块
============

提供统一的日志配置和管理功能，支持同时输出到控制台和文件。

核心功能:
    - get_logger(): 创建自定义日志器
    - logger: 全局默认日志器实例

日志格式:
    时间 - 日志器名称 - 日志级别 - 文件名:行号 - 消息内容

使用示例:
    >>> from utils.logger_handler import logger, get_logger
    >>> logger.info("系统启动")
    >>> logger.error("发生错误")

    # 创建自定义日志器
    >>> custom_logger = get_logger("my_module")
    >>> custom_logger.info("自定义日志")

作者: UVAIS 开发团队
创建时间: 2026-04
"""

import logging
import os
from datetime import datetime

from utils.path_tool import get_abs_path


# ==================== 日志配置 ====================

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 默认日志格式
# 格式说明: 时间 - 日志器名 - 级别 - 文件:行号 - 消息
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
    name: str = "agent",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: str = None,
) -> logging.Logger:
    """
    创建并返回一个配置好的日志器

    日志器同时输出到控制台和文件，可以分别设置不同的日志级别。

    Args:
        name (str): 日志器名称，用于标识日志来源
        console_level (int): 控制台输出的最低日志级别，默认 INFO
        file_level (int): 文件输出的最低日志级别，默认 DEBUG
        log_file (str): 日志文件路径，默认为 logs/{name}_{日期}.log

    Returns:
        logging.Logger: 配置好的日志器实例

    Example:
        >>> logger = get_logger("my_app")
        >>> logger.info("应用启动")
        >>> logger.debug("调试信息")  # 只写入文件，不显示在控制台

    Note:
        - 控制台默认显示 INFO 及以上级别的日志
        - 文件默认记录 DEBUG 及以上级别的日志
        - 同一个 name 只会创建一个日志器实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 日志器级别设为最低，由 Handler 控制输出

    # 避免重复添加 Handler（防止日志重复输出）
    if logger.handlers:
        return logger

    # ========== 控制台 Handler ==========
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # ========== 文件 Handler ==========
    if not log_file:
        # 默认日志文件名格式: {name}_{YYYYMMDD}.log
        log_file = os.path.join(
            LOG_ROOT,
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


# ==================== 全局默认日志器 ====================
# 提供便捷的全局日志器，大多数情况下可直接使用

logger = get_logger()


# ==================== 模块测试入口 ====================
if __name__ == '__main__':
    print("测试日志功能...\n")

    # 测试各级别日志
    logger.debug("这是 DEBUG 级别日志 - 仅写入文件")
    logger.info("这是 INFO 级别日志 - 控制台和文件都显示")
    logger.warning("这是 WARNING 级别日志")
    logger.error("这是 ERROR 级别日志")

    print(f"\n日志文件保存在: {LOG_ROOT}")
