"""
路径工具模块
============

提供项目路径相关的工具函数，用于获取绝对路径和项目根目录。

核心功能:
    - get_project_root(): 获取项目根目录
    - get_abs_path(): 将相对路径转换为绝对路径

使用示例:
    >>> from utils.path_tool import get_abs_path, get_project_root
    >>> root = get_project_root()
    >>> config_path = get_abs_path("config/rag.yml")
    >>> print(config_path)

作者: UVAIS 开发团队
创建时间: 2026-04
"""

import os


def get_project_root() -> str:
    """
    获取项目根目录

    通过当前文件的位置向上查找，返回项目根目录的绝对路径。

    Returns:
        str: 项目根目录的绝对路径

    Example:
        >>> root = get_project_root()
        >>> print(root)
        '/home/user/UVAIS'

    Note:
        此函数假设当前文件位于 utils/ 目录下，
        项目结构为: project_root/utils/path_tool.py
    """
    # 获取当前文件的绝对路径
    current_file = os.path.abspath(__file__)

    # 获取当前文件所在目录 (utils/)
    current_dir = os.path.dirname(current_file)

    # 获取项目根目录 (utils 的上级目录)
    project_root = os.path.dirname(current_dir)

    return project_root


def get_abs_path(relative_path: str) -> str:
    """
    将相对路径转换为绝对路径

    以项目根目录为基准，将相对路径转换为绝对路径。
    这使得代码可以在任何目录下运行，而不用担心路径问题。

    Args:
        relative_path (str): 相对于项目根目录的路径
                            例如: "config/rag.yml", "data/document.txt"

    Returns:
        str: 绝对路径

    Example:
        >>> path = get_abs_path("config/rag.yml")
        >>> print(path)
        '/home/user/UVAIS/config/rag.yml'

        >>> path = get_abs_path("models/bge-reranker-base")
        >>> print(path)
        '/home/user/UVAIS/models/bge-reranker-base'
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


# ==================== 模块测试入口 ====================
if __name__ == '__main__':
    print("=== 路径工具测试 ===")
    print(f"项目根目录: {get_project_root()}")
    print(f"配置文件路径: {get_abs_path('config/rag.yml')}")
    print(f"数据目录路径: {get_abs_path('data/')}")
    print(f"模型路径: {get_abs_path('models/bge-reranker-base')}")
