"""
配置处理模块
============

负责加载和管理 YAML 格式的配置文件，为各模块提供配置访问接口。

配置文件:
    - rag.yml: RAG 模块配置 (模型名称等)
    - milvus.yml: Milvus 向量库配置 (分片参数、检索配置等)
    - prompts.yml: 提示词配置 (文件路径)
    - agent.yml: Agent 配置 (外部数据路径等)

使用示例:
    >>> from utils.config_handler import rag_conf, milvus_conf
    >>> print(rag_conf["chat_model_name"])
    >>> print(milvus_conf["chunk"]["chunk_size"])

作者: UVAIS 开发团队
创建时间: 2026-04
"""

import os
import yaml

from utils.path_tool import get_abs_path


def load_rag_config(
    config_path: str = get_abs_path("config/rag.yml"),
    encoding: str = "utf-8"
) -> dict:
    """
    加载 RAG 模块配置

    Args:
        config_path (str): 配置文件路径
        encoding (str): 文件编码

    Returns:
        dict: 配置字典，包含:
            - chat_model_name: 对话模型名称
            - embedding_model_name: 嵌入模型名称

    Example:
        >>> conf = load_rag_config()
        >>> print(conf["chat_model_name"])
        'qwen3-max'
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(
    config_path: str = get_abs_path("config/milvus.yml"),  # ✅ 修改为正确的
    encoding: str = "utf-8"
) -> dict:
    """
    加载向量库配置

    Args:
        config_path (str): 配置文件路径
        encoding (str): 文件编码

    Returns:
        dict: 配置字典，包含:
            - collection_name: 向量库集合名称
            - persist_directory: 持久化目录
            - k: 检索返回数量
            - chunk_size: 分片大小
            - chunk_overlap: 分片重叠

    Example:
        >>> conf = load_chroma_config()
        >>> print(conf["chunk_size"])
        200
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(
    config_path: str = get_abs_path("config/prompts.yml"),
    encoding: str = "utf-8"
) -> dict:
    """
    加载提示词配置

    Args:
        config_path (str): 配置文件路径
        encoding (str): 文件编码

    Returns:
        dict: 配置字典，包含各提示词文件的路径

    Example:
        >>> conf = load_prompts_config()
        >>> print(conf["main_prompt_path"])
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(
    config_path: str = get_abs_path("config/agent.yml"),
    encoding: str = "utf-8"
) -> dict:
    """
    加载 Agent 配置（文件不存在时返回空字典）

    Args:
        config_path (str): 配置文件路径
        encoding (str): 文件编码

    Returns:
        dict: 配置字典
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_milvus_config(
    config_path: str = get_abs_path("config/milvus.yml"),
    encoding: str = "utf-8"
) -> dict:
    """
    加载 Milvus 向量库配置

    Args:
        config_path (str): 配置文件路径
        encoding (str): 文件编码

    Returns:
        dict: 配置字典，包含:
            - db_path: Milvus Lite 数据文件路径
            - collection_name: 集合名称
            - vector_dim: 向量维度
            - search: 检索配置 (vector_top_k, bm25_top_k, final_top_k)
            - rrf_k: RRF 融合参数
            - chunk: 分片配置
            - data_path: 数据路径

    Example:
        >>> conf = load_milvus_config()
        >>> print(conf["collection_name"])
        'uvais_knowledge'
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# ==================== 全局配置实例 ====================
# 模块加载时读取配置，后续可直接使用

rag_conf = load_rag_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
milvus_conf = load_milvus_config()  # Milvus 配置

# ==================== 环境变量覆盖 ====================
# 支持通过环境变量覆盖 Milvus URI 和模型路径（用于 Docker 部署）

if os.getenv("MILVUS_URI"):
    milvus_conf["uri"] = os.getenv("MILVUS_URI")
    # 日志将在 MilvusService 初始化时输出

if os.getenv("MILVUS_COLLECTION_NAME"):
    milvus_conf["collection_name"] = os.getenv("MILVUS_COLLECTION_NAME")

if os.getenv("BGE_M3_MODEL_PATH"):
    milvus_conf["model_path"] = os.getenv("BGE_M3_MODEL_PATH")

if os.getenv("RERANKER_MODEL_PATH"):
    milvus_conf["reranker"]["model_path"] = os.getenv("RERANKER_MODEL_PATH")


# ==================== 模块测试入口 ====================
if __name__ == '__main__':
    print("=== RAG 配置 ===")
    for k, v in rag_conf.items():
        print(f"  {k}: {v}")

    print("\n=== Milvus 配置 ===")
    for k, v in milvus_conf.items():
        print(f"  {k}: {v}")

    print("\n=== Prompts 配置 ===")
    for k, v in prompts_conf.items():
        print(f"  {k}: {v}")

    print("\n=== Agent 配置 ===")
    for k, v in agent_conf.items():
        print(f"  {k}: {v}")
