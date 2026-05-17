"""
文件处理工具模块
================

提供文件操作相关的工具函数，包括：
- MD5 哈希计算：用于文档去重
- 目录遍历：获取指定类型的文件列表
- 文档加载：PDF/TXT 文件内容提取

使用示例:
    >>> from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type
    >>> md5 = get_file_md5_hex("data/document.txt")
    >>> files = listdir_with_allowed_type("data/", (".txt", ".pdf"))

作者: UVAIS 开发团队
创建时间: 2026-04
"""

import os
import hashlib
from typing import Tuple, List, Optional

from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(filepath: str) -> Optional[str]:
    """
    计算文件的 MD5 十六进制哈希值

    用于知识库文档去重，避免重复加载相同内容的文件。
    采用 4KB 分块读取，支持大文件处理。

    Args:
        filepath (str): 目标文件的绝对路径或相对路径

    Returns:
        Optional[str]: 文件的 MD5 十六进制字符串 (32位小写)。
                      如果文件不存在或读取失败，返回 None。

    Example:
        >>> md5 = get_file_md5_hex("data/SOP_网箱维修.txt")
        >>> print(md5)
        'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'
    """
    # 验证文件存在性
    if not os.path.exists(filepath):
        logger.error(f"[md5计算] 文件 {filepath} 不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算] 路径 {filepath} 不是文件")
        return None

    md5_obj = hashlib.md5()

    # 4KB 分块读取，避免大文件导致内存溢出
    chunk_size = 4096
    try:
        with open(filepath, "rb") as f:  # 必须以二进制模式读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

        # 返回 32 位十六进制哈希字符串
        md5_hex = md5_obj.hexdigest()
        return md5_hex

    except Exception as e:
        logger.error(f"计算文件 {filepath} MD5 失败: {str(e)}")
        return None


def listdir_with_allowed_type(
    path: str,
    allowed_types: Tuple[str, ...]
) -> Tuple[str, ...]:
    """
    获取目录内指定类型的文件列表

    遍历指定目录，筛选出符合允许后缀名的文件路径。

    Args:
        path (str): 目标目录路径
        allowed_types (Tuple[str, ...]): 允许的文件后缀名元组，如 (".txt", ".pdf")

    Returns:
        Tuple[str, ...]: 符合条件的文件绝对路径元组

    Example:
        >>> files = listdir_with_allowed_type("data/", (".txt", ".pdf"))
        >>> print(files)
        ('data/document1.txt', 'data/document2.pdf')

    Note:
        - 如果目录不存在，返回空元组
        - 返回的是元组而非列表，便于作为不可变序列使用
    """
    files = []

    # 验证目录存在性
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type] {path} 不是有效的文件夹")
        # 【修复】返回空元组而非 allowed_types
        return tuple()

    # 遍历目录，筛选符合后缀名的文件
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


def pdf_loader(filepath: str, passwd: Optional[str] = None) -> List[Document]:
    """
    加载 PDF 文件内容

    使用 LangChain 的 PyPDFLoader 解析 PDF 文件，
    将每页内容转换为独立的 Document 对象。

    Args:
        filepath (str): PDF 文件的绝对路径
        passwd (Optional[str]): PDF 密码（如果有加密）

    Returns:
        List[Document]: 文档对象列表，每个对象包含页面内容和元数据

    Example:
        >>> docs = pdf_loader("data/manual.pdf")
        >>> print(f"共 {len(docs)} 页")
        >>> print(docs[0].page_content[:100])  # 打印第一页前100字
    """
    return PyPDFLoader(filepath, password=passwd).load()


def txt_loader(filepath: str) -> List[Document]:
    """
    加载 TXT 文本文件内容

    使用 LangChain 的 TextLoader 读取文本文件，
    将内容转换为 Document 对象。

    Args:
        filepath (str): TXT 文件的绝对路径

    Returns:
        List[Document]: 包含单个文档对象的列表

    Note:
        默认使用 UTF-8 编码读取，如遇乱码请检查文件编码
    """
    return TextLoader(filepath, encoding="utf-8").load()


# ==================== 模块测试入口 ====================
if __name__ == '__main__':
    # 测试 MD5 计算
    print("测试 MD5 计算...")
    test_md5 = get_file_md5_hex(__file__)  # 计算当前文件的 MD5
    print(f"当前文件 MD5: {test_md5}")

    # 测试目录遍历
    print("\n测试目录遍历...")
    test_files = listdir_with_allowed_type("data/", (".txt", ".pdf"))
    print(f"找到 {len(test_files)} 个文件: {test_files}")
