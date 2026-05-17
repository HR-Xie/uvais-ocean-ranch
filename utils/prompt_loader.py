"""
提示词加载模块
==============

负责从文件系统加载各类提示词模板，供 Agent 和 RAG 模块使用。

提示词类型:
    - 系统提示词 (main_prompt): 定义 Agent 的角色和行为准则
    - RAG 提示词 (rag_summarize): 定义知识检索与总结的格式
    - 报告提示词 (workorder_prompt): 定义巡检报告生成的格式

使用示例:
    >>> from utils.prompt_loader import load_system_prompts
    >>> prompt = load_system_prompts()
    >>> print(prompt[:100])

作者: UVAIS 开发团队
创建时间: 2026-04
"""

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompts() -> str:
    """
    加载系统提示词

    加载 Agent 的主系统提示词，定义其角色、能力和行为准则。

    Returns:
        str: 系统提示词内容

    Raises:
        KeyError: 配置文件中缺少 main_prompt_path 配置项
        Exception: 文件读取失败

    Example:
        >>> prompt = load_system_prompts()
        >>> print(prompt[:100])
    """
    # 从配置文件获取提示词路径
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error("[load_system_prompts] 配置文件中缺少 main_prompt_path 配置项")
        raise e

    # 读取提示词文件
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_system_prompts] 解析系统提示词出错: {str(e)}")
        raise e


def load_rag_prompts() -> str:
    """
    加载 RAG 总结提示词

    加载用于知识检索与总结的提示词模板。

    Returns:
        str: RAG 提示词内容

    Raises:
        KeyError: 配置文件中缺少 rag_summarize_prompt_path 配置项
        Exception: 文件读取失败

    Example:
        >>> prompt = load_rag_prompts()
        >>> print(prompt[:100])
    """
    # 从配置文件获取提示词路径
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error("[load_rag_prompts] 配置文件中缺少 rag_summarize_prompt_path 配置项")
        raise e

    # 读取提示词文件
    try:
        with open(rag_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] 解析 RAG 总结提示词出错: {str(e)}")
        raise e


def load_report_prompts() -> str:
    """
    加载巡检报告提示词

    加载用于生成结构化巡检报告/工单的提示词模板，自动注入当前日期时间。

    Returns:
        str: 报告生成提示词内容（已注入当前时间）

    Raises:
        KeyError: 配置文件中缺少 report_prompt_path 配置项
        Exception: 文件读取失败

    Example:
        >>> prompt = load_report_prompts()
        >>> print(prompt[:100])
    """
    from datetime import datetime

    # 从配置文件获取提示词路径
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error("[load_report_prompts] 配置文件中缺少 report_prompt_path 配置项")
        raise e

    # 读取提示词文件
    try:
        with open(report_prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
    except Exception as e:
        logger.error(f"[load_report_prompts] 解析报告生成提示词出错: {str(e)}")
        raise e

    # 注入当前日期时间，确保工单编号和时间为实时
    now = datetime.now()
    current_info = (
        f"\n\n【系统时间】当前时间为 {now.strftime('%Y')} 年 {now.strftime('%m')} 月 {now.strftime('%d')} 日，"
        f"{now.strftime('%H')}:{now.strftime('%M')}。"
        f"工单编号中的年月日必须使用当前日期 {now.strftime('%Y%m%d')}。"
    )
    return prompt + current_info


# ==================== 模块测试入口 ====================
if __name__ == '__main__':
    print("=== 系统提示词 ===")
    print(load_system_prompts()[:200] + "...\n")

    print("=== RAG 提示词 ===")
    print(load_rag_prompts()[:200] + "...\n")

    print("=== 报告提示词 ===")
    print(load_report_prompts()[:200] + "...")
