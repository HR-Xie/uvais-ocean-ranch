"""提示词模板加载"""

from datetime import datetime
from utils.config import prompts_conf, get_abs_path
from utils.logging import logger


def _load(key: str) -> str:
    path = get_abs_path(prompts_conf[key])
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"[prompts] 读取失败: {path}: {e}")
        raise


def load_system_prompts() -> str:
    return _load("main_prompt_path")


def load_rag_prompts() -> str:
    return _load("rag_summarize_prompt_path")


def load_report_prompts() -> str:
    prompt = _load("report_prompt_path")
    now = datetime.now()
    return prompt + (
        f"\n\n【系统时间】当前时间为 {now.strftime('%Y')} 年 {now.strftime('%m')} 月 "
        f"{now.strftime('%d')} 日，{now.strftime('%H')}:{now.strftime('%M')}。"
        f"工单编号中的年月日必须使用当前日期 {now.strftime('%Y%m%d')}。"
    )
