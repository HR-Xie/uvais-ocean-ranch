"""双通道日志：控制台(INFO+) + 文件(DEBUG+)"""

import logging
import os
from datetime import datetime

from utils.config import get_abs_path

LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
    name: str = "agent",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: str = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    if not log_file:
        log_file = os.path.join(
            LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()


if __name__ == '__main__':
    logger.debug("DEBUG - 仅写入文件")
    logger.info("INFO - 控制台+文件")
    logger.warning("WARNING")
    logger.error("ERROR")
    print(f"\n日志目录: {LOG_ROOT}")
