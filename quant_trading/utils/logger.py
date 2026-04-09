"""Logger utility module"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "量化之神",
    log_dir: str = "./logs",
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 3
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的备份文件数量

    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（带轮转）
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称，默认为"量化之神"

    Returns:
        日志记录器
    """
    if name:
        return logging.getLogger(f"量化之神.{name}")
    return logging.getLogger("量化之神")


# 预设的日志记录器
_default_logger: Optional[logging.Logger] = None


def get_default_logger() -> logging.Logger:
    """获取默认日志记录器（单例）"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger


# 导出函数
__all__ = ["setup_logger", "get_logger", "get_default_logger"]
