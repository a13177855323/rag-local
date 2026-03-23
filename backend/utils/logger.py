"""
日志工具模块
提供统一的日志格式，包含错误等级、时间戳、模块标识和详细描述
遵循项目统一的日志格式要求
"""

import logging
import sys
from datetime import datetime
from typing import Optional
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RAGLogger:
    """RAG系统统一日志记录器 - 单例模式"""
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化日志记录器"""
        self._logger = logging.getLogger("RAG")
        self._logger.setLevel(logging.DEBUG)

        # 避免重复添加处理器
        if not self._logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)

            # 统一格式: [时间戳] [级别] [模块] 消息
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

    def _format_message(self, module: str, message: str, details: Optional[dict] = None) -> str:
        """
        格式化日志消息

        Args:
            module: 模块标识
            message: 主要消息
            details: 详细描述字典

        Returns:
            格式化后的消息
        """
        formatted = f"[{module}] {message}"
        if details:
            details_str = " | ".join([f"{k}={v}" for k, v in details.items()])
            formatted += f" | {details_str}"
        return formatted

    def debug(self, module: str, message: str, details: Optional[dict] = None):
        """记录DEBUG级别日志"""
        self._logger.debug(self._format_message(module, message, details))

    def info(self, module: str, message: str, details: Optional[dict] = None):
        """记录INFO级别日志"""
        self._logger.info(self._format_message(module, message, details))

    def warning(self, module: str, message: str, details: Optional[dict] = None):
        """记录WARNING级别日志"""
        self._logger.warning(self._format_message(module, message, details))

    def error(self, module: str, message: str, details: Optional[dict] = None):
        """记录ERROR级别日志"""
        self._logger.error(self._format_message(module, message, details))

    def critical(self, module: str, message: str, details: Optional[dict] = None):
        """记录CRITICAL级别日志"""
        self._logger.critical(self._format_message(module, message, details))

    def log_exception(self, module: str, exception: Exception, context: Optional[dict] = None):
        """
        记录异常信息

        Args:
            module: 模块标识
            exception: 异常对象
            context: 上下文信息
        """
        details = {
            "exception_type": type(exception).__name__,
            "exception_msg": str(exception)
        }
        if context:
            details.update(context)
        self.error(module, f"Exception occurred", details)


# 全局单例实例
_logger_instance = None


def get_logger() -> RAGLogger:
    """获取 RAGLogger 单例实例"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = RAGLogger()
    return _logger_instance


def log_info(module: str, message: str, details: Optional[dict] = None):
    """便捷函数：记录INFO日志"""
    get_logger().info(module, message, details)


def log_error(module: str, message: str, details: Optional[dict] = None):
    """便捷函数：记录ERROR日志"""
    get_logger().error(module, message, details)


def log_warning(module: str, message: str, details: Optional[dict] = None):
    """便捷函数：记录WARNING日志"""
    get_logger().warning(module, message, details)


def log_debug(module: str, message: str, details: Optional[dict] = None):
    """便捷函数：记录DEBUG日志"""
    get_logger().debug(module, message, details)
