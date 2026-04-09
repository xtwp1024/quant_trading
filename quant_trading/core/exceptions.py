# -*- coding: utf-8 -*-
"""
自定义异常类层次结构 - Custom Exception Hierarchy

提供细粒度的异常类型，避免使用宽泛的 Exception 捕获。
"""

from typing import Optional, Any
from decimal import Decimal


class TitanException(Exception):
    """Titan V13 基础异常类"""

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        初始化异常

        Args:
            message: 错误消息
            details: 额外详细信息字典
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ==================== 配置相关异常 ====================

class ConfigurationError(TitanException):
    """配置错误基类"""


class ConfigValidationError(ConfigurationError):
    """配置验证失败"""

    def __init__(self, config_key: str, reason: str, provided_value: Any = None):
        message = f"配置验证失败: '{config_key}' - {reason}"
        details = {'key': config_key, 'reason': reason}
        if provided_value is not None:
            details['provided_value'] = str(provided_value)
        super().__init__(message, details)


class MissingConfigError(ConfigurationError):
    """缺少必需的配置项"""

    def __init__(self, config_key: str, section: Optional[str] = None):
        if section:
            message = f"缺少必需配置项: '{section}.{config_key}'"
        else:
            message = f"缺少必需配置项: '{config_key}'"
        super().__init__(message, {'key': config_key, 'section': section})


class InvalidConfigValueError(ConfigurationError):
    """配置值无效"""

    def __init__(self, config_key: str, invalid_value: Any, expected_format: str):
        message = f"配置值无效: '{config_key}' = {invalid_value}, 期望格式: {expected_format}"
        super().__init__(message, {
            'key': config_key,
            'invalid_value': str(invalid_value),
            'expected_format': expected_format
        })


# ==================== 数据库相关异常 ====================

class DatabaseError(TitanException):
    """数据库错误基类"""


class ConnectionPoolError(DatabaseError):
    """连接池错误"""

    def __init__(self, message: str, pool_size: Optional[int] = None):
        details = {'pool_size': pool_size} if pool_size else {}
        super().__init__(message, details)


class ConnectionTimeoutError(ConnectionPoolError):
    """连接超时"""

    def __init__(self, dsn: str, timeout: float):
        message = f"数据库连接超时: {timeout}秒"
        super().__init__(message, {'dsn': dsn, 'timeout': timeout})


class TransactionError(DatabaseError):
    """事务执行错误"""

    def __init__(self, operation: str, reason: str):
        message = f"事务失败: {operation} - {reason}"
        super().__init__(message, {'operation': operation, 'reason': reason})


class QueryExecutionError(DatabaseError):
    """查询执行错误"""

    def __init__(self, query: str, reason: str, params: Optional[tuple] = None):
        message = f"查询执行失败: {reason}"
        details = {'query': query[:100], 'reason': reason}  # 截断长查询
        if params:
            details['params'] = str(params)
        super().__init__(message, details)


class RecordNotFoundError(DatabaseError):
    """记录未找到"""

    def __init__(self, table: str, criteria: dict):
        message = f"记录未找到: {table}"
        super().__init__(message, {'table': table, 'criteria': criteria})


# ==================== 交易相关异常 ====================

class TradingError(TitanException):
    """交易错误基类"""


class OrderExecutionError(TradingError):
    """订单执行失败"""

    def __init__(self, symbol: str, side: str, reason: str, order_id: Optional[str] = None):
        message = f"订单执行失败: {symbol} {side} - {reason}"
        details = {'symbol': symbol, 'side': side, 'reason': reason}
        if order_id:
            details['order_id'] = order_id
        super().__init__(message, details)


class InsufficientBalanceError(TradingError):
    """余额不足"""

    def __init__(self, required: Decimal, available: Decimal, asset: str):
        message = f"余额不足: {asset}，需要 {required}，可用 {available}"
        super().__init__(message, {
            'asset': asset,
            'required': str(required),
            'available': str(available)
        })


class InvalidOrderError(TradingError):
    """无效订单参数"""

    def __init__(self, field: str, value: Any, reason: str):
        message = f"无效订单参数: {field} = {value} - {reason}"
        super().__init__(message, {'field': field, 'value': str(value), 'reason': reason})


class PositionLimitExceededError(TradingError):
    """持仓超限"""

    def __init__(self, symbol: str, current_size: Decimal, max_limit: Decimal):
        message = f"持仓超限: {symbol}，当前 {current_size}，上限 {max_limit}"
        super().__init__(message, {
            'symbol': symbol,
            'current': str(current_size),
            'limit': str(max_limit)
        })


# ==================== 市场数据相关异常 ====================

class MarketDataError(TitanException):
    """市场数据错误基类"""


class TickerNotFoundError(MarketDataError):
    """行情数据未找到"""

    def __init__(self, symbol: str, exchange: Optional[str] = None):
        if exchange:
            message = f"行情数据未找到: {symbol} @ {exchange}"
        else:
            message = f"行情数据未找到: {symbol}"
        super().__init__(message, {'symbol': symbol, 'exchange': exchange})


class InvalidSymbolError(MarketDataError):
    """无效的交易对"""

    def __init__(self, symbol: str, reason: str):
        message = f"无效交易对: '{symbol}' - {reason}"
        super().__init__(message, {'symbol': symbol, 'reason': reason})


class DataFeedError(MarketDataError):
    """数据源错误"""

    def __init__(self, feed_name: str, reason: str):
        message = f"数据源错误: {feed_name} - {reason}"
        super().__init__(message, {'feed': feed_name, 'reason': reason})


# ==================== 策略相关异常 ====================

class StrategyError(TitanException):
    """策略错误基类"""


class IndicatorCalculationError(StrategyError):
    """指标计算错误"""

    def __init__(self, indicator: str, reason: str):
        message = f"指标计算失败: {indicator} - {reason}"
        super().__init__(message, {'indicator': indicator, 'reason': reason})


class InsufficientDataError(StrategyError):
    """数据不足"""

    def __init__(self, required_length: int, actual_length: int, context: str):
        message = f"数据不足: {context}，需要 {required_length} 条，实际 {actual_length} 条"
        super().__init__(message, {
            'required': required_length,
            'actual': actual_length,
            'context': context
        })


class SignalGenerationError(StrategyError):
    """信号生成失败"""

    def __init__(self, strategy: str, reason: str):
        message = f"信号生成失败: {strategy} - {reason}"
        super().__init__(message, {'strategy': strategy, 'reason': reason})


# ==================== 风险管理相关异常 ====================

class RiskError(TitanException):
    """风险管理错误基类"""


class RiskLimitExceededError(RiskError):
    """风险限制超限"""

    def __init__(self, limit_type: str, current_value: Decimal, max_limit: Decimal):
        message = f"风险限制超限: {limit_type}，当前 {current_value}，上限 {max_limit}"
        super().__init__(message, {
            'type': limit_type,
            'current': str(current_value),
            'limit': str(max_limit)
        })


class DailyLossLimitExceededError(RiskError):
    """每日亏损超限"""

    def __init__(self, current_loss: Decimal, max_limit: Decimal, date: str):
        message = f"每日亏损超限: {date}，亏损 {current_loss}，上限 {max_limit}"
        super().__init__(message, {
            'date': date,
            'current_loss': str(current_loss),
            'max_limit': str(max_limit)
        })


class CircuitBreakerTriggeredError(RiskError):
    """熔断器触发"""

    def __init__(self, reason: str, cooldown_seconds: Optional[int] = None):
        message = f"熔断器触发: {reason}"
        details = {'reason': reason}
        if cooldown_seconds:
            details['cooldown_seconds'] = cooldown_seconds
        super().__init__(message, details)


class LeverageLimitExceededError(RiskError):
    """杠杆超限"""

    def __init__(self, current_leverage: int, max_leverage: int):
        message = f"杠杆超限: 当前 {current_leverage}x，最大 {max_leverage}x"
        super().__init__(message, {
            'current': current_leverage,
            'maximum': max_leverage
        })


# ==================== API 相关异常 ====================

class APIError(TitanException):
    """API 错误基类"""


class ExchangeAPIError(APIError):
    """交易所 API 错误"""

    def __init__(self, exchange: str, endpoint: str, status_code: Optional[int] = None, reason: str = ""):
        message = f"交易所API错误: {exchange} - {endpoint}"
        if status_code:
            message += f" (HTTP {status_code})"
        if reason:
            message += f" - {reason}"
        super().__init__(message, {
            'exchange': exchange,
            'endpoint': endpoint,
            'status_code': status_code,
            'reason': reason
        })


class RateLimitError(APIError):
    """API 速率限制"""

    def __init__(self, exchange: str, retry_after: Optional[int] = None):
        message = f"API 速率限制: {exchange}"
        if retry_after:
            message += f"，{retry_after}秒后重试"
        super().__init__(message, {
            'exchange': exchange,
            'retry_after': retry_after
        })


class AuthenticationError(APIError):
    """认证失败"""

    def __init__(self, exchange: str, reason: str = "Invalid credentials"):
        message = f"认证失败: {exchange} - {reason}"
        super().__init__(message, {'exchange': exchange, 'reason': reason})


class NetworkTimeoutError(APIError):
    """网络超时"""

    def __init__(self, url: str, timeout: float):
        message = f"网络超时: {url} (>{timeout}秒)"
        super().__init__(message, {'url': url, 'timeout': timeout})


# ==================== 工具函数 ====================

def handle_exception(
    exc: Exception,
    context: str,
    raise_if_critical: bool = True,
    log_level: str = "error"
) -> None:
    """
    统一异常处理函数

    Args:
        exc: 捕获的异常
        context: 错误上下文描述
        raise_if_critical: 是否重新抛出严重异常
        log_level: 日志级别 (error, warning, info)

    Example:
        try:
            risky_operation()
        except TitanException as e:
            handle_exception(e, "执行交易", raise_if_critical=True)
        except Exception as e:
            # 转换未知异常为 TitanException
            handle_exception(
                TitanException(f"未知错误: {e}"),
                "执行交易"
            )
    """
    from core.logger import logger

    # 记录日志
    log_msg = f"[{context}] {exc}"
    if log_level == "error":
        logger.error(log_msg)
    elif log_level == "warning":
        logger.warning(log_msg)
    else:
        logger.info(log_msg)

    # 严重异常重新抛出
    if raise_if_critical:
        critical_exceptions = (
            DailyLossLimitExceededError,
            CircuitBreakerTriggeredError,
            AuthenticationError
        )
        if isinstance(exc, critical_exceptions):
            raise
