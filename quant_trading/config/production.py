# -*- coding: utf-8 -*-
"""
Production Configuration
========================

生产环境配置 - 包含API密钥管理、风险限制、熔断机制和紧急止损配置

IMPORTANT: 所有敏感配置通过环境变量管理，禁止硬编码

环境变量:
    BINANCE_API_KEY - Binance API密钥
    BINANCE_API_SECRET - Binance API密钥密码
    TELEGRAM_BOT_TOKEN - Telegram机器人Token (可选)
    TELEGRAM_CHAT_ID - Telegram聊天ID (可选)
    SLACK_WEBHOOK_URL - Slack Webhook URL (可选)
    PRODUCTION_MODE - 设置为"true"启用生产模式
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class TradingMode(Enum):
    """交易模式"""
    DRY_RUN = "dry_run"      # 干跑模式 - 仅监控，不交易
    PAPER = "paper"          # Paper交易 - 模拟执行
    LIVE = "live"            # 实盘交易


class RiskLevel(Enum):
    """风险等级"""
    MINIMAL = "minimal"      # 最小风险
    LOW = "low"             # 低风险
    MEDIUM = "medium"       # 中风险
    HIGH = "high"           # 高风险
    CRITICAL = "critical"   # 严重风险


# ===================== API密钥管理 =====================

def get_api_credentials() -> Dict[str, str]:
    """
    从环境变量获取API凭证

    Returns:
        Dict with 'api_key', 'api_secret', 'testnet'
    """
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    # 检查是否为测试网
    testnet = os.environ.get("BINANCE_TESTNET", "false").lower() == "true"

    # 生产模式检查
    production_mode = os.environ.get("PRODUCTION_MODE", "false").lower() == "true"

    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "testnet": testnet,
        "production_mode": production_mode,
    }


def validate_api_credentials(creds: Dict[str, str]) -> List[str]:
    """
    验证API凭证

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not creds.get("api_key"):
        errors.append("BINANCE_API_KEY is not set")

    if not creds.get("api_secret"):
        errors.append("BINANCE_API_SECRET is not set")

    if len(creds.get("api_key", "")) < 10:
        errors.append("BINANCE_API_KEY appears to be invalid")

    return errors


# ===================== 风险配置 =====================

@dataclass
class PositionLimits:
    """仓位限制"""
    max_position_size: float = 1000.0       # 单币种最大持仓价值(USDT)
    max_total_exposure: float = 0.6         # 总仓位上限 (60% of capital)
    max_single_trade_pct: float = 0.15      # 单笔交易最大占比 (15% of capital)
    min_trade_interval_seconds: int = 300    # 最小交易间隔 (5分钟)
    max_positions: int = 4                  # 最大同时持仓数


@dataclass
class LossLimits:
    """亏损限制"""
    max_daily_loss: float = 500.0           # 日最大亏损(USDT)
    max_single_loss: float = 100.0          # 单笔最大亏损(USDT)
    max_consecutive_losses: int = 3         # 连续亏损熔断次数
    max_drawdown_pct: float = 0.10          # 最大回撤限制 (10%)
    daily_loss_limit_pct: float = 0.05      # 日亏损限制 (5% of capital)
    circuit_break_after_losses: int = 3     # 连续亏损后熔断次数


@dataclass
class CircuitBreakerConfig:
    """熔断机制配置"""
    enabled: bool = True
    # 价格异常熔断
    price_change_threshold: float = 0.05     # 5分钟内价格变化超过5%则熔断
    # 成交量异常熔断
    volume_spike_multiplier: float = 5.0     # 成交量超过5倍均值则熔断
    # 连接异常熔断
    max_retry_attempts: int = 3             # 最大重试次数
    retry_cooldown_seconds: int = 60         # 重试冷却时间
    # API限流熔断
    rate_limit_threshold: int = 1200         # 1分钟最大请求数
    rate_limit_window_seconds: int = 60      # 限流时间窗口


@dataclass
class EmergencyStopConfig:
    """紧急止损配置"""
    enabled: bool = True
    # 自动平仓阈值
    auto_close_loss_pct: float = 0.08        # 亏损8%自动平仓
    auto_close_profit_pct: float = 0.15     # 盈利15%自动平仓
    # 手动终止
    kill_switch_enabled: bool = True         # Kill Switch开关
    kill_switch_key: str = "KILL"           # 终止密钥
    # 通知
    notify_on_emergency: bool = True         # 紧急情况通知
    notify_on_circuit_break: bool = True    # 熔断通知
    # 冷却期
    emergency_cooldown_minutes: int = 30     # 紧急停止后的冷却期


@dataclass
class TradeApprovalConfig:
    """交易审批配置"""
    # 大额交易审批
    large_trade_threshold: float = 500.0     # 大额交易阈值(USDT)
    require_approval_above: float = 1000.0  # 需要审批的交易阈值
    approval_notification: bool = True       # 发送审批通知
    # 高风险交易审批
    high_risk_approval_required: bool = True  # 高风险交易需审批


@dataclass
class ProductionRiskConfig:
    """生产环境风险配置"""
    # 模式
    trading_mode: TradingMode = TradingMode.DRY_RUN

    # 仓位限制
    position: PositionLimits = field(default_factory=PositionLimits)

    # 亏损限制
    loss: LossLimits = field(default_factory=LossLimits)

    # 熔断机制
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # 紧急止损
    emergency_stop: EmergencyStopConfig = field(default_factory=EmergencyStopConfig)

    # 交易审批
    trade_approval: TradeApprovalConfig = field(default_factory=TradeApprovalConfig)

    # 风控参数
    stop_loss_pct: float = 0.03             # 止损 3%
    take_profit_pct: float = 0.07           # 止盈 7%
    time_stop_hours: int = 14               # 时间止损 14小时

    # 监控参数
    health_check_interval: int = 60         # 健康检查间隔(秒)
    metrics_log_interval: int = 300         # 指标日志间隔(秒)

    # 调试模式
    debug_mode: bool = False                # 调试模式开关
    verbose_logging: bool = False            # 详细日志


# ===================== 告警配置 =====================

@dataclass
class AlertConfig:
    """告警配置"""
    enabled: bool = True

    # Telegram配置
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Slack配置
    slack_enabled: bool = False
    slack_webhook_url: str = ""

    # 邮件配置 (可选)
    email_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_recipients: List[str] = field(default_factory=list)

    def load_from_env(self) -> "AlertConfig":
        """从环境变量加载告警配置"""
        self.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.telegram_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)

        self.slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        self.slack_enabled = bool(self.slack_webhook_url)

        return self


# ===================== 生产配置 =====================

class ProductionConfig:
    """生产环境完整配置"""

    def __init__(self):
        # 加载API凭证
        self.credentials = get_api_credentials()

        # 风险配置
        self.risk = ProductionRiskConfig()

        # 告警配置
        self.alerts = AlertConfig().load_from_env()

        # 数据库配置
        self.database_path = os.environ.get(
            "TRADING_DB_PATH",
            "data/production_trading.db"
        )

        # 日志配置
        self.log_path = os.environ.get(
            "TRADING_LOG_PATH",
            "logs/production_trading.log"
        )

    def validate(self) -> List[str]:
        """
        验证配置完整性

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # 验证API凭证
        cred_errors = validate_api_credentials(self.credentials)
        errors.extend([f"API Credentials: {e}" for e in cred_errors])

        # 验证风险配置
        if self.risk.trading_mode == TradingMode.LIVE:
            if not self.credentials.get("api_key"):
                errors.append("Cannot run in LIVE mode without API credentials")

        # 验证告警配置
        if self.risk.emergency_stop.enabled and not self.alerts.enabled:
            errors.append("Emergency stop is enabled but alerts are disabled")

        return errors

    def is_safe_to_trade(self) -> bool:
        """检查是否安全交易"""
        if self.risk.trading_mode == TradingMode.LIVE:
            return (
                self.credentials.get("production_mode", False) and
                len(self.validate()) == 0
            )
        return True


# ===================== 辅助函数 =====================

def get_production_config() -> ProductionConfig:
    """获取生产配置单例"""
    return ProductionConfig()


def print_config_summary(config: ProductionConfig) -> None:
    """打印配置摘要"""
    print("=" * 60)
    print("Production Configuration Summary")
    print("=" * 60)
    print(f"Trading Mode: {config.risk.trading_mode.value}")
    print(f"API Testnet: {config.credentials.get('testnet', False)}")
    print(f"Production Mode: {config.credentials.get('production_mode', False)}")
    print()
    print("Risk Limits:")
    print(f"  Max Position Size: ${config.risk.position.max_position_size}")
    print(f"  Max Total Exposure: {config.risk.position.max_total_exposure:.0%}")
    print(f"  Max Single Trade: {config.risk.position.max_single_trade_pct:.0%}")
    print(f"  Max Positions: {config.risk.position.max_positions}")
    print()
    print(f"Loss Limits:")
    print(f"  Max Daily Loss: ${config.risk.loss.max_daily_loss}")
    print(f"  Max Single Loss: ${config.risk.loss.max_single_loss}")
    print(f"  Circuit Break After: {config.risk.loss.max_consecutive_losses} losses")
    print(f"  Max Drawdown: {config.risk.loss.max_drawdown_pct:.0%}")
    print()
    print(f"Circuit Breaker: {'Enabled' if config.risk.circuit_breaker.enabled else 'Disabled'}")
    print(f"Emergency Stop: {'Enabled' if config.risk.emergency_stop.enabled else 'Disabled'}")
    print()
    print(f"Telegram Alerts: {'Enabled' if config.alerts.telegram_enabled else 'Disabled'}")
    print(f"Slack Alerts: {'Enabled' if config.alerts.slack_enabled else 'Disabled'}")
    print("=" * 60)


# 导出
__all__ = [
    "ProductionConfig",
    "ProductionRiskConfig",
    "PositionLimits",
    "LossLimits",
    "CircuitBreakerConfig",
    "EmergencyStopConfig",
    "TradeApprovalConfig",
    "AlertConfig",
    "TradingMode",
    "RiskLevel",
    "get_api_credentials",
    "validate_api_credentials",
    "get_production_config",
    "print_config_summary",
]
