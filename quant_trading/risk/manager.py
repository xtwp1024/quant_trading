"""Risk Manager - Centralized risk management for the trading system"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskConfig:
    """风险配置"""
    max_position_size: float = 1000.0
    max_single_loss: float = 100.0
    max_daily_loss: float = 500.0
    max_daily_trades: int = 10
    max_portfolio_exposure: float = 0.5  # 最大仓位暴露比例
    stop_loss_pct: float = 0.02  # 2%止损
    take_profit_pct: float = 0.05  # 5%止盈
    cooldown_after_loss: int = 1800000  # 30分钟冷却


@dataclass
class RiskMetrics:
    """风险指标"""
    current_exposure: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    max_drawdown: float = 0.0
    current_risk_level: RiskLevel = RiskLevel.LOW


class RiskManager:
    """风险管理器"""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.metrics = RiskMetrics()
        self.trade_log: List[Dict] = []
        self._last_trade_time = 0

    def check_position_size(self, symbol: str, size: float, price: float) -> Dict[str, Any]:
        """
        检查仓位大小是否合规

        Returns:
            Dict with 'allowed' (bool) and 'reason' (str)
        """
        position_value = size * price

        # 检查单笔仓位大小
        if position_value > self.config.max_position_size:
            return {
                "allowed": False,
                "reason": f"仓位价值 {position_value:.2f} 超过限制 {self.config.max_position_size:.2f}"
            }

        # 检查总暴露度
        new_exposure = self.metrics.current_exposure + position_value
        if new_exposure > self.config.max_portfolio_exposure * self.config.max_position_size * 10:
            return {
                "allowed": False,
                "reason": f"总暴露度 {new_exposure:.2f} 超过限制"
            }

        return {"allowed": True}

    def check_trade_allowed(self, symbol: str, potential_loss: float = 0) -> Dict[str, Any]:
        """
        检查是否允许交易

        Returns:
            Dict with 'allowed' (bool) and 'reason' (str)
        """
        # 检查冷却期
        if time.time() * 1000 - self._last_trade_time < self.config.cooldown_after_loss:
            remaining = (self.config.cooldown_after_loss - (time.time() * 1000 - self._last_trade_time)) / 60000
            return {
                "allowed": False,
                "reason": f"冷却期中，还需等待 {remaining:.1f} 分钟"
            }

        # 检查日亏损
        if abs(self.metrics.daily_pnl) > self.config.max_daily_loss:
            return {
                "allowed": False,
                "reason": f"日亏损 {abs(self.metrics.daily_pnl):.2f} 超过限制 {self.config.max_daily_loss:.2f}"
            }

        # 检查日交易次数
        if self.metrics.daily_trades >= self.config.max_daily_trades:
            return {
                "allowed": False,
                "reason": f"日交易次数 {self.metrics.daily_trades} 达到限制"
            }

        # 检查潜在亏损
        if potential_loss < -self.config.max_single_loss:
            return {
                "allowed": False,
                "reason": f"潜在亏损 {potential_loss:.2f} 超过单笔限制"
            }

        # 检查连续亏损
        if self.metrics.consecutive_losses >= 3:
            return {
                "allowed": False,
                "reason": f"连续亏损 {self.metrics.consecutive_losses} 次，强制停止"
            }

        return {"allowed": True}

    def calculate_stop_loss(self, entry_price: float, position_type: str) -> float:
        """计算止损价"""
        if position_type.lower() == "long":
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, position_type: str) -> float:
        """计算止盈价"""
        if position_type.lower() == "long":
            return entry_price * (1 + self.config.take_profit_pct)
        else:
            return entry_price * (1 - self.config.take_profit_pct)

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """记录交易并更新风险指标"""
        self.trade_log.append({
            **trade,
            "timestamp": int(time.time() * 1000)
        })
        self._last_trade_time = time.time() * 1000
        self.metrics.daily_trades += 1

        # 更新盈亏
        if "pnl" in trade:
            self.metrics.daily_pnl += trade["pnl"]

            # 更新连续亏损
            if trade["pnl"] < 0:
                self.metrics.consecutive_losses += 1
            else:
                self.metrics.consecutive_losses = 0

        # 更新暴露度
        if "value" in trade:
            self.metrics.current_exposure += trade["value"]

    def update_risk_level(self) -> RiskLevel:
        """更新风险等级"""
        if self.metrics.consecutive_losses >= 5 or abs(self.metrics.daily_pnl) > self.config.max_daily_loss * 2:
            self.metrics.current_risk_level = RiskLevel.CRITICAL
        elif self.metrics.consecutive_losses >= 3 or abs(self.metrics.daily_pnl) > self.config.max_daily_loss:
            self.metrics.current_risk_level = RiskLevel.HIGH
        elif self.metrics.consecutive_losses >= 2 or abs(self.metrics.daily_pnl) > self.config.max_daily_loss * 0.5:
            self.metrics.current_risk_level = RiskLevel.MEDIUM
        else:
            self.metrics.current_risk_level = RiskLevel.LOW

        return self.metrics.current_risk_level

    def get_risk_report(self) -> Dict[str, Any]:
        """获取风险报告"""
        return {
            "metrics": {
                "current_exposure": self.metrics.current_exposure,
                "daily_pnl": self.metrics.daily_pnl,
                "daily_trades": self.metrics.daily_trades,
                "consecutive_losses": self.metrics.consecutive_losses,
                "max_drawdown": self.metrics.max_drawdown,
                "risk_level": self.metrics.current_risk_level.value
            },
            "config": {
                "max_position_size": self.config.max_position_size,
                "max_single_loss": self.config.max_single_loss,
                "max_daily_loss": self.config.max_daily_loss,
                "max_daily_trades": self.config.max_daily_trades,
                "stop_loss_pct": self.config.stop_loss_pct,
                "take_profit_pct": self.config.take_profit_pct
            }
        }

    def reset_daily(self) -> None:
        """重置日度风险指标"""
        self.metrics.daily_pnl = 0.0
        self.metrics.daily_trades = 0
        self.metrics.consecutive_losses = 0


# 导出主要类
__all__ = ["RiskManager", "RiskConfig", "RiskMetrics", "RiskLevel"]
