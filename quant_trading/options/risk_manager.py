"""
期权风险管理 - 从 Options-Trading-Bot 仓库吸收 + OptionSuite 持仓周期管理
Phase 3: 风险管理
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum

import numpy as np

from .pricing.greeks import calculate_greeks


__all__ = [
    # Enums
    "RiskLimitType",
    "OptionsStrategyType",
    "PositionLifecycle",
    # Dataclasses
    "RiskLimit",
    "RiskMetrics",
    "PositionLifecycleData",
    # Managers
    "PositionCycleManager",
    "RiskManager",
    "PositionSizer",
    "StopLoss",
    # New from OptionSuite
    "PutVerticalRiskManager",
    "StrangleRiskManager",
    # Aliases
    "OptionsRiskManager",
]


# ============================================================================
# 风险限制类型
# ============================================================================

class RiskLimitType(Enum):
    """风险限制类型"""
    MAX_POSITION_SIZE = "max_position_size"
    MAX_LOSS_PER_TRADE = "max_loss_per_trade"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_DELTA = "max_delta"
    MAX_GAMMA = "max_gamma"
    MAX_VEGA = "max_vega"
    MAX_THETA = "max_theta"
    MAX_PORTFOLIO_LOSS = "max_portfolio_loss"
    MAX_DRAWDOWN = "max_drawdown"
    VAR_LIMIT = "var_limit"


# ============================================================================
# 风险限制
# ============================================================================

@dataclass
class RiskLimit:
    """风险限制"""
    limit_type: RiskLimitType
    value: float
    action: str = "CLOSE"  # "CLOSE", "REDUCE", "WARN"

    def check(self, current_value: float) -> bool:
        """是否超出限制"""
        return abs(current_value) > self.value


# ============================================================================
# 风险指标
# ============================================================================

@dataclass
class RiskMetrics:
    """风险指标"""
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0
    position_count: int = 0
    total_premium: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    var: float = 0.0  # Value at Risk
    max_drawdown: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "delta": self.portfolio_delta,
            "gamma": self.portfolio_gamma,
            "vega": self.portfolio_vega,
            "theta": self.portfolio_theta,
            "position_count": self.position_count,
            "total_premium": self.total_premium,
            "unrealized_pnl": self.unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "var": self.var,
            "max_drawdown": self.max_drawdown,
        }


# ============================================================================
# 持仓周期管理（OptionSuite 核心功能）
# ============================================================================

@dataclass
class PositionLifecycleData:
    """
    持仓生命周期跟踪

    从 OptionSuite 吸收：持仓周期管理功能
    """
    entry_time: datetime
    last_adjustment_time: datetime
    entry_price: float
    adjustments: List[Dict] = field(default_factory=list)
    max_holding_period: timedelta = field(default_factory=lambda: timedelta(days=21))
    target_exit_dte: int = 14  # 目标在 DTE <= 14 时平仓

    def days_held(self, current_time: datetime = None) -> float:
        """持仓天数"""
        if current_time is None:
            current_time = datetime.now()
        return (current_time - self.entry_time).total_seconds() / 86400

    def dte_threshold_check(self, current_dte: int) -> bool:
        """检查是否达到 DTE 阈值"""
        return current_dte <= self.target_exit_dte

    def holding_period_check(self, current_time: datetime = None) -> bool:
        """检查是否超过最大持仓期"""
        return self.days_held(current_time) >= self.max_holding_period.total_seconds() / 86400

    def should_roll(self, current_dte: int, current_time: datetime = None) -> bool:
        """是否应该滚动（移仓）"""
        # 超过持仓期 或 DTE 过低
        return self.holding_period_check(current_time) or current_dte <= 7

    def add_adjustment(self, adjustment_type: str, details: Dict):
        """记录调整"""
        self.adjustments.append({
            "time": datetime.now(),
            "type": adjustment_type,
            "details": details,
        })
        self.last_adjustment_time = datetime.now()


class PositionCycleManager:
    """
    持仓周期管理器

    功能：
    - 跟踪每个持仓的生命周期
    - 触发自动平仓/滚动
    - 管理仓位调整历史
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.positions: Dict[str, PositionLifecycleData] = {}
        self.max_holding_days = self.config.get("max_holding_days", 21)
        self.target_dte_exit = self.config.get("target_dte_exit", 14)
        self.roll_threshold_dte = self.config.get("roll_threshold_dte", 7)

    def track_position(
        self,
        position_id: str,
        entry_time: datetime,
        entry_price: float,
    ):
        """开始跟踪一个持仓"""
        self.positions[position_id] = PositionLifecycleData(
            entry_time=entry_time,
            last_adjustment_time=entry_time,
            entry_price=entry_price,
            max_holding_period=timedelta(days=self.max_holding_days),
            target_exit_dte=self.target_dte_exit,
        )

    def untrack_position(self, position_id: str):
        """停止跟踪一个持仓"""
        if position_id in self.positions:
            del self.positions[position_id]

    def record_adjustment(self, position_id: str, adjustment_type: str, details: Dict):
        """记录持仓调整"""
        if position_id in self.positions:
            self.positions[position_id].add_adjustment(adjustment_type, details)

    def check_exit_signals(
        self,
        position_id: str,
        current_dte: int,
        current_time: datetime = None,
    ) -> Dict:
        """
        检查是否应该退出

        Returns:
            {"should_exit": bool, "reason": str, "action": str}
        """
        if position_id not in self.positions:
            return {"should_exit": False, "reason": None, "action": "UNKNOWN"}

        lifecycle = self.positions[position_id]

        # 1. DTE 过低检查
        if current_dte <= 1:
            return {
                "should_exit": True,
                "reason": "DTE_TOO_LOW",
                "action": "CLOSE",
            }

        # 2. 超过目标 DTE
        if lifecycle.dte_threshold_check(current_dte):
            return {
                "should_exit": True,
                "reason": "TARGET_DTE_REACHED",
                "action": "CLOSE",
            }

        # 3. 超过最大持仓期
        if lifecycle.holding_period_check(current_time):
            return {
                "should_exit": True,
                "reason": "MAX_HOLDING_PERIOD",
                "action": "CLOSE_OR_ROLL",
            }

        # 4. 应该滚动检查
        if lifecycle.should_roll(current_dte, current_time):
            return {
                "should_exit": True,
                "reason": "SHOULD_ROLL",
                "action": "ROLL",
            }

        return {"should_exit": False, "reason": None, "action": "HOLD"}

    def get_position_age(self, position_id: str, current_time: datetime = None) -> float:
        """获取持仓年龄（天）"""
        if position_id not in self.positions:
            return 0.0
        return self.positions[position_id].days_held(current_time)


# ============================================================================
# 风险管理器
# ============================================================================

class RiskManager:
    """
    期权风险管理器

    功能：
    - 实时风险监控
    - Greeks 敞口限制
    - 止损/止盈
    - 仓位限制
    - VaR 计算
    - 最大回撤检查
    - 持仓周期管理
    """

    def __init__(self, limits: List[RiskLimit] = None, config: Dict = None):
        self.config = config or {}
        self.limits = limits or []
        self.alerts: List[Dict] = []
        self.daily_loss = 0.0
        self.peak_equity = 0.0
        self.equity_curve: List[float] = []

        # 持仓周期管理器
        self.cycle_manager = PositionCycleManager(self.config.get("cycle_manager", {}))

        # 默认限制
        self._setup_default_limits()

    def _setup_default_limits(self):
        """设置默认限制"""
        if not self.limits:
            self.limits = [
                RiskLimit(RiskLimitType.MAX_POSITION_SIZE, 10),
                RiskLimit(RiskLimitType.MAX_DELTA, 5.0),
                RiskLimit(RiskLimitType.MAX_GAMMA, 0.5),
                RiskLimit(RiskLimitType.MAX_VEGA, 2.0),
                RiskLimit(RiskLimitType.MAX_THETA, 1.0),
                RiskLimit(RiskLimitType.MAX_LOSS_PER_TRADE, 0.05),  # 5%
                RiskLimit(RiskLimitType.MAX_DAILY_LOSS, 0.10),  # 10%
                RiskLimit(RiskLimitType.MAX_DRAWDOWN, 0.20),  # 20%
                RiskLimit(RiskLimitType.VAR_LIMIT, 0.02),  # 2% VaR
            ]

    def calculate_portfolio_risk(
        self,
        positions: List,
        underlying_price: float,
        risk_free_rate: float = 0.05,
        volatility: float = 0.80,
    ) -> RiskMetrics:
        """计算组合风险"""
        metrics = RiskMetrics()

        for pos in positions:
            # 计算单个期权的 Greeks
            from .strategies import OptionPositionSide

            T = (pos.contract.expiration_timestamp - 0) / (365 * 24 * 60 * 60 * 1000)
            if T <= 0:
                T = 1 / 365

            greeks = calculate_greeks(
                S=underlying_price,
                K=pos.contract.strike_price,
                T=T,
                r=risk_free_rate,
                sigma=volatility,
                option_type=pos.contract.option_type.value,
            )

            # 方向调整
            direction = 1 if pos.side == OptionPositionSide.LONG else -1
            size = pos.size * direction

            metrics.portfolio_delta += greeks.delta * size
            metrics.portfolio_gamma += greeks.gamma * size
            metrics.portfolio_vega += greeks.vega * size
            metrics.portfolio_theta += greeks.theta * size

            metrics.total_premium += pos.entry_value
            metrics.unrealized_pnl += (
                (pos.contract.mark_price - pos.entry_premium) * pos.size * direction
            )

        metrics.position_count = len(positions)

        return metrics

    def check_risk(
        self,
        positions: List,
        underlying_price: float,
        cash: float,
        equity: float,
    ) -> List[Dict]:
        """
        检查风险限制

        Returns:
            触发限制的警告列表
        """
        warnings = []
        metrics = self.calculate_portfolio_risk(positions, underlying_price)

        # 更新 equity curve 用于 VaR 计算
        self.equity_curve.append(equity)
        if len(self.equity_curve) > 252:  # 保持一年数据
            self.equity_curve.pop(0)

        # 计算 VaR
        metrics.var = self.calculate_var()
        metrics.max_drawdown = self.get_current_drawdown(equity)

        # 检查各项限制
        for limit in self.limits:
            if limit.limit_type == RiskLimitType.MAX_POSITION_SIZE:
                if metrics.position_count > limit.value:
                    warnings.append({
                        "type": "POSITION_SIZE",
                        "limit": limit.value,
                        "current": metrics.position_count,
                        "action": limit.action,
                        "message": f"Position count {metrics.position_count} exceeds limit {limit.value}",
                    })

            elif limit.limit_type == RiskLimitType.MAX_DELTA:
                if abs(metrics.portfolio_delta) > limit.value:
                    warnings.append({
                        "type": "DELTA",
                        "limit": limit.value,
                        "current": metrics.portfolio_delta,
                        "action": limit.action,
                        "message": f"Delta {metrics.portfolio_delta:.2f} exceeds limit {limit.value}",
                    })

            elif limit.limit_type == RiskLimitType.MAX_GAMMA:
                if abs(metrics.portfolio_gamma) > limit.value:
                    warnings.append({
                        "type": "GAMMA",
                        "limit": limit.value,
                        "current": metrics.portfolio_gamma,
                        "action": limit.action,
                        "message": f"Gamma {metrics.portfolio_gamma:.4f} exceeds limit {limit.value}",
                    })

            elif limit.limit_type == RiskLimitType.MAX_VEGA:
                if abs(metrics.portfolio_vega) > limit.value:
                    warnings.append({
                        "type": "VEGA",
                        "limit": limit.value,
                        "current": metrics.portfolio_vega,
                        "action": limit.action,
                        "message": f"Vega {metrics.portfolio_vega:.2f} exceeds limit {limit.value}",
                    })

            elif limit.limit_type == RiskLimitType.MAX_THETA:
                if abs(metrics.portfolio_theta) > limit.value:
                    warnings.append({
                        "type": "THETA",
                        "limit": limit.value,
                        "current": metrics.portfolio_theta,
                        "action": limit.action,
                        "message": f"Theta {metrics.portfolio_theta:.2f} exceeds limit {limit.value}",
                    })

            elif limit.limit_type == RiskLimitType.MAX_DAILY_LOSS:
                if abs(self.daily_loss) > limit.value * equity:
                    warnings.append({
                        "type": "DAILY_LOSS",
                        "limit": limit.value * equity,
                        "current": self.daily_loss,
                        "action": limit.action,
                        "message": f"Daily loss {self.daily_loss:.2f} exceeds limit {limit.value * equity:.2f}",
                    })

            elif limit.limit_type == RiskLimitType.MAX_DRAWDOWN:
                if abs(metrics.max_drawdown) > limit.value:
                    warnings.append({
                        "type": "MAX_DRAWDOWN",
                        "limit": limit.value,
                        "current": metrics.max_drawdown,
                        "action": limit.action,
                        "message": f"Max drawdown {metrics.max_drawdown * 100:.2f}% exceeds limit {limit.value * 100:.2f}%",
                    })

            elif limit.limit_type == RiskLimitType.VAR_LIMIT:
                if metrics.var > limit.value:
                    warnings.append({
                        "type": "VAR",
                        "limit": limit.value,
                        "current": metrics.var,
                        "action": limit.action,
                        "message": f"VaR {metrics.var * 100:.2f}% exceeds limit {limit.value * 100:.2f}%",
                    })

        if warnings:
            self.alerts.extend(warnings)

        return warnings

    def should_close_position(
        self,
        position,
        underlying_price: float,
        entry_price: float,
        pnl_pct: float,
    ) -> tuple:
        """
        检查是否应该止损/止盈

        Returns:
            (should_close, reason)
        """
        # 止损：亏损超过 50%
        if pnl_pct < -0.50:
            return True, "STOP_LOSS_50%"

        # 止盈：盈利超过 100%
        if pnl_pct > 1.00:
            return True, "TAKE_PROFIT_100%"

        # 检查 Greeks 风险
        if abs(position.contract.delta) > 0.95:
            return True, "DELTA_EXTREME"

        return False, None

    def update_daily_pnl(self, pnl: float):
        """更新每日盈亏"""
        self.daily_loss += pnl

    def reset_daily(self):
        """重置每日数据"""
        self.daily_loss = 0.0

    def update_peak_equity(self, equity: float):
        """更新峰值权益"""
        if equity > self.peak_equity:
            self.peak_equity = equity

    def get_current_drawdown(self, current_equity: float) -> float:
        """计算当前回撤"""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - current_equity) / self.peak_equity

    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        计算 Value at Risk（历史模拟法）

        Args:
            confidence_level: 置信水平（默认 95%）

        Returns:
            VaR 值（相对于权益的比例）
        """
        if len(self.equity_curve) < 50:
            return 0.0

        # 计算收益率序列
        returns = []
        for i in range(1, len(self.equity_curve)):
            if self.equity_curve[i - 1] > 0:
                ret = (self.equity_curve[i] - self.equity_curve[i - 1]) / self.equity_curve[i - 1]
                returns.append(ret)

        if not returns:
            return 0.0

        # VaR 是收益率分布的分位数
        var = np.percentile(returns, 100 * (1 - confidence_level))
        return abs(var)

    def check_max_drawdown_limit(self, current_equity: float) -> bool:
        """检查是否超过最大回撤限制"""
        max_dd_limit = next(
            (l.value for l in self.limits if l.limit_type == RiskLimitType.MAX_DRAWDOWN),
            0.20  # 默认 20%
        )

        current_dd = self.get_current_drawdown(current_equity)

        if current_dd > max_dd_limit:
            return False
        return True

    def check_trade_risk(self, potential_loss: float, capital: float) -> bool:
        """检查单笔交易风险"""
        max_risk_pct = next(
            (l.value for l in self.limits if l.limit_type == RiskLimitType.MAX_LOSS_PER_TRADE),
            0.05  # 默认 5%
        )

        max_allowed_loss = capital * max_risk_pct

        if potential_loss > max_allowed_loss:
            return False
        return True


# ============================================================================
# 仓位计算器
# ============================================================================

class PositionSizer:
    """
    仓位计算器

    根据账户规模和风险承受能力计算仓位大小
    """

    def __init__(
        self,
        method: str = "kelly",
        max_risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.10,
    ):
        """
        Args:
            method: "kelly", "fixed", "volatility"
            max_risk_per_trade: 每笔交易最大风险（比例）
            max_portfolio_risk: 组合最大风险（比例）
        """
        self.method = method
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk

    def calculate_size(
        self,
        account_equity: float,
        option_price: float,
        strike_price: float,
        volatility: float,
        probability_otm: float,
    ) -> float:
        """
        计算仓位大小

        Args:
            account_equity: 账户权益
            option_price: 期权价格
            strike_price: 行权价
            volatility: 波动率
            probability_otm: 期权归零概率

        Returns:
            建议仓位大小
        """
        if self.method == "fixed":
            return self._fixed_size(account_equity, option_price)

        elif self.method == "kelly":
            return self._kelly_size(
                account_equity, option_price, probability_otm
            )

        elif self.method == "volatility":
            return self._volatility_size(
                account_equity, option_price, volatility, strike_price
            )

        return 1.0

    def _fixed_size(self, equity: float, option_price: float) -> float:
        """固定仓位"""
        max_cost = equity * self.max_risk_per_trade
        return max(1.0, max_cost / option_price)

    def _kelly_size(
        self,
        equity: float,
        option_price: float,
        probability_otm: float,
    ) -> float:
        """Kelly Criterion"""
        win_rate = 1 - probability_otm
        avg_win = option_price * 2  # 简化：平均盈利为期权费的2倍
        avg_loss = option_price

        if avg_loss == 0:
            return 1.0

        kelly_fraction = (win_rate * avg_win - avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # 限制最大25%

        max_cost = equity * self.max_portfolio_risk
        return max(1.0, (equity * kelly_fraction) / option_price)

    def _volatility_size(
        self,
        equity: float,
        option_price: float,
        volatility: float,
        strike_price: float,
    ) -> float:
        """基于波动率的仓位"""
        # VaR 风格：波动率越高，仓位越小
        vol_scalar = 1.0 / (1 + volatility)

        max_cost = equity * self.max_risk_per_trade
        base_size = max_cost / option_price

        return max(1.0, base_size * vol_scalar)


# ============================================================================
# 止损策略
# ============================================================================

class StopLoss:
    """
    止损策略
    """

    def __init__(self, method: str = "percentage", value: float = 0.50):
        """
        Args:
            method: "percentage", "trailing", "greeks"
            value: 止损值（百分比或固定值）
        """
        self.method = method
        self.value = value
        self.peak_price = 0.0

    def should_trigger(
        self,
        entry_price: float,
        current_price: float,
        high_since_entry: float = None,
    ) -> tuple:
        """
        检查是否触发止损

        Returns:
            (triggered, current_stop_price)
        """
        if self.method == "percentage":
            stop_price = entry_price * (1 - self.value)
            return current_price <= stop_price, stop_price

        elif self.method == "trailing":
            if high_since_entry and high_since_entry > self.peak_price:
                self.peak_price = high_since_entry

            trailing_stop = self.peak_price * (1 - self.value)
            return current_price <= trailing_stop, trailing_stop

        return False, 0


# ============================================================================
# OptionSuite 风控策略类型枚举（从 D:/Hive/Data/trading_repos/OptionSuite 吸收）
# ============================================================================

class OptionsStrategyType(Enum):
    """期权策略类型.

    从 OptionSuite PutVerticalManagementStrategyTypes 和 StrangleManagementStrategyTypes 吸收并统一.
    """
    HOLD_TO_EXPIRATION = "hold_to_expiration"
    CLOSE_AT_50_PERCENT = "close_at_50_percent"
    CLOSE_AT_50_PERCENT_OR_21_DAYS = "close_at_50_pct_or_21_days"
    CLOSE_AT_21_DAYS = "close_at_21_days"
    ROLL_ON_THRESHOLD = "roll_on_threshold"


class PositionLifecycle(Enum):
    """持仓生命周期状态.

    描述持仓在不同阶段的状态，用于风控决策.
    """
    OPEN = "open"
    PARTIAL_PROFIT = "partial_profit"  # 已达50%利润
    NEAR_EXPIRY = "near_expiry"        # 接近到期（DTE <= 21）
    EXPIRY_PENDING = "expiry_pending"   # 到期前一小时（DTE <= 1）
    CLOSED = "closed"


# ============================================================================
# Put Vertical Spread 风控管理器（从 OptionSuite 吸收）
# ============================================================================

class PutVerticalRiskManager:
    """Put Vertical Spread风控管理器.

    策略: 卖出较低行权价Put, 买入更低保额更高行权价Put

    从 D:/Hive/Data/trading_repos/OptionSuite/riskManager/putVerticalRiskManagement.py 吸收

    规则:
    1. 利润达50% -> 止盈
    2. 距到期<=21天 -> 平仓
    3. DTE<30 -> 提前处理
    4. 到期 -> 平仓
    """

    def __init__(
        self,
        strategy_type: OptionsStrategyType = OptionsStrategyType.CLOSE_AT_50_PERCENT_OR_21_DAYS,
        profit_target_pct: float = 0.50,
        max_hold_days: int = 21,
        dte_threshold: int = 30,
    ):
        """
        Args:
            strategy_type: 策略管理类型
            profit_target_pct: 止盈目标（默认50%）
            max_hold_days: 最大持仓天数（默认21天）
            dte_threshold: DTE阈值，低于此值触发早期处理（默认30）
        """
        self.strategy_type = strategy_type
        self.profit_target_pct = profit_target_pct
        self.max_hold_days = max_hold_days
        self.dte_threshold = dte_threshold

    def should_close(
        self,
        position: dict,
        current_price: float,
        days_to_expiry: int,
    ) -> Tuple[bool, str]:
        """
        判断是否应该平仓.

        Args:
            position: 持仓信息字典，包含:
                - profit_pct: 利润百分比（小数形式，如0.6表示60%）
                - entry_price: 入场价格
                - strike_low: 较低行权价（short put）
                - strike_high: 较高行权价（long put）
                - premium: 权利金
            current_price: 当前标的价格
            days_to_expiry: 距到期天数（DTE）

        Returns:
            (should_close, reason): 是否应该平仓及原因

        Rules:
            1. 利润达到 profit_target_pct -> TAKE_PROFIT
            2. 距到期 <= max_hold_days -> TIME_EXIT
            3. DTE < dte_threshold -> EARLY_HANDLING
            4. DTE <= 1 -> EXPIRY
            5. 亏损达到50% -> STOP_LOSS（仅 CLOSE_AT_50_PERCENT_OR_21_DAYS_OR_HALFLOSS 策略）
        """
        profit_pct = position.get("profit_pct", 0.0)

        # 规则4: 到期
        if days_to_expiry <= 1:
            return True, "EXPIRY"

        # 规则2: 超过最大持仓期
        if days_to_expiry <= self.max_hold_days:
            return True, "TIME_EXIT"

        # 规则3: DTE低于阈值，触发早期处理
        if days_to_expiry < self.dte_threshold:
            return True, "EARLY_HANDLING"

        # 规则1: 止盈
        if profit_pct >= self.profit_target_pct:
            return True, "TAKE_PROFIT"

        # 规则5: 止损（仅特定策略）
        if self.strategy_type == OptionsStrategyType.CLOSE_AT_50_PERCENT_OR_21_DAYS:
            if profit_pct <= -0.50:
                return True, "STOP_LOSS"

        return False, "HOLD"

    def calculate_position_health(
        self,
        position: dict,
        current_price: float,
    ) -> dict:
        """
        计算持仓健康度.

        Args:
            position: 持仓信息字典
            current_price: 当前标的价格

        Returns:
            dict: {
                profit_pct: 利润百分比,
                days_to_expiry: 距到期天数,
                breakeven: 盈亏平衡点,
                max_loss_pct: 最大亏损百分比,
                health_score: 健康度评分 (0-100)
            }
        """
        profit_pct = position.get("profit_pct", 0.0)
        entry_price = position.get("entry_price", 0.0)
        strike_low = position.get("strike_low", 0.0)  # short put strike
        strike_high = position.get("strike_high", 0.0)  # long put strike
        premium = position.get("premium", 0.0)

        # 盈亏平衡点（对于put vertical: strike_low - net_credit）
        net_credit = premium  # 卖出低行权价put收取的权利金 - 买入高行权价put付出的权利金
        breakeven = strike_low - net_credit if net_credit > 0 else strike_low

        # 最大亏损（ Put Vertical Spread的最大亏损 = 行权价差 - 净权利金）
        max_loss = abs(strike_high - strike_low) - abs(net_credit)
        max_loss_pct = max_loss / entry_price if entry_price > 0 else 0.0

        # 健康度评分
        health_score = 50.0  # 基础分
        if profit_pct > 0:
            health_score += min(profit_pct * 50, 30)  # 最多加30分
        elif profit_pct < 0:
            health_score += max(profit_pct * 50, -50)  # 最多减50分

        return {
            "profit_pct": profit_pct,
            "days_to_expiry": position.get("days_to_expiry", 0),
            "breakeven": breakeven,
            "max_loss_pct": max_loss_pct,
            "health_score": max(0, min(100, health_score)),
        }

    def get_roll_recommendation(self, position: dict) -> dict:
        """
        获取移仓建议（如果应该持有但接近到期）.

        Args:
            position: 持仓信息字典

        Returns:
            dict: {
                should_roll: bool,
                current_strike: 当前行权价,
                recommended_strike: 推荐行权价,
                roll_cost: 移仓成本,
                reason: str
            }
        """
        strike_low = position.get("strike_low", 0.0)
        strike_high = position.get("strike_high", 0.0)
        days_to_expiry = position.get("days_to_expiry", 0)
        current_price = position.get("current_price", 0.0)

        # 推荐的移仓：向同一方向移动一个行权价间隔
        strike_interval = abs(strike_high - strike_low)
        recommended_strike_low = strike_low + strike_interval
        recommended_strike_high = strike_high + strike_interval

        roll_cost = (recommended_strike_high - recommended_strike_low) * 100  # 每点$100

        return {
            "should_roll": days_to_expiry < self.dte_threshold and days_to_expiry > 7,
            "current_strike_low": strike_low,
            "current_strike_high": strike_high,
            "recommended_strike_low": recommended_strike_low,
            "recommended_strike_high": recommended_strike_high,
            "roll_cost": roll_cost,
            "reason": f"DTE={days_to_expiry} below threshold {self.dte_threshold}, consider rolling to next expiration",
        }


# ============================================================================
# Strangle 风控管理器（从 OptionSuite 吸收）
# ============================================================================

class StrangleRiskManager:
    """Strangle风控管理器.

    策略: 卖出OTM Call + 卖出OTM Put（Short Strangle）

    从 D:/Hive/Data/trading_repos/OptionSuite/riskManager/strangleRiskManagement.py 吸收

    当价格触及某一翅膀时:
    - 短期: 快速调整
    - 中期: 等待反弹
    """

    def __init__(
        self,
        strategy_type: OptionsStrategyType = OptionsStrategyType.CLOSE_AT_50_PERCENT_OR_21_DAYS,
        profit_target_pct: float = 0.50,
        max_hold_days: int = 30,
        wing_adjust_threshold: float = 0.10,
    ):
        """
        Args:
            strategy_type: 策略管理类型
            profit_target_pct: 止盈目标（默认50%）
            max_hold_days: 最大持仓天数（默认30天，strangle可持有更久）
            wing_adjust_threshold: 翅膀调整阈值（当价格移动10%时调整）
        """
        self.strategy_type = strategy_type
        self.profit_target_pct = profit_target_pct
        self.max_hold_days = max_hold_days
        self.wing_adjust_threshold = wing_adjust_threshold

    def should_adjust(
        self,
        position: dict,
        current_price: float,
    ) -> dict:
        """
        判断是否需要调整翅膀.

        当价格移动触及某一翅膀时，根据持仓时间和策略类型决定是否调整.

        Args:
            position: 持仓信息字典，包含:
                - put_strike: Put行权价（翅膀）
                - call_strike: Call行权价（翅膀）
                - days_held: 已持仓天数
                - wing_width: 翅膀宽度（两行权价之间的距离）
            current_price: 当前标的价格

        Returns:
            dict: {
                should_adjust: bool,
                adjustment_type: str,  # "WIDEN", "NARROW", "ROLL_UP", "ROLL_DOWN"
                reason: str
            }
        """
        put_strike = position.get("put_strike", 0.0)
        call_strike = position.get("call_strike", 0.0)
        days_held = position.get("days_held", 0)
        current_price = position.get("current_price", current_price)

        wing_width = abs(call_strike - put_strike)

        adjustment_needed = False
        adjustment_type = ""
        reason = ""

        # 检查是否价格触及put翅膀（下限）
        if current_price <= put_strike * (1 + self.wing_adjust_threshold):
            if days_held < 7:  # 短期：快速调整
                adjustment_type = "WIDEN"
                reason = f"Short-term: Price {current_price} near put wing {put_strike}, widen spread"
            else:  # 中期：等待反弹
                adjustment_type = "HOLD"
                reason = f"Medium-term: Price {current_price} near put wing, waiting for rebound"

        # 检查是否价格触及call翅膀（上限）
        elif current_price >= call_strike * (1 - self.wing_adjust_threshold):
            if days_held < 7:  # 短期：快速调整
                adjustment_type = "WIDEN"
                reason = f"Short-term: Price {current_price} near call wing {call_strike}, widen spread"
            else:  # 中期：等待反弹
                adjustment_type = "HOLD"
                reason = f"Medium-term: Price {current_price} near call wing, waiting for pullback"

        # 价格安全地在两翅膀之间
        else:
            adjustment_type = "HOLD"
            reason = "Price within safe range"

        return {
            "should_adjust": adjustment_type in ("WIDEN", "NARROW", "ROLL_UP", "ROLL_DOWN"),
            "adjustment_type": adjustment_type,
            "reason": reason,
            "current_put_strike": put_strike,
            "current_call_strike": call_strike,
        }

    def should_close(
        self,
        position: dict,
        current_price: float,
    ) -> Tuple[bool, str]:
        """
        判断是否应该平仓.

        Args:
            position: 持仓信息字典，包含:
                - profit_pct: 利润百分比（小数形式）
                - days_held: 已持仓天数
                - days_to_expiry: 距到期天数（DTE）
            current_price: 当前标的价格

        Returns:
            (should_close, reason): 是否应该平仓及原因
        """
        profit_pct = position.get("profit_pct", 0.0)
        days_held = position.get("days_held", 0)
        days_to_expiry = position.get("days_to_expiry", 30)

        # 规则1: 到期
        if days_to_expiry <= 1:
            return True, "EXPIRY"

        # 规则2: 超过最大持仓期
        if days_held >= self.max_hold_days:
            return True, "MAX_HOLDING_PERIOD"

        # 规则3: 止盈
        if profit_pct >= self.profit_target_pct:
            return True, "TAKE_PROFIT"

        # 规则4: 亏损止损
        if profit_pct <= -0.50:
            return True, "STOP_LOSS"

        return False, "HOLD"

    def calculate_strangle_metrics(self, position: dict, current_price: float) -> dict:
        """
        计算Strangle持仓指标.

        Args:
            position: 持仓信息字典
            current_price: 当前标的价格

        Returns:
            dict: {
                profit_pct: 利润百分比,
                max_profit_pct: 最大盈利百分比,
                max_loss_pct: 最大亏损百分比,
                wing_utilization: 翅膀利用率,
                days_to_expiry: 距到期天数
            }
        """
        profit_pct = position.get("profit_pct", 0.0)
        put_strike = position.get("put_strike", 0.0)
        call_strike = position.get("call_strike", 0.0)
        premium_collected = position.get("premium_collected", 0.0)
        days_to_expiry = position.get("days_to_expiry", 0)

        wing_width = abs(call_strike - put_strike)
        max_profit_pct = premium_collected / wing_width if wing_width > 0 else 0.0
        max_loss_pct = 1.0  # Strangle最大亏损理论上无限（但实际受限于保证金）

        # 翅膀利用率：当前价格距两边翅膀的平均距离
        mid_strike = (put_strike + call_strike) / 2
        distance_to_put = abs(current_price - put_strike) / put_strike
        distance_to_call = abs(call_strike - current_price) / call_strike
        wing_utilization = 1.0 - (distance_to_put + distance_to_call) / 2

        return {
            "profit_pct": profit_pct,
            "max_profit_pct": max_profit_pct,
            "max_loss_pct": max_loss_pct,
            "wing_utilization": wing_utilization,
            "days_to_expiry": days_to_expiry,
        }


# ============================================================================
# 别名（向后兼容）
# ============================================================================

OptionsRiskManager = RiskManager


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    # 测试风险管理
    from .pricing.greeks import calculate_greeks

    print("=== Risk Manager + Position Cycle Test ===\n")

    manager = RiskManager()

    # 测试持仓周期管理
    print("--- Position Cycle Manager ---")
    cycle_mgr = manager.cycle_manager

    # 模拟持仓
    import uuid
    pos_id = str(uuid.uuid4())

    from datetime import datetime, timedelta
    entry_time = datetime.now() - timedelta(days=15)

    cycle_mgr.track_position(pos_id, entry_time, 100.0)
    print(f"Tracked position: {pos_id}")
    print(f"  Days held: {cycle_mgr.get_position_age(pos_id):.1f}")

    # 检查退出信号（DTE 正常）
    signal = cycle_mgr.check_exit_signals(pos_id, current_dte=20)
    print(f"  Exit signal (DTE=20): {signal}")

    # 检查退出信号（DTE 过低）
    signal = cycle_mgr.check_exit_signals(pos_id, current_dte=5)
    print(f"  Exit signal (DTE=5): {signal}")

    # 模拟希腊字母暴露
    print("\n--- Portfolio Risk Metrics ---")
    metrics = RiskMetrics(
        portfolio_delta=2.5,
        portfolio_gamma=0.3,
        portfolio_vega=1.5,
        portfolio_theta=-0.8,
        position_count=4,
        total_premium=-500,
        unrealized_pnl=150,
    )

    print(f"  Delta:  {metrics.portfolio_delta:.2f}")
    print(f"  Gamma:  {metrics.portfolio_gamma:.4f}")
    print(f"  Vega:   {metrics.portfolio_vega:.2f}")
    print(f"  Theta:  {metrics.portfolio_theta:.2f} (per day)")

    # 测试仓位计算
    sizer = PositionSizer(method="kelly", max_risk_per_trade=0.02)
    size = sizer.calculate_size(
        account_equity=10000,
        option_price=100,
        strike_price=2200,
        volatility=0.80,
        probability_otm=0.50,
    )
    print(f"\nRecommended Position Size: {size:.2f} contracts")

    # 测试止损
    stop = StopLoss(method="trailing", value=0.25)
    print(f"\nStop Loss (Trailing 25%):")
    print(f"  Entry: $100, High: $150 -> Stop: ${150 * 0.75:.2f}")
