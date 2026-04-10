# -*- coding: utf-8 -*-
"""
Multi-Strategy Portfolio Manager - 多策略组合管理器

管理多个策略的资金分配和风险控制:
- 资金在策略间分配
- 跟踪整体组合表现
- 处理仓位再平衡
- 风险限制

Usage:
    from quant_trading.portfolio.multi_strategy_portfolio import MultiStrategyPortfolio

    portfolio = MultiStrategyPortfolio(initial_cash=100000)
    portfolio.allocate("v36", 0.6)
    portfolio.update("v36", position_value=60000)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np

logger = logging.getLogger("MultiStrategyPortfolio")


@dataclass
class Position:
    """仓位信息"""
    strategy_name: str
    allocation_ratio: float   # 分配比例
    current_value: float      # 当前市值
    entry_value: float        # 入场价值
    pnl: float = 0.0          # 盈亏
    pnl_pct: float = 0.0     # 盈亏百分比
    trades: int = 0           # 交易次数
    wins: int = 0             # 盈利次数


@dataclass
class PortfolioMetrics:
    """组合指标"""
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    cash: float
    positions_value: float
    leverage: float
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0


@dataclass
class RebalanceResult:
    """再平衡结果"""
    strategy: str
    old_allocation: float
    new_allocation: float
    value_change: float
    reason: str


class MultiStrategyPortfolio:
    """
    多策略组合管理器

    功能:
    - 策略资金分配
    - 仓位跟踪
    - 动态再平衡
    - 风险控制
    - 表现归因
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        max_strategies: int = 3,
        max_position_per_strategy: float = 0.5,
        rebalance_threshold: float = 0.1,
        stop_loss: float = -0.15,
        max_drawdown_limit: float = -0.20
    ):
        """
        Args:
            initial_cash: 初始资金
            max_strategies: 最大策略数
            max_position_per_strategy: 单策略最大仓位比例
            rebalance_threshold: 再平衡阈值
            stop_loss: 组合止损线
            max_drawdown_limit: 最大回撤限制
        """
        self.initial_cash = initial_cash
        self.max_strategies = max_strategies
        self.max_position_per_strategy = max_position_per_strategy
        self.rebalance_threshold = rebalance_threshold
        self.stop_loss = stop_loss
        self.max_drawdown_limit = max_drawdown_limit

        self._cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._allocations: Dict[str, float] = {}  # 目标分配比例
        self._equity_curve: List[float] = [initial_cash]
        self._peak_equity = initial_cash
        self._trade_history: List[Dict[str, Any]] = []
        self._rebalance_history: List[RebalanceResult] = []

    @property
    def total_value(self) -> float:
        """总权益"""
        positions_value = sum(p.current_value for p in self._positions.values())
        return self._cash + positions_value

    @property
    def positions_value(self) -> float:
        """持仓市值"""
        return sum(p.current_value for p in self._positions.values())

    @property
    def cash(self) -> float:
        """可用资金"""
        return self._cash

    @property
    def positions(self) -> Dict[str, Position]:
        """所有仓位"""
        return self._positions.copy()

    def allocate(
        self,
        strategy_name: str,
        ratio: float,
        initial_value: Optional[float] = None
    ) -> bool:
        """
        分配资金给策略

        Args:
            strategy_name: 策略名称
            ratio: 分配比例 (0.0 - 1.0)
            initial_value: 初始市值 (默认按比例计算)

        Returns:
            bool: 分配是否成功
        """
        if len(self._allocations) >= self.max_strategies:
            if strategy_name not in self._allocations:
                logger.warning(f"Max strategies reached ({self.max_strategies})")
                return False

        if ratio > self.max_position_per_strategy:
            ratio = self.max_position_per_strategy
            logger.warning(f"Capped allocation to {ratio}")

        # 检查总分配
        total_alloc = sum(self._allocations.values()) - self._allocations.get(strategy_name, 0)
        if total_alloc + ratio > 1.0:
            ratio = 1.0 - total_alloc
            logger.warning(f"Adjusted allocation to {ratio} to fit total")

        self._allocations[strategy_name] = ratio

        if strategy_name not in self._positions:
            value = initial_value if initial_value is not None else self.total_value * ratio
            self._positions[strategy_name] = Position(
                strategy_name=strategy_name,
                allocation_ratio=ratio,
                current_value=value,
                entry_value=value
            )
            logger.info(f"Allocated {ratio:.0%} to {strategy_name} (${value:,.0f})")
        else:
            self._positions[strategy_name].allocation_ratio = ratio

        return True

    def deallocate(self, strategy_name: str) -> Optional[float]:
        """
        取消策略分配，回收资金

        Args:
            strategy_name: 策略名称

        Returns:
            float: 回收的资金
        """
        if strategy_name not in self._positions:
            return None

        pos = self._positions[strategy_name]
        recovered = pos.current_value + self._cash * self._allocations.get(strategy_name, 0)

        # 更新现金
        self._cash += pos.current_value

        # 删除仓位
        del self._positions[strategy_name]
        del self._allocations[strategy_name]

        logger.info(f"Deallocated {strategy_name}, recovered ${recovered:,.0f}")
        return recovered

    def update(
        self,
        strategy_name: str,
        current_value: float,
        trade_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        更新策略仓位

        Args:
            strategy_name: 策略名称
            current_value: 当前市值
            trade_result: 交易结果 {"win": bool, "pnl": float}
        """
        if strategy_name not in self._positions:
            logger.warning(f"Position not found for {strategy_name}")
            return

        pos = self._positions[strategy_name]
        pos.entry_value = pos.current_value  # 更新入场价值
        pos.current_value = current_value
        pos.pnl = current_value - pos.entry_value
        pos.pnl_pct = pos.pnl / pos.entry_value if pos.entry_value > 0 else 0

        if trade_result:
            pos.trades += 1
            if trade_result.get("win", False):
                pos.wins += 1
            self._trade_history.append({
                "strategy": strategy_name,
                "timestamp": datetime.now().isoformat(),
                **trade_result
            })

        # 更新权益曲线
        self._equity_curve.append(self.total_value)

    def update_allocation(
        self,
        strategy_name: str,
        new_ratio: float
    ) -> Optional[RebalanceResult]:
        """
        更新策略分配比例

        Args:
            strategy_name: 策略名称
            new_ratio: 新比例

        Returns:
            RebalanceResult: 再平衡结果
        """
        if strategy_name not in self._allocations:
            return None

        old_ratio = self._allocations[strategy_name]

        # 计算价值变化
        target_value = self.total_value * new_ratio
        current_value = self._positions.get(strategy_name, Position("", 0, 0, 0)).current_value
        value_change = target_value - current_value

        # 执行再分配
        if value_change > 0:
            # 增加仓位 - 从现金扣除
            if value_change > self._cash:
                value_change = self._cash
            self._cash -= value_change
        else:
            # 减少仓位 - 回收现金
            self._cash += abs(value_change)

        self._allocations[strategy_name] = new_ratio

        result = RebalanceResult(
            strategy=strategy_name,
            old_allocation=old_ratio,
            new_allocation=new_ratio,
            value_change=value_change,
            reason="manual_adjustment"
        )
        self._rebalance_history.append(result)

        logger.info(
            f"Rebalanced {strategy_name}: {old_ratio:.0%} -> {new_ratio:.0%} "
            f"(value change: ${value_change:+,.0f})"
        )
        return result

    def check_and_rebalance(self) -> List[RebalanceResult]:
        """
        检查并自动再平衡

        Returns:
            List[RebalanceResult]: 再平衡结果列表
        """
        results = []
        total_value = self.total_value

        for name, target_ratio in self._allocations.items():
            if name not in self._positions:
                continue

            target_value = total_value * target_ratio
            current_value = self._positions[name].current_value
            drift = (current_value - target_value) / total_value

            if abs(drift) > self.rebalance_threshold:
                # 需要再平衡
                self.update_allocation(name, target_ratio)
                results.append(RebalanceResult(
                    strategy=name,
                    old_allocation=self._allocations[name] - drift,
                    new_allocation=target_ratio,
                    value_change=current_value - target_value,
                    reason=f"drift={drift:.1%}"
                ))

        return results

    def get_metrics(self) -> PortfolioMetrics:
        """获取组合指标"""
        positions_value = self.positions_value
        total_value = self.total_value
        total_pnl = total_value - self.initial_cash
        total_pnl_pct = total_pnl / self.initial_cash if self.initial_cash > 0 else 0

        # 更新峰值
        if total_value > self._peak_equity:
            self._peak_equity = total_value

        # 最大回撤
        max_dd = 0.0
        for eq in self._equity_curve:
            if self._peak_equity > 0:
                dd = (self._peak_equity - eq) / self._peak_equity
                max_dd = max(max_dd, dd)

        # 夏普比率
        if len(self._equity_curve) > 1:
            returns = np.diff(self._equity_curve) / self._equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0

        # 总胜率
        total_trades = sum(p.trades for p in self._positions.values())
        total_wins = sum(p.wins for p in self._positions.values())
        win_rate = total_wins / total_trades if total_trades > 0 else 0.0

        # 杠杆
        leverage = positions_value / total_value if total_value > 0 else 0.0

        return PortfolioMetrics(
            total_value=total_value,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            cash=self._cash,
            positions_value=positions_value,
            leverage=leverage,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate
        )

    def get_position_summary(self) -> Dict[str, Any]:
        """获取仓位摘要"""
        return {
            name: {
                "allocation": pos.allocation_ratio,
                "value": pos.current_value,
                "pnl": pos.pnl,
                "pnl_pct": pos.pnl_pct,
                "trades": pos.trades,
                "wins": pos.wins,
                "win_rate": pos.wins / pos.trades if pos.trades > 0 else 0
            }
            for name, pos in self._positions.items()
        }

    def check_risk_limits(self) -> Dict[str, bool]:
        """
        检查风险限制

        Returns:
            Dict[str, bool]: 各风险指标是否触发
        """
        metrics = self.get_metrics()
        alerts = {}

        # 止损检查
        alerts["stop_loss_triggered"] = metrics.total_pnl_pct <= self.stop_loss

        # 最大回撤检查
        alerts["max_drawdown_triggered"] = metrics.max_drawdown >= abs(self.max_drawdown_limit)

        # 杠杆过高
        alerts["high_leverage"] = metrics.leverage > 1.5

        # 现金不足 (低于10%)
        cash_ratio = metrics.cash / metrics.total_value if metrics.total_value > 0 else 0
        alerts["low_cash"] = cash_ratio < 0.1

        return alerts

    def reset(self) -> None:
        """重置组合"""
        self._cash = self.initial_cash
        self._positions = {}
        self._allocations = {}
        self._equity_curve = [self.initial_cash]
        self._peak_equity = self.initial_cash
        self._trade_history = []
        self._rebalance_history = []
        logger.info("Portfolio reset")


# ===================== 资金管理辅助 =====================

class CapitalManager:
    """
    资金管理器

    负责计算策略的实际资金分配
    """

    def __init__(self, total_capital: float = 100000.0):
        self.total_capital = total_capital
        self._reserved: Dict[str, float] = {}

    def reserve(self, strategy_name: str, amount: float) -> bool:
        """预留资金"""
        available = self.total_capital - sum(self._reserved.values())
        if amount > available:
            logger.warning(f"Insufficient capital for {strategy_name}: {amount} > {available}")
            return False
        self._reserved[strategy_name] = amount
        return True

    def release(self, strategy_name: str) -> float:
        """释放预留"""
        if strategy_name in self._reserved:
            amount = self._reserved[strategy_name]
            del self._reserved[strategy_name]
            return amount
        return 0.0

    def get_available(self) -> float:
        """获取可用资金"""
        return self.total_capital - sum(self._reserved.values())

    def rebalance(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """
        计算实际分配金额

        Args:
            allocations: Dict[策略名, 比例]

        Returns:
            Dict[策略名, 金额]
        """
        available = self.get_available()
        return {
            name: available * ratio
            for name, ratio in allocations.items()
        }
