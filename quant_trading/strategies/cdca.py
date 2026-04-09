# -*- coding: utf-8 -*-
"""
cDCA (Constant Dollar Cost Averaging) 定投策略模块 / cDCA Strategy Module

定时定额定投 — Regular fixed-amount investing.

策略逻辑 / Strategy Logic:
1. 固定时间间隔买入 / Fixed time interval purchases
2. 固定金额 / Fixed amount
3. 低波动时加大仓位, 高波动时减少 / Increase position in low volatility, decrease in high volatility

特点 / Features:
- 长期定投, 摊平成本 / Long-term DCA, average cost reduction
- 波动率调整仓位 / Volatility-adjusted position sizing
- 无需预测市场 / No market prediction needed
- 适合长期持有 / Suitable for long-term holding

Ref: kraken-infinity-grid (cDCA)
    https://github.com/btschwertfeger/kraken-infinity-grid
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from time import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams


__all__ = [
    "CDCAParams",
    "CDCAStrategy",
]


@dataclass
class CDCAParams(StrategyParams):
    """cDCA策略参数 / cDCA strategy parameters."""
    amount_per_interval: float = 100.0   # 每个间隔投资金额(quote) / Investment per interval (quote)
    interval_seconds: int = 86400       # 投资间隔秒数 (default: 1 day) / Interval in seconds (default: 1 day)
    volatility_adjust: bool = True        # 是否启用波动率调整 / Enable volatility adjustment
    high_vol_threshold: float = 0.5      # 高波动阈值 (>此值减少投资) / High volatility threshold
    low_vol_threshold: float = 0.2      # 低波动阈值 (<此值增加投资) / Low volatility threshold
    max_vol_multiplier: float = 2.0     # 最大波动调整倍数 / Max volatility multiplier
    min_vol_multiplier: float = 0.5     # 最小波动调整倍数 / Min volatility multiplier
    fee: float = 0.0016                 # 交易费率 / Trading fee
    max_investment: float = 10000.0     # 最大投资额 / Maximum investment


class CDCAStrategy(BaseStrategy):
    """
    Constant Dollar Cost Averaging — 定时定额定投 / cDCA Strategy.

    在固定时间间隔用固定金额购买资产, 通过波动率调整每次购买数量,
    长期持有不卖出.

    Invest fixed amount at regular intervals, adjust purchase quantity
    based on volatility, hold long-term without selling.

    策略特点 / Key Features:
    - 时间驱动买入, 非价格驱动 / Time-driven buying, not price-driven
    - 波动率调整仓位 / Volatility-adjusted position sizing
    - 低波动多买, 高波动少买 / Buy more in low volatility, less in high volatility
    - 只买不卖, 长期持有 / Buy only, hold long-term

    与GridHODL的区别 / vs GridHODL:
    - GridHODL: 网格交易+持有, 有买有卖 / Grid trading + holding, buys and sells
    - cDCA: 纯定投, 只买不卖 / Pure DCA, buy only, no selling

    与SWING的区别 / vs SWING:
    - SWING: 区间交易, 高卖低买 / Range trading, sell high buy low
    - cDCA: 定时买入, 无需判断市场 / Regular buying, no market timing needed
    """

    name: str = "cdca"

    def __init__(
        self,
        symbol: str,
        params: Optional[CDCAParams] = None,
    ) -> None:
        """
        初始化cDCA策略 / Initialize cDCA strategy.

        Args:
            symbol: 交易对符号 / Trading pair symbol
            params: 策略参数 / Strategy parameters
        """
        super().__init__(symbol, params)
        p = params or CDCAParams()
        self.params = p

        # 内部状态 / Internal state
        self._accumulated_base: float = 0.0      # 积累的基础货币 / Accumulated base currency
        self._total_invested: float = 0.0        # 总投入金额 / Total invested amount
        self._avg_cost: float = 0.0              # 平均成本 / Average cost
        self._last_purchase_price: float = 0.0   # 上次购买价格 / Last purchase price
        self._last_execute_time: float = 0.0     # 上次执行时间戳 / Last execution timestamp
        self._volatility: float = 0.0            # 当前波动率 / Current volatility
        self._vol_multiplier: float = 1.0       # 波动调整倍数 / Volatility multiplier
        self._purchase_count: int = 0             # 购买次数 / Purchase count
        self._current_price: float = 0.0         # 当前价格 / Current price

        # 价格历史用于计算波动率 / Price history for volatility calculation
        self._price_history: List[float] = []

    def compute_purchase_amount(self, current_vol: float) -> float:
        """
        根据波动率计算实际购买金额 / Calculate actual purchase amount based on volatility.

        波动率调整规则 / Volatility adjustment rules:
        - current_vol > high_vol_threshold: 减少购买量 * min_vol_multiplier / Decrease * min multiplier
        - current_vol < low_vol_threshold: 增加购买量 * max_vol_multiplier / Increase * max multiplier
        - 否则使用基础金额 / Otherwise use base amount

        Args:
            current_vol: 当前波动率 / Current volatility

        Returns:
            调整后的购买金额 / Adjusted purchase amount
        """
        p = self.params
        base = p.amount_per_interval

        if not p.volatility_adjust:
            return base

        if current_vol > p.high_vol_threshold:
            # 高波动: 减少购买量 / High volatility: decrease amount
            self._vol_multiplier = p.min_vol_multiplier
        elif current_vol < p.low_vol_threshold:
            # 低波动: 增加购买量 / Low volatility: increase amount
            self._vol_multiplier = p.max_vol_multiplier
        else:
            # 正常区间: 线性插值 / Normal range: linear interpolation
            range_size = p.high_vol_threshold - p.low_vol_threshold
            if range_size > 0:
                normalized = (current_vol - p.low_vol_threshold) / range_size
                self._vol_multiplier = p.max_vol_multiplier - normalized * (
                    p.max_vol_multiplier - p.min_vol_multiplier
                )

        return base * self._vol_multiplier

    def should_execute(self, elapsed_seconds: float) -> bool:
        """
        判断是否应该执行购买 / Determine if purchase should be executed.

        Args:
            elapsed_seconds: 距上次执行的秒数 / Seconds since last execution

        Returns:
            True if should execute, False otherwise
        """
        return elapsed_seconds >= self.params.interval_seconds

    def execute_purchase(self, current_price: float) -> Dict[str, Any]:
        """
        执行一次购买 / Execute a purchase.

        Args:
            current_price: 当前价格 / Current price

        Returns:
            Dict with purchase details:
            - action: "buy"
            - price: purchase price
            - volume: purchased volume
            - amount: quote amount spent
            - vol_multiplier: volatility multiplier used
            - message: str description
        """
        p = self.params
        amount = self.compute_purchase_amount(self._volatility)

        # 确保不超过最大投资额 / Ensure not exceeding max investment
        if self._total_invested + amount > p.max_investment:
            amount = max(0.0, p.max_investment - self._total_invested)

        if amount <= 0:
            return {
                "action": "hold",
                "price": current_price,
                "volume": 0.0,
                "amount": 0.0,
                "vol_multiplier": self._vol_multiplier,
                "message": "Max investment reached, no more purchases",
            }

        # 计算购买数量 / Calculate volume
        volume = float(Decimal(str(amount)) / Decimal(str(current_price)))

        # 更新状态 / Update state
        self._accumulated_base += volume
        self._total_invested += amount
        self._avg_cost = self._total_invested / self._accumulated_base
        self._last_purchase_price = current_price
        self._last_execute_time = time()
        self._purchase_count += 1

        return {
            "action": "buy",
            "price": round(current_price, 2),
            "volume": round(volume, 8),
            "amount": round(amount, 2),
            "vol_multiplier": round(self._vol_multiplier, 4),
            "message": (
                f"cDCA: Bought {volume:.8f} @ {current_price:.2f} "
                f"(vol_mult={self._vol_multiplier:.2f})"
            ),
        }

    def on_price_update(
        self,
        current_price: float,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        价格更新时调用 (REST接口) / Call on price update (REST API).

        Args:
            current_price: 当前价格 / Current price
            current_time: 当前时间戳 (可选) / Current timestamp (optional)

        Returns:
            Dict with action info
        """
        self._current_price = current_price
        now = current_time or time()

        # 更新价格历史和波动率 / Update price history and volatility
        self._update_volatility(current_price)

        # 检查是否应该执行购买 / Check if should execute purchase
        elapsed = now - self._last_execute_time
        if self.should_execute(elapsed):
            return self.execute_purchase(current_price)

        return {
            "action": "hold",
            "price": current_price,
            "volume": 0.0,
            "amount": 0.0,
            "vol_multiplier": round(self._vol_multiplier, 4),
            "message": (
                f"Holding. Next purchase in "
                f"{max(0, self.params.interval_seconds - elapsed):.0f}s"
            ),
        }

    def _update_volatility(self, current_price: float) -> None:
        """更新波动率计算 / Update volatility calculation."""
        self._price_history.append(current_price)

        # 保持足够的历史数据 / Keep enough history
        max_history = max(self.params.interval_seconds // 60, 100)
        if len(self._price_history) > max_history:
            self._price_history = self._price_history[-max_history:]

        # 计算波动率 (使用收益率标准差) / Calculate volatility (std of returns)
        if len(self._price_history) >= 2:
            returns = np.diff(self._price_history) / np.array(self._price_history[:-1])
            self._volatility = float(np.std(returns))

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号 (基于时间) / Generate trading signals (time-based).

        Args:
            data: K线数据 (需要close列和时间戳) / Candlestick data (needs close and timestamp)

        Returns:
            List of trading signals (buy signals when interval elapsed)
        """
        signals = []
        if len(data) < 2:
            return signals

        current_price = float(data["close"].iloc[-1])
        self._current_price = current_price

        # 获取时间戳列或使用索引 / Get timestamp column or use index
        if "timestamp" in data.columns:
            current_time = float(data["timestamp"].iloc[-1])
        else:
            # 假设每行代表一个时间单位 / Assume each row represents one time unit
            current_time = time()

        self._update_volatility(current_price)

        elapsed = current_time - self._last_execute_time
        if self.should_execute(elapsed):
            result = self.execute_purchase(current_price)
            if result["action"] == "buy":
                signals.append(
                    Signal(
                        symbol=self.symbol,
                        direction=SignalDirection.LONG,
                        price=result["price"],
                        strength=1.0,
                        metadata={
                            "order_type": "dca_buy",
                            "volume": result["volume"],
                            "amount": result["amount"],
                            "vol_multiplier": result["vol_multiplier"],
                        },
                    )
                )

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: Any,
    ) -> float:
        """计算仓位大小 / Calculate position size."""
        return signal.metadata.get("amount", 0.0) / signal.price

    def get_state(self) -> Dict[str, Any]:
        """
        获取当前策略状态 / Get current strategy state.

        Returns:
            Dict with current state info
        """
        p = self.params
        elapsed = time() - self._last_execute_time
        next_purchase_in = max(0, p.interval_seconds - elapsed)

        unrealized_pnl = 0.0
        if self._accumulated_base > 0 and self._current_price > 0:
            unrealized_pnl = (
                self._current_price * self._accumulated_base - self._total_invested
            )

        return {
            "accumulated_base": round(self._accumulated_base, 8),
            "total_invested": round(self._total_invested, 2),
            "avg_cost": round(self._avg_cost, 2),
            "current_price": round(self._current_price, 2),
            "current_volatility": round(self._volatility, 6),
            "vol_multiplier": round(self._vol_multiplier, 4),
            "purchase_count": self._purchase_count,
            "next_purchase_in_seconds": round(next_purchase_in, 0),
            "max_investment_reached": self._total_invested >= p.max_investment,
            "unrealized_pnl": round(unrealized_pnl, 2),
        }

    def get_profit(self) -> Dict[str, float]:
        """
        计算当前盈亏 / Calculate current profit/loss.

        Returns:
            Dict with profit info:
            - total_invested: 总投入 / Total invested
            - current_value: 当前价值 / Current value
            - profit: 利润 / Profit
            - roi: 投资回报率 / Return on investment
            - avg_cost: 平均成本 / Average cost
        """
        if self._accumulated_base == 0:
            return {
                "total_invested": 0.0,
                "current_value": 0.0,
                "profit": 0.0,
                "roi": 0.0,
                "avg_cost": 0.0,
            }

        current_value = self._current_price * self._accumulated_base
        profit = current_value - self._total_invested
        roi = (profit / self._total_invested * 100) if self._total_invested > 0 else 0.0

        return {
            "total_invested": round(self._total_invested, 2),
            "current_value": round(current_value, 2),
            "profit": round(profit, 2),
            "roi": round(roi, 2),
            "avg_cost": round(self._avg_cost, 2),
        }

    def reset(self) -> None:
        """重置策略状态 / Reset strategy state."""
        self._accumulated_base = 0.0
        self._total_invested = 0.0
        self._avg_cost = 0.0
        self._last_purchase_price = 0.0
        self._last_execute_time = 0.0
        self._volatility = 0.0
        self._vol_multiplier = 1.0
        self._purchase_count = 0
        self._current_price = 0.0
        self._price_history = []

    def get_required_history(self) -> int:
        """获取所需历史数据长度 / Get required history length."""
        # 需要足够数据计算波动率 / Need enough data for volatility calculation
        return max(self.params.interval_seconds // 60, 100)
