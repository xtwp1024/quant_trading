"""
HFT Latency Arbitrage Strategy — 延迟套利策略适配器
=================================================

基于 `quant_trading.strategy.hft_strategies.LatencyArbitrageStrategy` 的 BaseStrategy 适配器。

利用跨交易所传输延迟导致的价格差异进行套利。

Classes
-------
HFTLatencyArbStrategy
    BaseStrategy adapter for latency arbitrage trading.
HFTLatencyArbParams
    Strategy parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalDirection, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class HFTLatencyArbParams(StrategyParams):
    """延迟套利策略参数"""
    venues: List[str] = field(default_factory=lambda: ["binance", "okx", "kucoin"])
    """监控的交易所列表"""
    spread_threshold: float = 0.0002
    """价差阈值（百分比）"""
    position_size: float = 100.0
    """每次交易仓位大小"""
    max_spread_age: float = 0.001
    """价格过期时间（秒）"""


class HFTLatencyArbStrategy(BaseStrategy):
    """
    延迟套利策略

    利用跨交易所传输延迟导致的价格差异进行套利：
    - 监控多个交易所的价格
    - 价格差异 > 阈值：买入便宜的，卖出贵的
    - 价差收拢时平仓

    Parameters
    ----------
    symbol : str
        交易标的符号
    params : HFTLatencyArbParams, optional
        策略参数
    """

    name = "hft_latency_arb"

    def __init__(
        self,
        symbol: str,
        params: Optional[HFTLatencyArbParams] = None,
    ) -> None:
        super().__init__(symbol, params or HFTLatencyArbParams())

        # 交易所价格: {venue: (price, timestamp)}
        self._venue_prices: Dict[str, Tuple[float, float]] = {
            v: (0.0, 0.0) for v in self.params.venues
        }

        # 净仓位
        self._net_position: float = 0.0

        # 历史数据
        self._spread_history: List[float] = []

    def update_venue_price(
        self,
        venue: str,
        price: float,
        timestamp: float,
    ) -> None:
        """
        更新特定交易所的价格。

        Parameters
        ----------
        venue : str
            交易所名称
        price : float
            价格
        timestamp : float
            时间戳
        """
        self._venue_prices[venue] = (price, timestamp)

    def _compute_cross_venue_spread(self) -> Tuple[float, float, List[str]]:
        """
        计算跨交易所价差。

        Returns
        -------
        Tuple[float, float, List[str]]
            (最大价差, 加权价差, 活跃交易所列表)
        """
        active = [
            (v, p, t)
            for v, (p, t) in self._venue_prices.items()
            if t > 0 and self._venue_prices[v][0] > 0
        ]

        if len(active) < 2:
            return 0.0, 0.0, []

        prices = [(v, p) for v, p, _ in active]
        max_idx = max(range(len(prices)), key=lambda i: prices[i][1])
        min_idx = min(range(len(prices)), key=lambda i: prices[i][1])

        max_spread = prices[max_idx][1] - prices[min_idx][1]
        weighted_spread = max_spread / (prices[min_idx][1] + 1e-9)

        return max_spread, weighted_spread, [prices[max_idx][0], prices[min_idx][0]]

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号。

        Parameters
        ----------
        data : pd.DataFrame
            市场数据，需包含 venue, price, timestamp 列

        Returns
        -------
        List[Signal]
            交易信号列表
        """
        signals = []

        if len(data) < 1:
            return signals

        # 更新各交易所价格
        for _, row in data.iterrows():
            venue = str(row.get("venue", ""))
            if venue in self._venue_prices:
                self.update_venue_price(
                    venue,
                    float(row.get("price", 0.0)),
                    float(row.get("timestamp", 0.0)),
                )

        max_spread, weighted_spread, active_venues = self._compute_cross_venue_spread()
        self._spread_history.append(weighted_spread)

        if len(self._spread_history) > 1000:
            self._spread_history.pop(0)

        if len(active_venues) < 2:
            return signals

        # 检查价格是否过期
        now = float(data.iloc[-1].get("timestamp", 0.0))
        for v, (p, t) in self._venue_prices.items():
            if t > 0 and now - t > self.params.max_spread_age:
                return signals  # 价格太旧

        # 信号生成
        if weighted_spread > self.params.spread_threshold:
            sig_type = SignalType.SELL  # 卖出贵的
        elif weighted_spread < -self.params.spread_threshold:
            sig_type = SignalType.BUY  # 买入便宜的
        else:
            return signals

        timestamp = int(now)

        signals.append(
            Signal(
                type=sig_type,
                symbol=self.symbol,
                timestamp=timestamp,
                price=float(data.iloc[-1].get("price", 0.0)),
                strength=1.0,
                metadata={
                    "strategy": "hft_latency_arb",
                    "max_spread": max_spread,
                    "weighted_spread": weighted_spread,
                    "active_venues": active_venues,
                },
            )
        )

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """
        计算仓位大小。

        Parameters
        ----------
        signal : Signal
            交易信号
        context : StrategyContext
            策略上下文

        Returns
        -------
        float
            仓位大小
        """
        return self.params.position_size

    def on_bar(self, bar: pd.Series) -> Optional[Signal]:
        """处理单根K线数据"""
        df = pd.DataFrame([bar])
        signals = self.generate_signals(df)
        return signals[-1] if signals else None

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """处理tick数据"""
        if "venue" not in tick or "price" not in tick:
            return None
        df = pd.DataFrame([tick])
        signals = self.generate_signals(df)
        return signals[-1] if signals else None

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """订单成交回调"""
        pass

    def on_position_changed(self, position: Dict[str, Any]) -> None:
        """持仓变化回调"""
        self._net_position = position.get("quantity", self._net_position)

    def get_required_history(self) -> int:
        """返回所需历史数据长度"""
        return 10

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "venues": self.params.venues,
                "spread_threshold": self.params.spread_threshold,
                "position_size": self.params.position_size,
                "max_spread_age": self.params.max_spread_age,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HFTLatencyArbStrategy":
        """从字典反序列化"""
        params = HFTLatencyArbParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
