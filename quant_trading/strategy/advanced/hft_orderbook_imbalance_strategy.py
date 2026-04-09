"""
HFT Order Book Imbalance Strategy — 订单簿不平衡策略适配器
=====================================================

基于 `quant_trading.strategy.hft_strategies.OrderBookImbalanceStrategy` 的 BaseStrategy 适配器。

分析限价订单簿的不平衡来预测短期价格方向。

Classes
-------
HFTOrderBookImbalanceStrategy
    BaseStrategy adapter for order book imbalance trading.
HFTOrderBookImbalanceParams
    Strategy parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalDirection, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


@dataclass
class HFTOrderBookImbalanceParams(StrategyParams):
    """订单簿不平衡策略参数"""
    imbalance_threshold: float = 0.3
    """不平衡阈值"""
    depth_levels: int = 5
    """聚合的档位数"""
    position_size: float = 100.0
    """每次交易仓位大小"""


class HFTOrderBookImbalanceStrategy(BaseStrategy):
    """
    订单簿不平衡策略

    分析限价订单簿(LOB)的不平衡来预测短期价格方向：
    - OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    - OBI > 阈值：看涨，价格可能上升
    - OBI < -阈值：看跌，价格可能下降

    Parameters
    ----------
    symbol : str
        交易标的符号
    params : HFTOrderBookImbalanceParams, optional
        策略参数
    """

    name = "hft_orderbook_imbalance"

    def __init__(
        self,
        symbol: str,
        params: Optional[HFTOrderBookImbalanceParams] = None,
    ) -> None:
        super().__init__(symbol, params or HFTOrderBookImbalanceParams())

        self._obi_history: List[float] = []
        self._position: float = 0.0

    def _compute_imbalance(
        self,
        bids: np.ndarray,
        asks: np.ndarray,
    ) -> float:
        """
        计算订单簿不平衡度。

        Parameters
        ----------
        bids : np.ndarray
            买单数组 [price, quantity] per level
        asks : np.ndarray
            卖单数组 [price, quantity] per level

        Returns
        -------
        float
            不平衡度 in [-1, 1]
        """
        # 聚合前N档
        bid_vol = np.sum(bids[:self.params.depth_levels, 1])
        ask_vol = np.sum(asks[:self.params.depth_levels, 1])

        total = bid_vol + ask_vol + 1e-9
        imbalance = (bid_vol - ask_vol) / total

        return imbalance

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号。

        Parameters
        ----------
        data : pd.DataFrame
            订单簿数据，需包含 bids, asks 列（嵌套列表或 numpy 数组）

        Returns
        -------
        List[Signal]
            交易信号列表
        """
        signals = []

        if len(data) < 1:
            return signals

        # 提取订单簿数据
        row = data.iloc[-1]

        bids_raw = row.get("bids", np.array([[0, 0]]))
        asks_raw = row.get("asks", np.array([[0, 0]]))

        # 如果是列表，转换为numpy数组
        if isinstance(bids_raw, list):
            bids = np.array(bids_raw)
        else:
            bids = bids_raw
        if isinstance(asks_raw, list):
            asks = np.array(asks_raw)
        else:
            asks = asks_raw

        if len(bids) == 0 or len(asks) == 0:
            return signals

        imbalance = self._compute_imbalance(bids, asks)
        self._obi_history.append(imbalance)

        # 限制长度
        if len(self._obi_history) > 1000:
            self._obi_history.pop(0)

        if len(self._obi_history) < 10:
            return signals

        # OBI移动平均
        obi_ma = np.mean(self._obi_history[-10:])

        # 信号生成
        if imbalance > self.params.imbalance_threshold and obi_ma > 0:
            sig_type = SignalType.BUY
        elif imbalance < -self.params.imbalance_threshold and obi_ma < 0:
            sig_type = SignalType.SELL
        else:
            return signals

        timestamp = int(row.get("timestamp", 0))
        mid_price = float(row.get("mid_price", 0.0)) if "mid_price" in row.index else 0.0

        signals.append(
            Signal(
                type=sig_type,
                symbol=self.symbol,
                timestamp=timestamp,
                price=mid_price,
                strength=1.0,
                metadata={
                    "strategy": "hft_orderbook_imbalance",
                    "imbalance": imbalance,
                    "obi_ma": obi_ma,
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
        if "bids" not in tick and "asks" not in tick:
            return None
        df = pd.DataFrame([tick])
        signals = self.generate_signals(df)
        return signals[-1] if signals else None

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """订单成交回调"""
        pass

    def on_position_changed(self, position: Dict[str, Any]) -> None:
        """持仓变化回调"""
        self._position = position.get("quantity", self._position)

    def get_required_history(self) -> int:
        """返回所需历史数据长度"""
        return 20

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "imbalance_threshold": self.params.imbalance_threshold,
                "depth_levels": self.params.depth_levels,
                "position_size": self.params.position_size,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HFTOrderBookImbalanceStrategy":
        """从字典反序列化"""
        params = HFTOrderBookImbalanceParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
