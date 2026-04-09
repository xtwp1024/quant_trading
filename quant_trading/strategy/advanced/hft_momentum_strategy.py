"""
HFT Momentum Strategy — 高频动量策略适配器
=========================================

基于 `quant_trading.strategy.hft_strategies.MomentumSignalStrategy` 的 BaseStrategy 适配器。

使用短期价格动量和成交量加速度生成高频交易信号。

Classes
-------
HFTMomentumStrategy
    BaseStrategy adapter for momentum signal trading.
HFTMomentumParams
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
class HFTMomentumParams(StrategyParams):
    """高频动量策略参数"""
    short_window: int = 5
    """短期窗口"""
    long_window: int = 20
    """长期窗口"""
    volume_window: int = 10
    """成交量窗口"""
    momentum_threshold: float = 0.0005
    """动量阈值"""
    position_size: float = 100.0
    """每次交易仓位大小"""


class HFTMomentumStrategy(BaseStrategy):
    """
    高频动量策略

    使用短期价格动量和成交量加速度生成信号：
    - 价格动量：短期MA vs 长期MA交叉
    - 成交量加速度：成交量变化确认动量
    - 信号：动量 + 成交量确认

    Parameters
    ----------
    symbol : str
        交易标的符号
    params : HFTMomentumParams, optional
        策略参数
    """

    name = "hft_momentum"

    def __init__(
        self,
        symbol: str,
        params: Optional[HFTMomentumParams] = None,
    ) -> None:
        super().__init__(symbol, params or HFTMomentumParams())

        self._prices: List[float] = []
        self._volumes: List[float] = []
        self._position: float = 0.0

    def _compute_momentum(self) -> tuple[float, float]:
        """计算价格动量和成交量加速度"""
        if len(self._prices) < self.params.long_window:
            return 0.0, 0.0

        prices = np.array(self._prices[-self.params.long_window:])
        volumes = (
            np.array(self._volumes[-self.params.volume_window:])
            if len(self._volumes) >= self.params.volume_window
            else np.array([1.0])
        )

        # 价格动量：对数收益率
        log_returns = np.diff(np.log(prices + 1e-9))
        momentum = np.mean(log_returns[-self.params.short_window:])

        # 成交量加速度
        volume_gradient = np.gradient(volumes)
        volume_accel = np.mean(np.gradient(volume_gradient))

        return momentum, volume_accel

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号。

        Parameters
        ----------
        data : pd.DataFrame
            K线数据，需包含 close, volume 列

        Returns
        -------
        List[Signal]
            交易信号列表
        """
        signals = []

        if len(data) < 2:
            return signals

        price = float(data["close"].iloc[-1])
        volume = float(data.get("volume", 0).iloc[-1]) if "volume" in data.columns else 0.0

        self._prices.append(price)
        self._volumes.append(volume)

        # 限制长度
        if len(self._prices) > 500:
            self._prices.pop(0)
            self._volumes.pop(0)

        momentum, volume_accel = self._compute_momentum()

        if len(self._prices) < self.params.long_window:
            return signals

        # 成交量确认
        volume_confirm = 1 if volume_accel > 0 else 0.5

        # 信号生成
        if momentum > self.params.momentum_threshold * volume_confirm:
            sig_type = SignalType.BUY
        elif momentum < -self.params.momentum_threshold * volume_confirm:
            sig_type = SignalType.SELL
        else:
            return signals

        timestamp = int(data.iloc[-1].get("timestamp", 0))

        signals.append(
            Signal(
                type=sig_type,
                symbol=self.symbol,
                timestamp=timestamp,
                price=price,
                strength=1.0,
                metadata={
                    "strategy": "hft_momentum",
                    "momentum": momentum,
                    "volume_accel": volume_accel,
                    "volume_confirm": volume_confirm,
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
        return None

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """订单成交回调"""
        pass

    def on_position_changed(self, position: Dict[str, Any]) -> None:
        """持仓变化回调"""
        self._position = position.get("quantity", self._position)

    def get_required_history(self) -> int:
        """返回所需历史数据长度"""
        return self.params.long_window + 10

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "short_window": self.params.short_window,
                "long_window": self.params.long_window,
                "volume_window": self.params.volume_window,
                "momentum_threshold": self.params.momentum_threshold,
                "position_size": self.params.position_size,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HFTMomentumStrategy":
        """从字典反序列化"""
        params = HFTMomentumParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
