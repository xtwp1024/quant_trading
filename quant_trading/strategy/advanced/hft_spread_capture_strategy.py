"""
HFT Spread Capture Strategy — 高频价差捕捉策略适配器
=================================================

基于 `quant_trading.strategy.hft_strategies.SpreadCaptureStrategy` 的 BaseStrategy 适配器。

使用 Kalman 滤波器估计真实中间价，当价差偏离均值时入场。
适用于高频做市和价差捕捉。

Classes
-------
HFTSpreadCaptureStrategy
    BaseStrategy adapter for spread capture trading.
HFTSpreadCaptureParams
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
class HFTSpreadCaptureParams(StrategyParams):
    """高频价差捕捉策略参数"""
    spread_threshold: float = 0.001
    """价差 z-score 阈值"""
    kalman_covariance: float = 1e-4
    """Kalman 滤波器过程噪声"""
    kalman_observation_cov: float = 1.0
    """Kalman 滤波器观测噪声"""
    position_size: float = 100.0
    """每次交易仓位大小"""


class HFTSpreadCaptureStrategy(BaseStrategy):
    """
    高频价差捕捉策略

    使用 Kalman 滤波器平滑价格噪声，当价差偏离均值时入场：
    - 价差扩大超过阈值：预期均值回归，fade the move
    - 价差收窄：捕捉价差做市

    Parameters
    ----------
    symbol : str
        交易标的符号
    params : HFTSpreadCaptureParams, optional
        策略参数
    """

    name = "hft_spread_capture"

    def __init__(
        self,
        symbol: str,
        params: Optional[HFTSpreadCaptureParams] = None,
    ) -> None:
        super().__init__(symbol, params or HFTSpreadCaptureParams())

        # Kalman 滤波器状态
        self._state_mean: float = 0.0
        self._state_covariance: float = 1.0

        # 仓位
        self._position: float = 0.0
        self._avg_price: float = 0.0

        # 历史数据
        self._price_history: List[float] = []
        self._spread_history: List[float] = []

    def _update_kalman(self, observed_price: float) -> None:
        """更新 Kalman 滤波器"""
        # 预测步骤
        pred_mean = self._state_mean
        pred_cov = self._state_covariance + self.params.kalman_covariance

        # 更新步骤
        kalman_gain = pred_cov / (pred_cov + self.params.kalman_observation_cov)
        self._state_mean = pred_mean + kalman_gain * (observed_price - pred_mean)
        self._state_covariance = pred_cov * (1 - kalman_gain)

    def _compute_signal_from_spread(self, spread: float) -> Optional[SignalType]:
        """根据价差计算信号方向"""
        if len(self._spread_history) < 20:
            return None

        spread_mean = np.mean(self._spread_history[-20:])
        spread_std = np.std(self._spread_history[-20:]) + 1e-9
        spread_zscore = (spread - spread_mean) / spread_std

        if spread_zscore > self.params.spread_threshold:
            return SignalType.SELL  # 价差扩大，平仓
        elif spread_zscore < -self.params.spread_threshold:
            return SignalType.BUY  # 价差收窄，入场
        return None

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号。

        Parameters
        ----------
        data : pd.DataFrame
            K线或tick数据，需包含 bid, ask 列（或 high, low, close）

        Returns
        -------
        List[Signal]
            交易信号列表
        """
        signals = []

        if len(data) < 2:
            return signals

        # 提取 bid/ask
        if 'bid' in data.columns and 'ask' in data.columns:
            bid = float(data['bid'].iloc[-1])
            ask = float(data['ask'].iloc[-1])
        else:
            # 假设 high/low 作为 ask/bid 的近似
            ask = float(data['high'].iloc[-1])
            bid = float(data['low'].iloc[-1])

        spread = ask - bid
        mid_price = (bid + ask) / 2.0

        self._price_history.append(mid_price)
        self._spread_history.append(spread)

        # 限制历史长度
        if len(self._price_history) > 1000:
            self._price_history.pop(0)
            self._spread_history.pop(0)

        # 更新 Kalman 滤波器
        self._update_kalman(mid_price)

        # 计算信号
        sig_type = self._compute_signal_from_spread(spread)
        if sig_type is None:
            return signals

        # 获取时间戳
        timestamp = int(data.iloc[-1].get("timestamp", 0))

        signals.append(
            Signal(
                type=sig_type,
                symbol=self.symbol,
                timestamp=timestamp,
                price=mid_price,
                strength=1.0,
                metadata={
                    "strategy": "hft_spread_capture",
                    "spread": spread,
                    "mid_price": mid_price,
                    "position": self._position,
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
        if 'bid' not in tick and 'high' not in tick:
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
        self._avg_price = position.get("avg_price", self._avg_price)

    def get_required_history(self) -> int:
        """返回所需历史数据长度"""
        return 50

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "spread_threshold": self.params.spread_threshold,
                "kalman_covariance": self.params.kalman_covariance,
                "kalman_observation_cov": self.params.kalman_observation_cov,
                "position_size": self.params.position_size,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HFTSpreadCaptureStrategy":
        """从字典反序列化"""
        params = HFTSpreadCaptureParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
