"""
Kalman Pairs Strategy — 卡尔曼滤波配对交易策略适配器
====================================================

基于 `quant_trading.strategy.kalman_bot.PairsTradingStrategy` 的 BaseStrategy 适配器。

使用卡尔曼滤波跟踪两个资产之间的时变对冲比率（协整系数）。

模型
----
price_s2_t = β_t[0] · price_s1_t + β_t[1] + ε_t
β_t       = β_{t-1} + η_t         (random walk)

状态   x_t = β_t  ∈ ℝ²
观察   y_t = price_s2_t
H_t     = [price_s1_t, 1]

交易规则 (Bollinger):
    long  spread  when e_t < −√Q_t   (spread undervalued → expect rebound)
    short spread  when e_t > +√Q_t   (spread overvalued  → expect contraction)
    exit         when e_t crosses 0

Classes
-------
KalmanPairsStrategyAdapter
    BaseStrategy adapter for Kalman Filter-based pairs trading.
KalmanPairsParams
    Strategy parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_trading.signal.types import Signal, SignalType, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext
from quant_trading.strategy.kalman_bot import (
    KalmanFilter,
    SpreadSignalGenerator,
)


@dataclass
class KalmanPairsParams(StrategyParams):
    """卡尔曼配对交易策略参数"""
    q: float = 1e-3
    """对冲比率的过程噪声标准差（控制 β 变化速度）。越小越稳定，越大越灵活。典型范围: 1e-4 到 1e-2。"""
    r: float = 1e-4
    """观测噪声标准差。典型值：OLS 回归残差标准差 × sqrt(1e-4)。"""
    threshold_mult: float = 1.0
    """布林带乘数（默认 1.0）。"""
    hedge: str = "long"
    """对冲方向: 'long' = 做多价差（做空 s1，做多 s2），'short' = 相反。"""
    symbol1: str = "ASSET1"
    """第一个资产符号。"""
    symbol2: str = "ASSET2"
    """第二个资产符号。"""
    position_size: float = 0.1
    """基础持仓大小（占组合的比例）。"""


class KalmanPairsStrategyAdapter(BaseStrategy):
    """
    基于卡尔曼滤波的配对交易策略。

    使用卡尔曼滤波跟踪两个资产之间的时变对冲比率（协整系数），
    当价差偏离均值时生成交易信号。

    信号逻辑：
      - e_t < −√Q_t → 做多价差（预期反弹）
      - e_t > +√Q_t → 做空价差（预期收缩）
      - e_t 穿过 0 → 平仓

    Parameters
    ----------
    symbol : str
        主要交易对符号（例如 "PAIR1/PAIR2"）。
    params : KalmanPairsParams, optional
        策略参数。
    """

    name = "kalman_pairs"

    def __init__(
        self,
        symbol: str,
        params: Optional[KalmanPairsParams] = None,
    ) -> None:
        super().__init__(symbol, params or KalmanPairsParams())

        # Initialize Kalman Filter
        self.F = np.eye(2)  # random-walk state transition
        self.Q = (self.params.q ** 2) * np.eye(2)
        self.R = (self.params.r ** 2) * np.eye(1)

        self._kf: KalmanFilter = KalmanFilter(
            F=self.F, Q=self.Q, R=self.R, m=2, n=1
        )

        self._signal_generator = SpreadSignalGenerator(self.params.threshold_mult)

        # State
        self._x_history: List[np.ndarray] = []
        self._spread_history: List[float] = []
        self._signal_history: List[int] = []
        self._current_position: int = 0
        self._last_spread: Optional[float] = None

    def _init_filter(self) -> None:
        """Initialize the Kalman filter with t=0 priors."""
        x0 = np.array([[0.0], [0.0]])  # initial state [hedge_ratio, intercept]
        P0 = np.eye(2) * 1e-3  # initial uncertainty
        self._kf.init_sequence(x0, P0)

    def _update_filter(self, price_s1: float, price_s2: float) -> Tuple[float, float]:
        """
        Update Kalman filter with new price observations.

        Returns
        -------
        Tuple[float, float]
            (spread, spread_variance)
        """
        H_t = np.array([[price_s1, 1.0]])
        y_t = np.array([[price_s2]])

        x_post = self._kf.update(y_t, H_t)
        hedge_ratio = x_post[0, 0]
        intercept = x_post[1, 0]

        # Spread is the innovation: e_t = y_t - H_t · x_t
        spread = price_s2 - (hedge_ratio * price_s1 + intercept)

        # Innovation variance (approximation using observation noise)
        spread_var = float(self._kf.R[0, 0])

        return spread, spread_var

    def _get_spread_and_variance(
        self, data: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract price series and compute spread/variance series.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns for two assets' prices.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            (spreads, spread_vars) or (None, None) if insufficient data
        """
        # Try to find price columns for both assets
        price1_col = None
        price2_col = None

        for col in data.columns:
            col_lower = col.lower()
            if self.params.symbol1.lower() in col_lower or 'price1' in col_lower or 's1' in col_lower:
                price1_col = col
            if self.params.symbol2.lower() in col_lower or 'price2' in col_lower or 's2' in col_lower:
                price2_col = col

        # Fallback: look for generic price columns
        if price1_col is None:
            for col in data.columns:
                if 'close' in col.lower() or 'price' in col.lower():
                    if price1_col is None:
                        price1_col = col
                    elif price2_col is None:
                        price2_col = col
                        break

        if price1_col is None or price2_col is None or price1_col == price2_col:
            # Cannot find two distinct price columns
            return None, None

        prices1 = data[price1_col].values
        prices2 = data[price2_col].values

        if len(prices1) < 2 or len(prices2) < 2:
            return None, None

        # Initialize filter on first observation
        if len(self._kf.x_history) <= 1:
            self._init_filter()
            # Warm up filter
            for i in range(min(50, len(prices1) - 1)):
                self._update_filter(prices1[i], prices2[i])

        spreads = []
        spread_vars = []

        for i in range(len(prices1)):
            spread, spread_var = self._update_filter(prices1[i], prices2[i])
            spreads.append(spread)
            spread_vars.append(spread_var)

        return np.array(spreads), np.array(spread_vars)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from Kalman filter-based spread analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns for two assets' prices.

        Returns
        -------
        List[Signal]
            Trading signals for the pairs trade.
        """
        signals = []

        spreads, spread_vars = self._get_spread_and_variance(data)

        if spreads is None or len(spreads) == 0:
            return signals

        # Generate discrete signals
        raw_signals, raw_positions = self._signal_generator.generate(spreads, spread_vars)

        # Handle hedge direction
        if self.params.hedge == "short":
            raw_signals = -raw_signals
            raw_positions = -raw_positions

        # Take the last signal
        if len(raw_signals) > 0:
            current_signal = int(raw_signals[-1])
            self._current_position = current_signal

            last_row = data.iloc[-1]
            timestamp = int(last_row.get("timestamp", 0))
            price = float(last_row.get("close", 1.0))

            # Map spread signal to Signal object
            if current_signal == 1:
                signal_type = SignalType.BUY
                reason = f"Kalman Pairs: Long spread (hedge_ratio={self._kf.x_posterior[0,0]:.4f})"
            elif current_signal == -1:
                signal_type = SignalType.SELL
                reason = f"Kalman Pairs: Short spread (hedge_ratio={self._kf.x_posterior[0,0]:.4f})"
            else:
                signal_type = SignalType.HOLD
                reason = "Kalman Pairs: Flat position"

            signals.append(
                Signal(
                    type=signal_type,
                    symbol=self.symbol,
                    timestamp=timestamp,
                    price=price,
                    strength=1.0,
                    reason=reason,
                    metadata={
                        "strategy": "kalman_pairs",
                        "hedge_ratio": float(self._kf.x_posterior[0, 0]),
                        "intercept": float(self._kf.x_posterior[1, 0]),
                        "spread": float(spreads[-1]) if len(spreads) > 0 else 0.0,
                        "spread_var": float(spread_vars[-1]) if len(spread_vars) > 0 else 0.0,
                        "signal1": int(raw_signals[-1]) if len(raw_signals) > 0 else 0,
                        "position": float(raw_positions[-1]) if len(raw_positions) > 0 else 0.0,
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
        Calculate position size for pairs trade.

        For pairs trading, the position size is typically the hedge ratio
        multiplied by the base position size.

        Parameters
        ----------
        signal : Signal
            The trading signal.
        context : StrategyContext
            Current strategy context.

        Returns
        -------
        float
            Position size as fraction of portfolio.
        """
        # Use the hedge ratio to scale position
        hedge_ratio = signal.metadata.get("hedge_ratio", 1.0) if signal.metadata else 1.0
        base_size = self.params.position_size

        return base_size * abs(hedge_ratio)

    def get_required_history(self) -> int:
        """Require sufficient history for Kalman filter warm-up."""
        return 100

    def get_current_hedge_ratio(self) -> float:
        """Get the current estimated hedge ratio."""
        if len(self._kf.x_history) > 0:
            return float(self._kf.x_history[-1][0, 0])
        return 0.0

    def to_dict(self) -> dict:
        """Serialize strategy to dict."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "q": self.params.q,
                "r": self.params.r,
                "threshold_mult": self.params.threshold_mult,
                "hedge": self.params.hedge,
                "symbol1": self.params.symbol1,
                "symbol2": self.params.symbol2,
                "position_size": self.params.position_size,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KalmanPairsStrategyAdapter":
        """Deserialize strategy from dict."""
        params = KalmanPairsStrategyParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
