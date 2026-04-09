"""
Volatility Gate Risk Control.

波动率门限风控模块:
- 当短期波动率 > 长期均值 * bound系数 → 减仓50%
- 当波动率回归正常 → 恢复满仓

论文逻辑:
short_vol > long_vol_mean * threshold → position *= reduction_factor
vol_ratio < threshold * 0.8 → 恢复满仓 (回撤门限防止抖动)

Adopted from PortfolioStrategyBacktestUS (risk-management concepts).
Pure NumPy/Pandas implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["VolatilityGate"]


class VolatilityGate:
    """Volatility gate risk controller.

    规则 / Rules:
        - 当短期波动率 > 长期均值 * threshold_multiplier → 减仓50%
        - 当波动率回归正常 (vol_ratio < threshold * 0.8) → 恢复满仓

    论文逻辑 / Paper Logic:
        short_vol > long_vol_mean * threshold → position *= reduction_factor
        vol_ratio < threshold * 0.8 → full position (hysteresis to avoid churn)

    Attributes:
        short_window: 短期窗口 (天) / Short rolling window in days.
        long_window: 长期窗口 (天) / Long rolling window in days.
        threshold_multiplier: 门限系数 / Threshold multiplier (e.g. 1.5 = 50% above long-run mean).
        reduction_factor: 减仓系数 / Position reduction factor (0.5 = cut to 50%).
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        threshold_multiplier: float = 1.5,
        reduction_factor: float = 0.5,
    ):
        """Initialize VolatilityGate.

        Args:
            short_window: 短期窗口 (天). Default 5.
            long_window: 长期窗口 (天). Default 20.
            threshold_multiplier: 门限系数. Default 1.5.
            reduction_factor: 减仓系数. Default 0.5 (cut to 50%).
        """
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Windows must be positive integers.")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window.")
        if not (0 < reduction_factor <= 1):
            raise ValueError("reduction_factor must be in (0, 1].")
        if threshold_multiplier <= 0:
            raise ValueError("threshold_multiplier must be positive.")

        self.short_window = short_window
        self.long_window = long_window
        self.threshold_multiplier = threshold_multiplier
        self.reduction_factor = reduction_factor

        # Hysteresis band: recovery only when vol_ratio drops below threshold * 0.8
        self._recovery_threshold = threshold_multiplier * 0.8

    # ------------------------------------------------------------------
    # Core computation methods
    # ------------------------------------------------------------------

    @staticmethod
    def compute_volatility(returns: np.ndarray | pd.Series) -> float:
        """计算波动率 (年化) / Compute annualized volatility.

        Args:
            returns: 收益率序列 / Return series.

        Returns:
            年化波动率 (float). / Annualized volatility.
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        else:
            returns = np.asarray(returns).flatten()
            returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return 0.0

        # Daily std, annualized by sqrt(252)
        daily_std = np.std(returns, ddof=1)
        return float(daily_std * np.sqrt(252))

    def compute_long_vol_mean(self, returns: pd.Series) -> float:
        """Compute long-run average volatility over long_window.

        Args:
            returns: 收益率序列 / Return series.

        Returns:
            Long-run mean annualized volatility.
        """
        if len(returns) < self.long_window:
            # Not enough data — use all available
            return self.compute_volatility(returns)

        rolling_vol = returns.rolling(window=self.long_window).std().dropna()
        # Annualize each rolling window vol
        rolling_vol_ann = rolling_vol * np.sqrt(252)
        return float(rolling_vol_ann.mean())

    def should_reduce(self, returns: pd.Series) -> tuple[bool, float]:
        """判断是否应该减仓 / Determine whether to reduce position.

        Args:
            returns: 最近 short_window 天的收益率序列 / Return series for recent short_window days.

        Returns:
            (should_reduce, current_vol_ratio):
                should_reduce: 是否应减仓
                current_vol_ratio: 当前波动率 / 长期均值
        """
        if len(returns) < self.short_window:
            return False, 0.0

        short_vol = self.compute_volatility(returns.iloc[-self.short_window:])
        long_vol_mean = self.compute_long_vol_mean(returns)

        if long_vol_mean <= 0:
            return False, 0.0

        vol_ratio = short_vol / long_vol_mean
        should_reduce = vol_ratio > self.threshold_multiplier

        return bool(should_reduce), float(vol_ratio)

    def get_target_position(
        self,
        current_position: float,
        returns: pd.Series,
    ) -> float:
        """获取目标仓位 / Get target position size.

        逻辑 / Logic:
            vol_ratio > threshold → position * reduction_factor
            vol_ratio < threshold * 0.8 → 恢复满仓 (hysteresis)
            otherwise → maintain current position

        Args:
            current_position: 当前仓位 (0–1). / Current position (0–1).
            returns: 收益率序列 / Return series.

        Returns:
            目标仓位 (float, 0–1). / Target position.
        """
        should_reduce, vol_ratio = self.should_reduce(returns)

        if should_reduce:
            # Volatility breakout — reduce position
            return float(current_position * self.reduction_factor)

        # Recovery check: only restore full position when vol_ratio
        # falls well below threshold (hysteresis band)
        if vol_ratio < self._recovery_threshold and vol_ratio > 0:
            return 1.0

        # Maintain current position within the hysteresis zone
        return float(current_position)

    def compute_signal(self, returns: pd.Series) -> dict:
        """返回完整风控信号 / Return complete risk-control signal.

        Args:
            returns: 收益率序列 / Return series.

        Returns:
            dict with keys:
                should_reduce: bool — 是否应减仓
                current_vol: float — 当前短期年化波动率
                long_vol_mean: float — 长期平均年化波动率
                vol_ratio: float — 波动率比率
                target_position: float — 目标仓位
                vol_state: str — 'low' | 'normal' | 'high' | 'extreme'
        """
        if len(returns) < self.short_window:
            return {
                "should_reduce": False,
                "current_vol": 0.0,
                "long_vol_mean": 0.0,
                "vol_ratio": 0.0,
                "target_position": 1.0,
                "vol_state": "normal",
            }

        short_vol = self.compute_volatility(returns.iloc[-self.short_window:])
        long_vol_mean = self.compute_long_vol_mean(returns)

        if long_vol_mean <= 0:
            vol_ratio = 0.0
        else:
            vol_ratio = short_vol / long_vol_mean

        # Volatility regime classification
        if vol_ratio < 0.6:
            vol_state = "low"
        elif vol_ratio < self.threshold_multiplier:
            vol_state = "normal"
        elif vol_ratio < 2.0:
            vol_state = "high"
        else:
            vol_state = "extreme"

        should_reduce = vol_ratio > self.threshold_multiplier
        target_position = (
            self.reduction_factor if should_reduce else
            (1.0 if vol_ratio < self._recovery_threshold else 1.0)
        )

        return {
            "should_reduce": bool(should_reduce),
            "current_vol": float(short_vol),
            "long_vol_mean": float(long_vol_mean),
            "vol_ratio": float(vol_ratio),
            "target_position": float(target_position),
            "vol_state": vol_state,
        }
