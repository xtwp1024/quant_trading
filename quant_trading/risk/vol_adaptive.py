"""
Volatility-Adaptive Risk Manager.

波动率自适应风控管理器:
根据市场波动状态动态调整:
  1. 仓位大小 / Position size
  2. 止损幅度 / Stop loss
  3. 最大持仓数 / Max positions
  4. 止盈目标 / Profit target

Adopted from PortfolioStrategyBacktestUS and five_factors_riskControl concepts.
Pure NumPy/Pandas implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .volatility_gate import VolatilityGate

__all__ = ["VolAdaptiveRiskManager"]


# ------------------------------------------------------------------
# Regime tables — calibrated for equity markets (252-day year)
# ------------------------------------------------------------------

# Position size by volatility state (fraction of base position)
_POSITION_SIZE_TABLE = {
    "low": 1.0,      # Low vol: full exposure
    "normal": 0.85,  # Normal: slight reduction
    "high": 0.5,     # High vol: cut to half
    "extreme": 0.25, # Extreme vol: quarter position
}

# Stop loss by volatility state (annualized %)
_STOP_LOSS_TABLE = {
    "low": 0.20,      # 20% stop in low-vol regime
    "normal": 0.15,   # 15% stop in normal regime
    "high": 0.10,     # 10% stop in high-vol regime
    "extreme": 0.06,  # 6% stop in extreme-vol regime
}

# Max positions by volatility state
_MAX_POSITIONS_TABLE = {
    "low": 20,
    "normal": 15,
    "high": 8,
    "extreme": 4,
}

# Profit target (annualized %)
_PROFIT_TARGET_TABLE = {
    "low": 0.40,
    "normal": 0.25,
    "high": 0.15,
    "extreme": 0.08,
}


class VolAdaptiveRiskManager:
    """Volatility-adaptive risk manager.

    根据市场波动状态动态调整风险参数:
    - 仓位大小 (position_size)
    - 止损幅度 (stop_loss_pct)
    - 最大持仓数 (max_positions)
    - 止盈目标 (profit_target_pct)

    Attributes:
        base_position: 基础仓位 (default 1.0 = full).
        vol_gate: VolatilityGate instance for volatility breakout detection.
    """

    def __init__(
        self,
        base_position: float = 1.0,
        vol_gate: Optional[VolatilityGate] = None,
    ):
        """Initialize VolAdaptiveRiskManager.

        Args:
            base_position: 基础仓位 (0–1). Default 1.0.
            vol_gate: VolatilityGate instance. Creates a default one if None.
        """
        if not (0 < base_position <= 1):
            raise ValueError("base_position must be in (0, 1].")

        self.base_position = base_position
        self.vol_gate = vol_gate if vol_gate is not None else VolatilityGate()

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    @staticmethod
    def classify_vol_regime(vol_ratio: float, threshold: float = 1.5) -> str:
        """Classify volatility regime from vol_ratio.

        Args:
            vol_ratio: 当前波动率 / 长期均值.
            threshold: 门限系数 (default 1.5).

        Returns:
            'low' | 'normal' | 'high' | 'extreme'
        """
        if vol_ratio < 0.6:
            return "low"
        elif vol_ratio < threshold:
            return "normal"
        elif vol_ratio < 2.0:
            return "high"
        else:
            return "extreme"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_risk_params(
        self,
        returns: pd.Series,
        regime: str = "normal",
    ) -> dict:
        """计算当前风险参数 / Compute current risk parameters.

        Args:
            returns: 收益率序列 / Return series.
            regime: 手动指定波动状态 (default 'normal').
                   If 'auto', derives from vol_gate signal.

        Returns:
            dict with keys:
                position_size: float — 仓位比例 (0–1)
                stop_loss_pct: float — 止损幅度 (年化小数, e.g. 0.15 = 15%)
                max_positions: int — 最大持仓数
                profit_target_pct: float — 止盈目标 (年化小数)
                vol_state: str — 'low' | 'normal' | 'high' | 'extreme'
        """
        if regime == "auto":
            signal = self.vol_gate.compute_signal(returns)
            vol_state = signal["vol_state"]
        else:
            vol_state = regime

        # Look up tables using vol_state
        position_size = _POSITION_SIZE_TABLE.get(vol_state, 1.0)
        stop_loss_pct = _STOP_LOSS_TABLE.get(vol_state, 0.15)
        max_positions = _MAX_POSITIONS_TABLE.get(vol_state, 15)
        profit_target_pct = _PROFIT_TARGET_TABLE.get(vol_state, 0.25)

        # Apply base_position scaling
        effective_position = self.base_position * position_size

        return {
            "position_size": float(effective_position),
            "stop_loss_pct": float(stop_loss_pct),
            "max_positions": int(max_positions),
            "profit_target_pct": float(profit_target_pct),
            "vol_state": vol_state,
        }

    # ------------------------------------------------------------------
    # Convenience: full signal
    # ------------------------------------------------------------------

    def get_full_signal(self, returns: pd.Series) -> dict:
        """Return combined volatility gate + adaptive risk params.

        Args:
            returns: 收益率序列 / Return series.

        Returns:
            Merged signal dict from VolatilityGate + computed risk params.
        """
        gate_signal = self.vol_gate.compute_signal(returns)
        risk_params = self.compute_risk_params(returns, regime="auto")

        return {
            **gate_signal,
            **risk_params,
        }
