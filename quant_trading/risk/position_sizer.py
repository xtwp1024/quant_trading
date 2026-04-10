"""Position Sizing Implementations.

Provides multiple position sizing strategies:
- Kelly criterion (full Kelly and fractional)
- Fixed fraction / fixed ratio
- ATR-based position sizing
- Volatility-adjusted position sizing
- Risk-parity position sizing
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Dict, Any


class PositionSizingMethod(Enum):
    """Position sizing method types."""
    KELLY = "kelly"
    FRACTIONAL = "fractional"
    ATR_BASED = "atr_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    FIXED_RATIO = "fixed_ratio"


class KellySizer:
    """Kelly criterion position sizer.

    Kelly % = W - (1 - W) / R
    where W = winning rate, R = win/loss ratio
    """

    def __init__(self, fractional: float = 1.0):
        """
        Args:
            fractional: Fraction of Kelly to use (0.0-1.0). 0.5 = half Kelly is recommended.
        """
        self.fractional = fractional

    def calculate(
        self,
        account_balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_position_pct: float = 0.25,
    ) -> float:
        """
        Calculate position size using Kelly criterion.

        Args:
            account_balance: Current account balance
            win_rate: Historical win rate (0.0-1.0)
            avg_win: Average winning amount
            avg_loss: Average losing amount (positive value)
            max_position_pct: Maximum position as fraction of account

        Returns:
            Position size in account currency
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply fractional Kelly (reduce by half for risk management)
        kelly_pct *= self.fractional

        # Clamp to valid range
        kelly_pct = max(0.0, min(kelly_pct, max_position_pct))

        return account_balance * kelly_pct


class FractionalSizer:
    """Fixed fractional position sizer.

    Position = Account * Risk Percentage / Stop Loss %
    """

    def __init__(self, risk_pct: float = 0.02):
        """
        Args:
            risk_pct: Risk percentage per trade (default 2%)
        """
        self.risk_pct = risk_pct

    def calculate(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> float:
        """
        Calculate position size based on fixed fractional risk.

        Args:
            account_balance: Current account balance
            entry_price: Entry price of the position
            stop_loss_price: Stop loss price (exit price)

        Returns:
            Number of units to purchase
        """
        if entry_price <= 0 or stop_loss_price <= 0 or entry_price == stop_loss_price:
            return 0.0

        risk_amount = account_balance * self.risk_pct
        risk_per_unit = abs(entry_price - stop_loss_price)

        if risk_per_unit == 0:
            return 0.0

        position_size = risk_amount / risk_per_unit
        return position_size


class ATRSizer:
    """ATR-based position sizer.

    Position = Account * Risk% / (ATR * Multiplier)
    Uses Average True Range for stop distance.
    """

    def __init__(self, risk_pct: float = 0.02, atr_multiplier: float = 2.0):
        """
        Args:
            risk_pct: Risk percentage per trade
            atr_multiplier: ATR multiplier for stop distance
        """
        self.risk_pct = risk_pct
        self.atr_multiplier = atr_multiplier

    def calculate(
        self,
        account_balance: float,
        atr: float,
        entry_price: float,
    ) -> float:
        """
        Calculate position size using ATR-based stops.

        Args:
            account_balance: Current account balance
            atr: Average True Range value
            entry_price: Entry price

        Returns:
            Number of units to purchase
        """
        if atr <= 0 or entry_price <= 0:
            return 0.0

        risk_amount = account_balance * self.risk_pct
        stop_distance = atr * self.atr_multiplier

        position_size = risk_amount / stop_distance
        return max(0.0, position_size)


class VolatilitySizer:
    """Volatility-adjusted position sizer.

    Position = Base Position * (Target Vol / Current Vol)
    Adjusts size inversely to volatility for consistent risk.
    """

    def __init__(
        self,
        base_position: float,
        target_vol: float = 0.15,
        max_vol: float = 0.50,
    ):
        """
        Args:
            base_position: Base position size (e.g., full Kelly size)
            target_vol: Target annualized volatility (default 15%)
            max_vol: Maximum volatility before position goes to zero
        """
        self.base_position = base_position
        self.target_vol = target_vol
        self.max_vol = max_vol

    def calculate(
        self,
        current_vol: float,
        regime_multiplier: float = 1.0,
    ) -> float:
        """
        Calculate volatility-adjusted position size.

        Args:
            current_vol: Current annualized volatility (e.g., 0.20 = 20%)
            regime_multiplier: Additional multiplier from regime detection

        Returns:
            Adjusted position size
        """
        if current_vol <= 0:
            return self.base_position

        # Target vol adjustment
        vol_ratio = self.target_vol / current_vol
        vol_ratio = min(vol_ratio, 2.0)  # Cap at 2x

        adjusted = self.base_position * vol_ratio * regime_multiplier

        # If volatility exceeds max, reduce to zero
        if current_vol > self.max_vol:
            adjusted *= (self.max_vol / current_vol)

        return min(adjusted, self.base_position)


class FixedRatioSizer:
    """Fixed ratio position sizer.

    Increases position size as account grows, decreases on losses.
    Based on Ryan Jones's fixed ratio money management.
    """

    def __init__(self, delta: float = 0.5):
        """
        Args:
            delta: Risk aggressiveness (0.0-1.0). Higher = faster growth.
        """
        self.delta = delta
        self.peak_balance = 0.0

    def calculate(
        self,
        account_balance: float,
        unit_value: float = 1000.0,
    ) -> float:
        """
        Calculate position using fixed ratio method.

        Args:
            account_balance: Current account balance
            unit_value: Value of one trading unit

        Returns:
            Number of units to trade
        """
        if account_balance <= 0 or unit_value <= 0:
            return 0.0

        # Update peak if new high
        if account_balance > self.peak_balance:
            self.peak_balance = account_balance

        # Calculate risk capital
        risk_cap = account_balance - self.peak_balance

        # Calculate units based on delta
        if self.peak_balance == 0:
            return 1.0

        # Units = 1 + sqrt(2 * delta * risk_cap / unit_value)
        units = 1 + math.sqrt(2 * self.delta * risk_cap / unit_value)

        return max(1.0, units)


class PositionSizerFactory:
    """Factory for creating position sizers."""

    _sizers = {
        PositionSizingMethod.KELLY: KellySizer,
        PositionSizingMethod.FRACTIONAL: FractionalSizer,
        PositionSizingMethod.ATR_BASED: ATRSizer,
        PositionSizingMethod.VOLATILITY_ADJUSTED: VolatilitySizer,
        PositionSizingMethod.FIXED_RATIO: FixedRatioSizer,
    }

    @classmethod
    def create(
        cls,
        method: PositionSizingMethod,
        **kwargs,
    ) -> Any:
        """Create a position sizer by method type."""
        sizer_class = cls._sizers.get(method)
        if sizer_class is None:
            raise ValueError(f"Unknown position sizing method: {method}")
        return sizer_class(**kwargs)

    @classmethod
    def register(cls, method: PositionSizingMethod, sizer_class: type) -> None:
        """Register a custom position sizer."""
        cls._sizers[method] = sizer_class


# Convenience functions

def kelly_size(
    account_balance: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fractional: float = 0.5,
) -> float:
    """Calculate Kelly-based position size."""
    sizer = KellySizer(fractional=fractional)
    return sizer.calculate(account_balance, win_rate, avg_win, avg_loss)


def fractional_size(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float = 0.02,
) -> float:
    """Calculate fixed fractional position size."""
    sizer = FractionalSizer(risk_pct=risk_pct)
    return sizer.calculate(account_balance, entry_price, stop_loss_price)


def atr_size(
    account_balance: float,
    atr: float,
    entry_price: float,
    risk_pct: float = 0.02,
    atr_multiplier: float = 2.0,
) -> float:
    """Calculate ATR-based position size."""
    sizer = ATRSizer(risk_pct=risk_pct, atr_multiplier=atr_multiplier)
    return sizer.calculate(account_balance, atr, entry_price)


__all__ = [
    "PositionSizingMethod",
    "KellySizer",
    "FractionalSizer",
    "ATRSizer",
    "VolatilitySizer",
    "FixedRatioSizer",
    "PositionSizerFactory",
    "kelly_size",
    "fractional_size",
    "atr_size",
]
