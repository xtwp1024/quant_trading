"""Volatility Forecasting and Regime Detection.

Adopted from ATLAS (D:/Hive/Data/trading_repos/ATLAS/) with adaptations:
- Pure numpy/pandas implementation (no external dependencies like numba/xgboost)
- Inlined helper functions
- Relative imports within quant_trading.risk

Features:
- Multiple volatility estimators: Close-to-close, Parkinson, Garman-Klass,
  Rogers-Satchell, Yang-Zhang
- Market regime detection (LOW/MEDIUM/HIGH volatility)
- Risk-adjusted position sizing based on volatility regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Optional, Union

# ------------------------------------------------------------------
# Volatility Regime Enum
# ------------------------------------------------------------------


class VolatilityRegime(Enum):
    """Market volatility regime classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ------------------------------------------------------------------
# Helper: safe rolling window helper
# ------------------------------------------------------------------

def _rolling_apply(
    series: pd.Series,
    window: int,
    func: callable,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Apply a function over a rolling window with NaN handling."""
    if min_periods is None:
        min_periods = window
    return series.rolling(window=window, min_periods=min_periods).apply(func, raw=True)


# ------------------------------------------------------------------
# Volatility Estimators
# ------------------------------------------------------------------

def close_to_close_vol(
    close: Union[pd.Series, np.ndarray],
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Standard close-to-close volatility estimator.

    Formula: std(log(close_t / close_{t-1})) * sqrt(252) if annualized

    Args:
        close: Close price series
        window: Rolling window in periods (default 21 trading days)
        annualize: If True, annualize the volatility (default True)

    Returns:
        Rolling close-to-close volatility series
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    log_returns = np.log(close / close.shift(1)).fillna(0)
    vol = log_returns.rolling(window=window, min_periods=window).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol


def parkinson_vol(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Parkinson volatility estimator using high/low prices.

    Formula: sqrt(H * L / (4 * ln(2)))
             where H = high, L = low over the window
             Annualized: result * sqrt(252 / window) * sqrt(252)

    The Parkinson estimator is more efficient than close-to-close as it
    uses the full high-low range to capture intra-day volatility.

    Args:
        high: High price series
        low: Low price series
        window: Rolling window in periods
        annualize: If True, annualize the volatility (default True)

    Returns:
        Rolling Parkinson volatility series
    """
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)

    # Intra-day log high-low ratio
    hl_ratio = np.log(high / low)

    # Rolling mean of squared log HL ratio
    hl_var = (hl_ratio ** 2).rolling(window=window, min_periods=window).mean()

    # Parkinson volatility: sqrt(H*L / (4*ln(2))) = sqrt(ln(H/L)^2 / (4*ln(2)))
    # Variance of log(H/L) over window is approximated by mean(log(H/L)^2)
    # Parkinson = sqrt(var / (4 * ln(2)))
    parkinson = np.sqrt(hl_var / (4 * np.log(2)))

    if annualize:
        # Scale from per-period to annualised
        # Using the standard approach: parkinson * sqrt(252 / window) * sqrt(252)
        # Simplifies to: parkinson * sqrt(252 * 252 / window) - not quite right
        # Better: scale by sqrt(252 / window) to get per-day vol, then * sqrt(252)
        parkinson = parkinson * np.sqrt(252 / window) * np.sqrt(252)

    return parkinson


def garman_klass_vol(
    open_prices: Union[pd.Series, np.ndarray],
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Garman-Klass volatility estimator using OHLC prices.

    Formula (per period):
        sigma_GK^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2

    The Garman-Klass estimator is up to 5x more efficient than close-to-close
    and captures intra-day price movement using the full OHLC data.

    Args:
        open_prices: Open price series
        high: High price series
        low: Low price series
        close: Close price series
        window: Rolling window in periods
        annualize: If True, annualize the volatility (default True)

    Returns:
        Rolling Garman-Klass volatility series
    """
    if isinstance(open_prices, np.ndarray):
        open_prices = pd.Series(open_prices)
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    # Intra-day log ratios
    log_hl = np.log(high / low)
    log_co = np.log(close / open_prices)

    # Garman-Klass variance component
    gk_var = (
        0.5 * (log_hl ** 2)
        - (2 * np.log(2) - 1) * (log_co ** 2)
    )

    # Rolling mean of the Garman-Klass component
    gk_mean = gk_var.rolling(window=window, min_periods=window).mean()

    garman_klass = np.sqrt(gk_mean)

    if annualize:
        garman_klass = garman_klass * np.sqrt(252 / window) * np.sqrt(252)

    return garman_klass


def rogers_satchell_vol(
    open_prices: Union[pd.Series, np.ndarray],
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Rogers-Satchell volatility estimator using OHLC prices.

    Formula (per period):
        sigma_RS^2 = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)

    This estimator captures the drift-adjusted intra-day range and
    is particularly useful for estimating volatility in trending markets.

    Args:
        open_prices: Open price series
        high: High price series
        low: Low price series
        close: Close price series
        window: Rolling window in periods
        annualize: If True, annualize the volatility (default True)

    Returns:
        Rolling Rogers-Satchell volatility series
    """
    if isinstance(open_prices, np.ndarray):
        open_prices = pd.Series(open_prices)
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    # Log ratios
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_prices)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_prices)

    # Rogers-Satchell variance
    rs_var = (log_hc * log_ho) + (log_lc * log_lo)

    # Rolling mean
    rs_mean = rs_var.rolling(window=window, min_periods=window).mean()

    rogers_satchell = np.sqrt(rs_mean)

    if annualize:
        rogers_satchell = rogers_satchell * np.sqrt(252 / window) * np.sqrt(252)

    return rogers_satchell


def yang_zhang_vol(
    open_prices: Union[pd.Series, np.ndarray],
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    """Yang-Zhang volatility estimator using OHLC prices.

    Formula:
        sigma^2 = N/(N-1) * (sigma_close^2 + k * sigma_open^2 + (1-k) * sigma_rs^2)

    where:
        sigma_close^2 = running variance of log(C/O)
        sigma_open^2  = running variance of log(O/ref), ref = close_prev
        sigma_rs^2    = Rogers-Satchell component
        k             = 0.64 / (0.64 + (N+1)/(N-1))

    The Yang-Zhang estimator combines overnight (close-to-open),
    open-to-close, and the Rogers-Satchell intra-day components,
    providing the most accurate volatility estimate among the OHLC-based
    estimators.

    Args:
        open_prices: Open price series
        high: High price series
        low: Low price series
        close: Close price series
        window: Rolling window in periods
        annualize: If True, annualize the volatility (default True)

    Returns:
        Rolling Yang-Zhang volatility series
    """
    if isinstance(open_prices, np.ndarray):
        open_prices = pd.Series(open_prices)
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    # Close-to-close log returns
    log_co = np.log(close / open_prices).fillna(0)

    # Overnight log returns: open / previous close
    log_oc_prev = np.log(open_prices / close.shift(1)).fillna(0)

    # Rogers-Satchell component
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_prices)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_prices)
    rs_var = (log_hc * log_ho) + (log_lc * log_lo)

    # Individual variance estimates
    var_close = log_co.rolling(window=window, min_periods=window).var()
    var_open = log_oc_prev.rolling(window=window, min_periods=window).var()
    var_rs = rs_var.rolling(window=window, min_periods=window).mean()

    # Yang-Zhang weighting factor
    # k = 0.64 / (0.64 + (N+1)/(N-1))
    # Simplified using the standard 0.64 coefficient
    k = 0.64 / (0.64 + (window + 1) / (window - 1))

    # Combined variance
    yz_var = (var_close + k * var_open + (1 - k) * var_rs)

    yang_zhang = np.sqrt(yz_var)

    if annualize:
        yang_zhang = yang_zhang * np.sqrt(252 / window) * np.sqrt(252)

    return yang_zhang


# ------------------------------------------------------------------
# Composite Volatility (average of multiple estimators)
# ------------------------------------------------------------------

def composite_volatility(
    open_prices: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    estimators: Tuple[str, ...] = ("garman_klass", "yang_zhang", "rogers_satchell", "parkinson"),
    annualize: bool = True,
) -> pd.Series:
    """Composite volatility from multiple estimators.

    Args:
        open_prices: Open price series
        high: High price series
        low: Low price series
        close: Close price series
        window: Rolling window in periods
        estimators: Tuple of estimator names to average
        annualize: If True, annualize the volatility

    Returns:
        Average volatility from specified estimators
    """
    available = {
        "close_to_close": close_to_close_vol(close, window, annualize),
        "parkinson": parkinson_vol(high, low, window, annualize),
        "garman_klass": garman_klass_vol(open_prices, high, low, close, window, annualize),
        "rogers_satchell": rogers_satchell_vol(open_prices, high, low, close, window, annualize),
        "yang_zhang": yang_zhang_vol(open_prices, high, low, close, window, annualize),
    }

    selected = [available[e] for e in estimators if e in available]

    if not selected:
        raise ValueError(f"No valid estimators provided. Available: {list(available.keys())}")

    return pd.concat(selected, axis=1).mean(axis=1)


# ------------------------------------------------------------------
# Regime Detection
# ------------------------------------------------------------------

def detect_regime(
    volatility: Union[pd.Series, float, np.ndarray],
    thresholds: Optional[Tuple[float, float]] = None,
) -> Union[VolatilityRegime, pd.Series]:
    """Detect market volatility regime from a volatility series or value.

    Args:
        volatility: Volatility series or single volatility value.
                    Expected to be annualized (e.g., 0.15 = 15% annualized vol).
        thresholds: Tuple of (low_threshold, high_threshold) volatility values.
                    If None, defaults are used:
                    - LOW: vol < 0.10 (10%)
                    - MEDIUM: 0.10 <= vol < 0.25
                    - HIGH: vol >= 0.25

    Returns:
        VolatilityRegime enum value or series of enum values
    """
    if thresholds is None:
        low_thresh = 0.10
        high_thresh = 0.25
    else:
        low_thresh, high_thresh = thresholds

    if isinstance(volatility, (float, np.floating, np.integer)):
        vol = float(volatility)
        if vol < low_thresh:
            return VolatilityRegime.LOW
        elif vol < high_thresh:
            return VolatilityRegime.MEDIUM
        else:
            return VolatilityRegime.HIGH

    # Series case
    def _classify(v):
        if pd.isna(v):
            return VolatilityRegime.MEDIUM  # default for NaN
        if v < low_thresh:
            return VolatilityRegime.LOW
        elif v < high_thresh:
            return VolatilityRegime.MEDIUM
        else:
            return VolatilityRegime.HIGH

    return volatility.apply(_classify)


def regime_multipliers(regime: VolatilityRegime) -> float:
    """Return position size multiplier based on volatility regime.

    Higher volatility regimes warrant smaller positions to control risk.
    These multipliers are applied to the base position size.

    Args:
        regime: VolatilityRegime enum value

    Returns:
        Position size multiplier (0 < multiplier <= 1)
    """
    multipliers = {
        VolatilityRegime.LOW: 1.0,
        VolatilityRegime.MEDIUM: 0.75,
        VolatilityRegime.HIGH: 0.5,
    }
    return multipliers.get(regime, 0.75)


# ------------------------------------------------------------------
# Volatility-Adjusted Position Sizing
# ------------------------------------------------------------------

def volatility_adjusted_position(
    base_size: float,
    current_vol: float,
    target_vol: float = 0.15,
    regime: Optional[VolatilityRegime] = None,
) -> float:
    """Calculate volatility-adjusted position size.

    Uses the volatility targeting formula:
        adjusted_size = base_size * (target_vol / current_vol)

    Optionally applies a regime multiplier to further reduce risk
    during high-volatility regimes.

    Args:
        base_size: Base position size (e.g., 10000 units)
        current_vol: Current annualized volatility (e.g., 0.20 = 20%)
        target_vol: Target annualized volatility (default 0.15 = 15%)
        regime: Optional volatility regime to apply additional scaling

    Returns:
        Adjusted position size (clipped to base_size as maximum)
    """
    if current_vol <= 0:
        return base_size

    # Volatility targeting adjustment
    vol_ratio = target_vol / current_vol

    # Cap the adjustment at 2x to avoid excessive leverage
    vol_ratio = min(vol_ratio, 2.0)

    adjusted = base_size * vol_ratio

    # Apply regime-based reduction
    if regime is not None:
        regime_mult = regime_multipliers(regime)
        adjusted = adjusted * regime_mult

    # Never exceed base_size to avoid accidental leverage
    return min(adjusted, base_size)


def calculate_volatility_forecast(
    close: pd.Series,
    open_prices: Optional[pd.Series] = None,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    window: int = 21,
    method: str = "yang_zhang",
    annualize: bool = True,
) -> pd.Series:
    """Calculate a volatility forecast/estimate from price data.

    Automatically selects the best available estimator based on
    which OHLC columns are provided.

    Priority:
        1. yang_zhang (requires O, H, L, C)
        2. garman_klass (requires O, H, L, C)
        3. rogers_satchell (requires O, H, L, C)
        4. parkinson (requires H, L)
        5. close_to_close (requires C only)

    Args:
        close: Close price series (required)
        open_prices: Open price series (optional)
        high: High price series (optional)
        low: Low price series (optional)
        window: Rolling window in periods
        method: Preferred estimator name if all columns available
        annualize: If True, annualize the volatility

    Returns:
        Volatility forecast series
    """
    has_ohlc = all(x is not None for x in [open_prices, high, low])

    if method == "yang_zhang" and has_ohlc:
        return yang_zhang_vol(open_prices, high, low, close, window, annualize)
    elif method == "garman_klass" and has_ohlc:
        return garman_klass_vol(open_prices, high, low, close, window, annualize)
    elif method == "rogers_satchell" and has_ohlc:
        return rogers_satchell_vol(open_prices, high, low, close, window, annualize)
    elif high is not None and low is not None:
        return parkinson_vol(high, low, window, annualize)
    else:
        return close_to_close_vol(close, window, annualize)


# ------------------------------------------------------------------
# Exports
# ------------------------------------------------------------------

__all__ = [
    # Regime
    "VolatilityRegime",
    "detect_regime",
    "regime_multipliers",
    # Estimators
    "close_to_close_vol",
    "parkinson_vol",
    "garman_klass_vol",
    "rogers_satchell_vol",
    "yang_zhang_vol",
    "composite_volatility",
    "calculate_volatility_forecast",
    # Position sizing
    "volatility_adjusted_position",
]
