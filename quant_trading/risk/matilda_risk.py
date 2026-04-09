"""Matilda Risk Quantification Library — Comprehensive risk metrics.

Adapted from Quantropy/matilda/quantitative_analysis/risk_quantification.py.
Unique features NOT in finclaw: LPM/HPM, Kappa ratios, Modigliani ratio,
Roys safety first, rolling Sharpe, enhanced drawdown analysis.

All functions work with numpy arrays or pandas Series.
"""

import math
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(returns: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Convert input to numpy array, handling pandas Series index."""
    if isinstance(returns, pd.Series):
        return returns.values
    if isinstance(returns, list):
        return np.array(returns)
    return np.asarray(returns)


def _mean(returns: Union[np.ndarray, pd.Series]) -> float:
    if isinstance(returns, pd.Series):
        return float(returns.mean())
    return float(np.mean(returns))


def _std(returns: Union[np.ndarray, pd.Series]) -> float:
    if isinstance(returns, pd.Series):
        return float(returns.std(ddof=1))
    return float(np.std(returns, ddof=1))


# ---------------------------------------------------------------------------
# Lower / Upper Partial Moments
# ---------------------------------------------------------------------------

def lower_partial_moments(
    returns: Union[np.ndarray, pd.Series, list],
    target: float = 0.0,
    threshold: Optional[float] = None,
) -> float:
    """Lower Partial Moment (LPM) of order n.

    LPM measures downside risk by summing powered deviations below a threshold.

    Args:
        returns: Return series.
        target: Threshold (minimum acceptable return). Defaults to 0.
        threshold: Alias for target (for compatibility).

    Returns:
        LPM value (lower is riskier).
    """
    if threshold is not None:
        target = threshold
    arr = _to_array(returns)
    diff = target - arr
    diff = np.clip(diff, a_min=0, a_max=None)
    return float(np.sum(diff ** 1) / len(arr))


def upper_partial_moments(
    returns: Union[np.ndarray, pd.Series, list],
    target: float = 0.0,
    threshold: Optional[float] = None,
) -> float:
    """Upper Partial Moment (HPM) of order n.

    HPM measures upside potential by summing powered deviations above a threshold.

    Args:
        returns: Return series.
        target: Threshold. Defaults to 0.
        threshold: Alias for target (for compatibility).

    Returns:
        HPM value.
    """
    if threshold is not None:
        target = threshold
    arr = _to_array(returns)
    diff = arr - target
    diff = np.clip(diff, a_min=0, a_max=None)
    return float(np.sum(diff ** 1) / len(arr))


def lpm_n(
    returns: Union[np.ndarray, pd.Series, list],
    target: float,
    order: int,
) -> float:
    """Lower Partial Moment of arbitrary order.

    Args:
        returns: Return series.
        target: Minimum acceptable return.
        order: Moment order (1 = first LPM, 2 = variance-style LPM, etc.).

    Returns:
        LPM value.
    """
    arr = _to_array(returns)
    diff = target - arr
    diff = np.clip(diff, a_min=0, a_max=None)
    return float(np.sum(diff ** order) / len(arr))


def hpm_n(
    returns: Union[np.ndarray, pd.Series, list],
    target: float,
    order: int,
) -> float:
    """Upper Partial Moment of arbitrary order.

    Args:
        returns: Return series.
        target: Threshold return.
        order: Moment order.

    Returns:
        HPM value.
    """
    arr = _to_array(returns)
    diff = arr - target
    diff = np.clip(diff, a_min=0, a_max=None)
    return float(np.sum(diff ** order) / len(arr))


# ---------------------------------------------------------------------------
# Value at Risk — Historical Simulation
# ---------------------------------------------------------------------------

def var_historical(
    returns: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> float:
    """Historical simulation VaR.

    Uses empirical quantiles of the return distribution.

    Args:
        returns: Return series.
        confidence: Confidence level (e.g. 0.95 = 95th percentile loss).

    Returns:
        VaR as a positive loss number.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    sorted_returns = np.sort(arr)
    index = int((1 - confidence) * len(sorted_returns))
    index = max(0, min(index, len(sorted_returns) - 1))
    return abs(float(sorted_returns[index]))


# ---------------------------------------------------------------------------
# Value at Risk — Parametric (Variance-Covariance)
# ---------------------------------------------------------------------------

def var_variance_covariance(
    returns: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
    period: int = 252,
) -> float:
    """Parametric VaR assuming normal distribution.

    Uses the inverse normal CDF (z-score).

    Args:
        returns: Return series.
        confidence: Confidence level.
        period: Annualisation factor (252 for daily).

    Returns:
        Annualised VaR as a positive loss number.
    """
    from scipy.stats import norm
    arr = _to_array(returns)
    mu = _mean(arr) * period
    sigma = _std(arr) * np.sqrt(period)
    return abs(float(norm.ppf(1 - confidence, loc=mu, scale=sigma)))


# ---------------------------------------------------------------------------
# CVaR / Expected Shortfall
# ---------------------------------------------------------------------------

def cvar_expected_shortfall(
    returns: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> float:
    """CVaR / Expected Shortfall — average loss beyond VaR threshold.

    Args:
        returns: Return series.
        confidence: Confidence level.

    Returns:
        CVaR as a positive loss number.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    var = var_historical(arr, confidence)
    tail = arr[arr <= -var]
    if len(tail) == 0:
        return var
    return abs(float(np.mean(tail)))


# ---------------------------------------------------------------------------
# Omega Ratio
# ---------------------------------------------------------------------------

def omega_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    threshold: float = 0.0,
) -> float:
    """Omega ratio — probability-weighted gain/loss ratio above threshold.

    Uses LPM as denominator (matches matilda implementation).

    Args:
        returns: Return series.
        threshold: Minimum acceptable return.

    Returns:
        Omega ratio (higher is better).
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    denom = lower_partial_moments(arr, target=threshold, threshold=threshold)
    if denom == 0:
        return float('inf') if _mean(arr) > threshold else 1.0
    excess = _mean(arr) - threshold
    return float(excess / denom)


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------

def sortino_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    target: float = 0.0,
    threshold: Optional[float] = None,
    rf: float = 0.0,
    period: int = 252,
) -> float:
    """Sortino ratio — excess return over downside deviation.

    Uses LPM with order=2 as downside denominator.

    Args:
        returns: Return series.
        target: Minimum acceptable return.
        threshold: Alias for target.
        rf: Risk-free rate (annualised).
        period: Annualisation factor.

    Returns:
        Annualised Sortino ratio (higher is better).
    """
    if threshold is not None:
        target = threshold
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    excess_return = _mean(arr) - (rf / period)
    downside = lpm_n(arr, target=target, order=2)
    if downside == 0:
        return float('inf') if excess_return > 0 else 0.0
    return float((excess_return * period) / (np.sqrt(downside) * np.sqrt(period)))


# ---------------------------------------------------------------------------
# Kappa Ratios
# ---------------------------------------------------------------------------

def kappa_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    target: float = 0.0,
    n: int = 2,
    rf: float = 0.0,
    period: int = 252,
) -> float:
    """Kappa ratio of order n.

    Generalised downside risk-adjusted performance measure.
    Matches Sortino when n=2, VaR-kappa when n=1, CVaR-kappa when tail.

    Args:
        returns: Return series.
        target: Minimum acceptable return.
        n: Order (1=VaR-kappa, 2=Sortino-kappa, 3=third-order kappa).
        rf: Risk-free rate (annualised).
        period: Annualisation factor.

    Returns:
        Annualised Kappa ratio.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    excess = _mean(arr) - (rf / period)
    lpm_val = lpm_n(arr, target=target, order=n)
    if lpm_val == 0:
        return float('inf') if excess > 0 else 0.0
    return float((excess * period) / (math.pow(lpm_val, float(1 / n)) * np.sqrt(period)))


def kappa_var_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    target: float = 0.0,
    confidence: float = 0.95,
    rf: float = 0.0,
    period: int = 252,
) -> float:
    """VaR-based Kappa ratio (order 1).

    Args:
        returns: Return series.
        target: Minimum acceptable return.
        confidence: VaR confidence level.
        rf: Risk-free rate.
        period: Annualisation factor.

    Returns:
        VaR-kappa ratio.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    excess = _mean(arr) - (rf / period)
    var = var_historical(arr, confidence)
    if var == 0:
        return float('inf') if excess > 0 else 0.0
    return float((excess * period) / var)


def kappa_cvar_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    target: float = 0.0,
    confidence: float = 0.95,
    rf: float = 0.0,
    period: int = 252,
) -> float:
    """CVaR-based Kappa ratio (first-order lower partial moment).

    Args:
        returns: Return series.
        target: Minimum acceptable return.
        confidence: CVaR confidence level.
        rf: Risk-free rate.
        period: Annualisation factor.

    Returns:
        CVaR-kappa ratio.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    excess = _mean(arr) - (rf / period)
    cvar = cvar_expected_shortfall(arr, confidence)
    if cvar == 0:
        return float('inf') if excess > 0 else 0.0
    return float((excess * period) / cvar)


def kappa_target_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    target: float = 0.0,
    rf: float = 0.0,
    period: int = 252,
) -> float:
    """Target-based Kappa ratio (uses LPM).

    Args:
        returns: Return series.
        target: Minimum acceptable return.
        rf: Risk-free rate.
        period: Annualisation factor.

    Returns:
        Target-kappa ratio.
    """
    return kappa_ratio(returns, target=target, n=2, rf=rf, period=period)


# ---------------------------------------------------------------------------
# Gain Loss Ratio & Upside Potential Ratio
# ---------------------------------------------------------------------------

def gain_loss_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    target: float = 0.0,
) -> float:
    """Gain-to-Loss ratio — HPM / LPM (above/below threshold)."""
    arr = _to_array(returns)
    lpm_val = lower_partial_moments(arr, target=target)
    if lpm_val == 0:
        return float('inf')
    return upper_partial_moments(arr, target=target) / lpm_val


def upside_potential_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    target: float = 0.0,
) -> float:
    """Upside potential ratio — HPM / sqrt(LPM)."""
    arr = _to_array(returns)
    lpm2 = lpm_n(arr, target=target, order=2)
    if lpm2 == 0:
        return float('inf') if _mean(arr) > target else 0.0
    return upper_partial_moments(arr, target=target) / math.sqrt(lpm2)


# ---------------------------------------------------------------------------
# Modigliani Ratio (Modigliani Risk-Adjusted Performance)
# ---------------------------------------------------------------------------

def modigliani_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    benchmark_returns: Union[np.ndarray, pd.Series, list],
    rf: float = 0.0,
    period: int = 252,
) -> float:
    """Modigliani risk-adjusted performance measure.

    Combines Sharpe and information ratios:
    M2 = Sharpe * sigma_benchmark + rf

    Args:
        returns: Portfolio return series.
        benchmark_returns: Benchmark return series.
        rf: Risk-free rate (annualised).
        period: Annualisation factor.

    Returns:
        Modigliani ratio (higher is better).
    """
    arr = _to_array(returns)
    bm_arr = _to_array(benchmark_returns)
    if len(arr) == 0 or len(bm_arr) == 0:
        return 0.0
    bm_vol = _std(bm_arr) * np.sqrt(period)
    excess_return = _mean(arr) - (rf / period)
    port_vol = _std(arr) * np.sqrt(period)
    if port_vol == 0:
        return 0.0
    sharpe = (excess_return * period) / port_vol
    return float(sharpe * bm_vol + (rf / period) * period)


# ---------------------------------------------------------------------------
# Information Ratio
# ---------------------------------------------------------------------------

def information_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    benchmark_returns: Union[np.ndarray, pd.Series, list],
    period: int = 252,
) -> float:
    """Information ratio — excess return / tracking error.

    Args:
        returns: Portfolio return series.
        benchmark_returns: Benchmark return series.
        period: Annualisation factor.

    Returns:
        Information ratio.
    """
    arr = _to_array(returns)
    bm_arr = _to_array(benchmark_returns)
    if len(arr) == 0 or len(bm_arr) == 0 or len(arr) != len(bm_arr):
        return 0.0
    diff = arr - bm_arr
    mean_diff = _mean(diff)
    std_diff = _std(diff)
    if std_diff == 0:
        return 0.0
    return float((mean_diff * period) / (std_diff * np.sqrt(period)))


# ---------------------------------------------------------------------------
# Treynor Ratio
# ---------------------------------------------------------------------------

def treynor_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    benchmark_returns: Union[np.ndarray, pd.Series, list],
    rf: float = 0.0,
    period: int = 252,
) -> float:
    """Treynor ratio — excess return / portfolio beta.

    Args:
        returns: Portfolio return series.
        benchmark_returns: Benchmark return series.
        rf: Risk-free rate (annualised).
        period: Annualisation factor.

    Returns:
        Treynor ratio (higher is better).
    """
    arr = _to_array(returns)
    bm_arr = _to_array(benchmark_returns)
    if len(arr) == 0 or len(bm_arr) == 0 or len(arr) != len(bm_arr):
        return 0.0

    # Compute beta: cov(portfolio, benchmark) / var(benchmark)
    mean_r = _mean(arr)
    mean_b = _mean(bm_arr)
    n = len(arr)
    cov = sum((r - mean_r) * (b - mean_b) for r, b in zip(arr, bm_arr)) / n
    var_b = sum((b - mean_b) ** 2 for b in bm_arr) / n
    if var_b == 0:
        return 0.0
    beta = cov / var_b
    if beta == 0:
        return 0.0
    excess = _mean(arr) - (rf / period)
    return float((excess * period) / beta)


# ---------------------------------------------------------------------------
# Jensens Alpha
# ---------------------------------------------------------------------------

def jensens_alpha(
    returns: Union[np.ndarray, pd.Series, list],
    benchmark_returns: Union[np.ndarray, pd.Series, list],
    rf: float = 0.0,
    beta: Optional[float] = None,
    period: int = 252,
) -> float:
    """Jensen's alpha — abnormal return above CAPM prediction.

    Args:
        returns: Portfolio return series.
        benchmark_returns: Benchmark return series.
        rf: Risk-free rate (annualised).
        beta: Portfolio beta (computed from data if None).
        period: Annualisation factor.

    Returns:
        Alpha (positive = outperformance).
    """
    arr = _to_array(returns)
    bm_arr = _to_array(benchmark_returns)
    if len(arr) == 0 or len(bm_arr) == 0:
        return 0.0

    if beta is None:
        mean_r = _mean(arr)
        mean_b = _mean(bm_arr)
        n = len(arr)
        cov = sum((r - mean_r) * (b - mean_b) for r, b in zip(arr, bm_arr)) / n
        var_b = sum((b - mean_b) ** 2 for b in bm_arr) / n
        if var_b == 0:
            return 0.0
        beta = cov / var_b

    # CAPM expected return: rf + beta * (market_rf - rf)
    capm_return = (rf / period) + beta * (_mean(bm_arr) - rf / period)
    ann_return = _mean(arr) * period
    return float(ann_return - capm_return)


# ---------------------------------------------------------------------------
# Roys Safety First Criterion
# ---------------------------------------------------------------------------

def roys_safety_first(
    returns: Union[np.ndarray, pd.Series, list],
    rf: float = 0.02,
    period: int = 252,
) -> float:
    """Roys Safety First criterion — probability of shortfall.

    SF = (E[return] - MAR) / std_dev

    Args:
        returns: Return series.
        rf: Minimum acceptable return threshold (annualised).
        period: Annualisation factor.

    Returns:
        Safety-first ratio (higher = safer).
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    mar = rf / period
    excess = _mean(arr) - mar
    std = _std(arr)
    if std == 0:
        return float('inf') if excess > 0 else 0.0
    return float((excess * period) / (std * np.sqrt(period)))


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------

def rolling_sharpe(
    returns: Union[np.ndarray, pd.Series, list],
    window: int = 252,
    rf: float = 0.0,
    period: int = 252,
) -> pd.Series:
    """Rolling Sharpe ratio over a lookback window.

    Args:
        returns: Return series (pandas Series preferred).
        window: Lookback window in periods.
        rf: Annualised risk-free rate.
        period: Annualisation factor.

    Returns:
        pandas Series of rolling Sharpe ratios.
    """
    if isinstance(returns, pd.Series):
        arr = returns.copy()
    else:
        arr = pd.Series(_to_array(returns))

    rf_period = rf / period
    excess = arr - rf_period
    rolling_mean = excess.rolling(window=window, min_periods=window // 2).mean()
    rolling_std = excess.rolling(window=window, min_periods=window // 2).std(ddof=1)
    result = (rolling_mean / rolling_std) * np.sqrt(period)
    return result


# ---------------------------------------------------------------------------
# Drawdown Analysis with Duration Tracking
# ---------------------------------------------------------------------------

def drawdown_analysis(
    returns: Union[np.ndarray, pd.Series, list],
    trailing_period: Optional[int] = None,
) -> dict:
    """Enhanced drawdown analysis with duration tracking.

    Args:
        returns: Return series.
        trailing_period: Window for rolling max (None = full history).

    Returns:
        Dict with:
            - daily_drawdown: drawdown series
            - max_daily_drawdown: rolling max drawdown series
            - max_drawdown: worst drawdown (negative)
            - max_drawdown_duration: longest drawdown period in bars
            - average_daily_drawdown: mean drawdown
            - current_drawdown: latest drawdown
            - drawdown_duration: current drawdown duration
    """
    if isinstance(returns, pd.Series):
        arr = returns.copy()
    else:
        arr = pd.Series(_to_array(returns))

    if trailing_period is not None:
        rolling_max = arr.rolling(trailing_period, min_periods=1).max()
    else:
        rolling_max = arr.cummax()

    daily_drawdown = arr / rolling_max - 1.0

    # Max drawdown in the window
    max_daily_drawdown = daily_drawdown.rolling(
        trailing_period or len(arr), min_periods=1
    ).min()

    # Duration tracking
    peak = arr.cummax()
    is_drawdown = daily_drawdown < 0

    # Current drawdown duration
    current_dd = float(daily_drawdown.iloc[-1]) if len(daily_drawdown) > 0 else 0.0

    # Longest drawdown duration
    dd_series = daily_drawdown.values
    max_dd = float(np.nanmin(dd_series)) if len(dd_series) > 0 else 0.0

    # Compute duration
    durations = []
    current_duration = 0
    in_dd = False
    dd_start = 0
    for i, dd in enumerate(dd_series):
        if dd < 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif dd >= 0 and in_dd:
            in_dd = False
            durations.append(i - dd_start)
    if in_dd:
        durations.append(len(dd_series) - dd_start)

    max_duration = max(durations) if durations else 0
    current_duration = len(dd_series) - dd_start if in_dd else 0

    return {
        'daily_drawdown': daily_drawdown,
        'max_daily_drawdown': max_daily_drawdown,
        'max_drawdown': max_dd,
        'max_drawdown_duration': max_duration,
        'average_daily_drawdown': float(np.nanmean(dd_series)),
        'current_drawdown': current_dd,
        'drawdown_duration': current_duration,
    }


# ---------------------------------------------------------------------------
# Conditional Sharpe Ratio
# ---------------------------------------------------------------------------

def conditional_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, list],
    rf: float = 0.0,
    confidence: float = 0.95,
    period: int = 252,
) -> float:
    """Conditional Sharpe ratio — excess return / CVaR.

    Args:
        returns: Return series.
        rf: Risk-free rate (annualised).
        confidence: CVaR confidence level.
        period: Annualisation factor.

    Returns:
        Conditional Sharpe ratio.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    excess = _mean(arr) - (rf / period)
    cvar = cvar_expected_shortfall(arr, confidence)
    if cvar == 0:
        return float('inf') if excess > 0 else 0.0
    return float((excess * period) / cvar)


# ---------------------------------------------------------------------------
# Excess Return VaR (ERVaR)
# ---------------------------------------------------------------------------

def excess_return_var(
    returns: Union[np.ndarray, pd.Series, list],
    rf: float = 0.0,
    confidence: float = 0.95,
    period: int = 252,
) -> float:
    """Excess return VaR — excess return / historical VaR.

    Args:
        returns: Return series.
        rf: Risk-free rate (annualised).
        confidence: VaR confidence level.
        period: Annualisation factor.

    Returns:
        ERVaR ratio.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return 0.0
    excess = _mean(arr) - (rf / period)
    var = var_historical(arr, confidence)
    if var == 0:
        return float('inf') if excess > 0 else 0.0
    return float((excess * period) / var)
