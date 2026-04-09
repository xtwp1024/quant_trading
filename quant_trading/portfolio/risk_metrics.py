"""
quant_trading.portfolio.risk_metrics — Portfolio-Level Risk Metrics.

Evaluates portfolio return series for risk: VaR, CVaR, CDaR, Omega,
downside/upside metrics, and portfolio-specific performance ratios.

Complements quant_trading/risk/risk_metrics.py (which focuses on
trade-level and aggregate portfolio metrics) with portfolio-return-series
focused calculations including CDaR and Omega ratio.

Key metrics provided:
- VaR (historical, parametric, Gaussian) at multiple confidence levels
- CVaR / Expected Shortfall at multiple confidence levels
- CDaR (Conditional Drawdown at Risk) — Ulcer-style tail drawdown
- Omega ratio (gain/loss threshold measure)
- Maximum drawdown, underwater series, drawdown duration
- Tail ratio, downside deviation, upside potential ratio
- Information ratio, Treynor ratio, Jensens alpha (when benchmark provided)
- Kapa ratios (LPM-based): kappa_1, kappa_2, kappa_3
- Sortino ratio (LPM-based, uses target return)
- Roy's safety-first ratio
- Skewness and kurtosis (return distribution shape)
- Rolling Sharpe and drawdown analysis

References
----------
- Sortino & van der Meer (1991) "Downside Risk"
- Keel & Heidorn (1997) "Deriving the Risk-Adjusted Performance Measures"
- Maillard, Roncalli & Teiletche (2010) "On the Properties of Equally-Weighted
  Risk Contributions Portfolios"
- Rockafellar & Uryasev (2000) "Optimization of Conditional VaR"
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

try:
    import numpy as np
    import pandas as pd
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None
    pd = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DrawdownInfo:
    """Detailed drawdown analysis for a portfolio equity curve."""
    max_drawdown: float           # worst peak-to-trough (negative fraction)
    max_drawdown_duration: int    # periods in the longest drawdown
    current_drawdown: float       # current drawdown fraction
    drawdown_series: List[float]  # full underwater curve
    peak_index: int               # index of the peak before max drawdown
    trough_index: int             # index of the trough of max drawdown


@dataclass
class PortfolioRiskReport:
    """Comprehensive risk report for a portfolio return series."""
    # Return
    total_return: float
    annualized_return: float
    # Volatility
    volatility: float
    downside_deviation: float
    # Risk metrics
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    cdar_95: float
    cdar_99: float
    max_drawdown: float
    max_drawdown_duration: int
    # Performance ratios
    sharpe_ratio: float
    sortino_ratio: float
    omega_ratio: float
    calmar_ratio: float
    tail_ratio: float
    # Distribution shape
    skewness: float
    kurtosis: float
    # Benchmarked ratios (when benchmark provided)
    information_ratio: Optional[float] = None
    treynor_ratio: Optional[float] = None
    jensens_alpha: Optional[float] = None
    # Kappa ratios
    kappa_1: Optional[float] = None
    kappa_2: Optional[float] = None
    kappa_3: Optional[float] = None
    # Safety first
    roys_safety_first: Optional[float] = None
    # Rolling
    rolling_sharpe_mean: Optional[float] = None
    rolling_sharpe_std: Optional[float] = None


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _percentile(sorted_xs: List[float], p: float) -> float:
    """Nth percentile from a pre-sorted list."""
    if not sorted_xs:
        return 0.0
    idx = max(int(len(sorted_xs) * p), 0)
    idx = min(idx, len(sorted_xs) - 1)
    return sorted_xs[idx]


# ---------------------------------------------------------------------------
# VaR / CVaR (portfolio return series)
# ---------------------------------------------------------------------------

def value_at_risk(
    returns: List[float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Value at Risk — worst expected loss at a given confidence level.

    Args:
        returns: List of portfolio period returns.
        confidence: Confidence level (e.g. 0.95 → 95%).
        method: "historical", "parametric" (Gaussian), or " Cornish-Fisher"
                (moment-expanded Gaussian).

    Returns:
        VaR as a negative fraction (e.g. -0.025 means 2.5% expected loss).
    """
    if len(returns) < 10:
        return 0.0

    if method == "historical":
        sorted_r = sorted(returns)
        return _percentile(sorted_r, 1 - confidence)

    # Parametric (Gaussian)
    mu = _mean(returns)
    s = _std(returns)
    z_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326, 0.975: 1.96, 0.99: 2.326}
    z = z_map.get(confidence, 1.645)
    return mu - z * s


def conditional_var(
    returns: List[float],
    confidence: float = 0.95,
) -> float:
    """
    Conditional VaR (Expected Shortfall) — average loss beyond VaR.

    Args:
        returns: List of portfolio period returns.
        confidence: Confidence level.

    Returns:
        CVaR as a negative fraction.
    """
    if len(returns) < 10:
        return 0.0
    sorted_r = sorted(returns)
    cutoff = max(int(len(sorted_r) * (1 - confidence)), 1)
    tail = sorted_r[:cutoff]
    return _mean(tail) if tail else sorted_r[0]


def cvar_from_var(
    returns: List[float],
    confidence: float = 0.95,
) -> float:
    """Alias for conditional_var (CVaR = Expected Shortfall)."""
    return conditional_var(returns, confidence)


# ---------------------------------------------------------------------------
# CDaR — Conditional Drawdown at Risk (Chek et al., 2003)
# ---------------------------------------------------------------------------

def conditional_drawdown_at_risk(
    returns: List[float],
    confidence: float = 0.95,
) -> float:
    """
    Conditional Drawdown at Risk (CDaR / Ulcer CVaR).

    Average of drawdowns beyond the (1-confidence) percentile of the
    drawdown distribution. More intuitive than CVaR for portfolio
    risk because it measures severity of drawdowns, not just losses.

    Args:
        returns: List of portfolio period returns.
        confidence: Confidence level.

    Returns:
        CDaR as a positive fraction (higher = riskier).
    """
    if len(returns) < 10 or not _HAS_NUMPY:
        return _cdar_pure(returns, confidence)

    arr = np.array(returns)
    equity = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak

    sorted_dd = sorted(drawdowns.tolist())
    cutoff = max(int(len(sorted_dd) * (1 - confidence)), 1)
    tail = [d for d in drawdowns.tolist() if d <= sorted_dd[cutoff]]
    return abs(_mean(tail)) if tail else 0.0


def _cdar_pure(returns: List[float], confidence: float) -> float:
    """Pure-python CDaR when numpy is not available."""
    if not returns:
        return 0.0
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    peak = equity[0]
    drawdowns = []
    for eq in equity:
        if eq > peak:
            peak = eq
        drawdowns.append((eq - peak) / peak if peak > 0 else 0.0)

    sorted_dd = sorted(drawdowns)
    cutoff = max(int(len(sorted_dd) * (1 - confidence)), 1)
    tail = [d for d in drawdowns if d <= sorted_dd[cutoff]]
    return abs(_mean(tail)) if tail else 0.0


# ---------------------------------------------------------------------------
# Drawdown Analysis
# ---------------------------------------------------------------------------

def max_drawdown_analysis(returns: List[float]) -> DrawdownInfo:
    """
    Comprehensive drawdown analysis.

    Returns a DrawdownInfo with max drawdown, duration, underwater
    series, and the peak/trough indices.
    """
    if not returns:
        return DrawdownInfo(0.0, 0, 0.0, [], 0, 0)

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))

    peak = equity[0]
    max_dd = 0.0
    dd_series = []
    in_drawdown = False
    dd_start_idx = 0
    max_dd_duration = 0
    peak_idx = 0
    trough_idx = 0
    current_dd_start = 0

    for i, eq in enumerate(equity):
        if eq >= peak:
            peak = eq
            peak_idx = i
            if in_drawdown:
                duration = i - dd_start_idx
                max_dd_duration = max(max_dd_duration, duration)
                in_drawdown = False
        dd = (eq - peak) / peak if peak > 0 else 0.0
        dd_series.append(dd)
        if dd < max_dd:
            max_dd = dd
            trough_idx = i
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            dd_start_idx = i

    if in_drawdown:
        max_dd_duration = max(max_dd_duration, len(equity) - dd_start_idx)

    return DrawdownInfo(
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        current_drawdown=dd_series[-1] if dd_series else 0.0,
        drawdown_series=dd_series,
        peak_index=peak_idx,
        trough_index=trough_idx,
    )


# ---------------------------------------------------------------------------
# Omega Ratio
# ---------------------------------------------------------------------------

def omega_ratio(
    returns: List[float],
    threshold: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Omega Ratio — probability-weighted ratio of gains to losses beyond threshold.

    omega = sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))

    Higher is better. At threshold=0 this measures total upside vs downside.
    With threshold = risk_free_rate / periods, measures excess return efficiency.

    Args:
        returns: List of period returns.
        threshold: Minimum acceptable return (default 0).
        periods_per_year: Annualisation factor.

    Returns:
        Omega ratio (>= 0).
    """
    if not returns:
        return 0.0
    if _HAS_NUMPY:
        arr = np.array(returns)
        excess = arr - threshold
        upside = float(np.sum(np.maximum(excess, 0)))
        downside = float(np.sum(np.maximum(-excess, 0)))
    else:
        upside = sum(max(r - threshold, 0) for r in returns)
        downside = sum(max(threshold - r, 0) for r in returns)

    if downside < 1e-12:
        return float('inf') if upside > 0 else 1.0
    return upside / downside


# ---------------------------------------------------------------------------
# Lower / Upper Partial Moments (LPM / HPM)
# ---------------------------------------------------------------------------

def lower_partial_moment(
    returns: List[float],
    target: float = 0.0,
    n: int = 1,
) -> float:
    """
    Lower Partial Moment of order n with threshold `target`.

    LPM_n = sum(max(target - r, 0)^n) / len(returns)

    n=2 with target=0 gives Sortino's downside variance denominator.
    """
    if not returns:
        return 0.0
    if _HAS_NUMPY:
        diff = np.maximum(target - np.array(returns), 0)
        return float(np.sum(diff ** n)) / len(returns)
    return _mean([max(target - r, 0) ** n for r in returns])


def upper_partial_moment(
    returns: List[float],
    target: float = 0.0,
    n: int = 1,
) -> float:
    """Upper Partial Moment (mirrors LPM for upside)."""
    if not returns:
        return 0.0
    if _HAS_NUMPY:
        diff = np.maximum(np.array(returns) - target, 0)
        return float(np.sum(diff ** n)) / len(returns)
    return _mean([max(r - target, 0) ** n for r in returns])


# ---------------------------------------------------------------------------
# Kappa Ratios (keel & Heidorn / Sortino)
# ---------------------------------------------------------------------------

def kappa_ratio(
    returns: List[float],
    target: float = 0.0,
    n: int = 2,
    periods_per_year: int = 252,
) -> float:
    """
    Kappa Ratio of order n.

    kappa_n = (annualised_return - target) / (LPM_n * periods_per_year)^(1/n)

    n=2 is the Sortino ratio. n=1 is the turnover ratio variant.
    """
    if not returns:
        return 0.0
    ann_ret = _mean(returns) * periods_per_year
    lpm_n = lower_partial_moment(returns, target, n)
    scaled_lpm = lpm_n * periods_per_year
    denom = scaled_lpm ** (1.0 / n)
    if denom < 1e-12:
        return float('inf') if ann_ret - target > 0 else 0.0
    return (ann_ret - target) / denom


# ---------------------------------------------------------------------------
# Standard performance ratios
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe Ratio."""
    if len(returns) < 2:
        return 0.0
    rf = risk_free_rate / periods_per_year
    excess = [r - rf for r in returns]
    mu = _mean(excess)
    s = _std(excess)
    if s < 1e-12:
        return 0.0
    return (mu / s) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: List[float],
    target: float = 0.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sortino Ratio using LPM-based downside deviation."""
    if len(returns) < 2:
        return 0.0
    ann_ret = _mean(returns) * periods_per_year
    rf_excess = (risk_free_rate - target) / periods_per_year
    lpm_2 = lower_partial_moment(returns, target, 2)
    down_dev = math.sqrt(lpm_2 * periods_per_year)
    if down_dev < 1e-12:
        return float('inf') if ann_ret > risk_free_rate else 0.0
    return (ann_ret - risk_free_rate) / down_dev


def calmar_ratio(
    returns: List[float],
    periods_per_year: int = 252,
) -> float:
    """Annualised Calmar Ratio: return / max drawdown (absolute)."""
    if not returns:
        return 0.0
    ann_ret = _mean(returns) * periods_per_year
    dd_info = max_drawdown_analysis(returns)
    if abs(dd_info.max_drawdown) < 1e-12:
        return float('inf') if ann_ret > 0 else 0.0
    return ann_ret / abs(dd_info.max_drawdown)


def tail_ratio(returns: List[float]) -> float:
    """
    Tail Ratio — ratio of the 95th percentile return to the 5th percentile.

    Values > 1 indicate positive skew (more large gains than large losses).
    """
    if len(returns) < 10:
        return 1.0
    sorted_r = sorted(returns)
    n = len(sorted_r)
    upper = sorted_r[int(n * 0.95)]
    lower = sorted_r[int(n * 0.05)]
    if abs(lower) < 1e-12:
        return float('inf') if upper > 0 else 0.0
    return upper / abs(lower)


def upside_potential_ratio(
    returns: List[float],
    target: float = 0.0,
) -> float:
    """
    Upside Potential Ratio — mean upside / downside deviation.

    Uses target as the minimum acceptable return.
    """
    if not returns:
        return 0.0
    hpm = math.sqrt(upper_partial_moment(returns, target, 2))
    lpm = math.sqrt(lower_partial_moment(returns, target, 2))
    if lpm < 1e-12:
        return float('inf') if hpm > 0 else 0.0
    return hpm / lpm


def roys_safety_first(
    returns: List[float],
    threshold: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Roy's Safety-First Ratio.

    (annualised_return - threshold) / (annualised_max_drawdown)

    Evaluates the probability of returns falling below a threshold.
    """
    if not returns:
        return 0.0
    ann_ret = _mean(returns) * periods_per_year
    dd_info = max_drawdown_analysis(returns)
    denom = abs(dd_info.max_drawdown)
    if denom < 1e-12:
        return float('inf') if ann_ret > threshold else 0.0
    return (ann_ret - threshold) / denom


# ---------------------------------------------------------------------------
# Benchmarked ratios (require benchmark return series)
# ---------------------------------------------------------------------------

def information_ratio(
    returns: List[float],
    benchmark: List[float],
    periods_per_year: int = 252,
) -> float:
    """Information Ratio — active return / tracking error."""
    if len(returns) != len(benchmark) or len(returns) < 2:
        return 0.0
    active = [r - b for r, b in zip(returns, benchmark)]
    mu = _mean(active)
    te = _std(active)
    if te < 1e-12:
        return 0.0
    return (mu / te) * math.sqrt(periods_per_year)


def jensens_alpha(
    returns: List[float],
    benchmark: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Jensen's Alpha — average excess return over CAPM prediction.

    alpha = ann_return - (rf + beta * (benchmark_return - rf))
    """
    if len(returns) != len(benchmark) or len(returns) < 2:
        return 0.0
    rf = risk_free_rate / periods_per_year
    port_excess = [r - rf for r in returns]
    bench_excess = [b - rf for b in benchmark]
    port_mean = _mean(port_excess)
    bench_mean = _mean(bench_excess)
    bench_var = sum((b - bench_mean) ** 2 for b in bench_excess)
    if bench_var < 1e-12:
        return 0.0
    beta = sum((port_excess[i] - port_mean) * (bench_excess[i] - bench_mean)
               for i in range(len(port_excess))) / bench_var
    alpha = port_mean - beta * bench_mean
    return alpha * periods_per_year


def treynor_ratio(
    returns: List[float],
    benchmark: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Treynor Ratio — (return - rf) / beta."""
    if len(returns) != len(benchmark) or len(returns) < 2:
        return 0.0
    rf = risk_free_rate / periods_per_year
    port_excess = [r - rf for r in returns]
    bench_excess = [b - rf for b in benchmark]
    port_mean = _mean(port_excess)
    bench_mean = _mean(bench_excess)
    bench_var = sum((b - bench_mean) ** 2 for b in bench_excess)
    if bench_var < 1e-12:
        return 0.0
    beta = sum((port_excess[i] - port_mean) * (bench_excess[i] - bench_mean)
               for i in range(len(port_excess))) / bench_var
    if abs(beta) < 1e-12:
        return 0.0
    return (port_mean * periods_per_year) / beta


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------

def rolling_sharpe(
    returns: List[float],
    window: int = 60,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> List[float]:
    """Rolling Sharpe ratio over a sliding window."""
    if len(returns) < window:
        return []
    rf = risk_free_rate / periods_per_year
    result = []
    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window:i]
        mu = _mean([r - rf for r in window_returns])
        s = _std([r - rf for r in window_returns])
        result.append((mu / s) * math.sqrt(periods_per_year) if s > 1e-12 else 0.0)
    return result


# ---------------------------------------------------------------------------
# Skewness / Kurtosis
# ---------------------------------------------------------------------------

def skewness(returns: List[float]) -> float:
    """Return distribution skewness (3rd moment)."""
    n = len(returns)
    if n < 3:
        return 0.0
    m = _mean(returns)
    s = _std(returns)
    if s < 1e-12:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * sum(((r - m) / s) ** 3 for r in returns)


def kurtosis(returns: List[float]) -> float:
    """Excess kurtosis (4th moment - 3)."""
    n = len(returns)
    if n < 4:
        return 0.0
    m = _mean(returns)
    s = _std(returns)
    if s < 1e-12:
        return 0.0
    m4 = sum(((r - m) / s) ** 4 for r in returns) / n
    return m4 - 3.0


# ---------------------------------------------------------------------------
# Full risk report
# ---------------------------------------------------------------------------

def portfolio_risk_report(
    returns: List[float],
    benchmark: Optional[List[float]] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target: float = 0.0,
) -> PortfolioRiskReport:
    """
    Generate a comprehensive portfolio risk report.

    Args:
        returns: Portfolio period return series.
        benchmark: Optional benchmark return series for alpha/IR/Treynor.
        risk_free_rate: Annualised risk-free rate.
        periods_per_year: Number of periods per year for annualisation.
        target: Target return for LPM-based ratios (default 0).

    Returns:
        PortfolioRiskReport dataclass.
    """
    if not returns:
        return PortfolioRiskReport(*([0.0] * 17))  # type: ignore[arg-type]

    # Equity curve
    equity = 1.0
    for r in returns:
        equity *= (1 + r)
    total_return = equity - 1.0

    ann_ret = _mean(returns) * periods_per_year
    ann_vol = _std(returns) * math.sqrt(periods_per_year)
    downside_dev = math.sqrt(lower_partial_moment(returns, target, 2) * periods_per_year)

    dd_info = max_drawdown_analysis(returns)

    # Benchmarked
    ir = None
    treynor = None
    alpha = None
    if benchmark is not None and len(benchmark) == len(returns):
        ir = information_ratio(returns, benchmark, periods_per_year)
        treynor = treynor_ratio(returns, benchmark, risk_free_rate, periods_per_year)
        alpha = jensens_alpha(returns, benchmark, risk_free_rate, periods_per_year)

    rolling_sharpes = rolling_sharpe(returns, window=min(60, len(returns)),
                                     risk_free_rate=risk_free_rate,
                                     periods_per_year=periods_per_year)

    return PortfolioRiskReport(
        total_return=round(total_return, 6),
        annualized_return=round(ann_ret, 6),
        volatility=round(ann_vol, 6),
        downside_deviation=round(downside_dev, 6),
        var_95=round(value_at_risk(returns, 0.95), 6),
        cvar_95=round(conditional_var(returns, 0.95), 6),
        var_99=round(value_at_risk(returns, 0.99), 6),
        cvar_99=round(conditional_var(returns, 0.99), 6),
        cdar_95=round(conditional_drawdown_at_risk(returns, 0.95), 6),
        cdar_99=round(conditional_drawdown_at_risk(returns, 0.99), 6),
        max_drawdown=round(dd_info.max_drawdown, 6),
        max_drawdown_duration=dd_info.max_drawdown_duration,
        sharpe_ratio=round(sharpe_ratio(returns, risk_free_rate, periods_per_year), 4),
        sortino_ratio=round(sortino_ratio(returns, target, risk_free_rate, periods_per_year), 4),
        omega_ratio=round(omega_ratio(returns, threshold=target, periods_per_year=periods_per_year), 4),
        calmar_ratio=round(calmar_ratio(returns, periods_per_year), 4),
        tail_ratio=round(tail_ratio(returns), 4),
        skewness=round(skewness(returns), 4),
        kurtosis=round(kurtosis(returns), 4),
        information_ratio=round(ir, 4) if ir is not None else None,
        treynor_ratio=round(treynor, 4) if treynor is not None else None,
        jensens_alpha=round(alpha, 6) if alpha is not None else None,
        kappa_1=round(kappa_ratio(returns, target, 1, periods_per_year), 4),
        kappa_2=round(kappa_ratio(returns, target, 2, periods_per_year), 4),
        kappa_3=round(kappa_ratio(returns, target, 3, periods_per_year), 4),
        roys_safety_first=round(roys_safety_first(returns, target, periods_per_year), 4),
        rolling_sharpe_mean=round(_mean(rolling_sharpes), 4) if rolling_sharpes else None,
        rolling_sharpe_std=round(_std(rolling_sharpes), 4) if rolling_sharpes else None,
    )


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

#: Alias for conditional_var
cvar = conditional_var

#: Alias for conditional_drawdown_at_risk
cdar = conditional_drawdown_at_risk

#: Alias for portfolio_risk_report
risk_report = portfolio_risk_report
