"""Bootstrap重采样评估 — 计算指标置信区间.

Absorbs pybroker's bootstrap evaluation concepts into 量化之神:

- ``BootstrapEvaluator`` — BCa bootstrap计算Sharpe/ProfitFactor等指标的置信区间
- ``BootstrapResult`` — 结果容器, 包含点估计和置信区间
- ``normal_cdf``, ``inverse_normal_cdf`` — 标准正态分布函数

Usage
-----
```python
from quant_trading.backtester import BootstrapEvaluator

evaluator = BootstrapEvaluator(n_bootstrap=1000, seed=42)
result = evaluator.evaluate(returns, metric='sharpe')
print(f"Sharpe: {result['point_estimate']:.3f} "
      f"CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
```

English:
    Bootstrap resampling evaluation for computing confidence intervals
    of trading performance metrics (Sharpe Ratio, Profit Factor, etc.)
    using the bias-corrected and accelerated (BCa) method.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

__all__ = [
    "BootstrapEvaluator",
    "BootstrapResult",
    "normal_cdf",
    "inverse_normal_cdf",
]


# ---------------------------------------------------------------------------
# Numba availability check
# ---------------------------------------------------------------------------
try:
    from numba import jit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


# ---------------------------------------------------------------------------
# Statistical helper functions (same implementation as pybroker/vect.py)
# ---------------------------------------------------------------------------
if _NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def normal_cdf(z: float) -> float:
        """Standard normal CDF."""
        zz = np.fabs(z)
        pdf = np.exp(-0.5 * zz * zz) / np.sqrt(2.0 * np.pi)
        t = 1.0 / (1.0 + zz * 0.2316419)
        poly = (
            (((1.330274429 * t - 1.821255978) * t + 1.781477937) * t
             - 0.356563782)
            * t
            + 0.319381530
        ) * t
        return 1.0 - pdf * poly if z > 0 else pdf * poly

    @jit(nopython=True, cache=True)
    def inverse_normal_cdf(p: float) -> float:
        """Inverse standard normal CDF (approx)."""
        pp = p if p <= 0.5 else 1.0 - p
        if pp <= 0.0:
            pp = 1.0e-10
        t = np.sqrt(np.log(1.0 / (pp * pp)))
        numer = (0.010328 * t + 0.802853) * t + 2.515517
        denom = ((0.001308 * t + 0.189269) * t + 1.432788) * t + 1.0
        x = t - numer / denom
        return x if p <= 0.5 else -x
else:
    def normal_cdf(z: float) -> float:
        """Standard normal CDF."""
        import math
        zz = abs(z)
        pdf = math.exp(-0.5 * zz * zz) / math.sqrt(2.0 * math.pi)
        t = 1.0 / (1.0 + zz * 0.2316419)
        poly = (
            (((1.330274429 * t - 1.821255978) * t + 1.781477937) * t
             - 0.356563782)
            * t
            + 0.319381530
        ) * t
        return 1.0 - pdf * poly if z > 0 else pdf * poly

    def inverse_normal_cdf(p: float) -> float:
        """Inverse standard normal CDF (approx)."""
        import math
        pp = p if p <= 0.5 else 1.0 - p
        if pp <= 0.0:
            pp = 1.0e-10
        t = math.sqrt(math.log(1.0 / (pp * pp)))
        numer = (0.010328 * t + 0.802853) * t + 2.515517
        denom = ((0.001308 * t + 0.189269) * t + 1.432788) * t + 1.0
        x = t - numer / denom
        return x if p <= 0.5 else -x


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
if _NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _sharpe_metric(returns: np.ndarray) -> float:
        """Compute Sharpe Ratio (annualized)."""
        n = len(returns)
        if n == 0:
            return 0.0
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(returns) / std * np.sqrt(252)

    @jit(nopython=True, cache=True)
    def _sortino_metric(returns: np.ndarray) -> float:
        """Compute Sortino Ratio (annualized)."""
        n = len(returns)
        if n == 0:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0
        std = np.std(downside)
        if std == 0:
            return 0.0
        return np.mean(returns) / std * np.sqrt(252)

    @jit(nopython=True, cache=True)
    def _profit_factor_metric(pnls: np.ndarray) -> float:
        """Compute Profit Factor."""
        profits = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_profit = np.sum(profits) if len(profits) > 0 else 0.0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @jit(nopython=True, cache=True)
    def _total_return_metric(values: np.ndarray) -> float:
        """Compute total return."""
        if len(values) < 2:
            return 0.0
        return values[-1] / values[0] - 1.0
else:
    def _sharpe_metric(returns: np.ndarray) -> float:
        n = len(returns)
        if n == 0:
            return 0.0
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(returns) / std * np.sqrt(252)

    def _sortino_metric(returns: np.ndarray) -> float:
        n = len(returns)
        if n == 0:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0
        std = np.std(downside)
        if std == 0:
            return 0.0
        return np.mean(returns) / std * np.sqrt(252)

    def _profit_factor_metric(pnls: np.ndarray) -> float:
        profits = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_profit = np.sum(profits) if len(profits) > 0 else 0.0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _total_return_metric(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        return values[-1] / values[0] - 1.0


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
_METRIC_FUNCS: dict[str, Callable[[np.ndarray], float]] = {
    "sharpe": _sharpe_metric,
    "sortino": _sortino_metric,
    "profit_factor": _profit_factor_metric,
    "total_return": _total_return_metric,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    """Bootstrap evaluation result.

    Attributes:
        point_estimate: Point estimate of the metric.
        ci_lower: Lower bound of the confidence interval (95% default).
        ci_upper: Upper bound of the confidence interval (95% default).
        std_error: Standard error of the bootstrap estimate.
        metric_name: Name of the evaluated metric.
        n_bootstrap: Number of bootstrap samples used.
        bias: Estimated bias (point_estimate - bootstrap_mean).
        confidence_intervals: Dict of confidence levels → (lower, upper).
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    metric_name: str
    n_bootstrap: int
    bias: float
    confidence_intervals: dict[str, tuple[float, float]]


# ---------------------------------------------------------------------------
# BCa helper (bias-corrected and accelerated bootstrap)
# ---------------------------------------------------------------------------
def _bca_confidence_intervals(
    boot_samples: np.ndarray,
    point_estimate: float,
    original: np.ndarray,
    metric_func: Callable[[np.ndarray], float],
) -> dict[str, tuple[float, float]]:
    """Compute BCa confidence intervals at 90%, 95%, and 99% levels.

    Parameters
    ----------
    boot_samples : np.ndarray
        Bootstrap sample metric values.
    point_estimate : float
        Point estimate on original data.
    original : np.ndarray
        Original return/data array.
    metric_func : callable
        Metric function.

    Returns
    -------
    dict
        Mapping from level string → (lower, upper) tuple.
    """
    n_boot = len(boot_samples)
    n_orig = len(original)

    # Bias correction factor (z0)
    below_count = np.sum(boot_samples < point_estimate)
    below_count = max(min(below_count, n_boot - 1), 1)
    z0 = inverse_normal_cdf(below_count / n_boot)

    # Acceleration factor (a) — jackknife
    theta_dot = np.zeros(n_orig)
    for i in range(n_orig):
        jackknife_sample = np.concatenate([original[:i], original[i + 1 :]])
        theta_dot[i] = metric_func(jackknife_sample)
    theta_dot_mean = np.mean(theta_dot)

    numer = np.sum((theta_dot_mean - theta_dot) ** 3)
    denom = 6.0 * (np.sum((theta_dot_mean - theta_dot) ** 2) ** 1.5)
    accel = numer / (denom + 1e-60)

    # Compute intervals
    boot_sorted = np.sort(boot_samples)
    intervals: dict[str, tuple[float, float]] = {}

    for alpha, label in [(0.05, "95%"), (0.10, "90%"), (0.01, "99%")]:
        z_lo = inverse_normal_cdf(alpha / 2)
        z_hi = inverse_normal_cdf(1 - alpha / 2)

        a_lo = normal_cdf(z0 + (z0 + z_lo) / (1 - accel * (z0 + z_lo)))
        a_hi = normal_cdf(z0 + (z0 + z_hi) / (1 - accel * (z0 + z_hi)))

        k_lo = max(int(a_lo * (n_boot + 1)) - 1, 0)
        k_hi = min(int((1 - a_hi) * (n_boot + 1)) - 1, n_boot - 1)

        lower = boot_sorted[k_lo]
        upper = boot_sorted[n_boot - 1 - k_hi]
        intervals[label] = (float(lower), float(upper))

    return intervals


# ---------------------------------------------------------------------------
# Bootstrap Evaluator
# ---------------------------------------------------------------------------
class BootstrapEvaluator:
    """Bootstrap重采样评估 — 计算指标置信区间.

    Uses the bias-corrected and accelerated (BCa) bootstrap method to
    compute confidence intervals for trading performance metrics.

    Parameters
    ----------
    n_bootstrap : int, default 1000
        Number of bootstrap resamples.
    seed : int, default 42
        Random seed for reproducibility.
    sample_size : int, optional
        Size of each bootstrap sample. Defaults to len(returns).
    confidence_level : float, default 0.95
        Primary confidence level for CI reporting.

    Example
    -------
    ```python
    evaluator = BootstrapEvaluator(n_bootstrap=2000, seed=42)
    result = evaluator.evaluate(returns, metric='sharpe')

    print(f"Sharpe: {result.point_estimate:.3f}")
    print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print(f"Std Error: {result.std_error:.3f}")
    ```
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        seed: int = 42,
        sample_size: Optional[int] = None,
        confidence_level: float = 0.95,
    ):
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive.")
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be between 0 and 1.")

        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.sample_size = sample_size
        self.confidence_level = confidence_level

    def evaluate(
        self,
        returns: np.ndarray,
        metric: str = "sharpe",
        equity_curve: Optional[np.ndarray] = None,
    ) -> BootstrapResult:
        """Evaluate a metric using bootstrap resampling.

        Parameters
        ----------
        returns : np.ndarray
            Array of returns (centered at 0) or PnLs.
        metric : str, default 'sharpe'
            Metric to evaluate. Options: 'sharpe', 'sortino',
            'profit_factor', 'total_return'.
        equity_curve : np.ndarray, optional
            Equity curve array. Required if metric='total_return'.

        Returns
        -------
        BootstrapResult
            Contains point estimate, CI bounds, std error, and bias.
        """
        if metric not in _METRIC_FUNCS:
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Supported: {list(_METRIC_FUNCS.keys())}"
            )

        metric_func = _METRIC_FUNCS[metric]

        # Compute point estimate
        if metric == "total_return":
            if equity_curve is None:
                raise ValueError("equity_curve required for total_return metric.")
            point_estimate = float(metric_func(equity_curve))
        else:
            point_estimate = float(metric_func(returns))

        # Bootstrap resampling
        rng = np.random.default_rng(self.seed)
        n = self.sample_size if self.sample_size is not None else len(returns)
        n = min(n, len(returns))

        boot_values = np.zeros(self.n_bootstrap)

        if metric == "total_return":
            if equity_curve is None:
                raise ValueError("equity_curve required for total_return.")
            for i in range(self.n_bootstrap):
                idx = rng.integers(0, len(equity_curve), size=n)
                sample = equity_curve[idx]
                boot_values[i] = _total_return_metric(sample)
        else:
            for i in range(self.n_bootstrap):
                idx = rng.integers(0, len(returns), size=n)
                sample = returns[idx]
                boot_values[i] = metric_func(sample)

        # Standard error
        std_error = float(np.std(boot_values))

        # Bias
        bias = float(point_estimate - np.mean(boot_values))

        # BCa confidence intervals
        if metric == "total_return":
            if equity_curve is None:
                raise ValueError("equity_curve required for total_return.")
            conf_intervals = _bca_confidence_intervals(
                boot_values, point_estimate, equity_curve, _total_return_metric
            )
        else:
            conf_intervals = _bca_confidence_intervals(
                boot_values, point_estimate, returns, metric_func
            )

        # Primary CI (confidence_level)
        alpha = 1 - self.confidence_level
        if self.confidence_level == 0.95:
            ci_label = "95%"
        elif self.confidence_level == 0.90:
            ci_label = "90%"
        elif self.confidence_level == 0.99:
            ci_label = "99%"
        else:
            ci_label = f"{self.confidence_level:.0%}"
            z_lo = inverse_normal_cdf(alpha / 2)
            z_hi = inverse_normal_cdf(1 - alpha / 2)
            boot_sorted = np.sort(boot_values)
            k_lo = max(int((alpha / 2) * (self.n_bootstrap + 1)) - 1, 0)
            k_hi = min(
                int((1 - alpha / 2) * (self.n_bootstrap + 1)) - 1,
                self.n_bootstrap - 1,
            )
            conf_intervals[ci_label] = (
                float(boot_sorted[k_lo]),
                float(boot_sorted[self.n_bootstrap - 1 - k_hi]),
            )

        ci_lower, ci_upper = conf_intervals.get(ci_label, (point_estimate, point_estimate))

        return BootstrapResult(
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=std_error,
            metric_name=metric,
            n_bootstrap=self.n_bootstrap,
            bias=bias,
            confidence_intervals=conf_intervals,
        )
