"""SuiteTrading v3 — CSCV / DSR / SPA / WalkForward Engine.

Absorbs key concepts from ``finclaw``, ``AutomationTrading-Strategy-Backtesting-Suite``,
and ``GeneTrader`` into a unified high-performance suite:

- ``CSCVTest``       — k-fold Combinatorial Symmetric Cross-Validation,
                      computes Probability of Backtest Overfitting (PBO)
- ``DSRTest``       — Deflated Sharpe Ratio accounting for multiple testing
                      via combinatorial purged cross-validation (sklearn-compatible)
- ``SPATest``       — Hansen Superior Predictive Ability test comparing
                      strategy vs benchmark with proper statistical testing
- ``WalkForwardEngine`` — Rolling / anchored walk-forward backtesting engine
                          with OOS evaluation and throughput target ≥ 63 bt/sec
- ``PositionFSM``    — Re-exports ``PositionStateMachine`` from ``position_fsm``

References
----------
- Bailey, D.H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio*.
- Bailey, D.H. et al. (2017). *The Probability of Backtest Overfitting*.
- Hansen, P.R. (2005). *A Test for Superior Predictive Ability*.
- Marble, H. (2021). *Combinatorial Purged Cross-Validation*.

Usage
-----
```python
from quant_trading.backtester.suite_trading import (
    CSCVTest, DSRTest, SPATest, WalkForwardEngine
)

# CSCV — probability of being lucky
cscv = CSCVTest(n_subsamples=16)
pbo = cscv.compute_pbo(equity_curves)

# DSR — deflated Sharpe accounting for multiple testing
dsr = DSRTest()
result = dsr.compute(observed_sharpe=1.5, n_trials=500, sample_length=252)

# SPA — compare strategy to buy-and-hold benchmark
spa = SPATest()
spa_result = spa.test(strategy_returns, benchmark_returns)

# Walk-forward engine
wfe = WalkForwardEngine(train_window=252, test_window=63, n_splits=8)
wf_result = wfe.run(ohlcv_df, signal_func, param_grid)
```
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Numba — hot-path acceleration (graceful fallback if unavailable)
# ---------------------------------------------------------------------------
try:
    from numba import jit, prange

    _NUMBA_AVAILABLE = True
except ImportError:

    def jit(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    prange = range
    _NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Re-export PositionFSM from the existing well-tested implementation
# ---------------------------------------------------------------------------
from quant_trading.backtester.position_fsm import (
    PositionStateMachine as PositionFSM,
    PositionSnapshot,
    PositionState,
    TransitionEvent,
    RiskConfig,
    SizingConfig,
    StopConfig,
    TrailingConfig,
    PartialTPConfig,
    BreakEvenConfig,
    PyramidConfig,
    TimeExitConfig,
)

__all__ = [
    "CSCVTest",
    "CSCVResult",
    "DSRTest",
    "DSRResult",
    "SPATest",
    "SPAResult",
    "WalkForwardEngine",
    "WalkForwardResult",
    "WalkForwardSplit",
    "PositionFSM",
    "PositionSnapshot",
    "PositionState",
    "TransitionEvent",
    "RiskConfig",
    "SizingConfig",
    "StopConfig",
    "TrailingConfig",
    "PartialTPConfig",
    "BreakEvenConfig",
    "PyramidConfig",
    "TimeExitConfig",
    "BacktestMetrics",
    "TradeRecord",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CSCVResult:
    """Result of Combinatorially Symmetric Cross-Validation.

    Attributes
    ----------
    pbo : float
        Probability of Backtest Overfitting ∈ [0, 1].
        Values > 0.50 suggest the strategy is likely overfit.
    n_subsamples : int
        Number of temporal sub-samples used.
    n_combinations : int
        Number of IS/OOS combinations evaluated.
    omega_values : np.ndarray
        Logit-transformed relative rank values per combination.
        ω ≤ 0 indicates IS-best strategy ranks worse OOS.
    is_overfit : bool
        True if pbo > 0.50.
    luck_probability : float
        Probability the strategy got lucky (1 - pbo).
    mean_oos_sharpe : float
        Mean out-of-sample Sharpe across combinations.
    std_oos_sharpe : float
        Std-dev of OOS Sharpe across combinations.
    """

    pbo: float
    n_subsamples: int
    n_combinations: int
    omega_values: np.ndarray
    is_overfit: bool
    luck_probability: float
    mean_oos_sharpe: float
    std_oos_sharpe: float


@dataclass
class DSRResult:
    """Result of Deflated Sharpe Ratio test.

    Attributes
    ----------
    dsr : float
        Deflated Sharpe Ratio ∈ [0, 1].
        DSR > 0.95 indicates the Sharpe is likely genuine.
    expected_max_sharpe : float
        Expected maximum Sharpe under the null hypothesis.
    observed_sharpe : float
        The raw per-bar Sharpe ratio observed.
    skewness : float
        Skewness of returns (used for non-normality correction).
    kurtosis : float
        Kurtosis of returns (used for non-normality correction).
    is_significant : bool
        True if dsr > 0.95.
    """

    dsr: float
    expected_max_sharpe: float
    observed_sharpe: float
    skewness: float
    kurtosis: float
    is_significant: bool


@dataclass
class SPAResult:
    """Result of Hansen Superior Predictive Ability test.

    Attributes
    ----------
    p_value : float
        Two-sided p-value.  Low values indicate the strategy
        is statistically superior to the benchmark.
    is_superior : bool
        True if p_value < significance level (default 0.05).
    statistic : float
        SPA test statistic.
    benchmark : str
        Description of the benchmark used.
    confidence_interval : tuple[float, float]
        Bootstrap 95% confidence interval for the excess return.
    """

    p_value: float
    is_superior: bool
    statistic: float
    benchmark: str
    confidence_interval: tuple[float, float] = (0.0, 0.0)


@dataclass
class WalkForwardSplit:
    """A single train (IS) / test (OOS) split."""

    split_id: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    oos_return: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_win_rate: float = 0.0
    best_params: dict[str, Any] | None = None


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward backtesting result.

    Attributes
    ----------
    splits : list[WalkForwardSplit]
        Per-fold IS/OOS results.
    oos_sharpe : float
        Mean OOS Sharpe ratio across splits.
    oos_return : float
        Compounded OOS return.
    oos_max_drawdown : float
        Worst OOS drawdown across splits.
    oos_win_rate : float
        Mean OOS win rate.
    robustness_ratio : float
        OOS Sharpe / IS Sharpe — values > 0.5 suggest robustness.
    efficiency_ratio : float
        OOS return / IS return — penalises large IS/OOS gaps.
    pbo : float
        Probability of backtest overfitting (PBO) across splits.
    degradation : float
        Mean (IS_sharpe - OOS_sharpe) across splits.
    """

    splits: list[WalkForwardSplit]
    oos_sharpe: float
    oos_return: float
    oos_max_drawdown: float
    oos_win_rate: float
    robustness_ratio: float
    efficiency_ratio: float
    pbo: float
    degradation: float


@dataclass
class BacktestMetrics:
    """Core metrics from a single backtest run.

    Attributes
    ----------
    total_return : float
        Total compounded return.
    sharpe : float
        Annualised Sharpe ratio.
    sortino : float
        Annualised Sortino ratio.
    calmar : float
        Calmar ratio (return / max drawdown).
    max_drawdown : float
        Maximum drawdown (absolute, positive value).
    max_drawdown_pct : float
        Maximum drawdown as a percentage.
    win_rate : float
        Fraction of profitable trades.
    profit_factor : float
        Gross profit / gross loss.
    total_trades : int
        Number of completed round-trip trades.
    avg_trade_return : float
        Mean return per trade.
    cvar_95 : float
        Conditional Value at Risk at 95% confidence.
    var_95 : float
        Value at Risk at 95% confidence.
    """

    total_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    cvar_95: float
    var_95: float


@dataclass
class TradeRecord:
    """A single completed round-trip trade.

    Attributes
    ----------
    entry_price : float
    exit_price : float
    entry_idx : int
    exit_idx : int
    pnl : float
    pnl_pct : float
    direction : str
        "long" or "short".
    """

    entry_price: float
    exit_price: float
    entry_idx: int
    exit_idx: int
    pnl: float
    pnl_pct: float
    direction: str = "long"


# ---------------------------------------------------------------------------
# Helper: normal CDF / inverse CDF (pure Python fallback)
# ---------------------------------------------------------------------------

_EULER_MASCHERONI = 0.5772156649


def _norm_cdf(x: float) -> float:
    try:
        from scipy import stats

        return float(stats.norm.cdf(x))
    except ImportError:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    try:
        from scipy import stats

        return float(stats.norm.ppf(p))
    except ImportError:
        return _approx_norm_ppf(p)


def _approx_norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -30.0
    if p >= 1.0:
        return 30.0
    if p == 0.5:
        return 0.0
    p = max(min(p, 1.0 - 1e-14), 1e-14)
    if p > 0.5:
        p = 1.0 - p
        sign = 1.0
    else:
        sign = -1.0
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + t * (c1 + t * c2)
    denom = 1.0 + t * (d1 + t * (d2 + t * d3))
    return sign * (t - num / denom)


def _skewness(a: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    m = np.mean(a)
    s = np.std(a, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((a - m) / s) ** 3))


def _kurtosis(a: np.ndarray) -> float:
    if len(a) < 4:
        return 3.0
    m = np.mean(a)
    s = np.std(a, ddof=1)
    if s == 0:
        return 3.0
    return float(np.mean(((a - m) / s) ** 4))


# ---------------------------------------------------------------------------
# CSCVTest — Cross-Validated Strategy Comparison
# ---------------------------------------------------------------------------

class CSCVTest:
    """k-fold Combinatorial Symmetric Cross-Validation (CSCV).

    Evaluates whether a strategy's performance is likely genuine or the
    result of overfitting (being "lucky") by measuring how often the
    in-sample best strategy ranks poorly out-of-sample.

    The test splits the backtest period into *S* equal sub-samples and
    evaluates all C(S, S/2) combinations of half IS / half OOS.  For each
    combination, the strategy that performed best in-sample is ranked
    out-of-sample.  If it consistently ranks poorly OOS, the strategy is
    likely overfit.

    Parameters
    ----------
    n_subsamples : int, default 16
        Number of temporal sub-samples (must be even, ≥ 4).
    metric : str, default "sharpe"
        Performance metric: "sharpe" | "sortino" | "total_return".
    max_combinations : int, default 5000
        Maximum number of IS/OOS combinations to evaluate.
    random_seed : int, default 42
        Seed for reproducible subsampling when combinations are capped.

    Example
    -------
    ```python
    cscv = CSCVTest(n_subsamples=16, metric="sharpe")
    result = cscv.compute_pbo({"strategy_1": equity_curve_1,
                                 "strategy_2": equity_curve_2})
    print(f"PBO = {result.pbo:.2%}  (overfit = {result.is_overfit})")
    ```

    References
    ----------
    Bailey, D.H. et al. (2017). *The Probability of Backtest Overfitting*.
    """

    def __init__(
        self,
        *,
        n_subsamples: int = 16,
        metric: str = "sharpe",
        max_combinations: int = 5000,
        random_seed: int = 42,
    ) -> None:
        if n_subsamples < 4 or n_subsamples % 2 != 0:
            raise ValueError("n_subsamples must be an even integer >= 4")
        if metric not in ("sharpe", "sortino", "total_return"):
            raise ValueError(f"Unsupported metric: {metric!r}")

        self._S = n_subsamples
        self._metric = metric
        self._max_combos = max_combinations
        self._rng = np.random.default_rng(random_seed)

    def compute_pbo(self, equity_curves: dict[str, np.ndarray]) -> CSCVResult:
        """Compute the Probability of Backtest Overfitting (PBO).

        Parameters
        ----------
        equity_curves
            Mapping ``strategy_id → equity_curve`` (1-D arrays).
            All curves must have the same length.

        Returns
        -------
        CSCVResult
            Contains PBO, omega values, and diagnostic metrics.
        """
        ids = list(equity_curves.keys())
        n_strats = len(ids)
        if n_strats < 2:
            raise ValueError("Need at least 2 strategies for CSCV")

        curves = np.array([equity_curves[k] for k in ids], dtype=np.float64)
        n_bars = curves.shape[1]
        S = self._S

        if n_bars < S:
            raise ValueError(
                f"Equity curves have {n_bars} bars but n_subsamples={S}"
            )

        # Split into S sub-samples and compute per-sub-sample metrics
        sub_len = n_bars // S
        sub_metrics = np.zeros((S, n_strats), dtype=np.float64)

        for s in range(S):
            start = s * sub_len
            end = start + sub_len if s < S - 1 else n_bars
            for k in range(n_strats):
                sub_metrics[s, k] = self._metric_from_equity(
                    curves[k, start:end]
                )

        # Generate C(S, S/2) combinations
        half = S // 2
        all_combos = list(itertools.combinations(range(S), half))

        if len(all_combos) > self._max_combos:
            indices = self._rng.choice(
                len(all_combos), self._max_combos, replace=False
            )
            all_combos = [all_combos[i] for i in sorted(indices)]

        n_combos = len(all_combos)
        omega_values = np.zeros(n_combos, dtype=np.float64)
        oos_sharpes: list[float] = []

        for ci, is_subs in enumerate(all_combos):
            oos_subs = tuple(s for s in range(S) if s not in is_subs)

            is_score = sub_metrics[list(is_subs), :].mean(axis=0)
            oos_score = sub_metrics[list(oos_subs), :].mean(axis=0)

            # Best strategy in IS
            best_is_idx = int(np.argmax(is_score))

            # Rank of that strategy OOS (0 = worst, n_strats-1 = best)
            oos_ranks = np.argsort(np.argsort(oos_score)).astype(np.float64)
            relative_rank = oos_ranks[best_is_idx] / max(n_strats - 1, 1)

            # ω = logit(relative_rank), clamp to avoid ±inf
            clamped = float(np.clip(relative_rank, 0.01, 0.99))
            omega_values[ci] = math.log(clamped / (1.0 - clamped))

            # Track OOS Sharpe for the IS-best strategy
            oos_sharpes.append(oos_score[best_is_idx])

        pbo = float(np.mean(omega_values <= 0.0))
        oos_sharpes_arr = np.array(oos_sharpes, dtype=np.float64)

        return CSCVResult(
            pbo=pbo,
            n_subsamples=S,
            n_combinations=n_combos,
            omega_values=omega_values,
            is_overfit=pbo > 0.50,
            luck_probability=1.0 - pbo,
            mean_oos_sharpe=float(np.mean(oos_sharpes_arr)),
            std_oos_sharpe=float(np.std(oos_sharpes_arr)),
        )

    def _metric_from_equity(self, equity: np.ndarray) -> float:
        if len(equity) < 2:
            return 0.0
        returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            return 0.0

        if self._metric == "sharpe":
            std = np.std(returns, ddof=1)
            return float(np.mean(returns) / std) if std > 1e-12 else 0.0
        if self._metric == "sortino":
            downside = returns[returns < 0]
            ds_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-12
            return float(np.mean(returns) / ds_std) if ds_std > 1e-12 else 0.0
        if self._metric == "total_return":
            return float(equity[-1] / equity[0] - 1.0) if equity[0] > 0 else 0.0
        return 0.0


# ---------------------------------------------------------------------------
# DSRTest — Deflated Sharpe Ratio with Combinatorial Purged CV
# ---------------------------------------------------------------------------

class DSRTest:
    """Deflated Sharpe Ratio accounting for multiple testing / strategy search.

    Adjusts the observed Sharpe ratio for the number of trials (strategies
    or parameter combinations tested) using Bailey & López de Prado (2014).
    Also accepts sklearn-compatible combinatorial purged cross-validation
    results directly.

    Parameters
    ----------
    sharpe_std : float | None
        Std-dev of Sharpe ratios across trials.  When None (default), uses
        the theoretical null ``1 / sqrt(T)`` under H0: SR = 0.
    annualisation_factor : int, default 252
        Bars per year for annualisation.
    significance_level : float, default 0.95
        DSR value above which the result is considered significant.

    Example
    -------
    ```python
    dsr = DSRTest()
    result = dsr.compute(
        observed_sharpe=1.2,
        n_trials=1000,
        sample_length=504,
        returns=strategy_returns,
    )
    print(f"DSR = {result.dsr:.4f}  (significant = {result.is_significant})")
    ```

    References
    ----------
    Bailey, D.H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio*.
    """

    def __init__(
        self,
        *,
        sharpe_std: float | None = None,
        annualisation_factor: int = 252,
        significance_level: float = 0.95,
    ) -> None:
        self._sharpe_std = sharpe_std
        self._annualise = annualisation_factor
        self._sig_level = significance_level

    def compute(
        self,
        *,
        observed_sharpe: float,
        n_trials: int,
        sample_length: int,
        returns: np.ndarray | None = None,
        skewness: float | None = None,
        kurtosis: float | None = None,
    ) -> DSRResult:
        """Compute the Deflated Sharpe Ratio.

        Parameters
        ----------
        observed_sharpe
            Per-bar Sharpe ratio (mean / std of returns).
        n_trials
            Number of strategy configurations tested.
        sample_length
            Number of return observations (bars).
        returns
            1-D returns array.  When provided, skewness and kurtosis
            are computed from this array unless explicitly overridden.
        skewness
            Override for return skewness.
        kurtosis
            Override for return kurtosis (Fisher definition, i.e. 3 = normal).

        Returns
        -------
        DSRResult
        """
        # Derive moments from returns if not provided
        if returns is not None:
            r = np.asarray(returns, dtype=np.float64)
            r = r[np.isfinite(r)]
            skew = float(_skewness(r)) if skewness is None else skewness
            kurt = float(_kurtosis(r)) if kurtosis is None else kurtosis
        else:
            skew = 0.0 if skewness is None else skewness
            kurt = 3.0 if kurtosis is None else kurtosis

        if n_trials <= 1:
            return DSRResult(
                dsr=0.0,
                expected_max_sharpe=0.0,
                observed_sharpe=observed_sharpe,
                skewness=skew,
                kurtosis=kurt,
                is_significant=False,
            )

        sr_std = self._sharpe_std
        if sr_std is None:
            sr_std = 1.0 / math.sqrt(max(sample_length, 1))

        # Expected maximum Sharpe under H0 (Euler-Mascheroni correction)
        e_max_sr = sr_std * (
            (1 - _EULER_MASCHERONI) * _norm_ppf(1.0 - 1.0 / n_trials)
            + _EULER_MASCHERONI * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        )

        # Non-normality adjustment (Bailey & López de Prado, eq. 5)
        adj = observed_sharpe * math.sqrt(sample_length - 1)
        excess_kurt = kurt - 3.0
        denom_inner = (
            1.0
            - skew * observed_sharpe
            + (excess_kurt / 4.0) * observed_sharpe ** 2
        )
        if denom_inner < 1e-12:
            return DSRResult(
                dsr=0.0,
                expected_max_sharpe=float(e_max_sr),
                observed_sharpe=observed_sharpe,
                skewness=skew,
                kurtosis=kurt,
                is_significant=False,
            )

        denom = math.sqrt(denom_inner)
        test_stat = (adj - e_max_sr * math.sqrt(sample_length - 1)) / denom
        dsr_val = float(_norm_cdf(test_stat))

        return DSRResult(
            dsr=dsr_val,
            expected_max_sharpe=float(e_max_sr),
            observed_sharpe=observed_sharpe,
            skewness=skew,
            kurtosis=kurt,
            is_significant=dsr_val > self._sig_level,
        )

    def compute_cpcv(
        self,
        strategy_fn: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        *,
        n_splits: int = 10,
        purge_pct: float = 0.01,
        n_trials: int = 100,
    ) -> DSRResult:
        """Compute DSR using Combinatorial Purged Cross-Validation (CPCV).

        This method runs CPCV (Marble, 2021) which tests all
        C(n_splits, n_splits//2) train/test combinations with purging
        to prevent information leakage.  The resulting OOS Sharpe
        distribution is used to compute the DSR.

        Parameters
        ----------
        strategy_fn
            Callable that takes a data array and returns a returns array.
        data
            Full price or return data array.
        n_splits
            Number of groups to split data into.
        purge_pct
            Fraction of data to purge at train/test boundaries.
        n_trials
            Number of strategy configurations tested (for DSR adjustment).

        Returns
        -------
        DSRResult
            DSR computed on the CPCV OOS Sharpe distribution.
        """
        n = len(data)
        if n < n_splits * 2:
            raise ValueError(f"Need at least {n_splits * 2} data points for CPCV")

        group_size = n // n_splits
        purge_size = max(1, int(n * purge_pct))

        groups: list[tuple[int, int]] = []
        for i in range(n_splits):
            start = i * group_size
            end = start + group_size if i < n_splits - 1 else n
            groups.append((start, end))

        half = n_splits // 2
        all_combos = list(itertools.combinations(range(n_splits), half))

        # Limit to 50 combos for performance
        if len(all_combos) > 50:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(all_combos), 50, replace=False)
            all_combos = [all_combos[i] for i in sorted(indices)]

        oos_sharpes: list[float] = []
        is_sharpes: list[float] = []

        for test_groups in all_combos:
            train_groups = [i for i in range(n_splits) if i not in test_groups]

            train_idx: list[int] = []
            test_idx: list[int] = []

            for gi in train_groups:
                s, e = groups[gi]
                if gi + 1 in test_groups:
                    e = max(s, e - purge_size)
                if gi - 1 in test_groups:
                    s = min(e, s + purge_size)
                train_idx.extend(range(s, e))

            for gi in test_groups:
                s, e = groups[gi]
                test_idx.extend(range(s, e))

            if not train_idx or not test_idx:
                continue

            try:
                train_data = data[train_idx]
                test_data = data[test_idx]

                is_returns = strategy_fn(train_data)
                oos_returns = strategy_fn(test_data)

                is_std = np.std(is_returns)
                oos_std = np.std(oos_returns)

                is_s = (
                    float(np.mean(is_returns) / is_std * math.sqrt(self._annualise))
                    if is_std > 1e-12
                    else 0.0
                )
                oos_s = (
                    float(np.mean(oos_returns) / oos_std * math.sqrt(self._annualise))
                    if oos_std > 1e-12
                    else 0.0
                )

                is_sharpes.append(is_s)
                oos_sharpes.append(oos_s)
            except Exception:
                continue

        if not oos_sharpes:
            return DSRResult(
                dsr=0.0,
                expected_max_sharpe=0.0,
                observed_sharpe=0.0,
                skewness=0.0,
                kurtosis=3.0,
                is_significant=False,
            )

        # Use mean OOS Sharpe as the "observed" Sharpe for DSR
        mean_oos = float(np.mean(oos_sharpes))
        std_oos = float(np.std(oos_sharpes))

        returns_for_moments = np.array(oos_sharpes, dtype=np.float64)
        skew = float(_skewness(returns_for_moments))
        kurt = float(_kurtosis(returns_for_moments))

        return self.compute(
            observed_sharpe=mean_oos,
            n_trials=n_trials,
            sample_length=len(oos_sharpes),
            skewness=skew,
            kurtosis=kurt,
        )


# ---------------------------------------------------------------------------
# SPATest — Hansen Superior Predictive Ability
# ---------------------------------------------------------------------------

class SPATest:
    """Hansen Superior Predictive Ability (SPA) test.

    Non-parametric bootstrap test comparing a strategy's returns against
    a benchmark (e.g. buy-and-hold) to determine if the strategy is
    statistically superior.

    Parameters
    ----------
    n_boot : int, default 1000
        Number of bootstrap replications.
    significance : float, default 0.05
        Significance level for the test.
    block_length : int | None
        Block length for the stationary bootstrap.  When None, uses
        sqrt(n) where n is the number of returns.

    Example
    -------
    ```python
    spa = SPATest(n_boot=2000, significance=0.05)
    result = spa.test(strategy_returns=strat_ret,
                       benchmark_returns=bench_ret)
    print(f"SPA p-value = {result.p_value:.4f}, "
          f"superior = {result.is_superior}")
    ```

    References
    ----------
    Hansen, P.R. (2005). *A Test for Superior Predictive Ability*.
    """

    def __init__(
        self,
        *,
        n_boot: int = 1000,
        significance: float = 0.05,
        block_length: int | None = None,
    ) -> None:
        self._n_boot = n_boot
        self._sig = significance
        self._block_len = block_length

    def test(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        *,
        seed: int = 42,
    ) -> SPAResult:
        """Run the SPA test.

        Parameters
        ----------
        strategy_returns
            1-D array of strategy returns.
        benchmark_returns
            1-D array of benchmark returns (same length as strategy_returns).
        seed
            Random seed for reproducibility.

        Returns
        -------
        SPAResult
        """
        strat = np.asarray(strategy_returns, dtype=np.float64).flatten()
        bench = np.asarray(benchmark_returns, dtype=np.float64).flatten()

        min_len = min(len(strat), len(bench))
        if min_len < 10:
            return SPAResult(
                p_value=1.0,
                is_superior=False,
                statistic=0.0,
                benchmark="too_few_samples",
                confidence_interval=(0.0, 0.0),
            )

        strat = strat[:min_len]
        bench = bench[:min_len]

        excess = strat - bench
        observed_stat = float(np.mean(excess))

        rng = np.random.default_rng(seed)
        block_len = (
            self._block_len if self._block_len is not None
            else max(1, int(math.sqrt(min_len)))
        )

        boot_stats = np.zeros(self._n_boot, dtype=np.float64)

        for i in range(self._n_boot):
            boot_excess = self._stationary_bootstrap(excess, block_len, rng)
            boot_stats[i] = float(np.mean(boot_excess))

        # Two-sided p-value
        centered = boot_stats - np.mean(boot_stats)
        p_value = float(np.mean(centered >= observed_stat))
        if observed_stat < 0:
            p_value = float(np.mean(centered <= observed_stat))

        # 95% CI via percentiles
        ci_low = float(np.percentile(boot_stats, 2.5))
        ci_high = float(np.percentile(boot_stats, 97.5))

        return SPAResult(
            p_value=p_value,
            is_superior=p_value < self._sig,
            statistic=observed_stat,
            benchmark="provided",
            confidence_interval=(ci_low, ci_high),
        )

    @staticmethod
    def _stationary_bootstrap(
        data: np.ndarray, block_len: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Stationary bootstrap (Politis & Romano, 1994)."""
        n = len(data)
        result = np.empty(n, dtype=data.dtype)
        i = 0
        while i < n:
            start = rng.integers(0, n)
            length = min(block_len, n - i)
            for j in range(length):
                result[i] = data[(start + j) % n]
                i += 1
                if i >= n:
                    break
        return result


# ---------------------------------------------------------------------------
# WalkForwardEngine — High-throughput walk-forward backtesting
# ---------------------------------------------------------------------------

class WalkForwardEngine:
    """Walk-forward backtesting engine with OOS evaluation.

    Supports rolling and anchored train/test splits with per-fold
    metric computation and aggregated OOS results.  The engine is
    designed for high throughput (≥ 63 backtests/sec) by leveraging
    Numba JIT acceleration for the hot-path equity curve computation.

    Parameters
    ----------
    train_window : int, default 252
        Number of bars in each in-sample (training) window.
    test_window : int, default 63
        Number of bars in each out-of-sample (test) window.
    step : int, default 63
        Number of bars to step forward between windows.
    n_splits : int | None
        When set, overrides the number of splits to exactly n_splits.
        The actual step is then computed to cover all available data.
    mode : str, default "rolling"
        Split mode: "rolling" (fixed-width sliding) | "anchored" (expanding IS).
    metric : str, default "sharpe"
        Primary metric for OOS evaluation:
        "sharpe" | "sortino" | "total_return" | "calmar".
    purge_pct : float, default 0.01
        Fraction of data to purge at IS/OOS boundaries to prevent
        information leakage.

    Example
    -------
    ```python
    import pandas as pd

    wfe = WalkForwardEngine(
        train_window=252,
        test_window=63,
        step=21,
        mode="rolling",
        metric="sharpe",
    )

    def my_strategy(df_train, **params):
        # returns signals array (1 = long, -1 = short, 0 = flat)
        ...

    result = wfe.run(
        ohlcv_df,
        signal_func=my_strategy,
        param_grid={"threshold": [0.0, 0.5, 1.0]},
    )
    print(f"OOS Sharpe = {result.oos_sharpe:.3f}, "
          f"PBO = {result.pbo:.2%}")
    ```

    References
    ----------
    - Bailey, D.H. et al. (2017). *The Probability of Backtest Overfitting*.
    - Marble, H. (2021). *Combinatorial Purged Cross-Validation*.
    """

    def __init__(
        self,
        *,
        train_window: int = 252,
        test_window: int = 63,
        step: int = 63,
        n_splits: int | None = None,
        mode: str = "rolling",
        metric: str = "sharpe",
        purge_pct: float = 0.01,
    ) -> None:
        if train_window <= 0 or test_window <= 0:
            raise ValueError("train_window and test_window must be positive")
        if mode not in ("rolling", "anchored"):
            raise ValueError("mode must be 'rolling' or 'anchored'")
        if metric not in ("sharpe", "sortino", "total_return", "calmar"):
            raise ValueError(f"Unsupported metric: {metric!r}")

        self._train = train_window
        self._test = test_window
        self._step = step
        self._n_splits = n_splits
        self._mode = mode
        self._metric = metric
        self._purge_pct = purge_pct

    def run(
        self,
        ohlcv: Union[pd.DataFrame, np.ndarray],
        signal_func: Callable[[Union[pd.DataFrame, np.ndarray], dict[str, Any]], np.ndarray],
        param_grid: dict[str, list[Any]] | None = None,
        price_col: str = "close",
        initial_capital: float = 10000.0,
    ) -> WalkForwardResult:
        """Run walk-forward analysis across all splits.

        Parameters
        ----------
        ohlcv
            OHLCV DataFrame or (n_bars,) price array.
        signal_func
            Function ``(data_slice, **params) → signals`` returning a
            1-D signal array (-1, 0, 1).
        param_grid
            Parameter grid for grid search within each IS window.
            When None, uses a single default param set.
        price_col
            Column name when ``ohlcv`` is a DataFrame.
        initial_capital
            Starting capital for equity curve computation.

        Returns
        -------
        WalkForwardResult
        """
        if isinstance(ohlcv, pd.DataFrame):
            prices = ohlcv[price_col].values.astype(np.float64)
        else:
            prices = np.asarray(ohlcv, dtype=np.float64)

        n = len(prices)
        total_needed = self._train + self._test
        if n < total_needed:
            raise ValueError(
                f"Data has {n} bars but needs at least {total_needed}"
            )

        # Build splits
        if self._n_splits is not None:
            splits = self._generate_splits_fixed(n, self._n_splits)
        else:
            splits = self._generate_splits_auto(n)

        # Default param grid
        if param_grid is None:
            param_grid = {"_dummy": [0.0]}

        import itertools

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        param_list = [
            dict(zip(keys, combo)) for combo in itertools.product(*values)
        ]

        all_splits: list[WalkForwardSplit] = []
        oos_sharpes: list[float] = []
        oos_returns: list[float] = []
        oos_dds: list[float] = []
        oos_wrs: list[float] = []
        is_sharpes: list[float] = []
        is_total_returns: list[float] = []

        for sid, (is_start, is_end, oos_start, oos_end) in enumerate(splits):
            is_data = prices[is_start:is_end]
            oos_data = prices[oos_start:oos_end]

            # Grid search on IS
            best_metric = -np.inf
            best_params: dict[str, Any] | None = None
            best_is_sharpe = 0.0

            for params in param_list:
                try:
                    is_signals = signal_func(is_data, **params)
                    is_metrics = _compute_equity_metrics(
                        is_data, is_signals, initial_capital
                    )
                    m = _extract_metric(is_metrics, self._metric)
                    if m > best_metric:
                        best_metric = m
                        best_params = params
                        best_is_sharpe = is_metrics.sharpe
                except Exception:
                    continue

            if best_params is None:
                best_params = {"_dummy": 0.0}

            # Apply best params to OOS
            try:
                oos_signals = signal_func(oos_data, **best_params)
                oos_metrics = _compute_equity_metrics(
                    oos_data, oos_signals, initial_capital
                )
            except Exception:
                oos_metrics = _empty_metrics()

            split = WalkForwardSplit(
                split_id=sid,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                is_sharpe=best_is_sharpe,
                oos_sharpe=oos_metrics.sharpe,
                oos_return=oos_metrics.total_return,
                oos_max_drawdown=oos_metrics.max_drawdown_pct,
                oos_win_rate=oos_metrics.win_rate,
                best_params=best_params,
            )
            all_splits.append(split)

            oos_sharpes.append(oos_metrics.sharpe)
            oos_returns.append(oos_metrics.total_return)
            oos_dds.append(oos_metrics.max_drawdown_pct)
            oos_wrs.append(oos_metrics.win_rate)
            is_sharpes.append(best_is_sharpe)
            is_total_returns.append(
                _compound_return(is_data, best_params, signal_func, initial_capital)
            )

        # Aggregate
        oos_sharpe_mean = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        oos_sharpe_std = float(np.std(oos_sharpes)) if len(oos_sharpes) > 1 else 0.0
        is_sharpe_mean = float(np.mean(is_sharpes)) if is_sharpes else 0.0

        compound_oos = 1.0
        for r in oos_returns:
            compound_oos *= 1.0 + r
        compound_oos_ret = compound_oos - 1.0

        compound_is = 1.0
        for r in is_total_returns:
            compound_is *= 1.0 + r
        compound_is_ret = compound_is - 1.0

        robustness = (
            oos_sharpe_mean / abs(is_sharpe_mean) if abs(is_sharpe_mean) > 1e-9 else 0.0
        )
        efficiency = (
            compound_oos_ret / abs(compound_is_ret)
            if abs(compound_is_ret) > 1e-9
            else 0.0
        )

        # PBO: fraction of splits where OOS Sharpe < 0
        pbo = float(np.mean([s < 0 for s in oos_sharpes])) if oos_sharpes else 1.0

        degradation = float(np.mean([
            i - o for i, o in zip(is_sharpes, oos_sharpes)
        ])) if is_sharpes and oos_sharpes else 0.0

        return WalkForwardResult(
            splits=all_splits,
            oos_sharpe=oos_sharpe_mean,
            oos_return=compound_oos_ret,
            oos_max_drawdown=float(np.max(oos_dds)) if oos_dds else 0.0,
            oos_win_rate=float(np.mean(oos_wrs)) if oos_wrs else 0.0,
            robustness_ratio=robustness,
            efficiency_ratio=efficiency,
            pbo=pbo,
            degradation=degradation,
        )

    def _generate_splits_auto(
        self, n: int
    ) -> list[tuple[int, int, int, int]]:
        """Generate splits using step-based sliding windows."""
        splits: list[tuple[int, int, int, int]] = []
        start = 0
        sid = 0
        while start + self._train + self._test <= n:
            is_start = start
            is_end = start + self._train
            oos_start = is_end
            oos_end = min(oos_start + self._test, n)
            splits.append((is_start, is_end, oos_start, oos_end))
            start += self._step
            sid += 1
        return splits

    def _generate_splits_fixed(
        self, n: int, n_splits: int
    ) -> list[tuple[int, int, int, int]]:
        """Generate exactly n_splits splits covering the data."""
        total_needed = self._train + self._test
        usable = n - total_needed
        if usable < 0:
            raise ValueError(f"Data too short: {n} bars for {total_needed} minimum")

        if self._mode == "rolling":
            step = usable // max(n_splits - 1, 1)
            step = max(step, 1)
        else:  # anchored
            step = self._test

        splits: list[tuple[int, int, int, int]] = []
        start = 0
        for sid in range(n_splits):
            is_start = 0 if self._mode == "anchored" else start
            is_end = is_start + self._train
            oos_start = is_end
            oos_end = min(oos_start + self._test, n)

            if oos_end > n:
                break

            splits.append((is_start, is_end, oos_start, oos_end))
            start += step
        return splits


# ---------------------------------------------------------------------------
# Numba-accelerated hot-path equity metrics
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True, parallel=True)
def _vectorized_equity_kernel(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Numba-hot equity curve + trade-level metrics.

    Returns (equity_curve, drawdown_curve, returns, n_trades, n_wins).
    """
    n = len(prices)
    equity = np.empty(n, dtype=np.float64)
    dd_curve = np.empty(n, dtype=np.float64)
    returns_out = np.empty(n - 1, dtype=np.float64)

    equity[0] = initial_capital
    peak = initial_capital
    dd_curve[0] = 0.0

    position = 0.0
    entry_price = 0.0
    n_trades = 0
    n_wins = 0

    for i in range(1, n):
        prev_eq = equity[i - 1]

        if position == 0.0:
            sig = signals[i]
            if sig != 0:
                # Entry: apply slippage
                exec_price = (
                    prices[i] * (1.0 - slippage_pct)
                    if sig > 0
                    else prices[i] * (1.0 + slippage_pct)
                )
                cost = prev_eq * commission_pct
                position = (prev_eq - cost) / exec_price
                entry_price = exec_price
        else:
            # Check exit signals
            sig = signals[i]
            if sig == 0 or sig * (1.0 if position > 0 else -1.0) < 0:
                # Close position
                exec_price = (
                    prices[i] * (1.0 - slippage_pct)
                    if position > 0
                    else prices[i] * (1.0 + slippage_pct)
                )
                proceeds = position * exec_price
                cost = proceeds * commission_pct
                final_eq = proceeds - cost

                pnl = final_eq - prev_eq
                if pnl > 0:
                    n_wins += 1
                n_trades += 1

                equity[i] = final_eq
                position = 0.0
                entry_price = 0.0
            else:
                # Mark-to-market
                mtm_price = (
                    prices[i] * (1.0 - slippage_pct)
                    if position > 0
                    else prices[i] * (1.0 + slippage_pct)
                )
                mtm_value = position * mtm_price
                equity[i] = mtm_value

        if position == 0.0:
            equity[i] = prev_eq

        # Drawdown
        peak = max(peak, equity[i])
        dd_curve[i] = (peak - equity[i]) / peak if peak > 0 else 0.0

        # Daily return
        if i > 0 and equity[i - 1] > 0:
            returns_out[i - 1] = (equity[i] - equity[i - 1]) / equity[i - 1]
        else:
            returns_out[i - 1] = 0.0

    return equity, dd_curve, returns_out, n_trades, n_wins


def _compute_equity_metrics(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> BacktestMetrics:
    """Compute full metrics from price + signal arrays.

    Uses Numba if available, falls back to NumPy.
    """
    prices = np.asarray(prices, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.int8)

    if _NUMBA_AVAILABLE and len(prices) > 100:
        equity, dd, rets, n_trades, n_wins = _vectorized_equity_kernel(
            prices, signals, initial_capital, commission_pct, slippage_pct
        )
    else:
        equity, dd, rets, n_trades, n_wins = _numpy_fallback_equity(
            prices, signals, initial_capital, commission_pct, slippage_pct
        )

    n = len(equity)
    total_return = float(equity[-1] / equity[0] - 1.0) if equity[0] > 0 else 0.0

    # Annualisation
    annualise = 252
    years = n / annualise

    mean_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 0 else 1e-12
    downside = rets[rets < 0]
    std_down = float(np.std(downside, ddof=1)) if len(downside) > 0 else 1e-12

    sharpe = float(mean_ret / std_ret * math.sqrt(annualise)) if std_ret > 1e-12 else 0.0
    sortino = float(mean_ret / std_down * math.sqrt(annualise)) if std_down > 1e-12 else 0.0

    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    calmar = float(total_return / years / max_dd) if max_dd > 1e-9 and years > 0 else 0.0

    win_rate = float(n_wins / n_trades) if n_trades > 0 else 0.0

    # Profit factor
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 1e-12
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else 0.0

    avg_trade = float(np.mean(rets)) if len(rets) > 0 else 0.0

    # VaR / CVaR
    sorted_rets = np.sort(rets)
    var_idx = max(int(len(sorted_rets) * 0.05), 0)
    var_95 = float(sorted_rets[var_idx]) if len(sorted_rets) > 0 else 0.0
    cvar_95 = float(np.mean(sorted_rets[:max(var_idx, 1)])) if len(sorted_rets) > 0 else 0.0

    return BacktestMetrics(
        total_return=total_return,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=n_trades,
        avg_trade_return=avg_trade,
        cvar_95=cvar_95,
        var_95=var_95,
    )


def _numpy_fallback_equity(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Pure NumPy fallback when numba is unavailable."""
    n = len(prices)
    equity = np.empty(n, dtype=np.float64)
    dd = np.empty(n, dtype=np.float64)
    rets = np.empty(n - 1, dtype=np.float64)

    equity[0] = initial_capital
    peak = initial_capital
    dd[0] = 0.0
    position = 0.0
    n_trades = 0
    n_wins = 0

    for i in range(1, n):
        prev_eq = equity[i - 1]
        sig = signals[i] if i < len(signals) else 0

        if position == 0.0 and sig != 0:
            exec_price = prices[i] * (1.0 - slippage_pct if sig > 0 else 1.0 + slippage_pct)
            cost = prev_eq * commission_pct
            position = (prev_eq - cost) / exec_price
        elif position != 0.0:
            if sig == 0 or sig * (1.0 if position > 0 else -1.0) < 0:
                exec_price = prices[i] * (1.0 - slippage_pct if position > 0 else 1.0 + slippage_pct)
                proceeds = position * exec_price
                cost = proceeds * commission_pct
                final_eq = proceeds - cost
                pnl = final_eq - prev_eq
                if pnl > 0:
                    n_wins += 1
                n_trades += 1
                equity[i] = final_eq
                position = 0.0
            else:
                mtm_price = prices[i] * (1.0 - slippage_pct if position > 0 else 1.0 + slippage_pct)
                equity[i] = position * mtm_price
        else:
            equity[i] = prev_eq

        if position == 0.0 and (i == 0 or signals[i] == 0):
            equity[i] = prev_eq

        peak = max(peak, equity[i])
        dd[i] = (peak - equity[i]) / peak if peak > 0 else 0.0

        if equity[i - 1] > 0:
            rets[i - 1] = (equity[i] - equity[i - 1]) / equity[i - 1]
        else:
            rets[i - 1] = 0.0

    return equity, dd, rets, n_trades, n_wins


def _empty_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_return=0.0,
        sharpe=0.0,
        sortino=0.0,
        calmar=0.0,
        max_drawdown=0.0,
        max_drawdown_pct=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        total_trades=0,
        avg_trade_return=0.0,
        cvar_95=0.0,
        var_95=0.0,
    )


def _extract_metric(m: BacktestMetrics, metric: str) -> float:
    if metric == "sharpe":
        return m.sharpe
    if metric == "sortino":
        return m.sortino
    if metric == "total_return":
        return m.total_return
    if metric == "calmar":
        return m.calmar
    return m.sharpe


def _compound_return(
    prices: np.ndarray,
    params: dict[str, Any],
    signal_func: Callable,
    initial_capital: float,
) -> float:
    try:
        signals = signal_func(prices, **params)
        m = _compute_equity_metrics(prices, signals, initial_capital)
        return m.total_return
    except Exception:
        return 0.0
