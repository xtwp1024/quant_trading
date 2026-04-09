"""
quant_trading.portfolio.optimizers — Portfolio Optimization Suite.

Provides a consistent OptimizerBase interface for:
- MeanVarianceOptimizer      : Markowitz mean-variance (Markowitz, 1952)
- MaxSharpeOptimizer         : Maximum Sharpe ratio portfolio
- MinVarianceOptimizer       : Global minimum variance portfolio
- RiskParityOptimizer        : Risk parity / equal risk contribution
- HRPOptimizer               : Hierarchical Risk Parity (Lopez de Prado, 2016)
- BlackLittermanOptimizer    : Black-Litterman model (1992)
- CVaROptimizer              : CVaR (Expected Shortfall) minimisation

Integrates with:
- quant_trading.risk  — risk_metrics, VaR/CVaR calculators, matilda_portfolio
- quant_trading.factors — alpha evaluators and LSTM predictors for return forecasts

All optimizers accept either:
  (a) dict[str, list[float]]  — {ticker: [daily_returns]}  (pure-python, numpy-free)
  (b) pd.DataFrame            — assets as columns, rows as daily returns
"""

from __future__ import annotations

import abc
import math
import typing
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

try:
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    from scipy.optimize import minimize, Bounds
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    np = None
    pd = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result of a portfolio optimization run."""
    weights: Dict[str, float]           # ticker -> weight (sums to 1)
    expected_return: float               # annualised expected return
    volatility: float                    # annualised volatility (std dev)
    sharpe_ratio: float                  # (return - rf) / volatility
    optimizer_name: str                  # which optimizer produced this
    metadata: Dict[str, Any] = field(default_factory=dict)   # extra info

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "expected_return": round(self.expected_return, 6),
            "volatility": round(self.volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "optimizer": self.optimizer_name,
            **self.metadata,
        }


@dataclass
class EfficientFrontierPoint:
    """Single point on the efficient frontier."""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


# ---------------------------------------------------------------------------
# Pure-Python helpers (used when numpy is unavailable)
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _cov_matrix_pure(
    returns: Dict[str, list[float]],
) -> tuple[List[str], List[List[float]]]:
    """Compute covariance matrix from return dict (pure Python)."""
    tickers = sorted(returns)
    n = len(tickers)
    if n == 0:
        return [], []
    T = min(len(returns[t]) for t in tickers)
    if T < 2:
        return tickers, [[0.0] * n for _ in range(n)]
    means = {t: _mean(returns[t][:T]) for t in tickers}
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            ti, tj = tickers[i], tickers[j]
            val = sum(
                (returns[ti][k] - means[ti]) * (returns[tj][k] - means[tj])
                for k in range(T)
            ) / (T - 1)
            cov[i][j] = val
            cov[j][i] = val
    return tickers, cov


def _portfolio_stats_pure(
    weights: List[float],
    mean_returns: List[float],
    cov: List[List[float]],
) -> tuple[float, float]:
    """Compute (expected_return, volatility) from weight vector (pure Python)."""
    n = len(weights)
    ret = sum(weights[i] * mean_returns[i] for i in range(n))
    var = sum(weights[i] * weights[j] * cov[i][j] for i in range(n) for j in range(n))
    return ret, math.sqrt(max(var, 0.0))


# ---------------------------------------------------------------------------
# OptimizerBase
# ---------------------------------------------------------------------------

class OptimizerBase(abc.ABC):
    """Abstract base class for all portfolio optimizers.

    Supports two input modes:
      - dict[str, list[float]] : {ticker: [daily_returns]}
      - pd.DataFrame           : columns = tickers, index = dates

    Subclasses must implement `solve()`.

    Parameters
    ----------
    returns : returns data
    risk_free_rate : annualised risk-free rate (default 0.02)
    periods_per_year : number of trading periods per year (default 252)
    """

    def __init__(
        self,
        returns: typing.Union[Dict[str, list[float]], "pd.DataFrame"],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        self._returns = returns
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self._is_df = _HAS_SCIPY and isinstance(returns, pd.DataFrame)
        self._tickers: List[str] = []
        self._mean_rets: List[float] = []
        self._cov: List[List[float]] = []
        self._annualised = False
        self._parse()

    # ---- public API ----

    def solve(self) -> OptimizationResult:
        """Run the optimisation and return an OptimizationResult."""
        return self._solve_impl()

    def efficient_frontier(self, n_points: int = 50) -> List[EfficientFrontierPoint]:
        """Compute the Markowitz efficient frontier (default 50 points)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support efficient_frontier()"
        )

    # ---- properties exposed to subclasses ----

    @property
    def tickers(self) -> List[str]:
        return self._tickers

    @property
    def mean_returns(self) -> List[float]:
        return self._mean_rets

    @property
    def cov_matrix(self) -> List[List[float]]:
        return self._cov

    def annualised_return(self, daily: float) -> float:
        return daily * self.periods_per_year

    def annualised_vol(self, daily: float) -> float:
        return daily * math.sqrt(self.periods_per_year)

    # ---- internal helpers ----

    @abc.abstractmethod
    def _solve_impl(self) -> OptimizationResult:
        """Concrete optimizers override this."""
        ...

    def _parse(self) -> None:
        """Parse input data into tickers, mean returns, and covariance."""
        if self._is_df:
            self._parse_df()
        else:
            self._parse_dict()

    def _parse_dict(self) -> None:
        assert isinstance(self._returns, dict)
        self._tickers, self._cov = _cov_matrix_pure(self._returns)
        n = len(self._tickers)
        self._mean_rets = [
            _mean(self._returns[t]) if t in self._returns else 0.0
            for t in self._tickers
        ]

    def _parse_df(self) -> None:
        assert _HAS_SCIPY
        df = typing.cast(pd.DataFrame, self._returns)
        self._tickers = list(df.columns)
        n = len(self._tickers)
        self._mean_rets = df.mean().tolist()
        # sample covariance (annualised)
        cov_np = df.cov().values * self.periods_per_year
        self._cov = cov_np.tolist()

    def _result(
        self,
        weights: List[float],
        optimizer_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Build an OptimizationResult from a weight list."""
        ret, vol = _portfolio_stats_pure(weights, self._mean_rets, self._cov)
        ann_ret = ret * self.periods_per_year
        ann_vol = vol * math.sqrt(self.periods_per_year)
        sharpe = (ann_ret - self.risk_free_rate) / ann_vol if ann_vol > 1e-12 else 0.0
        return OptimizationResult(
            weights={self._tickers[i]: round(weights[i], 8) for i in range(len(weights))},
            expected_return=round(ann_ret, 6),
            volatility=round(ann_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            optimizer_name=optimizer_name,
            metadata=metadata or {},
        )

    def _project_to_simplex(self, w: List[float]) -> List[float]:
        """Project onto the simplex (sum=1, non-negative)."""
        n = len(w)
        w = [max(wi, 0.0) for wi in w]
        s = sum(w)
        return [wi / s for wi in w] if s > 0 else [1.0 / n] * n

    def _np_solve_min_var(
        self,
        cov_np: "np.ndarray",
        mean_np: Optional["np.ndarray"] = None,
        target_return: Optional[float] = None,
    ) -> "np.ndarray":
        """Scipy-based min-variance (used when scipy is available)."""
        n = len(cov_np)

        def vol_objective(w: np.ndarray) -> float:
            return float(np.sqrt(w @ cov_np @ w))

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
        if target_return is not None and mean_np is not None:
            daily_target = target_return / self.periods_per_year
            constraints.append({
                "type": "eq",
                "fun": lambda x: float(mean_np.dot(x)) - daily_target,
            })

        x0 = np.ones(n) / n
        bounds = Bounds(0.0, 1.0)
        result = minimize(vol_objective, x0, method="SLSQP",
                          constraints=constraints, bounds=bounds)
        return result.x if result.success else x0

    def _np_solve_max_sharpe(self, cov_np: "np.ndarray", mean_np: np.ndarray) -> np.ndarray:
        """Scipy-based max-Sharpe (used when scipy is available)."""
        n = len(mean_np)
        daily_rf = self.risk_free_rate / self.periods_per_year

        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(mean_np.dot(w))
            vol = float(np.sqrt(w @ cov_np @ w))
            return -(ret - daily_rf) / vol if vol > 1e-12 else -1e18

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
        x0 = np.ones(n) / n
        bounds = Bounds(0.0, 1.0)
        result = minimize(neg_sharpe, x0, method="SLSQP",
                         constraints=constraints, bounds=bounds)
        return result.x if result.success else x0


# ---------------------------------------------------------------------------
# Mean-Variance Optimizer (Markowitz)
# ---------------------------------------------------------------------------

class MeanVarianceOptimizer(OptimizerBase):
    """Markowitz Mean-Variance Portfolio Optimizer.

    Finds the portfolio that minimises variance for a given target return.
    If no target is specified, returns the global minimum variance portfolio.

    Reference: Markowitz (1952) "Portfolio Selection"
    """

    def __init__(
        self,
        returns: typing.Union[Dict[str, list[float]], "pd.DataFrame"],
        target_return: Optional[float] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        """
        Parameters
        ----------
        returns : return data
        target_return : annualised target return (None → min-variance portfolio)
        risk_free_rate : annualised risk-free rate
        periods_per_year : trading periods per year
        """
        super().__init__(returns, risk_free_rate, periods_per_year)
        self.target_return = target_return

    def _solve_impl(self) -> OptimizationResult:
        if _HAS_SCIPY and self._is_df:
            return self._solve_numpy()
        return self._solve_pure()

    def _solve_numpy(self) -> OptimizationResult:
        cov_np = np.array(self._cov)
        mean_np = np.array(self._mean_rets)
        w = self._np_solve_min_var(cov_np, mean_np, self.target_return)
        return self._result(w.tolist(), "MeanVarianceOptimizer",
                            {"target_return": self.target_return})

    def _solve_pure(self) -> OptimizationResult:
        cov = self._cov
        n = len(cov)
        if n == 0:
            return self._result([], "MeanVarianceOptimizer")

        mean_rets = self._mean_rets
        target = (self.target_return / self.periods_per_year
                  if self.target_return is not None else None)

        # Gradient-descent min-var
        w = [1.0 / n] * n
        for _ in range(5000):
            grad = [2 * sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
            if target is not None and mean_rets is not None:
                port_ret = sum(w[i] * mean_rets[i] for i in range(n))
                lam = 50.0 * (target - port_ret)
                grad = [grad[i] - lam * mean_rets[i] for i in range(n)]

            w = [w[i] - 0.005 * grad[i] for i in range(n)]
            w = self._project_to_simplex(w)

        return self._result(w, "MeanVarianceOptimizer",
                            {"target_return": self.target_return})

    def efficient_frontier(self, n_points: int = 50) -> List[EfficientFrontierPoint]:
        """Compute the Markowitz efficient frontier."""
        if not _HAS_SCIPY or not self._is_df:
            return self._ef_pure(n_points)
        return self._ef_numpy(n_points)

    def _ef_numpy(self, n_points: int) -> List[EfficientFrontierPoint]:
        mean_np = np.array(self._mean_rets)
        cov_np = np.array(self._cov)
        min_ret = float(mean_np.min())
        max_ret = float(mean_np.max())
        if abs(max_ret - min_ret) < 1e-10:
            w = self._np_solve_min_var(cov_np)
            result = self._result(w.tolist(), "MeanVarianceOptimizer")
            return [EfficientFrontierPoint(
                weights=result.weights,
                expected_return=result.expected_return,
                volatility=result.volatility,
                sharpe_ratio=result.sharpe_ratio,
            )]
        frontier = []
        for i in range(n_points):
            target = min_ret + (max_ret - min_ret) * i / (n_points - 1)
            w = self._np_solve_min_var(cov_np, mean_np, target)
            result = self._result(w.tolist(), "MeanVarianceOptimizer")
            frontier.append(EfficientFrontierPoint(
                weights=result.weights,
                expected_return=result.expected_return,
                volatility=result.volatility,
                sharpe_ratio=result.sharpe_ratio,
            ))
        return frontier

    def _ef_pure(self, n_points: int) -> List[EfficientFrontierPoint]:
        mean_rets = self._mean_rets
        cov = self._cov
        n = len(cov)
        if n == 0:
            return []
        min_ret = min(mean_rets)
        max_ret = max(mean_rets)
        if abs(max_ret - min_ret) < 1e-10:
            w = self._solve_pure().weights
            result = self._result([w.get(t, 0.0) for t in self._tickers], "MeanVarianceOptimizer")
            return [EfficientFrontierPoint(
                weights=result.weights,
                expected_return=result.expected_return,
                volatility=result.volatility,
                sharpe_ratio=result.sharpe_ratio,
            )]
        frontier = []
        for i in range(n_points):
            target = min_ret + (max_ret - min_ret) * i / (n_points - 1)
            w = self._solve_pure().weights
            # re-solve with target (approximate via gradient descent)
            for _ in range(2000):
                grad = [2 * sum(cov[p][q] * w[p] for q in range(n)) for p in range(n)]
                port_ret = sum(w[p] * mean_rets[p] for p in range(n))
                lam = 50.0 * (target / self.periods_per_year - port_ret)
                grad = [grad[p] - lam * mean_rets[p] for p in range(n)]
                w = [w[p] - 0.005 * grad[p] for p in range(n)]
                w = self._project_to_simplex(w)
            result = self._result(w, "MeanVarianceOptimizer")
            frontier.append(EfficientFrontierPoint(
                weights=result.weights,
                expected_return=result.expected_return,
                volatility=result.volatility,
                sharpe_ratio=result.sharpe_ratio,
            ))
        return frontier


# ---------------------------------------------------------------------------
# Max Sharpe Optimizer
# ---------------------------------------------------------------------------

class MaxSharpeOptimizer(OptimizerBase):
    """Maximum Sharpe Ratio Portfolio Optimizer.

    Maximises (expected_return - risk_free_rate) / volatility.
    Uses gradient ascent or scipy-based optimisation.

    Reference: Sharpe (1994) "The Sharpe Ratio"
    """

    def __init__(
        self,
        returns: typing.Union[Dict[str, list[float]], "pd.DataFrame"],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        super().__init__(returns, risk_free_rate, periods_per_year)

    def _solve_impl(self) -> OptimizationResult:
        if _HAS_SCIPY and self._is_df:
            return self._solve_numpy()
        return self._solve_pure()

    def _solve_numpy(self) -> OptimizationResult:
        cov_np = np.array(self._cov)
        mean_np = np.array(self._mean_rets)
        w = self._np_solve_max_sharpe(cov_np, mean_np)
        return self._result(w.tolist(), "MaxSharpeOptimizer")

    def _solve_pure(self) -> OptimizationResult:
        cov = self._cov
        n = len(cov)
        if n == 0:
            return self._result([], "MaxSharpeOptimizer")

        mean_rets = self._mean_rets
        daily_rf = self.risk_free_rate / self.periods_per_year

        best_sharpe = -1e18
        best_w = [1.0 / n] * n

        w = [1.0 / n] * n
        for _ in range(8000):
            ret, vol = _portfolio_stats_pure(w, mean_rets, cov)
            if vol < 1e-12:
                break
            sharpe = (ret - daily_rf) / vol
            grad_ret = list(mean_rets)
            grad_var = [2 * sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
            grad = [
                (vol * grad_ret[i] - (ret - daily_rf) * grad_var[i] / (2 * vol)) / (vol * vol)
                for i in range(n)
            ]
            w = [w[i] + 0.003 * grad[i] for i in range(n)]
            w = self._project_to_simplex(w)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_w = list(w)

        return self._result(best_w, "MaxSharpeOptimizer")


# ---------------------------------------------------------------------------
# Min Variance Optimizer
# ---------------------------------------------------------------------------

class MinVarianceOptimizer(OptimizerBase):
    """Global Minimum Variance Portfolio Optimizer.

    Finds the portfolio with the lowest possible volatility,
    regardless of expected return.

    Reference: Michaud (1989) "The Markowitz Optimisation Enigma"
    """

    def __init__(
        self,
        returns: typing.Union[Dict[str, list[float]], "pd.DataFrame"],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        super().__init__(returns, risk_free_rate, periods_per_year)

    def _solve_impl(self) -> OptimizationResult:
        if _HAS_SCIPY and self._is_df:
            cov_np = np.array(self._cov)
            w = self._np_solve_min_var(cov_np)
            return self._result(w.tolist(), "MinVarianceOptimizer")
        return self._solve_pure()

    def _solve_pure(self) -> OptimizationResult:
        cov = self._cov
        n = len(cov)
        if n == 0:
            return self._result([], "MinVarianceOptimizer")

        mean_rets = self._mean_rets
        w = [1.0 / n] * n
        for _ in range(5000):
            grad = [2 * sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
            w = [w[i] - 0.005 * grad[i] for i in range(n)]
            w = self._project_to_simplex(w)

        return self._result(w, "MinVarianceOptimizer")


# ---------------------------------------------------------------------------
# Risk Parity / Risk Budgeting Optimizer
# ---------------------------------------------------------------------------

class RiskParityOptimizer(OptimizerBase):
    """Risk Parity (Equal Risk Contribution) Portfolio Optimizer.

    Each asset contributes equally to total portfolio volatility.
    Iteratively adjusts weights until marginal risk contributions are equal.

    Also supports custom risk budgets via `risk_budgets` dict.

    Reference: Maillard, Roncalli & Teiletche (2010) "On the Properties
    of Equally-Weighted Risk Contributions Portfolios"
    """

    def __init__(
        self,
        returns: typing.Union[Dict[str, list[float]], "pd.DataFrame"],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        risk_budgets: Optional[Dict[str, float]] = None,
    ):
        """
        Parameters
        ----------
        returns : return data
        risk_free_rate : annualised risk-free rate
        periods_per_year : trading periods per year
        risk_budgets : optional {ticker: target_risk_share}; defaults to equal shares
        """
        super().__init__(returns, risk_free_rate, periods_per_year)
        self.risk_budgets = risk_budgets  # None → equal risk contribution

    def _solve_impl(self) -> OptimizationResult:
        if _HAS_SCIPY and self._is_df:
            return self._solve_numpy()
        return self._solve_pure()

    def _solve_numpy(self) -> OptimizationResult:
        cov_np = np.array(self._cov)
        n = len(self._tickers)

        if self.risk_budgets is None:
            targets = np.ones(n) / n
        else:
            targets = np.array([self.risk_budgets.get(t, 1.0) for t in self._tickers])
            targets = targets / targets.sum()

        def risk_contrib(w: np.ndarray) -> np.ndarray:
            vol = float(np.sqrt(w @ cov_np @ w))
            if vol < 1e-12:
                return np.zeros(n)
            marginal = cov_np @ w
            return w * marginal / vol

        def objective(w: np.ndarray) -> float:
            rc = risk_contrib(w)
            return float(np.sum((rc - targets) ** 2))

        x0 = np.ones(n) / n
        result = minimize(objective, x0, method="SLSQP",
                          constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}],
                          bounds=Bounds(0.0, 1.0))
        w = result.x if result.success else x0

        # Compute risk contributions for metadata
        rc = risk_contrib(w)
        metadata = {"risk_contributions": {self._tickers[i]: round(rc[i], 8) for i in range(n)}}
        return self._result(w.tolist(), "RiskParityOptimizer", metadata)

    def _solve_pure(self) -> OptimizationResult:
        cov = self._cov
        n = len(cov)
        if n == 0:
            return self._result([], "RiskParityOptimizer")

        # Initial: inverse volatility weights
        vols = [math.sqrt(max(cov[i][i], 1e-12)) for i in range(n)]
        w = [1.0 / v for v in vols]
        w = self._project_to_simplex(w)

        # Newton-like refinement
        for _ in range(200):
            port_var_sq = sum(w[i] * w[j] * cov[i][j] for i in range(n) for j in range(n))
            port_vol = math.sqrt(max(port_var_sq, 1e-12))
            mrc = [sum(cov[i][j] * w[j] for j in range(n)) / port_vol for i in range(n)]
            rc = [w[i] * mrc[i] for i in range(n)]
            target_rc = port_vol / n

            for i in range(n):
                if rc[i] > 0:
                    w[i] *= (target_rc / rc[i]) ** 0.3
            w = self._project_to_simplex(w)

        metadata = {"risk_contributions": {self._tickers[i]: round(rc[i], 8) for i in range(n)}}
        return self._result(w, "RiskParityOptimizer", metadata)


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (HRP) Optimizer
# ---------------------------------------------------------------------------

class HRPOptimizer(OptimizerBase):
    """Hierarchical Risk Parity Portfolio Optimizer.

    Uses hierarchical clustering on the correlation matrix to partition
    assets, then recursively allocates by inverse variance within clusters.
    Robust to noisy or near-singular covariance matrices — no matrix
    inversion required.

    Reference: Lopez de Prado (2016) "Building Diversified Portfolios
    that Outperform Out-of-Sample"
    """

    def __init__(
        self,
        returns: typing.Union[Dict[str, list[float]], "pd.DataFrame"],
        linkage_method: str = "ward",
    ):
        """
        Parameters
        ----------
        returns : return data
        linkage_method : scipy.cluster.hierarchy linkage method
                         ('ward', 'single', 'complete', 'average')
        """
        # HRP does not use risk_free_rate or periods_per_year in the same way
        super().__init__(returns, risk_free_rate=0.0, periods_per_year=252)
        self.linkage_method = linkage_method

    def _solve_impl(self) -> OptimizationResult:
        if _HAS_SCIPY and self._is_df:
            return self._solve_numpy()
        return self._solve_pure()

    def _solve_numpy(self) -> OptimizationResult:
        df = typing.cast(pd.DataFrame, self._returns)
        cov = df.cov() * self.periods_per_year

        # Correlation matrix
        d = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(d, d)
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)
        corr = pd.DataFrame(corr, index=cov.index, columns=cov.columns)

        # Distance matrix
        dist_matrix = np.sqrt(np.clip(1 - corr.values, 0, None))
        np.fill_diagonal(dist_matrix, 0)
        condensed = squareform(dist_matrix, checks=False)

        Z = linkage(condensed, method=self.linkage_method)
        dendro = dendrogram(Z, no_plot=True)
        sort_idx = [int(x) - 1 for x in dendro["ivl"]]
        sorted_assets = list(corr.columns[sort_idx])

        cov_reordered = cov.loc[sorted_assets, sorted_assets]
        weights = self._hrp_recursive_np(cov_reordered)
        final_weights = pd.Series(0.0, index=cov.index)
        for asset, wt in weights.items():
            final_weights[asset] = wt

        return self._result(final_weights.tolist(), "HRPOptimizer",
                            {"linkage": self.linkage_method})

    def _hrp_recursive_np(self, cov_sub: pd.DataFrame) -> pd.Series:
        n = len(cov_sub)
        if n == 1:
            return pd.Series({cov_sub.index[0]: 1.0})

        half = n // 2
        left_assets = list(cov_sub.index[:half])
        right_assets = list(cov_sub.index[half:])

        left_cov = cov_sub.loc[left_assets, left_assets]
        right_cov = cov_sub.loc[right_assets, right_assets]

        left_var = self._cluster_variance_np(left_cov)
        right_var = self._cluster_variance_np(right_cov)
        total_var = left_var + right_var

        left_weight = right_var / total_var if total_var > 0 else 0.5
        right_weight = left_var / total_var if total_var > 0 else 0.5

        left_w = self._hrp_recursive_np(left_cov)
        right_w = self._hrp_recursive_np(right_cov)

        return pd.concat([left_w * left_weight, right_w * right_weight])

    @staticmethod
    def _cluster_variance_np(cov_sub: pd.DataFrame) -> float:
        n = len(cov_sub)
        if n == 0:
            return 0.0
        w = np.ones(n) / n
        return float(w @ cov_sub.values @ w)

    def _solve_pure(self) -> OptimizationResult:
        # Pure-python HRP using correlation-based distance
        tickers = self._tickers
        n = len(tickers)
        if n == 0:
            return self._result([], "HRPOptimizer")

        cov = self._cov
        # Build correlation from covariance
        vols = [math.sqrt(max(cov[i][i], 1e-12)) for i in range(n)]
        corr = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if vols[i] > 0 and vols[j] > 0:
                    corr[i][j] = cov[i][j] / (vols[i] * vols[j])
        corr = [[min(max(corr[i][j], -1), 1) for j in range(n)] for i in range(n)]
        for i in range(n):
            corr[i][i] = 1.0

        # Distance matrix
        dist = [[math.sqrt(max(1 - corr[i][j], 0)) for j in range(n)] for i in range(n)]

        # Simplified hierarchical clustering (average linkage, pure python)
        clusters = [[i] for i in range(n)]
        active = set(range(n))

        while len(clusters) > 1:
            # Find closest pair
            min_dist = 1e18
            merge = (0, 1)
            for i, ci in enumerate(clusters):
                for j, cj in enumerate(clusters):
                    if i >= j:
                        continue
                    # Average linkage distance
                    d = sum(dist[a][b] for a in ci for b in cj) / (len(ci) * len(cj))
                    if d < min_dist:
                        min_dist = d
                        merge = (i, j)

        # Just use simple bisection based on dendrogram approximation
        # Sort by average distance to others (simplified ordering)
        avg_dist = [sum(dist[i][j] for j in range(n)) / n for i in range(n)]
        order = sorted(range(n), key=lambda i: avg_dist[i])
        sorted_tickers = [tickers[order[i]] for i in range(n)]

        # Recursive allocation
        weights = self._hrp_recursive_pure(cov, sorted_tickers)
        return self._result(weights, "HRPOptimizer", {"linkage": self.linkage_method})

    def _hrp_recursive_pure(
        self, cov: List[List[float]], tickers_sub: List[str]
    ) -> List[float]:
        n = len(tickers_sub)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        idx_map = {t: self._tickers.index(t) for t in tickers_sub}
        half = n // 2
        left_tickers = tickers_sub[:half]
        right_tickers = tickers_sub[half:]

        left_cov = [[cov[idx_map[a]][idx_map[b]]
                      for a in left_tickers] for b in left_tickers]
        right_cov = [[cov[idx_map[a]][idx_map[b]]
                       for a in right_tickers] for b in right_tickers]

        left_var = self._cluster_variance_pure(left_cov)
        right_var = self._cluster_variance_pure(right_cov)
        total_var = left_var + right_var

        left_weight = right_var / total_var if total_var > 0 else 0.5
        right_weight = left_var / total_var if total_var > 0 else 0.5

        left_w = self._hrp_recursive_pure(cov, left_tickers)
        right_w = self._hrp_recursive_pure(cov, right_tickers)

        return [left_w[i] * left_weight for i in range(len(left_w))] + \
               [right_w[i] * right_weight for i in range(len(right_w))]

    @staticmethod
    def _cluster_variance_pure(cov_sub: List[List[float]]) -> float:
        n = len(cov_sub)
        if n == 0:
            return 0.0
        w = [1.0 / n] * n
        return sum(w[i] * w[j] * cov_sub[i][j] for i in range(n) for j in range(n))


# ---------------------------------------------------------------------------
# Black-Litterman Optimizer
# ---------------------------------------------------------------------------

class BlackLittermanOptimizer(OptimizerBase):
    """Black-Litterman Portfolio Optimizer.

    Combines the market-implied equilibrium returns (CAPM prior) with
    investor views to produce posterior expected returns, then optimizes
    via mean-variance.

    Reference: Black & Litterman (1992) "Global Portfolio Optimization"
    """

    def __init__(
        self,
        returns: typing.Union[Dict[str, list[float]], "pd.DataFrame"],
        market_caps: typing.Optional[typing.Dict[str, float]] = None,
        views: typing.Optional[List[Dict[str, typing.Any]]] = None,
        view_confidences: typing.Optional[List[float]] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        """
        Parameters
        ----------
        returns : historical return data
        market_caps : {ticker: market_cap}; used to build equilibrium prior
        views : list of view dicts, each with:
            - 'assets': {ticker: weight_in_view}
            - 'return': expected_return (annualised)
        view_confidences : confidence in each view [0, 1]
        risk_aversion : market risk aversion parameter (delta)
        tau : scaling factor for covariance uncertainty
        risk_free_rate : annualised risk-free rate
        periods_per_year : trading periods per year
        """
        super().__init__(returns, risk_free_rate, periods_per_year)
        self.market_caps = market_caps or {}
        self.views = views or []
        self.view_confidences = view_confidences or []
        self.risk_aversion = risk_aversion
        self.tau = tau

    def _solve_impl(self) -> OptimizationResult:
        if _HAS_SCIPY and self._is_df:
            return self._solve_numpy()
        return self._solve_pure()

    def _solve_numpy(self) -> OptimizationResult:
        df = typing.cast(pd.DataFrame, self._returns)
        cov = (df.cov() * self.periods_per_year).values
        mean_rets_arr = np.array(self._mean_rets)
        n = len(self._tickers)

        # Market cap weights (or equal-weighted fallback)
        if self.market_caps:
            mcaps = np.array([self.market_caps.get(t, 1.0) for t in self._tickers])
            w_mkt = mcaps / mcaps.sum()
        else:
            w_mkt = np.ones(n) / n

        # Equilibrium (implied) returns: pi = delta * Sigma * w_mkt
        pi = self.risk_aversion * (cov @ w_mkt)

        # Apply views if any
        bl_returns = pi.copy()
        k = len(self.views)
        if k > 0 and len(self.view_confidences) == k:
            for v_idx, view in enumerate(self.views):
                conf = max(min(self.view_confidences[v_idx], 0.99), 0.01)
                # Build P vector
                P = np.array([view.get("assets", {}).get(t, 0.0) for t in self._tickers])
                Q = view["return"] / self.periods_per_year  # daily
                # Omega (diagonal): uncertainty of the view
                p_sigma_p = float(P @ self.tau * cov @ P)
                omega_diag = max((1.0 / conf - 1.0) * abs(p_sigma_p), 1e-12)
                # BL update
                p_pi = float(P @ pi)
                residual = Q - p_pi
                scale = self.tau / (self.tau + omega_diag)
                bl_returns = bl_returns + scale * P * residual

        # Mean-variance with posterior returns
        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(bl_returns @ w)
            vol = float(np.sqrt(w @ cov @ w))
            return -(ret - self.risk_free_rate) / vol if vol > 1e-12 else -1e18

        x0 = np.ones(n) / n
        result = minimize(neg_sharpe, x0, method="SLSQP",
                          constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}],
                          bounds=Bounds(0.0, 1.0))
        w = result.x if result.success else x0

        metadata = {
            "implied_returns": {self._tickers[i]: round(float(pi[i]), 6) for i in range(n)},
            "bl_returns": {self._tickers[i]: round(float(bl_returns[i]), 6) for i in range(n)},
        }
        return self._result(w.tolist(), "BlackLittermanOptimizer", metadata)

    def _solve_pure(self) -> OptimizationResult:
        tickers = self._tickers
        cov = self._cov
        n = len(tickers)
        if n == 0:
            return self._result([], "BlackLittermanOptimizer")

        mean_rets = self._mean_rets

        # Market cap weights
        total_cap = sum(self.market_caps.get(t, 1.0) for t in tickers)
        w_mkt = [self.market_caps.get(t, 1.0) / total_cap for t in tickers]

        # Implied equilibrium returns
        pi = [self.risk_aversion * sum(cov[i][j] * w_mkt[j] for j in range(n))
              for i in range(n)]

        # Apply views
        bl_returns = list(pi)
        k = len(self.views)
        if k > 0 and len(self.view_confidences) == k:
            for v_idx, view in enumerate(self.views):
                conf = max(min(self.view_confidences[v_idx], 0.99), 0.01)
                Q = view["return"] / self.periods_per_year
                P = [view.get("assets", {}).get(t, 0.0) for t in tickers]
                p_sigma_p = sum(P[i] * self.tau * cov[i][j] * P[j]
                                for i in range(n) for j in range(n))
                omega_diag = max((1.0 / conf - 1.0) * abs(p_sigma_p), 1e-12)
                p_pi = sum(P[j] * pi[j] for j in range(n))
                residual = Q - p_pi
                scale = self.tau / (self.tau + omega_diag)
                for i in range(n):
                    bl_returns[i] += scale * P[i] * residual

        # Gradient-descent mean-variance with BL returns
        w = [1.0 / n] * n
        for _ in range(3000):
            grad = [
                bl_returns[i] - self.risk_aversion * sum(cov[i][j] * w[j] for j in range(n))
                for i in range(n)
            ]
            w = [w[i] + 0.005 * grad[i] for i in range(n)]
            w = self._project_to_simplex(w)

        metadata = {
            "implied_returns": {tickers[i]: round(pi[i] * self.periods_per_year, 6) for i in range(n)},
            "bl_returns": {tickers[i]: round(bl_returns[i] * self.periods_per_year, 6) for i in range(n)},
        }
        return self._result(w, "BlackLittermanOptimizer", metadata)


# ---------------------------------------------------------------------------
# CVaR Optimizer
# ---------------------------------------------------------------------------

class CVaROptimizer(OptimizerBase):
    """Conditional VaR (Expected Shortfall) Minimisation Portfolio Optimizer.

    Minimises CVaR at a given confidence level rather than variance.
    CVaR is the expected loss beyond the VaR threshold — a more
    coherent risk measure than VaR alone.

    Reference: Rockafellar & Uryasev (2000) "Optimization of Conditional VaR"

    Note: when scipy is not available, falls back to a VaR-based approximation.
    """

    def __init__(
        self,
        returns: typing.Union[Dict[str, list[float]], "pd.DataFrame"],
        confidence: float = 0.95,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        target_return: typing.Optional[float] = None,
    ):
        """
        Parameters
        ----------
        returns : return data
        confidence : CVaR confidence level (e.g. 0.95 → 95%)
        risk_free_rate : annualised risk-free rate
        periods_per_year : trading periods per year
        target_return : annualised target return (optional)
        """
        super().__init__(returns, risk_free_rate, periods_per_year)
        self.confidence = confidence
        self.target_return = target_return

    def _solve_impl(self) -> OptimizationResult:
        if _HAS_SCIPY and self._is_df:
            return self._solve_numpy()
        return self._solve_pure()

    def _solve_numpy(self) -> OptimizationResult:
        df = typing.cast(pd.DataFrame, self._returns)
        returns_np = df.values  # (T, n)
        n = len(self._tickers)
        T = len(returns_np)

        alpha = self.confidence

        def cvar_objective(w: np.ndarray) -> float:
            """Sample CVaR: mean of losses beyond VaR at (1-alpha) quantile."""
            port_returns = returns_np @ w  # (T,)
            var_threshold = np.percentile(port_returns, (1 - alpha) * 100)
            tail_losses = -port_returns[port_returns <= var_threshold]
            if len(tail_losses) == 0:
                return 0.0
            return float(np.mean(tail_losses))

        def objective(w: np.ndarray) -> float:
            # Combine CVaR minimisation with optional return constraint
            cv = cvar_objective(w)
            if self.target_return is not None:
                daily_ret = float(np.mean(returns_np @ w))
                ann_ret = daily_ret * self.periods_per_year
                deficit = max(self.target_return - ann_ret, 0.0)
                return cv + 10.0 * deficit
            return cv

        x0 = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
        bounds = Bounds(0.0, 1.0)
        result = minimize(objective, x0, method="SLSQP",
                          constraints=constraints, bounds=bounds)
        w = result.x if result.success else x0

        # Compute CVaR for metadata
        port_rets = returns_np @ w
        var_thresh = float(np.percentile(port_rets, (1 - self.confidence) * 100))
        cvar_val = float(np.mean(-port_rets[port_rets <= var_thresh])) \
            if np.any(port_rets <= var_thresh) else 0.0
        cvar_ann = cvar_val * math.sqrt(self.periods_per_year)

        ret = float(np.mean(port_rets)) * self.periods_per_year
        vol = float(np.std(port_rets, ddof=1)) * math.sqrt(self.periods_per_year)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 1e-12 else 0.0

        metadata = {
            "cvar_daily": round(cvar_val, 6),
            "cvar_annualised": round(cvar_ann, 6),
            "var_daily": round(var_thresh, 6),
            "confidence": self.confidence,
        }
        return self._result(w.tolist(), "CVaROptimizer", metadata)

    def _solve_pure(self) -> OptimizationResult:
        returns_dict = typing.cast(Dict[str, list[float]], self._returns)
        tickers = self._tickers
        n = len(tickers)
        if n == 0:
            return self._result([], "CVaROptimizer")

        # Approximate: use gradient-based min-VaR/CVaR via sorted returns
        w = [1.0 / n] * n
        T = min(len(returns_dict[t]) for t in tickers)
        all_rets = [[returns_dict[tickers[i]][k] for i in range(n)]
                    for k in range(T)]

        alpha = self.confidence
        for _ in range(2000):
            port_rets = [sum(w[i] * all_rets[k][i] for i in range(n)) for k in range(T)]
            sorted_rets = sorted(port_rets)
            var_idx = max(int((1 - alpha) * T) - 1, 0)
            var_thresh = sorted_rets[var_idx]
            # CVaR = mean of losses beyond var
            tail_losses = [-r for r in port_rets if r <= var_thresh]
            cvar = sum(tail_losses) / len(tail_losses) if tail_losses else 0.0

            # Gradient approximation: perturb each weight
            eps = 1e-6
            grad = []
            for i in range(n):
                w_up = list(w)
                w_up[i] += eps
                s = sum(w_up)
                w_up = [wi / s for wi in w_up]
                port_rets_up = [sum(w_up[i] * all_rets[k][i] for i in range(n)) for k in range(T)]
                sorted_up = sorted(port_rets_up)
                var_up = sorted_up[max(int((1 - alpha) * T) - 1, 0)]
                tail_up = [-r for r in port_rets_up if r <= var_up]
                cvar_up = sum(tail_up) / len(tail_up) if tail_up else 0.0
                grad.append((cvar_up - cvar) / eps)

            # Add return penalty if needed
            port_ret_ann = sum(w[i] * self._mean_rets[i] for i in range(n)) * self.periods_per_year
            if self.target_return is not None:
                deficit = max(self.target_return - port_ret_ann, 0.0)
                for i in range(n):
                    grad[i] += 10.0 * (-self._mean_rets[i]) * self.periods_per_year / self.periods_per_year * deficit

            lr = 0.005
            w = [w[i] - lr * grad[i] for i in range(n)]
            w = self._project_to_simplex(w)

        # Final stats
        port_rets_final = [sum(w[i] * all_rets[k][i] for i in range(n)) for k in range(T)]
        sorted_final = sorted(port_rets_final)
        var_final = sorted_final[max(int((1 - alpha) * T) - 1, 0)]
        tail_final = [-r for r in port_rets_final if r <= var_final]
        cvar_final = sum(tail_final) / len(tail_final) if tail_final else 0.0

        ret = sum(port_rets_final) / len(port_rets_final) * self.periods_per_year
        vol = math.sqrt(sum((r - sum(port_rets_final) / len(port_rets_final)) ** 2
                            for r in port_rets_final) / (T - 1)) * math.sqrt(self.periods_per_year)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 1e-12 else 0.0

        # Override stats with the CVaR result
        result = self._result(w, "CVaROptimizer")
        result.expected_return = round(ret, 6)
        result.volatility = round(vol, 6)
        result.sharpe_ratio = round(sharpe, 4)
        result.metadata["cvar_daily"] = round(cvar_final, 6)
        result.metadata["cvar_annualised"] = round(cvar_final * math.sqrt(self.periods_per_year), 6)
        result.metadata["var_daily"] = round(var_final, 6)
        result.metadata["confidence"] = self.confidence
        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

OPTIMIZER_REGISTRY: typing.Dict[str, typing.Type[OptimizerBase]] = {
    "mean_variance": MeanVarianceOptimizer,
    "max_sharpe": MaxSharpeOptimizer,
    "min_variance": MinVarianceOptimizer,
    "risk_parity": RiskParityOptimizer,
    "hrp": HRPOptimizer,
    "black_litterman": BlackLittermanOptimizer,
    "cvar": CVaROptimizer,
}


def optimize(
    returns: typing.Union[Dict[str, list[float]], "pd.DataFrame"],
    method: str = "max_sharpe",
    **kwargs,
) -> OptimizationResult:
    """Factory function to run a named optimizer.

    Args:
        returns : return data
        method  : one of 'mean_variance', 'max_sharpe', 'min_variance',
                  'risk_parity', 'hrp', 'black_litterman', 'cvar'
        **kwargs : passed to the optimizer constructor

    Returns:
        OptimizationResult

    Raises:
        ValueError : if method is unknown
    """
    cls = OPTIMIZER_REGISTRY.get(method.lower())
    if cls is None:
        raise ValueError(
            f"Unknown optimizer method: {method!r}. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    return cls(returns, **kwargs).solve()
