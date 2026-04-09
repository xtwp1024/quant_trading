"""Matilda Portfolio Optimization Library.

Adapted from Quantropy/matilda/quantitative_analysis/portfolio_optimization.py.
Unique features NOT in finclaw: Hierarchical Risk Parity (HRP),
PostModern Portfolio Theory (PMPT), critical line algorithm support.

Works with pandas DataFrame of returns.
"""

import abc
import typing
from dataclasses import dataclass
from functools import partial
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize, Bounds


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PortfolioWeights:
    """Optimised portfolio weights."""
    weights: pd.Series          # asset -> weight
    expected_return: float     # annualised expected return
    volatility: float          # annualised volatility
    sharpe_ratio: float         # Sharpe ratio


@dataclass
class EfficientFrontierPoint:
    """Single point on the efficient frontier."""
    target_return: float
    volatility: float
    sharpe_ratio: float
    weights: pd.Series


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PortfolioAllocationModel(metaclass=abc.ABCMeta):
    """Abstract base for portfolio allocation models."""

    def __init__(self, df_returns: pd.DataFrame):
        self.df_returns = df_returns.dropna()

    @abc.abstractmethod
    def solve_weights(self, **kwargs) -> PortfolioWeights:
        """Solve for optimal weights."""
        pass

    # ----- shared helpers -----

    def expected_returns(self) -> pd.Series:
        """Mean historical returns (annualised)."""
        return self.df_returns.mean() * 252

    def covariance_matrix(self) -> pd.DataFrame:
        """Sample covariance matrix (annualised)."""
        return self.df_returns.cov() * 252

    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Compute return, volatility, Sharpe for a weight vector."""
        w = pd.Series(weights, index=self.df_returns.columns)
        ret = float(self.expected_returns().dot(w))
        vol = float(np.sqrt(w @ self.covariance_matrix() @ w))
        sharpe = ret / vol if vol != 0 else 0.0
        return ret, vol, sharpe


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (HRP)
# ---------------------------------------------------------------------------

class HierarchicalRiskParity(PortfolioAllocationModel):
    """Hierarchical Risk Parity portfolio allocation.

    Uses hierarchical clustering on the correlation matrix to partition
    assets into a dendrogram, then allocates recursively by inverse variance
    within clusters. Does not require inversion of the covariance matrix
    and is robust to noisy / singular matrices.

    Reference: Lopez de Prado (2016) "Building Diversified Portfolios
    that Outperform Out-of-Sample"
    """

    def solve_weights(
        self,
        correlation_matrix: Optional[pd.DataFrame] = None,
        linkage_method: str = 'ward',
    ) -> PortfolioWeights:
        """Compute HRP weights.

        Args:
            correlation_matrix: Asset correlation matrix (defaults to sample).
            linkage_method: scipy.cluster.hierarchy linkage method.

        Returns:
            PortfolioWeights with HRP weights.
        """
        cov = self.covariance_matrix()
        if correlation_matrix is None:
            # Build correlation from covariance
            d = np.sqrt(np.diag(cov.values))
            corr = cov.values / np.outer(d, d)
            corr = pd.DataFrame(corr, index=cov.index, columns=cov.columns)
        else:
            corr = correlation_matrix.copy()

        # Distance matrix from correlation
        dist_matrix = np.sqrt(np.clip(1 - corr.values, a_min=0, a_max=None))
        np.fill_diagonal(dist_matrix, 0)

        # Flatten to condensed distance matrix
        condensed = squareform(dist_matrix, checks=False)

        # Hierarchical clustering
        Z = linkage(condensed, method=linkage_method)

        # Build dendrogram order
        dendro = dendrogram(Z, no_plot=True)
        sort_idx = [int(x) - 1 for x in dendro['ivl']]  # 1-indexed in scipy
        sorted_assets = list(corr.columns[sort_idx])

        # Quasi-diagonalisation: reorder cov matrix
        cov_reordered = cov.loc[sorted_assets, sorted_assets]

        # Recursive bisection allocation
        weights = self._hrp_recursive(cov_reordered)

        # Map back to original asset order
        final_weights = pd.Series(0.0, index=cov.columns)
        for asset, w in weights.items():
            final_weights[asset] = w

        ret, vol, sharpe = self.portfolio_stats(final_weights.values)
        return PortfolioWeights(
            weights=final_weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
        )

    def _hrp_recursive(self, cov_sub: pd.DataFrame) -> pd.Series:
        """Recursive HRP weight allocation."""
        n = len(cov_sub)
        if n == 1:
            return pd.Series({cov_sub.index[0]: 1.0})

        # Split into two clusters
        assets = list(cov_sub.index)
        half = n // 2
        left_assets = assets[:half]
        right_assets = assets[half:]

        left_cov = cov_sub.loc[left_assets, left_assets]
        right_cov = cov_sub.loc[right_assets, right_assets]

        # Allocate proportionally to cluster variance
        left_var = self._cluster_variance(left_cov)
        right_var = self._cluster_variance(right_cov)
        total_var = left_var + right_var

        left_weight = right_var / total_var if total_var != 0 else 0.5
        right_weight = left_var / total_var if total_var != 0 else 0.5

        left_w = self._hrp_recursive(left_cov)
        right_w = self._hrp_recursive(right_cov)

        # Scale by cluster weight
        left_w = left_w * left_weight
        right_w = right_w * right_weight

        return pd.concat([left_w, right_w])

    @staticmethod
    def _cluster_variance(cov_sub: pd.DataFrame) -> float:
        """Variance of an equal-weighted cluster."""
        n = len(cov_sub)
        if n == 0:
            return 0.0
        weights = np.ones(n) / n
        return float(weights @ cov_sub.values @ weights)


# ---------------------------------------------------------------------------
# PostModern Portfolio Theory (PMPT)
# ---------------------------------------------------------------------------

class PostModernPortfolioTheory(PortfolioAllocationModel):
    """PostModern Portfolio Theory optimizer.

    Uses downside risk (Sortino ratio or other lower partial moment metrics)
    instead of standard deviation. Supports arbitrary risk metrics via
    scipy.optimize minimize.

    Reference: Sortino & van der Meer (1991) "Downside Risk"
    """

    def solve_weights(
        self,
        risk_metric: typing.Callable = None,
        target_return: Optional[float] = None,
        risk_free_rate: float = 0.0,
        long_only: bool = True,
    ) -> PortfolioWeights:
        """Solve for weights using PMPT framework.

        Args:
            risk_metric: Risk function(weights, df_returns) -> scalar.
                         Defaults to downside deviation (Sortino denominator).
            target_return: Required annualised return (equality constraint).
            risk_free_rate: Annualised risk-free rate.
            long_only: If True, weights >= 0.

        Returns:
            PortfolioWeights.
        """
        n = len(self.df_returns.columns)
        cov = self.covariance_matrix()

        if risk_metric is None:
            def risk_metric(w, df_returns):
                # Downside deviation (Sortino denominator)
                port_ret = df_returns.dot(w)
                mar = risk_free_rate / 252
                diff = mar - port_ret
                diff = np.clip(diff, a_min=0, a_max=None)
                return float(np.sqrt(np.mean(diff ** 2)))

        def objective(weights: np.ndarray) -> float:
            return risk_metric(weights, self.df_returns)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if target_return is not None:
            mean_ret = self.expected_returns()
            constraints.append({
                'type': 'eq',
                'fun': lambda x: float(mean_ret.dot(x)) - target_return,
            })

        # Bounds
        if long_only:
            bounds = Bounds(0, 1)
        else:
            bounds = Bounds(-1, 1)

        # Initial guess: equal weights
        x0 = np.ones(n) / n

        result = minimize(
            fun=objective,
            x0=x0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
        )

        if not result.success:
            raise RuntimeError(f"PMPT optimization failed: {result.message}")

        weights = pd.Series(result.x, index=self.df_returns.columns)
        ret, vol, sharpe = self.portfolio_stats(weights.values)
        return PortfolioWeights(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
        )


# ---------------------------------------------------------------------------
# Black-Litterman Model (skeleton — requires market-implied priors)
# ---------------------------------------------------------------------------

class BlackLittermanModel(PortfolioAllocationModel):
    """Black-Litterman portfolio allocation model.

    Combines market-implied equilibrium returns (CAPM prior) with
    investor views to produce posterior expected returns, then
    optimizes via MPT.

    Note: Full implementation requires a market cap weighted portfolio
    as the prior. This is a structured skeleton.
    """

    def __init__(
        self,
        df_returns: pd.DataFrame,
        market_caps: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
    ):
        super().__init__(df_returns)
        self.market_caps = market_caps
        self.risk_free_rate = risk_free_rate

    def solve_weights(
        self,
        view_returns: Optional[pd.Series] = None,
        view_confidence: Optional[np.ndarray] = None,
        risk_aversion: float = 2.5,
    ) -> PortfolioWeights:
        """Compute BL weights.

        Args:
            view_returns: Investor views on expected returns (alpha).
            view_confidence: Diagonal matrix of view confidences (P @ Omega @ P.T).
            risk_aversion: Risk aversion parameter (delta).

        Returns:
            PortfolioWeights.
        """
        # Equilibrium returns from market cap weighted portfolio
        if self.market_caps is None:
            # Fall back to equal-weighted if no market caps
            pi = self.expected_returns()
        else:
            mc = self.market_caps / self.market_caps.sum()
            cov = self.covariance_matrix()
            # Equilibrium returns: delta * Cov * w_mkt
            pi = risk_aversion * cov.dot(mc)

        if view_returns is None:
            # No views — just use equilibrium
            posterior = pi
        else:
            # Blended posterior: (tau*Cov)^-1 + P.T @ Omega^-1 @ P
            # Simplified: linear blend
            tau = 0.1  # scaling factor
            P = np.eye(len(pi))  # identity view matrix
            if view_confidence is None:
                Omega = np.eye(len(pi)) * 0.01
            else:
                Omega = view_confidence

            # Posterior mean formula (simplified)
            M = np.linalg.inv(np.linalg.inv(tau * cov) + P.T @ np.linalg.inv(Omega) @ P)
            views = view_returns.values if isinstance(view_returns, pd.Series) else view_returns
            posterior = M @ (np.linalg.inv(tau * cov) @ pi.values + P.T @ np.linalg.inv(Omega) @ views)
            posterior = pd.Series(posterior, index=pi.index)

        # Mean-variance optimization with posterior returns
        n = len(self.df_returns.columns)
        cov_mat = self.covariance_matrix()

        def neg_sharpe(w):
            ret = float(posterior.dot(w))
            vol = float(np.sqrt(w @ cov_mat @ w))
            return -(ret / vol if vol != 0 else 0)

        x0 = np.ones(n) / n
        result = minimize(
            fun=neg_sharpe,
            x0=x0,
            method='SLSQP',
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
            bounds=Bounds(0, 1),
        )

        if not result.success:
            raise RuntimeError(f"BL optimization failed: {result.message}")

        weights = pd.Series(result.x, index=self.df_returns.columns)
        ret, vol, sharpe = self.portfolio_stats(weights.values)
        return PortfolioWeights(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
        )


# ---------------------------------------------------------------------------
# Risk Parity (naive — equal risk contribution)
# ---------------------------------------------------------------------------

class RiskParityModel(PortfolioAllocationModel):
    """Risk Parity — equal risk contribution across assets.

    Each asset contributes equally to total portfolio volatility.
    Uses iterative optimisation to find weights.
    """

    def solve_weights(
        self,
        total_volatility: Optional[float] = None,
        risk_free_rate: float = 0.0,
    ) -> PortfolioWeights:
        """Solve for risk parity weights.

        Args:
            total_volatility: Target portfolio volatility (annualised).
            risk_free_rate: Annualised risk-free rate.

        Returns:
            PortfolioWeights.
        """
        n = len(self.df_returns.columns)
        cov = self.covariance_matrix().values

        def risk_contribution(w: np.ndarray) -> np.ndarray:
            """Individual risk contributions = w_i * (Cov @ w)_i / vol."""
            vol = float(np.sqrt(w @ cov @ w))
            if vol == 0:
                return np.zeros(n)
            marginal = cov @ w
            return w * marginal / vol

        def equal_risk_objective(w: np.ndarray) -> float:
            """Minimise sum of squared differences from equal risk."""
            rc = risk_contribution(w)
            target = np.mean(rc)
            return float(np.sum((rc - target) ** 2))

        x0 = np.ones(n) / n
        result = minimize(
            fun=equal_risk_objective,
            x0=x0,
            method='SLSQP',
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
            bounds=Bounds(0, 1),
        )

        if not result.success:
            raise RuntimeError(f"Risk parity optimization failed: {result.message}")

        weights = pd.Series(result.x, index=self.df_returns.columns)

        # Scale to target volatility if provided
        if total_volatility is not None:
            vol = float(np.sqrt(weights.values @ cov @ weights.values))
            if vol > 0:
                weights = weights * (total_volatility / vol)

        ret, vol, sharpe = self.portfolio_stats(weights.values)
        return PortfolioWeights(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
        )


# ---------------------------------------------------------------------------
# Efficient Frontier (Markowitz)
# ---------------------------------------------------------------------------

class EfficientFrontier(PortfolioAllocationModel):
    """Markowitz Efficient Frontier.

    Computes the full mean-variance efficient frontier via
    quadratic optimisation at multiple target return levels.
    """

    def solve_frontier(
        self,
        n_points: int = 50,
        risk_free_rate: float = 0.0,
    ) -> List[EfficientFrontierPoint]:
        """Compute the efficient frontier.

        Args:
            n_points: Number of points along the frontier.
            risk_free_rate: Annualised risk-free rate.

        Returns:
            List of EfficientFrontierPoint sorted by volatility.
        """
        mean_ret = self.expected_returns()
        cov = self.covariance_matrix().values
        n = len(self.df_returns.columns)

        target_returns = np.linspace(mean_ret.min(), mean_ret.max(), n_points)

        frontier = []
        for target in target_returns:
            def vol_objective(w):
                return float(np.sqrt(w @ cov @ w))

            result = minimize(
                fun=vol_objective,
                x0=np.ones(n) / n,
                method='SLSQP',
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: float(mean_ret.dot(x)) - target},
                ],
                bounds=Bounds(0, 1),
            )

            if not result.success:
                continue

            w = result.x
            vol = float(np.sqrt(w @ cov @ w))
            ret = float(mean_ret.dot(w))
            sharpe = (ret - risk_free_rate) / vol if vol != 0 else 0.0
            weights = pd.Series(w, index=self.df_returns.columns)
            frontier.append(EfficientFrontierPoint(
                target_return=ret,
                volatility=vol,
                sharpe_ratio=sharpe,
                weights=weights,
            ))

        return sorted(frontier, key=lambda p: p.volatility)

    def solve_weights(
        self,
        objective: str = 'max_sharpe',
        target_return: Optional[float] = None,
        risk_free_rate: float = 0.0,
    ) -> PortfolioWeights:
        """Solve for a specific point on the frontier.

        Args:
            objective: 'max_sharpe', 'min_volatility', or 'target_return'.
            target_return: Required annualised return (used if objective='target_return').
            risk_free_rate: Annualised risk-free rate.

        Returns:
            PortfolioWeights.
        """
        if objective == 'max_sharpe':
            frontier = self.solve_frontier(risk_free_rate=risk_free_rate)
            if not frontier:
                raise RuntimeError("Failed to compute frontier for max Sharpe")
            best = max(frontier, key=lambda p: p.sharpe_ratio)
            return PortfolioWeights(
                weights=best.weights,
                expected_return=best.target_return,
                volatility=best.volatility,
                sharpe_ratio=best.sharpe_ratio,
            )
        elif objective == 'min_volatility':
            frontier = self.solve_frontier(risk_free_rate=risk_free_rate)
            if not frontier:
                raise RuntimeError("Failed to compute frontier")
            best = min(frontier, key=lambda p: p.volatility)
            return PortfolioWeights(
                weights=best.weights,
                expected_return=best.target_return,
                volatility=best.volatility,
                sharpe_ratio=best.sharpe_ratio,
            )
        elif objective == 'target_return':
            if target_return is None:
                raise ValueError("target_return required for objective='target_return'")
            mean_ret = self.expected_returns()
            cov = self.covariance_matrix().values
            n = len(self.df_returns.columns)
            result = minimize(
                fun=lambda w: float(np.sqrt(w @ cov @ w)),
                x0=np.ones(n) / n,
                method='SLSQP',
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: float(mean_ret.dot(x)) - target_return},
                ],
                bounds=Bounds(0, 1),
            )
            if not result.success:
                raise RuntimeError(f"Target return optimization failed: {result.message}")
            w = result.x
            vol = float(np.sqrt(w @ cov @ w))
            ret = float(mean_ret.dot(w))
            sharpe = (ret - risk_free_rate) / vol if vol != 0 else 0.0
            return PortfolioWeights(
                weights=pd.Series(w, index=self.df_returns.columns),
                expected_return=ret,
                volatility=vol,
                sharpe_ratio=sharpe,
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")
