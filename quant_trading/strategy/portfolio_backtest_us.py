#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
US Portfolio Strategy Backtesting Module / 美国股票组合策略回测模块

Absorbed from PortfolioStrategyBacktestUS repository.
Incorporates volatility regime detection, mean-variance optimization,
risk parity strategies, and full backtesting capabilities.

从 PortfolioStrategyBacktestUS 仓库吸收而来。
整合了波动率区间检测、均值-方差优化、风险平价策略及完整回测功能。

@author: federico (original)
"""

import math
import warnings
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm


# =============================================================================
# Volatility Regime Detector / 波动率区间检测器
# =============================================================================


class VolatilityRegimeDetector:
    """
    Detect high/low volatility regimes using rolling windows.

    使用滚动窗口检测高/低波动率区间。

    This class analyzes historical volatility to classify market regimes,
    which can be used to enable or disable certain trading strategies.

    该类分析历史波动率以分类市场区间，可用于启用或禁用某些交易策略。

    Parameters
    ----------
    window : int, default 20
        Rolling window size for volatility calculation (in trading days).
        波动率计算的滚动窗口大小（以交易日计）。
    threshold_percentile : float, default 0.75
        Percentile threshold for regime classification (0.0 to 1.0).
        Values above this percentile are classified as HIGH volatility regime.
        区间分类的百分位阈值（0.0 到 1.0）。高于此百分位的值被分类为高波动率区间。
    annualization_factor : float, default 252
        Factor to annualize daily volatility. 每日波动率的年化因子。

    Example
    -------
    >>> detector = VolatilityRegimeDetector(window=20, threshold_percentile=0.75)
    >>> regimes = detector.detect_regimes(volatility_series)
    >>> is_high_vol = detector.is_high_regime(current_vol)
    """

    def __init__(
        self,
        window: int = 20,
        threshold_percentile: float = 0.75,
        annualization_factor: float = 252.0,
    ):
        self.window = window
        self.threshold_percentile = threshold_percentile
        self.annualization_factor = annualization_factor
        self._threshold: Optional[float] = None
        self._vol_history: Optional[pd.Series] = None

    def fit(self, returns: pd.Series) -> "VolatilityRegimeDetector":
        """
        Fit the detector by computing the volatility threshold from historical data.

        从历史数据计算波动率阈值来拟合检测器。

        Parameters
        ----------
        returns : pd.Series
            Historical daily returns series.
            历史每日收益率序列。

        Returns
        -------
        self : VolatilityRegimeDetector
            Returns self for method chaining.
            返回自身以支持方法链式调用。
        """
        # Compute rolling volatility (annualized)
        vol = (
            returns.rolling(window=self.window, min_periods=1)
            .std()
            .dropna()
            * math.sqrt(self.annualization_factor)
        )
        self._vol_history = vol
        self._threshold = vol.quantile(self.threshold_percentile)
        return self

    def detect_regimes(self, returns: pd.Series) -> pd.Series:
        """
        Detect volatility regimes over time.

        检测时间序列上的波动率区间。

        Parameters
        ----------
        returns : pd.Series
            Daily returns series.
            每日收益率序列。

        Returns
        -------
        regimes : pd.Series
            Series of regime labels: 1 for HIGH, 0 for LOW.
            区间标签序列：1 表示高波动，0 表示低波动。
        """
        if self._threshold is None:
            self.fit(returns)

        vol = (
            returns.rolling(window=self.window, min_periods=self.window)
            .std()
            * math.sqrt(self.annualization_factor)
        )
        regimes = (vol > self._threshold).astype(int)
        return regimes

    def is_high_regime(self, value: float) -> bool:
        """
        Check if a given volatility value is in the HIGH regime.

        检查给定的波动率值是否处于高波动区间。

        Parameters
        ----------
        value : float
            Volatility value (annualized).
            波动率值（年化）。

        Returns
        -------
        bool
            True if HIGH volatility regime, False otherwise.
            如果是高波动区间返回 True，否则返回 False。
        """
        if self._threshold is None:
            raise ValueError("Detector has not been fitted. Call fit() first.")
        return value > self._threshold

    @property
    def threshold(self) -> Optional[float]:
        """Get the computed volatility threshold / 获取计算的波动率阈值。"""
        return self._threshold


# =============================================================================
# Volatility Threshold Gate / 波动率阈值门控
# =============================================================================


class VolatilityThresholdGate:
    """
    Volatility-based regime gate for enabling/disabling strategies.

    基于波动率的区间门控，用于启用/禁用策略。

    This gate uses volatility regime detection to determine whether
    a strategy should be active or inactive. When market volatility
    exceeds the threshold, the strategy is disabled (gate closed).

    该门控使用波动率区间检测来决定策略是否应处于活跃状态。
    当市场波动率超过阈值时，策略被禁用（门关闭）。

    Parameters
    ----------
    threshold : float, optional
        Fixed volatility threshold (annualized). If None, computed from data.
        固定波动率阈值（年化）。如果为 None，则从数据计算。
    window : int, default 20
        Rolling window for volatility estimation.
        波动率估计的滚动窗口。
    mode : str, default 'strict'
        Gate mode: 'strict' (disable when above threshold) or 'flexible' (reduce exposure).
        门控模式：'strict'（高于阈值时禁用）或 'flexible'（降低敞口）。

    Example
    -------
    >>> gate = VolatilityThresholdGate(threshold=0.20, mode='strict')
    >>> is_enabled = gate.is_enabled(current_volatility)
    >>> exposure = gate.get_exposure(current_volatility)  # for flexible mode
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        window: int = 20,
        mode: str = "strict",
    ):
        self.threshold = threshold
        self.window = window
        self.mode = mode
        self._detector: Optional[VolatilityRegimeDetector] = None

    def fit(self, returns: pd.Series, threshold_percentile: float = 0.75) -> "VolatilityThresholdGate":
        """
        Fit the gate on historical returns data.

        在历史收益率数据上拟合门控。

        Parameters
        ----------
        returns : pd.Series
            Historical daily returns.
            历史每日收益率。
        threshold_percentile : float, default 0.75
            Percentile for automatic threshold detection.
            自动阈值检测的百分位。

        Returns
        -------
        self : VolatilityThresholdGate
            Returns self for method chaining.
        """
        if self.threshold is None:
            self._detector = VolatilityRegimeDetector(
                window=self.window, threshold_percentile=threshold_percentile
            )
            self._detector.fit(returns)
            self.threshold = self._detector.threshold
        return self

    def is_enabled(self, volatility: float) -> bool:
        """
        Determine if the strategy should be enabled.

        判断策略是否应被启用。

        Parameters
        ----------
        volatility : float
            Current annualized volatility.
            当前年化波动率。

        Returns
        -------
        bool
            True if strategy is enabled (gate open), False if disabled (gate closed).
            如果策略启用（门打开）返回 True，如果禁用（门关闭）返回 False。
        """
        if self.mode == "strict":
            return volatility <= self.threshold
        else:  # flexible mode - still enable but with reduced exposure
            return True

    def get_exposure(self, volatility: float) -> float:
        """
        Get recommended portfolio exposure based on volatility.

        根据波动率获取建议的投资组合敞口。

        Parameters
        ----------
        volatility : float
            Current annualized volatility.
            当前年化波动率。

        Returns
        -------
        float
            Exposure multiplier between 0.0 and 1.0.
            0.0 到 1.0 之间的敞口乘数。
        """
        if self.mode == "strict":
            return 1.0 if self.is_enabled(volatility) else 0.0
        else:
            # Flexible mode: scale exposure inversely with volatility
            if volatility <= self.threshold:
                return 1.0
            else:
                # Scale down exposure as volatility increases
                ratio = self.threshold / volatility
                return max(0.0, min(1.0, ratio))

    def get_gate_status(self, volatility: float) -> str:
        """
        Get the gate status as a string.

        获取门控状态的字符串表示。

        Parameters
        ----------
        volatility : float
            Current annualized volatility.
            当前年化波动率。

        Returns
        -------
        str
            'OPEN' if enabled, 'CLOSED' if disabled.
        """
        return "OPEN" if self.is_enabled(volatility) else "CLOSED"


# =============================================================================
# Mean-Variance Optimizer / 均值-方差优化器
# =============================================================================


class MeanVarianceOptimizer:
    """
    Markowitz Mean-Variance Portfolio Optimization (Markowitz 1952).

    Markowitz 均值-方差投资组合优化。

    Implements Markowitz's modern portfolio theory using scipy optimization
    (no cvxpy dependency). Supports both maximum Sharpe ratio and
    minimum variance portfolios.

    使用 scipy 优化实现 Markowitz 现代投资组合理论（无 cvxpy 依赖）。
    支持最大 Sharpe 比率和最小方差投资组合。

    Parameters
    ----------
    method : str, default 'max_sharpe'
        Optimization method: 'max_sharpe' (maximum Sharpe ratio)
        or 'min_variance' (minimum variance).
        优化方法：'max_sharpe'（最大 Sharpe 比率）或 'min_variance'（最小方差）。
    allow_short : bool, default False
        Whether to allow short selling (negative weights).
        是否允许卖空（负权重）。
    max_leverage : float, default 1.0
        Maximum allowed leverage (sum of absolute weights).
        最大允许杠杆（绝对权重之和）。
    risk_aversion : float, optional
        Risk aversion coefficient (for quadratic utility).
        If None, computes tangency portfolio.
        风险厌恶系数（用于二次效用）。如果为 None，则计算切点组合。
    annualization_factor : float, default 252.0
        Factor to annualize returns and volatility.
        收益率和波动率的年化因子。

    Example
    -------
    >>> optimizer = MeanVarianceOptimizer(method='max_sharpe', allow_short=False)
    >>> weights = optimizer.optimize(mean_returns, cov_matrix)
    >>> # or with risk aversion
    >>> optimizer_ra = MeanVarianceOptimizer(method='max_sharpe', risk_aversion=2.0)
    >>> weights_ra = optimizer_ra.optimize(mean_returns, cov_matrix)
    """

    def __init__(
        self,
        method: str = "max_sharpe",
        allow_short: bool = False,
        max_leverage: float = 1.0,
        risk_aversion: Optional[float] = None,
        annualization_factor: float = 252.0,
    ):
        if method not in ("max_sharpe", "min_variance"):
            raise ValueError("method must be 'max_sharpe' or 'min_variance'")
        self.method = method
        self.allow_short = allow_short
        self.max_leverage = max_leverage
        self.risk_aversion = risk_aversion
        self.annualization_factor = annualization_factor

    def _quadratic_utility(
        self, w: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray, phi: Optional[float]
    ) -> float:
        """
        Compute quadratic utility function for portfolio optimization.

        计算投资组合优化的二次效用函数。

        U(w) = phi/2 * w'*Sigma*w - w'*mu  (for finite phi)
        U(w) = -w'*mu / sqrt(w'*Sigma*w)     (tangency portfolio when phi=None)
        """
        port_var = w @ cov_mat @ w
        port_var = abs(port_var)  # prevent numerical issues
        port_ret = w @ mu

        if phi is None:
            # Tangency portfolio (maximum Sharpe ratio)
            return -port_ret / math.sqrt(port_var)
        elif math.isinf(phi):
            # Minimum variance portfolio
            return port_var
        else:
            # Quadratic utility with risk aversion
            return phi / 2.0 * port_var - port_ret

    def _leverage(self, w: np.ndarray) -> float:
        """Compute leverage (sum of absolute weights)."""
        return np.abs(w).sum()

    def optimize(
        self,
        mean_returns: Union[np.ndarray, pd.Series],
        cov_matrix: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Optimize portfolio weights.

        优化投资组合权重。

        Parameters
        ----------
        mean_returns : array-like
            Expected returns for each asset (N assets).
            每个资产的预期收益率（N 个资产）。
        cov_matrix : array-like
            Covariance matrix of returns (N x N).
            收益率协方差矩阵（N x N）。

        Returns
        -------
        weights : np.ndarray
            Optimized portfolio weights.
            优化后的投资组合权重。
        """
        # Convert to numpy arrays
        if isinstance(mean_returns, pd.Series):
            asset_names = mean_returns.index
            mu = mean_returns.values
        else:
            asset_names = np.arange(len(mean_returns))
            mu = np.asarray(mean_returns)

        if isinstance(cov_matrix, pd.DataFrame):
            Sigma = cov_matrix.values
        else:
            Sigma = np.asarray(cov_matrix)

        n_assets = len(mu)

        # Constraints
        # 1. Sum of weights = 1
        linear_constraint = opt.LinearConstraint(np.ones(n_assets), lb=1.0, ub=1.0)

        # 2. Leverage constraint
        nonlinear_constraint = opt.NonlinearConstraint(
            self._leverage, lb=1.0, ub=self.max_leverage
        )

        # 3. Bounds (short selling)
        if self.allow_short:
            bounds = opt.Bounds(lb=-np.inf, ub=np.inf)
        else:
            bounds = opt.Bounds(lb=0.0, ub=np.inf)

        # Initial guess: equal weights
        w0 = np.ones(n_assets) / n_assets

        # Compute optimal weights
        try:
            if self.method == "min_variance":
                phi = float("inf")  # Minimize variance
            elif self.risk_aversion is not None:
                phi = self.risk_aversion
            else:
                phi = None  # Tangency portfolio (max Sharpe)

            result = opt.minimize(
                self._quadratic_utility,
                w0,
                args=(mu, Sigma, phi),
                method="trust-constr",
                options={"gtol": 1e-6, "xtol": 1e-6},
                constraints=(linear_constraint, nonlinear_constraint),
                bounds=bounds,
            )

            if not result.success:
                # Fallback to equal weights
                warnings.warn(
                    f"Optimization failed: {result.message}. Using equal weights."
                )
                return w0

            return result.x

        except Exception as e:
            warnings.warn(f"Optimization error: {e}. Using equal weights.")
            return w0

    def optimize_portfolio(
        self,
        returns_df: pd.DataFrame,
        risk_aversion: Optional[float] = None,
    ) -> pd.Series:
        """
        Optimize portfolio directly from returns DataFrame.

        直接从收益率 DataFrame 优化投资组合。

        Parameters
        ----------
        returns_df : pd.DataFrame
            Historical returns with assets as columns.
            资产作为列的历史收益率。
        risk_aversion : float, optional
            Risk aversion parameter. If None, computes max Sharpe.
            风险厌恶参数。如果为 None，则计算最大 Sharpe。

        Returns
        -------
        weights : pd.Series
            Optimized weights indexed by asset names.
            以资产名称为索引的优化权重。
        """
        mu = returns_df.mean()
        Sigma = returns_df.cov()

        if risk_aversion is not None:
            self.risk_aversion = risk_aversion

        weights = self.optimize(mu, Sigma)

        if isinstance(mu, pd.Series):
            return pd.Series(weights, index=mu.index)
        else:
            return pd.Series(weights)


# =============================================================================
# Risk Parity Strategy / 风险平价策略
# =============================================================================


class RiskParityStrategy:
    """
    Risk Parity Portfolio Construction.

    风险平价投资组合构建。

    In a risk parity portfolio, each asset contributes equally to the
    total portfolio risk. This is achieved by weighting assets inversely
    proportional to their volatility (or risk contribution).

    在风险平价组合中，每个资产对总风险的贡献相等。
    这是通过将资产权重与波动率（或风险贡献）成反比来实现的。

    Parameters
    ----------
    method : str, default 'volatility'
        Risk measure: 'volatility' (inverse volatility weighting)
        or 'risk_contribution' (exact risk parity via optimization).
        风险度量：'volatility'（逆波动率加权）或 'risk_contribution'（通过优化实现精确风险平价）。
    allow_short : bool, default False
        Whether to allow short selling.
        是否允许卖空。
    max_leverage : float, default 1.0
        Maximum allowed leverage.
        最大允许杠杆。

    Example
    -------
    >>> rp = RiskParityStrategy(method='volatility')
    >>> weights = rp.compute_weights(volatility_series)
    >>> # or with risk contribution method
    >>> rp_rc = RiskParityStrategy(method='risk_contribution')
    >>> weights_rc = rp_rc.compute_weights(volatility_series, cov_matrix)
    """

    def __init__(
        self,
        method: str = "volatility",
        allow_short: bool = False,
        max_leverage: float = 1.0,
    ):
        if method not in ("volatility", "risk_contribution"):
            raise ValueError("method must be 'volatility' or 'risk_contribution'")
        self.method = method
        self.allow_short = allow_short
        self.max_leverage = max_leverage

    def compute_weights(
        self,
        volatility: Union[np.ndarray, pd.Series],
        cov_matrix: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> np.ndarray:
        """
        Compute risk parity weights.

        计算风险平价权重。

        Parameters
        ----------
        volatility : array-like
            Asset volatilities (can be raw or from covariance matrix).
            资产波动率（可以是原始值或来自协方差矩阵）。
        cov_matrix : array-like, optional
            Covariance matrix. Required for 'risk_contribution' method.
            协方差矩阵。对于 'risk_contribution' 方法是必需的。

        Returns
        -------
        weights : np.ndarray
            Risk parity portfolio weights.
            风险平价投资组合权重。
        """
        if isinstance(volatility, pd.Series):
            asset_names = volatility.index
            vol = volatility.values
        else:
            asset_names = np.arange(len(volatility))
            vol = np.asarray(volatility)

        # Avoid division by zero
        vol = np.where(vol <= 0, np.finfo(float).eps, vol)

        if self.method == "volatility":
            # Inverse volatility weighting
            inv_vol = 1.0 / vol
            weights = inv_vol / inv_vol.sum()

        elif self.method == "risk_contribution":
            if cov_matrix is None:
                raise ValueError(
                    "cov_matrix required for 'risk_contribution' method"
                )
            if isinstance(cov_matrix, pd.DataFrame):
                Sigma = cov_matrix.values
            else:
                Sigma = np.asarray(cov_matrix)

            n_assets = len(vol)
            weights = self._risk_contribution_optimization(Sigma, n_assets)

        # Apply leverage constraint
        current_leverage = np.abs(weights).sum()
        if current_leverage > self.max_leverage:
            weights = weights * (self.max_leverage / current_leverage)

        # Apply short selling constraint
        if not self.allow_short:
            weights = np.maximum(weights, 0.0)
            # Re-normalize if we clipped
            if weights.sum() > 0:
                weights = weights / weights.sum()

        return weights

    def _risk_contribution_optimization(
        self, Sigma: np.ndarray, n_assets: int
    ) -> np.ndarray:
        """
        Optimize portfolio to achieve equal risk contribution.

        优化投资组合以实现相等风险贡献。

        Each asset's contribution to portfolio variance is:
        w_i * (Sigma * w)_i / sqrt(w' * Sigma * w)

        We minimize the squared difference between each asset's
        contribution and the target (1/n of total risk).
        """

        def risk_contribution(w: np.ndarray) -> np.ndarray:
            """Compute vector of risk contributions."""
            port_var = w @ Sigma @ w
            if port_var <= 0:
                return np.ones(n_assets) * float("inf")
            marginal_contrib = Sigma @ w
            risk_contrib = w * marginal_contrib / math.sqrt(port_var)
            return risk_contrib

        def objective(w: np.ndarray) -> float:
            """Minimize squared difference from equal risk contribution."""
            rc = risk_contribution(w)
            target_rc = np.ones(n_assets) / n_assets * np.sum(rc)
            return np.sum((rc - target_rc) ** 2)

        # Constraints
        linear_constraint = opt.LinearConstraint(
            np.ones(n_assets), lb=1.0, ub=1.0
        )
        nonlinear_constraint = opt.NonlinearConstraint(
            lambda w: np.abs(w).sum(), lb=1.0, ub=self.max_leverage
        )

        if self.allow_short:
            bounds = opt.Bounds(lb=-np.inf, ub=np.inf)
        else:
            bounds = opt.Bounds(lb=0.0, ub=np.inf)

        w0 = np.ones(n_assets) / n_assets

        try:
            result = opt.minimize(
                objective,
                w0,
                method="trust-constr",
                options={"gtol": 1e-6, "xtol": 1e-6},
                constraints=(linear_constraint, nonlinear_constraint),
                bounds=bounds,
            )

            if not result.success:
                warnings.warn(
                    f"Risk contribution optimization failed. Using inverse volatility."
                )
                vol = np.sqrt(np.diag(Sigma))
                inv_vol = 1.0 / np.where(vol <= 0, np.finfo(float).eps, vol)
                return inv_vol / inv_vol.sum()

            return result.x
        except Exception as e:
            warnings.warn(f"Risk contribution error: {e}. Using inverse volatility.")
            vol = np.sqrt(np.diag(Sigma))
            inv_vol = 1.0 / np.where(vol <= 0, np.finfo(float).eps, vol)
            return inv_vol / inv_vol.sum()

    def compute_weights_from_returns(
        self, returns_df: pd.DataFrame
    ) -> pd.Series:
        """
        Compute risk parity weights from returns DataFrame.

        从收益率 DataFrame 计算风险平价权重。

        Parameters
        ----------
        returns_df : pd.DataFrame
            Historical returns with assets as columns.
            资产作为列的历史收益率。

        Returns
        -------
        weights : pd.Series
            Risk parity weights indexed by asset names.
            以资产名称为索引的风险平价权重。
        """
        vol = returns_df.std()  # Daily volatility
        vol_annual = vol * math.sqrt(252)  # Annualize

        if self.method == "risk_contribution":
            cov_annual = returns_df.cov() * 252  # Annualize covariance
            weights = self.compute_weights(vol_annual.values, cov_annual.values)
        else:
            weights = self.compute_weights(vol_annual.values)

        return pd.Series(weights, index=returns_df.columns)


# =============================================================================
# Portfolio Backtester / 投资组合回测器
# =============================================================================


class PortfolioBacktester:
    """
    Backtest portfolio strategies with transaction costs.

    带交易成本的投资组合策略回测。

    This class simulates portfolio trading with realistic transaction costs,
    including fixed fees and proportional fees. It tracks portfolio value,
    positions, and computes performance metrics.

    该类模拟带真实交易成本的投资组合交易，包括固定费用和比例费用。
    它跟踪组合价值、持仓并计算绩效指标。

    Parameters
    ----------
    initial_capital : float, default 1_000_000
        Initial portfolio capital.
        初始组合资本。
    transaction_fee_fixed : float, default 0.0
        Fixed transaction fee per trade.
        每笔交易的固定交易费用。
    transaction_fee_proportional : float, default 0.001
        Proportional transaction fee (e.g., 0.001 = 0.1%).
        比例交易费用（例如，0.001 = 0.1%）。
    allow_short : bool, default False
        Whether to allow short selling.
        是否允许卖空。
    max_leverage : float, default 1.0
        Maximum allowed leverage.
        最大允许杠杆。

    Example
    -------
    >>> backtester = PortfolioBacktester(initial_capital=1_000_000,
    ...                                   transaction_fee_proportional=0.001)
    >>> results = backtester.backtest(returns_df, weights_df, prices_df)
    >>> print(results['final_value'], results['sharpe_ratio'])
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        transaction_fee_fixed: float = 0.0,
        transaction_fee_proportional: float = 0.001,
        allow_short: bool = False,
        max_leverage: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.transaction_fee_fixed = transaction_fee_fixed
        self.transaction_fee_proportional = transaction_fee_proportional
        self.allow_short = allow_short
        self.max_leverage = max_leverage

        # State variables (reset during backtest)
        self._holdings: Dict[str, float] = {"cash": initial_capital}
        self._portfolio_value: float = initial_capital
        self._history: List[Dict] = []

    def reset(self) -> "PortfolioBacktester":
        """Reset the backtester state."""
        self._holdings = {"cash": self.initial_capital}
        self._portfolio_value = self.initial_capital
        self._history = []
        return self

    def _compute_transaction_cost(
        self, value: float, quantity: float, price: float
    ) -> float:
        """Compute total transaction cost."""
        trade_value = abs(quantity * price)
        cost = (
            self.transaction_fee_fixed
            + self.transaction_fee_proportional * trade_value
        )
        return min(cost, trade_value)  # Cost cannot exceed trade value

    def trade(
        self,
        timestamp: pd.Timestamp,
        ticker: str,
        quantity: float,
        price: float,
    ) -> Dict[str, float]:
        """
        Execute a trade.

        执行交易。

        Parameters
        ----------
        timestamp : pd.Timestamp
            Trade timestamp.
            交易时间戳。
        ticker : str
            Asset ticker symbol.
            资产代码。
        quantity : float
            Number of shares (positive = buy, negative = sell).
            股数（正 = 买入，负 = 卖出）。
        price : float
            Price per share.
            每股价格。

        Returns
        -------
        dict : Trade details including cost.
            包含成本的交易详情字典。
        """
        trade_value = quantity * price

        # Compute transaction cost
        cost = self._compute_transaction_cost(trade_value, quantity, price)

        # Update cash
        self._holdings["cash"] -= trade_value + cost

        # Update holdings
        if ticker in self._holdings:
            self._holdings[ticker] += quantity
        else:
            self._holdings[ticker] = quantity

        # Remove zero holdings
        if self._holdings.get(ticker, 0) == 0:
            del self._holdings[ticker]

        # Update portfolio value
        self._portfolio_value -= cost

        return {
            "timestamp": timestamp,
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "trade_value": trade_value,
            "cost": cost,
        }

    def rebalance(
        self,
        timestamp: pd.Timestamp,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[Dict]:
        """
        Rebalance portfolio to target weights.

        将投资组合重新平衡到目标权重。

        Parameters
        ----------
        timestamp : pd.Timestamp
            Rebalance timestamp.
            再平衡时间戳。
        target_weights : dict
            Target weights {ticker: weight}.
            目标权重 {代码: 权重}。
        prices : dict
            Current prices {ticker: price}.
            当前价格 {代码: 价格}。

        Returns
        -------
        trades : list
            List of executed trades.
            已执行交易的列表。
        """
        trades = []

        # Get current tickers
        current_tickers = set(self._holdings.keys()) - {"cash"}
        target_tickers = set(target_weights.keys())

        # Sell positions no longer needed
        for ticker in current_tickers - target_tickers:
            quantity = -self._holdings[ticker]
            price = prices.get(ticker, 0.0)
            if not math.isnan(price) and price > 0:
                trade = self.trade(timestamp, ticker, quantity, price)
                trades.append(trade)

        # Update existing positions
        for ticker in current_tickers & target_tickers:
            target_value = self._portfolio_value * target_weights[ticker]
            current_value = self._holdings[ticker] * prices.get(ticker, 0.0)
            diff_value = target_value - current_value
            price = prices.get(ticker, 0.0)
            if not math.isnan(price) and price > 0:
                quantity = diff_value / price
                if abs(quantity) >= 1:  # Only trade if at least 1 share
                    trade = self.trade(timestamp, ticker, quantity, price)
                    trades.append(trade)

        # Buy new positions
        for ticker in target_tickers - current_tickers:
            target_value = self._portfolio_value * target_weights[ticker]
            price = prices.get(ticker, 0.0)
            if not math.isnan(price) and price > 0:
                quantity = target_value / price
                if quantity >= 1:
                    trade = self.trade(timestamp, ticker, quantity, price)
                    trades.append(trade)

        return trades

    def update_value(self, prices: Dict[str, float]) -> float:
        """
        Update and return current portfolio value.

        更新并返回当前投资组合价值。

        Parameters
        ----------
        prices : dict
            Current prices {ticker: price}.
            当前价格 {代码: 价格}。

        Returns
        -------
        value : float
            Current portfolio value.
            当前投资组合价值。
        """
        total_value = self._holdings.get("cash", 0.0)

        for ticker, quantity in self._holdings.items():
            if ticker != "cash" and ticker in prices:
                price = prices[ticker]
                if not math.isnan(price):
                    total_value += quantity * price

        self._portfolio_value = total_value
        return total_value

    def record_state(self, timestamp: pd.Timestamp) -> Dict:
        """
        Record current portfolio state.

        记录当前投资组合状态。

        Parameters
        ----------
        timestamp : pd.Timestamp
            Record timestamp.
            记录时间戳。

        Returns
        -------
        state : dict
            Portfolio state snapshot.
            投资组合状态快照。
        """
        state = {
            "timestamp": timestamp,
            "portfolio_value": self._portfolio_value,
            "holdings": self._holdings.copy(),
        }
        self._history.append(state)
        return state

    def backtest(
        self,
        returns_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        prices_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run backtest on portfolio strategy.

        运行投资组合策略回测。

        Parameters
        ----------
        returns_df : pd.DataFrame
            Asset returns (index = dates, columns = tickers).
            资产收益率（索引 = 日期，列 = 代码）。
        weights_df : pd.DataFrame
            Target portfolio weights over time (same index as returns_df).
            目标投资组合权重（与 returns_df 索引相同）。
        prices_df : pd.DataFrame, optional
            Asset prices. If None, inferred from returns.
            资产价格。如果为 None，从收益率推断。

        Returns
        -------
        results : dict
            Backtest results including metrics.
            回测结果，包含指标。
        """
        self.reset()

        # Prepare prices if not provided
        if prices_df is None:
            # Assume returns_df index is date and we start at 1.0
            prices_df = (1 + returns_df).cumprod()

        # Ensure aligned dates
        common_dates = returns_df.index.intersection(weights_df.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between returns and weights")
        common_dates = common_dates.sort_values()

        # Initialize with first valid prices
        first_date = common_dates[0]
        self.record_state(first_date)

        # Process each period
        for date in common_dates:
            prices = prices_df.loc[date].to_dict()

            # Update portfolio value with current prices
            self.update_value(prices)

            # Get target weights
            target_weights = weights_df.loc[date].to_dict()

            # Remove any NaN weights
            target_weights = {
                k: v for k, v in target_weights.items() if not math.isnan(v)
            }

            # Normalize weights to sum to 1 (handle cash)
            total_weight = sum(target_weights.values())
            if total_weight > 0:
                target_weights = {
                    k: v / total_weight for k, v in target_weights.items()
                }

            # Rebalance if weights differ significantly
            current_weights = {}
            for ticker, qty in self._holdings.items():
                if ticker != "cash" and ticker in prices:
                    current_weights[ticker] = (
                        qty * prices[ticker]
                    ) / self._portfolio_value

            needs_rebalance = False
            for ticker, target_w in target_weights.items():
                current_w = current_weights.get(ticker, 0.0)
                if abs(target_w - current_w) > 0.01:  # 1% threshold
                    needs_rebalance = True
                    break

            if needs_rebalance:
                self.rebalance(date, target_weights, prices)

            # Record state
            self.record_state(date)

        # Compute final metrics
        results = self.compute_results(returns_df.loc[common_dates])
        return results

    def compute_results(self, returns_df: pd.DataFrame) -> Dict:
        """
        Compute backtest performance metrics.

        计算回测绩效指标。

        Parameters
        ----------
        returns_df : pd.DataFrame
            Actual asset returns for the backtest period.
            回测期间的实际资产收益率。

        Returns
        -------
        metrics : dict
            Performance metrics dictionary.
            绩效指标字典。
        """
        # Extract portfolio value history
        values = pd.Series(
            [h["portfolio_value"] for h in self._history],
            index=[h["timestamp"] for h in self._history],
        )

        # Compute returns
        portfolio_returns = values.pct_change().dropna()

        if len(portfolio_returns) == 0:
            return {
                "final_value": self.initial_capital,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }

        # Total return
        total_return = (values.iloc[-1] / values.iloc[0]) - 1

        # Annualized return
        n_years = len(portfolio_returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Volatility (annualized)
        volatility = portfolio_returns.std() * math.sqrt(252)

        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative = values / values.iloc[0]
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = abs(drawdown.min())

        return {
            "final_value": values.iloc[-1],
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "portfolio_values": values,
            "portfolio_returns": portfolio_returns,
        }

    def compute_weights(
        self,
        mean_returns: Union[pd.Series, np.ndarray],
        cov_matrix: Union[pd.DataFrame, np.ndarray],
        method: str = "max_sharpe",
    ) -> np.ndarray:
        """
        Compute portfolio weights using MeanVarianceOptimizer.

        使用 MeanVarianceOptimizer 计算投资组合权重。

        Parameters
        ----------
        mean_returns : array-like
            Expected returns.
            预期收益率。
        cov_matrix : array-like
            Covariance matrix.
            协方差矩阵。
        method : str
            Optimization method ('max_sharpe' or 'min_variance').
            优化方法（'max_sharpe' 或 'min_variance'）。

        Returns
        -------
        weights : np.ndarray
            Portfolio weights.
            投资组合权重。
        """
        optimizer = MeanVarianceOptimizer(
            method=method,
            allow_short=self.allow_short,
            max_leverage=self.max_leverage,
        )
        return optimizer.optimize(mean_returns, cov_matrix)


# =============================================================================
# Module Exports / 模块导出
# =============================================================================

__all__ = [
    "VolatilityRegimeDetector",
    "VolatilityThresholdGate",
    "MeanVarianceOptimizer",
    "RiskParityStrategy",
    "PortfolioBacktester",
]
