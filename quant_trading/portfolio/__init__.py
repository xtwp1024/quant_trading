"""
quant_trading.portfolio — Portfolio Optimization & Risk Metrics.

A comprehensive portfolio management module providing:
- Seven optimizer classes (Markowitz, Max Sharpe, Min Variance,
  Risk Parity, HRP, Black-Litterman, CVaR)
- A complete portfolio risk metrics suite (VaR, CVaR, CDaR, Omega,
  Sortino, Calmar, Kappa ratios, drawdown analysis, and more)
- Seamless integration with quant_trading.risk and quant_trading.factors

Quick start
----------
>>> from quant_trading.portfolio import (
...     MeanVarianceOptimizer,
...     RiskParityOptimizer,
...     HRPOptimizer,
...     BlackLittermanOptimizer,
...     CVaROptimizer,
...     optimize,          # factory function
...     portfolio_risk_report,
...     DrawdownInfo,
... )
>>> # Dict-style (pure Python, no numpy needed)
>>> returns = {"AAPL": [0.01, -0.02, 0.03], "GOOG": [0.02, -0.01, 0.015]}
>>> opt = MeanVarianceOptimizer(returns, target_return=0.12)
>>> result = opt.solve()
>>> print(result.weights)
{'AAPL': 0.42, 'GOOG': 0.58}

>>> # pandas DataFrame style (numpy/scipy required)
>>> import pandas as pd
>>> df = pd.DataFrame(returns)
>>> result = optimize(df, method="hrp")
>>> print(result.weights)
{'AAPL': 0.38, 'GOOG': 0.62}

>>> # Full risk report
>>> from quant_trading.portfolio.risk_metrics import portfolio_risk_report
>>> report = portfolio_risk_report([0.01, -0.02, 0.03] * 60, risk_free_rate=0.02)
>>> print(report.sharpe_ratio, report.cvar_95, report.max_drawdown)
"""

from quant_trading.portfolio.optimizers import (
    # Optimizer base
    OptimizerBase,
    OptimizationResult,
    EfficientFrontierPoint,
    # Concrete optimizers
    MeanVarianceOptimizer,
    MaxSharpeOptimizer,
    MinVarianceOptimizer,
    RiskParityOptimizer,
    HRPOptimizer,
    BlackLittermanOptimizer,
    CVaROptimizer,
    # Factory
    optimize,
    OPTIMIZER_REGISTRY,
)

from quant_trading.portfolio.risk_metrics import (
    # Data structures
    DrawdownInfo,
    PortfolioRiskReport,
    # Individual metrics
    value_at_risk,
    conditional_var,
    cvar_from_var,
    conditional_drawdown_at_risk,
    cdar,
    cvar,
    max_drawdown_analysis,
    omega_ratio,
    lower_partial_moment,
    upper_partial_moment,
    kappa_ratio,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    tail_ratio,
    upside_potential_ratio,
    roys_safety_first,
    information_ratio,
    jensens_alpha,
    treynor_ratio,
    rolling_sharpe,
    skewness,
    kurtosis,
    # Full report
    portfolio_risk_report,
    risk_report,
)

__all__ = [
    # ---- Optimizers ----
    "OptimizerBase",
    "OptimizationResult",
    "EfficientFrontierPoint",
    "MeanVarianceOptimizer",
    "MaxSharpeOptimizer",
    "MinVarianceOptimizer",
    "RiskParityOptimizer",
    "HRPOptimizer",
    "BlackLittermanOptimizer",
    "CVaROptimizer",
    "optimize",
    "OPTIMIZER_REGISTRY",
    # ---- Risk metrics ----
    "DrawdownInfo",
    "PortfolioRiskReport",
    "value_at_risk",
    "conditional_var",
    "cvar_from_var",
    "conditional_drawdown_at_risk",
    "cdar",
    "cvar",
    "max_drawdown_analysis",
    "omega_ratio",
    "lower_partial_moment",
    "upper_partial_moment",
    "kappa_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "tail_ratio",
    "upside_potential_ratio",
    "roys_safety_first",
    "information_ratio",
    "jensens_alpha",
    "treynor_ratio",
    "rolling_sharpe",
    "skewness",
    "kurtosis",
    "portfolio_risk_report",
    "risk_report",
]
