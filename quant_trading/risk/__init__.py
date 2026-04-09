"""Risk management module.

Adopted from finclaw risk library:
- VaRCalculator, VaRResult: historical and parametric VaR/CVaR
- RiskMetrics (as RiskMetricsCalculator), RiskReport, DrawdownInfo: core risk metrics
- AdvancedRiskMetrics: omega, tail ratio, downside deviation, info/treynor ratios, capture ratios
- PortfolioOptimizer: mean-variance, max sharpe, min variance, risk-parity, Black-Litterman, efficient frontier

Adopted from Quantropy/matilda library (quant_trading/risk/matilda_risk.py):
- var_historical, var_variance_covariance, cvar_expected_shortfall
- lower_partial_moments, upper_partial_moments (LPM/HPM)
- kappa_ratio, kappa_var_ratio, kappa_cvar_ratio, kappa_target_ratio
- omega_ratio (LPM-based), sortino_ratio (LPM-based)
- modigliani_ratio, information_ratio, treynor_ratio, jensens_alpha
- roys_safety_first, rolling_sharpe, drawdown_analysis
- gain_loss_ratio, upside_potential_ratio, conditional_sharpe_ratio, excess_return_var

Portfolio optimization (quant_trading/risk/matilda_portfolio.py):
- HierarchicalRiskParity, PostModernPortfolioTheory, RiskParityModel
- BlackLittermanModel, EfficientFrontier
"""

from .manager import RiskManager, RiskConfig, RiskMetrics as RiskMetricsData, RiskLevel
from .maverick_risk import PositionSizeTool, TechnicalStopsTool, RiskMetricsTool

# Volatility gate risk control (adopted from five_factors_riskControl + PortfolioStrategyBacktestUS)
from .volatility_gate import VolatilityGate
from .vol_adaptive import VolAdaptiveRiskManager

# VaR
from .var_calculator import VaRCalculator, VaRResult
from .var_calculator import historical_var, parametric_var, cvar_var

# Core risk metrics (aliased to avoid conflict with RiskMetricsData from manager.py)
from .risk_metrics import RiskMetrics as RiskMetricsCalculator
from .risk_metrics import RiskReport, DrawdownInfo
from .risk_metrics import sharpe, sortino, calmar, max_drawdown, win_rate, profit_factor

# Advanced metrics
from .advanced_metrics import AdvancedRiskMetrics
from .advanced_metrics import omega_ratio, tail_ratio, downside_deviation
from .advanced_metrics import information_ratio, treynor_ratio, capture_ratios

# Portfolio optimizer
from .portfolio_optimizer import PortfolioOptimizer

# ---- Matilda risk quantification (unique: LPM/HPM, Kappa, Modigliani, Roy, HRP) ----
from .matilda_risk import (
    var_historical,
    var_variance_covariance,
    cvar_expected_shortfall,
    lower_partial_moments,
    upper_partial_moments,
    lpm_n,
    hpm_n,
    kappa_ratio,
    kappa_var_ratio,
    kappa_cvar_ratio,
    kappa_target_ratio,
    omega_ratio as matilda_omega_ratio,
    sortino_ratio as matilda_sortino_ratio,
    modigliani_ratio,
    information_ratio as matilda_information_ratio,
    treynor_ratio as matilda_treynor_ratio,
    jensens_alpha,
    roys_safety_first,
    rolling_sharpe,
    drawdown_analysis,
    gain_loss_ratio,
    upside_potential_ratio,
    conditional_sharpe_ratio,
    excess_return_var,
)

# ---- Matilda portfolio optimization (unique: HRP, PMPT) ----
from .matilda_portfolio import (
    HierarchicalRiskParity,
    PostModernPortfolioTheory,
    RiskParityModel,
    BlackLittermanModel,
    EfficientFrontier,
    PortfolioWeights,
    EfficientFrontierPoint,
)

# Aliases for RiskManager compatibility
VaRCalculator = VaRCalculator
RiskMetrics = RiskMetricsCalculator

# ---- OptionSuite Greeks-based risk (adopted from D:/Hive/Data/trading_repos/OptionSuite/) ----
from .option_suite_risk import (
    OptionPosition,
    PortfolioGreeks,
    GreeksAggregator,
    MarginCalculator,
    RiskScenarioAnalyzer,
    ScenarioResult,
    RiskReport as OptionSuiteRiskReport,
)

# GARCH volatility models (pure NumPy + scipy.optimize)
from .garch_risk import (
    GARCHModel,
    EGARCHModel,
    GJRGARCHModel,
    VolatilityForecaster,
    RiskMetrics as GARCHRiskMetrics,
    fit_garch11,
    forecast_volatility,
    compute_garch_var,
)

__all__ = [
    # Existing
    "RiskManager",
    "RiskConfig",
    "RiskMetricsData",
    "RiskLevel",
    "PositionSizeTool",
    "TechnicalStopsTool",
    "RiskMetricsTool",
    # Volatility gate (波动率门限风控)
    "VolatilityGate",
    "VolAdaptiveRiskManager",
    # VaR
    "VaRCalculator",
    "VaRResult",
    "historical_var",
    "parametric_var",
    "cvar_var",
    # Core risk metrics (aliased)
    "RiskMetricsCalculator",
    "RiskReport",
    "DrawdownInfo",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    # Advanced metrics
    "AdvancedRiskMetrics",
    "omega_ratio",
    "tail_ratio",
    "downside_deviation",
    "information_ratio",
    "treynor_ratio",
    "capture_ratios",
    # Portfolio optimizer
    "PortfolioOptimizer",
    # Aliases for RiskManager chain
    "VaRCalculator",
    "RiskMetrics",
    # Matilda risk (unique: LPM/HPM, Kappa, Modigliani, Roy, rolling Sharpe)
    "var_historical",
    "var_variance_covariance",
    "cvar_expected_shortfall",
    "lower_partial_moments",
    "upper_partial_moments",
    "lpm_n",
    "hpm_n",
    "kappa_ratio",
    "kappa_var_ratio",
    "kappa_cvar_ratio",
    "kappa_target_ratio",
    "matilda_omega_ratio",
    "matilda_sortino_ratio",
    "modigliani_ratio",
    "matilda_information_ratio",
    "matilda_treynor_ratio",
    "jensens_alpha",
    "roys_safety_first",
    "rolling_sharpe",
    "drawdown_analysis",
    "gain_loss_ratio",
    "upside_potential_ratio",
    "conditional_sharpe_ratio",
    "excess_return_var",
    # Matilda portfolio (unique: HRP, PMPT)
    "HierarchicalRiskParity",
    "PostModernPortfolioTheory",
    "RiskParityModel",
    "BlackLittermanModel",
    "EfficientFrontier",
    "PortfolioWeights",
    "EfficientFrontierPoint",
    # GARCH volatility models (pure NumPy)
    "GARCHModel",
    "EGARCHModel",
    "GJRGARCHModel",
    "VolatilityForecaster",
    "GARCHRiskMetrics",
    "fit_garch11",
    "forecast_volatility",
    "compute_garch_var",
    # OptionSuite Greeks-based risk
    "OptionPosition",
    "PortfolioGreeks",
    "GreeksAggregator",
    "MarginCalculator",
    "RiskScenarioAnalyzer",
    "ScenarioResult",
    "OptionSuiteRiskReport",
]
