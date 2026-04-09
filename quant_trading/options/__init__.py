"""
Options trading modules - 从 Options-Trading-Bot 仓库吸收整合
核心功能：
- Black-Scholes 定价 + Implied Volatility Newton-Raphson 求解器
- Greeks 计算 (Delta/Gamma/Vega/Theta/Rho + 高阶 Greeks)
- 策略: Straddle, Iron Condor, Strangle, RSI Momentum
- 完整风险管理 + 持仓周期管理
- Deribit Options Trading Bot (VavaBot)
"""

# 定价模块
from .pricing.black_scholes import (
    BlackScholes,
    bs_price,
    bs_greeks,
    black_scholes_greeks,
    implied_volatility,
    implied_volatility_newton_raphson,
    implied_volatility_bisection,
    calculate_iv_smile,
    volatility_smile_fit,
    svi_volatility,
    # 独立 Greeks 函数
    greeks,
    delta,
    gamma,
    vega,
    theta,
    rho,
)

from .pricing.greeks import (
    Greeks,
    calculate_greeks,
    calculate_portfolio_greeks,
    implied_volatility as iv_from_greeks,
)

# 策略模块
from .strategies import (
    OptionStrategy,
    OptionsStrategy,
    OptionPositionSide,
    StrategyType,
    StrategySignal,
    StrategyResult,
    # 策略类
    StraddleStrategy,
    IronCondorStrategy,
    StrangleStrategy,
    RSIMomentumStrategy,
    # 盈亏计算
    calculate_straddle_pnl,
    calculate_iron_condor_pnl,
    calculate_strangle_pnl,
    # 工具
    get_strategy,
    list_strategies,
    STRATEGY_REGISTRY,
)

# 风险管理模块
from .risk_manager import (
    RiskManager,
    OptionsRiskManager,
    RiskManager as RiskManagerClass,
    RiskLimit,
    RiskLimitType,
    RiskMetrics,
    PositionSizer,
    StopLoss,
    # 持仓周期管理
    PositionLifecycle,
    PositionCycleManager,
)

# VavaBot Options Trading 模块 (从 vavabot_options_strategy 吸收)
# 注: 以下为实际存在于 vavabot_options.py 中的类
from .vavabot_options import (
    DeribitOptionsAPI,
    DeribitWebSocketClient,
    DeribitAPIError,
    DeribitAuthError,
    BlackScholesGreeks,
    OptionContract,
    PositionMonitor,
    PositionMonitorError,
    VolatilitySurface,
    VavaBotOptionsStrategy,
)


__all__ = [
    # 定价
    "BlackScholes",
    "bs_price",
    "bs_greeks",
    "black_scholes_greeks",
    "implied_volatility",
    "implied_volatility_newton_raphson",
    "implied_volatility_bisection",
    "calculate_iv_smile",
    "volatility_smile_fit",
    "svi_volatility",
    # 独立 Greeks
    "greeks",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    # Greeks
    "Greeks",
    "calculate_greeks",
    "calculate_portfolio_greeks",
    # 策略
    "OptionStrategy",
    "OptionsStrategy",
    "OptionPositionSide",
    "StrategyType",
    "StrategySignal",
    "StrategyResult",
    "StraddleStrategy",
    "IronCondorStrategy",
    "StrangleStrategy",
    "RSIMomentumStrategy",
    "calculate_straddle_pnl",
    "calculate_iron_condor_pnl",
    "calculate_strangle_pnl",
    "get_strategy",
    "list_strategies",
    "STRATEGY_REGISTRY",
    # 风险管理
    "RiskManager",
    "OptionsRiskManager",
    "RiskLimit",
    "RiskLimitType",
    "RiskMetrics",
    "PositionSizer",
    "StopLoss",
    "PositionLifecycle",
    "PositionCycleManager",
    # VavaBot Options (实际存在于 vavabot_options.py)
    "DeribitOptionsAPI",
    "DeribitWebSocketClient",
    "DeribitAPIError",
    "DeribitAuthError",
    "BlackScholesGreeks",
    "OptionContract",
    "PositionMonitor",
    "PositionMonitorError",
    "VolatilitySurface",
    "VavaBotOptionsStrategy",
]
