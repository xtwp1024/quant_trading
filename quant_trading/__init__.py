"""
量化之神 (God of Quant Trading)
Unified Quant Trading System combining the best of 5 projects

Architecture:
- Core: TitanBrain (RAG), TitanCortex (Self-healing), EventBus
- Five Force: Five Force Cognitive Architecture
- Gene Lab: Gene Bank + Strategy Breeder
- Strategy: Advanced academic paper strategies (Hawkes, Info Theory, PA-AMM, Onchain)
- Options: Black-Scholes pricing, Greeks, Options strategies
- Knowledge: Loss Recovery, Obsidian Integration, Sentiment Analysis
- Optimization: Kelly Calculator, Position Sizer, Strategy Selector
"""

__version__ = "1.0.0"
__author__ = "Quant God Team"

# Options modules - fully functional
from .options.pricing.black_scholes import BlackScholes
from .options.pricing.greeks import Greeks, calculate_greeks

# Optimization modules - fully functional
from .optimization.kelly_calculator import KellyCalculator
from .optimization import PositionSizer

# Arbitrage modules - spot-futures arbitrage framework
from .arbitrage.spot_futures import (
    SpotPortfolioConstruction,
    FuturesPricingArbitrageIntervals,
    SpotFuturesArbitrageur,
)
from .arbitrage.lead_lag import SpotFuturesRelationship
from .arbitrage.arbitrage_predictor import ArbitragePredictor

# Backtester modules — SuiteTrading v2 dual-runner engine
from .backtester.suite_engine import (
    SuiteEngine,
    BacktestDataset,
    StrategySignals,
    SuiteRiskConfig,
    run_fsm_backtest,
    run_simple_backtest,
)
from .backtester.position_fsm import PositionStateMachine
from .backtester.anti_overfit import AntiOverfitPipeline
from .backtester.optuna_optimizer import OptunaOptimizer, grid_search

__all__ = [
    # Options
    "BlackScholes",
    "Greeks",
    "calculate_greeks",
    # Optimization
    "KellyCalculator",
    "PositionSizer",
    # Arbitrage
    "SpotPortfolioConstruction",
    "FuturesPricingArbitrageIntervals",
    "SpotFuturesArbitrageur",
    "SpotFuturesRelationship",
    "ArbitragePredictor",
    # Backtester (SuiteTrading v2)
    "SuiteEngine",
    "BacktestDataset",
    "StrategySignals",
    "SuiteRiskConfig",
    "PositionStateMachine",
    "AntiOverfitPipeline",
    "OptunaOptimizer",
    "grid_search",
]
