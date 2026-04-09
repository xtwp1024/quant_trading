"""
Spot-Futures Arbitrage Module
=============================

期现套利模块 - 提供期现套利策略的核心框架

核心组件:
- spot_futures: 期现套利核心框架 (FuturesPricingArbitrageIntervals, SpotPortfolioConstruction)
- lead_lag: Lead-lag关系分析 (SpotFuturesRelationship)
- arbitrage_predictor: ML价差预测 (LSTM/BP/SVR)
- triangular: 三角套利引擎 (TriangularArbitrageEngine, TrianglePath, ArbitrageOpportunity)
- ibkr_pairs: Interactive Brokers 配对交易 (IBKRPairTrader, PairsExecutionManager, IBKRPairsMonitor)

典型用法:
    from quant_trading.arbitrage import SpotFuturesArbitrageur

    arb = SpotFuturesArbitrageur(spot_df, futures_df)
    interval = arb.calculate_arbitrage_interval()
    signal = arb.generate_signal(predicted_spread, interval)
"""

from quant_trading.arbitrage.spot_futures import (
    SpotPortfolioConstruction,
    FuturesPricingArbitrageIntervals,
    SpotFuturesArbitrageur,
)
from quant_trading.arbitrage.lead_lag import SpotFuturesRelationship
from quant_trading.arbitrage.arbitrage_predictor import ArbitragePredictor
from quant_trading.arbitrage.polymarket import (
    PolymarketArbitrageur,
    MarketSignal,
    ArbitrageSignal,
    PriceCandle,
    ProbabilityDeviationStrategy,
    LiquidityProvisionStrategy,
    FundingRateStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
)
from quant_trading.arbitrage.polymarket_api import (
    PolymarketClient,
    CrossPlatformFeeds,
)
from quant_trading.arbitrage.triangular import (
    TriangularArbitrageEngine,
    TrianglePath,
    ArbitrageOpportunity,
)
from quant_trading.arbitrage.options_stat_arb import (
    IVMeanReversionStrategy,
    OptionsStatArb,
    calculate_iv_rank,
    calculate_iv_percentile,
    detect_volatility_regime,
)
from quant_trading.arbitrage.ibkr_pairs import (
    IBKRPairTrader,
    RESTFallbackPairTrader,
    PairsExecutionManager,
    IBKRPairsMonitor,
    IBKRContract,
    IBKROrder,
    PairPosition,
    SpreadSignal,
    create_ibkr_trader,
)

__all__ = [
    "SpotPortfolioConstruction",
    "FuturesPricingArbitrageIntervals",
    "SpotFuturesArbitrageur",
    "SpotFuturesRelationship",
    "ArbitragePredictor",
    # Polymarket
    "PolymarketArbitrageur",
    "MarketSignal",
    "ArbitrageSignal",
    "PriceCandle",
    "ProbabilityDeviationStrategy",
    "LiquidityProvisionStrategy",
    "FundingRateStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "PolymarketClient",
    "CrossPlatformFeeds",
    # Triangular Arbitrage
    "TriangularArbitrageEngine",
    "TrianglePath",
    "ArbitrageOpportunity",
    # Options Statistical Arbitrage
    "IVMeanReversionStrategy",
    "OptionsStatArb",
    "calculate_iv_rank",
    "calculate_iv_percentile",
    "detect_volatility_regime",
    # IBKR Pairs Trading
    "IBKRPairTrader",
    "RESTFallbackPairTrader",
    "PairsExecutionManager",
    "IBKRPairsMonitor",
    "IBKRContract",
    "IBKROrder",
    "PairPosition",
    "SpreadSignal",
    "create_ibkr_trader",
]
