"""Signal module — types, generators, fusion, quality, and state.

Exports
-------
Types
    SignalDirection : LONG / SHORT / NEUTRAL
    SignalType      : BUY / SELL / EXIT_LONG / EXIT_SHORT / CLOSE_ALL / HOLD
    Signal          : Main signal dataclass (compatible with both paradigms)

Generators
    SignalGenerator  : Abstract base for all generators
    RSIGenerator     : RSI overbought/oversold signals
    MACDGenerator    : MACD line crossover signals
    BollingerGenerator : Bollinger Band bounce signals
    VolumeGenerator  : Volume spike signals
    ATRGenerator     : ATR trend / trailing-stop signals
    MultiGenerator   : Weighted multi-indicator generator
    TimesFMGenerator : Google TimesFM 2.5 时间序列预测信号

Fusion
    FusionStrategy           : Abstract fusion base
    WeightedAverageFusion     : Strength-weighted average + threshold
    VotingFusion              : Majority vote
    EnsembleFusion            : All-generators-must-agree
    BayesianFusion            : Bayesian probabilistic fusion
    SignalCombinator          : User-facing fusion wrapper

Quality
    SignalMetrics : Quality metrics dataclass
    SignalQuality : Computes quality from signal + price series
    SignalFilter  : Filters signals by quality thresholds

State
    SignalState : Current position state from signal stream
    SignalStore : Chronological signal history per symbol
    SignalCache : LRU cache for computed signals
"""
from quant_trading.signal_modules.types import (
    SignalDirection,
    SignalType,
    Signal,
)
from quant_trading.signal_modules.generators import (
    SignalGenerator,
    RSIGenerator,
    MACDGenerator,
    BollingerGenerator,
    VolumeGenerator,
    ATRGenerator,
    MultiGenerator,
)
from quant_trading.signal_modules.timesfm_generator import TimesFMGenerator
from quant_trading.signal_modules.combinator import (
    FusionStrategy,
    WeightedAverageFusion,
    VotingFusion,
    EnsembleFusion,
    BayesianFusion,
    SignalCombinator,
)
from quant_trading.signal_modules.quality import (
    SignalMetrics,
    SignalQuality,
    SignalFilter,
)
from quant_trading.signal_modules.cache import (
    SignalState,
    SignalStore,
    SignalCache,
)
from quant_trading.signal_modules.stock_pool import (
    StockPoolManager,
    StockPoolClassifier,
    PoolType,
    PoolClassification,
    AStockDataProvider,
)

__all__ = [
    # Types
    "SignalDirection",
    "SignalType",
    "Signal",
    # Generators
    "SignalGenerator",
    "RSIGenerator",
    "MACDGenerator",
    "BollingerGenerator",
    "VolumeGenerator",
    "ATRGenerator",
    "MultiGenerator",
    "TimesFMGenerator",
    # Fusion
    "FusionStrategy",
    "WeightedAverageFusion",
    "VotingFusion",
    "EnsembleFusion",
    "BayesianFusion",
    "SignalCombinator",
    # Quality
    "SignalMetrics",
    "SignalQuality",
    "SignalFilter",
    # State
    "SignalState",
    "SignalStore",
    "SignalCache",
    # Stock Pool
    "StockPoolManager",
    "StockPoolClassifier",
    "PoolType",
    "PoolClassification",
    "AStockDataProvider",
]
