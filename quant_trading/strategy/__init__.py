"""Strategy modules - Advanced technical analysis theories"""
from .base import BaseStrategy
from .loader import StrategyLoader

# Advanced theory modules (absorbed from AbuQuant)
from .chan_theory import (
    ChanTheoryAnalyzer,
    Bi,
    XSegment,
    ZhongShu,
    TrendType,
    BeiChi,
    FenXing,
    KLineDirection,
)
from .elliott_wave import (
    ElliottWaveAnalyzer,
    Wave,
    WaveStructure,
    WaveDegree,
    WaveType,
    FibonacciLevel,
)
from .harmonic import (
    HarmonicPatternRecognizer,
    HarmonicPattern,
    HarmonicStructure,
    HarmonicPoint,
)
from .candlestick_patterns import (
    CandlestickPatternAnalyzer,
    CandlePattern,
    CandlePatternResult,
    CandleStatistics,
)
from .profit_hunter import ProfitHunterStrategy

# KalmanBOT — Kalman filter & KalmanNet for pairs trading / mean reversion
from .kalman_bot import (
    KalmanFilter,
    KalmanNet,
    SpreadSignalGenerator,
    PairsTradingStrategy,
    MeanReversionStrategy,
    create_pairs_strategy,
)

# US Portfolio Strategy Backtesting (absorbed from PortfolioStrategyBacktestUS)
from .portfolio_backtest_us import (
    VolatilityRegimeDetector,
    VolatilityThresholdGate,
    MeanVarianceOptimizer,
    RiskParityStrategy,
    PortfolioBacktester,
)

# Binance Algorithmic Trading - GA-optimized strategies
from .binance_algo import (
    # Environment
    TradingGymEnv,
    TradingState,
    BacktestResult,
    # Strategies
    MACrossoverStrategy,
    RSIStrategy,
    BollingerStrategy,
    # Optimizers
    GeneticOptimizer,
    StrategyOptimizer,
    # Indicators
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_atr,
    # Constants
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
    STRATEGY_MA_CROSSOVER,
    STRATEGY_RSI,
    STRATEGY_BOLLINGER,
)

# HMM Regime — Gaussian HMM (pure NumPy, no hmmlearn) for market regime detection
from .hmm_regime import (
    GaussianHMM,
    MarketRegimeDetector,
    MarketRegimeDetectorConfig,
    RegimeAwareStrategy,
    RegimeAwareStrategyParams,
)

# Copula Pairs Trading — Gaussian/Student-t copula + DCC-GARCH (pure NumPy + scipy)
from .copula_pairs import (
    GaussianCopula,
    StudentTCopula,
    DCCGARCHModel,
    CopulaPairsStrategy,
    CopulaPairsParams,
    to_uniform_margins,
    kendall_tau,
    select_best_copula,
)

# Adaptive Multi-Regime Engine (absorbed from finclaw)
from .advanced.adaptive_regime_engine import (
    AdaptiveRegimeEngine,
    AdaptiveRegimeParams,
    MarketRegime,
    RegimeSignalResult,
)

# HFT Strategies — High-Frequency Trading (absorbed from HFT-strategies repo)
from .hft_strategies import (
    SignalType,
    HFTOrder,
    Position,
    OrderBook,
    TradeTick,
    SpreadCaptureStrategy,
    MomentumSignalStrategy,
    OrderBookImbalanceStrategy,
    LatencyArbitrageStrategy,
    HFTPositionManager,
)

# 3Commas Cyber Bots — DCA bot with trailing stop loss and profit compounding
# (absorbed from 3commas-cyber-bots repo)
from .commas_cyber import (
    # Core Classes (verified to exist)
    ThreeCommasAPI,
    TrailingStopLoss,
    CompoundStrategy,
    AltRankStrategy,
    GalaxyScoreStrategy,
    DCABotStrategy,
    DealCluster,
    MarketCollector,
)

__all__ = [
    # Base
    "BaseStrategy",
    "StrategyLoader",
    # Chan Theory (缠论)
    "ChanTheoryAnalyzer",
    "Bi",
    "XSegment",
    "ZhongShu",
    "TrendType",
    "BeiChi",
    "FenXing",
    "KLineDirection",
    # Elliott Wave (艾略特波浪)
    "ElliottWaveAnalyzer",
    "Wave",
    "WaveStructure",
    "WaveDegree",
    "WaveType",
    "FibonacciLevel",
    # Harmonic (谐波)
    "HarmonicPatternRecognizer",
    "HarmonicPattern",
    "HarmonicStructure",
    "HarmonicPoint",
    # Candlestick (K线形态)
    "CandlestickPatternAnalyzer",
    "CandlePattern",
    "CandlePatternResult",
    "CandleStatistics",
    # Profit Hunter (价格异常检测)
    "ProfitHunterStrategy",
    # KalmanBOT — Kalman filter / KalmanNet pairs trading
    "KalmanFilter",
    "KalmanNet",
    "SpreadSignalGenerator",
    "PairsTradingStrategy",
    "MeanReversionStrategy",
    "create_pairs_strategy",
    # US Portfolio Strategy Backtesting
    "VolatilityRegimeDetector",
    "VolatilityThresholdGate",
    "MeanVarianceOptimizer",
    "RiskParityStrategy",
    "PortfolioBacktester",
    # Binance Algorithmic Trading
    "TradingGymEnv",
    "TradingState",
    "BacktestResult",
    "MACrossoverStrategy",
    "RSIStrategy",
    "BollingerStrategy",
    "GeneticOptimizer",
    "StrategyOptimizer",
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_bollinger_bands",
    "calculate_macd",
    "calculate_atr",
    "ACTION_HOLD",
    "ACTION_BUY",
    "ACTION_SELL",
    "STRATEGY_MA_CROSSOVER",
    "STRATEGY_RSI",
    "STRATEGY_BOLLINGER",
    # HMM Regime — Gaussian HMM (pure NumPy) for market regime detection
    "GaussianHMM",
    "MarketRegimeDetector",
    "MarketRegimeDetectorConfig",
    "RegimeAwareStrategy",
    "RegimeAwareStrategyParams",
    # Copula Pairs Trading
    "GaussianCopula",
    "StudentTCopula",
    "DCCGARCHModel",
    "CopulaPairsStrategy",
    "CopulaPairsParams",
    "to_uniform_margins",
    "kendall_tau",
    "select_best_copula",
    # HFT Strategies
    "SignalType",
    "HFTOrder",
    "Position",
    "OrderBook",
    "TradeTick",
    "SpreadCaptureStrategy",
    "MomentumSignalStrategy",
    "OrderBookImbalanceStrategy",
    "LatencyArbitrageStrategy",
    "HFTPositionManager",
    # Adaptive Multi-Regime Engine (finclaw)
    "AdaptiveRegimeEngine",
    "AdaptiveRegimeParams",
    "MarketRegime",
    "RegimeSignalResult",
    # 3Commas Cyber Bots — DCA + Trailing Stop + Compound
    "ThreeCommasAPI",
    "TrailingStopLoss",
    "CompoundStrategy",
    "AltRankStrategy",
    "GalaxyScoreStrategy",
    "DCABotStrategy",
    "DealCluster",
    "MarketCollector",
]
