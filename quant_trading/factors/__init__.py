"""
quant_trading.factors — Formulaic alpha library, evaluator, sentiment, ML predictors,
high-frequency tick factors, and multi-agent alpha system.

Key exports
-----------
Alpha Library
- Alpha101              : all 101 formulaic alphas (Kakushadze, 2016)
- AlphaEvaluator        : IC, turnover, half-life, correlation analysis

IC/IR Factor Analysis
- ICAnalyzer            : IC/IR computation, rolling IC, factor selection, IC decay

High-Frequency Factors (Chinese A-Share tick data)
- HighFrequencyFactors : all 39 high-frequency factors (A1-A39)
- FactorCalculator      : core calculator with all 39 factor methods
- OrderFlowFactors      : order-side factors A1-A27 (arrival, buy/sell, cancel, ratios)
- TradeFlowFactors      : trade-side factors A28-A39 (VWAP, direction, imbalance)
  Key derived metrics:
    - trade_imbalance    : (V_buy - V_sell) / (V_buy + V_sell)
    - order_trade_ratio  : # orders / # trades (60s window)

MyTT — Pure Python Technical Indicators (zero Ta-Lib dependency)
- Core helpers    : RD, RET, ABS, MAX, MIN, MA, REF, DIFF, STD, IF, SUM,
                    HHV, LLV, EMA, SMA, AVEDEV, SLOPE
- Composite       : COUNT, EVERY, LAST, EXIST, BARSLAST, FORCAST, CROSS
- Indicators      : MACD, KDJ, RSI, WR, BIAS, BOLL, PSY, CCI, ATR, BBI,
                    DMI, TURTLES, KTN, TRIX, VR, EMV, DPO, BRAR, DMA,
                    MTM, MASS, ROC, EXPMA, OBV, MFI, ASI

Sentiment & ML
- SentimentAgent     : news-based sentiment (OpenAI GPT, optional; rule-based fallback)
- LSTMPredictor      : LSTM / XGBoost prediction wrapper

Alpha Agent System
- AlphaAgent         : top-level orchestrator
- CommunicationManager : thread-safe pub/sub + direct messaging
- ProphetPredictor   : Facebook/Meta Prophet time-series forecasting
- DataAgent          : OHLCV ingestion and normalisation
- PredictionAgent    : LSTM + XGBoost + Prophet ensemble predictions
- SignalAgent        : formulaic alpha computation and dynamic factor selection
- RiskAgent          : volatility, drawdown, and VaR risk overlays

LOB Microstructure & HFT Prediction
- LOBFeatureEngine       : LOB tick-by-tick feature engineering (spread, imbalance,
                            diffs, lags, rolling stats over count & time windows)
- LOBPreprocessor        : raw LOB data cleaning, type enforcement, null-filling
- HFTPredictor           : LightGBM + Random Forest ensemble for 1s/3s/5s
                            limit-order-book price movement classification (76-78% accuracy)
- HFTLabel               : label constants (NO_CHANGE=0, BID_DOWN=1, ASK_UP=2)

Pairs Trading (KalmanBOT_ICASSP23)
- KalmanFilter           : Traditional linear Kalman filter for state estimation
- LinearSystemModel      : Linear Gaussian state-space model
- KalmanNetNN            : GRU-based adaptive Kalman filter (ICASSP 2023)
- KalmanNetSystemModel   : System model for KalmanNet
- KalmanFilterStrategy   : Pairs trading with linear KF hedge ratio
- KalmanNetStrategy      : Pairs trading with KalmanNet (deep learning enhanced)
- BollingerSignal        : Bollinger band entry/exit signals
- HedgeRatioEstimator    : sklearn-compatible Kalman Filter hedge ratio estimator

GARCH Volatility Modeling (IBApi-GARCH-CrackSpreadTrading-Algo)
- GARCHModel             : GARCH/GARCH-M/EGARCH/TGARCH/PGARCH family (pure NumPy/SciPy)
- GARCHSignalGenerator   : z-score trigger signal generator (+1 mean-reversion, -1 trend)
"""

from quant_trading.factors.alpha_101 import Alpha101
from quant_trading.factors.alpha_evaluator import AlphaEvaluator
from quant_trading.factors.alpha101 import (
    ALPHA_FACTORS,
    ALPHA_NAMES,
    Alpha101,
    Alpha101Compute,
    rank,
    ts_rank,
    correlation,
    covariance,
    decay_linear,
    delta,
    ts_argmax,
    ts_argmin,
    ts_max,
    ts_min,
    delay,
    signedpower,
)
from quant_trading.factors.high_freq import HighFrequencyFactors, FactorCalculator
from quant_trading.factors.order_flow import OrderFlowFactors
from quant_trading.factors.trade_flow import TradeFlowFactors
from quant_trading.factors.hfreq_features import (
    # Core microstructure
    order_imbalance, vpin, flow_toxicity, spread_decomposition,
    order_flow_skew, trade_direction, quote_velocity, trade_activity_ratio,
    price_impact, midprice_ma_diff, volume_curve_slope,
    order_arrival_intensity, price_reversion, volatility_ratio,
    trade_size_imbalance, quote_cluster_quality, liquidity_score,
    # Order-side A1-A16
    order_arrival_rate, cum_order_count, order_volume_sum, cum_order_volume,
    buy_order_arrival_rate, sell_order_arrival_rate, buy_order_volume,
    sell_order_volume, fill_kill_count, cancel_count, cancel_volume,
    buy_cancel_count, sell_cancel_count, buy_cancel_volume, sell_cancel_volume,
    cum_cancel_count,
    # VWAP A17-A19
    cancel_vwap, cancel_buy_vwap, cancel_sell_vwap,
    # Ratios A20-A27
    cancel_rate, cancel_volume_rate, buy_cancel_rate, sell_cancel_rate,
    avg_order_size, avg_buy_order_size, avg_sell_order_size, order_size_std,
    # Trade-side A28-A39
    trade_arrival_rate, cum_trade_count, trade_volume_sum, cum_trade_volume,
    buy_trade_volume, sell_trade_volume, trade_imbalance, order_trade_ratio,
    buy_trade_ratio, trade_size_avg, trade_size_skew, trade_vwap,
)
from quant_trading.factors.lob_processor import LOBProcessor
from quant_trading.factors.mytt import (
    # Level 0 — core helpers
    RD, RET, ABS, MAX, MIN, MA, REF, DIFF, STD, IF, SUM,
    HHV, LLV, EMA, SMA, AVEDEV, SLOPE,
    # Level 1 — composite helpers
    COUNT, EVERY, LAST, EXIST, BARSLAST, FORCAST, CROSS,
    # Level 2 — technical indicators
    MACD, KDJ, RSI, WR, BIAS, BOLL, PSY, CCI, ATR, BBI,
    DMI, TURTLES, KTN, TRIX, VR, EMV, DPO, BRAR, DMA,
    MTM, MASS, ROC, EXPMA, OBV, MFI, ASI,
)
from quant_trading.factors.sentiment_agent import SentimentAgent
from quant_trading.factors.lstm_predictor import LSTMPredictor, XGBoostPredictor as _XGBoostFromLSTM
from quant_trading.factors.ic_analyzer import ICAnalyzer
from quant_trading.factors.xgb_predictor import XGBoostPredictor
from quant_trading.factors.alpha_agent import (
    AlphaAgent,
    CommunicationManager,
    ProphetPredictor,
    DataAgent,
    PredictionAgent,
    SignalAgent,
    RiskAgent,
    # AlphaFactor system (new)
    AlphaFactor,
    Alpha101Bundle,
    FactorCategory,
    MarketRegime,
    # Key alpha factor classes
    Alpha001, Alpha002, Alpha003, Alpha004,
    Alpha006, Alpha007, Alpha008, Alpha009, Alpha010,
    Alpha012, Alpha013, Alpha014, Alpha015, Alpha016,
    Alpha017, Alpha018, Alpha019, Alpha020, Alpha021,
    Alpha023, Alpha024, Alpha026, Alpha027, Alpha028,
    Alpha029, Alpha031, Alpha032, Alpha033, Alpha036,
    Alpha037, Alpha039, Alpha040, Alpha044, Alpha046,
    Alpha051, Alpha056, Alpha057, Alpha059, Alpha060,
    Alpha071, Alpha074, Alpha101,
    # Utility functions
    neutralize_zscore,
    neutralize_rank,
    compute_ic,
    compute_ir,
    compute_rolling_ic,
)
from quant_trading.factors.hft_predictor import HFTPredictor, HFTLabel
from quant_trading.factors.lob_features import LOBFeatureEngine, LOBPreprocessor

# Kalman filter & pairs trading (KalmanBOT_ICASSP23)
from quant_trading.factors.kalman_filter import (
    KalmanFilter,
    LinearSystemModel,
    KalmanNetNN,
    KalmanNetSystemModel,
    KNetDelta,
)
from quant_trading.factors.pairs_trading import (
    KalmanFilterStrategy,
    KalmanNetStrategy,
    BollingerSignal,
    LearnableBollingerSignal,
    HedgeRatioEstimator,
    PairsTradingStrategy,
    PairsPosition,
    prepare_forex_data,
    compute_innovation_portfolio,
)

# GARCH volatility modeling (IBApi-GARCH-CrackSpreadTrading-Algo source)
from quant_trading.factors.garch_model import GARCHModel, GARCHSignalGenerator

# HMM Regime Detection (pure NumPy Gaussian HMM, no hmmlearn required)
from quant_trading.factors.hmm_regime import (
    MarketRegime,
    RegimeState,
    HMMRegimeDetector,
    RegimeSwitchingStrategy,
)

# Crypto Absorption Factors (absorbed from finclaw)
from quant_trading.factors.crypto_absorption import (
    CryptoAbsorption,
    compute_absorption,
    compute_absorption_series,
)
from quant_trading.factors.crypto_market_timing import (
    CryptoMarketTiming,
    compute_timing_score,
    compute_timing_series,
)

__all__ = [
    # Alpha library
    "Alpha101",
    # New alpha101 module
    "ALPHA_FACTORS",
    "ALPHA_NAMES",
    "Alpha101Compute",
    # Alpha helpers
    "rank", "ts_rank", "correlation", "covariance", "decay_linear",
    "delta", "ts_argmax", "ts_argmin", "ts_max", "ts_min",
    "delay", "signedpower",
    # Evaluator
    "AlphaEvaluator",
    # IC/IR factor analysis
    "ICAnalyzer",
    # High-frequency factors
    "HighFrequencyFactors",
    "FactorCalculator",
    "OrderFlowFactors",
    "TradeFlowFactors",
    # High-frequency factor library (pure NumPy/Pandas)
    # Core microstructure
    "order_imbalance", "vpin", "flow_toxicity", "spread_decomposition",
    "order_flow_skew", "trade_direction", "quote_velocity", "trade_activity_ratio",
    "price_impact", "midprice_ma_diff", "volume_curve_slope",
    "order_arrival_intensity", "price_reversion", "volatility_ratio",
    "trade_size_imbalance", "quote_cluster_quality", "liquidity_score",
    # Order-side A1-A16
    "order_arrival_rate", "cum_order_count", "order_volume_sum", "cum_order_volume",
    "buy_order_arrival_rate", "sell_order_arrival_rate", "buy_order_volume",
    "sell_order_volume", "fill_kill_count", "cancel_count", "cancel_volume",
    "buy_cancel_count", "sell_cancel_count", "buy_cancel_volume", "sell_cancel_volume",
    "cum_cancel_count",
    # VWAP A17-A19
    "cancel_vwap", "cancel_buy_vwap", "cancel_sell_vwap",
    # Ratios A20-A27
    "cancel_rate", "cancel_volume_rate", "buy_cancel_rate", "sell_cancel_rate",
    "avg_order_size", "avg_buy_order_size", "avg_sell_order_size", "order_size_std",
    # Trade-side A28-A39
    "trade_arrival_rate", "cum_trade_count", "trade_volume_sum", "cum_trade_volume",
    "buy_trade_volume", "sell_trade_volume", "trade_imbalance", "order_trade_ratio",
    "buy_trade_ratio", "trade_size_avg", "trade_size_skew", "trade_vwap",
    # LOB Processor
    "LOBProcessor",
    # MyTT — pure-Python technical indicators
    # Level 0 — core helpers
    "RD", "RET", "ABS", "MAX", "MIN",
    "MA", "REF", "DIFF", "STD", "IF", "SUM",
    "HHV", "LLV", "EMA", "SMA", "AVEDEV", "SLOPE",
    # Level 1 — composite helpers
    "COUNT", "EVERY", "LAST", "EXIST", "BARSLAST", "FORCAST", "CROSS",
    # Level 2 — technical indicators
    "MACD", "KDJ", "RSI", "WR", "BIAS", "BOLL", "PSY", "CCI",
    "ATR", "BBI", "DMI", "TURTLES", "KTN", "TRIX", "VR", "EMV",
    "DPO", "BRAR", "DMA", "MTM", "MASS", "ROC", "EXPMA", "OBV",
    "MFI", "ASI",
    # Sentiment
    "SentimentAgent",
    # ML predictors
    "LSTMPredictor",
    "XGBoostPredictor",
    # Alpha Agent system
    "AlphaAgent",
    "CommunicationManager",
    "ProphetPredictor",
    "DataAgent",
    "PredictionAgent",
    "SignalAgent",
    "RiskAgent",
    # AlphaFactor system (absorbed from alpha-agent repo)
    "AlphaFactor",
    "Alpha101Bundle",
    "FactorCategory",
    "MarketRegime",
    # Key alpha factor classes
    "Alpha001", "Alpha002", "Alpha003", "Alpha004",
    "Alpha006", "Alpha007", "Alpha008", "Alpha009", "Alpha010",
    "Alpha012", "Alpha013", "Alpha014", "Alpha015", "Alpha016",
    "Alpha017", "Alpha018", "Alpha019", "Alpha020", "Alpha021",
    "Alpha023", "Alpha024", "Alpha026", "Alpha027", "Alpha028",
    "Alpha029", "Alpha031", "Alpha032", "Alpha033", "Alpha036",
    "Alpha037", "Alpha039", "Alpha040", "Alpha044", "Alpha046",
    "Alpha051", "Alpha056", "Alpha057", "Alpha059", "Alpha060",
    "Alpha071", "Alpha074", "Alpha101",
    # Utility functions
    "neutralize_zscore",
    "neutralize_rank",
    "compute_ic",
    "compute_ir",
    "compute_rolling_ic",
    # LOB Microstructure & HFT Prediction
    "LOBFeatureEngine",
    "LOBPreprocessor",
    "HFTPredictor",
    "HFTLabel",
    # Kalman filter & pairs trading (KalmanBOT_ICASSP23)
    "KalmanFilter",
    "LinearSystemModel",
    "KalmanNetNN",
    "KalmanNetSystemModel",
    "KNetDelta",
    "KalmanFilterStrategy",
    "KalmanNetStrategy",
    "BollingerSignal",
    "LearnableBollingerSignal",
    "HedgeRatioEstimator",
    "PairsTradingStrategy",
    "PairsPosition",
    "prepare_forex_data",
    "compute_innovation_portfolio",
    # HMM Regime Detection
    "MarketRegime",
    "RegimeState",
    "HMMRegimeDetector",
    "RegimeSwitchingStrategy",
    # GARCH volatility modeling
    "GARCHModel",
    "GARCHSignalGenerator",
    # Crypto Absorption Factors (finclaw)
    "CryptoAbsorption",
    "compute_absorption",
    "compute_absorption_series",
    "CryptoMarketTiming",
    "compute_timing_score",
    "compute_timing_series",
]
