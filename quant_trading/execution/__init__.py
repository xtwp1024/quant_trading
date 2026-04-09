"""Execution module — order execution, commission, position tracking, and routing.

Provides adapters for: Freqtrade, Hummingbot, PyCryptoBot, Lumibot, OctoBot.
Plus: commission models, position tracking, and smart order routing.
"""
from .executor import Executor
from .order_engine import OrderEngine
from .commission import (
    CommissionModel,
    FixedCommission,
    MakerTakerCommission,
    TieredCommission,
    CryptoCommission,
    BINANCE_SPOT_COMMISSION,
    BINANCE_FUTURES_COMMISSION,
    COINBASE_COMMISSION,
    KRAKEN_COMMISSION,
)
from .position_tracker import (
    Position,
    PositionState,
    PositionTracker,
)
from .router import (
    ExchangeQuote,
    RoutedOrder,
    SmartOrderRouter,
)
from .freqtrade_adapter import (
    FreqtradeExchangeAdapter,
    FreqtradeMultiExchangeAdapter,
    HAS_FREQTRADE,
)
from .freqtrade_strategy import (
    FreqtradeSignal,
    BaseStrategy,
    FreqtradeStrategyWrapper,
    IStrategyAdapter,
    StrategyRunner,
    create_strategy_from_template,
)
from .freqtrade_wrapper import (
    HAS_FREQTRADE as _FT_WRAPPER_HAS_FREQTRADE,
    FreqtradeStrategy,
    EdgePositionSizer,
    EMASpreadStrategy,
    MACDStrategy,
    PriceBandStrategy,
    FreqtradeExecutor,
    SignalDirection,
    EntryExit,
    ROI,
    StrategyConfig,
    TradeSignal,
    Position,
)
from .hummingbot_adapter import (
    HummingbotExchangeAdapter,
    HummingbotMultiExchangeAdapter,
    HummingbotConfig,
    MarketDataCollector,
    ArbitrageDetector,
    ExecutionRouter,
    ArbitrageOpportunity,
    BestExchange,
    OrderType,
    TradeType,
    PositionSide,
    SUPPORTED_EXCHANGES,
    EXCHANGE_NAME_MAPPING,
    EXCHANGE_DISPLAY_NAMES,
)
from .pycryptobot import (
    PyCryptoBotConfig,
    AppState,
    MarketData,
    Granularity,
    Action,
    BaseStrategy,
    EMAStrategy,
    THUMBStrategy,
    SMARoCStrategy,
    QuadencyConnector,
    PyCryptoBotExecutor,
    calculate_ema,
    calculate_sma,
    calculate_macd,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_obv,
    calculate_atr,
)
from .lumibot_adapter import (
    # Core entities / 核心实体
    Asset,
    Order,
    Position,
    Bar,
    Bars,
    # Enums / 枚举
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    # Broker classes / 券商类
    LumibotBroker,
    PolygonBroker,
    AlpacaBroker,
    # Data source classes / 数据源类
    PolygonDataSource,
    AlpacaDataSource,
    # Strategy / 策略
    LumibotStrategy,
    # Backtest / 回测
    LumibotBacktest,
    # Exceptions / 异常
    LumibotBrokerAPIError,
    LumibotDataError,
)
from .octobot_core import (
    # Config / 配置
    OctoBotConfig,
    # Data classes / 数据类
    TradeOrder,
    ExecutionReport,
    # Enums / 枚举
    TradeSignalType,
    AlertLevel,
    ExchangeSyncMode,
    # Main classes / 主类
    TelegramNotifier,
    TradingAligner,
    OctoBotExecutor,
)
from .dynamic_spread import (
    DynamicSpreadPricing,
    DynamicSpreadConfig,
)

__all__ = [
    # Core executor
    "Executor",
    "OrderEngine",
    # Commission models
    "CommissionModel",
    "FixedCommission",
    "MakerTakerCommission",
    "TieredCommission",
    "CryptoCommission",
    "BINANCE_SPOT_COMMISSION",
    "BINANCE_FUTURES_COMMISSION",
    "COINBASE_COMMISSION",
    "KRAKEN_COMMISSION",
    # Position tracking
    "Position",
    "PositionState",
    "PositionTracker",
    # Smart order routing
    "ExchangeQuote",
    "RoutedOrder",
    "SmartOrderRouter",
    # Freqtrade Exchange Adapter
    "FreqtradeExchangeAdapter",
    "FreqtradeMultiExchangeAdapter",
    "HAS_FREQTRADE",
    # Freqtrade Strategy Interface
    "FreqtradeSignal",
    "BaseStrategy",
    "FreqtradeStrategyWrapper",
    "IStrategyAdapter",
    "StrategyRunner",
    "create_strategy_from_template",
    # Freqtrade Strategy Wrapper (pure Python, no freqtrade dependency)
    "FreqtradeStrategy",
    "EdgePositionSizer",
    "EMASpreadStrategy",
    "MACDStrategy",
    "PriceBandStrategy",
    "FreqtradeExecutor",
    "SignalDirection",
    "EntryExit",
    "ROI",
    "StrategyConfig",
    "TradeSignal",
    "Position",
    # Hummingbot Exchange Adapter
    "HummingbotExchangeAdapter",
    "HummingbotMultiExchangeAdapter",
    "HummingbotConfig",
    "MarketDataCollector",
    "ArbitrageDetector",
    "ExecutionRouter",
    "ArbitrageOpportunity",
    "BestExchange",
    "OrderType",
    "TradeType",
    "PositionSide",
    "SUPPORTED_EXCHANGES",
    "EXCHANGE_NAME_MAPPING",
    "EXCHANGE_DISPLAY_NAMES",
    # PyCryptoBot
    "PyCryptoBotConfig",
    "AppState",
    "MarketData",
    "Granularity",
    "Action",
    "BaseStrategy",
    "EMAStrategy",
    "THUMBStrategy",
    "SMARoCStrategy",
    "QuadencyConnector",
    "PyCryptoBotExecutor",
    "calculate_ema",
    "calculate_sma",
    "calculate_macd",
    "calculate_rsi",
    "calculate_bollinger_bands",
    "calculate_obv",
    "calculate_atr",
    # Lumibot Adapter (pure Python urllib, no lumibot SDK dependency)
    # 核心实体
    "Asset",
    "Order",
    "Position",
    "Bar",
    "Bars",
    # 枚举
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    # 券商类
    "LumibotBroker",
    "PolygonBroker",
    "AlpacaBroker",
    # 数据源类
    "PolygonDataSource",
    "AlpacaDataSource",
    # 策略
    "LumibotStrategy",
    # 回测
    "LumibotBacktest",
    # 异常
    "LumibotBrokerAPIError",
    "LumibotDataError",
    # OctoBot Core Adapter (pure Python urllib, Telegram integration)
    # OctoBot 核心适配器（纯 Python urllib 实现，Telegram 集成）
    # 配置
    "OctoBotConfig",
    # 数据类
    "TradeOrder",
    "ExecutionReport",
    # 枚举
    "TradeSignalType",
    "AlertLevel",
    "ExchangeSyncMode",
    # 主类
    "TelegramNotifier",
    "TradingAligner",
    "OctoBotExecutor",
    # Dynamic Spread Market Making (from hummingbot pmm_dynamic)
    "DynamicSpreadPricing",
    "DynamicSpreadConfig",
]
