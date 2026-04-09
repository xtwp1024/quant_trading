"""
Quant Trading Connectors Package

This package provides exchange connector abstractions inspired by the Hummingbot framework.
It offers a standardized interface for connecting to cryptocurrency exchanges.

Key Components:
- base_connector: Abstract base class for exchange connectors
- order_types: Order, trade, and position type enumerations
- binance_rest: Binance REST API connector
- binance_ws: Binance WebSocket connector
- binance_orderbook: Binance order book cache manager

Architecture:
    The connector pattern separates exchange-specific implementation details
    from the core trading logic, enabling:
    - Multiple exchange support with unified interface
    - Order lifecycle management via InFlightOrder
    - Event-driven architecture for real-time updates
    - Consistent fee calculation across exchanges

Example:
    from quant_trading.connectors import BinanceRESTConnector, BinanceWSConnector, OrderType, TradeType
"""

from quant_trading.connectors.order_types import (
    CloseType,
    InFlightOrderBase,
    OrderState,
    OrderType,
    PositionAction,
    PriceType,
    TradeFee,
    TradeType,
    TokenAmount,
)

# Import Binance connectors
from quant_trading.connectors.binance_rest import (
    BinanceRESTConnector,
    BinanceRateLimiter,
    BinanceAPIError,
    BinanceRateLimitError,
    BinanceConnectionError,
)

from quant_trading.connectors.binance_ws import (
    BinanceWSConnector,
    BinanceWSManager,
    WSSubscription,
    BinanceWSMessage,
)

from quant_trading.connectors.binance_orderbook import (
    BinanceOrderBookManager,
    OrderBookState,
    OrderBookSnapshot,
    PriceLevel,
)

from quant_trading.connectors.cex_adapter import (
    CEXAdapter,
    SUPPORTED_EXCHANGES,
)

from quant_trading.connectors.unicorn_binance import (
    UnicornBinanceREST,
    UnicornBinanceWebSocket,
    UnicornBinanceAPIError,
    UnicornBinanceConnectionError,
    UnicornBinanceRateLimitError,
)

from quant_trading.connectors.ccxt_adapter import (
    CCXTAdapter,
    ExchangeNormalizer,
    MarketDataFetcher,
    OrderExecutor,
    MarginCalculator,
    SUPPORTED_EXCHANGES,
    EXCHANGE_CONFIGS,
    Market,
    OrderBook,
    OHLCV,
    Ticker,
    Order,
    Trade,
    Position,
)

__all__ = [
    # Order types
    "OrderType",
    "TradeType",
    "OrderState",
    "PositionAction",
    "PriceType",
    "CloseType",
    "TokenAmount",
    "TradeFee",
    "InFlightOrderBase",
    # Binance REST
    "BinanceRESTConnector",
    "BinanceRateLimiter",
    "BinanceAPIError",
    "BinanceRateLimitError",
    "BinanceConnectionError",
    # Binance WebSocket
    "BinanceWSConnector",
    "BinanceWSManager",
    "WSSubscription",
    "BinanceWSMessage",
    # Order Book
    "BinanceOrderBookManager",
    "OrderBookState",
    "OrderBookSnapshot",
    "PriceLevel",
    # CEX Multi-Exchange Adapter
    "CEXAdapter",
    "SUPPORTED_EXCHANGES",
    # Unicorn Binance Suite
    "UnicornBinanceREST",
    "UnicornBinanceWebSocket",
    "UnicornBinanceAPIError",
    "UnicornBinanceConnectionError",
    "UnicornBinanceRateLimitError",
    # CCXT Unified Adapter (100+ exchanges)
    "CCXTAdapter",
    "ExchangeNormalizer",
    "MarketDataFetcher",
    "OrderExecutor",
    "MarginCalculator",
    "EXCHANGE_CONFIGS",
    "Market",
    "OrderBook",
    "OHLCV",
    "Ticker",
    "Order",
    "Trade",
    "Position",
]
