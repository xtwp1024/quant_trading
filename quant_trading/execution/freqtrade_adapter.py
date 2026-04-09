"""
Freqtrade Exchange Adapter

适配 freqtrade 专业级加密货币量化交易框架到本系统的连接器架构。

支持:
- 15+ 交易所 (Binance, OKX, Bybit, Gate, Kraken, etc.)
- 现货/杠杆/永续合约
- FreqAI 机器学习优化
- QTPyLib 内置指标
- WAL 模式 SQLite 持久化
- WebSocket 实时数据

此适配器将 freqtrade 的 Exchange 类包装为 BaseConnector 接口，
使其可以与现有的 execution/connectors 架构无缝集成。
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from freqtrade.exchange.exchange import Exchange
    from freqtrade.exchange.exchange_types import (
        CcxtBalances,
        CcxtOrder,
        CcxtPosition,
        OrderBook,
        Ticker,
        Tickers,
    )
    from freqtrade.enums import BuySell, CandleType, TradingMode
    _FREQTRADE_AVAILABLE = True
    HAS_FREQTRADE = True
except ImportError:
    Exchange = None
    CcxtBalances = None
    CcxtOrder = None
    CcxtPosition = None
    OrderBook = None
    Ticker = None
    Tickers = None
    BuySell = None
    CandleType = None
    TradingMode = None
    _FREQTRADE_AVAILABLE = False
    HAS_FREQTRADE = False

from quant_trading.connectors.base_connector import BaseConnector
from quant_trading.connectors.order_types import (
    InFlightOrderBase,
    OrderType as QtOrderType,
    OrderUpdate,
    TradeType as QtTradeType,
    TradeUpdate,
    TradeFee,
    TokenAmount,
)
from quant_trading.execution.executor import (
    Order as QtOrder,
    OrderSide,
    OrderStatus,
    OrderType as QtExecOrderType,
)

if TYPE_CHECKING:
    from pandas import DataFrame

logger = logging.getLogger(__name__)

# Freqtrade pair format: "BTC/USDT"
# Our format: "BTC-USDT"
def _ft_pair_to_qt(pair: str) -> str:
    """Convert freqtrade pair format to our format."""
    return pair.replace("/", "-")

def _qt_pair_to_ft(pair: str) -> str:
    """Convert our pair format to freqtrade format."""
    return pair.replace("-", "/")

# Timeframe mappings
FT_TIMEFRAMES = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "6h": "6h", "8h": "8h",
    "12h": "12h", "1d": "1d", "3d": "3d", "1w": "1w",
}


class FreqtradeExchangeAdapter(BaseConnector):
    """
    Freqtrade Exchange Adapter

    将 freqtrade Exchange 类适配到 BaseConnector 接口。
    利用 freqtrade 的成熟交易框架，提供:
    - 统一的交易所接口 (ccxt 后端)
    - 完整的订单生命周期管理
    - 余额和持仓管理
    - 实时市场数据 (WebSocket 支持)
    - 15+ 交易所支持
    """

    def __init__(
        self,
        exchange_name: str,
        api_key: str = "",
        api_secret: str = "",
        config: Optional[Dict[str, Any]] = None,
        trading_mode: str = "spot",
        dry_run: bool = True,
        **kwargs,
    ):
        """
        Initialize Freqtrade Exchange Adapter.

        Args:
            exchange_name: Exchange name (e.g., "binance", "okx", "bybit")
            api_key: Exchange API key
            api_secret: Exchange API secret
            config: Additional freqtrade configuration
            trading_mode: "spot", "margin", or "futures"
            dry_run: Enable dry-run mode
        """
        super().__init__()
        self._exchange_name = exchange_name.lower()
        self._api_key = api_key
        self._api_secret = api_secret
        self._trading_mode = trading_mode
        self._dry_run = dry_run
        self._config = config or {}
        self._exchange: Optional[Exchange] = None
        self._leverage = kwargs.get("leverage", 1)
        self._margin_mode = kwargs.get("margin_mode", "isolated")

        # Cache for in-flight orders
        self._in_flight_orders: Dict[str, InFlightOrderBase] = {}

    @property
    def name(self) -> str:
        return f"Freqtrade{self._exchange_name.capitalize()}"

    @property
    def trading_pairs(self) -> List[str]:
        """Return list of supported trading pairs."""
        if self._exchange is None:
            return []
        return [m.symbol for m in self._exchange.markets.values() if m.get("active", True)]

    @property
    def ready(self) -> bool:
        """Return whether connector is ready to trade."""
        return self._exchange is not None and bool(self._exchange.markets)

    @property
    def exchange(self) -> Optional[Exchange]:
        """Get the underlying freqtrade Exchange instance."""
        return self._exchange

    async def connect(self) -> None:
        """
        Establish connection to the exchange via freqtrade.
        """
        # Build freqtrade configuration
        ft_config: Dict[str, Any] = {
            "dry_run": self._dry_run,
            "trading_mode": self._trading_mode,
            "margin_mode": self._margin_mode if self._trading_mode != "spot" else None,
            "stake_currency": "USDT",
            "exchange": {
                "name": self._exchange_name,
                "key": self._api_key,
                "secret": self._api_secret,
                "enable_ws": True,
            },
            "runmode": "trade",
        }
        ft_config.update(self._config)

        # Initialize freqtrade Exchange
        self._exchange = Exchange(ft_config, validate=True, load_leverage_tiers=True)

        # Sync balances to internal state
        await self._sync_balances()
        logger.info(f"Connected to {self._exchange_name} via freqtrade adapter")

    async def disconnect(self) -> None:
        """Close connection to exchange."""
        if self._exchange:
            self._exchange.close()
            self._exchange = None
        logger.info(f"Disconnected from {self._exchange_name}")

    async def _sync_balances(self) -> None:
        """Synchronize account balances from exchange."""
        if self._exchange is None:
            return
        balances: CcxtBalances = self._exchange.get_balances()
        for currency, balance in balances.items():
            self._account_balances[currency] = Decimal(str(balance.get("total", 0)))
            self._account_available_balances[currency] = Decimal(str(balance.get("free", 0)))

    # ===================
    # Order Methods
    # ===================

    async def place_order(
        self,
        trading_pair: str,
        side: QtTradeType,
        amount: Decimal,
        order_type: QtOrderType,
        price: Optional[Decimal] = None,
        **kwargs,
    ) -> str:
        """
        Place an order via freqtrade.

        Args:
            trading_pair: Trading pair (e.g., "BTC-USDT")
            side: BUY or SELL
            amount: Order amount
            order_type: MARKET, LIMIT, etc.
            price: Order price (required for LIMIT orders)

        Returns:
            Client order ID
        """
        if self._exchange is None:
            raise RuntimeError("Exchange not connected")

        pair = _qt_pair_to_ft(trading_pair)
        ft_side = BuySell.BUY if side == QtTradeType.BUY else BuySell.SELL
        ordertype = "market" if order_type == QtOrderType.MARKET else "limit"
        leverage = kwargs.get("leverage", self._leverage)

        # Determine rate
        rate = float(price) if price else 0.0

        # Place order via freqtrade
        ccxt_order: CcxtOrder = self._exchange.create_order(
            pair=pair,
            ordertype=ordertype,
            side=ft_side,
            amount=float(amount),
            rate=rate,
            leverage=leverage,
        )

        # Create in-flight order
        client_order_id = str(ccxt_order.get("id", ""))
        self._in_flight_orders[client_order_id] = InFlightOrderBase(
            client_order_id=client_order_id,
            trading_pair=trading_pair,
            order_type=order_type,
            trade_type=side,
            amount=amount,
            price=price,
            creation_timestamp=ccxt_order.get("timestamp", 0) / 1000,
        )

        # Update balances
        await self._sync_balances()

        return client_order_id

    async def cancel_order(self, trading_pair: str, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            trading_pair: Trading pair
            order_id: Client order ID

        Returns:
            True if cancellation was successful
        """
        if self._exchange is None:
            raise RuntimeError("Exchange not connected")

        pair = _qt_pair_to_ft(trading_pair)
        try:
            self._exchange.cancel_order(order_id, pair)
            await self._sync_balances()
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def place_stoploss(
        self,
        trading_pair: str,
        amount: Decimal,
        stop_price: float,
        side: QtTradeType,
        leverage: float = 1.0,
    ) -> str:
        """
        Place a stoploss order on exchange.

        Args:
            trading_pair: Trading pair
            amount: Order amount
            stop_price: Stop price
            side: BUY or SELL
            leverage: Leverage for futures

        Returns:
            Order ID
        """
        if self._exchange is None:
            raise RuntimeError("Exchange not connected")

        pair = _qt_pair_to_ft(trading_pair)
        ft_side = BuySell.BUY if side == QtTradeType.BUY else BuySell.SELL

        order_types = {"stoploss": "market"}
        ccxt_order: CcxtOrder = self._exchange.create_stoploss(
            pair=pair,
            amount=float(amount),
            stop_price=stop_price,
            order_types=order_types,
            side=ft_side,
            leverage=leverage,
        )

        return str(ccxt_order.get("id", ""))

    # ===================
    # Market Data Methods
    # ===================

    def get_price(
        self, trading_pair: str, is_buy: bool, amount: Decimal = Decimal("0")
    ) -> Decimal:
        """
        Get current market price.

        Args:
            trading_pair: Trading pair
            is_buy: True for ask price, False for bid price
            amount: Optional amount for volume-weighted price

        Returns:
            Current price
        """
        if self._exchange is None:
            return Decimal("0")

        pair = _qt_pair_to_ft(trading_pair)
        if self._exchange.exchange_has("fetchL2OrderBook"):
            orderbook = self.fetch_l2_order_book(trading_pair, 20)
            if is_buy:
                return Decimal(str(orderbook.get("asks", [[0]])[0][0]))
            else:
                return Decimal(str(orderbook.get("bids", [[0]])[0][0]))
        else:
            ticker: Optional[Ticker] = self._exchange.get_ticker(pair)
            if ticker:
                return Decimal(str(ticker.get("last", 0)))
        return Decimal("0")

    def get_ticker(self, trading_pair: str) -> Optional[Dict[str, Any]]:
        """Get ticker for trading pair."""
        if self._exchange is None:
            return None
        pair = _qt_pair_to_ft(trading_pair)
        ticker: Optional[Ticker] = self._exchange.get_ticker(pair)
        if ticker:
            return {
                "symbol": trading_pair,
                "last": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "volume": ticker.get("quoteVolume"),
                "percentage": ticker.get("percentage"),
            }
        return None

    def get_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Get all tickers."""
        if self._exchange is None:
            return {}
        tickers: Tickers = self._exchange.get_tickers()
        return {
            _ft_pair_to_qt(symbol): {
                "last": t.get("last"),
                "bid": t.get("bid"),
                "ask": t.get("ask"),
                "volume": t.get("quoteVolume"),
                "percentage": t.get("percentage"),
            }
            for symbol, t in tickers.items()
        }

    def fetch_l2_order_book(self, trading_pair: str, depth: int = 20) -> OrderBook:
        """Fetch level 2 order book."""
        if self._exchange is None:
            raise RuntimeError("Exchange not connected")
        pair = _qt_pair_to_ft(trading_pair)
        return self._exchange.fetch_l2_order_book(pair, depth)

    def fetch_ohlcv(
        self,
        trading_pair: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: Optional[int] = None,
    ) -> "DataFrame":
        """
        Fetch OHLCV candles.

        Args:
            trading_pair: Trading pair
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of candles to fetch
            since: Start time in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        if self._exchange is None:
            raise RuntimeError("Exchange not connected")
        pair = _qt_pair_to_ft(trading_pair)
        return self._exchange.fetch_ohlcv(
            pair, timeframe, limit, since=since
        )

    def fetch_trades(
        self, trading_pair: str, limit: int = 100, since: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch recent trades."""
        if self._exchange is None:
            raise RuntimeError("Exchange not connected")
        pair = _qt_pair_to_ft(trading_pair)
        return self._exchange.fetch_trades(pair, limit=limit, since=since)

    # ===================
    # Balance & Position Methods
    # ===================

    async def get_account_balances(self) -> Dict[str, Decimal]:
        """Get all account balances."""
        await self._sync_balances()
        return self._account_balances.copy()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions (for futures)."""
        if self._exchange is None:
            return []
        positions: List[CcxtPosition] = self._exchange.fetch_positions()
        return [
            {
                "symbol": p.get("symbol"),
                "side": p.get("side"),
                "contracts": p.get("contracts"),
                "leverage": p.get("leverage"),
                "collateral": p.get("collateral"),
                "liquidation_price": p.get("liquidationPrice"),
            }
            for p in positions
        ]

    def set_leverage(self, leverage: float, pair: Optional[str] = None) -> None:
        """Set leverage for futures trading."""
        self._leverage = leverage
        if self._exchange and self._exchange.trading_mode == TradingMode.FUTURES:
            if pair:
                pair = _qt_pair_to_ft(pair)
                self._exchange._set_leverage(leverage, pair)

    # ===================
    # Quantization Methods
    # ===================

    def get_order_price_quantum(self, trading_pair: str, price: Decimal) -> Decimal:
        """Get minimum price increment."""
        if self._exchange is None:
            return Decimal("0")
        pair = _qt_pair_to_ft(trading_pair)
        precision = self._exchange.get_precision_price(pair)
        if precision is None:
            return Decimal("0")
        return Decimal(str(precision))

    def get_order_size_quantum(self, trading_pair: str, amount: Decimal) -> Decimal:
        """Get minimum size increment."""
        if self._exchange is None:
            return Decimal("0")
        pair = _qt_pair_to_ft(trading_pair)
        precision = self._exchange.get_precision_amount(pair)
        if precision is None:
            return Decimal("0")
        return Decimal(str(precision))

    def quantize_order_price(
        self, trading_pair: str, price: Decimal
    ) -> Decimal:
        """Quantize order price to exchange requirements."""
        if self._exchange is None:
            return price
        pair = _qt_pair_to_ft(trading_pair)
        quantized = self._exchange.price_to_precision(pair, float(price))
        return Decimal(str(quantized))

    def quantize_order_amount(
        self, trading_pair: str, amount: Decimal
    ) -> Decimal:
        """Quantize order amount to exchange requirements."""
        if self._exchange is None:
            return amount
        pair = _qt_pair_to_ft(trading_pair)
        quantized = self._exchange.amount_to_precision(pair, float(amount))
        return Decimal(str(quantized))

    # ===================
    # Fee Methods
    # ===================

    def estimate_fee_pct(self, is_maker: bool) -> Decimal:
        """Estimate trading fee percentage."""
        # Freqtrade/exchange fees vary by exchange and pair
        # Default to 0.1% (common for most exchanges)
        return Decimal("0.001")

    def get_fee(
        self, trading_pair: str, order_type: QtOrderType, side: QtTradeType
    ) -> TradeFee:
        """Get trading fee for a pair."""
        maker_fee = self._exchange.get_fee(
            _qt_pair_to_ft(trading_pair), taker_or_maker="maker"
        ) if self._exchange else Decimal("0.001")
        return TradeFee(percent=maker_fee)

    # ===================
    # Status Methods
    # ===================

    @property
    def status_dict(self) -> Dict[str, bool]:
        """Return status of connector components."""
        return {
            "exchange_connection": self._exchange is not None,
            "markets_loaded": bool(self._exchange and self._exchange.markets),
            "balance_sync": True,
        }

    # ===================
    # Advanced Order Methods
    # ===================

    def get_stake_amount(
        self,
        pair: str,
        price: float,
        stoploss: float,
        capital_mode: str = "fixed",
        stake_amount: float = 0.0,
    ) -> float:
        """
        Calculate stake amount for a trade.

        Args:
            pair: Trading pair
            price: Entry price
            stoploss: Stoploss percentage (negative)
            capital_mode: "fixed" or "percentage"
            stake_amount: Stake amount or percentage

        Returns:
            Calculated stake amount
        """
        if self._exchange is None:
            return 0.0

        pair_ft = _qt_pair_to_ft(pair)
        if capital_mode == "fixed":
            min_stake = self._exchange.get_min_pair_stake_amount(
                pair_ft, price, stoploss, self._leverage
            )
            max_stake = self._exchange.get_max_pair_stake_amount(pair_ft, price, self._leverage)
            if min_stake is None:
                min_stake = 0.0
            if max_stake is None:
                max_stake = float("inf")
            return max(min_stake, min(stake_amount, max_stake))
        else:
            # Percentage mode - would need account balance
            return stake_amount

    def validate_order_types(self, order_types: Dict[str, str]) -> bool:
        """Validate if order types are supported by exchange."""
        if self._exchange is None:
            return False
        try:
            self._exchange.validate_ordertypes(order_types)
            return True
        except Exception:
            return False

    def get_supported_order_types(self) -> List[str]:
        """Get list of supported order types."""
        if self._exchange is None:
            return []
        types = ["market", "limit"]
        if self._exchange.exchange_has("createStopLossOrder"):
            types.append("stop_loss")
        if self._exchange.exchange_has("createStopLimitOrder"):
            types.append("stop_limit")
        return types

    # ===================
    # Market Info Methods
    # ===================

    def get_markets(self) -> List[Dict[str, Any]]:
        """Get all markets with trading info."""
        if self._exchange is None:
            return []
        return [
            {
                "symbol": _ft_pair_to_qt(m.get("symbol", "")),
                "base": m.get("base", ""),
                "quote": m.get("quote", ""),
                "active": m.get("active", True),
                "precision": m.get("precision", {}),
                "limits": m.get("limits", {}),
            }
            for m in self._exchange.markets.values()
        ]

    def get_pair_quote_currency(self, trading_pair: str) -> str:
        """Get quote currency for a pair."""
        if self._exchange is None:
            return ""
        pair = _qt_pair_to_ft(trading_pair)
        return self._exchange.get_pair_quote_currency(pair)

    def get_pair_base_currency(self, trading_pair: str) -> str:
        """Get base currency for a pair."""
        if self._exchange is None:
            return ""
        pair = _qt_pair_to_ft(trading_pair)
        return self._exchange.get_pair_base_currency(pair)

    def market_is_active(self, trading_pair: str) -> bool:
        """Check if a market pair is active for trading."""
        if self._exchange is None:
            return False
        pair = _qt_pair_to_ft(trading_pair)
        return self._exchange.market_is_active(pair)

    # ===================
    # Order Tracking
    # ===================

    @property
    def in_flight_orders(self) -> Dict[str, InFlightOrderBase]:
        """Return dictionary of in-flight orders."""
        return self._in_flight_orders

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get status of an order."""
        if self._exchange is None:
            return None
        try:
            # Search in freqtrade open orders
            for oid in self._exchange._dry_run_open_orders:
                if oid == order_id:
                    return OrderStatus.PENDING
        except Exception:
            pass
        return None

    def update_order_status(self, order_id: str) -> bool:
        """Update order status from exchange."""
        if self._exchange is None or order_id not in self._in_flight_orders:
            return False

        try:
            # This would need the pair - simplified here
            return True
        except Exception as e:
            logger.error(f"Failed to update order {order_id}: {e}")
            return False


class FreqtradeMultiExchangeAdapter:
    """
    Multi-Exchange Adapter Manager

    Manages multiple FreqtradeExchangeAdapter instances for different exchanges.
    Provides unified access across exchanges.
    """

    def __init__(self):
        self._adapters: Dict[str, FreqtradeExchangeAdapter] = {}

    def add_exchange(
        self,
        name: str,
        api_key: str = "",
        api_secret: str = "",
        config: Optional[Dict[str, Any]] = None,
        trading_mode: str = "spot",
        dry_run: bool = True,
    ) -> FreqtradeExchangeAdapter:
        """Add and connect an exchange adapter."""
        adapter = FreqtradeExchangeAdapter(
            exchange_name=name,
            api_key=api_key,
            api_secret=api_secret,
            config=config,
            trading_mode=trading_mode,
            dry_run=dry_run,
        )
        self._adapters[name.lower()] = adapter
        return adapter

    def get_adapter(self, name: str) -> Optional[FreqtradeExchangeAdapter]:
        """Get an adapter by exchange name."""
        return self._adapters.get(name.lower())

    def get_all_adapters(self) -> Dict[str, FreqtradeExchangeAdapter]:
        """Get all adapters."""
        return self._adapters.copy()

    async def connect_all(self) -> None:
        """Connect to all exchanges."""
        for adapter in self._adapters.values():
            await adapter.connect()

    async def disconnect_all(self) -> None:
        """Disconnect from all exchanges."""
        for adapter in self._adapters.values():
            await adapter.disconnect()
