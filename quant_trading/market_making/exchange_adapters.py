"""
Exchange Adapter Patterns for PassivBot

Adapted from passivbot's src/exchanges/ directory.

This module captures the exchange integration patterns from passivbot's
CCXT-based architecture, without depending on passivbot itself. It provides
abstract base classes and concrete exchange adapter examples that show
how to integrate the PassivBotStrategy with various cryptocurrency exchanges.

Supported exchanges: Binance, OKX, Bybit, Bitget, GateIO, Hyperliquid, KuCoin

Key patterns:
- Hook-based customization: can_*, _do_*, _get_*, _normalize_*, _build_*
- Hedge mode vs one-way mode handling
- Position side determination (long/short)
- Balance fetching and normalization
- Order execution and cancellation
- Ticker and OHLCV data fetching

Usage:
    adapter = BinanceAdapter(api_key, api_secret)
    balance = await adapter.fetch_balance()
    positions = await adapter.fetch_positions()
    orders = await adapter.execute_orders([order_dict])
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ExchangeAdapter",
    "BinanceAdapter",
    "OKXAdapter",
    "BybitAdapter",
    "BitgetAdapter",
    "GateIOAdapter",
    "HyperliquidAdapter",
    "KuCoinAdapter",
    "OrderSpec",
    "Position",
    "Ticker",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """Normalized position data."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    margin: float = 0.0

    @property
    def is_open(self) -> bool:
        return abs(self.size) > 1e-10


@dataclass
class OrderSpec:
    """Specification for creating an order."""
    symbol: str
    side: str  # 'buy' or 'sell'
    position_side: str  # 'long' or 'short'
    qty: float
    price: float
    order_type: str = "limit"  # 'limit', 'market'
    custom_id: str = ""
    post_only: bool = False


@dataclass
class Ticker:
    """Normalized ticker data."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float = 0.0
    timestamp_ms: int = 0


@dataclass
class Balance:
    """Normalized balance data."""
    total: float
    free: float
    used: float
    quote: str = "USDT"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ExchangeAdapter(ABC):
    """
    Abstract base class for exchange adapters.

    Exchange adapters bridge between the exchange's API and the
    PassivBotStrategy. They handle:
    - API session management (REST + optional WebSocket)
    - Data normalization (positions, balances, orders)
    - Order execution and cancellation
    - Exchange-specific quirks (position side, margin mode, etc.)

    Subclass this for each exchange and override the hook methods.
    """

    exchange_name: str = "unknown"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        password: str = "",
        passphrase: str = "",
        testnet: bool = False,
        recv_window_ms: int = 5000,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.password = password
        self.passphrase = passphrase
        self.testnet = testnet
        self.recv_window_ms = recv_window_ms

        # Normalized state
        self.balance: Balance = Balance(total=0.0, free=0.0, used=0.0)
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, List[Dict]] = {}
        self.markets: Dict[str, Dict] = {}

        # Internal
        self._session = None
        self._ws_session = None
        self._leverage: Dict[str, int] = {}
        self._margin_mode: Dict[str, str] = {}
        self._hedge_mode: bool = True
        self._last_balance_fetch_ms: int = 0

    # -------------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Initialize API sessions and load markets."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close API sessions."""
        pass

    async def fetch_balance(self) -> Balance:
        """Fetch and normalize account balance."""
        raw = await self._do_fetch_balance()
        return self._normalize_balance(raw)

    async def fetch_positions(self) -> List[Position]:
        """Fetch and normalize all positions."""
        raw = await self._do_fetch_positions()
        return self._normalize_positions(raw)

    async def fetch_tickers(self) -> Dict[str, Ticker]:
        """Fetch and normalize tickers for all symbols."""
        raw = await self._do_fetch_tickers()
        return self._normalize_tickers(raw)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 1000
    ) -> np.ndarray:
        """
        Fetch OHLCV candles.

        Returns:
            Array of shape (n, 5) with [open, high, low, close, volume]
        """
        raw = await self._do_fetch_ohlcv(symbol, timeframe, limit)
        return self._normalize_ohlcv(raw)

    # -------------------------------------------------------------------------
    # Order execution
    # -------------------------------------------------------------------------

    async def execute_orders(self, orders: List[OrderSpec]) -> List[Dict]:
        """Execute multiple orders in parallel."""
        if not orders:
            return []
        tasks = [self.execute_order(o) for o in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    @abstractmethod
    async def execute_order(self, order: OrderSpec) -> Dict:
        """Execute a single order."""
        pass

    async def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an open order."""
        return await self._do_cancel_order(symbol, order_id)

    # -------------------------------------------------------------------------
    # Exchange configuration
    # -------------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol."""
        await self._do_set_leverage(symbol, leverage)
        self._leverage[symbol] = leverage

    async def set_margin_mode(self, symbol: str, mode: str) -> None:
        """
        Set margin mode for a symbol.

        Args:
            mode: 'cross' or 'isolated'
        """
        await self._do_set_margin_mode(symbol, mode)
        self._margin_mode[symbol] = mode

    async def set_hedge_mode(self, enabled: bool) -> None:
        """Enable or disable hedge mode (two-way positions)."""
        await self._do_set_hedge_mode(enabled)
        self._hedge_mode = enabled

    # -------------------------------------------------------------------------
    # Abstract hooks (override per exchange)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _do_fetch_balance(self) -> Dict:
        """Fetch raw balance from exchange API."""
        pass

    @abstractmethod
    async def _do_fetch_positions(self) -> List[Dict]:
        """Fetch raw positions from exchange API."""
        pass

    @abstractmethod
    async def _do_fetch_tickers(self) -> Dict:
        """Fetch raw tickers from exchange API."""
        pass

    @abstractmethod
    async def _do_fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[List[float]]:
        """Fetch raw OHLCV data from exchange API."""
        pass

    @abstractmethod
    async def _do_execute_order(self, order: OrderSpec) -> Dict:
        """Execute order via exchange API."""
        pass

    @abstractmethod
    async def _do_cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel order via exchange API."""
        pass

    @abstractmethod
    async def _do_set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage via exchange API."""
        pass

    @abstractmethod
    async def _do_set_margin_mode(self, symbol: str, mode: str) -> None:
        """Set margin mode via exchange API."""
        pass

    @abstractmethod
    async def _do_set_hedge_mode(self, enabled: bool) -> None:
        """Set hedge mode via exchange API."""
        pass

    # -------------------------------------------------------------------------
    # Normalization hooks (override per exchange)
    # -------------------------------------------------------------------------

    def _normalize_balance(self, raw: Dict) -> Balance:
        """Normalize balance from exchange format to Balance dataclass."""
        # Default: assume exchange returns total/free/used in quote currency
        return Balance(
            total=float(raw.get("total", 0)),
            free=float(raw.get("free", raw.get("available", 0))),
            used=float(raw.get("used", 0)),
            quote=self._default_quote(),
        )

    def _normalize_positions(self, raw: List[Dict]) -> List[Position]:
        """Normalize positions from exchange format."""
        positions = {}
        for elm in raw:
            contracts = float(elm.get("contracts", elm.get("size", 0)))
            if contracts == 0:
                continue
            symbol = elm["symbol"]
            side = self._get_position_side(elm)
            key = f"{symbol}:{side}"
            pos = Position(
                symbol=symbol,
                side=side,
                size=contracts,
                entry_price=float(elm.get("entryPrice", elm.get("price", 0))),
                unrealized_pnl=float(elm.get("unrealizedPnl", 0)),
                margin=float(elm.get("margin", 0)),
            )
            positions[key] = pos
        return list(positions.values())

    def _normalize_tickers(self, raw: Dict) -> Dict[str, Ticker]:
        """Normalize tickers from exchange format."""
        tickers = {}
        for symbol, data in raw.items():
            tickers[symbol] = Ticker(
                symbol=symbol,
                bid=float(data.get("bid", 0)),
                ask=float(data.get("ask", 0)),
                last=float(data.get("last", data.get("lastPrice", 0))),
                volume=float(data.get("volume", data.get("quoteVolume", 0))),
            )
        return tickers

    def _normalize_ohlcv(self, raw: List[List[float]]) -> np.ndarray:
        """
        Normalize OHLCV data.

        Input: list of [timestamp, open, high, low, close, volume]
        Output: ndarray of shape (n, 5) with [open, high, low, close, volume]
        """
        if not raw:
            return np.zeros((0, 5))
        result = np.zeros((len(raw), 5))
        for i, candle in enumerate(raw):
            # Handle both 5-element and 6-element formats
            result[i, 0] = float(candle[1])  # open
            result[i, 1] = float(candle[2])  # high
            result[i, 2] = float(candle[3])  # low
            result[i, 3] = float(candle[4])  # close
            result[i, 4] = float(candle[5]) if len(candle) > 5 else 0.0  # volume
        return result

    def _get_position_side(self, elm: Dict) -> str:
        """
        Determine position side from exchange position data.

        Default implementation uses common CCXT patterns.
        Override for exchange-specific logic.
        """
        # Check for positionSide field
        if "positionSide" in elm:
            return str(elm["positionSide"]).lower()
        if "position_side" in elm:
            return str(elm["position_side"]).lower()
        # Check side field (buy = long, sell = short)
        side = elm.get("side", "").lower()
        if side == "buy":
            return "long"
        if side == "sell":
            return "short"
        # Check if net position is positive or negative
        size = float(elm.get("size", elm.get("contracts", 0)))
        return "long" if size > 0 else "short"

    def _default_quote(self) -> str:
        return "USDT"

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def _symbol_to_exchange(self, symbol: str) -> str:
        """Convert normalized symbol to exchange-specific format."""
        # Default: assume symbol is already in exchange format
        return symbol

    def _symbol_from_exchange(self, exchange_symbol: str) -> str:
        """Convert exchange-specific symbol to normalized format."""
        return exchange_symbol

    def _parse_price(self, price: float, symbol: str) -> float:
        """Round price to exchange's price step for the symbol."""
        market = self.markets.get(symbol, {})
        price_step = market.get("precision", {}).get("price", 1e-8)
        return round(price / price_step) * price_step

    def _parse_qty(self, qty: float, symbol: str) -> float:
        """Round quantity to exchange's lot size for the symbol."""
        market = self.markets.get(symbol, {})
        qty_step = market.get("precision", {}).get("amount", 1e-8)
        return round(qty / qty_step) * qty_step

    async def _retry_async(
        self, fn, *args, max_retries: int = 3, delay: float = 1.0, **kwargs
    ):
        """Simple retry wrapper for async API calls."""
        last_exc = None
        for attempt in range(max_retries):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (attempt + 1))
                    logger.warning(
                        f"{self.exchange_name}: retrying after error: {exc}"
                    )
        raise last_exc


# ---------------------------------------------------------------------------
# Binance adapter
# ---------------------------------------------------------------------------


class BinanceAdapter(ExchangeAdapter):
    """
    Binance futures adapter.

    Key characteristics:
    - Uses positionSide in order params (Binance hedge mode)
    - Symbols: BTCUSDT, ETHUSDT, etc.
    - Balance: totalCrossWalletBalance field
    - Position fetch: fapiprivatev3_get_positionrisk
    """

    exchange_name = "binance"

    def __init__(self, api_key: str = "", api_secret: str = "", **kwargs):
        super().__init__(api_key, api_secret, **kwargs)
        self.broker_code = kwargs.get("broker_code", "")

    async def connect(self) -> None:
        # In a real implementation, this would create aiohttp sessions
        # and load markets via self._session.fetch_markets()
        pass

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
        if self._ws_session:
            await self._ws_session.close()

    async def _do_fetch_balance(self) -> Dict:
        # Would use: self._session.fapiprivate_get_account()
        # Returns: {"info": {"totalCrossWalletBalance": "10000.00", ...}}
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def _do_fetch_positions(self) -> List[Dict]:
        # Would use: self._session.fapiprivatev3_get_positionrisk()
        return []

    async def _do_fetch_tickers(self) -> Dict:
        # Would use: self._session.fapipublic_get_ticker_bookticker()
        return {}

    async def _do_fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[List[float]]:
        # Would use: self._session.fetch_ohlcv(symbol, timeframe, limit)
        return []

    async def _do_execute_order(self, order: OrderSpec) -> Dict:
        # Would use: self._session.create_order(...)
        # Binance requires: positionSide, newClientOrderId
        return {}

    async def _do_cancel_order(self, symbol: str, order_id: str) -> Dict:
        return {}

    async def _do_set_leverage(self, symbol: str, leverage: int) -> None:
        # self._session.set_leverage(leverage, symbol=symbol)
        pass

    async def _do_set_margin_mode(self, symbol: str, mode: str) -> None:
        # self._session.set_margin_mode(mode.lower(), symbol=symbol)
        pass

    async def _do_set_hedge_mode(self, enabled: bool) -> None:
        # self._session.set_position_mode(enabled)
        pass

    def _normalize_balance(self, raw: Dict) -> Balance:
        info = raw.get("info", raw)
        total = float(info.get("totalCrossWalletBalance", 0))
        free = float(info.get("availableBalance", info.get("crossWalletBalance", 0)))
        used = total - free
        return Balance(total=total, free=free, used=used, quote="USDT")

    def _normalize_positions(self, raw: List[Dict]) -> List[Position]:
        positions = []
        for elm in raw:
            if float(elm.get("positionAmt", 0)) != 0:
                pos = Position(
                    symbol=elm["symbol"].replace("USDT", ""),  # normalize
                    side=elm.get("positionSide", "long").lower(),
                    size=abs(float(elm["positionAmt"])),
                    entry_price=float(elm["entryPrice"]),
                )
                positions.append(pos)
        return positions


# ---------------------------------------------------------------------------
# OKX adapter
# ---------------------------------------------------------------------------


class OKXAdapter(ExchangeAdapter):
    """
    OKX futures adapter.

    Key characteristics:
    - Uses posSide in order params
    - Symbols: BTC-USDT-SWAP (perpetual)
    - Account modes: long_short_mode (hedge) vs net_mode (one-way)
    - Portfolio margin (PM) accounts have different behavior
    """

    exchange_name = "okx"

    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = "", **kwargs):
        super().__init__(api_key, api_secret, passphrase=passphrase, **kwargs)
        self.dual_side = True  # Hedge mode by default

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def _do_fetch_balance(self) -> Dict:
        # self._session.fetch_balance()
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def _do_fetch_positions(self) -> List[Dict]:
        return []

    async def _do_fetch_tickers(self) -> Dict:
        return {}

    async def _do_fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[List[float]]:
        return []

    async def _do_execute_order(self, order: OrderSpec) -> Dict:
        return {}

    async def _do_cancel_order(self, symbol: str, order_id: str) -> Dict:
        return {}

    async def _do_set_leverage(self, symbol: str, leverage: int) -> None:
        pass

    async def _do_set_margin_mode(self, symbol: str, mode: str) -> None:
        pass

    async def _do_set_hedge_mode(self, enabled: bool) -> None:
        self._hedge_mode = enabled
        self.dual_side = enabled

    def _get_position_side(self, elm: Dict) -> str:
        return elm.get("posSide", "long").lower()

    def _symbol_from_exchange(self, exchange_symbol: str) -> str:
        # OKX: BTC-USDT-SWAP -> BTC/USDT:USDT
        return exchange_symbol.replace("-", "/").replace("SWAP", ":USDT")


# ---------------------------------------------------------------------------
# Bybit adapter
# ---------------------------------------------------------------------------


class BybitAdapter(ExchangeAdapter):
    """
    Bybit futures adapter.

    Key characteristics:
    - Unified trading account (UTA) vs classic account
    - positionIdx: 1=long, 2=short (in hedge mode)
    - Symbols: BTCUSDT, ETHUSDT, etc.
    """

    exchange_name = "bybit"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def _do_fetch_balance(self) -> Dict:
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def _do_fetch_positions(self) -> List[Dict]:
        return []

    async def _do_fetch_tickers(self) -> Dict:
        return {}

    async def _do_fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[List[float]]:
        return []

    async def _do_execute_order(self, order: OrderSpec) -> Dict:
        return {}

    async def _do_cancel_order(self, symbol: str, order_id: str) -> Dict:
        return {}

    async def _do_set_leverage(self, symbol: str, leverage: int) -> None:
        pass

    async def _do_set_margin_mode(self, symbol: str, mode: str) -> None:
        pass

    async def _do_set_hedge_mode(self, enabled: bool) -> None:
        self._hedge_mode = enabled

    def _get_position_side(self, elm: Dict) -> str:
        idx = float(elm.get("positionIdx", 0))
        if idx == 1.0:
            return "long"
        if idx == 2.0:
            return "short"
        side = elm.get("side", "").lower()
        return "long" if side == "buy" else "short"


# ---------------------------------------------------------------------------
# Bitget adapter
# ---------------------------------------------------------------------------


class BitgetAdapter(ExchangeAdapter):
    """
    Bitget futures adapter.

    Key characteristics:
    - Symbols: BTCUSDT_UMCBL (perpetual swap)
    - positionSide: long/short/net_mode
    - Margin mode: cross/isolated
    """

    exchange_name = "bitget"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def _do_fetch_balance(self) -> Dict:
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def _do_fetch_positions(self) -> List[Dict]:
        return []

    async def _do_fetch_tickers(self) -> Dict:
        return {}

    async def _do_fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[List[float]]:
        return []

    async def _do_execute_order(self, order: OrderSpec) -> Dict:
        return {}

    async def _do_cancel_order(self, symbol: str, order_id: str) -> Dict:
        return {}

    async def _do_set_leverage(self, symbol: str, leverage: int) -> None:
        pass

    async def _do_set_margin_mode(self, symbol: str, mode: str) -> None:
        pass

    async def _do_set_hedge_mode(self, enabled: bool) -> None:
        self._hedge_mode = enabled


# ---------------------------------------------------------------------------
# GateIO adapter
# ---------------------------------------------------------------------------


class GateIOAdapter(ExchangeAdapter):
    """
    Gate.io futures adapter.

    Key characteristics:
    - Symbols: BTC_USD (perpetual futures)
    - Uses contract type: 'perpetual'
    - Mode: cross_margin / isolated
    """

    exchange_name = "gateio"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def _do_fetch_balance(self) -> Dict:
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def _do_fetch_positions(self) -> List[Dict]:
        return []

    async def _do_fetch_tickers(self) -> Dict:
        return {}

    async def _do_fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[List[float]]:
        return []

    async def _do_execute_order(self, order: OrderSpec) -> Dict:
        return {}

    async def _do_cancel_order(self, symbol: str, order_id: str) -> Dict:
        return {}

    async def _do_set_leverage(self, symbol: str, leverage: int) -> None:
        pass

    async def _do_set_margin_mode(self, symbol: str, mode: str) -> None:
        pass

    async def _do_set_hedge_mode(self, enabled: bool) -> None:
        self._hedge_mode = enabled


# ---------------------------------------------------------------------------
# Hyperliquid adapter
# ---------------------------------------------------------------------------


class HyperliquidAdapter(ExchangeAdapter):
    """
    Hyperliquid perpetuals adapter.

    Key characteristics:
    - Uses WebSocket for all data (no REST for fills)
    - Symbols: BTC, ETH (no quote suffix)
    - Uses Hyperliquid's "Vault" concept
    - No position side in orders (one-way per user)
    """

    exchange_name = "hyperliquid"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def _do_fetch_balance(self) -> Dict:
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def _do_fetch_positions(self) -> List[Dict]:
        return []

    async def _do_fetch_tickers(self) -> Dict:
        return {}

    async def _do_fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[List[float]]:
        return []

    async def _do_execute_order(self, order: OrderSpec) -> Dict:
        return {}

    async def _do_cancel_order(self, symbol: str, order_id: str) -> Dict:
        return {}

    async def _do_set_leverage(self, symbol: str, leverage: int) -> None:
        pass

    async def _do_set_margin_mode(self, symbol: str, mode: str) -> None:
        pass

    async def _do_set_hedge_mode(self, enabled: bool) -> None:
        # Hyperliquid is always one-way (no hedge mode)
        self._hedge_mode = False

    def _symbol_from_exchange(self, exchange_symbol: str) -> str:
        # Hyperliquid: BTC -> BTC/USDT:USDT
        return f"{exchange_symbol}/USDT:USDT"


# ---------------------------------------------------------------------------
# KuCoin adapter
# ---------------------------------------------------------------------------


class KuCoinAdapter(ExchangeAdapter):
    """
    KuCoin futures adapter.

    Key characteristics:
    - Symbols: XBTUSDTM, ETHUSDTM (perpetual)
    - Uses positionSide in hedge mode
    - Margin mode: cross/isolated
    """

    exchange_name = "kucoin"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def _do_fetch_balance(self) -> Dict:
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    async def _do_fetch_positions(self) -> List[Dict]:
        return []

    async def _do_fetch_tickers(self) -> Dict:
        return {}

    async def _do_fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int
    ) -> List[List[float]]:
        return []

    async def _do_execute_order(self, order: OrderSpec) -> Dict:
        return {}

    async def _do_cancel_order(self, symbol: str, order_id: str) -> Dict:
        return {}

    async def _do_set_leverage(self, symbol: str, leverage: int) -> None:
        pass

    async def _do_set_margin_mode(self, symbol: str, mode: str) -> None:
        pass

    async def _do_set_hedge_mode(self, enabled: bool) -> None:
        self._hedge_mode = enabled


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------


def create_adapter(
    exchange: str,
    api_key: str = "",
    api_secret: str = "",
    password: str = "",
    **kwargs,
) -> ExchangeAdapter:
    """
    Factory to create an exchange adapter by name.

    Args:
        exchange: Exchange name (binance, okx, bybit, bitget, gateio, hyperliquid, kucoin)
        api_key: API key
        api_secret: API secret
        password: Password (for some exchanges)

    Returns:
        ExchangeAdapter instance
    """
    exchange = exchange.lower()
    adapters = {
        "binance": BinanceAdapter,
        "okx": OKXAdapter,
        "bybit": BybitAdapter,
        "bitget": BitgetAdapter,
        "gateio": GateIOAdapter,
        "gate": GateIOAdapter,
        "hyperliquid": HyperliquidAdapter,
        "kucoin": KuCoinAdapter,
    }
    if exchange not in adapters:
        raise ValueError(f"Unknown exchange: {exchange}")
    return adapters[exchange](api_key, api_secret, password=password, **kwargs)
