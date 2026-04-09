"""
Binance Order Book Cache Manager

Provides local order book caching and management for Binance trading pairs,
with support for real-time depth cache updates via WebSocket.

Features:
- Local order book state management
- Depth cache with snapshot and delta updates
- Best bid/ask calculation
- Spread calculation
- Price level aggregation
- Volume calculation
- Thread-safe operations

Inspired by unicorn-binance-local-depth-cache package patterns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

# Constants
MAX_ORDER_BOOK_DEPTH = 5000
DEFAULT_DEPTH = 100
PRICE_LEVELS_AGGREGATION = {
    "0.01": 1,
    "0.1": 1,
    "1": 1,
    "10": 1,
    "100": 1,
    "1000": 1,
    "10000": 1,
}


@dataclass
class PriceLevel:
    """Represents a price level in the order book."""

    price: Decimal
    quantity: Decimal

    @property
    def total_value(self) -> Decimal:
        """Return total value at this price level (price * quantity)."""
        return self.price * self.quantity

    def __repr__(self):
        return f"PriceLevel(price={self.price}, qty={self.quantity})"


@dataclass
class OrderBookSnapshot:
    """Full order book snapshot."""

    symbol: str
    bids: List[PriceLevel]  # Sorted by price descending
    asks: List[PriceLevel]  # Sorted by price ascending
    last_update_id: int
    timestamp: float

    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Return best bid (highest price)."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Return best ask (lowest price)."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Return bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    @property
    def spread_percent(self) -> Optional[Decimal]:
        """Return spread as percentage of mid price."""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 100
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Return mid price (average of best bid and best ask)."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None

    def get_volume_at_price(
        self, price: Decimal, side: str, depth: Decimal = Decimal("1000000")
    ) -> Decimal:
        """
        Get volume available at a specific price level.

        Args:
            price: Target price
            side: 'bid' or 'ask'
            depth: Maximum distance from best price to search

        Returns:
            Available volume
        """
        if side == "bid":
            levels = self.bids
            best = self.best_bid.price if self.best_bid else None
        else:
            levels = self.asks
            best = self.best_ask.price if self.best_ask else None

        if best is None:
            return Decimal("0")

        volume = Decimal("0")
        for level in levels:
            if abs(level.price - best) > depth:
                break
            volume += level.quantity

        return volume


@dataclass
class OrderBookState:
    """
    Current state of the order book for a symbol.

    Maintains sorted bid/ask lists with efficient update operations.
    """

    symbol: str
    bids: Dict[Decimal, Decimal] = field(default_factory=dict)  # price -> quantity
    asks: Dict[Decimal, Decimal] = field(default_factory=dict)
    last_update_id: int = 0
    last_update_time: float = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def update_bids(self, updates: List[Tuple[Decimal, Decimal]]):
        """Update bid levels with price -> quantity pairs."""
        async with self._lock:
            for price, qty in updates:
                if qty == 0:
                    self.bids.pop(price, None)
                else:
                    self.bids[price] = qty

    async def update_asks(self, updates: List[Tuple[Decimal, Decimal]]):
        """Update ask levels with price -> quantity pairs."""
        async with self._lock:
            for price, qty in updates:
                if qty == 0:
                    self.asks.pop(price, None)
                else:
                    self.asks[price] = qty

    async def apply_depth_update(
        self, update_id: int, bids: List[Tuple[Decimal, Decimal]], asks: List[Tuple[Decimal, Decimal]]
    ):
        """
        Apply depth cache update from WebSocket.

        Updates are applied only if update_id is higher than current.
        Zero quantities remove price levels.
        """
        async with self._lock:
            if update_id <= self.last_update_id:
                return False  # Stale update

            # Apply bid updates
            for price, qty in bids:
                if qty == 0:
                    self.bids.pop(price, None)
                else:
                    self.bids[price] = qty

            # Apply ask updates
            for price, qty in asks:
                if qty == 0:
                    self.asks.pop(price, None)
                else:
                    self.asks[price] = qty

            self.last_update_id = update_id
            self.last_update_time = time.time()
            return True

    async def get_snapshot(self, depth: int = DEFAULT_DEPTH) -> OrderBookSnapshot:
        """Get current order book snapshot."""
        async with self._lock:
            # Sort and limit bids (descending by price)
            sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:depth]
            # Sort and limit asks (ascending by price)
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]

            return OrderBookSnapshot(
                symbol=self.symbol,
                bids=[PriceLevel(price=p, quantity=q) for p, q in sorted_bids],
                asks=[PriceLevel(price=p, quantity=q) for p, q in sorted_asks],
                last_update_id=self.last_update_id,
                timestamp=self.last_update_time,
            )

    async def reset(self):
        """Reset order book state."""
        async with self._lock:
            self.bids.clear()
            self.asks.clear()
            self.last_update_id = 0
            self.last_update_time = 0


class BinanceOrderBookManager:
    """
    Manager for local Binance order books.

    Provides caching and management of order book depth for multiple trading pairs
    with support for snapshot and delta updates.

    Features:
    - Multiple symbol support
    - Snapshot initialization
    - Real-time delta updates
    - Best bid/ask queries
    - Spread calculation
    - Volume calculations
    - Callbacks on updates

    Args:
        rest_connector: BinanceRESTConnector for fetching snapshots
        ws_connector: BinanceWSConnector for real-time updates
        default_depth: Default order book depth to maintain
    """

    def __init__(
        self,
        rest_connector: Optional["BinanceRESTConnector"] = None,
        ws_connector: Optional["BinanceWSConnector"] = None,
        default_depth: int = DEFAULT_DEPTH,
    ):
        self._rest = rest_connector
        self._ws = ws_connector
        self._default_depth = default_depth

        # Order book caches per symbol
        self._order_books: Dict[str, OrderBookState] = {}

        # Subscriptions
        self._subscriptions: Dict[str, asyncio.Task] = {}

        # Update callbacks
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Statistics
        self._update_counts: Dict[str, int] = defaultdict(int)
        self._last_update_times: Dict[str, float] = {}

        self._logger = logging.getLogger("BinanceOrderBookManager")

    @property
    def order_books(self) -> Dict[str, OrderBookState]:
        """Return all managed order books."""
        return self._order_books

    def get_order_book(self, symbol: str) -> Optional[OrderBookState]:
        """Get order book state for a symbol."""
        return self._order_books.get(symbol.upper())

    async def initialize_order_book(
        self, symbol: str, depth: int = DEFAULT_DEPTH
    ) -> OrderBookSnapshot:
        """
        Initialize order book for a symbol with REST API snapshot.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            depth: Order book depth to fetch

        Returns:
            Initial order book snapshot
        """
        symbol = symbol.upper()

        if not self._rest:
            raise ValueError("REST connector not provided")

        # Fetch snapshot from REST API
        raw_book = await self._rest.get_order_book(symbol, depth)

        # Create order book state
        state = OrderBookState(symbol=symbol)
        state.last_update_id = int(raw_book.get("lastUpdateId", 0))

        # Parse bids
        for price_str, qty_str in raw_book.get("bids", []):
            price = Decimal(price_str)
            qty = Decimal(qty_str)
            state.bids[price] = qty

        # Parse asks
        for price_str, qty_str in raw_book.get("asks", []):
            price = Decimal(price_str)
            qty = Decimal(qty_str)
            state.asks[price] = qty

        self._order_books[symbol] = state
        self._logger.info(f"Initialized order book for {symbol} with {depth} levels")

        return await state.get_snapshot(depth)

    async def start_depth_stream(self, symbol: str, levels: int = 100):
        """
        Start real-time depth stream for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            levels: Depth levels to subscribe (5, 10, 20, 100, 500, 1000)
        """
        symbol = symbol.upper()

        if not self._ws:
            raise ValueError("WebSocket connector not provided")

        if symbol in self._subscriptions:
            self._logger.warning(f"Already subscribed to {symbol}")
            return

        # Ensure order book is initialized
        if symbol not in self._order_books:
            await self.initialize_order_book(symbol, levels)

        # Subscribe to depth stream
        callback = lambda data: asyncio.create_task(self._handle_depth_update(symbol, data))
        stream_name = await self._ws.subscribe_depth(symbol, levels, callback)

        self._logger.info(f"Started depth stream for {symbol}: {stream_name}")

    async def stop_depth_stream(self, symbol: str):
        """Stop depth stream for a symbol."""
        symbol = symbol.upper()

        if symbol in self._subscriptions:
            task = self._subscriptions.pop(symbol)
            task.cancel()
            self._logger.info(f"Stopped depth stream for {symbol}")

    async def _handle_depth_update(self, symbol: str, data: Dict[str, Any]):
        """Handle incoming depth update from WebSocket."""
        state = self._order_books.get(symbol)
        if not state:
            return

        # Parse update data
        update_id = data.get("update_id", 0)
        bids = [(Decimal(p), Decimal(q)) for p, q in data.get("bids", [])]
        asks = [(Decimal(p), Decimal(q)) for p, q in data.get("asks", [])]

        # Apply update
        applied = await state.apply_depth_update(update_id, bids, asks)

        if applied:
            self._update_counts[symbol] += 1
            self._last_update_times[symbol] = time.time()

            # Notify callbacks
            snapshot = await state.get_snapshot()
            for callback in self._callbacks[symbol]:
                try:
                    callback(snapshot)
                except Exception as e:
                    self._logger.error(f"Callback error for {symbol}: {e}")

    async def get_best_bid_ask(self, symbol: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Get best bid and ask for a symbol.

        Returns:
            Tuple of (best_bid, best_ask)
        """
        state = self._order_books.get(symbol)
        if not state:
            return None, None

        async with state._lock:
            if not state.bids or not state.asks:
                return None, None

            best_bid = max(state.bids.keys())
            best_ask = min(state.asks.keys())

            return best_bid, best_ask

    async def get_spread(self, symbol: str) -> Optional[Decimal]:
        """
        Get bid-ask spread for a symbol.

        Returns:
            Spread as Decimal, or None if order book is empty
        """
        best_bid, best_ask = await self.get_best_bid_ask(symbol)
        if best_bid and best_ask:
            return best_ask - best_bid
        return None

    async def get_mid_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get mid price for a symbol.

        Returns:
            Mid price as Decimal, or None if order book is empty
        """
        state = self._order_books.get(symbol)
        if not state:
            return None

        async with state._lock:
            if not state.bids or not state.asks:
                return None

            best_bid = max(state.bids.keys())
            best_ask = min(state.asks.keys())

            return (best_bid + best_ask) / 2

    async def get_vwap(
        self, symbol: str, side: str, amount: Decimal
    ) -> Optional[Decimal]:
        """
        Calculate volume-weighted average price for a given amount.

        Args:
            symbol: Trading pair
            side: 'bid' (buy) or 'ask' (sell)
            amount: Target amount to calculate VWAP for

        Returns:
            VWAP as Decimal, or None if insufficient liquidity
        """
        state = self._order_books.get(symbol)
        if not state:
            return None

        async with state._lock:
            if side == "bid":
                levels = sorted(state.asks.items(), key=lambda x: x[0])  # Lowest ask first
            else:
                levels = sorted(state.bids.items(), key=lambda x: -x[0])  # Highest bid first

            remaining = amount
            total_cost = Decimal("0")

            for price, qty in levels:
                if remaining <= 0:
                    break

                fill_qty = min(remaining, qty)
                total_cost += fill_qty * price
                remaining -= fill_qty

            if remaining > 0:
                return None  # Insufficient liquidity

            return total_cost / amount

    def add_callback(self, symbol: str, callback: Callable):
        """Add update callback for a symbol."""
        self._callbacks[symbol.upper()].append(callback)

    def remove_callback(self, symbol: str, callback: Callable):
        """Remove update callback for a symbol."""
        symbol = symbol.upper()
        if symbol in self._callbacks:
            self._callbacks[symbol].remove(callback)

    async def get_aggregated_levels(
        self, symbol: str, side: str, tick_size: Decimal
    ) -> List[PriceLevel]:
        """
        Get aggregated price levels for a symbol.

        Args:
            symbol: Trading pair
            side: 'bid' or 'ask'
            tick_size: Aggregation tick size (e.g., Decimal("0.01"))

        Returns:
            List of aggregated price levels
        """
        state = self._order_books.get(symbol)
        if not state:
            return []

        async with state._lock:
            if side == "bid":
                levels = state.bids
            else:
                levels = state.asks

            # Aggregate by tick size
            aggregated: Dict[Decimal, Decimal] = {}
            for price, qty in levels.items():
                agg_price = (price // tick_size) * tick_size
                aggregated[agg_price] = aggregated.get(agg_price, Decimal("0")) + qty

            return [
                PriceLevel(price=p, quantity=q)
                for p, q in sorted(aggregated.items(), reverse=(side == "bid"))
            ]

    async def get_depth(
        self, symbol: str, side: str, depth_percent: Decimal = Decimal("1")
    ) -> Decimal:
        """
        Get total depth (volume) within a percentage of best price.

        Args:
            symbol: Trading pair
            side: 'bid' or 'ask'
            depth_percent: Percentage distance from best price

        Returns:
            Total volume within depth
        """
        state = self._order_books.get(symbol)
        if not state:
            return Decimal("0")

        async with state._lock:
            if side == "bid":
                levels = sorted(state.bids.items(), key=lambda x: -x[0])
                best = max(state.bids.keys()) if state.bids else None
            else:
                levels = sorted(state.asks.items(), key=lambda x: x[0])
                best = min(state.asks.keys()) if state.asks else None

            if best is None:
                return Decimal("0")

            depth_price = best * (depth_percent / 100)
            total_volume = Decimal("0")

            for price, qty in levels:
                if abs(price - best) <= depth_price:
                    total_volume += qty
                else:
                    break

            return total_volume

    def get_stats(self, symbol: str) -> Dict[str, Any]:
        """Get statistics for a symbol's order book."""
        return {
            "update_count": self._update_counts.get(symbol, 0),
            "last_update_time": self._last_update_times.get(symbol, 0),
            "bid_levels": len(self._order_books.get(symbol, OrderBookState(symbol="")).bids),
            "ask_levels": len(self._order_books.get(symbol, OrderBookState(symbol="")).asks),
        }

    async def close(self):
        """Close all order book streams and cleanup."""
        for symbol in list(self._subscriptions.keys()):
            await self.stop_depth_stream(symbol)
        self._order_books.clear()
        self._logger.info("Order book manager closed")
