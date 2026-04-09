"""
Local Depth Cache

Local order book depth cache — maintains recent N levels of quotes.
Supports:
- Real-time incremental updates
- Trade matching
- Spread calculation
- Depth calculation

Bilingual docstrings (English/Chinese).

Usage:
    >>> from quant_trading.connectors.binance_depth_cache import LocalDepthCache
    >>> cache = LocalDepthCache("BTCUSDT", levels=20)
    >>> cache.update_bids([("50000.0", "1.5"), ("49900.0", "2.0")])
    >>> cache.update_asks([("50100.0", "1.0"), ("50200.0", "3.0")])
    >>> print(f"Spread: {cache.get_spread()}")
    >>> print(f"Midprice: {cache.get_midprice()}")
"""

from __future__ import annotations

import logging
import threading
import time
from decimal import Decimal
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("LocalDepthCache")


class LocalDepthCache:
    """
    Local order book depth cache — maintains recent N price levels.

    Features:
    - Real-time incremental updates from WebSocket or REST
    - Trade matching for market-making
    - Spread calculation
    - Depth (volume) calculation
    - Thread-safe operations

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        levels: Number of price levels to maintain (default 20)
        price_precision: Decimal precision for prices (default 2)
        qty_precision: Decimal precision for quantities (default 8)

    Attributes:
        symbol: Trading pair symbol
        levels: Number of price levels
        bid_prices: Sorted list of bid prices (descending)
        ask_prices: Sorted list of ask prices (ascending)

    Example:
        >>> cache = LocalDepthCache("BTCUSDT", levels=20)
        >>> cache.update_bids([("50000.0", "1.5"), ("49900.0", "2.0")])
        >>> cache.update_asks([("50100.0", "1.0"), ("50200.0", "3.0")])
        >>> print(f"Bid 0: {cache.get_bid_price(0)}, Ask 0: {cache.get_ask_price(0)}")
    """

    def __init__(
        self,
        symbol: str,
        levels: int = 20,
        price_precision: int = 2,
        qty_precision: int = 8,
    ):
        self.symbol = symbol.upper()
        self.levels = levels
        self.price_precision = price_precision
        self.qty_precision = qty_precision

        # Order book storage: price -> quantity
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._lock = threading.RLock()

        # Statistics
        self._last_update_time: float = 0
        self._update_count: int = 0
        self._last_trade_price: Decimal = Decimal("0")
        self._last_trade_side: str = ""

        # Callbacks
        self._on_update: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None

        self._logger = logging.getLogger(f"LocalDepthCache.{self.symbol}")

    @property
    def bid_prices(self) -> List[Decimal]:
        """Return sorted list of bid prices (descending)."""
        with self._lock:
            return sorted(self._bids.keys(), reverse=True)

    @property
    def ask_prices(self) -> List[Decimal]:
        """Return sorted list of ask prices (ascending)."""
        with self._lock:
            return sorted(self._asks.keys(), reverse=False)

    @property
    def last_update_time(self) -> float:
        """Return timestamp of last update."""
        return self._last_update_time

    @property
    def update_count(self) -> int:
        """Return number of updates received."""
        return self._update_count

    # ===================
    # Update Methods / 更新方法
    # ===================

    def update_bids(self, bids: List[tuple]) -> None:
        """
        Update bid levels with list of (price, quantity) tuples.

        Zero quantity removes the price level.

        Args:
            bids: List of (price, quantity) tuples as strings or numbers
                e.g., [("50000.0", "1.5"), ("49900.0", "2.0")]

        Example:
            >>> cache.update_bids([("50000.0", "1.5"), ("49800.0", "0")])
        """
        with self._lock:
            for price_str, qty_str in bids:
                price = self._to_decimal(price_str)
                qty = self._to_decimal(qty_str)

                if qty == 0:
                    self._bids.pop(price, None)
                else:
                    self._bids[price] = qty

            # Trim to levels limit
            self._trim_bids()
            self._update_stats()

    def update_asks(self, asks: List[tuple]) -> None:
        """
        Update ask levels with list of (price, quantity) tuples.

        Zero quantity removes the price level.

        Args:
            asks: List of (price, quantity) tuples as strings or numbers
                e.g., [("50100.0", "1.0"), ("50200.0", "3.0")]

        Example:
            >>> cache.update_asks([("50100.0", "1.0"), ("50300.0", "0")])
        """
        with self._lock:
            for price_str, qty_str in asks:
                price = self._to_decimal(price_str)
                qty = self._to_decimal(qty_str)

                if qty == 0:
                    self._asks.pop(price, None)
                else:
                    self._asks[price] = qty

            # Trim to levels limit
            self._trim_asks()
            self._update_stats()

    def apply_update(
        self,
        bids: Optional[List[tuple]] = None,
        asks: Optional[List[tuple]] = None,
        last_update_id: Optional[int] = None,
    ) -> bool:
        """
        Apply depth update from WebSocket.

        Args:
            bids: List of bid (price, qty) tuples
            asks: List of ask (price, qty) tuples
            last_update_id: Update sequence ID (optional, for validation)

        Returns:
            True if update was applied successfully

        Note:
            This method is used for WebSocket depth updates which may
            contain only changed levels.
        """
        if bids:
            self.update_bids(bids)
        if asks:
            self.update_asks(asks)
        return True

    def _trim_bids(self) -> None:
        """Trim bids to maintain levels limit."""
        if len(self._bids) > self.levels:
            sorted_prices = sorted(self._bids.keys(), reverse=True)
            for price in sorted_prices[self.levels:]:
                del self._bids[price]

    def _trim_asks(self) -> None:
        """Trim asks to maintain levels limit."""
        if len(self._asks) > self.levels:
            sorted_prices = sorted(self._asks.keys(), reverse=False)
            for price in sorted_prices[self.levels:]:
                del self._asks[price]

    def _update_stats(self) -> None:
        """Update statistics after an update."""
        self._update_count += 1
        self._last_update_time = time.time()

    def _to_decimal(self, value) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    # ===================
    # Price Queries / 价格查询
    # ===================

    def get_bid_price(self, level: int = 0) -> float:
        """
        Get bid price at specific level.

        Args:
            level: Level index (0 = best bid, 1 = next, etc.)

        Returns:
            Bid price as float, or 0.0 if level doesn't exist

        Example:
            >>> cache.get_bid_price(0)   # Best bid
            50000.0
            >>> cache.get_bid_price(1)   # Second level
            49900.0
        """
        with self._lock:
            sorted_bids = sorted(self._bids.keys(), reverse=True)
            if level < len(sorted_bids):
                return float(sorted_bids[level])
            return 0.0

    def get_ask_price(self, level: int = 0) -> float:
        """
        Get ask price at specific level.

        Args:
            level: Level index (0 = best ask, 1 = next, etc.)

        Returns:
            Ask price as float, or 0.0 if level doesn't exist

        Example:
            >>> cache.get_ask_price(0)   # Best ask
            50100.0
            >>> cache.get_ask_price(1)   # Second level
            50200.0
        """
        with self._lock:
            sorted_asks = sorted(self._asks.keys(), reverse=False)
            if level < len(sorted_asks):
                return float(sorted_asks[level])
            return 0.0

    def get_spread(self) -> float:
        """
        Get bid-ask spread.

        Returns:
            Spread as float (best_ask - best_bid), or 0.0 if not available

        Example:
            >>> cache.get_spread()
            100.0
        """
        best_bid = self.get_bid_price(0)
        best_ask = self.get_ask_price(0)

        if best_bid > 0 and best_ask > 0:
            return best_ask - best_bid
        return 0.0

    def get_midprice(self) -> float:
        """
        Get mid price (average of best bid and best ask).

        Returns:
            Mid price as float, or 0.0 if not available

        Example:
            >>> cache.get_midprice()
            50050.0
        """
        best_bid = self.get_bid_price(0)
        best_ask = self.get_ask_price(0)

        if best_bid > 0 and best_ask > 0:
            return (best_bid + best_ask) / 2.0
        return 0.0

    def get_depth(self) -> Tuple[List[tuple], List[tuple]]:
        """
        Get full order book depth.

        Returns:
            Tuple of (bids, asks) where each is a list of (price, quantity) tuples
            sorted: bids descending by price, asks ascending by price

        Example:
            >>> bids, asks = cache.get_depth()
            >>> print(f"Bids: {bids[:3]}, Asks: {asks[:3]}")
        """
        with self._lock:
            sorted_bids = sorted(self._bids.keys(), reverse=True)
            sorted_asks = sorted(self._asks.keys(), reverse=False)

            bids = [(float(p), float(self._bids[p])) for p in sorted_bids]
            asks = [(float(p), float(self._asks[p])) for p in sorted_asks]

            return bids, asks

    def get_bids_qty_at_price(self, price: float, side: str = "bid") -> float:
        """
        Get total quantity at a specific price level.

        Args:
            price: Price level
            side: 'bid' or 'ask'

        Returns:
            Quantity at that price level, or 0.0
        """
        with self._lock:
            p = self._to_decimal(price)
            if side == "bid":
                return float(self._bids.get(p, Decimal("0")))
            else:
                return float(self._asks.get(p, Decimal("0")))

    # ===================
    # Depth Calculations / 深度计算
    # ===================

    def get_bid_depth(self, levels: int = None) -> float:
        """
        Get total bid quantity across N levels.

        Args:
            levels: Number of levels to sum (None = all levels)

        Returns:
            Total bid quantity

        Example:
            >>> cache.get_bid_depth(10)   # Sum of top 10 bid levels
        """
        with self._lock:
            sorted_bids = sorted(self._bids.keys(), reverse=True)
            if levels:
                sorted_bids = sorted_bids[:levels]

            return sum(float(self._bids[p]) for p in sorted_bids)

    def get_ask_depth(self, levels: int = None) -> float:
        """
        Get total ask quantity across N levels.

        Args:
            levels: Number of levels to sum (None = all levels)

        Returns:
            Total ask quantity
        """
        with self._lock:
            sorted_asks = sorted(self._asks.keys(), reverse=False)
            if levels:
                sorted_asks = sorted_asks[:levels]

            return sum(float(self._asks[p]) for p in sorted_asks)

    def get_vwap(self, side: str, amount: float) -> float:
        """
        Calculate volume-weighted average price for a given amount.

        Args:
            side: 'bid' (buy) or 'ask' (sell)
            amount: Target amount to calculate VWAP for

        Returns:
            VWAP as float, or 0.0 if insufficient liquidity

        Example:
            >>> cache.get_vwap('bid', 1.0)   # VWAP for buying 1 BTC
        """
        with self._lock:
            if side == "bid":
                # Buy: sweep asks from lowest
                levels = sorted(self._asks.items(), key=lambda x: x[0])
            else:
                # Sell: sweep bids from highest
                levels = sorted(self._bids.items(), key=lambda x: -x[0])

            remaining = Decimal(str(amount))
            total_cost = Decimal("0")

            for price, qty in levels:
                if remaining <= 0:
                    break

                fill_qty = min(remaining, qty)
                total_cost += fill_qty * price
                remaining -= fill_qty

            if remaining > 0:
                return 0.0  # Insufficient liquidity

            return float(total_cost / Decimal(str(amount)))

    # ===================
    # Trade Processing / 成交处理
    # ===================

    def process_trade(
        self,
        trade_price: float,
        trade_qty: float,
        trade_side: str,
    ) -> None:
        """
        Process a trade and update local book.

        For trade-driven updates, adjusts local book based on trade direction.

        Args:
            trade_price: Trade execution price
            trade_qty: Trade quantity
            trade_side: 'buy' or 'sell' (from taker's perspective)

        Note:
            When a trade occurs:
            - If buy (taker buys): best ask is consumed, reduce ask qty
            - If sell (taker sells): best bid is consumed, reduce bid qty

        Example:
            >>> cache.process_trade(50000.0, 0.5, 'buy')  # Buyer took 0.5 from ask
        """
        with self._lock:
            self._last_trade_price = self._to_decimal(trade_price)
            self._last_trade_side = trade_side

            price = self._to_decimal(trade_price)
            qty = self._to_decimal(trade_qty)

            if trade_side.lower() == "buy":
                # Buyer is buying (taking from asks)
                if price in self._asks:
                    remaining = self._asks[price] - qty
                    if remaining <= 0:
                        self._asks.pop(price, None)
                    else:
                        self._asks[price] = remaining
            else:
                # Seller is selling (taking from bids)
                if price in self._bids:
                    remaining = self._bids[price] - qty
                    if remaining <= 0:
                        self._bids.pop(price, None)
                    else:
                        self._bids[price] = remaining

            # Notify trade callback
            if self._on_trade:
                try:
                    self._on_trade({
                        "price": trade_price,
                        "qty": trade_qty,
                        "side": trade_side,
                        "symbol": self.symbol,
                    })
                except Exception as e:
                    self._logger.error(f"Trade callback error: {e}")

    def set_trade_callback(self, callback: Callable) -> None:
        """
        Set callback for trade events.

        Args:
            callback: Function to call on trade events
        """
        self._on_trade = callback

    def set_update_callback(self, callback: Callable) -> None:
        """
        Set callback for depth update events.

        Args:
            callback: Function to call on depth updates
        """
        self._on_update = callback

    # ===================
    # Utility Methods / 工具方法
    # ===================

    def reset(self) -> None:
        """
        Reset the order book to empty state.

        Example:
            >>> cache.reset()
        """
        with self._lock:
            self._bids.clear()
            self._asks.clear()
            self._update_count = 0
            self._last_update_time = 0
            self._logger.info("Depth cache reset")

    def get_stats(self) -> dict:
        """
        Get depth cache statistics.

        Returns:
            dict with statistics:
            {
                'symbol': 'BTCUSDT',
                'bid_levels': 20,
                'ask_levels': 20,
                'update_count': 1500,
                'spread': 100.0,
                'midprice': 50050.0,
            }

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Spread: {stats['spread']}")
        """
        with self._lock:
            return {
                "symbol": self.symbol,
                "bid_levels": len(self._bids),
                "ask_levels": len(self._asks),
                "update_count": self._update_count,
                "spread": self.get_spread(),
                "midprice": self.get_midprice(),
                "best_bid": self.get_bid_price(0),
                "best_ask": self.get_ask_price(0),
                "last_trade_price": float(self._last_trade_price),
                "last_trade_side": self._last_trade_side,
            }

    def __repr__(self) -> str:
        return (
            f"LocalDepthCache(symbol={self.symbol}, "
            f"levels={self.levels}, "
            f"bids={len(self._bids)}, "
            f"asks={len(self._asks)}, "
            f"spread={self.get_spread():.2f})"
        )


__all__ = [
    "LocalDepthCache",
]
