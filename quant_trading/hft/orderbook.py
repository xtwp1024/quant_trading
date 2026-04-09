"""
quant_trading.hft.orderbook
===========================
Real-time order book manager for HFT market making.

Adapted from py_polymarket_hft_mm OrderBook:
- Thread-safe order book state management
- Depth tracking (bids/asks with price and size)
- Micro-price and BPS signal calculation
- WebSocket update handling
- Incremental book updates via bisect

Key concepts:
- Price-time priority (standard limit order book)
- Micro-price = volume-weighted mid (signal generation)
- BPS threshold monitoring for trading signals
"""

import bisect
import json
import logging
import threading
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal Enum
# ---------------------------------------------------------------------------


class BPSSignal(Enum):
    """Trading signal derived from order book micro-price vs mid BPS spread."""
    UP = "UP"       # Micro > Mid by threshold -> bid pressure -> UP
    DOWN = "DOWN"   # Micro < Mid by threshold -> ask pressure -> DOWN
    NEUTRAL = "NEUTRAL"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class PriceLevel:
    """A single price level in the order book."""
    price: float
    size: float

    def __repr__(self) -> str:
        return f"PriceLevel(price={self.price}, size={self.size})"


@dataclass
class MarketData:
    """Snapshot of current market state."""
    best_bid_price: float = 0.0
    best_bid_volume: float = 0.0
    best_ask_price: float = 0.0
    best_ask_volume: float = 0.0
    micro_price: float = 0.0
    mid_price: float = 0.0
    micro_vs_mid_bps: float = 0.0
    spread_bps: float = 0.0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Order Book Manager
# ---------------------------------------------------------------------------


class OrderBookManager:
    """
    Thread-safe real-time order book manager.

    Maintains an in-memory order book with:
    - Sorted bid/ask lists (price descending for bids, ascending for asks)
    - Micro-price calculation (volume-weighted mid)
    - BPS signal monitoring
    - Incremental update support via bisect

    Usage:
        book = OrderBookManager()
        book.update_snapshot(bids, asks)
        data = book.get_market_data()
        signal = book.get_signal(threshold_bps=50)
    """

    def __init__(
        self,
        price_precision: int = 2,
        size_precision: int = 4,
    ):
        """
        Args:
            price_precision: Decimal places for price levels (default 2).
            size_precision: Decimal places for size levels (default 4).
        """
        self.price_precision = price_precision
        self.size_precision = size_precision

        # Internal state: lists of [price, size], kept sorted
        self._bids: List[List[float]] = []   # sorted descending by price
        self._asks: List[List[float]] = []   # sorted ascending by price
        self._lock = threading.Lock()

        # Signal state
        self._last_signal = BPSSignal.NEUTRAL

        # Book health
        self._last_update_time: Optional[float] = None
        self._update_count: int = 0

    # -------------------------------------------------------------------------
    # Thread-safe accessors
    # -------------------------------------------------------------------------

    def get_market_data(self) -> Optional[MarketData]:
        """
        Return a MarketData snapshot, or None if the book is empty.

        All values are computed from the current sorted bid/ask lists.
        """
        with self._lock:
            if not self._bids or not self._asks:
                return None

            best_bid_price, best_bid_volume = self._bids[0]
            best_ask_price, best_ask_volume = self._asks[0]

            total_vol = best_bid_volume + best_ask_volume
            if total_vol > 0:
                micro_price = (
                    best_bid_price * best_ask_volume
                    + best_ask_price * best_bid_volume
                ) / total_vol
            else:
                micro_price = (best_bid_price + best_ask_price) / 2

            mid_price = (best_bid_price + best_ask_price) / 2
            micro_vs_mid_bps = (micro_price - mid_price) * 10000
            spread_bps = (best_ask_price - best_bid_price) / best_ask_price * 10000

            return MarketData(
                best_bid_price=best_bid_price,
                best_bid_volume=best_bid_volume,
                best_ask_price=best_ask_price,
                best_ask_volume=best_ask_volume,
                micro_price=micro_price,
                mid_price=mid_price,
                micro_vs_mid_bps=micro_vs_mid_bps,
                spread_bps=spread_bps,
                timestamp=self._last_update_time or time.time(),
            )

    def get_signal(self, threshold_bps: float = 50.0) -> BPSSignal:
        """
        Return a trading signal based on micro-price vs mid BPS spread.

        Args:
            threshold_bps: Minimum BPS deviation to trigger a signal.

        Returns:
            BPSSignal.UP   if micro_vs_mid_bps > +threshold_bps
            BPSSignal.DOWN if micro_vs_mid_bps < -threshold_bps
            BPSSignal.NEUTRAL otherwise.
        """
        data = self.get_market_data()
        if data is None:
            return BPSSignal.NEUTRAL

        if data.micro_vs_mid_bps > threshold_bps:
            self._last_signal = BPSSignal.UP
        elif data.micro_vs_mid_bps < -threshold_bps:
            self._last_signal = BPSSignal.DOWN
        else:
            self._last_signal = BPSSignal.NEUTRAL

        return self._last_signal

    @property
    def last_signal(self) -> BPSSignal:
        """Return the most recently computed signal."""
        return self._last_signal

    @property
    def is_healthy(self) -> bool:
        """Return True if the book has received a recent update."""
        with self._lock:
            return self._last_update_time is not None

    @property
    def update_count(self) -> int:
        """Number of updates processed since initialization."""
        with self._lock:
            return self._update_count

    def get_depth(self, levels: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """
        Return the top N price levels for bids and asks.

        Args:
            levels: Number of levels to return per side.

        Returns:
            Dict with 'bids' and 'asks', each a list of (price, size) tuples.
        """
        with self._lock:
            bids = [(round(p, self.price_precision), round(s, self.size_precision))
                    for p, s in self._bids[:levels]]
            asks = [(round(p, self.price_precision), round(s, self.size_precision))
                    for p, s in self._asks[:levels]]
        return {"bids": bids, "asks": asks}

    def get_weighted_midpoint(self) -> Optional[float]:
        """
        Return the volume-weighted midpoint (micro-price).

        Alias for calculate_micro_price on current best bid/ask.
        """
        data = self.get_market_data()
        return data.micro_price if data else None

    # -------------------------------------------------------------------------
    # Snapshot updates (full book replacement)
    # -------------------------------------------------------------------------

    def update_snapshot(
        self,
        bids: List[Any],
        asks: List[Any],
    ) -> None:
        """
        Replace the order book with a full snapshot from WebSocket.

        Args:
            bids: List of bid entries. Each entry can be:
                  - dict with 'price' and 'size' keys, or
                  - list/tuple [price, size].
            asks: Same format for ask entries.
        """
        parsed_bids = self._parse_levels(bids)
        parsed_asks = self._parse_levels(asks)

        with self._lock:
            # Sort: bids descending by price, asks ascending by price
            parsed_bids.sort(key=lambda x: x[0], reverse=True)
            parsed_asks.sort(key=lambda x: x[0])

            self._bids = parsed_bids
            self._asks = parsed_asks
            self._last_update_time = time.perf_counter()
            self._update_count += 1

    def _parse_levels(
        self,
        levels: List[Any],
    ) -> List[List[float]]:
        """Parse a list of raw price-level entries into [price, size] pairs."""
        result = []
        for entry in levels:
            if isinstance(entry, dict):
                price = float(entry.get("price", 0))
                size = float(entry.get("size", 0))
            else:
                price = float(entry[0])
                size = float(entry[1])
            if price > 0 and size > 0:
                result.append([price, size])
        return result

    # -------------------------------------------------------------------------
    # Incremental updates (delta processing via bisect)
    # -------------------------------------------------------------------------

    def apply_incremental_update(
        self,
        price: float,
        size: float,
        side: str,  # "BUY" = bid, "SELL" = ask
    ) -> None:
        """
        Apply an incremental order book change (single price level update).

        Uses binary search (bisect) for O(log N) insertion/deletion.

        Args:
            price: Price level to update.
            size:  New size at this price (0 = remove level).
            side:  "BUY" for bids, "SELL" for asks.
        """
        with self._lock:
            book_side = self._bids if side == "BUY" else self._asks

            # Normalize to [price, size] format
            if book_side and isinstance(book_side[0], list) and len(book_side[0]) == 2:
                # Already in flat format
                target = book_side
            else:
                # Dict format -- convert on demand
                target = book_side

            idx = bisect.bisect_left(target, [price, 0.0])

            if idx < len(target) and target[idx][0] == price:
                # Level exists
                if size == 0:
                    del target[idx]  # Remove level
                else:
                    target[idx][1] = size  # Update size
            elif size > 0:
                # New level
                bisect.insort(target, [price, size])

            # Re-sort just in case
            if side == "BUY":
                target.sort(key=lambda x: x[0], reverse=True)
            else:
                target.sort(key=lambda x: x[0])

            self._last_update_time = time.perf_counter()
            self._update_count += 1

    def cull_opposite_side(
        self,
        price: float,
        side: str,
    ) -> None:
        """
        Remove all price levels on the opposite side that would cross.

        When a bid is placed at 'price', all asks with price <= price
        would cross, so they are removed (and vice versa).

        Args:
            price: The trigger price.
            side:  "BUY" for bid, "SELL" for ask.
        """
        with self._lock:
            if side == "BUY":
                # Remove asks with price <= price (would cross with new bid)
                self._asks = [[p, s] for p, s in self._asks if p > price]
            else:
                # Remove bids with price >= price (would cross with new ask)
                self._bids = [[p, s] for p, s in self._bids if p < price]

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear the entire order book."""
        with self._lock:
            self._bids.clear()
            self._asks.clear()
            self._last_update_time = None

    def __repr__(self) -> str:
        data = self.get_market_data()
        if data is None:
            return "OrderBookManager(empty)"
        return (
            f"OrderBookManager("
            f"bid={data.best_bid_price}/{data.best_bid_volume}, "
            f"ask={data.best_ask_price}/{data.best_ask_volume}, "
            f"micro={data.micro_price:.4f}, "
            f"signal={self._last_signal.value})"
        )
