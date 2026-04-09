"""Signal cache — stores and manages signal state across time.

Provides:
- SignalStore: Persistent signal history with symbol/time indexing
- SignalState: Current position state derived from signal stream
- SignalCache: LRU cache for computed signals (avoids recalculation)

Usage
-----
```python
from quant_trading.signal.cache import SignalStore, SignalState

store = SignalStore()
store.append(signal)
state = store.get_state("BTC/USDT")

# Check if we have a live position
if state.position != 0:
    print(f"Holding {state.position} from {state.entry_price}")
```
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from quant_trading.signal.types import Signal, SignalType, SignalDirection


@dataclass
class SignalState:
    """Current position state derived from signal stream.

    Attributes
    ----------
    symbol : str
        Trading pair.
    position : int
        Current position: 1 = long, -1 = short, 0 = flat.
    entry_price : float
        Average entry price.
    entry_timestamp : int
        Unix timestamp (ms) of entry.
    bars_held : int
        Number of bars since entry.
    stop_loss : float
        Active stop-loss price.
    take_profit : float
        Active take-profit price.
    unrealized_pnl : float
        Unrealized PnL from entry price to last known price.
    last_price : float
        Most recent known price.
    last_signal : Signal
        The most recent signal that affected state.
    """
    symbol: str
    position: int = 0  # 1=long, -1=short, 0=flat
    entry_price: float = 0.0
    entry_timestamp: int = 0
    bars_held: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    last_price: float = 0.0
    last_signal: Optional[Signal] = None

    def update_price(self, price: float, bar_index: int) -> None:
        """Update current price and unrealized PnL."""
        self.last_price = price
        if self.position != 0 and self.entry_price > 0:
            if self.position == 1:
                self.unrealized_pnl = (price - self.entry_price) / self.entry_price
            elif self.position == -1:
                self.unrealized_pnl = (self.entry_price - price) / self.entry_price
        if self.entry_timestamp > 0:
            self.bars_held = bar_index  # caller should manage bar index

    def check_stops(self, price: float) -> Optional[SignalType]:
        """Check if price has hit stop-loss or take-profit.

        Returns
        -------
        SignalType or None
            EXIT_LONG if stop-loss hit on long
            EXIT_SHORT if stop-loss hit on short
            CLOSE_ALL if TP hit (depending on config)
        """
        if self.position == 1:
            if 0 < self.stop_loss < price:
                return SignalType.EXIT_LONG
            if self.take_profit > 0 and price >= self.take_profit:
                return SignalType.EXIT_LONG
        elif self.position == -1:
            if self.stop_loss > 0 and price > self.stop_loss:
                return SignalType.EXIT_SHORT
            if self.take_profit > 0 and price <= self.take_profit:
                return SignalType.EXIT_SHORT
        return None


@dataclass
class SignalStore:
    """Chronological signal store per symbol.

    Supports:
    - append: Add a signal
    - get_since: Get signals since timestamp
    - get_by_type: Filter by SignalType
    - latest: Most recent signal
    - purge_before: Remove old signals (GDPR, memory)
    """

    _signals: Dict[str, List[Signal]] = field(default_factory=dict)
    _latest: Dict[str, Signal] = field(default_factory=dict)

    def append(self, signal: Signal) -> None:
        """Add a signal to the store."""
        sym = signal.symbol
        if sym not in self._signals:
            self._signals[sym] = []
        self._signals[sym].append(signal)
        self._latest[sym] = signal

    def get_all(self, symbol: str) -> List[Signal]:
        """Get all signals for a symbol."""
        return list(self._signals.get(symbol, []))

    def get_since(self, symbol: str, timestamp: int) -> List[Signal]:
        """Get signals since a given timestamp (inclusive)."""
        return [s for s in self._signals.get(symbol, []) if s.timestamp >= timestamp]

    def get_by_type(
        self, symbol: str, sig_type: SignalType
    ) -> List[Signal]:
        """Get all signals of a specific type for a symbol."""
        return [s for s in self._signals.get(symbol, []) if s.type == sig_type]

    def latest(self, symbol: str) -> Optional[Signal]:
        """Most recent signal for a symbol."""
        return self._latest.get(symbol)

    def purge_before(self, timestamp: int, symbol: Optional[str] = None) -> int:
        """Remove signals older than timestamp.

        Parameters
        ----------
        timestamp : int
            Purge signals before this timestamp.
        symbol : str, optional
            Purge only this symbol. If None, purge all.

        Returns
        -------
        int
            Number of signals purged.
        """
        if symbol:
            before = [s for s in self._signals.get(symbol, []) if s.timestamp < timestamp]
            self._signals[symbol] = [s for s in self._signals.get(symbol, []) if s.timestamp >= timestamp]
            return len(before)
        else:
            total = 0
            for sym in self._signals:
                before = [s for s in self._signals[sym] if s.timestamp < timestamp]
                self._signals[sym] = [s for s in self._signals[sym] if s.timestamp >= timestamp]
                total += len(before)
            return total

    def symbols(self) -> List[str]:
        """All symbols in the store."""
        return list(self._signals.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._signals.values())


class SignalCache:
    """LRU cache for computed signal streams.

    Avoids recomputing expensive indicator-based signals
    on the same data.
    """

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, List[Signal]] = {}
        self._access_order: List[str] = []
        self._max_size = max_size

    def _key(self, generator_id: str, df_hash: int) -> str:
        return f"{generator_id}::{df_hash}"

    def get(self, generator_id: str, df_hash: int) -> Optional[List[Signal]]:
        k = self._key(generator_id, df_hash)
        if k in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(k)
            self._access_order.append(k)
            return list(self._cache[k])
        return None

    def put(self, generator_id: str, df_hash: int, signals: List[Signal]) -> None:
        k = self._key(generator_id, df_hash)
        if k in self._cache:
            self._access_order.remove(k)
        elif len(self._cache) >= self._max_size:
            # Evict least recently used
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[k] = list(signals)
        self._access_order.append(k)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)
