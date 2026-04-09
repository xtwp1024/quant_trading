"""
quant_trading.hft.hft_utils
============================
HFT utility helpers: CPU affinity, GC control, latency measurement, BPS calculation.

Key features from py_polymarket_hft_mm:
- CPU affinity for dedicated cores (reduces context switching)
- GC.disable() / GC.enable() to eliminate GC pauses during trading
- Latency measurement for performance profiling
- BPS (basis points) spread calculation for signal detection
- Micro-price weighted midpoint calculation
"""

import gc
import os
import time
import logging
from typing import Callable, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CPU Affinity
# ---------------------------------------------------------------------------


def set_cpu_affinity() -> Optional[List[int]]:
    """
    Pin the current process to dedicated CPU cores for low-latency trading.

    Uses the last 1-2 cores (leaving headroom for OS/interrupts).
    Falls back gracefully if cpu_affinity is not available on the platform.

    Returns:
        List of core indices assigned, or None if not supported.

    Raises:
        No exceptions -- all failures are logged as warnings.
    """
    try:
        import psutil

        process = psutil.Process()
        cpu_count = os.cpu_count() or 1
        logger.info(f"System has {cpu_count} CPU cores")

        if cpu_count >= 4:
            affinity_cores = [cpu_count - 2, cpu_count - 1]
        elif cpu_count >= 2:
            affinity_cores = [cpu_count - 1]
        else:
            affinity_cores = [0]

        process.cpu_affinity(affinity_cores)
        logger.info(f"CPU affinity set to cores: {affinity_cores}")

        # Set high process priority
        if os.name == "nt":  # Windows
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            logger.info("Process priority set to HIGH (Windows)")
        else:  # Linux/Unix
            process.nice(-10)  # Lower nice = higher priority
            logger.info("Process nice value set to -10 (high priority)")

        return affinity_cores

    except ImportError:
        logger.warning(
            "psutil not installed; CPU affinity not available. "
            "Install with: pip install psutil"
        )
    except Exception as e:
        logger.warning(f"Failed to set CPU affinity: {e}")

    return None


# ---------------------------------------------------------------------------
# Garbage Collection Control
# ---------------------------------------------------------------------------

_gc_was_enabled: bool = True


def disable_gc() -> None:
    """
    Disable Python's garbage collector to eliminate GC pauses during trading.

    This is critical for HFT -- even a minor GC pause can cause missed
    opportunities or order staleness.

    Call once before the trading loop starts. Pair with enable_gc() when done.

    Note:
        Objects allocated during trading will not be freed until GC is
        re-enabled. Monitor memory usage in long sessions.
    """
    global _gc_was_enabled
    _gc_was_enabled = gc.is_enabled()
    gc.disable()
    logger.info("Garbage collector DISABLED (HFT mode)")


def enable_gc() -> None:
    """
    Re-enable the garbage collector and optionally run a full collection.

    Call when the trading loop exits or during quiet/idle periods.
    A full gc.collect() is run to clean up any accumulated objects.

    Note:
        This may cause a brief pause -- only call during non-trading periods.
    """
    global _gc_was_enabled
    if not gc.is_enabled():
        gc.enable()
        logger.info("Garbage collector ENABLED")
        # Clean up accumulated objects from trading session
        gc.collect()
        logger.info("GC collection completed")
    _gc_was_enabled = True


def is_gc_enabled() -> bool:
    """Return True if the garbage collector is currently enabled."""
    return gc.is_enabled()


# ---------------------------------------------------------------------------
# Latency Measurement
# ---------------------------------------------------------------------------


class LatencyTracker:
    """
    Rolling window latency tracker for HFT performance profiling.

    Usage:
        tracker = LatencyTracker(window=1000)
        with tracker.measure():
            ...  # code to profile
        print(tracker.stats())  # mean, p50, p99 latencies
    """

    def __init__(self, window: int = 1000):
        self.window = window
        self._latencies: List[float] = []
        self._timestamps: List[float] = []

    def record(self, latency_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        self._latencies.append(latency_ms)
        self._timestamps.append(time.perf_counter())
        if len(self._latencies) > self.window:
            self._latencies.pop(0)
            self._timestamps.pop(0)

    def measure(self) -> "LatencyContext":
        """Context manager that records the duration of a code block."""
        return LatencyContext(self)

    def stats(self) -> dict:
        """Return latency statistics."""
        if not self._latencies:
            return {"count": 0, "mean_ms": 0, "p50_ms": 0, "p99_ms": 0, "max_ms": 0}
        sorted_lat = sorted(self._latencies)
        return {
            "count": len(sorted_lat),
            "mean_ms": round(sum(sorted_lat) / len(sorted_lat), 4),
            "p50_ms": round(sorted_lat[len(sorted_lat) // 2], 4),
            "p99_ms": round(sorted_lat[int(len(sorted_lat) * 0.99)], 4),
            "max_ms": round(sorted_lat[-1], 4),
        }


class LatencyContext:
    """Context manager returned by LatencyTracker.measure()."""

    def __init__(self, tracker: LatencyTracker):
        self.tracker = tracker
        self._start: float = 0.0

    def __enter__(self) -> "LatencyContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000  # ms
        self.tracker.record(elapsed)


def measure_latency(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that measures and logs the execution time of a function.

    Usage:
        @measure_latency
        def my_trading_function():
            ...
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed_ms:.3f} ms")
        return result
    return wrapper


# ---------------------------------------------------------------------------
# BPS (Basis Points) Calculation
# ---------------------------------------------------------------------------


def calculate_bps(price_a: float, price_b: float) -> float:
    """
    Calculate the spread between two prices in basis points.

    1 BPS = 0.01% = 0.0001 in decimal terms.

    Args:
        price_a: First price (e.g., best bid)
        price_b: Second price (e.g., best ask)

    Returns:
        Spread in basis points (e.g., 50 = 50 BPS = 0.50%)

    Formula:
        bps = (|price_a - price_b| / price_b) * 10000
    """
    if price_b == 0:
        return 0.0
    return abs(price_a - price_b) / price_b * 10000


def calculate_spread_bps(best_bid: float, best_ask: float) -> float:
    """
    Calculate bid-ask spread in BPS.

    Args:
        best_bid: Best bid price
        best_ask: Best ask price

    Returns:
        Bid-ask spread in BPS
    """
    if best_ask == 0:
        return 0.0
    return (best_ask - best_bid) / best_ask * 10000


def calculate_micro_price(
    best_bid: float,
    best_bid_vol: float,
    best_ask: float,
    best_ask_vol: float,
) -> float:
    """
    Calculate the micro-price (volume-weighted mid).

    Micro-price adjusts the mid-point by the volume imbalance:
        micro_price = (bid_price * ask_vol + ask_price * bid_vol) / (bid_vol + ask_vol)

    This gives a better "fair price" estimate than simple mid when
    there is asymmetric order book depth.

    Args:
        best_bid: Best bid price
        best_bid_vol: Volume at best bid
        best_ask: Best ask price
        best_ask_vol: Volume at best ask

    Returns:
        Micro-price (volume-weighted fair price estimate)
    """
    total_vol = best_bid_vol + best_ask_vol
    if total_vol == 0:
        return (best_bid + best_ask) / 2
    return (best_bid * best_ask_vol + best_ask * best_bid_vol) / total_vol


def calculate_micro_vs_mid_bps(
    best_bid: float,
    best_bid_vol: float,
    best_ask: float,
    best_ask_vol: float,
) -> float:
    """
    Calculate micro-price deviation from mid in BPS.

    This is the primary signal metric used by the HFT engine.
    Values above +TRADING_BPS_THRESHOLD suggest UP signal.
    Values below -TRADING_BPS_THRESHOLD suggest DOWN signal.

    Args:
        best_bid: Best bid price
        best_bid_vol: Volume at best bid
        best_ask: Best ask price
        best_ask_vol: Volume at best ask

    Returns:
        Micro-price deviation from mid in BPS.
        Positive = micro > mid (bid pressure), Negative = micro < mid (ask pressure).
    """
    micro_price = calculate_micro_price(best_bid, best_bid_vol, best_ask, best_ask_vol)
    mid_price = (best_bid + best_ask) / 2
    return (micro_price - mid_price) * 10000


# ---------------------------------------------------------------------------
# P&L Tracking
# ---------------------------------------------------------------------------


class PnLTracker:
    """
    Simple real-time P&L tracker for HFT sessions.

    Tracks:
    - Total realized P&L
    - Trade count
    - Win rate
    - Average profit per trade
    """

    def __init__(self):
        self._trades: List[float] = []  # signed P&L per trade
        self._reset()

    def _reset(self) -> None:
        self._trades.clear()

    def add_trade(self, pnl: float) -> None:
        """Record a completed trade's P&L (positive = profit, negative = loss)."""
        self._trades.append(pnl)

    @property
    def total_pnl(self) -> float:
        return sum(self._trades)

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    @property
    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        wins = sum(1 for p in self._trades if p > 0)
        return wins / len(self._trades)

    @property
    def avg_profit(self) -> float:
        if not self._trades:
            return 0.0
        return sum(self._trades) / len(self._trades)

    def stats(self) -> dict:
        return {
            "total_pnl": round(self.total_pnl, 4),
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 4),
            "avg_profit": round(self.avg_profit, 4),
            "best_trade": round(max(self._trades), 4) if self._trades else 0.0,
            "worst_trade": round(min(self._trades), 4) if self._trades else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"PnLTracker(total={self.total_pnl:.4f}, "
            f"trades={self.trade_count}, "
            f"win_rate={self.win_rate:.2%})"
        )
