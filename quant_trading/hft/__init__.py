"""
quant_trading.hft - High-Frequency Trading Module
=================================================
Adapted from py_polymarket_hft_mm architecture.

This module provides HFT infrastructure patterns:
- CPU affinity control for dedicated cores
- GC (garbage collection) management to eliminate pauses
- BPS (basis points) threshold signal detection
- Real-time order book management
- Async WebSocket client with auto-reconnection
- Automatic hedging system
- Real-time P&L tracking

Note: This is an ARCHITECTURAL REFERENCE implementation.
Not for live trading without exchange connection.

Architecture:
    HFTEngine (main trading loop)
        -> OrderBookManager (order book state)
        -> CLOBClient (WebSocket market data)
        -> HFTUtils (CPU affinity, GC, BPS helpers)
"""

__version__ = "1.0.0"
__author__ = "Quant God Team"

from .hft_engine import HFTEngine, HFTConfig, TradingState, HFTStats
from .orderbook import OrderBookManager, BPSSignal, MarketData, PriceLevel
from .clob_client import CLOBClient, Order, OrderSide, OrderStatus, MarketUpdate
from .hft_utils import (
    set_cpu_affinity,
    disable_gc,
    enable_gc,
    measure_latency,
    calculate_bps,
    calculate_micro_price,
    calculate_micro_vs_mid_bps,
    calculate_spread_bps,
    LatencyTracker,
    PnLTracker,
)

__all__ = [
    # Core engine
    "HFTEngine",
    "HFTConfig",
    "TradingState",
    "HFTStats",
    # Order book
    "OrderBookManager",
    "BPSSignal",
    "MarketData",
    "PriceLevel",
    # WebSocket client
    "CLOBClient",
    "Order",
    "OrderSide",
    "OrderStatus",
    "MarketUpdate",
    # Utilities
    "set_cpu_affinity",
    "disable_gc",
    "enable_gc",
    "measure_latency",
    "calculate_bps",
    "calculate_micro_price",
    "calculate_micro_vs_mid_bps",
    "calculate_spread_bps",
    "LatencyTracker",
    "PnLTracker",
]
