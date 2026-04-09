"""
quant_trading.hft.hft_engine
=============================
HFT trading loop -- ARCHITECTURAL REFERENCE implementation.

Adapted from py_polymarket_hft_mm main.py trading loop patterns.

This module provides the core HFT engine architecture:
- CPU affinity control (dedicated cores)
- GC management (gc.disable / gc.enable)
- BPS threshold signal detection
- Automatic hedging system
- Real-time P&L tracking
- Async architecture for non-blocking operations

NOTE: This is an ARCHITECTURAL REFERENCE. It is NOT ready for live trading.
It requires integration with a real exchange connection (order submission,
WebSocket market data feed, etc.). Use as a pattern for building a
production HFT system.

Key patterns:
1. gc.disable() before trading loop -- eliminates GC pauses
2. CPU affinity pinning to last 1-2 cores
3. BPS threshold signal detection (micro_vs_mid_bps)
4. Anchor + hedge dual-order placement
5. Thread-safe order book + P&L tracker
6. Market session windows (time-based reset)
7. Graceful shutdown on KeyboardInterrupt
"""

import asyncio
import gc
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Any

from .orderbook import BPSSignal, MarketData, OrderBookManager
from .clob_client import CLOBClient, Order, OrderSide
from .hft_utils import (
    LatencyTracker,
    PnLTracker,
    disable_gc,
    enable_gc,
    set_cpu_affinity,
    calculate_bps,
    calculate_micro_price,
    calculate_micro_vs_mid_bps,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class TradingState(Enum):
    IDLE = "IDLE"
    MONITORING = "MONITORING"
    TRADING = "TRADING"
    HEDGING = "HEDGING"
    STOPPED = "STOPPED"


@dataclass
class HFTConfig:
    """
    HFT Engine configuration parameters.

    Defaults match the py_polymarket_hft_mm config.py values.
    """
    # Signal thresholds
    trading_bps_threshold: float = 50.0      # Min BPS spread to trigger signal
    max_trading_bps_threshold: float = 100.0  # Max BPS spread (avoid volatile markets)

    # Position limits
    max_concurrent_trades: int = 30
    max_inventory: int = 1

    # Timing
    market_session_seconds: int = 900       # 15-minute trading windows
    min_delay_between_trades: float = 1.0  # Seconds between trades
    loop_sleep_seconds: float = 0.01       # Main loop sleep (10ms)

    # Profit
    profit_margin: float = 0.04           # 4% profit margin for hedge price

    # Hedging
    auto_hedge: bool = True                # Place opposite orders automatically

    # Performance
    enable_cpu_affinity: bool = True
    enable_gc_control: bool = True

    # WebSocket
    ws_market_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_user_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    reconnect_delay: float = 0.5
    max_reconnect_retries: int = 5

    def __repr__(self) -> str:
        return (
            f"HFTConfig(bps={self.trading_bps_threshold}/{self.max_trading_bps_threshold}, "
            f"max_trades={self.max_concurrent_trades}, "
            f"session={self.market_session_seconds}s)"
        )


@dataclass
class HFTStats:
    """Real-time HFT engine statistics."""
    state: TradingState = TradingState.IDLE
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0
    last_signal: BPSSignal = BPSSignal.NEUTRAL
    last_update_latency_ms: float = 0.0
    order_book_update_count: int = 0
    reconnect_count: int = 0
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# HFT Engine
# ---------------------------------------------------------------------------


class HFTEngine:
    """
    High-Frequency Trading engine.

    This is an ARCHITECTURAL REFERENCE implementing the core HFT loop pattern:
    1. Initialize: set CPU affinity, disable GC, set up WebSocket
    2. Main loop: read order book, detect signals, place orders
    3. Hedging: automatically place opposite orders to limit risk
    4. P&L tracking: real-time profit/loss monitoring
    5. Session management: time-based market windows
    6. Shutdown: re-enable GC, close connections

    Usage:
        config = HFTConfig(trading_bps_threshold=50)
        engine = HFTEngine(config)
        engine.start()
        # ... engine runs in background thread
        # engine.stop()

    Or with context manager (async):
        async with HFTEngine(config) as engine:
            await engine.run_async()
    """

    def __init__(
        self,
        config: Optional[HFTConfig] = None,
        clob_client: Optional[CLOBClient] = None,
        orderbook: Optional[OrderBookManager] = None,
        ws_market_url: Optional[str] = None,
        ws_user_url: Optional[str] = None,
    ):
        """
        Args:
            config: HFTConfig with trading parameters.
            clob_client: Pre-configured CLOBClient (optional).
            orderbook: Pre-configured OrderBookManager (optional).
            ws_market_url: Override WebSocket market data URL.
            ws_user_url: Override WebSocket user URL.
        """
        self.config = config or HFTConfig()
        if ws_market_url:
            self.config.ws_market_url = ws_market_url
        if ws_user_url:
            self.config.ws_user_url = ws_user_url

        # Core components
        self.orderbook = orderbook or OrderBookManager()
        self.clob_client = clob_client or CLOBClient(
            ws_url=self.config.ws_market_url,
            user_ws_url=self.config.ws_user_url,
            reconnect_delay=self.config.reconnect_delay,
            max_retries=self.config.max_reconnect_retries,
        )

        # P&L and latency tracking
        self.pnl_tracker = PnLTracker()
        self.latency_tracker = LatencyTracker(window=1000)

        # Trading state
        self._state = TradingState.IDLE
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        # Signal state
        self._trade_counter = 0
        self._last_trade_time = 0.0
        self._session_start_time: Optional[float] = None

        # Lock for thread-safe state access
        self._lock = threading.Lock()

        # Callbacks for external systems (e.g., TitanBrain, RiskManager)
        self._on_trade: Optional[Callable[[Order, float], None]] = None
        self._on_signal: Optional[Callable[[BPSSignal, MarketData], None]] = None
        self._on_state_change: Optional[Callable[[TradingState, TradingState], None]] = None

        # Statistics
        self._stats = HFTStats()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> TradingState:
        return self._state

    @property
    def stats(self) -> HFTStats:
        with self._lock:
            return HFTStats(
                state=self._state,
                trade_count=self._trade_counter,
                win_count=len([p for p in self.pnl_tracker._trades if p > 0]),
                loss_count=len([p for p in self.pnl_tracker._trades if p < 0]),
                total_pnl=self.pnl_tracker.total_pnl,
                last_signal=self.orderbook.last_signal,
                order_book_update_count=self.orderbook.update_count,
                uptime_seconds=(
                    time.time() - self._session_start_time
                    if self._session_start_time else 0.0
                ),
            )

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_trade(
        self,
        cb: Callable[[Order, float], None],
    ) -> "HFTEngine":
        """Set callback for trade execution events. Returns self for chaining."""
        self._on_trade = cb
        return self

    def on_signal(
        self,
        cb: Callable[[BPSSignal, MarketData], None],
    ) -> "HFTEngine":
        """Set callback for signal detection events. Returns self for chaining."""
        self._on_signal = cb
        return self

    def on_state_change(
        self,
        cb: Callable[[TradingState, TradingState], None],
    ) -> "HFTEngine":
        """Set callback for state transition events. Returns self for chaining."""
        self._on_state_change = cb
        return self

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the HFT engine in a background thread.

        Initializes:
        - CPU affinity (if enabled)
        - GC disabled (if enabled)
        - WebSocket connection
        - Background monitoring + trading threads
        """
        if self._running:
            logger.warning("HFTEngine already running")
            return

        self._running = True
        self._session_start_time = time.time()

        # Performance initialization
        if self.config.enable_cpu_affinity:
            set_cpu_affinity()

        if self.config.enable_gc_control:
            disable_gc()

        # Set up WebSocket callbacks
        self._setup_client_callbacks()

        # Start WebSocket in background
        self.clob_client.start()

        # Start trading worker thread
        self._worker_thread = threading.Thread(
            target=self._run_trading_loop,
            daemon=True,
            name="HFTEngine-trading",
        )
        self._worker_thread.start()

        self._set_state(TradingState.MONITORING)
        logger.info(f"HFTEngine started: {self.config}")

    def stop(self) -> None:
        """Stop the HFT engine and clean up resources."""
        if not self._running:
            return

        self._running = False
        self._set_state(TradingState.STOPPED)

        # Stop WebSocket
        self.clob_client.stop()

        # Re-enable GC
        if self.config.enable_gc_control:
            enable_gc()

        logger.info(
            f"HFTEngine stopped. Stats: {self.stats.trade_count} trades, "
            f"P&L={self.stats.total_pnl:.4f}"
        )

    # -------------------------------------------------------------------------
    # Async context manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "HFTEngine":
        return self

    async def __aexit__(self, *args) -> None:
        self.stop()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _setup_client_callbacks(self) -> None:
        """Wire up CLOBClient callbacks to update the order book."""

        def handle_book(update):
            if update.bids is not None and update.asks is not None:
                self.orderbook.update_snapshot(update.bids, update.asks)

        def handle_price_change(update):
            if update.price_changes:
                for change in update.price_changes:
                    price = float(change.get("price", 0))
                    size = float(change.get("size", 0))
                    side = change.get("side", "BUY")
                    if price > 0:
                        self.orderbook.apply_incremental_update(price, size, side)

        self.clob_client.on_book_update(handle_book)
        self.clob_client.on_price_change(handle_price_change)
        self.clob_client.on_connect(self._on_connect)
        self.clob_client.on_disconnect(self._on_disconnect)

    def _on_connect(self) -> None:
        logger.info("HFTEngine: CLOBClient connected")
        self._set_state(TradingState.MONITORING)

    def _on_disconnect(self) -> None:
        logger.warning("HFTEngine: CLOBClient disconnected")
        self._set_state(TradingState.IDLE)

    def _set_state(self, new_state: TradingState) -> None:
        old_state = self._state
        with self._lock:
            self._state = new_state
        if old_state != new_state and self._on_state_change:
            self._on_state_change(old_state, new_state)
        logger.debug(f"HFT state: {old_state.value} -> {new_state.value}")

    def _is_in_trading_window(self) -> bool:
        """Check if current time is within the active market session window."""
        if not self._session_start_time:
            return True
        elapsed = time.time() - self._session_start_time
        return elapsed < (self.config.market_session_seconds - 5)

    def _can_trade(self) -> bool:
        """Check if a new trade is allowed under current constraints."""
        now = time.time()
        return (
            self._trade_counter < self.config.max_concurrent_trades
            and self._is_in_trading_window()
            and (now - self._last_trade_time) >= self.config.min_delay_between_trades
        )

    def _get_elapsed_session_seconds(self) -> float:
        """Return seconds elapsed in the current market session."""
        ts = int(time.time())
        period_start = (
            ts // self.config.market_session_seconds
        ) * self.config.market_session_seconds
        return ts - period_start

    # -------------------------------------------------------------------------
    # Core trading logic (called from worker thread)
    # -------------------------------------------------------------------------

    def _run_trading_loop(self) -> None:
        """
        Main trading loop running in background thread.

        Loop structure:
        1. Check market session window (restart if expired)
        2. Get current market data from order book
        3. Validate price range + BPS threshold
        4. Detect signal (UP / DOWN / NEUTRAL)
        5. If signal and can_trade: place anchor + hedge
        6. Sleep briefly and repeat

        GC is disabled for the entire loop duration to eliminate pauses.
        """
        logger.info("HFT trading loop started")

        try:
            while self._running:
                # Check trading window -- restart session if expired
                if not self._is_in_trading_window():
                    self._restart_session()
                    continue

                # Get current market data
                market_data = self.orderbook.get_market_data()
                if market_data is None:
                    time.sleep(0.1)
                    continue

                # Filter: price range and BPS threshold
                up_ask = market_data.best_ask_price
                up_bid = market_data.best_bid_price

                price_ok = (0.2 < up_ask < 0.35) or (0.65 < up_bid < 0.8)
                bps_ok = abs(market_data.micro_vs_mid_bps) <= self.config.max_trading_bps_threshold

                if not (price_ok and bps_ok):
                    time.sleep(self.config.loop_sleep_seconds)
                    continue

                # Detect signal
                signal = self.orderbook.get_signal(
                    threshold_bps=self.config.trading_bps_threshold
                )

                if self._on_signal and signal != BPSSignal.NEUTRAL:
                    self._on_signal(signal, market_data)

                # Execute trade
                if signal != BPSSignal.NEUTRAL and self._can_trade():
                    self._execute_trade(signal, market_data)

                time.sleep(self.config.loop_sleep_seconds)

        except Exception as e:
            logger.error(f"Fatal error in HFT trading loop: {e}", exc_info=True)
        finally:
            self._running = False
            self._set_state(TradingState.STOPPED)
            if self.config.enable_gc_control:
                enable_gc()
            logger.info("HFT trading loop exited")

    def _execute_trade(self, signal: BPSSignal, market_data: MarketData) -> None:
        """
        Execute a market-making trade: anchor order + automatic hedge.

        Pattern from py_polymarket_hft_mm:
        1. Place anchor order on the signal side at bid price
        2. Place hedge order on the opposite side at (1 - price - profit_margin)
        3. Both orders submitted concurrently via ThreadPoolExecutor
        4. Update trade counter and last trade time
        """
        self._set_state(TradingState.TRADING)

        side_str = "UP" if signal == BPSSignal.UP else "DOWN"

        if signal == BPSSignal.UP:
            anchor_price = round(market_data.best_bid_price, self.orderbook.price_precision)
            hedge_price = round(
                1 - market_data.best_bid_price - self.config.profit_margin,
                self.orderbook.price_precision,
            )
        else:  # DOWN
            hedge_price = round(market_data.best_ask_price, self.orderbook.price_precision)
            anchor_price = round(
                1 - market_data.best_ask_price - self.config.profit_margin,
                self.orderbook.price_precision,
            )

        # Place anchor order
        with LatencyTracker().measure() as ctx:
            anchor_order_id = self._place_order(
                token_id="ANCHOR_TOKEN",  # TODO: wire to real token ID
                price=anchor_price,
                size=5,  # TODO: wire to position sizer
                side=OrderSide.BUY if side_str == "UP" else OrderSide.SELL,
            )

        self.latency_tracker.record(ctx.tracker.stats()["mean_ms"])

        # Auto-hedge
        if self.config.auto_hedge:
            self._set_state(TradingState.HEDGING)
            hedge_order_id = self._place_order(
                token_id="HEDGE_TOKEN",  # TODO: wire to real token ID
                price=hedge_price,
                size=5,
                side=OrderSide.SELL if side_str == "UP" else OrderSide.BUY,
            )

            logger.info(
                f"[HFT] Trade executed: {side_str} anchor={anchor_order_id} "
                f"hedge={hedge_order_id} @ {anchor_price}/{hedge_price}"
            )
        else:
            logger.info(
                f"[HFT] Trade executed: {side_str} anchor={anchor_order_id} @ {anchor_price}"
            )

        # Update state
        self._trade_counter += 1
        self._last_trade_time = time.time()
        self._set_state(TradingState.MONITORING)

        # Record P&L (simplified -- wire to real fill data)
        pnl = self._estimate_trade_pnl(anchor_price, hedge_price)
        self.pnl_tracker.add_trade(pnl)

    def _place_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: OrderSide,
    ) -> Optional[str]:
        """
        Place a limit order via the CLOB client.

        This is the INTERFACE method -- it must be connected to your
        exchange's order submission API.

        Args:
            token_id: Token/asset ID.
            price: Limit price.
            size: Order size.
            side: BUY or SELL.

        Returns:
            Exchange order ID, or None on failure.
        """
        try:
            return self.clob_client.submit_order(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
            )
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def _estimate_trade_pnl(
        self,
        anchor_price: float,
        hedge_price: float,
    ) -> float:
        """
        Estimate P&L for a completed anchor+hedge round-trip.

        Simplified model: profit = hedge_price - (1 - anchor_price).
        Wire this to real fill data (Order.filled_size, Order.status) for
        accurate P&L tracking.

        Args:
            anchor_price: Fill price of the anchor order.
            hedge_price: Fill price of the hedge order.

        Returns:
            Estimated profit from the round-trip.
        """
        # Model: we bought anchor at anchor_price, hedged by selling hedge at hedge_price
        # Net settlement value should be ~1.0 (binary outcome)
        # Profit ≈ hedge_price - (1 - anchor_price) if prices are within [0, 1]
        settlement = 1.0
        cost_basis = settlement - anchor_price  # Cost to go long at anchor price
        hedge_proceeds = hedge_price             # Proceeds from short hedge
        return hedge_proceeds - cost_basis

    def _restart_session(self) -> None:
        """
        Restart the trading session: reset state, GC collect, reconnect.

        Called when the market session window expires (every MARKET_SESSION_SECONDS).
        """
        logger.info(
            f"Market session ended after {self._get_elapsed_session_seconds():.0f}s. "
            "Restarting session..."
        )

        self._set_state(TradingState.IDLE)
        self._trade_counter = 0
        self._session_start_time = time.time()

        # Re-enable GC for cleanup during idle
        if self.config.enable_gc_control:
            enable_gc()
            gc.collect()
            disable_gc()

        # Clear order book
        self.orderbook.clear()

        # Restart WebSocket (reconnect)
        self.clob_client.stop()
        time.sleep(0.5)
        self.clob_client.start()

        self._set_state(TradingState.MONITORING)
        logger.info("Session restart complete")

    # -------------------------------------------------------------------------
    # Public utilities
    # -------------------------------------------------------------------------

    def get_pnl_report(self) -> Dict[str, Any]:
        """Return a full P&L and performance report."""
        return {
            "pnl": self.pnl_tracker.stats(),
            "latency": self.latency_tracker.stats(),
            "stats": self.stats.__dict__,
            "config": {
                "bps_threshold": self.config.trading_bps_threshold,
                "max_trades": self.config.max_concurrent_trades,
                "profit_margin": self.config.profit_margin,
                "auto_hedge": self.config.auto_hedge,
            },
        }

    def __repr__(self) -> str:
        return (
            f"HFTEngine(state={self._state.value}, "
            f"trades={self._trade_counter}, "
            f"pnl={self.pnl_tracker.total_pnl:.4f})"
        )
