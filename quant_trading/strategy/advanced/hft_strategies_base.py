"""
HFT Strategies — BaseStrategy Adapters
======================================

High-Frequency Trading Strategies adapted to BaseStrategy interface.

Absorbed from: quant_trading.strategy.hft_strategies

Classes
-------
SpreadCaptureStrategy
    Kalman filter-based spread capture strategy.
MomentumSignalStrategy
    Momentum signal strategy with volume confirmation.
OrderBookImbalanceStrategy
    Order book imbalance (OBI) strategy.
LatencyArbitrageStrategy
    Cross-venue latency arbitrage strategy.

Each strategy inherits from BaseStrategy and implements:
    - generate_signals(data) -> List[Signal]
    - calculate_position_size(signal, context) -> float
    - on_tick(tick) for tick-level signal generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_trading.signal.types import Signal, SignalType, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext
from quant_trading.strategy.hft_strategies import (
    SpreadCaptureStrategy as _HFTSpreadCapture,
    MomentumSignalStrategy as _HTMMomentum,
    OrderBookImbalanceStrategy as _HFTOrderBook,
    LatencyArbitrageStrategy as _HFTLatencyArb,
    SignalType as HFTSignalType,
    HFTOrder,
    Position,
    OrderBook,
)


# ============================================================================
# HFT Strategy Parameters
# ============================================================================

@dataclass
class SpreadCaptureParams(StrategyParams):
    """Parameters for SpreadCaptureStrategy."""
    spread_threshold: float = 0.001
    kalman_covariance: float = 1e-4
    kalman_observation_cov: float = 1.0
    position_size: float = 100.0


@dataclass
class MomentumSignalParams(StrategyParams):
    """Parameters for MomentumSignalStrategy."""
    short_window: int = 5
    long_window: int = 20
    volume_window: int = 10
    momentum_threshold: float = 0.0005
    position_size: float = 100.0


@dataclass
class OrderBookImbalanceParams(StrategyParams):
    """Parameters for OrderBookImbalanceStrategy."""
    imbalance_threshold: float = 0.3
    depth_levels: int = 5
    position_size: float = 100.0


@dataclass
class LatencyArbitrageParams(StrategyParams):
    """Parameters for LatencyArbitrageStrategy."""
    venues: List[str] = field(default_factory=lambda: ["binance", "coinbase"])
    spread_threshold: float = 0.0002
    position_size: float = 100.0
    max_spread_age: float = 0.001


# ============================================================================
# Signal Mapping Helpers
# ============================================================================

def _hft_signal_to_signal(
    hft_signal: HFTSignalType,
    symbol: str,
    price: float,
    timestamp: int,
    reason: str = "",
) -> Optional[Signal]:
    """Map HFT SignalType to BaseStrategy Signal."""
    if hft_signal == HFTSignalType.BUY:
        return Signal(
            type=SignalType.BUY,
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            strength=1.0,
            reason=reason,
        )
    elif hft_signal == HFTSignalType.SELL:
        return Signal(
            type=SignalType.SELL,
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            strength=1.0,
            reason=reason,
        )
    return None


# ============================================================================
# Spread Capture Strategy
# ============================================================================

class SpreadCaptureStrategy(BaseStrategy):
    """
    Kalman filter-based spread capture strategy.

    Uses a Kalman filter to estimate the true mid-price and detects when
    the bid-ask spread deviates from its mean, expecting mean reversion.

    For HFT usage, override on_tick() to process tick-level data directly.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g. "BTC/USDT").
    params : SpreadCaptureParams, optional
        Strategy parameters.
    """

    name = "spread_capture"

    def __init__(
        self,
        symbol: str,
        params: Optional[SpreadCaptureParams] = None,
    ) -> None:
        super().__init__(symbol, params or SpreadCaptureParams())
        self._hft = _HFTSpreadCapture(
            symbol=symbol,
            spread_threshold=self.params.spread_threshold,
            kalman_covariance=self.params.kalman_covariance,
            kalman_observation_cov=self.params.kalman_observation_cov,
            position_size=self.params.position_size,
        )

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate spread capture signals from price data.

        Uses the most recent row's close price as the mid-price reference.
        For real-time HFT, use on_tick() instead.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with at least 'close' column.

        Returns
        -------
        List[Signal]
            Trading signals based on spread deviation.
        """
        signals = []
        if len(data) < 20:
            return signals

        last_row = data.iloc[-1]
        price = float(last_row.get("close", 0))
        timestamp = int(last_row.get("timestamp", 0))

        # Compute spread from bid/ask if available, otherwise estimate
        bid = price * (1 - self.params.spread_threshold / 2)
        ask = price * (1 + self.params.spread_threshold / 2)

        hft_signal = self._hft.compute_signal(bid, ask, price, 0.0)
        signal = _hft_signal_to_signal(
            hft_signal, self.symbol, price, timestamp,
            reason="Spread deviation signal"
        )
        if signal:
            signals.append(signal)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """Calculate position size based on strategy parameter."""
        return self.params.position_size

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Process tick data for real-time HFT signal generation.

        Parameters
        ----------
        tick : Dict[str, Any]
            Tick data with 'bid', 'ask', 'price', 'volume', 'timestamp'.

        Returns
        -------
        Optional[Signal]
            Generated signal or None for HOLD.
        """
        bid = tick.get("bid", 0)
        ask = tick.get("ask", 0)
        price = tick.get("price", 0)
        volume = tick.get("volume", 0)
        timestamp = tick.get("timestamp", 0)

        hft_signal = self._hft.compute_signal(bid, ask, price, volume)
        return _hft_signal_to_signal(
            hft_signal, self.symbol, price, timestamp,
            reason="Spread capture tick signal"
        )

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Handle order fill callback."""
        self._hft.position = Position(
            symbol=self.symbol,
            quantity=order.get("quantity", 0),
            avg_price=order.get("price", 0),
        )

    def get_required_history(self) -> int:
        """Require at least 20 periods for spread analysis."""
        return 20

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "spread_threshold": self.params.spread_threshold,
                "kalman_covariance": self.params.kalman_covariance,
                "kalman_observation_cov": self.params.kalman_observation_cov,
                "position_size": self.params.position_size,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpreadCaptureStrategy":
        params = SpreadCaptureParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)


# ============================================================================
# Momentum Signal Strategy
# ============================================================================

class MomentumSignalStrategy(BaseStrategy):
    """
    Momentum signal strategy with volume confirmation.

    Uses short-term price momentum and volume acceleration to generate
    high-frequency trading signals.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    params : MomentumSignalParams, optional
        Strategy parameters.
    """

    name = "momentum_signal"

    def __init__(
        self,
        symbol: str,
        params: Optional[MomentumSignalParams] = None,
    ) -> None:
        super().__init__(symbol, params or MomentumSignalParams())
        self._hft = _HTMMomentum(
            symbol=symbol,
            short_window=self.params.short_window,
            long_window=self.params.long_window,
            volume_window=self.params.volume_window,
            momentum_threshold=self.params.momentum_threshold,
            position_size=self.params.position_size,
        )

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate momentum signals from OHLCV data.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with 'close' and 'volume' columns.

        Returns
        -------
        List[Signal]
            Trading signals based on momentum and volume.
        """
        signals = []
        if len(data) < self.params.long_window:
            return signals

        last_row = data.iloc[-1]
        price = float(last_row.get("close", 0))
        volume = float(last_row.get("volume", 0))
        timestamp = int(last_row.get("timestamp", 0))

        hft_signal = self._hft.compute_signal(price, volume, timestamp)
        signal = _hft_signal_to_signal(
            hft_signal, self.symbol, price, timestamp,
            reason="Momentum signal"
        )
        if signal:
            signals.append(signal)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """Calculate position size from parameters."""
        return self.params.position_size

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Process tick data for momentum signal generation.

        Parameters
        ----------
        tick : Dict[str, Any]
            Tick data with 'price', 'volume', 'timestamp'.

        Returns
        -------
        Optional[Signal]
            Generated signal or None.
        """
        price = tick.get("price", 0)
        volume = tick.get("volume", 0)
        timestamp = tick.get("timestamp", 0)

        hft_signal = self._hft.compute_signal(price, volume, timestamp)
        return _hft_signal_to_signal(
            hft_signal, self.symbol, price, timestamp,
            reason="Momentum tick signal"
        )

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Handle order fill callback."""
        self._hft.position = Position(
            symbol=self.symbol,
            quantity=order.get("quantity", 0),
            avg_price=order.get("price", 0),
        )

    def get_required_history(self) -> int:
        """Require long_window periods for momentum calculation."""
        return self.params.long_window

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "short_window": self.params.short_window,
                "long_window": self.params.long_window,
                "volume_window": self.params.volume_window,
                "momentum_threshold": self.params.momentum_threshold,
                "position_size": self.params.position_size,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MomentumSignalStrategy":
        params = MomentumSignalParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)


# ============================================================================
# Order Book Imbalance Strategy
# ============================================================================

class OrderBookImbalanceStrategy(BaseStrategy):
    """
    Order book imbalance (OBI) strategy.

    Analyzes the limit order book imbalance to predict short-term
    price direction. When bid volume dominates, price tends to rise.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    params : OrderBookImbalanceParams, optional
        Strategy parameters.
    """

    name = "orderbook_imbalance"

    def __init__(
        self,
        symbol: str,
        params: Optional[OrderBookImbalanceParams] = None,
    ) -> None:
        super().__init__(symbol, params or OrderBookImbalanceParams())
        self._hft = _HFTOrderBook(
            symbol=symbol,
            imbalance_threshold=self.params.imbalance_threshold,
            depth_levels=self.params.depth_levels,
            position_size=self.params.position_size,
        )

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate OBI signals from aggregated order book data.

        Note: For real OBI analysis, use on_tick() with raw order book data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'bid_volume', 'ask_volume' columns.

        Returns
        -------
        List[Signal]
            Trading signals based on order book imbalance.
        """
        signals = []
        if len(data) < 10:
            return signals

        last_row = data.iloc[-1]
        timestamp = int(last_row.get("timestamp", 0))
        mid_price = float(last_row.get("close", 0))

        # Estimate bid/ask volumes from data
        bid_vol = float(last_row.get("bid_volume", last_row.get("buy_volume", 0)))
        ask_vol = float(last_row.get("ask_volume", last_row.get("sell_volume", 0)))

        if bid_vol <= 0 and ask_vol <= 0:
            return signals

        # Construct synthetic order book
        order_book = OrderBook(
            bids=np.array([[mid_price * 0.999, bid_vol]]),
            asks=np.array([[mid_price * 1.001, ask_vol]]),
            timestamp=timestamp,
            venue=self.symbol,
        )

        hft_signal = self._hft.compute_signal(order_book, timestamp)
        signal = _hft_signal_to_signal(
            hft_signal, self.symbol, mid_price, timestamp,
            reason="Order book imbalance signal"
        )
        if signal:
            signals.append(signal)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """Calculate position size from parameters."""
        return self.params.position_size

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Process tick data with order book for OBI signal generation.

        Parameters
        ----------
        tick : Dict[str, Any]
            Tick data with 'bids', 'asks' arrays, 'timestamp'.

        Returns
        -------
        Optional[Signal]
            Generated signal or None.
        """
        bids = tick.get("bids")
        asks = tick.get("asks")
        timestamp = tick.get("timestamp", 0)

        if bids is None or asks is None:
            return None

        order_book = OrderBook(
            bids=np.array(bids),
            asks=np.array(asks),
            timestamp=timestamp,
            venue=self.symbol,
        )

        hft_signal = self._hft.compute_signal(order_book, timestamp)
        mid_price = float(np.mean([bids[0][0], asks[0][0]])) if bids and asks else 0
        return _hft_signal_to_signal(
            hft_signal, self.symbol, mid_price, timestamp,
            reason="OBI tick signal"
        )

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Handle order fill callback."""
        self._hft.position = Position(
            symbol=self.symbol,
            quantity=order.get("quantity", 0),
            avg_price=order.get("price", 0),
        )

    def get_required_history(self) -> int:
        """Require at least 10 periods for OBI moving average."""
        return 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "imbalance_threshold": self.params.imbalance_threshold,
                "depth_levels": self.params.depth_levels,
                "position_size": self.params.position_size,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderBookImbalanceStrategy":
        params = OrderBookImbalanceParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)


# ============================================================================
# Latency Arbitrage Strategy
# ============================================================================

class LatencyArbitrageStrategy(BaseStrategy):
    """
    Cross-venue latency arbitrage strategy.

    Exploits price discrepancies across multiple venues due to
    transmission latencies. When one venue's price deviates from
    another by more than the latency cost, arbitrage the spread.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    params : LatencyArbitrageParams, optional
        Strategy parameters.
    """

    name = "latency_arbitrage"

    def __init__(
        self,
        symbol: str,
        params: Optional[LatencyArbitrageParams] = None,
    ) -> None:
        super().__init__(symbol, params or LatencyArbitrageParams())
        self._hft = _HFTLatencyArb(
            symbol=symbol,
            venues=self.params.venues,
            spread_threshold=self.params.spread_threshold,
            position_size=self.params.position_size,
            max_spread_age=self.params.max_spread_age,
        )
        self._last_venue_prices: Dict[str, Tuple[float, float]] = {}

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate latency arbitrage signals.

        Note: This strategy requires multi-venue price data. For single-venue
        data, this will return no signals. Use update_venue_prices() to
        feed cross-venue data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with venue-specific price columns.

        Returns
        -------
        List[Signal]
            Trading signals based on cross-venue spread.
        """
        signals = []
        if len(self._last_venue_prices) < 2:
            return signals

        last_row = data.iloc[-1]
        timestamp = int(last_row.get("timestamp", 0))

        hft_signal = self._hft.compute_signal(timestamp)
        signal = _hft_signal_to_signal(
            hft_signal, self.symbol, 0, timestamp,
            reason="Latency arbitrage spread signal"
        )
        if signal:
            signals.append(signal)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """Calculate position size from parameters."""
        return self.params.position_size

    def update_venue_prices(
        self,
        venue: str,
        bid: float,
        ask: float,
        timestamp: float,
    ) -> None:
        """
        Update price for a specific venue.

        Call this method for each venue to feed cross-venue data.

        Parameters
        ----------
        venue : str
            Venue/exchange name.
        bid : float
            Bid price.
        ask : float
            Ask price.
        timestamp : float
            Unix timestamp.
        """
        self._last_venue_prices[venue] = (bid, ask)
        self._hft.update_venue_price(venue, (bid + ask) / 2, timestamp)

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Process multi-venue tick data for arbitrage signals.

        Parameters
        ----------
        tick : Dict[str, Any]
            Tick data with 'venue', 'bid', 'ask', 'timestamp'.

        Returns
        -------
        Optional[Signal]
            Generated signal or None.
        """
        venue = tick.get("venue")
        bid = tick.get("bid", 0)
        ask = tick.get("ask", 0)
        timestamp = tick.get("timestamp", 0)

        if venue:
            self.update_venue_prices(venue, bid, ask, timestamp)

        if len(self._last_venue_prices) < 2:
            return None

        hft_signal = self._hft.compute_signal(timestamp)

        # Calculate mid price across venues
        all_prices = [p for vals in self._last_venue_prices.values() for p in vals]
        mid_price = np.mean(all_prices) if all_prices else 0

        return _hft_signal_to_signal(
            hft_signal, self.symbol, mid_price, int(timestamp),
            reason="Latency arbitrage tick signal"
        )

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Handle order fill callback."""
        pass

    def get_required_history(self) -> int:
        """Arbitrage doesn't require historical data."""
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "venues": self.params.venues,
                "spread_threshold": self.params.spread_threshold,
                "position_size": self.params.position_size,
                "max_spread_age": self.params.max_spread_age,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatencyArbitrageStrategy":
        params = LatencyArbitrageParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
