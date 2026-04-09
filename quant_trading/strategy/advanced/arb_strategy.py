"""
Arbitrage Strategy — Triangular arbitrage opportunity detection and execution
============================================================================

Graph-based triangular arbitrage detection using Bellman-Ford style negative
cycle detection on currency graphs.

This strategy detects and evaluates triangular arbitrage opportunities such as:
    USDT -> BTC -> ETH -> USDT

When the cycle profit exceeds min_profit_pct after fees and slippage,
an executable arbitrage opportunity is flagged.

Absorbed from arb_detector.py in this project.

Classes
-------
ArbitrageStrategy
    BaseStrategy adapter for triangular arbitrage trading.
ArbitrageParams
    Strategy parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_trading.signal.types import Signal, SignalType, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext
from quant_trading.strategy.arb_detector import (
    ArbitrageDetector,
    ArbitrageOpportunity,
    TradingPair,
    OrderBook,
    OrderBookLevel,
)


@dataclass
class ArbitrageParams(StrategyParams):
    """Parameters for ArbitrageStrategy."""
    min_profit_pct: float = 0.1
    """Minimum profit percentage after fees/slippage to execute."""
    min_profit_threshold: float = 0.3
    """Gross profit threshold to trigger liquidity check."""
    initial_amount: float = 100.0
    """Initial capital in quote currency (USDT)."""
    max_opportunities: int = 5
    """Maximum number of opportunities to track."""


@dataclass
class ArbitrageSignalMetadata:
    """Metadata for arbitrage signals."""
    opportunity: ArbitrageOpportunity
    cycle: List[str]
    gross_profit_pct: float
    net_profit_pct: float
    is_executable: bool


class ArbitrageStrategy(BaseStrategy):
    """
    Triangular arbitrage strategy using graph-based detection.

    Monitors currency triads for arbitrage opportunities:
      - USDT -> X -> Y -> USDT

    When an opportunity's net profit exceeds min_profit_pct,
    an executable signal is generated.

    Note: This strategy requires real-time order book data and is
    typically used for high-frequency arbitrage trading.

    Parameters
    ----------
    symbol : str
        Primary trading pair symbol (e.g. "BTC/USDT").
    params : ArbitrageParams, optional
        Strategy parameters.
    """

    name = "arbitrage"

    def __init__(
        self,
        symbol: str,
        params: Optional[ArbitrageParams] = None,
    ) -> None:
        super().__init__(symbol, params or ArbitrageParams())
        self._detector = ArbitrageDetector(
            min_profit_pct=Decimal(str(self.params.min_profit_pct)),
            min_profit_threshold=Decimal(str(self.params.min_profit_threshold)),
            initial_amount=Decimal(str(self.params.initial_amount)),
        )
        self._opportunities: List[ArbitrageOpportunity] = []
        self._last_checked: int = 0

    def detect_opportunities(
        self,
        markets: Dict[str, TradingPair],
        tickers: Dict[str, Dict[str, Any]],
        order_books: Optional[Dict[str, OrderBook]] = None,
    ) -> List[ArbitrageOpportunity]:
        """
        Detect triangular arbitrage opportunities.

        Parameters
        ----------
        markets : Dict[str, TradingPair]
            Available trading pairs.
        tickers : Dict[str, Dict[str, Any]]
            Current ticker data with ask/bid prices.
        order_books : Dict[str, OrderBook], optional
            Order books for slippage calculation.

        Returns
        -------
        List[ArbitrageOpportunity]
            Detected opportunities sorted by gross profit.
        """
        return self._detector.find_triangular_opportunities(
            markets=markets,
            tickers=tickers,
            order_books=order_books,
        )

    def evaluate_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        order_books: Dict[str, OrderBook],
        fees: Tuple[Decimal, Decimal, Decimal],
    ) -> ArbitrageOpportunity:
        """
        Evaluate opportunity with fees and slippage.

        Parameters
        ----------
        opportunity : ArbitrageOpportunity
            The opportunity to evaluate.
        order_books : Dict[str, OrderBook]
            Order books for slippage calculation.
        fees : Tuple[Decimal, Decimal, Decimal]
            Trading fees for each leg.

        Returns
        -------
        ArbitrageOpportunity
            Updated opportunity with net profit calculated.
        """
        return self._detector.apply_fees_and_slippage(
            opportunity=opportunity,
            order_books=order_books,
            fees=fees,
        )

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate signals from detected arbitrage opportunities.

        Since arbitrage opportunities depend on external market data
        (multiple tickers, order books), this method returns signals
        based on cached opportunities updated via update_opportunities().

        Parameters
        ----------
        data : pd.DataFrame
            Current price data (used for timestamp).

        Returns
        -------
        List[Signal]
            Trading signals for executable opportunities.
        """
        signals = []

        if not self._opportunities:
            return signals

        for opp in self._opportunities[:self.params.max_opportunities]:
            if not opp.is_executable:
                continue

            # Create signal for the first leg of arbitrage
            last_row = data.iloc[-1]
            timestamp = int(last_row.get("timestamp", 0))

            signals.append(
                Signal(
                    type=SignalType.BUY,  # Represents entering arbitrage
                    symbol=opp.first_symbol,
                    timestamp=timestamp,
                    price=float(opp.first_price),
                    strength=min(1.0, float(opp.net_profit_pct) / 1.0),
                    reason=f"Arbitrage: {opp.first_symbol} -> {opp.second_symbol} -> {opp.third_symbol}, net profit: {opp.net_profit_pct:.4f}%",
                    metadata={
                        "arb_opportunity": True,
                        "cycle": [opp.first_symbol, opp.second_symbol, opp.third_symbol],
                        "gross_profit_pct": float(opp.gross_profit_pct),
                        "net_profit_pct": float(opp.net_profit_pct),
                        "is_executable": opp.is_executable,
                    },
                )
            )

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """
        Calculate position size for arbitrage.

        Arbitrage typically uses fixed capital per leg.
        Size is the initial_amount in quote currency.

        Parameters
        ----------
        signal : Signal
            The arbitrage signal.
        context : StrategyContext
            Current strategy context.

        Returns
        -------
        float
            Position size (base currency amount for first leg).
        """
        return float(self._detector.initial_amount)

    def update_opportunities(
        self,
        markets: Dict[str, TradingPair],
        tickers: Dict[str, Dict[str, Any]],
        order_books: Optional[Dict[str, OrderBook]] = None,
    ) -> None:
        """
        Update cached arbitrage opportunities.

        Call this method periodically with current market data
        to refresh the opportunity cache.

        Parameters
        ----------
        markets : Dict[str, TradingPair]
            Available trading pairs.
        tickers : Dict[str, Dict[str, Any]]
            Current ticker data.
        order_books : Dict[str, OrderBook], optional
            Order books for slippage.
        """
        self._opportunities = self.detect_opportunities(markets, tickers, order_books)

    def get_best_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """
        Get the most profitable executable opportunity.

        Returns
        -------
        ArbitrageOpportunity or None
            Best opportunity if any are executable.
        """
        for opp in self._opportunities:
            if opp.is_executable:
                return opp
        return None

    def get_required_history(self) -> int:
        """Arbitrage doesn't require historical data."""
        return 0

    def to_dict(self) -> dict:
        """Serialize strategy to dict."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "min_profit_pct": self.params.min_profit_pct,
                "min_profit_threshold": self.params.min_profit_threshold,
                "initial_amount": self.params.initial_amount,
                "max_opportunities": self.params.max_opportunities,
            },
            "opportunities_count": len(self._opportunities),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArbitrageStrategy":
        """Deserialize strategy from dict."""
        params = ArbitrageParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
