"""
Copula Pairs Strategy — BaseStrategy adapter for copula-based pairs trading
======================================================================

Adapts CopulaPairsStrategy from copula_pairs.py to BaseStrategy interface.

Copula-based pairs trading using Gaussian/Student-t copulas and optional
DCC-GARCH for dynamic correlation modeling.

References:
- Christopher Krauss & Johannes Stübinger:
  "Nonlinear dependence modeling with bivariate copulas: Statistical arbitrage pairs trading on the S&P 100"
- Robert Engle (2002): "Dynamic Conditional Correlation - A Simple Class of Multivariate GARCH Models"

Classes
-------
CopulaPairsStrategyAdapter
    BaseStrategy adapter for copula-based pairs trading.
CopulaPairsStrategyParams
    Strategy parameters (extends CopulaPairsParams).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_trading.signal.types import Signal, SignalType, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext
from quant_trading.strategy.copula_pairs import (
    CopulaPairsStrategy as OriginalCopulaPairsStrategy,
    CopulaPairsParams as OriginalCopulaPairsParams,
)


@dataclass
class CopulaPairsStrategyParams(OriginalCopulaPairsParams):
    """Parameters for CopulaPairsStrategyAdapter.

    Extends CopulaPairsParams with BaseStrategy-required fields.
    """
    symbol1: str = "ASSET1"
    """First asset symbol."""
    symbol2: str = "ASSET2"
    """Second asset symbol."""
    position_size: float = 0.1
    """Base position size as fraction of portfolio."""


class CopulaPairsStrategyAdapter(BaseStrategy):
    """
    BaseStrategy adapter for Copula-based Pairs Trading.

    This strategy monitors two cointegrated assets and generates trading
    signals based on copula-based conditional probability thresholds.

    Signal Logic:
      - H(u|v) >= entry_upper AND H(v|u) <= entry_lower → Short asset1, Long asset2
      - H(u|v) <= entry_lower AND H(v|u) >= entry_upper → Long asset1, Short asset2
      - Exit when both probabilities cross mid-threshold (0.5)

    Parameters
    ----------
    symbol : str
        Primary trading pair symbol (e.g. "PAIR1/PAIR2").
    params : CopulaPairsStrategyParams, optional
        Strategy parameters.
    """

    name = "copula_pairs"

    def __init__(
        self,
        symbol: str,
        params: Optional[CopulaPairsStrategyParams] = None,
    ) -> None:
        super().__init__(symbol, params or CopulaPairsStrategyParams())

        # Extract original params for CopulaPairsStrategy
        original_params = OriginalCopulaPairsParams(
            copula_type=self.params.copula_type,
            entry_upper=self.params.entry_upper,
            entry_lower=self.params.entry_lower,
            exit_upper=self.params.exit_upper,
            exit_lower=self.params.exit_lower,
            use_dcc=self.params.use_dcc,
            min_periods=self.params.min_periods,
            ecdf_margins=self.params.ecdf_margins,
            correlation_threshold=self.params.correlation_threshold,
        )

        self._strategy = OriginalCopulaPairsStrategy(
            symbol1=self.params.symbol1,
            symbol2=self.params.symbol2,
            params=original_params,
        )

        self._fitted: bool = False
        self._last_signal: Tuple[int, int] = (0, 0)  # (signal1, signal2)
        self._returns1_buffer: List[float] = []
        self._returns2_buffer: List[float] = []

    def fit(self, returns1: np.ndarray, returns2: np.ndarray) -> "CopulaPairsStrategyAdapter":
        """
        Fit the copula model on historical returns.

        Parameters
        ----------
        returns1 : np.ndarray
            First asset log returns.
        returns2 : np.ndarray
            Second asset log returns.

        Returns
        -------
        self
        """
        result = self._strategy.fit(returns1, returns2)
        self._fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from copula model.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns for two assets' returns or prices.
            Expected format: ['returns1', 'returns2'] or price columns.

        Returns
        -------
        List[Signal]
            Trading signals for both legs of the pairs trade.
        """
        signals = []

        if not self._fitted and len(data) < self.params.min_periods:
            return signals

        # Extract returns from DataFrame
        returns1, returns2 = self._extract_returns(data)

        if returns1 is None or returns2 is None:
            return signals

        # Fit if not yet fitted
        if not self._fitted:
            self.fit(returns1, returns2)

        # Generate signals from copula model
        try:
            result = self._strategy.generate_signals(returns1, returns2)
            signals1, signals2 = result.signals if hasattr(result, 'signals') else result

            self._last_signal = (signals1[-1] if len(signals1) > 0 else 0,
                                signals2[-1] if len(signals2) > 0 else 0)

        except Exception:
            return signals

        # Convert to Signal objects
        last_row = data.iloc[-1]
        timestamp = int(last_row.get("timestamp", 0))
        price1 = float(last_row.get("close", 0)) if "close" in data.columns else 1.0
        price2 = price1  # Fallback

        # Signal for first leg
        sig1 = self._last_signal[0]
        if sig1 != 0:
            signal_type = SignalType.BUY if sig1 > 0 else SignalType.SELL
            signals.append(
                Signal(
                    type=signal_type,
                    symbol=f"{self.params.symbol1}/{self.params.symbol2}",
                    timestamp=timestamp,
                    price=price1,
                    strength=abs(sig1),
                    reason=f"Copula pairs signal: {self.params.symbol1} {'long' if sig1 > 0 else 'short'}",
                    metadata={"leg": 1, "signal": sig1, "asset": self.params.symbol1},
                )
            )

        # Signal for second leg
        sig2 = self._last_signal[1]
        if sig2 != 0:
            signal_type = SignalType.SELL if sig2 > 0 else SignalType.BUY
            signals.append(
                Signal(
                    type=signal_type,
                    symbol=f"{self.params.symbol2}/{self.params.symbol1}",
                    timestamp=timestamp,
                    price=price2,
                    strength=abs(sig2),
                    reason=f"Copula pairs signal: {self.params.symbol2} {'long' if sig2 > 0 else 'short'}",
                    metadata={"leg": 2, "signal": sig2, "asset": self.params.symbol2},
                )
            )

        return signals

    def _extract_returns(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract returns arrays from DataFrame."""
        # Try to find returns columns
        if "returns1" in data.columns and "returns2" in data.columns:
            return data["returns1"].values, data["returns2"].values

        # Try to compute returns from price columns
        cols = data.columns.tolist()
        if len(cols) >= 4:
            # Assume first two price columns are the pairs
            try:
                prices1 = data.iloc[:, 0].values
                prices2 = data.iloc[:, 1].values
                returns1 = np.diff(np.log(prices1))
                returns2 = np.diff(np.log(prices2))
                return returns1, returns2
            except Exception:
                pass

        return None, None

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """
        Calculate position size for pairs trade.

        Size is scaled by the number of legs (2) to maintain
        dollar-neutral positions.

        Parameters
        ----------
        signal : Signal
            The signal to size.
        context : StrategyContext
            Current strategy context.

        Returns
        -------
        float
            Position size for the signal's leg.
        """
        base_size = context.portfolio_value * self.params.position_size
        # For pairs trading, each leg is half the position (dollar neutral)
        return base_size / (2.0 * signal.price)

    def get_required_history(self) -> int:
        """Return minimum required lookback periods."""
        return self.params.min_periods

    def to_dict(self) -> dict:
        """Serialize strategy to dict."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "copula_type": self.params.copula_type,
                "entry_upper": self.params.entry_upper,
                "entry_lower": self.params.entry_lower,
                "exit_upper": self.params.exit_upper,
                "exit_lower": self.params.exit_lower,
                "use_dcc": self.params.use_dcc,
                "min_periods": self.params.min_periods,
                "ecdf_margins": self.params.ecdf_margins,
                "correlation_threshold": self.params.correlation_threshold,
                "symbol1": self.params.symbol1,
                "symbol2": self.params.symbol2,
                "position_size": self.params.position_size,
            },
            "fitted": self._fitted,
            "last_signal": self._last_signal,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CopulaPairsStrategyAdapter":
        """Deserialize strategy from dict."""
        params = CopulaPairsStrategyParams(**data.get("params", {}))
        strategy = cls(symbol=data["symbol"], params=params)
        if data.get("fitted"):
            # Note: requires refitting with historical data
            pass
        return strategy
