"""
HMM Regime Strategy — Market regime-aware momentum strategy
=========================================================

Adapts position sizing and signal generation to detected market regimes
(BULL / NEUTRAL / BEAR) using Gaussian HMM on log-returns and volatility.

This is a BaseStrategy adapter for the RegimeAwareStrategy from hmm_regime.py.

Absorbed from:
  - D:/Hive/Data/trading_repos/RegimeSwitchingMomentumStrategy/ (regime_detection.py, hmm_model.py)
  - D:/Hive/Data/trading_repos/AI-Powered-Energy-Algorithmic-Trading-Integrating-Hidden-Markov-Models-with-Neural-Networks/ (alpha.py)

Classes
-------
HMMRegimeStrategy
    BaseStrategy adapter for regime-aware momentum trading.
HMMRegimeParams
    Strategy parameters.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal.types import Signal, SignalType, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext
from quant_trading.strategy.hmm_regime import (
    GaussianHMM,
    MarketRegimeDetector,
    MarketRegimeDetectorConfig,
)


@dataclass
class HMMRegimeParams(StrategyParams):
    """Parameters for HMMRegimeStrategy."""
    momentum_threshold: float = 0.01
    """Return threshold for generating signals within each regime."""
    bull_position_boost: float = 1.2
    """Position size multiplier when in BULL regime."""
    bear_position_scale: float = 0.5
    """Position size multiplier when in BEAR regime."""
    neutral_position_scale: float = 0.8
    """Position size multiplier when in NEUTRAL regime."""
    lookback: int = 60
    """Lookback window for regime detection."""
    return_window: int = 1
    """Window for computing log returns."""
    volatility_window: int = 20
    """Window for computing rolling volatility."""
    n_regimes: int = 3
    """Number of hidden regimes (default 3: bull, neutral, bear)."""
    max_iter: int = 200
    """Max Baum-Welch iterations per fit."""
    tol: float = 1e-4
    """Convergence tolerance."""


class HMMRegimeStrategy(BaseStrategy):
    """
    Market regime-aware momentum strategy using Gaussian HMM.

    Signals and position sizing adapt to detected market regimes:
      - BULL   : confirm momentum with higher conviction; larger positions.
      - BEAR   : require stronger momentum signals; reduced positions.
      - NEUTRAL: use SMA crossover for direction; moderate positions.

    The regime is detected online using a rolling Gaussian HMM fitted to
    recent log-returns and realised volatility.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g. "BTC/USDT").
    params : HMMRegimeParams, optional
        Strategy parameters.
    """

    name = "hmm_regime"

    # Regime constants
    BULL = MarketRegimeDetector.BULL
    NEUTRAL = MarketRegimeDetector.NEUTRAL
    BEAR = MarketRegimeDetector.BEAR

    def __init__(
        self,
        symbol: str,
        params: Optional[HMMRegimeParams] = None,
    ) -> None:
        super().__init__(symbol, params or HMMRegimeParams())
        self._detector: Optional[MarketRegimeDetector] = None
        self._last_regime: Optional[str] = None
        self._last_signal: Optional[int] = None
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize the HMM-based market regime detector."""
        config = MarketRegimeDetectorConfig(
            n_states=self.params.n_regimes,
            lookback=self.params.lookback,
            return_window=self.params.return_window,
            volatility_window=self.params.volatility_window,
            max_iter=self.params.max_iter,
            tol=self.params.tol,
        )
        self._detector = MarketRegimeDetector(config=config, verbose=False)

    def _get_close_prices(self, data: pd.DataFrame) -> np.ndarray:
        """Extract close prices from DataFrame."""
        if "close" in data.columns:
            return data["close"].values
        # Fallback: use last column as price
        return data.iloc[:, -1].values

    def _detect_regime(self, prices: np.ndarray) -> str:
        """Detect current market regime from price series."""
        if len(prices) < self.params.lookback:
            return self.NEUTRAL  # Default to neutral if not enough data

        try:
            win = prices[-self.params.lookback:]
            self._detector.fit(win)
            return self._detector.current_regime(win)
        except Exception:
            return self.NEUTRAL

    def _generate_signal_from_regime(
        self,
        regime: str,
        prices: np.ndarray,
    ) -> int:
        """
        Generate trading signal based on regime and momentum.

        Returns
        -------
        int
            1 (LONG), -1 (SHORT), or 0 (HOLD).
        """
        if regime == self.NEUTRAL:
            regime = self.BEAR  # Default fallback

        win = prices[-self.params.lookback:] if len(prices) >= self.params.lookback else prices
        ret_window = self.params.return_window

        if len(win) < ret_window + 1:
            return 0

        try:
            daily_return = np.log(win[-1] / win[-1 - ret_window])
        except Exception:
            return 0

        sma_20 = win[-20:].mean() if len(win) >= 20 else win.mean()
        sma_50 = win.mean() if len(win) >= 50 else win.mean()

        if regime == self.BULL:
            return 1 if daily_return > self.params.momentum_threshold else 0
        elif regime == self.BEAR:
            return 1 if daily_return > 1.5 * self.params.momentum_threshold else 0
        else:  # NEUTRAL
            if sma_20 > sma_50:
                return 1
            elif sma_20 < sma_50:
                return -1
            return 0

    def _get_regime_multiplier(self, regime: str) -> float:
        """Get position size multiplier for regime."""
        multipliers = {
            self.BULL: self.params.bull_position_boost,
            self.NEUTRAL: self.params.neutral_position_scale,
            self.BEAR: self.params.bear_position_scale,
        }
        return multipliers.get(regime, 1.0)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on regime detection.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with at least 'close' column.

        Returns
        -------
        List[Signal]
            Trading signals.
        """
        signals = []
        prices = self._get_close_prices(data)

        if len(prices) < self.params.lookback:
            return signals

        # Detect regime
        regime = self._detect_regime(prices)

        # Log regime changes
        if regime != self._last_regime:
            # Regime transition signal
            pass
        self._last_regime = regime

        # Generate signal
        signal_value = self._generate_signal_from_regime(regime, prices)
        self._last_signal = signal_value

        if signal_value == 0:
            return signals

        # Create signal
        last_row = data.iloc[-1]
        timestamp = int(last_row.get("timestamp", 0))
        price = float(last_row["close"]) if "close" in data.columns else float(prices[-1])

        if signal_value == 1:
            signal_type = SignalType.BUY
            reason = f"HMM Regime {regime}: momentum LONG"
        else:
            signal_type = SignalType.SELL
            reason = f"HMM Regime {regime}: momentum SHORT"

        signals.append(
            Signal(
                type=signal_type,
                symbol=self.symbol,
                timestamp=timestamp,
                price=price,
                strength=abs(signal_value),
                reason=reason,
                metadata={"regime": regime},
            )
        )

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """
        Calculate position size based on regime.

        Parameters
        ----------
        signal : Signal
            The signal to size.
        context : StrategyContext
            Current strategy context.

        Returns
        -------
        float
            Position size (number of units).
        """
        regime = signal.metadata.get("regime", self.NEUTRAL)
        multiplier = self._get_regime_multiplier(regime)

        # Base position size as fraction of portfolio
        base_size = context.portfolio_value * 0.1  # 10% base
        sized = (base_size * multiplier) / signal.price

        return sized

    def get_required_history(self) -> int:
        """Return required lookback window."""
        return self.params.lookback

    def on_bar(self, bar: pd.Series) -> Optional[Signal]:
        """Process single bar update."""
        if self._data is None:
            return None

        signals = self.generate_signals(self._data)
        return signals[-1] if signals else None

    def to_dict(self) -> dict:
        """Serialize strategy to dict."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "momentum_threshold": self.params.momentum_threshold,
                "bull_position_boost": self.params.bull_position_boost,
                "bear_position_scale": self.params.bear_position_scale,
                "neutral_position_scale": self.params.neutral_position_scale,
                "lookback": self.params.lookback,
                "return_window": self.params.return_window,
                "volatility_window": self.params.volatility_window,
                "n_regimes": self.params.n_regimes,
            },
            "last_regime": self._last_regime,
            "last_signal": self._last_signal,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HMMRegimeStrategy":
        """Deserialize strategy from dict."""
        params = HMMRegimeParams(**data.get("params", {}))
        strategy = cls(symbol=data["symbol"], params=params)
        return strategy
