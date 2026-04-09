"""
Statistical Arbitrage Strategy - Adapted from Hummingbot

Pairs trading strategy based on cointegration/z-score of spread.
This is a reference implementation adapted from Hummingbot's StatArb controller.

Key concepts:
- Cointegration: Two assets whose spread is mean-reverting
- Z-Score: (spread - mean) / std_dev of spread
- Hedge Ratio: Beta from linear regression of returns

Signal Logic:
    if z_score > entry_threshold:     # Spread too high
        signal = LONG_DOMINANT, SHORT_HEDGE
    elif z_score < -entry_threshold:  # Spread too low
        signal = SHORT_DOMINANT, LONG_HEDGE
    else:
        signal = NEUTRAL

Reference original: D:/Hive/Data/trading_repos/hummingbot/controllers/generic/stat_arb.py
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from quant_trading.signal import Signal, SignalDirection, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


class PositionSide(Enum):
    """Position side for pairs trading"""
    LONG_DOMINANT_SHORT_HEDGE = 1   # signal = 1
    SHORT_DOMINANT_LONG_HEDGE = -1  # signal = -1
    NEUTRAL = 0


@dataclass
class StatArbParams(StrategyParams):
    """Statistical Arbitrage Strategy Parameters"""
    # Trading pairs
    dominant_pair: str = "SOL-USDT"      # Primary asset to trade
    hedge_pair: str = "POPCAT-USDT"      # Hedging asset

    # Exchange connectors (for multi-exchange pairs trading)
    dominant_connector: str = "binance"
    hedge_connector: str = "binance"

    # Statistical parameters
    interval: str = "1m"                  # Candle interval
    lookback_period: int = 300            # Period for z-score calculation

    # Entry/exit thresholds
    entry_threshold: float = 2.0          # Z-score threshold to enter
    exit_threshold: float = 0.5            # Z-score threshold to exit (closer to 0)

    # Risk management
    take_profit_pct: float = 0.0008        # Per-trade take profit (0.08%)
    tp_global_pct: float = 0.01           # Global take profit (1%)
    sl_global_pct: float = 0.05           # Global stop loss (5%)

    # Position sizing
    min_amount_quote: float = 10.0        # Minimum order size in quote
    max_position_deviation: float = 0.1   # Max position imbalance %
    pos_hedge_ratio: float = 1.0          # Hedge ratio for position sizing

    # For perpetuals
    leverage: int = 1                     # Leverage for perpetual trading
    position_mode: str = "HEDGE"          # "HEDGE" or "ONEWAY"


@dataclass
class StatArbState:
    """Internal state for statistical arbitrage"""
    signal: PositionSide = PositionSide.NEUTRAL
    z_score: float = 0.0
    spread: float = 0.0
    hedge_ratio: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0

    # Position tracking (quote values)
    position_dominant_quote: float = 0.0
    position_hedge_quote: float = 0.0

    # Active orders tracking
    active_orders_dominant: List[Dict] = field(default_factory=list)
    active_orders_hedge: List[Dict] = field(default_factory=list)

    # Pair PnL
    pair_pnl_pct: float = 0.0

    # Historical prices cache
    dominant_prices: List[float] = field(default_factory=list)
    hedge_prices: List[float] = field(default_factory=list)

    # Theoretical positions
    theoretical_dominant_quote: float = 0.0
    theoretical_hedge_quote: float = 0.0


class StatArbStrategy(BaseStrategy):
    """
    Statistical Arbitrage Strategy (Pairs Trading)

    Trades two cointegrated assets when their spread deviates
    from the mean beyond a threshold, expecting reversion.
    """

    name: str = "stat_arb"
    params: StatArbParams

    def __init__(self, symbol: str, params: Optional[StatArbParams] = None):
        # For pairs trading, symbol is the dominant pair
        super().__init__(symbol, params)
        self.params = params or StatArbParams()

        # State for pairs trading
        self._state = StatArbState()

        # Total capital allocated
        self._total_amount_quote: float = 1000.0  # Should be set from context

        # Compute theoretical positions
        self._update_theoretical_positions()

    def _update_theoretical_positions(self):
        """Update theoretical position sizes based on hedge ratio"""
        total = self._total_amount_quote
        ratio = self.params.pos_hedge_ratio
        self._state.theoretical_dominant_quote = total * (1 / (1 + ratio))
        self._state.theoretical_hedge_quote = total * (ratio / (1 + ratio))

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on z-score of spread.

        Args:
            data: DataFrame with columns for dominant and hedge pair prices.
                  Expected columns: 'dominant_close', 'hedge_close' or similar

        Returns:
            List of Signal objects for both legs of the trade
        """
        if len(data) < self.params.lookback_period:
            return []

        # Extract prices
        dominant_prices = data['dominant_close'].values
        hedge_prices = data['hedge_close'].values

        # Calculate spread and z-score
        spread, z_score, hedge_ratio, alpha, beta = self._calculate_spread_and_zscore(
            dominant_prices, hedge_prices
        )

        # Update state
        self._state.spread = spread
        self._state.z_score = z_score
        self._state.hedge_ratio = hedge_ratio
        self._state.alpha = alpha
        self._state.beta = beta

        # Generate signal
        entry_thresh = self.params.entry_threshold
        exit_thresh = self.params.exit_threshold

        if z_score > entry_thresh:
            # Spread too high -> expect reversion downward
            # Long dominant (spread will fall when dominant falls)
            # Short hedge (spread will fall when hedge rises)
            signal = PositionSide.LONG_DOMINANT_SHORT_HEDGE
        elif z_score < -entry_thresh:
            # Spread too low -> expect reversion upward
            # Short dominant (spread will rise when dominant rises)
            # Long hedge (spread will rise when hedge falls)
            signal = PositionSide.SHORT_DOMINANT_LONG_HEDGE
        elif abs(z_score) < exit_thresh:
            # Near mean -> exit
            signal = PositionSide.NEUTRAL
        else:
            # Between thresholds -> maintain current position
            signal = self._state.signal

        self._state.signal = signal

        return self._create_signals(signal, data)

    def _calculate_spread_and_zscore(
        self,
        dominant_prices: np.ndarray,
        hedge_prices: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate spread and z-score using linear regression.

        Returns:
            Tuple of (spread, z_score, hedge_ratio, alpha, beta)
        """
        lookback = self.params.lookback_period

        # Use most recent data points
        dominant = dominant_prices[-lookback:]
        hedge = hedge_prices[-lookback:]

        if len(dominant) < lookback:
            return 0.0, 0.0, 1.0, 0.0, 1.0

        # Calculate percentage returns
        dominant_pct = np.diff(dominant) / dominant[:-1]
        hedge_pct = np.diff(hedge) / hedge[:-1]

        # Cumulative returns (normalize to start at 1)
        dominant_cum = np.cumprod(dominant_pct + 1)
        hedge_cum = np.cumprod(hedge_pct + 1)

        if len(dominant_cum) > 0:
            dominant_cum = dominant_cum / dominant_cum[0]
            hedge_cum = hedge_cum / hedge_cum[0]
        else:
            dominant_cum = np.array([1.0])
            hedge_cum = np.array([1.0])

        # Linear regression: hedge = alpha + beta * dominant
        try:
            dominant_reshaped = dominant_cum.reshape(-1, 1)
            reg = LinearRegression().fit(dominant_reshaped, hedge_cum)
            alpha = reg.intercept_
            beta = reg.coef_[0]
        except Exception:
            alpha = 0.0
            beta = 1.0

        # Calculate spread as percentage difference from predicted
        y_pred = alpha + beta * dominant_cum
        spread_pct = (hedge_cum - y_pred) / y_pred * 100

        # Z-score
        mean_spread = np.mean(spread_pct)
        std_spread = np.std(spread_pct)

        if std_spread == 0:
            return 0.0, 0.0, beta, alpha, beta

        current_spread = spread_pct[-1]
        z_score = (current_spread - mean_spread) / std_spread

        return current_spread, z_score, beta, alpha, beta

    def _create_signals(self, signal: PositionSide, data: pd.DataFrame) -> List[Signal]:
        """Create Signal objects for both legs of the pairs trade"""
        signals = []
        last_row = data.iloc[-1]

        dominant_price = float(last_row.get('dominant_close', 0))
        hedge_price = float(last_row.get('hedge_close', 0))

        if signal == PositionSide.LONG_DOMINANT_SHORT_HEDGE:
            # Long dominant, short hedge
            signals.append(Signal(
                symbol=self.params.dominant_pair,
                direction=SignalDirection.LONG,
                signal_type=SignalType.ENTRY,
                price=dominant_price,
                metadata={
                    'leg': 'dominant',
                    'z_score': self._state.z_score,
                    'hedge_ratio': self._state.hedge_ratio,
                    'side': signal
                }
            ))
            signals.append(Signal(
                symbol=self.params.hedge_pair,
                direction=SignalDirection.SHORT,
                signal_type=SignalType.ENTRY,
                price=hedge_price,
                metadata={
                    'leg': 'hedge',
                    'z_score': self._state.z_score,
                    'hedge_ratio': self._state.hedge_ratio,
                    'side': signal
                }
            ))

        elif signal == PositionSide.SHORT_DOMINANT_LONG_HEDGE:
            # Short dominant, long hedge
            signals.append(Signal(
                symbol=self.params.dominant_pair,
                direction=SignalDirection.SHORT,
                signal_type=SignalType.ENTRY,
                price=dominant_price,
                metadata={
                    'leg': 'dominant',
                    'z_score': self._state.z_score,
                    'hedge_ratio': self._state.hedge_ratio,
                    'side': signal
                }
            ))
            signals.append(Signal(
                symbol=self.params.hedge_pair,
                direction=SignalDirection.LONG,
                signal_type=SignalType.ENTRY,
                price=hedge_price,
                metadata={
                    'leg': 'hedge',
                    'z_score': self._state.z_score,
                    'hedge_ratio': self._state.hedge_ratio,
                    'side': signal
                }
            ))

        elif signal == PositionSide.NEUTRAL:
            # Exit signal for both legs
            if self._state.signal != PositionSide.NEUTRAL:
                signals.append(Signal(
                    symbol=self.params.dominant_pair,
                    direction=SignalDirection.NEUTRAL,
                    signal_type=SignalType.EXIT,
                    price=dominant_price,
                    metadata={
                        'leg': 'dominant',
                        'z_score': self._state.z_score,
                        'reason': 'mean_reversion'
                    }
                ))
                signals.append(Signal(
                    symbol=self.params.hedge_pair,
                    direction=SignalDirection.NEUTRAL,
                    signal_type=SignalType.EXIT,
                    price=hedge_price,
                    metadata={
                        'leg': 'hedge',
                        'z_score': self._state.z_score,
                        'reason': 'mean_reversion'
                    }
                ))

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """
        Calculate position size based on signal and risk parameters.

        For pairs trading:
        - Size based on min_amount_quote and current price
        - Adjusts for leverage
        """
        if signal.direction == SignalDirection.NEUTRAL:
            return 0.0

        # Get the appropriate theoretical position
        if signal.metadata.get('leg') == 'dominant':
            theoretical = self._state.theoretical_dominant_quote
        else:
            theoretical = self._state.theoretical_hedge_quote

        # Minimum order size constraint
        price = signal.price if signal.price else context.current_price
        if price <= 0:
            return 0.0

        amount = theoretical / price

        # Ensure minimum order size
        min_amount = self.params.min_amount_quote / price
        if amount < min_amount:
            amount = min_amount

        # Apply leverage for perpetual trading
        amount = amount * self.params.leverage

        return amount

    def check_take_profit_stop_loss(
        self,
        current_pnl_pct: float
    ) -> Tuple[bool, str]:
        """
        Check if take profit or stop loss is hit.

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        tp = self.params.tp_global_pct
        sl = self.params.sl_global_pct

        if current_pnl_pct >= tp:
            return True, "take_profit"
        elif current_pnl_pct <= -sl:
            return True, "stop_loss"

        return False, ""

    def update_state_from_fills(
        self,
        leg: str,
        filled_amount_quote: float,
        side: SignalDirection
    ) -> None:
        """Update internal state when an order is filled"""
        if leg == 'dominant':
            if side == SignalDirection.LONG:
                self._state.position_dominant_quote += filled_amount_quote
            else:
                self._state.position_dominant_quote -= filled_amount_quote
        else:
            if side == SignalDirection.LONG:
                self._state.position_hedge_quote += filled_amount_quote
            else:
                self._state.position_hedge_quote -= filled_amount_quote

        self._update_pair_pnl()

    def _update_pair_pnl(self) -> None:
        """Update pair PnL percentage"""
        total_d = abs(self._state.position_dominant_quote)
        total_h = abs(self._state.position_hedge_quote)
        total = total_d + total_h

        if total > 0:
            # Simplified PnL calculation
            # In real implementation, would track entry prices
            self._state.pair_pnl_pct = 0.0

    def get_current_signal(self) -> PositionSide:
        """Get current trading signal"""
        return self._state.signal

    def get_z_score(self) -> float:
        """Get current z-score"""
        return self._state.z_score

    def get_spread(self) -> float:
        """Get current spread"""
        return self._state.spread

    def get_hedge_ratio(self) -> float:
        """Get current hedge ratio (beta)"""
        return self._state.hedge_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Serialize strategy state"""
        return {
            'name': self.name,
            'symbol': self.symbol,
            'params': {
                'dominant_pair': self.params.dominant_pair,
                'hedge_pair': self.params.hedge_pair,
                'interval': self.params.interval,
                'lookback_period': self.params.lookback_period,
                'entry_threshold': self.params.entry_threshold,
                'exit_threshold': self.params.exit_threshold,
                'leverage': self.params.leverage,
            },
            'state': {
                'signal': self._state.signal.name,
                'z_score': self._state.z_score,
                'spread': self._state.spread,
                'hedge_ratio': self._state.hedge_ratio,
                'position_dominant_quote': self._state.position_dominant_quote,
                'position_hedge_quote': self._state.position_hedge_quote,
                'pair_pnl_pct': self._state.pair_pnl_pct,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatArbStrategy':
        """Deserialize strategy from dict"""
        params = StatArbParams(**data['params'])
        strategy = cls(symbol=data['symbol'], params=params)

        # Restore state if needed
        if 'state' in data:
            state_data = data['state']
            strategy._state.signal = PositionSide[state_data['signal']]
            strategy._state.z_score = state_data['z_score']
            strategy._state.spread = state_data['spread']
            strategy._state.hedge_ratio = state_data['hedge_ratio']
            strategy._state.position_dominant_quote = state_data['position_dominant_quote']
            strategy._state.position_hedge_quote = state_data['position_hedge_quote']
            strategy._state.pair_pnl_pct = state_data['pair_pnl_pct']

        return strategy


# Convenience function for creating pairs trading strategy
def create_stat_arb_strategy(
    dominant_pair: str,
    hedge_pair: str,
    total_capital: float = 10000.0,
    **kwargs
) -> StatArbStrategy:
    """
    Factory function to create a statistical arbitrage strategy.

    Args:
        dominant_pair: Primary trading pair (e.g., "SOL-USDT")
        hedge_pair: Hedging trading pair (e.g., "POPCAT-USDT")
        total_capital: Total capital to allocate
        **kwargs: Additional StatArbParams

    Returns:
        Configured StatArbStrategy instance
    """
    params = StatArbParams(
        dominant_pair=dominant_pair,
        hedge_pair=hedge_pair,
        **kwargs
    )
    strategy = StatArbStrategy(symbol=dominant_pair, params=params)
    strategy._total_amount_quote = total_capital
    strategy._update_theoretical_positions()
    return strategy
