"""
Freqtrade Strategy Wrapper
===========================

Absorbs freqtrade production crypto trading bot patterns into the quant_trading system.
Provides freqtrade-style strategy interfaces backed by existing connectors.

Patterns absorbed from freqtrade:
- IStrategy interface with populate_indicators/populate_entry_trend/populate_exit_trend
- Edge position sizing (win rate, risk-reward ratio, capital)
- Dynamic whitelist management
- ROI / stoploss / trailing stop
- Telegram integration hooks
- Multiple strategy templates (EMASpread, MACD, PriceBand)

This module does NOT import freqtrade itself (HAS_FREQTRADE=False).
It provides the same interfaces using pure Python with existing connectors as backend.

Classes:
    FreqtradeStrategy   : Base class for user-defined freqtrade-style strategies
    EdgePositionSizer   : Edge-based position sizing calculator
    EMASpreadStrategy    : EMA crossover spread strategy
    MACDStrategy        : MACD-based trend strategy
    PriceBandStrategy    : Price band breakout strategy
    FreqtradeExecutor    : Execution adapter using existing connectors
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

# No freqtrade imports — pure Python wrapper
HAS_FREQTRADE = False

if TYPE_CHECKING:
    from quant_trading.connectors.base_connector import BaseConnector

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Dataclasses
# =============================================================================

class SignalDirection(Enum):
    """Signal direction enum matching freqtrade."""
    LONG = "long"
    SHORT = "short"
    EXIT = "exit"
    NONE = "none"


class EntryExit(Enum):
    """Entry/exit signal type matching freqtrade."""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"


@dataclass
class ROI:
    """Minimal ROI table — maps duration (minutes) to profit ratio.

    Example: {0: 0.05, 30: 0.03, 60: 0.01}
    Means: 5% profit after 0min, 3% after 30min, 1% after 60min.
    """
    table: Dict[int, float] = field(default_factory=dict)

    def get_roi_at(self, minutes: int) -> float:
        """Get the ROI threshold active at the given age (minutes)."""
        if not self.table:
            return 0.0
        thresholds = sorted(self.table.keys(), reverse=True)
        for threshold in thresholds:
            if minutes >= threshold:
                return self.table[threshold]
        return 0.0


@dataclass
class StrategyConfig:
    """Shared strategy configuration matching freqtrade IStrategy attributes."""
    # Interface version
    INTERFACE_VERSION: int = 3

    # Position sizing
    minimal_roi: ROI = field(default_factory=ROI)
    stoploss: float = -0.10
    max_open_trades: int = -1  # -1 = unlimited

    # Trailing stop
    trailing_stop: bool = False
    trailing_stop_positive: Optional[float] = None
    trailing_stop_positive_offset: float = 0.0
    trailing_only_offset_is_reached: bool = False

    # Shorting
    can_short: bool = False

    # Timeframe
    timeframe: str = "1h"

    # Order types
    order_types: Dict[str, str] = field(default_factory=lambda: {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
    })
    order_time_in_force: Dict[str, str] = field(default_factory=lambda: {
        "entry": "GTC",
        "exit": "GTC",
    })

    # Processing
    process_only_new_candles: bool = True
    use_exit_signal: bool = True
    exit_profit_only: bool = False
    exit_profit_offset: float = 0.0
    ignore_roi_if_entry_signal: bool = False

    # Position adjustment
    position_adjustment_enable: bool = False
    max_entry_position_adjustment: int = -1

    # Startup
    startup_candle_count: int = 0
    ignore_buying_expired_candle_after: int = 0

    # Protections
    protections: List[Any] = field(default_factory=list)

    # Plotting
    plot_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSignal:
    """A single trade signal with entry/exit metadata."""
    pair: str
    direction: SignalDirection
    price: Optional[float] = None
    confidence: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    enter_tag: Optional[str] = None
    exit_tag: Optional[str] = None
    timeframe: str = "1h"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TradeSignal(pair={self.pair}, direction={self.direction.value}, "
            f"price={self.price}, confidence={self.confidence:.2f})"
        )


@dataclass
class Position:
    """Active trade position — mirrors freqtrade Trade persistence object."""
    id: str
    pair: str
    direction: SignalDirection
    entry_price: float
    current_price: float = 0.0
    stake_amount: float = 0.0
    amount: float = 0.0
    leverage: float = 1.0
    open_date: datetime = field(default_factory=datetime.now)
    entry_order_id: str = ""
    exit_order_id: Optional[str] = None
    realized_profit: float = 0.0
    is_open: bool = True
    enter_tag: Optional[str] = None
    exit_reason: Optional[str] = None
    stop_loss: Optional[float] = None
    # DCA tracking
    entries: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def profit_ratio(self) -> float:
        """Current profit as a ratio (e.g., 0.05 = 5%)."""
        if self.entry_price == 0:
            return 0.0
        if self.direction == SignalDirection.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

    @property
    def age_minutes(self) -> float:
        """Age of position in minutes."""
        return (datetime.now() - self.open_date).total_seconds() / 60.0


# =============================================================================
# FreqtradeStrategy — Base class for user strategies
# =============================================================================

class FreqtradeStrategy(ABC):
    """
    Freqtrade-style strategy base class.

    User strategies inherit from this and implement:
        - populate_indicators()   : Add technical indicators to DataFrame
        - populate_entry_trend()  : Set enter_long / enter_short columns
        - populate_exit_trend()  : Set exit_long / exit_short columns

    Attrs:
        strategy_config: StrategyConfig with all IStrategy-style settings

    Example:
        class MyStrategy(FreqtradeStrategy):
            strategy_config = StrategyConfig(
                stoploss=-0.05,
                minimal_roi=ROI({0: 0.04, 30: 0.02}),
                timeframe="15m",
            )

            def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
                dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
                dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
                return dataframe

            def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
                dataframe.loc[dataframe["ema20"] > dataframe["ema50"], "enter_long"] = 1
                return dataframe
    """

    # Subclasses should override this
    strategy_config: StrategyConfig = field(default_factory=StrategyConfig)

    _dataprovider: Optional["DataProvider"] = None  # type: ignore
    _config: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # IStrategy-compatible properties (delegated from strategy_config)
    # ------------------------------------------------------------------

    @property
    def minimal_roi(self) -> Dict[int, float]:
        """Minimal ROI table (int minutes -> float ratio)."""
        return self.strategy_config.minimal_roi.table

    @property
    def stoploss(self) -> float:
        """Stoploss as a negative ratio (e.g., -0.10 = 10% stop)."""
        return self.strategy_config.stoploss

    @property
    def timeframe(self) -> str:
        """Trading timeframe string (e.g., '1m', '5m', '1h', '4h', '1d')."""
        return self.strategy_config.timeframe

    @property
    def can_short(self) -> bool:
        return self.strategy_config.can_short

    @property
    def max_open_trades(self) -> int:
        return self.strategy_config.max_open_trades

    @property
    def trailing_stop(self) -> bool:
        return self.strategy_config.trailing_stop

    @property
    def use_exit_signal(self) -> bool:
        return self.strategy_config.use_exit_signal

    @property
    def position_adjustment_enable(self) -> bool:
        return self.strategy_config.position_adjustment_enable

    @property
    def process_only_new_candles(self) -> bool:
        return self.strategy_config.process_only_new_candles

    @property
    def startup_candle_count(self) -> int:
        return self.strategy_config.startup_candle_count

    # ------------------------------------------------------------------
    # DataProvider (placeholder for compatibility)
    # ------------------------------------------------------------------

    @property
    def dp(self):
        """DataProvider placeholder — compatible with freqtrade's dp."""
        return self._dataprovider

    def set_dataprovider(self, dp: Any) -> None:
        """Set the DataProvider (called by executor)."""
        self._dataprovider = dp

    # ------------------------------------------------------------------
    # Abstract methods — implement in subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Populate technical indicators for the strategy.

        Args:
            dataframe: DataFrame with OHLCV columns (open, high, low, close, volume)
            metadata: Dict with 'pair' and optionally 'timeframe'

        Returns:
            DataFrame with additional indicator columns added
        """
        ...

    @abstractmethod
    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Populate entry signal columns.

        Expected columns added to DataFrame:
            enter_long  : 1 when entering long position
            enter_short : 1 when entering short position (if can_short)
            enter_tag   : Optional string tag for the entry reason

        Args:
            dataframe: DataFrame with indicators
            metadata: Dict with 'pair' and optionally 'timeframe'

        Returns:
            DataFrame with entry columns populated
        """
        ...

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Populate exit signal columns (optional override).

        Expected columns:
            exit_long  : 1 when exiting long position
            exit_short : 1 when exiting short position

        Args:
            dataframe: DataFrame with indicators
            metadata: Dict with 'pair' and optionally 'timeframe'

        Returns:
            DataFrame with exit columns populated
        """
        return dataframe

    # ------------------------------------------------------------------
    # Optional hooks — override in subclasses for advanced behavior
    # ------------------------------------------------------------------

    def bot_start(self, **kwargs) -> None:
        """Called once after bot instantiation (freqtrade hook)."""
        pass

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """Called at the start of each bot iteration (freqtrade hook)."""
        pass

    def check_entry_timeout(
        self, pair: str, trade: Position, order: Any, current_time: datetime, **kwargs
    ) -> bool:
        """Return True to cancel unfilled entry order (default: False = keep)."""
        return False

    def check_exit_timeout(
        self, pair: str, trade: Position, order: Any, current_time: datetime, **kwargs
    ) -> bool:
        """Return True to cancel unfilled exit order (default: False = keep)."""
        return False

    def adjust_trade_position(
        self,
        trade: Position,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        **kwargs,
    ) -> Optional[float]:
        """
        DCA / position adjustment hook.

        Return None        : skip adjustment
        Return positive    : increase position by that stake amount
        Return negative    : decrease position (partial close)

        Args:
            trade: Active Position object
            current_time: Current datetime
            current_rate: Current market price
            current_profit: Current profit as ratio (e.g., 0.05 = 5%)
            min_stake: Minimum allowed stake from exchange
            max_stake: Maximum allowed stake from exchange

        Returns:
            Optional stake amount to adjust
        """
        return None

    def custom_stoploss(
        self,
        trade: Position,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Custom stoploss hook. Return new stoploss distance as ratio below current_rate.

        The returned value is the distance (e.g., -0.05 = 5% below current rate).
        Cannot be above self.stoploss (hard cap).
        """
        return self.stoploss

    def custom_roi(
        self,
        trade: Position,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """
        Custom ROI hook. Return a custom ROI ratio or None to fall back to minimal_roi.
        """
        return None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Return the leverage to use for the trade (default: 1.0 = no leverage).
        Override to implement leverage strategies.
        """
        return 1.0

    # ------------------------------------------------------------------
    # Signal helpers — used by executor
    # ------------------------------------------------------------------

    def get_entry_signal(
        self, pair: str, dataframe: pd.DataFrame
    ) -> Tuple[bool, bool]:
        """
        Get entry signal for a pair from the DataFrame.

        Returns:
            Tuple (enter_long, enter_short)
        """
        if dataframe is None or dataframe.empty:
            return False, False
        enter_long = bool(dataframe.get("enter_long", pd.Series([0])).iloc[-1])
        enter_short = bool(dataframe.get("enter_short", pd.Series([0])).iloc[-1])
        return enter_long, enter_short

    def get_exit_signal(
        self, pair: str, dataframe: pd.DataFrame
    ) -> Tuple[bool, bool]:
        """
        Get exit signal for a pair from the DataFrame.

        Returns:
            Tuple (exit_long, exit_short)
        """
        if dataframe is None or dataframe.empty:
            return False, False
        exit_long = bool(dataframe.get("exit_long", pd.Series([0])).iloc[-1])
        exit_short = bool(dataframe.get("exit_short", pd.Series([0])).iloc[-1])
        return exit_long, exit_short


# =============================================================================
# EdgePositionSizer — Edge-based position sizing
# =============================================================================

class EdgePositionSizer:
    """
    Edge-based position sizing calculator.

    Absorbs freqtrade's Edge position sizing logic:
    - Calculates position size from win rate, risk-reward ratio, and capital
    - Uses the Kelly Criterion-inspired sizing formula

    The core formula:
        position_size = (capital * KellyFraction) / entry_price
        KellyFraction = win_rate - (1 - win_rate) / risk_reward_ratio

    Args:
        capital          : Total available capital in stake currency
        win_rate         : Historical win rate (0.0 to 1.0)
        risk_reward_ratio: Average profit / average loss ratio
        max_stake_pct    : Maximum % of capital to risk per trade (default 0.02 = 2%)
        Kelly_reduction  : Kelly reduction factor for safety (default 0.5 = half Kelly)
        min_stake       : Minimum absolute stake amount
        max_stake       : Maximum absolute stake amount

    Example:
        sizer = EdgePositionSizer(
            capital=10000,
            win_rate=0.55,
            risk_reward_ratio=1.5,
            max_stake_pct=0.02,
        )
        stake = sizer.calculate_stake(entry_price=50000, stoploss=0.05)
    """

    def __init__(
        self,
        capital: float,
        win_rate: float = 0.0,
        risk_reward_ratio: float = 1.0,
        max_stake_pct: float = 0.02,
        Kelly_reduction: float = 0.5,
        min_stake: float = 10.0,
        max_stake: float = float("inf"),
    ):
        self.capital = capital
        self.win_rate = max(0.0, min(1.0, win_rate))
        self.risk_reward_ratio = max(0.01, risk_reward_ratio)
        self.max_stake_pct = max_stake_pct
        self.Kelly_reduction = Kelly_reduction
        self.min_stake = min_stake
        self.max_stake = max_stake

    @property
    def edge(self) -> float:
        """Edge = win_rate * risk_reward_ratio - (1 - win_rate)."""
        return self.win_rate * self.risk_reward_ratio - (1 - self.win_rate)

    @property
    def Kelly_fraction(self) -> float:
        """Kelly fraction capped at max_stake_pct, with Kelly reduction applied."""
        if self.risk_reward_ratio == 0:
            return 0.0
        kelly = self.win_rate - ((1 - self.win_rate) / self.risk_reward_ratio)
        kelly = max(0.0, kelly) * self.Kelly_reduction
        return min(kelly, self.max_stake_pct)

    @property
    def expected_value(self) -> float:
        """Expected value per trade as a ratio."""
        return self.win_rate * self.risk_reward_ratio - (1 - self.win_rate)

    def calculate_stake(
        self,
        entry_price: float,
        stoploss: float,
        leverage: float = 1.0,
    ) -> float:
        """
        Calculate the stake amount for a trade using edge sizing.

        Args:
            entry_price : Entry price per unit
            stoploss    : Stoploss ratio (e.g., 0.05 = 5% stop distance)
            leverage    : Leverage multiplier (default 1.0)

        Returns:
            Stake amount in quote currency
        """
        if entry_price <= 0 or stoploss <= 0:
            return self.min_stake

        # Kelly-based stake amount
        stake = self.capital * self.Kelly_fraction

        # Apply stoploss-adjusted stake (risk-based)
        # Risk = stake * stoploss_pct
        # We want risk to be controlled, so adjust:
        risk_pct = abs(stoploss)
        risk_adjusted_stake = stake

        # Apply leverage cap (reduce stake so effective risk is capped)
        effective_max_stake = self.max_stake / max(1.0, leverage)

        # Clamp to bounds
        stake = max(self.min_stake, min(stake, effective_max_stake))

        return round(stake, 8)

    def calculate_quantity(
        self,
        stake: float,
        entry_price: float,
        stoploss: float,
        leverage: float = 1.0,
    ) -> float:
        """
        Calculate the quantity (number of units) from a given stake.

        Args:
            stake       : Stake amount in quote currency
            entry_price : Entry price per unit
            stoploss    : Stoploss ratio
            leverage    : Leverage multiplier

        Returns:
            Quantity in base currency
        """
        if entry_price <= 0:
            return 0.0
        quantity = (stake / entry_price) * leverage
        return round(quantity, 8)

    def calculate_stoploss_price(
        self,
        entry_price: float,
        stoploss_ratio: float,
        direction: SignalDirection = SignalDirection.LONG,
    ) -> float:
        """
        Calculate the stoploss trigger price.

        Args:
            entry_price    : Entry price per unit
            stoploss_ratio : Stoploss as ratio (e.g., 0.05 = 5% below entry)
            direction      : LONG or SHORT

        Returns:
            Stoploss trigger price
        """
        if direction == SignalDirection.LONG:
            return entry_price * (1 - stoploss_ratio)
        else:
            return entry_price * (1 + stoploss_ratio)

    def calculate_take_profit_price(
        self,
        entry_price: float,
        profit_ratio: float,
        direction: SignalDirection = SignalDirection.LONG,
    ) -> float:
        """
        Calculate the take-profit trigger price.

        Args:
            entry_price  : Entry price per unit
            profit_ratio : Target profit as ratio (e.g., 0.05 = 5% profit)
            direction    : LONG or SHORT

        Returns:
            Take-profit trigger price
        """
        if direction == SignalDirection.LONG:
            return entry_price * (1 + profit_ratio)
        else:
            return entry_price * (1 - profit_ratio)

    def estimate_max_loss(self, stake: float, stoploss: float) -> float:
        """Estimate maximum loss for a trade."""
        return stake * abs(stoploss)

    def estimate_expected_profit(self, stake: float) -> float:
        """Estimate expected profit per trade given current edge."""
        return stake * self.expected_value


# =============================================================================
# EMASpreadStrategy — EMA crossover spread trading strategy
# =============================================================================

class EMASpreadStrategy(FreqtradeStrategy):
    """
    EMA Crossover Spread Strategy.

    Buys when the fast EMA crosses above the slow EMA (spread turns positive).
    Sells when the fast EMA crosses below the slow EMA (spread turns negative).

    Parameters:
        fast_ema_period  : Fast EMA period (default 10)
        slow_ema_period : Slow EMA period (default 20)
        spread_threshold: Minimum spread to trigger signal (default 0.0)
        trade_short     : Allow short trades (default False)

    Signals:
        enter_long : Fast EMA crosses above slow EMA
        enter_short: Fast EMA crosses below slow EMA (if trade_short=True)
        exit_long  : Fast EMA crosses below slow EMA
        exit_short : Fast EMA crosses above slow EMA

    Freqtrade patterns absorbed:
        - Strategy-level parameter definition
        - informative_pairs for multi-timeframe analysis
        - populate_indicators / populate_entry_trend / populate_exit_trend separation
    """

    strategy_config = StrategyConfig(
        stoploss=-0.05,
        minimal_roi=ROI({0: 0.04, 30: 0.02, 60: 0.01}),
        timeframe="1h",
        can_short=False,
        use_exit_signal=True,
    )

    # Strategy parameters
    fast_ema_period: int = 10
    slow_ema_period: int = 20
    spread_threshold: float = 0.0

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add EMA and spread indicators."""
        if "close" not in dataframe.columns:
            return dataframe

        # Fast and slow EMAs
        dataframe["ema_fast"] = self._ema(dataframe["close"], self.fast_ema_period)
        dataframe["ema_slow"] = self._ema(dataframe["close"], self.slow_ema_period)

        # Spread (difference between EMAs)
        dataframe["spread"] = dataframe["ema_fast"] - dataframe["ema_slow"]

        # Spread as percentage of slow EMA (normalized)
        dataframe["spread_pct"] = (
            (dataframe["ema_fast"] - dataframe["ema_slow"]) / dataframe["ema_slow"]
        )

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate entry signals based on EMA crossover."""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        if "spread" not in dataframe.columns or "ema_fast" not in dataframe.columns:
            return dataframe

        # Long entry: fast EMA crosses above slow EMA (spread turns positive)
        long_mask = (
            (dataframe["spread"] > self.spread_threshold)
            & (dataframe["spread"].shift(1) <= self.spread_threshold)
        )
        dataframe.loc[long_mask, "enter_long"] = 1
        dataframe.loc[long_mask, "enter_tag"] = "ema_cross_up"

        if self.can_short:
            # Short entry: fast EMA crosses below slow EMA
            short_mask = (
                (dataframe["spread"] < -self.spread_threshold)
                & (dataframe["spread"].shift(1) >= -self.spread_threshold)
            )
            dataframe.loc[short_mask, "enter_short"] = 1
            dataframe.loc[short_mask, "enter_tag"] = "ema_cross_down"

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate exit signals based on EMA crossover reversal."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        if "spread" not in dataframe.columns:
            return dataframe

        # Long exit: fast EMA crosses below slow EMA
        exit_long_mask = (
            (dataframe["spread"] < -self.spread_threshold)
            & (dataframe["spread"].shift(1) >= -self.spread_threshold)
        )
        dataframe.loc[exit_long_mask, "exit_long"] = 1

        if self.can_short:
            # Short exit: fast EMA crosses above slow EMA
            exit_short_mask = (
                (dataframe["spread"] > self.spread_threshold)
                & (dataframe["spread"].shift(1) <= self.spread_threshold)
            )
            dataframe.loc[exit_short_mask, "exit_short"] = 1

        return dataframe

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA manually (avoids talib dependency)."""
        return series.ewm(span=period, adjust=False).mean()


# =============================================================================
# MACDStrategy — MACD-based trend strategy
# =============================================================================

class MACDStrategy(FreqtradeStrategy):
    """
    MACD-based Trend Following Strategy.

    Uses the classic MACD indicator (12/26/9) to identify trend direction.
    Buys when MACD line crosses above signal line in uptrend.
    Sells when MACD line crosses below signal line.

    Parameters:
        macd_fast    : Fast EMA period for MACD (default 12)
        macd_slow    : Slow EMA period for MACD (default 26)
        macd_signal  : Signal line EMA period (default 9)
        macd_enabled : Enable MACD filter (default True)
        adx_enabled  : Enable ADX filter (default True)
        adx_threshold: Minimum ADX to consider trend (default 25)

    Signals:
        enter_long : MACD cross above signal line + ADX trend confirmation
        exit_long  : MACD cross below signal line

    Freqtrade patterns absorbed:
        - Multi-indicator confirmation
        - Signal tagging
        - Entry/exit signal separation
    """

    strategy_config = StrategyConfig(
        stoploss=-0.08,
        minimal_roi=ROI({0: 0.06, 45: 0.03, 90: 0.015}),
        timeframe="1h",
        can_short=False,
        use_exit_signal=True,
    )

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_enabled: bool = True
    adx_enabled: bool = True
    adx_threshold: float = 25.0

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add MACD and ADX indicators."""
        if "close" not in dataframe.columns:
            return dataframe

        # MACD
        ema_fast = dataframe["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = dataframe["close"].ewm(span=self.macd_slow, adjust=False).mean()
        dataframe["macd"] = ema_fast - ema_slow
        dataframe["macd_signal"] = dataframe["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        dataframe["macd_hist"] = dataframe["macd"] - dataframe["macd_signal"]

        # ADX (Average Directional Index) — trend strength
        dataframe["adx"] = self._adx(dataframe)

        # RSI (momentum confirmation)
        dataframe["rsi"] = self._rsi(dataframe["close"])

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate entry signals from MACD crossover."""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        if "macd_hist" not in dataframe.columns:
            return dataframe

        # MACD histogram crossover detection
        macd_cross_up = (
            (dataframe["macd_hist"] > 0)
            & (dataframe["macd_hist"].shift(1) <= 0)
        )
        macd_cross_down = (
            (dataframe["macd_hist"] < 0)
            & (dataframe["macd_hist"].shift(1) >= 0)
        )

        # Optional ADX trend filter
        if self.adx_enabled and "adx" in dataframe.columns:
            strong_trend = dataframe["adx"] > self.adx_threshold
            long_condition = macd_cross_up & strong_trend
        else:
            long_condition = macd_cross_up

        dataframe.loc[long_condition, "enter_long"] = 1
        dataframe.loc[long_condition, "enter_tag"] = "macd_cross_up"

        if self.can_short and "macd_hist" in dataframe.columns:
            if self.adx_enabled and "adx" in dataframe.columns:
                short_condition = macd_cross_down & strong_trend
            else:
                short_condition = macd_cross_down
            dataframe.loc[short_condition, "enter_short"] = 1
            dataframe.loc[short_condition, "enter_tag"] = "macd_cross_down"

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate exit signals from MACD reversal."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        if "macd_hist" not in dataframe.columns:
            return dataframe

        # Exit long: MACD crosses below signal
        dataframe.loc[
            (dataframe["macd_hist"] < 0) & (dataframe["macd_hist"].shift(1) >= 0),
            "exit_long",
        ] = 1

        if self.can_short:
            dataframe.loc[
                (dataframe["macd_hist"] > 0) & (dataframe["macd_hist"].shift(1) <= 0),
                "exit_short",
            ] = 1

        return dataframe

    @staticmethod
    def _adx(dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX manually."""
        high = dataframe["high"]
        low = dataframe["low"]
        close = dataframe["close"]

        # Plus Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = tr1.combine(tr2, max).combine(tr3, max)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        return adx

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# =============================================================================
# PriceBandStrategy — Price band breakout strategy
# =============================================================================

class PriceBandStrategy(FreqtradeStrategy):
    """
    Price Band Breakout Strategy.

    Buys when price breaks above the upper Bollinger Band.
    Sells when price falls below the lower Bollinger Band or the middle band.

    Parameters:
        bb_length      : Bollinger Band period (default 20)
        bb_std         : Number of standard deviations (default 2)
        rsi_length     : RSI period for confirmation (default 14)
        rsi_upper      : RSI upper threshold (default 60)
        rsi_lower      : RSI lower threshold (default 40)
        trade_short    : Allow short trades (default False)

    Signals:
        enter_long : Price crosses above upper Bollinger Band + RSI confirmation
        exit_long  : Price crosses below middle band or lower band

    Freqtrade patterns absorbed:
        - Bollinger Band integration
        - Multi-condition entry signals
        - Band crossover detection
    """

    strategy_config = StrategyConfig(
        stoploss=-0.06,
        minimal_roi=ROI({0: 0.05, 20: 0.025, 60: 0.01}),
        timeframe="1h",
        can_short=False,
        use_exit_signal=True,
    )

    bb_length: int = 20
    bb_std: float = 2.0
    rsi_length: int = 14
    rsi_upper: float = 60.0
    rsi_lower: float = 40.0

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """Add Bollinger Bands and RSI indicators."""
        if "close" not in dataframe.columns:
            return dataframe

        # Bollinger Bands
        rolling_mean = dataframe["close"].rolling(window=self.bb_length).mean()
        rolling_std = dataframe["close"].rolling(window=self.bb_length).std()

        dataframe["bb_middle"] = rolling_mean
        dataframe["bb_upper"] = rolling_mean + (rolling_std * self.bb_std)
        dataframe["bb_lower"] = rolling_mean - (rolling_std * self.bb_std)
        dataframe["bb_width"] = (
            (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_middle"]
        )

        # RSI
        dataframe["rsi"] = self._rsi(dataframe["close"])

        # Price position relative to bands
        dataframe["price_pct_b"] = (
            (dataframe["close"] - dataframe["bb_lower"])
            / (dataframe["bb_upper"] - dataframe["bb_lower"])
        )

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate entry signals from Bollinger Band breakout."""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        if "bb_upper" not in dataframe.columns or "rsi" not in dataframe.columns:
            return dataframe

        # Long entry: price breaks above upper band + RSI confirming strength
        long_mask = (
            (dataframe["close"] > dataframe["bb_upper"])
            & (dataframe["close"].shift(1) <= dataframe["bb_upper"].shift(1))
            & (dataframe["rsi"] < self.rsi_upper)
        )
        dataframe.loc[long_mask, "enter_long"] = 1
        dataframe.loc[long_mask, "enter_tag"] = "bb_breakout_up"

        if self.can_short:
            # Short entry: price breaks below lower band + RSI confirming weakness
            short_mask = (
                (dataframe["close"] < dataframe["bb_lower"])
                & (dataframe["close"].shift(1) >= dataframe["bb_lower"].shift(1))
                & (dataframe["rsi"] > self.rsi_lower)
            )
            dataframe.loc[short_mask, "enter_short"] = 1
            dataframe.loc[short_mask, "enter_tag"] = "bb_breakout_down"

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate exit signals from Bollinger Band mean reversion."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        if "bb_middle" not in dataframe.columns:
            return dataframe

        # Exit long: price crosses below middle band
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_middle"])
            & (dataframe["close"].shift(1) >= dataframe["bb_middle"].shift(1)),
            "exit_long",
        ] = 1

        if self.can_short:
            # Exit short: price crosses above middle band
            dataframe.loc[
                (dataframe["close"] > dataframe["bb_middle"])
                & (dataframe["close"].shift(1) <= dataframe["bb_middle"].shift(1)),
                "exit_short",
            ] = 1

        return dataframe

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# =============================================================================
# FreqtradeExecutor — Execution adapter backed by existing connectors
# =============================================================================

class FreqtradeExecutor:
    """
    Execution adapter providing freqtrade-style execution hooks.

    This executor bridges freqtrade strategy concepts to the existing
    connector architecture. It uses existing connectors (CEXAdapter,
    FreqtradeExchangeAdapter, etc.) as the backend for order execution.

    Key methods:
        calculate_stake_amount() : Edge-based or fixed stake sizing
        execute_entry()          : Place an entry order
        execute_exit()           : Place an exit order
        check_stoploss()         : Check and trigger stoploss if needed
        apply_roi()              : Check and apply ROI
        adjust_position()        : DCA / position adjustment

    Args:
        connector   : BaseConnector instance for exchange communication
        strategy    : FreqtradeStrategy instance
        capital     : Total available capital in stake currency
        edge_sizer  : Optional EdgePositionSizer for edge-based sizing

    Freqtrade patterns absorbed:
        - Stake amount calculation with edge sizing
        - ROI-based exit logic
        - Dynamic stoploss checking
        - Trailing stop logic
        - Position adjustment (DCA)
        - Order tracking and lifecycle management
    """

    def __init__(
        self,
        connector: "BaseConnector",
        strategy: FreqtradeStrategy,
        capital: float = 10000.0,
        edge_sizer: Optional[EdgePositionSizer] = None,
        stake_amount: float = 100.0,
        tradable_balance_ratio: float = 0.99,
        dry_run: bool = True,
    ):
        """
        Initialize FreqtradeExecutor.

        Args:
            connector                : Exchange connector
            strategy                 : FreqtradeStrategy instance
            capital                  : Total capital in stake currency
            edge_sizer               : Optional EdgePositionSizer
            stake_amount             : Fixed stake amount (if no edge_sizer)
            tradable_balance_ratio   : Ratio of balance available for trading
            dry_run                  : Dry-run mode (simulated fills)
        """
        self.connector = connector
        self.strategy = strategy
        self.capital = capital
        self.edge_sizer = edge_sizer
        self.stake_amount = stake_amount
        self.tradable_balance_ratio = tradable_balance_ratio
        self.dry_run = dry_run

        # Active positions
        self._positions: Dict[str, Position] = {}
        # Open trade stakes for available balance calculation
        self._open_trade_stakes: float = 0.0

        # Stoploss tracking
        self._stop_loss_prices: Dict[str, float] = {}

        # Trailing stop state
        self._trailing_stop_prices: Dict[str, float] = {}
        self._trailing_stop_offset: Dict[str, float] = {}

        # Order tracking
        self._pending_entries: Dict[str, Any] = {}
        self._pending_exits: Dict[str, Any] = {}

        # Counter for trade IDs
        self._trade_id_counter = 0

        logger.info(
            f"FreqtradeExecutor initialized: connector={connector.name}, "
            f"strategy={strategy.__class__.__name__}, capital={capital}"
        )

    # -------------------------------------------------------------------------
    # Stake Amount Calculation
    # -------------------------------------------------------------------------

    def calculate_stake_amount(
        self,
        pair: str,
        entry_price: float,
        stoploss: float,
        direction: SignalDirection = SignalDirection.LONG,
        max_open_trades: int = -1,
        leverage: float = 1.0,
    ) -> float:
        """
        Calculate the stake amount for an entry.

        If edge_sizer is configured, uses Kelly Criterion-based edge sizing.
        Otherwise, uses fixed stake_amount with open-trade adjustment.

        Args:
            pair         : Trading pair (e.g., 'BTC-USDT')
            entry_price  : Expected entry price
            stoploss     : Stoploss ratio (negative, e.g., -0.05)
            direction    : LONG or SHORT
            max_open_trades: Maximum concurrent trades (-1 = unlimited)
            leverage     : Leverage multiplier

        Returns:
            Stake amount in quote currency
        """
        # Edge-based sizing
        if self.edge_sizer is not None:
            stake = self.edge_sizer.calculate_stake(
                entry_price=entry_price,
                stoploss=abs(stoploss),
                leverage=leverage,
            )
        else:
            # Fixed stake with open-trade adjustment
            available = self.get_available_stake()
            stake = min(self.stake_amount, available)

        # Ensure stake respects tradable balance ratio
        max_allowed = self.capital * self.tradable_balance_ratio - self._open_trade_stakes
        stake = min(stake, max(0, max_allowed))

        # Ensure minimum stake from exchange
        min_stake = self._get_min_stake(pair, entry_price, stoploss, leverage)
        stake = max(stake, min_stake)

        return round(stake, 8)

    def _get_min_stake(
        self,
        pair: str,
        price: float,
        stoploss: float,
        leverage: float,
    ) -> float:
        """Get minimum stake from exchange connector."""
        try:
            if hasattr(self.connector, "get_stake_amount"):
                return self.connector.get_stake_amount(
                    pair, price, stoploss, leverage=leverage
                )
        except Exception:
            pass
        # Default minimum
        return 10.0

    def get_available_stake(self) -> float:
        """Get currently available stake amount."""
        return (self.capital * self.tradable_balance_ratio) - self._open_trade_stakes

    def _get_max_stake(
        self,
        pair: str,
        price: float,
        leverage: float = 1.0,
    ) -> float:
        """Get maximum stake from exchange connector."""
        try:
            if hasattr(self.connector, "get_max_pair_stake_amount"):
                return self.connector.get_max_pair_stake_amount(pair, price, leverage)
        except Exception:
            pass
        return float("inf")

    # -------------------------------------------------------------------------
    # Entry / Exit Execution
    # -------------------------------------------------------------------------

    def _generate_trade_id(self) -> str:
        """Generate a unique trade ID."""
        self._trade_id_counter += 1
        return f"FT_{int(time.time() * 1000)}_{self._trade_id_counter}"

    async def execute_entry(
        self,
        pair: str,
        direction: SignalDirection,
        stake_amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        enter_tag: Optional[str] = None,
        leverage: float = 1.0,
        dry_run: Optional[bool] = None,
    ) -> Tuple[Position, Optional[str]]:
        """
        Execute an entry order for a pair.

        Args:
            pair          : Trading pair
            direction     : LONG or SHORT
            stake_amount  : Stake amount in quote currency
            price         : Limit price (None = market order)
            order_type    : 'market' or 'limit'
            enter_tag     : Optional entry reason tag
            leverage      : Leverage for the position
            dry_run       : Override dry_run setting

        Returns:
            Tuple of (Position object, order_id or None if dry_run)
        """
        is_dry_run = dry_run if dry_run is not None else self.dry_run

        # Get current market price if not specified
        if price is None:
            price = self._get_market_price(pair)

        if price <= 0:
            raise ValueError(f"Invalid price for {pair}: {price}")

        # Calculate quantity
        quantity = (stake_amount / price) * leverage

        # Quantize to exchange precision
        if hasattr(self.connector, "quantize_order_amount"):
            quantity = float(
                self.connector.quantize_order_amount(pair, Decimal(str(quantity)))
            )

        # Place order via connector
        order_id = None
        if not is_dry_run:
            from quant_trading.execution.executor import OrderSide, OrderType

            side = OrderSide.BUY if direction == SignalDirection.LONG else OrderSide.SELL
            ot = OrderType.MARKET if order_type == "market" else OrderType.LIMIT

            result = await self.connector.place_order(
                trading_pair=pair,
                side=side,
                amount=Decimal(str(quantity)),
                order_type=ot,
                price=Decimal(str(price)) if price else None,
            )
            order_id = result.id if hasattr(result, "id") else str(result)

        # Create position
        trade_id = self._generate_trade_id()
        position = Position(
            id=trade_id,
            pair=pair,
            direction=direction,
            entry_price=price,
            current_price=price,
            stake_amount=stake_amount,
            amount=quantity,
            leverage=leverage,
            open_date=datetime.now(),
            entry_order_id=order_id or f"DRY_{trade_id}",
            enter_tag=enter_tag,
            stop_loss=self.strategy.stoploss,
            entries=[{
                "price": price,
                "amount": quantity,
                "date": datetime.now(),
                "stake": stake_amount,
            }],
        )

        # Track position
        self._positions[pair] = position
        self._open_trade_stakes += stake_amount

        # Set stoploss price
        sl_price = self._calculate_stoploss_price(position)
        self._stop_loss_prices[trade_id] = sl_price

        logger.info(
            f"Entry executed: pair={pair}, direction={direction.value}, "
            f"price={price}, qty={quantity}, stake={stake_amount}, "
            f"trade_id={trade_id}"
        )

        return position, order_id

    async def execute_exit(
        self,
        position: Position,
        exit_reason: str = "exit_signal",
        price: Optional[float] = None,
        order_type: str = "market",
        dry_run: Optional[bool] = None,
    ) -> Tuple[Position, Optional[str], float]:
        """
        Execute an exit order for a position.

        Args:
            position   : Active Position object
            exit_reason: Reason for exit (e.g., 'roi', 'stoploss', 'exit_signal')
            price      : Limit price (None = market order)
            order_type : 'market' or 'limit'
            dry_run    : Override dry_run setting

        Returns:
            Tuple of (updated Position, order_id or None, realized profit)
        """
        is_dry_run = dry_run if dry_run is not None else self.dry_run

        if price is None:
            price = self._get_market_price(position.pair)

        # Calculate exit quantity
        quantity = position.amount

        # Place order via connector
        order_id = None
        if not is_dry_run:
            from quant_trading.execution.executor import OrderSide, OrderType

            side = OrderSide.SELL if position.direction == SignalDirection.LONG else OrderSide.BUY
            ot = OrderType.MARKET if order_type == "market" else OrderType.LIMIT

            result = await self.connector.place_order(
                trading_pair=position.pair,
                side=side,
                amount=Decimal(str(quantity)),
                order_type=ot,
                price=Decimal(str(price)) if price else None,
            )
            order_id = result.id if hasattr(result, "id") else str(result)

        # Calculate realized profit
        if position.direction == SignalDirection.LONG:
            profit_ratio = (price - position.entry_price) / position.entry_price
        else:
            profit_ratio = (position.entry_price - price) / position.entry_price

        realized_profit = position.stake_amount * profit_ratio * position.leverage

        # Update position
        position.exit_order_id = order_id or f"DRY_EXIT_{position.id}"
        position.exit_reason = exit_reason
        position.is_open = False
        position.realized_profit = realized_profit

        # Remove from active positions
        if position.pair in self._positions:
            del self._positions[position.pair]
        self._open_trade_stakes -= position.stake_amount

        # Clean up tracking
        self._stop_loss_prices.pop(position.id, None)
        self._trailing_stop_prices.pop(position.id, None)

        logger.info(
            f"Exit executed: pair={position.pair}, reason={exit_reason}, "
            f"price={price}, profit_ratio={profit_ratio:.4f}, "
            f"realized_profit={realized_profit:.2f}"
        )

        return position, order_id, realized_profit

    # -------------------------------------------------------------------------
    # Stoploss / ROI Checking
    # -------------------------------------------------------------------------

    def _get_market_price(self, pair: str) -> float:
        """Get current market price for a pair."""
        try:
            if hasattr(self.connector, "get_price"):
                p = self.connector.get_price(pair, is_buy=True)
                return float(p) if p else 0.0
            elif hasattr(self.connector, "get_ticker"):
                ticker = self.connector.get_ticker(pair)
                return ticker.get("last", 0.0) if ticker else 0.0
        except Exception:
            pass
        return 0.0

    def _calculate_stoploss_price(
        self,
        position: Position,
        custom_stoploss: Optional[float] = None,
    ) -> float:
        """Calculate the stoploss trigger price for a position."""
        stoploss_ratio = custom_stoploss if custom_stoploss is not None else position.stop_loss
        if stoploss_ratio is None:
            stoploss_ratio = self.strategy.stoploss

        if position.direction == SignalDirection.LONG:
            return position.entry_price * (1 + stoploss_ratio)
        else:
            return position.entry_price * (1 - stoploss_ratio)

    async def check_stoploss(
        self,
        position: Position,
        current_price: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if stoploss is triggered for a position.

        Args:
            position    : Active position
            current_price: Current market price (auto-fetched if None)
            current_time: Current datetime (now if None)

        Returns:
            True if stoploss was triggered and exit was executed
        """
        if current_price is None:
            current_price = self._get_market_price(position.pair)

        position.current_price = current_price

        # Get custom stoploss from strategy
        now = current_time or datetime.now()
        custom_sl = self.strategy.custom_stoploss(
            trade=position,
            current_time=now,
            current_rate=current_price,
            current_profit=position.profit_ratio,
        )

        # Calculate stoploss price
        sl_price = self._calculate_stoploss_price(position, custom_stoploss=custom_sl)

        # Check trailing stop
        if self.strategy.trailing_stop and position.direction == SignalDirection.LONG:
            ts_offset = self.strategy.trailing_stop_positive or 0.02
            if self.strategy.trailing_only_offset_is_reached:
                if position.profit_ratio >= ts_offset:
                    trailing_price = position.entry_price * (1 + ts_offset)
                    if current_price < trailing_price:
                        await self.execute_exit(position, "trailing_stop")
                        return True
            else:
                # Always update trailing stop
                if position.profit_ratio > ts_offset:
                    trailing_sl = position.profit_ratio - 0.005  # 0.5% trail
                    if current_price < position.entry_price * (1 + trailing_sl):
                        await self.execute_exit(position, "trailing_stop")
                        return True

        # Check hard stoploss
        if position.direction == SignalDirection.LONG:
            triggered = current_price <= sl_price
        else:
            triggered = current_price >= sl_price

        if triggered:
            logger.info(
                f"Stoploss triggered: pair={position.pair}, "
                f"price={current_price}, sl_price={sl_price}"
            )
            await self.execute_exit(position, "stoploss", current_price)
            return True

        return False

    async def check_roi(
        self,
        position: Position,
        current_price: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if ROI threshold is reached for a position.

        Args:
            position     : Active position
            current_price: Current market price (auto-fetched if None)
            current_time : Current datetime (now if None)

        Returns:
            True if ROI was reached and exit was executed
        """
        if current_price is None:
            current_price = self._get_market_price(position.pair)

        position.current_price = current_price
        now = current_time or datetime.now()

        # Check custom ROI first
        custom_roi = self.strategy.custom_roi(
            trade=position,
            current_time=now,
            current_rate=current_price,
            current_profit=position.profit_ratio,
        )

        # Determine which ROI table to use
        if custom_roi is not None:
            roi_target = custom_roi
        else:
            roi_target = self.strategy.minimal_roi.get_roi_at(int(position.age_minutes))

        # Check if profit exceeds ROI target
        if position.profit_ratio >= roi_target:
            logger.info(
                f"ROI reached: pair={position.pair}, profit_ratio={position.profit_ratio:.4f}, "
                f"roi_target={roi_target:.4f}, age_minutes={position.age_minutes:.1f}"
            )
            if self.strategy.exit_profit_only or (
                not self.strategy.ignore_roi_if_entry_signal
            ):
                await self.execute_exit(position, "roi", current_price)
                return True

        return False

    # -------------------------------------------------------------------------
    # Position Adjustment (DCA)
    # -------------------------------------------------------------------------

    async def adjust_position(
        self,
        position: Position,
        current_price: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> Optional[float]:
        """
        Check and execute position adjustment (DCA) for a position.

        Args:
            position    : Active position
            current_price: Current market price (auto-fetched if None)
            current_time : Current datetime (now if None)

        Returns:
            New stake amount if adjustment was made, None otherwise
        """
        if not self.strategy.position_adjustment_enable:
            return None

        if current_price is None:
            current_price = self._get_market_price(position.pair)

        position.current_price = current_price
        now = current_time or datetime.now()

        # Get min/max stake from exchange
        min_stake = self._get_min_stake(
            position.pair, current_price, abs(self.strategy.stoploss), position.leverage
        )
        max_stake = self._get_max_stake(position.pair, current_price, position.leverage)
        available = self.get_available_stake()

        # Call strategy's adjust_trade_position
        adjustment = self.strategy.adjust_trade_position(
            trade=position,
            current_time=now,
            current_rate=current_price,
            current_profit=position.profit_ratio,
            min_stake=min_stake,
            max_stake=min_stake,
            current_entry_rate=position.entry_price,
            current_exit_rate=current_price,
            current_entry_profit=position.profit_ratio,
            current_exit_profit=position.profit_ratio,
        )

        if adjustment is None or adjustment == 0:
            return None

        # Check against limits
        if abs(adjustment) > available:
            adjustment = available if adjustment > 0 else -available

        new_total_stake = position.stake_amount + adjustment

        if new_total_stake < min_stake or new_total_stake > max_stake:
            return None

        # Execute adjustment as additional entry
        if adjustment > 0:
            qty = (adjustment / current_price) * position.leverage
            if hasattr(self.connector, "quantize_order_amount"):
                qty = float(
                    self.connector.quantize_order_amount(
                        position.pair, Decimal(str(qty))
                    )
                )
            position.entries.append({
                "price": current_price,
                "amount": qty,
                "date": now,
                "stake": adjustment,
            })
            position.stake_amount = new_total_stake
            position.amount += qty
            position.entry_price = (
                sum(e["price"] * e["amount"] for e in position.entries)
                / sum(e["amount"] for e in position.entries)
            )
            self._open_trade_stakes += adjustment
            logger.info(
                f"Position adjusted (DCA): pair={position.pair}, "
                f"additional_stake={adjustment}, new_stake={new_total_stake}"
            )
        else:
            # Partial close
            reduce_qty = (abs(adjustment) / current_price) * position.leverage
            if hasattr(self.connector, "quantize_order_amount"):
                reduce_qty = float(
                    self.connector.quantize_order_amount(
                        position.pair, Decimal(str(reduce_qty))
                    )
                )
            position.entries.append({
                "price": current_price,
                "amount": -reduce_qty,
                "date": now,
                "stake": adjustment,
            })
            position.amount = max(0, position.amount - reduce_qty)
            position.stake_amount = new_total_stake
            self._open_trade_stakes += adjustment
            logger.info(
                f"Position adjusted (partial close): pair={position.pair}, "
                f"reduced_stake={abs(adjustment)}, new_stake={new_total_stake}"
            )

        return new_total_stake

    # -------------------------------------------------------------------------
    # Position Management
    # -------------------------------------------------------------------------

    def get_open_positions(self) -> Dict[str, Position]:
        """Get all currently open positions."""
        return self._positions.copy()

    def get_position(self, pair: str) -> Optional[Position]:
        """Get the open position for a pair, if any."""
        return self._positions.get(pair)

    def has_position(self, pair: str) -> bool:
        """Check if there is an open position for a pair."""
        return pair in self._positions

    async def update_positions(
        self,
        pairs: List[str],
    ) -> List[Tuple[Position, str, Any]]:
        """
        Update all positions and check for exit conditions.

        For each open position, checks:
        1. Stoploss trigger
        2. ROI target
        3. Position adjustment

        Args:
            pairs: List of pairs to check

        Returns:
            List of (position, trigger_reason, result) for exits
        """
        results = []

        for pair, position in list(self._positions.items()):
            if pair not in pairs:
                continue

            current_price = self._get_market_price(pair)
            position.current_price = current_price

            # Check stoploss
            if await self.check_stoploss(position, current_price):
                results.append((position, "stoploss", True))
                continue

            # Check ROI
            if await self.check_roi(position, current_price):
                results.append((position, "roi", True))
                continue

            # Check position adjustment
            if self.strategy.position_adjustment_enable:
                await self.adjust_position(position, current_price)

        return results

    def get_equity(self) -> float:
        """Get current total equity (capital + open PnL)."""
        total_pnl = sum(
            p.stake_amount * p.profit_ratio * p.leverage
            for p in self._positions.values()
        )
        return self.capital + total_pnl


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core
    "HAS_FREQTRADE",
    "FreqtradeStrategy",
    "EdgePositionSizer",
    "EMASpreadStrategy",
    "MACDStrategy",
    "PriceBandStrategy",
    "FreqtradeExecutor",
    # Data classes
    "SignalDirection",
    "EntryExit",
    "ROI",
    "StrategyConfig",
    "TradeSignal",
    "Position",
]
