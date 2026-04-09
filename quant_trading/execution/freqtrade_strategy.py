"""
Freqtrade Strategy Interface Wrapper

将 freqtrade 的 IStrategy 接口封装为本地策略系统可用的接口。

Freqtrade 策略特点:
- populate_indicators() - 计算技术指标
- populate_buy_trend() / populate_sell_trend() - 生成交易信号
- adjust_trade_position() - 动态仓位调整
-确认止损止盈
- 自定义止损/止盈/保护机制
- FreqAI 集成 (机器学习特征)
- QTPyLib 内置 100+ 指标

此模块提供:
- FreqtradeStrategyWrapper: 将本地策略转换为 freqtrade 格式
- IStrategyAdapter: 将 freqtrade 策略适配到本地执行框架
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

try:
    from freqtrade.strategy import IStrategy
    from freqtrade.strategy.interface import IStrategy as FreqIStrategy
    from freqtrade.persistence import Trade, Order
    from freqtrade.data.dataprovider import DataProvider
    from freqtrade.enums import SignalDirection
    HAS_FREQTRADE = True
except ImportError:
    HAS_FREQTRADE = False
    FreqIStrategy = object

from quant_trading.execution.executor import OrderSide, OrderType

if TYPE_CHECKING:
    from freqtrade.exchange.exchange import Exchange

logger = logging.getLogger(__name__)


class FreqtradeSignal:
    """
    Freqtrade trading signal container.

    Represents a buy/sell signal from strategy analysis.
    """

    def __init__(
        self,
        pair: str,
        direction: str,  # "long", "short", "exit"
        price: Optional[float] = None,
        confidence: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        timeframe: str = "1h",
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.pair = pair
        self.direction = direction
        self.price = price
        self.confidence = confidence
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.timeframe = timeframe
        self.meta = meta or {}

    def __repr__(self):
        return (
            f"FreqtradeSignal(pair={self.pair}, direction={self.direction}, "
            f"price={self.price}, confidence={self.confidence})"
        )


class BaseStrategy(ABC):
    """
    Base strategy interface for local strategy system.

    Subclasses should implement the abstract methods to define
    their trading strategy.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @property
    @abstractmethod
    def timeframe(self) -> str:
        """Default timeframe for strategy."""
        pass

    @abstractmethod
    def analyze(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze dataframe and add indicators.

        Args:
            dataframe: Price data with OHLCV columns
            metadata: Additional metadata (pair, timeframe, etc.)

        Returns:
            DataFrame with added indicator columns
        """
        pass

    @abstractmethod
    def generate_signals(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> List[FreqtradeSignal]:
        """
        Generate trading signals from analyzed dataframe.

        Args:
            dataframe: DataFrame with indicators
            metadata: Additional metadata

        Returns:
            List of trading signals
        """
        pass

    def get_entry_logic(self) -> List[str]:
        """Return list of entry condition column names."""
        return ["enter_long"]

    def get_exit_logic(self) -> List[str]:
        """Return list of exit condition column names."""
        return ["exit_long"]


class FreqtradeStrategyWrapper:
    """
    Wrapper to convert local strategies to freqtrade IStrategy format.

    This allows local strategies written against BaseStrategy
    to run on the freqtrade execution framework.
    """

    def __init__(
        self,
        strategy: "BaseStrategy",
        stake_currency: str = "USDT",
        dry_run: bool = True,
    ):
        """
        Initialize strategy wrapper.

        Args:
            strategy: Local BaseStrategy implementation
            stake_currency: Stake currency for trading
            dry_run: Enable dry-run mode
        """
        if not HAS_FREQTRADE:
            raise ImportError("freqtrade not installed")

        self._strategy = strategy
        self._stake_currency = stake_currency
        self._dry_run = dry_run
        self._minimal_roi: Dict[int, float] = {}
        self._stoploss: float = -0.10
        self._timeframe: str = strategy.timeframe
        self._dataprovider: Optional[DataProvider] = None

    def set_dataprovider(self, dataprovider: DataProvider) -> None:
        """Set the freqtrade DataProvider."""
        self._dataprovider = dataprovider

    def set_minimal_roi(self, roi: Dict[int, float]) -> None:
        """Set minimal ROI table."""
        self._minimal_roi = roi

    def set_stoploss(self, stoploss: float) -> None:
        """Set stoploss percentage."""
        self._stoploss = stoploss

    # IStrategy interface methods (freqtrade expects these)
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Populate indicators for the strategy.

        Freqtrade will call this to add technical indicators
        to the dataframe before signal generation.
        """
        return self._strategy.analyze(dataframe, metadata)

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Populate buy signal.

        Freqtrade expects a 'enter_long' column with 1 for buy signals.
        """
        signals = self._strategy.generate_signals(dataframe, metadata)
        dataframe["enter_long"] = 0
        for signal in signals:
            if signal.direction == "long":
                pair_idx = dataframe[dataframe["pair"] == signal.pair].index
                if len(pair_idx) > 0:
                    dataframe.loc[pair_idx, "enter_long"] = 1
                    if signal.stop_loss:
                        dataframe.loc[pair_idx, "stoploss"] = signal.stop_loss
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Populate sell signal.

        Freqtrade expects an 'exit_long' column with 1 for sell signals.
        """
        signals = self._strategy.generate_signals(dataframe, metadata)
        dataframe["exit_long"] = 0
        for signal in signals:
            if signal.direction == "exit":
                pair_idx = dataframe[dataframe["pair"] == signal.pair].index
                if len(pair_idx) > 0:
                    dataframe.loc[pair_idx, "exit_long"] = 1
        return dataframe

    def get_entry_signal(self, pair: str, dataframe: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Get entry signal for a pair.

        Returns:
            Tuple of (enter_long, enter_short)
        """
        if len(dataframe) == 0:
            return False, False
        enter_long = bool(dataframe["enter_long"].iloc[-1])
        enter_short = bool(dataframe.get("enter_short", pd.Series([0])).iloc[-1])
        return enter_long, enter_short

    def get_exit_signal(self, pair: str, dataframe: pd.DataFrame) -> Tuple[bool, bool]:
        """
        Get exit signal for a pair.

        Returns:
            Tuple of (exit_long, exit_short)
        """
        if len(dataframe) == 0:
            return False, False
        exit_long = bool(dataframe["exit_long"].iloc[-1])
        exit_short = bool(dataframe.get("exit_short", pd.Series([0])).iloc[-1])
        return exit_long, exit_short

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_vol: float,
        dataframe: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Optional[float]:
        """
        Adjust trade position (DCA - Dollar Cost Averaging).

        Return None to skip position adjustment.
        Return a positive number to increase position.
        Return a negative number to decrease position.
        """
        return None

    # Properties expected by freqtrade
    @property
    def minimal_roi(self) -> Dict[int, float]:
        """Minimal ROI table."""
        return self._minimal_roi

    @property
    def stoploss(self) -> float:
        """Stoploss percentage."""
        return self._stoploss

    @property
    def timeframe(self) -> str:
        """Trading timeframe."""
        return self._timeframe

    @property
    def dataprovider(self) -> Optional[DataProvider]:
        """Freqtrade DataProvider."""
        return self._dataprovider

    def check_buy_timeout(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> bool:
        """Check if buy order should be cancelled due to timeout."""
        return False

    def check_sell_timeout(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> bool:
        """Check if sell order should be cancelled due to timeout."""
        return False

    def order_about_to_fill(self, trade: Trade, order: Order) -> None:
        """Hook called when an order is about to fill."""
        pass


class IStrategyAdapter(FreqtradeStrategyWrapper):
    """
    Adapter to run freqtrade IStrategy in local execution framework.

    This allows freqtrade strategies (from freqtrade ecosystem)
    to be used with the local execution system.
    """

    def __init__(
        self,
        freqtrade_strategy: Optional["FreqIStrategy"] = None,
        stake_currency: str = "USDT",
        dry_run: bool = True,
    ):
        """
        Initialize IStrategy adapter.

        Args:
            freqtrade_strategy: An IStrategy instance from freqtrade
            stake_currency: Stake currency
            dry_run: Dry-run mode
        """
        if not HAS_FREQTRADE:
            raise ImportError("freqtrade not installed")

        if freqtrade_strategy is None:
            # Create a minimal stub
            class MinimalStrategy(IStrategy):
                pass
            freqtrade_strategy = MinimalStrategy()

        self._freqtrade_strategy = freqtrade_strategy
        self._stake_currency = stake_currency
        self._dry_run = dry_run

        # Copy settings from freqtrade strategy
        self._minimal_roi = getattr(freqtrade_strategy, "minimal_roi", {})
        self._stoploss = getattr(freqtrade_strategy, "stoploss", -0.10)
        self._timeframe = getattr(freqtrade_strategy, "timeframe", "1h")
        self._dataprovider: Optional[DataProvider] = None

    def set_dataprovider(self, dataprovider: DataProvider) -> None:
        """Set the freqtrade DataProvider."""
        self._dataprovider = dataprovider
        self._freqtrade_strategy.dataprovider = dataprovider

    # Delegate to freqtrade strategy
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Populate indicators using freqtrade strategy."""
        return self._freqtrade_strategy.populate_indicators(dataframe, metadata)

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Populate buy signals using freqtrade strategy."""
        return self._freqtrade_strategy.populate_buy_trend(dataframe, metadata)

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Populate sell signals using freqtrade strategy."""
        return self._freqtrade_strategy.populate_sell_trend(dataframe, metadata)

    def get_entry_signal(self, pair: str, dataframe: pd.DataFrame) -> Tuple[bool, bool]:
        """Get entry signal from freqtrade strategy."""
        return self._freqtrade_strategy.get_entry_signal(pair, dataframe)

    def get_exit_signal(self, pair: str, dataframe: pd.DataFrame) -> Tuple[bool, bool]:
        """Get exit signal from freqtrade strategy."""
        return self._freqtrade_strategy.get_exit_signal(pair, dataframe)

    def adjust_trade_position(
        self,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_vol: float,
        dataframe: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> Optional[float]:
        """Adjust trade position using freqtrade strategy."""
        return self._freqtrade_strategy.adjust_trade_position(
            trade, current_time, current_rate, current_vol, dataframe, metadata
        )

    @property
    def freqtrade_strategy(self) -> "FreqIStrategy":
        """Get the underlying freqtrade strategy."""
        return self._freqtrade_strategy


class StrategyRunner:
    """
    Strategy Runner for freqtrade-style strategies.

    Runs strategies against a connector adapter and generates
    execution orders.
    """

    def __init__(
        self,
        connector: "FreqtradeExchangeAdapter",  # noqa: F821
        strategy: FreqtradeStrategyWrapper,
    ):
        """
        Initialize strategy runner.

        Args:
            connector: Exchange connector
            strategy: Strategy to run
        """
        self._connector = connector
        self._strategy = strategy
        self._running = False
        self._last_signals: Dict[str, List[FreqtradeSignal]] = {}

    async def start(self) -> None:
        """Start the strategy runner."""
        self._running = True
        logger.info(f"Strategy runner started: {self._strategy.name}")

    async def stop(self) -> None:
        """Stop the strategy runner."""
        self._running = False
        logger.info(f"Strategy runner stopped: {self._strategy.name}")

    async def analyze_and_signal(self, pair: str, dataframe: pd.DataFrame) -> List[FreqtradeSignal]:
        """
        Analyze data and generate signals for a pair.

        Args:
            pair: Trading pair
            dataframe: OHLCV data

        Returns:
            List of trading signals
        """
        metadata = {"pair": pair, "timeframe": self._strategy.timeframe}

        # Add indicators
        df_with_indicators = self._strategy.populate_indicators(dataframe, metadata)

        # Generate signals
        signals = self._strategy.generate_signals(df_with_indicators, metadata)
        self._last_signals[pair] = signals

        return signals

    def get_current_signals(self, pair: str) -> List[FreqtradeSignal]:
        """Get the most recent signals for a pair."""
        return self._last_signals.get(pair, [])

    @property
    def strategy(self) -> FreqtradeStrategyWrapper:
        """Get the strategy."""
        return self._strategy

    @property
    def connector(self) -> Any:
        """Get the connector."""
        return self._connector


# Utility functions for strategy creation
def create_strategy_from_template(
    name: str,
    timeframe: str,
    indicators: List[str],
    entry_columns: List[str],
    exit_columns: List[str],
    stoploss: float = -0.10,
    minimal_roi: Optional[Dict[int, float]] = None,
) -> BaseStrategy:
    """
    Create a strategy from a template.

    Args:
        name: Strategy name
        timeframe: Default timeframe
        indicators: List of indicator names to calculate
        entry_columns: Column names that trigger entry
        exit_columns: Column names that trigger exit
        stoploss: Stoploss percentage
        minimal_roi: Minimal ROI table

    Returns:
        A BaseStrategy subclass
    """
    minimal_roi = minimal_roi or {0: 0.05}

    class TemplatedStrategy(BaseStrategy):
        @property
        def name(self) -> str:
            return name

        @property
        def timeframe(self) -> str:
            return timeframe

        def analyze(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
            # Placeholder - implement actual indicators
            return dataframe

        def generate_signals(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> List[FreqtradeSignal]:
            signals = []
            pair = metadata.get("pair", "")

            # Check entry conditions
            for col in entry_columns:
                if col in dataframe.columns:
                    enter_mask = dataframe[col] == 1
                    for idx in dataframe[enter_mask].index:
                        signals.append(FreqtradeSignal(
                            pair=pair,
                            direction="long",
                            price=float(dataframe.loc[idx, "close"]),
                            confidence=1.0,
                            stop_loss=stoploss,
                        ))

            # Check exit conditions
            for col in exit_columns:
                if col in dataframe.columns:
                    exit_mask = dataframe[col] == 1
                    for idx in dataframe[exit_mask].index:
                        signals.append(FreqtradeSignal(
                            pair=pair,
                            direction="exit",
                            price=float(dataframe.loc[idx, "close"]),
                        ))

            return signals

    return TemplatedStrategy()


# Export
__all__ = [
    "FreqtradeSignal",
    "BaseStrategy",
    "FreqtradeStrategyWrapper",
    "IStrategyAdapter",
    "StrategyRunner",
    "create_strategy_from_template",
    "HAS_FREQTRADE",
]
