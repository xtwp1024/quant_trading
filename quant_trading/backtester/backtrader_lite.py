"""
backtrader_lite - 轻量级回测引擎 / Lightweight Backtesting Engine

A pure Python backtesting framework inspired by backtrader, providing:
- CerebroLite: Main backtesting engine (strategy runner, data feed management)
- DataFeed/PandasDataFeed/CSVDataFeed: Data feed interfaces
- BrokerSimulator: Broker simulation with cash, positions, orders
- Analyzer base class + implementations: SharpeRatio, MaxDrawdown, CalmarRatio, AnnualReturn, TradeAnalyzer
- Strategy base class: User strategy interface

纯 Python 实现，无 Cython 依赖。专注于核心回测引擎功能。

Usage / 用法:
```python
import pandas as pd
from quant_trading.backtester.backtrader_lite import (
    CerebroLite, PandasDataFeed, BrokerSimulator,
    Strategy, SharpeRatio, MaxDrawdown, TradeAnalyzer
)

# Prepare data
data = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')

# Create data feed
feed = PandasDataFeed(data)

# Define strategy
class MyStrategy(Strategy):
    params = (('fast_period', 10), ('slow_period', 30))

    def __init__(self):
        self.sma_fast = self.add_indicator('SMA', self.data.close, period=self.p.fast_period)
        self.sma_slow = self.add_indicator('SMA', self.data.close, period=self.p.slow_period)

    def next(self):
        if self.sma_fast > self.sma_slow and not self.get_position():
            self.buy()
        elif self.sma_fast < self.sma_slow and self.get_position():
            self.sell()

# Run backtest
cerebro = CerebroLite(initial_cash=100000)
cerebro.add_data(feed)
cerebro.add_strategy(MyStrategy)
cerebro.add_analyzer(SharpeRatio)
cerebro.add_analyzer(MaxDrawdown)
cerebro.add_analyzer(TradeAnalyzer)

results = cerebro.run()
print(f"Final Value: {results['final_value']:.2f}")
print(f"Sharpe Ratio: {results['analyzers']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['analyzers']['max_drawdown']:.2%}")
```
"""

from __future__ import annotations

import math
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes / 枚举和数据类
# =============================================================================

class OrderType(Enum):
    """Order type / 订单类型"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status / 订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionSide(Enum):
    """Position side / 持仓方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """
    Order representation / 订单表示

    Attributes:
        ref: Unique order reference / 唯一订单引用
        data: Data feed associated with order / 关联的数据源
        order_type: BUY or SELL / 买入或卖出
        size: Order size (positive for buy, negative for sell) / 订单数量
        price: Limit price (None for market order) / 限价（None为市价单）
        status: Current order status / 当前订单状态
        executed_price: Filled price / 成交价格
        executed_size: Filled size / 成交数量
        commission: Commission paid / 手续费
        dt_created: Datetime when order was created / 创建时间
        dt_filled: Datetime when order was filled / 成交时间
    """
    ref: int
    data: 'DataFeed'
    order_type: OrderType
    size: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    executed_price: float = 0.0
    executed_size: float = 0.0
    commission: float = 0.0
    dt_created: Optional[datetime] = None
    dt_filled: Optional[datetime] = None

    def isbuy(self) -> bool:
        """Check if order is a buy order / 检查是否为买入订单"""
        return self.order_type == OrderType.BUY

    def issell(self) -> bool:
        """Check if order is a sell order / 检查是否为卖出订单"""
        return self.order_type == OrderType.SELL

    def __repr__(self) -> str:
        return (f"Order(ref={self.ref}, {self.order_type.value}, "
                f"size={self.size}, price={self.price}, "
                f"status={self.status.value})")


@dataclass
class Position:
    """
    Position representation / 持仓表示

    Attributes:
        data: Data feed associated with position / 关联的数据源
        size: Position size (positive=long, negative=short) / 持仓数量
        avg_price: Average entry price / 平均入场价格
        unrealized_pnl: Unrealized profit/loss / 未实现盈亏
        realized_pnl: Realized profit/loss / 已实现盈亏
    """
    data: 'DataFeed'
    size: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def side(self) -> PositionSide:
        """Get position side / 获取持仓方向"""
        if self.size > 0:
            return PositionSide.LONG
        elif self.size < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    def get_value(self, current_price: float) -> float:
        """Get current position value / 获取当前持仓价值"""
        return self.size * current_price

    def update_price(self, current_price: float) -> None:
        """Update unrealized PnL based on current price / 根据当前价格更新未实现盈亏"""
        if self.avg_price != 0:
            self.unrealized_pnl = (current_price - self.avg_price) * self.size
        else:
            self.unrealized_pnl = 0.0


@dataclass
class Bar:
    """
    OHLCV bar data / OHLCV K线数据

    Attributes:
        datetime: Bar timestamp / K线时间戳
        open: Open price / 开盘价
        high: High price / 最高价
        low: Low price / 最低价
        close: Close price / 收盘价
        volume: Trading volume / 成交量
    """
    datetime: Any
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


# =============================================================================
# Data Feed Classes / 数据源类
# =============================================================================

class LineAccessor:
    """
    Line accessor for backtrader-style data access / backtrader风格的数据访问器

    Provides array-like access to historical values using negative indexing.
    Backtrader-style: feed.close[-1] = current close, feed.close[-2] = previous close

    使用负索引提供历史值的数组式访问。
    backtrader风格：feed.close[-1] = 当前收盘价，feed.close[-2] = 前一根K线收盘价
    """

    def __init__(self, datafeed: 'DataFeed', attr: str):
        """
        Initialize LineAccessor / 初始化访问器

        Args:
            datafeed: Parent DataFeed instance / 父数据源实例
            attr: Attribute name to access ('close', 'open', etc.) / 属性名
        """
        self._datafeed = datafeed
        self._attr = attr

    def __getitem__(self, key: int) -> float:
        """
        Get value by offset from current bar / 从当前K线按偏移获取值

        Args:
            key: Offset (e.g., -1 = current bar, -2 = previous bar) / 偏移量

        Returns:
            Value at the specified offset / 指定偏移量的值
        """
        datafeed = self._datafeed
        attr = self._attr

        if key < 0:
            # Negative index: offset back from current bar
            idx = datafeed._current_idx + key
        else:
            # Positive index: offset forward from start of data
            idx = key

        if 0 <= idx < len(datafeed._bars):
            bar = datafeed._bars[idx]
            return getattr(bar, attr, 0.0)
        return 0.0

    def __call__(self) -> float:
        """
        Get current value (same as [-1]) / 获取当前值（等同于[-1]）

        Returns:
            Current value / 当前值
        """
        return self.__getitem__(-1)

    def __len__(self) -> int:
        """Get number of bars / 获取K线数量"""
        return len(self._datafeed._bars)

    def __iter__(self):
        """Iterate over all values / 迭代所有值"""
        for i in range(len(self._datafeed._bars)):
            yield self.__getitem__(-(len(self._datafeed._bars) - i))

    def __repr__(self) -> str:
        return f"LineAccessor({self._attr})"


class DataFeed(ABC):
    """
    Abstract base class for data feeds / 数据源抽象基类

    All data feeds must implement this interface to provide
    OHLCV data to the backtesting engine.

    所有数据源必须实现此接口，向回测引擎提供OHLCV数据。

    Attributes:
        name: Name identifier for the data feed / 数据源名称
        params: Data feed parameters / 数据源参数
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Initialize data feed / 初始化数据源

        Args:
            name: Optional name identifier / 可选名称标识符
            **kwargs: Additional parameters / 额外参数
        """
        self.name = name or self.__class__.__name__
        self.params = kwargs
        self._bars: List[Bar] = []
        self._current_idx: int = 0
        self._started: bool = False

    @abstractmethod
    def load(self) -> None:
        """
        Load data into the feed / 将数据加载到数据源中

        Must be implemented by subclasses.
        必须由子类实现。
        """
        pass

    def start(self) -> None:
        """Start the data feed / 启动数据源"""
        if not self._bars:
            self.load()
        self._started = True
        self._current_idx = 0

    def stop(self) -> None:
        """Stop the data feed / 停止数据源"""
        self._started = False

    def reset(self) -> None:
        """Reset the data feed to beginning / 重置数据源到开始位置"""
        self._current_idx = 0

    @property
    def datetime(self) -> Any:
        """Get current bar datetime / 获取当前K线时间"""
        return self._bars[self._current_idx].datetime if self._bars else None

    @property
    def open(self) -> LineAccessor:
        """Get open price accessor (supports [-i] indexing) / 获取开盘价访问器"""
        return LineAccessor(self, 'open')

    @property
    def high(self) -> LineAccessor:
        """Get high price accessor (supports [-i] indexing) / 获取最高价访问器"""
        return LineAccessor(self, 'high')

    @property
    def low(self) -> LineAccessor:
        """Get low price accessor (supports [-i] indexing) / 获取最低价访问器"""
        return LineAccessor(self, 'low')

    @property
    def close(self) -> LineAccessor:
        """Get close price accessor (supports [-i] indexing) / 获取收盘价访问器"""
        return LineAccessor(self, 'close')

    @property
    def volume(self) -> LineAccessor:
        """Get volume accessor (supports [-i] indexing) / 获取成交量访问器"""
        return LineAccessor(self, 'volume')

    @property
    def openinterest(self) -> float:
        """Get current bar open interest / 获取当前K线持仓量"""
        return 0.0  # Default implementation / 默认实现

    def __len__(self) -> int:
        """Get number of bars / 获取K线数量"""
        return len(self._bars)

    def __getitem__(self, key: int) -> float:
        """
        Get value by index (0=oldest, -1=latest) / 通过索引获取值

        Args:
            key: Index (0 for oldest bar, -1 for latest bar) / 索引

        Returns:
            Close price at the given index / 给定索引的收盘价
        """
        idx = self._current_idx + key if key < 0 else key
        if 0 <= idx < len(self._bars):
            return self._bars[idx].close
        return 0.0

    def forward(self) -> bool:
        """
        Move to next bar / 移动到下一根K线

        Returns:
            True if there are more bars, False otherwise / 如果还有K线返回True，否则返回False
        """
        if self._current_idx < len(self._bars) - 1:
            self._current_idx += 1
            return True
        return False

    def backward(self) -> bool:
        """
        Move to previous bar / 移动到上一根K线

        Returns:
            True if there are more bars, False otherwise / 如果还有K线返回True，否则返回False
        """
        if self._current_idx > 0:
            self._current_idx -= 1
            return True
        return False

    def set_current(self, idx: int) -> None:
        """
        Set current bar index / 设置当前K线索引

        Args:
            idx: Index to set / 要设置的索引
        """
        if 0 <= idx < len(self._bars):
            self._current_idx = idx

    def get_bar(self, idx: int) -> Optional[Bar]:
        """
        Get bar by index / 通过索引获取K线

        Args:
            idx: Bar index / K线索引

        Returns:
            Bar at the given index or None / 给定索引的K线，如果不存在则返回None
        """
        if 0 <= idx < len(self._bars):
            return self._bars[idx]
        return None


class PandasDataFeed(DataFeed):
    """
    Data feed from pandas DataFrame / 从pandas DataFrame创建的数据源

    Accepts DataFrames with datetime index or 'date'/'datetime' column,
    and standard OHLCV columns.

    支持带有datetime索引或'date'/'datetime'列的DataFrame，以及标准OHLCV列。

    Required columns / 必需列:
        - date/datetime (index or column): Datetime values / 日期时间值
        - open (optional): Open price / 开盘价
        - high (optional): High price / 最高价
        - low (optional): Low price / 最低价
        - close: Close price / 收盘价
        - volume (optional): Volume / 成交量

    Example / 示例:
    ```python
    df = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
    feed = PandasDataFeed(df, name='AAPL')
    ```
    """

    # Standard column names / 标准列名
    STANDARD_COLS = {
        'date': ['date', 'datetime', 'timestamp', 'time'],
        'open': ['open', 'o'],
        'high': ['high', 'h'],
        'low': ['low', 'l'],
        'close': ['close', 'c'],
        'volume': ['volume', 'vol', 'v'],
    }

    def __init__(
        self,
        data: pd.DataFrame,
        name: Optional[str] = None,
        datetime_col: Optional[str] = None,
        open_col: Optional[str] = None,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        close_col: Optional[str] = None,
        volume_col: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize PandasDataFeed / 初始化Pandas数据源

        Args:
            data: DataFrame with OHLCV data / 包含OHLCV数据的DataFrame
            name: Optional name identifier / 可选名称标识符
            datetime_col: Name of datetime column (if not index) / 日期时间列名
            open_col: Name of open column / 开盘价列名
            high_col: Name of high column / 最高价列名
            low_col: Name of low column / 最低价列名
            close_col: Name of close column / 收盘价列名
            volume_col: Name of volume column / 成交量列名
            **kwargs: Additional parameters passed to parent / 传递给父类的额外参数
        """
        super().__init__(name=name, **kwargs)
        self.data = data.copy()

        # Detect or use specified column names / 检测或使用指定的列名
        self.datetime_col = datetime_col or self._find_column(
            data, self.STANDARD_COLS['date'], data.index.name
        )
        self.open_col = open_col or self._find_column(data, self.STANDARD_COLS['open'])
        self.high_col = high_col or self._find_column(data, self.STANDARD_COLS['high'])
        self.low_col = low_col or self._find_column(data, self.STANDARD_COLS['low'])
        self.close_col = close_col or self._find_column(data, self.STANDARD_COLS['close'])
        self.volume_col = volume_col or self._find_column(data, self.STANDARD_COLS['volume'])

        self._bars: List[Bar] = []

    def _find_column(self, df: pd.DataFrame, candidates: List[str], fallback: Optional[str] = None) -> Optional[str]:
        """Find first matching column name / 查找第一个匹配的列名"""
        # Check if any candidate is a dataframe column / 检查候选项是否为DataFrame列
        for col in candidates:
            if col in df.columns:
                return col
        # Return fallback if no match / 如果没有匹配则返回fallback
        return fallback

    def load(self) -> None:
        """Load data from DataFrame into bars / 从DataFrame加载数据到K线"""
        self._bars = []

        # Get datetime values / 获取日期时间值
        if self.datetime_col and self.datetime_col in self.data.columns:
            dates = pd.to_datetime(self.data[self.datetime_col])
        else:
            dates = pd.to_datetime(self.data.index)

        # Extract OHLCV columns / 提取OHLCV列
        open_prices = self._get_column(self.data, self.open_col, 'open')
        high_prices = self._get_column(self.data, self.high_col, 'high')
        low_prices = self._get_column(self.data, self.low_col, 'low')
        close_prices = self._get_column(self.data, self.close_col, 'close')
        volumes = self._get_column(self.data, self.volume_col, 'volume', default=0.0)

        # Handle missing OHLC by using close / 通过使用收盘价处理缺失的OHLC
        if open_prices is None:
            open_prices = close_prices.copy() if close_prices is not None else pd.Series([0.0] * len(dates))
        if high_prices is None:
            high_prices = close_prices.copy() if close_prices is not None else pd.Series([0.0] * len(dates))
        if low_prices is None:
            low_prices = close_prices.copy() if close_prices is not None else pd.Series([0.0] * len(dates))

        # Create bars / 创建K线
        for i in range(len(dates)):
            bar = Bar(
                datetime=dates.iloc[i] if hasattr(dates, 'iloc') else dates[i],
                open=float(open_prices.iloc[i]) if open_prices is not None else 0.0,
                high=float(high_prices.iloc[i]) if high_prices is not None else 0.0,
                low=float(low_prices.iloc[i]) if low_prices is not None else 0.0,
                close=float(close_prices.iloc[i]) if close_prices is not None else 0.0,
                volume=float(volumes.iloc[i]) if volumes is not None else 0.0,
            )
            self._bars.append(bar)

        logger.debug(f"PandasDataFeed '{self.name}' loaded {len(self._bars)} bars")

    def _get_column(self, df: pd.DataFrame, col_name: Optional[str], default_key: str, default: Any = None) -> Optional[pd.Series]:
        """Get column or return default / 获取列或返回默认值"""
        if col_name and col_name in df.columns:
            return df[col_name]
        default_col = self._find_column(df, self.STANDARD_COLS.get(default_key, [default_key]))
        if default_col and default_col in df.columns:
            return df[default_col]
        return default if default is not None else None


class CSVDataFeed(DataFeed):
    """
    Data feed from CSV file / 从CSV文件创建的数据源

    Reads OHLCV data from CSV files with automatic column detection.

    从具有自动列检测的CSV文件读取OHLCV数据。

    Example / 示例:
    ```python
    feed = CSVDataFeed('data.csv', name='AAPL')
    # or with custom columns / 或自定义列
    feed = CSVDataFeed('data.csv', name='AAPL', datetime_col='date',
                        open_col='open', high_col='high', low_col='low',
                        close_col='close', volume_col='volume')
    ```
    """

    def __init__(
        self,
        filepath: str,
        name: Optional[str] = None,
        datetime_col: Optional[str] = None,
        open_col: Optional[str] = None,
        high_col: Optional[str] = None,
        low_col: Optional[str] = None,
        close_col: Optional[str] = None,
        volume_col: Optional[str] = None,
        date_format: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize CSVDataFeed / 初始化CSV数据源

        Args:
            filepath: Path to CSV file / CSV文件路径
            name: Optional name identifier / 可选名称标识符
            datetime_col: Name of datetime column / 日期时间列名
            open_col: Name of open column / 开盘价列名
            high_col: Name of high column / 最高价列名
            low_col: Name of low column / 最低价列名
            close_col: Name of close column / 收盘价列名
            volume_col: Name of volume column / 成交量列名
            date_format: Datetime parse format / 日期时间解析格式
            **kwargs: Additional parameters / 额外参数
        """
        super().__init__(name=name, **kwargs)
        self.filepath = filepath
        self.datetime_col = datetime_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.date_format = date_format
        self._bars: List[Bar] = []

    def load(self) -> None:
        """Load data from CSV file / 从CSV文件加载数据"""
        # Read CSV file / 读取CSV文件
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file {self.filepath}: {e}")

        # If datetime_col specified, parse it / 如果指定了datetime_col，则解析它
        if self.datetime_col and self.datetime_col in df.columns:
            if self.date_format:
                df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], format=self.date_format)
            else:
                df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
            df = df.set_index(self.datetime_col)

        # Auto-detect column names / 自动检测列名
        cols_lower = {c.lower(): c for c in df.columns}

        def get_col(*names) -> Optional[str]:
            for n in names:
                if n.lower() in cols_lower:
                    return cols_lower[n.lower()]
            return None

        open_c = self.open_col or get_col('open', 'o')
        high_c = self.high_col or get_col('high', 'h')
        low_c = self.low_col or get_col('low', 'l')
        close_c = self.close_col or get_col('close', 'c')
        vol_c = self.volume_col or get_col('volume', 'vol', 'v')

        # Get index as datetime if it's the index / 如果是索引则获取索引作为日期时间
        if df.index.name:
            dates = pd.to_datetime(df.index)
        elif self.datetime_col is None:
            # Try to parse first column as datetime / 尝试将第一列解析为日期时间
            first_col = df.columns[0]
            dates = pd.to_datetime(df[first_col], format=self.date_format, errors='coerce')
            if dates.isna().all():
                # Use range as fallback / 使用范围作为后备
                dates = pd.RangeIndex(start=0, stop=len(df), step=1)
        else:
            dates = pd.RangeIndex(start=0, stop=len(df), step=1)

        # Extract OHLCV / 提取OHLCV
        opens = df[open_c].astype(float) if open_c else df[close_c].astype(float) if close_c else pd.Series([0.0] * len(df))
        highs = df[high_c].astype(float) if high_c else df[close_c].astype(float) if close_c else pd.Series([0.0] * len(df))
        lows = df[low_c].astype(float) if low_c else df[close_c].astype(float) if close_c else pd.Series([0.0] * len(df))
        closes = df[close_c].astype(float) if close_c else pd.Series([0.0] * len(df))
        vols = df[vol_c].astype(float) if vol_c else pd.Series([0.0] * len(df))

        # Create bars / 创建K线
        self._bars = []
        for i in range(len(dates)):
            bar = Bar(
                datetime=dates.iloc[i] if hasattr(dates, 'iloc') else dates[i],
                open=float(opens.iloc[i]),
                high=float(highs.iloc[i]),
                low=float(lows.iloc[i]),
                close=float(closes.iloc[i]),
                volume=float(vols.iloc[i]),
            )
            self._bars.append(bar)

        logger.debug(f"CSVDataFeed '{self.name}' loaded {len(self._bars)} bars from {self.filepath}")


# =============================================================================
# Broker Simulator / 经纪商模拟器
# =============================================================================

class BrokerSimulator:
    """
    Broker simulator for backtesting / 回测用经纪商模拟器

    Simulates broker functionality including:
    - Cash management / 现金管理
    - Position tracking / 持仓追踪
    - Order execution / 订单执行
    - Commission calculation / 手续费计算

    Example / 示例:
    ```python
    broker = BrokerSimulator(initial_cash=100000, commission=0.001)
    broker.setcommission(commission=0.0005)  # Update commission rate / 更新手续费率
    ```
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
    ):
        """
        Initialize broker / 初始化经纪商

        Args:
            initial_cash: Starting cash amount / 起始资金
            commission: Commission rate (e.g., 0.001 = 0.1%) / 手续费率
            slippage: Slippage rate (e.g., 0.001 = 0.1%) / 滑点率
        """
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._commission = commission
        self._slippage = slippage
        self._positions: Dict[str, Position] = {}  # data_name -> Position
        self._orders: List[Order] = []
        self._order_ref = 0
        self._value_history: List[float] = []

    @property
    def cash(self) -> float:
        """Get current cash / 获取当前现金"""
        return self._cash

    @property
    def value(self) -> float:
        """Get total portfolio value (cash + positions) / 获取总资产价值"""
        total = self._cash
        for pos in self._positions.values():
            total += abs(pos.size * pos.avg_price) if pos.size != 0 else 0
        return total

    @property
    def positions(self) -> Dict[str, Position]:
        """Get all positions / 获取所有持仓"""
        return self._positions.copy()

    @property
    def orders(self) -> List[Order]:
        """Get all orders / 获取所有订单"""
        return self._orders.copy()

    def getposition(self, data: DataFeed) -> Position:
        """
        Get position for a data feed / 获取特定数据源的持仓

        Args:
            data: Data feed to get position for / 要获取持仓的数据源

        Returns:
            Position object / 持仓对象
        """
        if data.name not in self._positions:
            self._positions[data.name] = Position(data=data)
        return self._positions[data.name]

    def setcash(self, cash: float) -> None:
        """
        Set cash amount / 设置现金金额

        Args:
            cash: New cash amount / 新现金金额
        """
        self._cash = cash
        logger.debug(f"Broker cash set to {cash}")

    def setcommission(self, commission: float = 0.0) -> None:
        """
        Set commission rate / 设置手续费率

        Args:
            commission: Commission rate (e.g., 0.001 = 0.1%) / 手续费率
        """
        self._commission = commission
        logger.debug(f"Broker commission set to {commission}")

    def setslippage(self, slippage: float = 0.0) -> None:
        """
        Set slippage rate / 设置滑点率

        Args:
            slippage: Slippage rate (e.g., 0.001 = 0.1%) / 滑点率
        """
        self._slippage = slippage
        logger.debug(f"Broker slippage set to {slippage}")

    def buy(
        self,
        data: DataFeed,
        size: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Place a buy order / 下买入订单

        Args:
            data: Data feed to buy / 要买入的数据源
            size: Number of units to buy / 买入数量
            price: Limit price (None for market order) / 限价（None为市价单）
            **kwargs: Additional order parameters / 额外订单参数

        Returns:
            Order object / 订单对象
        """
        self._order_ref += 1
        order = Order(
            ref=self._order_ref,
            data=data,
            order_type=OrderType.BUY,
            size=size,
            price=price,
            dt_created=datetime.now(),
        )
        self._orders.append(order)
        logger.debug(f"BUY order created: {order}")
        return order

    def sell(
        self,
        data: DataFeed,
        size: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Place a sell order / 下卖出订单

        Args:
            data: Data feed to sell / 要卖出的数据源
            size: Number of units to sell / 卖出数量
            price: Limit price (None for market order) / 限价（None为市价单）
            **kwargs: Additional order parameters / 额外订单参数

        Returns:
            Order object / 订单对象
        """
        self._order_ref += 1
        order = Order(
            ref=self._order_ref,
            data=data,
            order_type=OrderType.SELL,
            size=size,
            price=price,
            dt_created=datetime.now(),
        )
        self._orders.append(order)
        logger.debug(f"SELL order created: {order}")
        return order

    def close(self, data: DataFeed) -> Order:
        """
        Close position on a data feed / 平掉特定数据源的持仓

        Args:
            data: Data feed to close position for / 要平仓的数据源

        Returns:
            Order object / 订单对象
        """
        pos = self.getposition(data)
        if pos.size == 0:
            raise ValueError(f"No position to close for {data.name}")

        order_type = OrderType.SELL if pos.size > 0 else OrderType.BUY
        self._order_ref += 1
        order = Order(
            ref=self._order_ref,
            data=data,
            order_type=order_type,
            size=abs(pos.size),
            dt_created=datetime.now(),
        )
        self._orders.append(order)
        logger.debug(f"CLOSE order created: {order}")
        return order

    def execute_order(self, order: Order, current_price: float) -> Order:
        """
        Execute an order at given price / 以给定价格执行订单

        Args:
            order: Order to execute / 要执行的订单
            current_price: Current market price / 当前市场价格

        Returns:
            Executed order / 已执行的订单
        """
        if order.status != OrderStatus.PENDING:
            return order

        # Apply slippage / 应用滑点
        exec_price = current_price
        if order.isbuy():
            exec_price *= (1 + self._slippage)
        else:
            exec_price *= (1 - self._slippage)

        # Calculate commission / 计算手续费
        trade_value = abs(exec_price * order.size)
        commission = trade_value * self._commission

        # Check if enough cash for buy / 检查买入是否有足够现金
        if order.isbuy():
            required_cash = trade_value + commission
            if self._cash < required_cash:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected: insufficient cash. Required: {required_cash}, Available: {self._cash}")
                return order

        # Update position / 更新持仓
        pos = self.getposition(order.data)

        if order.isbuy():
            # Calculate new average price / 计算新的平均价格
            total_cost = pos.size * pos.avg_price + order.size * exec_price
            pos.size += order.size
            if pos.size > 0:
                pos.avg_price = total_cost / pos.size
            pos.update_price(exec_price)
            self._cash -= (trade_value + commission)
        else:
            # Sell / 卖出
            if abs(order.size) > abs(pos.size):
                # Closing long position / 平多仓
                realized_pnl = (exec_price - pos.avg_price) * pos.size
                pos.realized_pnl += realized_pnl
                self._cash += (trade_value - commission + realized_pnl)
                pos.size = 0
                pos.avg_price = 0.0
            else:
                # Partial sell or short sell / 部分卖出或做空卖出
                realized_pnl = (exec_price - pos.avg_price) * order.size
                pos.realized_pnl += realized_pnl
                pos.size -= order.size
                self._cash += (trade_value - commission)
            pos.update_price(exec_price)

        # Update order / 更新订单
        order.status = OrderStatus.FILLED
        order.executed_price = exec_price
        order.executed_size = order.size
        order.commission = commission
        order.dt_filled = datetime.now()

        logger.debug(f"Order executed: {order}")
        return order

    def get_value_history(self) -> List[float]:
        """Get portfolio value history / 获取资产价值历史"""
        return self._value_history.copy()

    def record_value(self) -> None:
        """Record current portfolio value / 记录当前资产价值"""
        self._value_history.append(self.value)


# =============================================================================
# Strategy Base Class / 策略基类
# =============================================================================

class Strategy:
    """
    Base class for trading strategies / 交易策略基类

    User strategies should inherit from this class and implement:
    - __init__: Initialize indicators and variables
    - next: Trading logic called on each bar

    用户策略应继承此类并实现：
    - __init__：初始化指标和变量
    - next：每根K线调用的交易逻辑

    Attributes:
        cerebro: Reference to Cerebro instance / Cerebro实例引用
        params: Strategy parameters / 策略参数
        datas: List of data feeds / 数据源列表
        broker: Broker simulator / 经纪商模拟器
        orders: List of orders / 订单列表

    Example / 示例:
    ```python
    class MyStrategy(Strategy):
        params = (('period', 20),)

        def __init__(self):
            self.sma = self.add_indicator('SMA', self.data.close, period=self.p.period)

        def next(self):
            if self.data.close > self.sma and not self.get_position():
                self.buy()
    ```
    """

    params: Tuple[Tuple[str, Any], ...] = ()

    def __init__(self, cerebro: Optional['CerebroLite'] = None):
        """
        Initialize strategy / 初始化策略

        Args:
            cerebro: Reference to Cerebro instance / Cerebro实例引用
        """
        self.cerebro = cerebro
        self.datas: List[DataFeed] = []
        self.broker: Optional[BrokerSimulator] = None
        self._indicators: Dict[str, Any] = {}
        self._order_callbacks: List[Callable[[Order], None]] = []

    def add_indicator(self, indicator_type: str, *args, **kwargs) -> Any:
        """
        Add an indicator to the strategy / 向策略添加指标

        Args:
            indicator_type: Type of indicator (SMA, EMA, etc.) / 指标类型
            *args: Positional arguments for indicator / 指标的位置参数
            **kwargs: Keyword arguments for indicator / 指标的关键字参数

        Returns:
            Indicator instance / 指标实例
        """
        indicator = self._create_indicator(indicator_type, *args, **kwargs)
        self._indicators[indicator_type] = indicator
        return indicator

    def _create_indicator(self, indicator_type: str, *args, **kwargs) -> Any:
        """
        Create an indicator / 创建指标

        Args:
            indicator_type: Type of indicator / 指标类型
            *args: Positional arguments / 位置参数
            **kwargs: Keyword arguments / 关键字参数

        Returns:
            Indicator value (currently returns close price for simple indicators) / 指标值
        """
        # Simple implementation - return close price for now
        # In a full implementation, this would create actual indicators
        if args:
            return args[0]  # Return first argument as indicator value
        return 0.0

    @property
    def data(self) -> DataFeed:
        """Get primary data feed / 获取主数据源"""
        return self.datas[0] if self.datas else None

    @property
    def data0(self) -> DataFeed:
        """Get data feed by index 0 / 获取索引0的数据源"""
        return self.datas[0] if self.datas else None

    def datai(self, i: int) -> DataFeed:
        """
        Get data feed by index / 通过索引获取数据源

        Args:
            i: Index of data feed / 数据源索引

        Returns:
            Data feed at index i / 索引i的数据源
        """
        return self.datas[i] if i < len(self.datas) else None

    def get_position(self, data: Optional[DataFeed] = None) -> Position:
        """
        Get current position / 获取当前持仓

        Args:
            data: Optional data feed / 可选数据源

        Returns:
            Position object / 持仓对象
        """
        if data is None:
            data = self.data
        if self.broker:
            return self.broker.getposition(data)
        return Position(data=data)

    def buy(
        self,
        data: Optional[DataFeed] = None,
        size: Optional[float] = None,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Place a buy order / 下买入订单

        Args:
            data: Data feed to buy (default: primary data) / 要买入的数据源
            size: Number of units to buy / 买入数量
            price: Limit price / 限价
            **kwargs: Additional parameters / 额外参数

        Returns:
            Order object / 订单对象
        """
        if data is None:
            data = self.data
        if self.broker and size is not None:
            return self.broker.buy(data, size, price, **kwargs)
        return Order(ref=0, data=data, order_type=OrderType.BUY, size=size or 0)

    def sell(
        self,
        data: Optional[DataFeed] = None,
        size: Optional[float] = None,
        price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Place a sell order / 下卖出订单

        Args:
            data: Data feed to sell (default: primary data) / 要卖出的数据源
            size: Number of units to sell / 卖出数量
            price: Limit price / 限价
            **kwargs: Additional parameters / 额外参数

        Returns:
            Order object / 订单对象
        """
        if data is None:
            data = self.data
        if self.broker and size is not None:
            return self.broker.sell(data, size, price, **kwargs)
        return Order(ref=0, data=data, order_type=OrderType.SELL, size=size or 0)

    def close(self, data: Optional[DataFeed] = None, **kwargs) -> Order:
        """
        Close position / 平仓

        Args:
            data: Data feed to close position for / 要平仓的数据源
            **kwargs: Additional parameters / 额外参数

        Returns:
            Order object / 订单对象
        """
        if data is None:
            data = self.data
        if self.broker:
            return self.broker.close(data)
        return Order(ref=0, data=data, order_type=OrderType.SELL, size=0)

    def notify_order(self, order: Order) -> None:
        """
        Callback when order is executed / 订单执行时的回调

        Override this method to handle order notifications.

        Args:
            order: Executed order / 已执行的订单
        """
        pass

    def notify_trade(self, trade: Dict[str, Any]) -> None:
        """
        Callback when trade is completed / 交易完成时的回调

        Override this method to handle trade notifications.

        Args:
            trade: Trade information dictionary / 交易信息字典
        """
        pass

    def __init_subclass__(cls, **kwargs):
        """Process params attribute from parent class / 从父类处理params属性"""
        super().__init_subclass__(**kwargs)
        # Collect params from parent classes / 从父类收集params
        base_params = {}
        for parent in cls.__mro__[1:]:
            if hasattr(parent, 'params'):
                base_params.update(dict(parent.params))
        # Merge with class's own params / 与类的自身params合并
        if hasattr(cls, 'params'):
            base_params.update(dict(cls.params))
        cls.params = tuple(base_params.items())

    @property
    def p(self) -> Any:
        """
        Get params as namespace / 获取params作为命名空间

        Returns:
            Object with params as attributes / params作为属性的对象
        """
        class Params:
            def __init__(self, params):
                for k, v in params:
                    setattr(self, k, v)
        return Params(self.params)


# =============================================================================
# Analyzer Base Class and Implementations / 分析器基类和实现
# =============================================================================

class Analyzer:
    """
    Base class for analyzers / 分析器基类

    Analyzers compute metrics on backtest results such as:
    - Sharpe ratio / 夏普比率
    - Maximum drawdown / 最大回撤
    - Trade statistics / 交易统计

    Users can create custom analyzers by inheriting from this class.

    用户可以通过继承此类创建自定义分析器。

    Attributes:
        strategy: Reference to strategy / 策略引用
    """

    def __init__(self, strategy: Optional[Strategy] = None):
        """
        Initialize analyzer / 初始化分析器

        Args:
            strategy: Reference to strategy / 策略引用
        """
        self.strategy = strategy
        self._results: Dict[str, Any] = {}

    def start(self) -> None:
        """
        Called at start of backtest / 回测开始时调用

        Override to perform initialization.
        """
        pass

    def stop(self) -> None:
        """
        Called at end of backtest / 回测结束时调用

        Override to perform final calculations.
        """
        pass

    def next(self) -> None:
        """
        Called on each bar / 每根K线调用

        Override to update analyzer state.
        """
        pass

    def get_analysis(self) -> Dict[str, Any]:
        """
        Get analysis results / 获取分析结果

        Returns:
            Dictionary of analysis results / 分析结果字典
        """
        return self._results.copy()


class SharpeRatio(Analyzer):
    """
    Sharpe Ratio Analyzer / 夏普比率分析器

    Calculates the Sharpe ratio of returns.
    Formula: (mean_return - risk_free_rate) / std_return

    计算收益率的夏普比率。
    公式：(平均收益率 - 无风险利率) / 收益率标准差

    Attributes:
        risk_free_rate: Annual risk-free rate (default: 0.02) / 年无风险利率
        annualization_factor: Factor to annualize returns (default: 252) / 年化因子
    """

    def __init__(
        self,
        strategy: Optional[Strategy] = None,
        risk_free_rate: float = 0.02,
        annualization_factor: float = 252.0,
    ):
        """
        Initialize SharpeRatio analyzer / 初始化夏普比率分析器

        Args:
            strategy: Reference to strategy / 策略引用
            risk_free_rate: Annual risk-free rate / 年无风险利率
            annualization_factor: Trading days per year / 年交易日数
        """
        super().__init__(strategy)
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self._returns: List[float] = []
        self._values: List[float] = []

    def start(self) -> None:
        """Reset and initialize / 重置和初始化"""
        self._returns = []
        self._values = []

    def next(self) -> None:
        """Record current value / 记录当前价值"""
        if self.strategy and self.strategy.broker:
            self._values.append(self.strategy.broker.value)

            # Calculate return / 计算收益率
            if len(self._values) > 1:
                ret = (self._values[-1] - self._values[-2]) / self._values[-2]
                self._returns.append(ret)

    def get_analysis(self) -> Dict[str, Any]:
        """Calculate and return Sharpe ratio / 计算并返回夏普比率"""
        if len(self._returns) < 2:
            self._results = {"sharpe_ratio": 0.0, "annual_return": 0.0, "annual_volatility": 0.0}
            return self._results

        import numpy as np
        returns = np.array(self._returns)

        # Annualized metrics / 年化指标
        annual_return = np.mean(returns) * self.annualization_factor
        annual_vol = np.std(returns, ddof=1) * np.sqrt(self.annualization_factor)

        # Sharpe ratio / 夏普比率
        sharpe_ratio = 0.0
        if annual_vol > 0:
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol

        self._results = {
            "sharpe_ratio": float(sharpe_ratio),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_vol),
            "daily_return_mean": float(np.mean(returns)),
            "daily_return_std": float(np.std(returns, ddof=1)),
            "num_days": len(returns),
        }
        return self._results


class MaxDrawdown(Analyzer):
    """
    Maximum Drawdown Analyzer / 最大回撤分析器

    Calculates maximum drawdown and related metrics.
    Formula: max((peak - trough) / peak)

    计算最大回撤及相关指标。
    公式：max((峰值 - 谷值) / 峰值)

    Attributes:
        drawdown_values: List of drawdown values / 回撤值列表
    """

    def __init__(self, strategy: Optional[Strategy] = None):
        """
        Initialize MaxDrawdown analyzer / 初始化最大回撤分析器

        Args:
            strategy: Reference to strategy / 策略引用
        """
        super().__init__(strategy)
        self._values: List[float] = []
        self._peak: float = 0.0
        self._current_drawdown: float = 0.0

    def start(self) -> None:
        """Reset and initialize / 重置和初始化"""
        self._values = []
        self._peak = 0.0
        self._current_drawdown = 0.0

    def next(self) -> None:
        """Update drawdown / 更新回撤"""
        if self.strategy and self.strategy.broker:
            value = self.strategy.broker.value
            self._values.append(value)

            if value > self._peak:
                self._peak = value

            if self._peak > 0:
                self._current_drawdown = (self._peak - value) / self._peak

    def get_analysis(self) -> Dict[str, Any]:
        """Calculate and return maximum drawdown / 计算并返回最大回撤"""
        if not self._values:
            self._results = {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}
            return self._results

        # Calculate drawdown series / 计算回撤序列
        peak = 0.0
        max_dd = 0.0
        max_dd_pct = 0.0

        for value in self._values:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd

        self._results = {
            "max_drawdown": float(max_dd * (self._values[0] if self._values else 100000)),
            "max_drawdown_pct": float(max_dd_pct),
            "peak_value": float(peak) if 'peak' in locals() else 0.0,
            "final_value": float(self._values[-1]) if self._values else 0.0,
            "num_days": len(self._values),
        }
        return self._results


class CalmarRatio(Analyzer):
    """
    Calmar Ratio Analyzer / 卡尔玛比率分析器

    Calculates Calmar ratio: annual return / max drawdown.
    A higher ratio indicates better risk-adjusted performance.

    计算卡尔玛比率：年化收益 / 最大回撤。
    比率越高表示风险调整后的表现越好。

    Attributes:
        annualization_factor: Factor to annualize returns / 年化因子
    """

    def __init__(
        self,
        strategy: Optional[Strategy] = None,
        annualization_factor: float = 252.0,
    ):
        """
        Initialize CalmarRatio analyzer / 初始化卡尔玛比率分析器

        Args:
            strategy: Reference to strategy / 策略引用
            annualization_factor: Trading days per year / 年交易日数
        """
        super().__init__(strategy)
        self.annualization_factor = annualization_factor
        self._values: List[float] = []
        self._returns: List[float] = []

    def start(self) -> None:
        """Reset and initialize / 重置和初始化"""
        self._values = []
        self._returns = []

    def next(self) -> None:
        """Record current value / 记录当前价值"""
        if self.strategy and self.strategy.broker:
            value = self.strategy.broker.value
            self._values.append(value)

            if len(self._values) > 1:
                ret = (value - self._values[-2]) / self._values[-2]
                self._returns.append(ret)

    def get_analysis(self) -> Dict[str, Any]:
        """Calculate and return Calmar ratio / 计算并返回卡尔玛比率"""
        if len(self._returns) < 2:
            self._results = {"calmar_ratio": 0.0, "annual_return": 0.0, "max_drawdown_pct": 0.0}
            return self._results

        import numpy as np

        # Annual return / 年化收益
        annual_return = np.mean(self._returns) * self.annualization_factor

        # Max drawdown / 最大回撤
        peak = 0.0
        max_dd = 0.0
        for value in self._values:
            if value > peak:
                peak = value
            if peak > 0:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd

        # Calmar ratio / 卡尔玛比率
        calmar_ratio = 0.0
        if max_dd > 0:
            calmar_ratio = annual_return / max_dd

        self._results = {
            "calmar_ratio": float(calmar_ratio),
            "annual_return": float(annual_return),
            "max_drawdown_pct": float(max_dd),
        }
        return self._results


class AnnualReturn(Analyzer):
    """
    Annual Return Analyzer / 年化收益分析器

    Calculates annual return for each year and summary statistics.

    计算每年收益和摘要统计。
    """

    def __init__(self, strategy: Optional[Strategy] = None):
        """
        Initialize AnnualReturn analyzer / 初始化年化收益分析器

        Args:
            strategy: Reference to strategy / 策略引用
        """
        super().__init__(strategy)
        self._values: List[Tuple[Any, float]] = []  # (datetime, value)

    def start(self) -> None:
        """Reset and initialize / 重置和初始化"""
        self._values = []

    def next(self) -> None:
        """Record current value with datetime / 记录当前价值和时间"""
        if self.strategy and self.strategy.broker:
            data = self.strategy.datas[0] if self.strategy.datas else None
            dt = data.datetime if data else None
            self._values.append((dt, self.strategy.broker.value))

    def get_analysis(self) -> Dict[str, Any]:
        """Calculate and return annual returns / 计算并返回年化收益"""
        if len(self._values) < 2:
            self._results = {"annual_returns": {}, "total_return": 0.0, "annual_return_mean": 0.0}
            return self._results

        # Group values by year / 按年份分组
        yearly_values: Dict[int, List[float]] = {}
        for dt, value in self._values:
            if dt is not None:
                year = dt.year if hasattr(dt, 'year') else int(dt)
                if year not in yearly_values:
                    yearly_values[year] = []
                yearly_values[year].append(value)

        # Calculate annual returns / 计算年化收益
        annual_returns = {}
        years = sorted(yearly_values.keys())

        for i, year in enumerate(years):
            values = yearly_values[year]
            if len(values) >= 2:
                year_return = (values[-1] - values[0]) / values[0]
                annual_returns[str(year)] = float(year_return)

        # Calculate summary / 计算摘要
        if annual_returns:
            returns_list = list(annual_returns.values())
            total_return = (self._values[-1][1] - self._values[0][1]) / self._values[0][1]
            mean_return = sum(returns_list) / len(returns_list)
        else:
            total_return = 0.0
            mean_return = 0.0

        self._results = {
            "annual_returns": annual_returns,
            "total_return": float(total_return),
            "total_return_pct": float(total_return * 100),
            "annual_return_mean": float(mean_return),
            "num_years": len(annual_returns),
        }
        return self._results


class TradeAnalyzer(Analyzer):
    """
    Trade Analyzer / 交易分析器

    Analyzes trade statistics including:
    - Total trades / 总交易数
    - Win rate / 胜率
    - Average profit/loss / 平均盈亏
    - Largest win/loss / 最大盈利/亏损

    分析交易统计，包括：
    - 总交易数
    - 胜率
    - 平均盈亏
    - 最大盈利/亏损
    """

    def __init__(self, strategy: Optional[Strategy] = None):
        """
        Initialize TradeAnalyzer / 初始化交易分析器

        Args:
            strategy: Reference to strategy / 策略引用
        """
        super().__init__(strategy)
        self._trades: List[Dict[str, Any]] = []
        self._open_trades: Dict[str, Dict[str, Any]] = {}  # data_name -> trade info

    def start(self) -> None:
        """Reset and initialize / 重置和初始化"""
        self._trades = []
        self._open_trades = {}

    def notify_order(self, order: Order) -> None:
        """Handle order notifications / 处理订单通知"""
        if order.status == OrderStatus.FILLED:
            data_name = order.data.name

            if order.isbuy():
                # Opening trade / 开仓交易
                self._open_trades[data_name] = {
                    "entry_date": order.dt_filled,
                    "entry_price": order.executed_price,
                    "size": order.executed_size,
                    "commission": order.commission,
                }
            elif order.issell() and data_name in self._open_trades:
                # Closing trade / 平仓交易
                open_trade = self._open_trades.pop(data_name)
                exit_price = order.executed_price
                size = open_trade["size"]

                # Calculate PnL / 计算盈亏
                pnl = (exit_price - open_trade["entry_price"]) * size
                net_pnl = pnl - open_trade["commission"] - order.commission

                # Record trade / 记录交易
                trade = {
                    "entry_date": open_trade["entry_date"],
                    "entry_price": open_trade["entry_price"],
                    "exit_date": order.dt_filled,
                    "exit_price": exit_price,
                    "size": size,
                    "pnl": float(pnl),
                    "net_pnl": float(net_pnl),
                    "commission": float(open_trade["commission"] + order.commission),
                    "return_pct": float((exit_price - open_trade["entry_price"]) / open_trade["entry_price"] * 100),
                }
                self._trades.append(trade)

    def get_analysis(self) -> Dict[str, Any]:
        """Calculate and return trade statistics / 计算并返回交易统计"""
        if not self._trades:
            self._results = {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "total_pnl": 0.0,
            }
            return self._results

        winning_trades = [t for t in self._trades if t["net_pnl"] > 0]
        losing_trades = [t for t in self._trades if t["net_pnl"] < 0]

        total_pnl = sum(t["net_pnl"] for t in self._trades)
        avg_profit = sum(t["net_pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t["net_pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0

        total_profit = sum(t["net_pnl"] for t in winning_trades)
        total_loss = abs(sum(t["net_pnl"] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0

        self._results = {
            "total_trades": len(self._trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": float(len(winning_trades) / len(self._trades) * 100),
            "avg_profit": float(avg_profit),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "largest_win": float(max(t["net_pnl"] for t in self._trades)) if self._trades else 0.0,
            "largest_loss": float(min(t["net_pnl"] for t in self._trades)) if self._trades else 0.0,
            "total_pnl": float(total_pnl),
            "avg_trade_pnl": float(total_pnl / len(self._trades)),
            "trades": self._trades,
        }
        return self._results


# =============================================================================
# CerebroLite - Main Backtesting Engine / 主回测引擎
# =============================================================================

class CerebroLite:
    """
    Main backtesting engine / 主回测引擎

    CerebroLite orchestrates the backtesting process:
    - Manages data feeds / 管理数据源
    - Runs strategies / 运行策略
    - Simulates broker / 模拟经纪商
    - Collects analyzer results / 收集分析器结果

    CerebroLite编排回测过程：
    - 管理数据源
    - 运行策略
    - 模拟经纪商
    - 收集分析器结果

    Example / 示例:
    ```python
    cerebro = CerebroLite(initial_cash=100000, commission=0.001)
    cerebro.add_data(PandasDataFeed(df, name='AAPL'))
    cerebro.add_strategy(MyStrategy)
    cerebro.add_analyzer(SharpeRatio)
    cerebro.add_analyzer(MaxDrawdown)

    results = cerebro.run()
    print(f"Final Value: {results['final_value']:.2f}")
    print(f"Sharpe: {results['analyzers']['sharpe_ratio']['sharpe_ratio']:.2f}")
    ```

    Attributes:
        initial_cash: Starting cash amount / 起始资金
        commission: Commission rate / 手续费率
        slippage: Slippage rate / 滑点率
        data_feeds: List of data feeds / 数据源列表
        strategies: List of strategy classes / 策略类列表
        analyzers: List of analyzer classes / 分析器类列表
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
    ):
        """
        Initialize CerebroLite / 初始化CerebroLite

        Args:
            initial_cash: Starting cash amount / 起始资金
            commission: Commission rate (e.g., 0.001 = 0.1%) / 手续费率
            slippage: Slippage rate (e.g., 0.001 = 0.1%) / 滑点率
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.data_feeds: List[DataFeed] = []
        self.strategy_classes: List[Type[Strategy]] = []
        self.analyzer_classes: List[Type[Analyzer]] = []
        self._broker: Optional[BrokerSimulator] = None

    def add_data(self, data: DataFeed, name: Optional[str] = None) -> None:
        """
        Add a data feed to Cerebro / 向Cerebro添加数据源

        Args:
            data: Data feed to add / 要添加的数据源
            name: Optional name for the data feed / 数据源的可选名称
        """
        if name:
            data.name = name
        self.data_feeds.append(data)
        logger.debug(f"Added data feed: {data.name}")

    def add_strategy(self, strategy_cls: Type[Strategy], **kwargs) -> None:
        """
        Add a strategy class to Cerebro / 向Cerebro添加策略类

        Args:
            strategy_cls: Strategy class to add / 要添加的策略类
            **kwargs: Parameters to pass to strategy / 传递给策略的参数
        """
        self.strategy_classes.append((strategy_cls, kwargs))
        logger.debug(f"Added strategy: {strategy_cls.__name__}")

    def add_analyzer(self, analyzer_cls: Type[Analyzer], **kwargs) -> None:
        """
        Add an analyzer class to Cerebro / 向Cerebro添加分析器类

        Args:
            analyzer_cls: Analyzer class to add / 要添加的分析器类
            **kwargs: Parameters to pass to analyzer / 传递给分析器的参数
        """
        self.analyzer_classes.append((analyzer_cls, kwargs))
        logger.debug(f"Added analyzer: {analyzer_cls.__name__}")

    def run(self) -> Dict[str, Any]:
        """
        Run the backtest / 运行回测

        Returns:
            Dictionary containing:
            - final_value: Final portfolio value / 最终资产价值
            - initial_value: Initial portfolio value / 初始资产价值
            - return_pct: Total return percentage / 总收益百分比
            - analyzers: Dictionary of analyzer results / 分析器结果字典

            包含以下内容的字典：
            - final_value：最终资产价值
            - initial_value：初始资产价值
            - return_pct：总收益百分比
            - analyzers：分析器结果字典
        """
        if not self.data_feeds:
            raise ValueError("No data feeds added. Use add_data() first.")

        if not self.strategy_classes:
            raise ValueError("No strategies added. Use add_strategy() first.")

        # Initialize broker / 初始化经纪商
        self._broker = BrokerSimulator(
            initial_cash=self.initial_cash,
            commission=self.commission,
            slippage=self.slippage,
        )

        # Prepare data feeds / 准备数据源
        for df in self.data_feeds:
            df.start()

        # Get minimum data length / 获取最小数据长度
        min_bars = min(len(df) for df in self.data_feeds)
        if min_bars == 0:
            raise ValueError("All data feeds are empty.")

        # Run each strategy / 运行每个策略
        all_results = []

        for strategy_cls, strategy_kwargs in self.strategy_classes:
            strategy_results = self._run_strategy(
                strategy_cls,
                strategy_kwargs,
                min_bars,
            )
            all_results.append(strategy_results)

        # Return results from first strategy (for backward compatibility)
        # 返回第一个策略的结果（向后兼容）
        return all_results[0] if all_results else {
            "final_value": self.initial_cash,
            "initial_value": self.initial_cash,
            "return_pct": 0.0,
            "analyzers": {},
        }

    def _run_strategy(
        self,
        strategy_cls: Type[Strategy],
        strategy_kwargs: Dict[str, Any],
        min_bars: int,
    ) -> Dict[str, Any]:
        """
        Run a single strategy / 运行单个策略

        Args:
            strategy_cls: Strategy class / 策略类
            strategy_kwargs: Strategy parameters / 策略参数
            min_bars: Minimum number of bars / 最小K线数

        Returns:
            Backtest results dictionary / 回测结果字典
        """
        # Create strategy instance / 创建策略实例
        strategy = strategy_cls(**strategy_kwargs)
        strategy.cerebro = self
        strategy.datas = self.data_feeds
        strategy.broker = self._broker

        # Create and start analyzers / 创建并启动分析器
        analyzers = []
        for analyzer_cls, analyzer_kwargs in self.analyzer_classes:
            analyzer = analyzer_cls(strategy, **analyzer_kwargs)
            analyzer.start()
            analyzers.append(analyzer)

        # Run through all bars / 遍历所有K线
        for bar_idx in range(min_bars):
            # Set current bar for all data feeds / 设置所有数据源的当前K线
            for df in self.data_feeds:
                df.set_current(bar_idx)

            # Record broker value / 记录经纪商价值
            self._broker.record_value()

            # Call strategy next() / 调用策略next()
            strategy.next()

            # Execute pending orders and update positions / 执行待处理订单并更新持仓
            self._execute_orders(bar_idx)

            # Update analyzers / 更新分析器
            for analyzer in analyzers:
                analyzer.next()

        # Final value calculation / 最终价值计算
        final_value = self._broker.value
        initial_value = self.initial_cash
        return_pct = (final_value - initial_value) / initial_value * 100 if initial_value > 0 else 0.0

        # Collect analyzer results / 收集分析器结果
        analyzer_results = {}
        for analyzer in analyzers:
            result = analyzer.get_analysis()
            analyzer_name = analyzer.__class__.__name__.lower()
            analyzer_results[analyzer_name] = result

        return {
            "final_value": float(final_value),
            "initial_value": float(initial_value),
            "return_pct": float(return_pct),
            "final_cash": float(self._broker.cash),
            "analyzers": analyzer_results,
        }

    def _execute_orders(self, bar_idx: int) -> None:
        """
        Execute pending orders at current bar / 执行当前K线的待处理订单

        Args:
            bar_idx: Current bar index / 当前K线索引
        """
        current_data = self.data_feeds[0] if self.data_feeds else None
        if not current_data:
            return

        current_price = current_data.close

        # Get pending orders / 获取待处理订单
        pending_orders = [o for o in self._broker.orders if o.status == OrderStatus.PENDING]

        for order in pending_orders:
            # For now, execute all orders at current close price
            # Currently, simply fill at current price for simplicity
            self._broker.execute_order(order, float(current_price()))


# =============================================================================
# Built-in Indicators / 内置指标
# =============================================================================

class Indicator(ABC):
    """
    Base class for indicators / 指标基类

    Provides common functionality for all indicators.

    提供所有指标的通用功能。
    """

    def __init__(self, data: DataFeed):
        """
        Initialize indicator / 初始化指标

        Args:
            data: Data feed to calculate indicator on / 要计算指标的数据源
        """
        self.data = data
        self._values: List[float] = []

    @abstractmethod
    def calculate(self) -> float:
        """
        Calculate current indicator value / 计算当前指标值

        Returns:
            Current indicator value / 当前指标值
        """
        pass

    def update(self) -> None:
        """Update indicator with current bar / 用当前K线更新指标"""
        value = self.calculate()
        self._values.append(value)

    def __call__(self) -> float:
        """Get current indicator value / 获取当前指标值"""
        return self._values[-1] if self._values else 0.0

    def __len__(self) -> int:
        """Get number of values / 获取值数量"""
        return len(self._values)

    def __getitem__(self, key: int) -> float:
        """Get value by index / 通过索引获取值"""
        if key < 0:
            idx = len(self._values) + key
        else:
            idx = key
        return self._values[idx] if 0 <= idx < len(self._values) else 0.0


class SMA(Indicator):
    """
    Simple Moving Average / 简单移动平均

    Calculates the simple moving average of a data series.

    计算数据序列的简单移动平均。
    """

    def __init__(self, data: DataFeed, period: int = 20):
        """
        Initialize SMA / 初始化SMA

        Args:
            data: Data feed / 数据源
            period: Period for SMA / SMA周期
        """
        super().__init__(data)
        self.period = period

    def calculate(self) -> float:
        """Calculate current SMA value / 计算当前SMA值"""
        if len(self.data) < self.period:
            return 0.0

        total = 0.0
        for i in range(self.period):
            idx = len(self.data) - 1 - i
            bar = self.data.get_bar(idx)
            if bar:
                total += bar.close
            else:
                return 0.0

        return total / self.period


class EMA(Indicator):
    """
    Exponential Moving Average / 指数移动平均

    Calculates the exponential moving average of a data series.

    计算数据序列的指数移动平均。
    """

    def __init__(self, data: DataFeed, period: int = 20):
        """
        Initialize EMA / 初始化EMA

        Args:
            data: Data feed / 数据源
            period: Period for EMA / EMA周期
        """
        super().__init__(data)
        self.period = period
        self._multiplier = 2.0 / (period + 1)
        self._sma = None

    def calculate(self) -> float:
        """Calculate current EMA value / 计算当前EMA值"""
        if len(self.data) < self.period:
            return 0.0

        if self._sma is None:
            # Initialize with SMA / 用SMA初始化
            total = 0.0
            for i in range(self.period):
                bar = self.data.get_bar(len(self.data) - 1 - i)
                if bar:
                    total += bar.close
                else:
                    return 0.0
            self._sma = total / self.period
            return self._sma

        # Calculate EMA / 计算EMA
        current_close = self.data.close
        ema = (current_close - self._sma) * self._multiplier + self._sma
        return ema


# =============================================================================
# Exports / 导出
# =============================================================================

__all__ = [
    # Core engine / 核心引擎
    "CerebroLite",
    # Data feeds / 数据源
    "DataFeed",
    "PandasDataFeed",
    "CSVDataFeed",
    "LineAccessor",
    # Broker / 经纪商
    "BrokerSimulator",
    # Strategy / 策略
    "Strategy",
    # Order and Position / 订单和持仓
    "Order",
    "OrderType",
    "OrderStatus",
    "Position",
    "PositionSide",
    "Bar",
    # Analyzers / 分析器
    "Analyzer",
    "SharpeRatio",
    "MaxDrawdown",
    "CalmarRatio",
    "AnnualReturn",
    "TradeAnalyzer",
    # Indicators / 指标
    "Indicator",
    "SMA",
    "EMA",
]
