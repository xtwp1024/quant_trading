"""
backtrader_framework - 纯 Python 回测框架 / Pure-Python Backtesting Framework

Inspired by backtrader (https://www.backtrader.com/), this module provides a
comprehensive, self-contained backtesting framework with:

- Cerebro: Main backtesting engine (strategy runner, data feed management, broker control)
- DataFeed: Abstract base + CSV / pandas loaders
- Broker: Broker simulator with cash, margin, positions, slippage & commission
- SlippageModel: Base class + FixedSlippage / VolumeShareSlippage
- CommissionScheme: Base class + PerShare / PerTrade / PercentCommission
- Analyzer: Base class + SharpeRatio / MaxDrawdown / CalmarRatio / TradeAnalyzer / SQN
- Strategy: User strategy base class with __init__ / next lifecycle hooks
- Built-in indicators: SMA, EMA, MACD, RSI, BollingerBands, ATR, Stochastic, VWAP, ADX, CCI, ROC

Pure Python + pandas. No Cython. Designed to be familiar to backtrader users.

Usage / 用法示例:
```python
import pandas as pd
from quant_trading.backtester.backtrader_framework import (
    Cerebro, PandasDataFeed, CSVDataFeed,
    Strategy, Broker,
    SMA, EMA, MACD, RSI, BollingerBands, ATR, Stochastic,
    SharpeRatio, MaxDrawdown, CalmarRatio, TradeAnalyzer, SQN,
    FixedSlippage, VolumeShareSlippage,
    PerShare, PerTrade, PercentCommission,
)

# 1. Prepare data
data = pd.read_csv("data.csv", parse_dates=["datetime"], index_col="datetime")

# 2. Create data feed
feed = PandasDataFeed(data, name="BTC/USDT")

# 3. Define strategy
class SMACross(Strategy):
    params = (("fast", 10), ("slow", 30))

    def __init__(self):
        self.sma_fast = SMA(self.data.close, period=self.p.fast)
        self.sma_slow = SMA(self.data.close, period=self.p.slow)

    def next(self):
        if self.sma_fast > self.sma_slow and not self.getposition().size:
            self.buy()
        elif self.sma_fast < self.sma_slow and self.getposition().size:
            self.sell()

# 4. Build Cerebro
cerebro = Cerebro(initial_cash=100_000.0)
cerebro.adddata(feed)
cerebro.addstrategy(SMACross, fast=10, slow=30)

# 5. Configure broker
cerebro.broker.setcash(100_000.0)
cerebro.broker.setcommission(commission=0.001)          # 0.1% per trade
cerebro.broker.set_slippage(FixedSlippage(slippage=0.01))  # Fixed 0.01 slippage

# 6. Add analyzers
cerebro.addanalyzer(SharpeRatio)
cerebro.addanalyzer(MaxDrawdown)
cerebro.addanalyzer(CalmarRatio)
cerebro.addanalyzer(TradeAnalyzer)
cerebro.addanalyzer(SQN)

# 7. Run
result = cerebro.run()
print(f"Final Value: {result['final_value']:.2f}")
print(f"Sharpe: {result['analyzers']['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {result['analyzers']['max_drawdown']:.2%}")
print(f"SQN: {result['analyzers']['sqn']:.2f}")
```
"""

from __future__ import annotations

import math
import bisect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums / 枚举
# =============================================================================

class OrderType(Enum):
    """Order direction / 订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order lifecycle status / 订单生命周期状态"""
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionSide(Enum):
    """Position direction / 持仓方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


# =============================================================================
# Order & Position / 订单与持仓
# =============================================================================

@dataclass
class Order:
    """
    Order record / 订单记录

    Attributes:
        ref: Unique order reference number / 唯一订单引用号
        data: Target data feed / 目标数据源
        order_type: BUY or SELL / 买入或卖出
        size: Order size (>0 always) / 订单数量（始终为正）
        price: Limit price; None = market order / 限价；None=市价单
        bar: Bar index when order was created / 创建订单时的K线索引
        status: Current status / 当前状态
        executed_price: Fill price / 成交价
        executed_size: Filled quantity / 成交数量
        executed_value: Total fill value / 成交总额
        commission: Commission paid on this fill / 本次成交手续费
        created_dt: Creation timestamp / 创建时间戳
        filled_dt: Fill timestamp / 成交时间戳
    """
    ref: int
    data: "DataFeedBase"
    order_type: OrderType
    size: float
    price: Optional[float] = None
    bar: int = 0
    status: OrderStatus = OrderStatus.SUBMITTED
    executed_price: float = 0.0
    executed_size: float = 0.0
    executed_value: float = 0.0
    commission: float = 0.0
    created_dt: Optional[datetime] = None
    filled_dt: Optional[datetime] = None

    def isbuy(self) -> bool:
        """True if this is a buy order / 是否为买入单"""
        return self.order_type == OrderType.BUY

    def issell(self) -> bool:
        """True if this is a sell order / 是否为卖出单"""
        return self.order_type == OrderType.SELL

    def __repr__(self) -> str:
        return (f"Order(ref={self.ref}, {self.order_type.value}, "
                f"size={self.size}, price={self.price}, "
                f"status={self.status.value})")


@dataclass
class Position:
    """
    Open position tracking / 持仓追踪

    Attributes:
        data: Associated data feed / 关联数据源
        size: Current size (+=long, -=short) / 当前数量（正=多，空=空）
        avg_price: Volume-weighted average entry price / 成交量加权平均入场价
        closed_pnl: Realized PnL / 已实现盈亏
    """
    data: "DataFeedBase"
    size: float = 0.0
    avg_price: float = 0.0
    closed_pnl: float = 0.0

    @property
    def side(self) -> PositionSide:
        """Position direction / 持仓方向"""
        if self.size > 0:
            return PositionSide.LONG
        elif self.size < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    @property
    def value(self) -> float:
        """Position notional value at current avg_price / 按当前均价计算的合约价值"""
        return self.size * self.avg_price

    def update(self, size: float, price: float) -> None:
        """
        Update position with a fill / 用成交更新持仓

        Args:
            size: Fill size (+ve = buy, -ve = sell) / 成交数量（正=买，负=卖）
            price: Fill price / 成交价格
        """
        if self.size == 0:
            # Opening a new position
            self.size = size
            self.avg_price = price
        elif (self.size > 0 and size > 0) or (self.size < 0 and size < 0):
            # Adding to the same direction
            total_value = self.value + size * price
            self.size += size
            self.avg_price = total_value / self.size if self.size != 0 else 0.0
        else:
            # Closing or reversing
            if abs(size) >= abs(self.size):
                # Full close or reversal
                if abs(size) > abs(self.size):
                    # Partial reversal: size exceeds current position
                    self.closed_pnl += (price - self.avg_price) * self.size
                    self.size = size + self.size  # net new size
                    self.avg_price = price
                else:
                    # Full close
                    self.closed_pnl += (price - self.avg_price) * self.size
                    self.size = 0
                    self.avg_price = 0.0
            else:
                # Partial close
                self.closed_pnl += (price - self.avg_price) * abs(size)
                self.size += size


# =============================================================================
# Data Feed / 数据源
# =============================================================================

class DataFeedBase(ABC):
    """
    Abstract base for all data feeds / 所有数据源的抽象基类

    Subclasses must implement: _load(), _bar_count(), datetime_at()
    """

    _order_starts: Dict[str, List[int]] = {}

    def __init__(self, name: Optional[str] = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.params = kwargs
        self._bars: List[Dict[str, Any]] = []
        self._current_idx: int = -1
        self._loaded: bool = False

        # Line accessors created lazily
        self._lines: Dict[str, "_Line"] = {}

    # --- backtrader-style line accessors ---
    @property
    def datetime(self) -> "_Line":
        return self._getline("datetime")

    @property
    def open(self) -> "_Line":
        return self._getline("open")

    @property
    def high(self) -> "_Line":
        return self._getline("high")

    @property
    def low(self) -> "_Line":
        return self._getline("low")

    @property
    def close(self) -> "_Line":
        return self._getline("close")

    @property
    def volume(self) -> "_Line":
        return self._getline("volume")

    @property
    def openinterest(self) -> "_Line":
        return self._getline("openinterest")

    def _getline(self, name: str) -> "_Line":
        if name not in self._lines:
            self._lines[name] = _Line(self, name)
        return self._lines[name]

    def __len__(self) -> int:
        """Number of bars available / 可用K线数量"""
        self._ensure_loaded()
        return len(self._bars)

    @property
    def datetime_arr(self) -> np.ndarray:
        """Array of datetime values / datetime 数组"""
        self._ensure_loaded()
        return np.array([b["datetime"] for b in self._bars], dtype="object")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    @abstractmethod
    def _load(self) -> None:
        """Subclasses implement to populate self._bars / 子类实现以填充 self._bars"""
        raise NotImplementedError

    @property
    def ibar(self) -> int:
        """Current bar index / 当前K线索引"""
        return self._current_idx

    @property
    def buflen(self) -> int:
        """Total bars loaded / 已加载K线总数"""
        self._ensure_loaded()
        return len(self._bars)

    def forward(self) -> None:
        """Advance to next bar / 前进到下一根K线"""
        self._current_idx += 1

    def rewind(self) -> bool:
        """Step back one bar; returns True if valid / 后退一根K线；有效则返回True"""
        if self._current_idx > 0:
            self._current_idx -= 1
            return True
        return False

    def rewindto(self, idx: int) -> None:
        """Jump to a specific bar index / 跳转到指定K线索引"""
        self._current_idx = idx

    def _bar(self, offset: int = 0) -> Dict[str, Any]:
        """Return bar at current_idx + offset / 返回当前偏移的K线"""
        idx = self._current_idx + offset
        if 0 <= idx < len(self._bars):
            return self._bars[idx]
        return {}

    def date(self, offset: int = 0) -> Any:
        """Datetime of bar at offset / 偏移K线的日期"""
        bar = self._bar(offset)
        return bar.get("datetime")

    def price(self, field: str, offset: int = 0) -> float:
        """Price field at offset / 偏移K线的价格字段"""
        bar = self._bar(offset)
        return float(bar.get(field, 0.0))

    def volume_val(self, offset: int = 0) -> float:
        """Volume at offset / 偏移K线的成交量"""
        return float(self._bar(offset).get("volume", 0.0))

    def advance(self) -> bool:
        """Alias for forward() / forward()的别名"""
        self.forward()
        return self._current_idx < len(self._bars)


class _Line:
    """
    Lazy line accessor for DataFeedBase / DataFeedBase 的惰性线访问器

    Supports backtrader-style indexing:
        line[-1] = current bar, line[-2] = previous bar, etc.
    """

    def __init__(self, feed: DataFeedBase, attr: str):
        self._feed = feed
        self._attr = attr

    def __getitem__(self, key: int) -> float:
        feed = self._feed
        if key < 0:
            idx = feed._current_idx + key
        else:
            idx = key
        if 0 <= idx < len(feed._bars):
            return float(feed._bars[idx].get(self._attr, 0.0))
        return 0.0

    def __call__(self) -> float:
        """Current value (equivalent to [-1]) / 当前值（等同于[-1]）"""
        return self[-1]

    def __len__(self) -> int:
        return len(self._feed._bars)

    def __iter__(self):
        for i in range(len(self._feed._bars)):
            yield float(self._feed._bars[i].get(self._attr, 0.0))

    @property
    def buflen(self) -> int:
        return len(self._feed._bars)


class PandasDataFeed(DataFeedBase):
    """
    Data feed from a pandas DataFrame / 从 pandas DataFrame 加载数据源

    Expected DataFrame index: datetime (parse_dates=True recommended)
    Expected columns: open, high, low, close, volume (case-insensitive)

    Args:
        df: DataFrame with OHLCV data / 含 OHLCV 数据的 DataFrame
        name: Optional feed name / 可选的数据源名称
        datetime_col: Name of the datetime column (default: index) / datetime 列名
    """

    def __init__(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        datetime_col: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._df = df
        self._dt_col = datetime_col
        self._cols_map: Dict[str, str] = {}

    def _load(self) -> None:
        df = self._df.copy()

        # Resolve datetime column
        if self._dt_col:
            dt_vals = df[self._dt_col]
        else:
            dt_vals = df.index
        df = df.reset_index(drop=True)

        # Normalise column names to lowercase
        self._cols_map = {c.lower(): c for c in df.columns}
        n = len(df)

        self._bars = []
        for i in range(n):
            row = df.iloc[i]
            self._bars.append({
                "datetime": dt_vals[i] if isinstance(dt_vals, pd.Index) else dt_vals.iloc[i],
                "open": float(row.get(self._cols_map.get("open", "open"), 0.0)),
                "high": float(row.get(self._cols_map.get("high", "high"), 0.0)),
                "low": float(row.get(self._cols_map.get("low", "low"), 0.0)),
                "close": float(row.get(self._cols_map.get("close", "close"), 0.0)),
                "volume": float(row.get(self._cols_map.get("volume", "volume"), 0.0)),
                "openinterest": float(row.get(self._cols_map.get("openinterest", "openinterest"), 0.0)),
            })


class CSVDataFeed(DataFeedBase):
    """
    Data feed loaded from a CSV file / 从 CSV 文件加载数据源

    Args:
        path: Path to CSV file / CSV 文件路径
        name: Optional feed name / 可选的数据源名称
        datetime_col: Column name or index for datetime (default: 0) / datetime 列名或索引
        datetime_format: strptime format string / strptime 格式字符串
        **kwargs: Passed to pd.read_csv / 透传给 pd.read_csv
    """

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        datetime_col: Union[str, int, None] = 0,
        datetime_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name or path)
        self._path = path
        self._dt_col = datetime_col
        self._dt_fmt = datetime_format
        self._csv_kwargs = kwargs

    def _load(self) -> None:
        df = pd.read_csv(self._path, **self._csv_kwargs)

        # Resolve datetime
        if isinstance(self._dt_col, str):
            dt_vals = pd.to_datetime(df[self._dt_col], format=self._dt_fmt)
        else:
            dt_vals = pd.to_datetime(df.iloc[:, self._dt_col], format=self._dt_fmt)

        # Normalise column names
        df.columns = df.columns.str.lower()

        self._bars = []
        for i in range(len(df)):
            row = df.iloc[i]
            self._bars.append({
                "datetime": dt_vals.iloc[i],
                "open": float(row.get("open", 0.0)),
                "high": float(row.get("high", 0.0)),
                "low": float(row.get("low", 0.0)),
                "close": float(row.get("close", 0.0)),
                "volume": float(row.get("volume", 0.0)),
                "openinterest": float(row.get("openinterest", 0.0)),
            })


# =============================================================================
# Slippage Models / 滑点模型
# =============================================================================

class SlippageModel(ABC):
    """
    Abstract base for slippage models / 滑点模型抽象基类

    Slippage is applied when an order is filled, modifying the fill price.
    """

    @abstractmethod
    def get_price(
        self,
        order: Order,
        current_price: float,
        volume: float,
    ) -> float:
        """
        Return the slippage-adjusted fill price / 返回滑点调整后的成交价格

        Args:
            order: The order being filled / 被执行的订单
            current_price: Base price (close or limit price) / 基础价格
            volume: Trading volume on this bar / 本K线交易量

        Returns:
            Adjusted price / 调整后的价格
        """
        raise NotImplementedError


class FixedSlippage(SlippageModel):
    """
    Fixed absolute slippage per share / 固定绝对滑点（每股）

    Args:
        slippage: Absolute slippage amount added to (buy) or subtracted from (sell)
                  the fill price / 固定滑点金额
    """

    def __init__(self, slippage: float = 0.01):
        self.slippage = slippage

    def get_price(
        self,
        order: Order,
        current_price: float,
        volume: float,
    ) -> float:
        if order.isbuy():
            return current_price + self.slippage
        return current_price - self.slippage


class VolumeShareSlippage(SlippageModel):
    """
    Volume-share proportional slippage / 成交量比例滑点

    Slippage = impact_coef * (order_size / volume) * price
    Higher order size relative to bar volume causes more slippage.

    Args:
        impact_coef: Sensitivity coefficient (default 0.1) / 敏感度系数
        max_slippage_pct: Maximum slippage as fraction of price (default 0.05) / 最大滑点比例
    """

    def __init__(self, impact_coef: float = 0.1, max_slippage_pct: float = 0.05):
        self.impact_coef = impact_coef
        self.max_slippage_pct = max_slippage_pct

    def get_price(
        self,
        order: Order,
        current_price: float,
        volume: float,
    ) -> float:
        if volume <= 0:
            return current_price
        ratio = abs(order.size) / volume
        slippage = self.impact_coef * ratio * current_price
        slippage = min(slippage, current_price * self.max_slippage_pct)
        if order.isbuy():
            return current_price + slippage
        return current_price - slippage


class NoSlippage(SlippageModel):
    """No slippage / 无滑点"""

    def get_price(
        self,
        order: Order,
        current_price: float,
        volume: float,
    ) -> float:
        return current_price


# =============================================================================
# Commission Schemes / 手续费方案
# =============================================================================

class CommissionScheme(ABC):
    """
    Abstract base for commission schemes / 手续费方案抽象基类

    Commission is calculated when an order is filled.
    """

    @abstractmethod
    def get_commission(
        self,
        size: float,
        price: float,
        position_value: float,
    ) -> float:
        """
        Calculate commission for a fill / 计算成交手续费

        Args:
            size: Filled quantity / 成交数量
            price: Fill price / 成交价格
            position_value: Total position value / 持仓总价值

        Returns:
            Commission amount / 手续费金额
        """
        raise NotImplementedError


class PerShare(CommissionScheme):
    """
    Fixed commission per share / 固定每股手续费

    Args:
        commission: Commission per share / 每股手续费
    """

    def __init__(self, commission: float = 0.0):
        self.commission = commission

    def get_commission(
        self,
        size: float,
        price: float,
        position_value: float,
    ) -> float:
        return abs(size) * self.commission


class PerTrade(CommissionScheme):
    """
    Fixed commission per trade (one side) / 每笔交易固定手续费

    Args:
        commission: Commission per trade / 每笔交易手续费
    """

    def __init__(self, commission: float = 0.0):
        self.commission = commission

    def get_commission(
        self,
        size: float,
        price: float,
        position_value: float,
    ) -> float:
        return self.commission


class PercentCommission(CommissionScheme):
    """
    Commission as a percentage of trade value / 成交金额比例手续费

    Args:
        commission: Fraction (e.g. 0.001 = 0.1%) / 比例（如 0.001 = 0.1%）
    """

    def __init__(self, commission: float = 0.0):
        self.commission = commission

    def get_commission(
        self,
        size: float,
        price: float,
        position_value: float,
    ) -> float:
        return abs(position_value) * self.commission


# =============================================================================
# Broker / 经纪商模拟器
# =============================================================================

class Broker:
    """
    Backtesting broker simulator / 回测用经纪商模拟器

    Manages cash, positions, order execution, slippage, and commission.
    Integrates with Cerebro to process orders each bar.

    Attributes:
        cash: Available cash / 可用资金
        slippage_model: Active slippage model / 当前滑点模型
        commission_scheme: Active commission scheme / 当前手续费方案
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_model: Optional[SlippageModel] = None,
        commission_scheme: Optional[CommissionScheme] = None,
    ):
        self.cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: Dict[str, Position] = {}  # keyed by data.name
        self._orders: List[Order] = []
        self._order_ref_counter = 0
        self._slippage = slippage_model or NoSlippage()
        self._commission = commission_scheme or PercentCommission(0.0)
        self._frozen_value: Optional[float] = None  # for margin calculation

    # ---- Position helpers ----

    def getposition(self, data: DataFeedBase) -> Position:
        """Get or create position for data feed / 获取或创建数据源对应持仓"""
        key = data.name
        if key not in self._positions:
            self._positions[key] = Position(data=data)
        return self._positions[key]

    def getpositionbyname(self, name: str) -> Optional[Position]:
        """Get position by data feed name / 按名称获取持仓"""
        return self._positions.get(name)

    def getvalue(self, data: Optional[DataFeedBase] = None) -> float:
        """
        Total portfolio value (cash + all positions) / 组合总价值

        If data is specified, returns value of that specific position.
        """
        if data is not None:
            pos = self.getposition(data)
            return self.cash + pos.size * pos.avg_price
        total = self.cash
        for pos in self._positions.values():
            if pos.size != 0:
                total += pos.size * pos.avg_price
        return total

    def getcash(self) -> float:
        """Available cash / 可用资金"""
        return self.cash

    # ---- Order management ----

    def submit(self, order: Order) -> Order:
        """Submit a new order / 提交新订单"""
        order.status = OrderStatus.ACCEPTED
        self._orders.append(order)
        return order

    def cancel(self, order: Order) -> None:
        """Cancel a pending order / 撤销待处理订单"""
        if order.status in (OrderStatus.SUBMITTED, OrderStatus.ACCEPTED):
            order.status = OrderStatus.CANCELLED

    def buy(
        self,
        data: DataFeedBase,
        size: float,
        price: Optional[float] = None,
    ) -> Order:
        """Place a buy order / 发送买入订单"""
        self._order_ref_counter += 1
        order = Order(
            ref=self._order_ref_counter,
            data=data,
            order_type=OrderType.BUY,
            size=size,
            price=price,
        )
        return self.submit(order)

    def sell(
        self,
        data: DataFeedBase,
        size: float,
        price: Optional[float] = None,
    ) -> Order:
        """Place a sell order / 发送卖出订单"""
        self._order_ref_counter += 1
        order = Order(
            ref=self._order_ref_counter,
            data=data,
            order_type=OrderType.SELL,
            size=size,
            price=price,
        )
        return self.submit(order)

    # ---- Per-bar execution ----

    def process_orders(self, data: DataFeedBase) -> List[Order]:
        """
        Execute all accepted orders for the current bar / 执行当前K线的所有已接受订单

        Returns list of filled orders.
        """
        filled = []
        current_price = data.close()
        bar_volume = data.volume()
        bar_idx = data.ibar

        for order in list(self._orders):
            if order.status not in (OrderStatus.ACCEPTED, OrderStatus.PARTIAL):
                continue
            if order.data is not data:
                continue

            order.status = OrderStatus.ACCEPTED
            order.bar = bar_idx

            # Determine fill price
            fill_price = self._slippage.get_price(order, current_price, bar_volume)
            if order.price is not None:
                # Limit order: fill only if price is met
                if order.isbuy() and fill_price > order.price:
                    fill_price = order.price
                elif order.issell() and fill_price < order.price:
                    fill_price = order.price

            fill_size = order.size
            fill_value = abs(fill_size * fill_price)
            commission = self._commission.get_commission(fill_size, fill_price, fill_value)

            # Check cash
            if order.isbuy() and (self.cash - fill_value - commission) < 0:
                order.status = OrderStatus.REJECTED
                continue

            # Execute
            self.cash -= fill_value + commission if order.isbuy() else -fill_value + commission
            order.executed_price = fill_price
            order.executed_size = fill_size
            order.executed_value = fill_value
            order.commission = commission
            order.status = OrderStatus.COMPLETED
            order.filled_dt = data.date()

            # Update position
            pos = self.getposition(data)
            signed_size = fill_size if order.isbuy() else -fill_size
            pos.update(signed_size, fill_price)
            # pos.closed_pnl is already updated inside pos.update()

            filled.append(order)

        # Remove completed / cancelled orders
        self._orders = [
            o for o in self._orders
            if o.status not in (OrderStatus.COMPLETED, OrderStatus.CANCELLED, OrderStatus.REJECTED)
        ]
        return filled

    # ---- Configuration ----

    def setcash(self, cash: float) -> None:
        """Set available cash / 设置可用资金"""
        self.cash = cash

    def setcommission(self, commission: float = 0.0) -> None:
        """
        Set commission scheme by rate / 按费率设置手续费方案

        Args:
            commission: Commission rate (fraction of trade value) / 手续费费率
        """
        self._commission = PercentCommission(commission)

    def setcommission_scheme(self, scheme: CommissionScheme) -> None:
        """Set commission scheme / 设置手续费方案"""
        self._commission = scheme

    def set_slippage(self, model: SlippageModel) -> None:
        """Set slippage model / 设置滑点模型"""
        self._slippage = model

    def reset(self, cash: Optional[float] = None) -> None:
        """Reset broker state / 重置经纪商状态"""
        self.cash = cash if cash is not None else self._initial_cash
        self._positions.clear()
        self._orders.clear()
        self._order_ref_counter = 0


# =============================================================================
# Strategy Base Class / 策略基类
# =============================================================================

class Strategy:
    """
    Base class for user-defined strategies / 用户策略基类

    Subclass and override:
        __init__(self)  — create indicators, store state
        next(self)      — called on each bar with new data

    Indicators are created via helper methods on self (e.g. SMA, EMA).
    Use self.buy() / self.sell() / self.close() to trade.
    Use self.getposition() to query current position.

    Parameters are defined as class attributes or a params tuple.

    Example:
        class MyStrategy(Strategy):
            params = (("period", 20),)

            def __init__(self):
                self.sma = SMA(self.data.close, period=self.params.period)

            def next(self):
                if self.data.close() > self.sma and not self.getposition().size:
                    self.buy()
    """

    params: Union[Tuple, Dict] = ()

    def __init__(self, cerebro: "Cerebro", data: DataFeedBase, args: Dict[str, Any]):
        self.cerebro = cerebro
        self.data = data
        self._args = args
        self._indicators: List["Indicator"] = []
        self._orders: Dict[int, Order] = {}
        self._broker: Broker = cerebro.broker

        # Copy params from class definition
        if isinstance(self.params, tuple):
            param_dict = {}
            for name, val in self.params:
                param_dict[name] = val
            self.p = ParamStore(param_dict)
        else:
            self.p = ParamStore(self.params if self.params else {})

        # Override with runtime args
        for k, v in args.items():
            setattr(self.p, k, v)

    # ---- Trading API ----

    def buy(
        self,
        size: Optional[float] = None,
        price: Optional[float] = None,
        data: Optional[DataFeedBase] = None,
    ) -> Order:
        """
        Place a buy order / 发送买入订单

        Args:
            size: Number of units (default: use sizer) / 数量（默认使用sizer）
            price: Limit price / 限价
            data: Target data feed (default: primary data) / 目标数据源

        Returns:
            Order instance / 订单实例
        """
        data = data or self.data
        size = size or self._sizer_size(data)
        order = self._broker.buy(data, size, price)
        self._orders[order.ref] = order
        return order

    def sell(
        self,
        size: Optional[float] = None,
        price: Optional[float] = None,
        data: Optional[DataFeedBase] = None,
    ) -> Order:
        """Place a sell order / 发送卖出订单"""
        data = data or self.data
        size = size or self._sizer_size(data)
        order = self._broker.sell(data, size, price)
        self._orders[order.ref] = order
        return order

    def close(self, data: Optional[DataFeedBase] = None) -> Order:
        """Close any open position on data feed / 平掉数据源上的所有仓位"""
        data = data or self.data
        pos = self._broker.getposition(data)
        if pos.size == 0:
            return None
        return self.sell(size=abs(pos.size), data=data)

    def getposition(self, data: Optional[DataFeedBase] = None) -> Position:
        """Current position on data feed / 当前数据源持仓"""
        data = data or self.data
        return self._broker.getposition(data)

    # ---- Sizer ----

    def _sizer_size(self, data: DataFeedBase) -> float:
        """Ask cerebro's sizer for default size / 从 Cerebro 获取默认交易数量"""
        if self.cerebro._sizer is not None:
            return self.cerebro._sizer.get_size(self, data)
        return 1.0

    # ---- Indicator helpers (lazy line wrappers) ----

    def SMA(self, line: "_Line", period: int) -> "SMA":
        """Add Simple Moving Average / 添加简单移动平均线"""
        ind = SMA(line, period=period)
        self._indicators.append(ind)
        return ind

    def EMA(self, line: "_Line", period: int) -> "EMA":
        """Add Exponential Moving Average / 添加指数移动平均线"""
        ind = EMA(line, period=period)
        self._indicators.append(ind)
        return ind

    def MACD(
        self,
        line: "_Line",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> "MACD":
        """Add MACD indicator / 添加 MACD 指标"""
        ind = MACD(line, fast=fast, slow=slow, signal=signal)
        self._indicators.append(ind)
        return ind

    def RSI(self, line: "_Line", period: int = 14) -> "RSI":
        """Add RSI indicator / 添加 RSI 指标"""
        ind = RSI(line, period=period)
        self._indicators.append(ind)
        return ind

    def BollingerBands(
        self,
        line: "_Line",
        period: int = 20,
        devfactor: float = 2.0,
    ) -> "BollingerBands":
        """Add Bollinger Bands / 添加布林带"""
        ind = BollingerBands(line, period=period, devfactor=devfactor)
        self._indicators.append(ind)
        return ind

    def ATR(self, data: DataFeedBase, period: int = 14) -> "ATR":
        """Add Average True Range / 添加 ATR"""
        ind = ATR(data, period=period)
        self._indicators.append(ind)
        return ind

    def Stochastic(
        self,
        data: DataFeedBase,
        k_period: int = 14,
        d_period: int = 3,
    ) -> "Stochastic":
        """Add Stochastic Oscillator / 添加随机振荡器"""
        ind = Stochastic(data, k_period=k_period, d_period=d_period)
        self._indicators.append(ind)
        return ind

    def VWAP(self, data: DataFeedBase) -> "VWAP":
        """Add Volume Weighted Average Price / 添加成交量加权平均价"""
        ind = VWAP(data)
        self._indicators.append(ind)
        return ind

    def ADX(self, data: DataFeedBase, period: int = 14) -> "ADX":
        """Add Average Directional Index / 添加平均趋向指数"""
        ind = ADX(data, period=period)
        self._indicators.append(ind)
        return ind

    def CCI(self, data: DataFeedBase, period: int = 20) -> "CCI":
        """Add Commodity Channel Index / 添加顺势指标"""
        ind = CCI(data, period=period)
        self._indicators.append(ind)
        return ind

    def ROC(self, line: "_Line", period: int = 12) -> "ROC":
        """Add Rate of Change / 添加变动率指标"""
        ind = ROC(line, period=period)
        self._indicators.append(ind)
        return ind

    # ---- Lifecycle hooks ----

    def on_start(self) -> None:
        """Called once before backtest starts / 回测开始前调用一次"""
        pass

    def on_end(self) -> None:
        """Called once after backtest ends / 回测结束后调用一次"""
        pass


class ParamStore:
    """Simple attribute-based parameter store / 简易属性参数存储"""

    def __init__(self, params: Dict[str, Any]):
        for k, v in params.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ParamStore({items})"


# =============================================================================
# Sizer / 仓位尺寸计算器
# =============================================================================

class Sizer(ABC):
    """Base class for position sizers / 仓位尺寸计算器基类"""

    @abstractmethod
    def get_size(
        self,
        strategy: Strategy,
        data: DataFeedBase,
    ) -> float:
        """Return number of units to trade / 返回交易数量"""
        raise NotImplementedError


class FixedSize(Sizer):
    """Fixed number of units per order / 固定数量"""

    def __init__(self, stake: float = 1.0):
        self._stake = stake

    def get_size(
        self,
        strategy: Strategy,
        data: DataFeedBase,
    ) -> float:
        return self._stake


class PercentSize(Sizer):
    """Size as a percentage of broker portfolio value / 按组合价值比例计算数量"""

    def __init__(self, percent: float = 0.05):
        self._percent = percent

    def get_size(
        self,
        strategy: Strategy,
        data: DataFeedBase,
    ) -> float:
        portfolio_value = strategy.cerebro.broker.getvalue()
        price = data.close()
        if price <= 0:
            return 0.0
        return math.floor(portfolio_value * self._percent / price)


# =============================================================================
# Indicators / 技术指标
# =============================================================================

class Indicator(ABC):
    """
    Abstract base for all indicators / 所有指标抽象基类

    Indicators track a line of data and compute values over a rolling window.
    """

    def __init__(self, *args, **kwargs):
        self.params = kwargs
        self._values: List[float] = []
        self._ready = False

    @abstractmethod
    def _calc(self) -> float:
        """Compute current indicator value / 计算当前指标值"""
        raise NotImplementedError

    def update(self) -> None:
        """Advance indicator by one bar / 指标前进一根K线"""
        val = self._calc()
        self._values.append(val)
        self._ready = len(self._values) > 0

    @property
    def lines(self):
        """Compatibility: return self for backtrader-style .lines attribute"""
        return self


class SMA(Indicator):
    """
    Simple Moving Average / 简单移动平均

    SMA(n) = (sum of last n prices) / n
    """

    def __init__(self, line: _Line, period: int = 30):
        super().__init__(period=period)
        self._line = line
        self._period = period
        self._buffer: List[float] = []

    def _calc(self) -> float:
        buf = self._buffer
        val = self._line[-1]
        buf.append(val)
        if len(buf) > self._period:
            buf.pop(0)
        if len(buf) < self._period:
            return 0.0
        return sum(buf) / self._period

    def __call__(self) -> float:
        return self._values[-1] if self._values else 0.0

    def __getitem__(self, key: int) -> float:
        if key < 0 and len(self._values) >= abs(key):
            return self._values[key]
        return 0.0

    def __len__(self) -> int:
        return len(self._values)


class EMA(Indicator):
    """
    Exponential Moving Average / 指数移动平均

    EMA(n) = alpha * price + (1 - alpha) * EMA(n-1)
    where alpha = 2 / (n + 1)
    """

    def __init__(self, line: _Line, period: int = 30):
        super().__init__(period=period)
        self._line = line
        self._period = period
        self._alpha = 2.0 / (period + 1)
        self._ema: Optional[float] = None

    def _calc(self) -> float:
        price = self._line[-1]
        if self._ema is None:
            self._ema = price
        else:
            self._ema = self._alpha * price + (1 - self._alpha) * self._ema
        return self._ema

    def __call__(self) -> float:
        return self._ema if self._ema is not None else 0.0

    def __getitem__(self, key: int) -> float:
        return self._values[key] if key < len(self._values) else 0.0

    def __len__(self) -> int:
        return len(self._values)


class MACD(Indicator):
    """
    MACD (Moving Average Convergence Divergence) / 移动平均收敛发散

    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal
    """

    def __init__(
        self,
        line: _Line,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):
        super().__init__(fast=fast, slow=slow, signal=signal)
        self._line = line
        self._fast = EMA(line, fast)
        self._slow = EMA(line, slow)
        self._macd_line: List[float] = []
        self._signal = EMA(_SimpleLineAccessor(self._macd_line), signal)
        self._hist: List[float] = []

    def _calc(self) -> float:
        f = self._fast[-1]
        s = self._slow[-1]
        macd = f - s
        self._macd_line.append(macd)
        sig = self._signal[-1]
        self._hist.append(macd - sig)
        return macd

    @property
    def signal(self) -> float:
        return self._signal[-1]

    @property
    def histogram(self) -> float:
        return self._hist[-1] if self._hist else 0.0

    def __call__(self) -> float:
        return self._values[-1] if self._values else 0.0

    def __len__(self) -> int:
        return len(self._values)


class _SimpleLineAccessor:
    """Wraps a plain list as a line for EMA inside MACD / 将普通列表包装为 Line"""
    def __init__(self, values: List[float]):
        self._vals = values
    def __call__(self) -> float:
        return self._vals[-1] if self._vals else 0.0
    def __getitem__(self, key: int) -> float:
        if key < 0 and len(self._vals) >= abs(key):
            return self._vals[key]
        return 0.0


class RSI(Indicator):
    """
    Relative Strength Index / 相对强弱指数

    RSI(n) = 100 - 100 / (1 + RS)
    RS = average gain / average loss over n periods
    """

    def __init__(self, line: _Line, period: int = 14):
        super().__init__(period=period)
        self._line = line
        self._period = period
        self._gains: List[float] = []
        self._losses: List[float] = []

    def _calc(self) -> float:
        cur = self._line[-1]
        prev = self._line[-2] if len(self._line) > 1 else cur
        gain = max(cur - prev, 0.0)
        loss = max(prev - cur, 0.0)
        self._gains.append(gain)
        self._losses.append(loss)
        if len(self._gains) < self._period:
            return 50.0
        avg_gain = sum(self._gains[-self._period:]) / self._period
        avg_loss = sum(self._losses[-self._period:]) / self._period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def __call__(self) -> float:
        return self._values[-1] if self._values else 50.0

    def __len__(self) -> int:
        return len(self._values)


class BollingerBands(Indicator):
    """
    Bollinger Bands / 布林带

    Middle = SMA(n)
    Upper = Middle + devfactor * stddev
    Lower = Middle - devfactor * stddev
    """

    def __init__(
        self,
        line: _Line,
        period: int = 20,
        devfactor: float = 2.0,
    ):
        super().__init__(period=period, devfactor=devfactor)
        self._line = line
        self._period = period
        self._devfactor = devfactor
        self._buffer: List[float] = []

    def _calc(self) -> float:
        self._buffer.append(self._line[-1])
        if len(self._buffer) > self._period:
            self._buffer.pop(0)
        if len(self._buffer) < self._period:
            return 0.0
        mean = sum(self._buffer) / self._period
        variance = sum((x - mean) ** 2 for x in self._buffer) / self._period
        stddev = math.sqrt(variance)
        return mean  # middle band

    @property
    def middle(self) -> float:
        return self._values[-1] if self._values else 0.0

    @property
    def upper(self) -> float:
        buf = self._buffer
        if len(buf) < self._period:
            return 0.0
        mean = sum(buf) / self._period
        variance = sum((x - mean) ** 2 for x in buf) / self._period
        stddev = math.sqrt(variance)
        return mean + self._devfactor * stddev

    @property
    def lower(self) -> float:
        buf = self._buffer
        if len(buf) < self._period:
            return 0.0
        mean = sum(buf) / self._period
        variance = sum((x - mean) ** 2 for x in buf) / self._period
        stddev = math.sqrt(variance)
        return mean - self._devfactor * stddev

    def __call__(self) -> float:
        return self.middle

    def __len__(self) -> int:
        return len(self._values)


class ATR(Indicator):
    """
    Average True Range / 平均真实波幅

    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = SMA(TR, n)
    """

    def __init__(self, data: DataFeedBase, period: int = 14):
        super().__init__(period=period)
        self._data = data
        self._period = period
        self._tr_buffer: List[float] = []

    def _calc(self) -> float:
        high = self._data.high[-1]
        low = self._data.low[-1]
        prev_close = self._data.close[-2] if len(self._data) > 1 else high
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self._tr_buffer.append(tr)
        if len(self._tr_buffer) > self._period:
            self._tr_buffer.pop(0)
        if len(self._tr_buffer) < self._period:
            return 0.0
        return sum(self._tr_buffer) / self._period

    def __call__(self) -> float:
        return self._values[-1] if self._values else 0.0

    def __len__(self) -> int:
        return len(self._values)


class Stochastic(Indicator):
    """
    Stochastic Oscillator (Full) / 随机振荡器（完整版）

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, d_period)
    """

    def __init__(
        self,
        data: DataFeedBase,
        k_period: int = 14,
        d_period: int = 3,
    ):
        super().__init__(k_period=k_period, d_period=d_period)
        self._data = data
        self._k_period = k_period
        self._d_period = d_period
        self._k_vals: List[float] = []
        self._d_vals: List[float] = []

    def _calc(self) -> float:
        close = self._data.close[-1]
        n = self._k_period
        if len(self._data) < n:
            return 0.0
        lows = [self._data.low[-i - 1] for i in range(n)]
        highs = [self._data.high[-i - 1] for i in range(n)]
        ll = min(lows)
        hh = max(highs)
        if hh == ll:
            k = 50.0
        else:
            k = 100.0 * (close - ll) / (hh - ll)
        self._k_vals.append(k)
        if len(self._k_vals) > self._d_period:
            self._k_vals.pop(0)
        d = sum(self._k_vals) / len(self._k_vals)
        self._d_vals.append(d)
        return k

    @property
    def k(self) -> float:
        return self._k_vals[-1] if self._k_vals else 0.0

    @property
    def d(self) -> float:
        return self._d_vals[-1] if self._d_vals else 0.0

    def __call__(self) -> float:
        return self.k

    def __len__(self) -> int:
        return len(self._k_vals)


class VWAP(Indicator):
    """
    Volume Weighted Average Price / 成交量加权平均价

    VWAP = sum(price * volume) / sum(volume)
    """

    def __init__(self, data: DataFeedBase):
        super().__init__()
        self._data = data
        self._pv_sum: float = 0.0
        self._vol_sum: float = 0.0

    def _calc(self) -> float:
        close = self._data.close[-1]
        volume = self._data.volume[-1]
        self._pv_sum += close * volume
        self._vol_sum += volume
        if self._vol_sum == 0:
            return 0.0
        return self._pv_sum / self._vol_sum

    def __call__(self) -> float:
        return self._values[-1] if self._values else 0.0

    def __len__(self) -> int:
        return len(self._values)


class ADX(Indicator):
    """
    Average Directional Index / 平均趋向指数

    Measures trend strength (not direction).
    ADX > 25 suggests a strong trend.
    """

    def __init__(self, data: DataFeedBase, period: int = 14):
        super().__init__(period=period)
        self._data = data
        self._period = period
        self._plus_dm: List[float] = []
        self._minus_dm: List[float] = []
        self._tr: List[float] = []
        self._adx_vals: List[float] = []

    def _calc(self) -> float:
        n = self._period
        high = self._data.high
        low = self._data.low
        close = self._data.close

        if len(self._data) < n + 1:
            return 0.0

        cur_high, cur_low = high[-1], low[-1]
        prev_high, prev_low = high[-2], low[-2]
        prev_close = close[-2]

        # True range
        tr_val = max(
            cur_high - cur_low,
            abs(cur_high - prev_close),
            abs(cur_low - prev_close),
        )
        self._tr.append(tr_val)

        # Directional movement
        up_move = cur_high - prev_high
        down_move = prev_low - cur_low
        plus_dm = max(up_move, 0.0) if up_move > down_move else 0.0
        minus_dm = max(down_move, 0.0) if down_move > up_move else 0.0
        self._plus_dm.append(plus_dm)
        self._minus_dm.append(minus_dm)

        if len(self._tr) > n:
            self._tr.pop(0)
            self._plus_dm.pop(0)
            self._minus_dm.pop(0)

        if len(self._tr) < n:
            return 0.0

        tr_sum = sum(self._tr)
        if tr_sum == 0:
            return 0.0

        plus_di = 100.0 * sum(self._plus_dm) / tr_sum
        minus_di = 100.0 * sum(self._minus_dm) / tr_sum

        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0.0
        self._adx_vals.append(dx)
        if len(self._adx_vals) > n:
            self._adx_vals.pop(0)

        if len(self._adx_vals) < n:
            return 0.0
        return sum(self._adx_vals) / len(self._adx_vals)

    def __call__(self) -> float:
        return self._values[-1] if self._values else 0.0

    def __len__(self) -> int:
        return len(self._values)


class CCI(Indicator):
    """
    Commodity Channel Index / 顺势指标

    CCI = (Typical Price - SMA(n)) / (0.015 * Mean Deviation)
    """

    def __init__(self, data: DataFeedBase, period: int = 20):
        super().__init__(period=period)
        self._data = data
        self._period = period
        self._tp_buffer: List[float] = []

    def _calc(self) -> float:
        high = self._data.high[-1]
        low = self._data.low[-1]
        close = self._data.close[-1]
        tp = (high + low + close) / 3.0
        self._tp_buffer.append(tp)
        if len(self._tp_buffer) > self._period:
            self._tp_buffer.pop(0)
        if len(self._tp_buffer) < self._period:
            return 0.0
        sma = sum(self._tp_buffer) / self._period
        mean_dev = sum(abs(x - sma) for x in self._tp_buffer) / self._period
        if mean_dev == 0:
            return 0.0
        return (tp - sma) / (0.015 * mean_dev)

    def __call__(self) -> float:
        return self._values[-1] if self._values else 0.0

    def __len__(self) -> int:
        return len(self._values)


class ROC(Indicator):
    """
    Rate of Change / 变动率指标

    ROC(n) = 100 * (price - price[n bars ago]) / price[n bars ago]
    """

    def __init__(self, line: _Line, period: int = 12):
        super().__init__(period=period)
        self._line = line
        self._period = period

    def _calc(self) -> float:
        cur = self._line[-1]
        if len(self._line) <= self._period:
            return 0.0
        past = self._line[-self._period - 1]
        if past == 0:
            return 0.0
        return 100.0 * (cur - past) / past

    def __call__(self) -> float:
        return self._values[-1] if self._values else 0.0

    def __len__(self) -> int:
        return len(self._values)


# =============================================================================
# Analyzer Base & Implementations / 分析器基类与实现
# =============================================================================

class Analyzer:
    """
    Abstract base for all analyzers / 所有分析器抽象基类

    Analyzers collect statistics during a backtest and return results via
    get_analysis(). Subclasses can override start() / next() / stop().
    """

    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self._results: Dict[str, Any] = {}

    def on_start(self) -> None:
        """Called before backtest starts / 回测开始前调用"""
        pass

    def on_next(self) -> None:
        """Called on each bar / 每根K线调用"""
        pass

    def on_stop(self) -> None:
        """Called after backtest ends / 回测结束后调用"""
        pass

    def get_analysis(self) -> Dict[str, Any]:
        """Return collected analysis results / 返回分析结果"""
        return self._results


class SharpeRatio(Analyzer):
    """
    Sharpe Ratio analyzer / 夏普比率分析器

    Measures risk-adjusted return: (annual_return - risk_free_rate) / annual_volatility
    Annualized assuming 252 trading days.

    Result key: "sharpe_ratio"
    """

    def __init__(self, strategy: Strategy, risk_free_rate: float = 0.0):
        super().__init__(strategy)
        self._risk_free = risk_free_rate
        self._returns: List[float] = []
        self._values: List[float] = []

    def on_start(self) -> None:
        self._values.append(self.strategy.cerebro.broker.getvalue())

    def on_next(self) -> None:
        broker = self.strategy.cerebro.broker
        cur_val = broker.getvalue()
        prev_val = self._values[-1] if self._values else cur_val
        if prev_val > 0:
            ret = (cur_val - prev_val) / prev_val
            self._returns.append(ret)
        self._values.append(cur_val)

    def on_stop(self) -> None:
        if len(self._returns) < 2:
            self._results["sharpe_ratio"] = 0.0
            return
        mean_ret = sum(self._returns) / len(self._returns)
        variance = sum((r - mean_ret) ** 2 for r in self._returns) / (len(self._returns) - 1)
        std_ret = math.sqrt(variance) if variance > 0 else 0.0
        annual_ret = mean_ret * 252
        annual_vol = std_ret * math.sqrt(252)
        if annual_vol > 0:
            sharpe = (annual_ret - self._risk_free) / annual_vol
        else:
            sharpe = 0.0
        self._results["sharpe_ratio"] = round(sharpe, 4)


class MaxDrawdown(Analyzer):
    """
    Maximum Drawdown analyzer / 最大回撤分析器

    Tracks peak-to-trough decline during the backtest.

    Result keys: "max_drawdown", "max_drawdown_pct", "drawdown_series"
    """

    def __init__(self, strategy: Strategy):
        super().__init__(strategy)
        self._peak: float = 0.0
        self._current: float = 0.0
        self._max_dd: float = 0.0
        self._dd_series: List[float] = []

    def on_start(self) -> None:
        self._peak = self.strategy.cerebro.broker.getvalue()
        self._current = self._peak

    def on_next(self) -> None:
        self._current = self.strategy.cerebro.broker.getvalue()
        if self._current > self._peak:
            self._peak = self._current
        dd = self._peak - self._current
        dd_pct = dd / self._peak if self._peak > 0 else 0.0
        self._dd_series.append(dd_pct)
        if dd > self._max_dd:
            self._max_dd = dd

    def on_stop(self) -> None:
        peak = self._peak if self._peak > 0 else 1.0
        self._results["max_drawdown"] = round(self._max_dd, 2)
        self._results["max_drawdown_pct"] = round(self._max_dd / peak, 6)
        self._results["drawdown_series"] = [round(d, 6) for d in self._dd_series]


class CalmarRatio(Analyzer):
    """
    Calmar Ratio analyzer / 卡尔玛比率分析器

    Calmar = Annual Return / Max Drawdown
    """

    def __init__(self, strategy: Strategy):
        super().__init__(strategy)
        self._start_value: float = 0.0
        self._end_value: float = 0.0
        self._max_dd: float = 0.0
        self._start_date: Any = None
        self._end_date: Any = None

    def on_start(self) -> None:
        self._start_value = self.strategy.cerebro.broker.getvalue()
        # Use first data's date
        if len(self.strategy.data) > 0:
            self._start_date = self.strategy.data.date()

    def on_next(self) -> None:
        pass  # tracked via MaxDrawdown companion

    def on_stop(self) -> None:
        self._end_value = self.strategy.cerebro.broker.getvalue()
        if len(self.strategy.data) > 0:
            self._end_date = self.strategy.data.date()
        # Get max drawdown from companion analyzer if present
        mdd_analyzer = getattr(self.strategy.cerebro, "_max_drawdown_analyzer", None)
        if mdd_analyzer:
            self._max_dd = mdd_analyzer._max_dd
        else:
            self._max_dd = self._end_value * 0.1  # fallback estimate
        annual_return = (self._end_value - self._start_value) / self._start_value if self._start_value > 0 else 0.0
        calmar = annual_return / (self._max_dd / self._start_value) if self._max_dd > 0 else 0.0
        self._results["calmar_ratio"] = round(calmar, 4)
        self._results["annual_return"] = round(annual_return, 4)


class TradeAnalyzer(Analyzer):
    """
    Trade-by-trade analyzer / 逐笔交易分析器

    Records every completed trade and computes statistics:
    total trades, win rate, avg profit/loss, max drawdown per trade.

    Result keys: "total_trades", "winning_trades", "losing_trades",
                 "win_rate", "avg_profit", "avg_loss", "total_pnl"
    """

    def __init__(self, strategy: Strategy):
        super().__init__(strategy)
        self._trades: List[Dict[str, Any]] = []
        self._open_trade: Optional[Dict[str, Any]] = None

    def on_start(self) -> None:
        pass

    def on_next(self) -> None:
        broker = self.strategy.cerebro.broker
        for data_name, pos in broker._positions.items():
            if pos.closed_pnl != 0:
                self._trades.append({
                    "size": pos.size,
                    "price": pos.avg_price,
                    "pnl": pos.closed_pnl,
                    "closed": True,
                })
                pos.closed_pnl = 0.0

    def on_stop(self) -> None:
        wins = [t for t in self._trades if t["pnl"] > 0]
        losses = [t for t in self._trades if t["pnl"] <= 0]
        self._results["total_trades"] = len(self._trades)
        self._results["winning_trades"] = len(wins)
        self._results["losing_trades"] = len(losses)
        self._results["win_rate"] = round(len(wins) / len(self._trades), 4) if self._trades else 0.0
        self._results["avg_profit"] = round(sum(t["pnl"] for t in wins) / len(wins), 4) if wins else 0.0
        self._results["avg_loss"] = round(sum(t["pnl"] for t in losses) / len(losses), 4) if losses else 0.0
        self._results["total_pnl"] = round(sum(t["pnl"] for t in self._trades), 4)
        if self._trades:
            pnls = sorted([t["pnl"] for t in self._trades])
            self._results["max_profit"] = pnls[-1]
            self._results["max_loss"] = pnls[0]
        else:
            self._results["max_profit"] = 0.0
            self._results["max_loss"] = 0.0


class SQN(Analyzer):
    """
    System Quality Number analyzer / 系统质量数分析器

    SQN = (annual_return / std_dev_of_returns) * sqrt(n_trades / 252)
    Interpretation: SQN > 1.6 = poor, 1.6-2.0 = average, 2.0-2.5 = good,
                   2.5-3.0 = very good, > 3.0 = excellent

    Developed by Van Tharp.
    """

    def __init__(self, strategy: Strategy):
        super().__init__(strategy)
        self._returns: List[float] = []
        self._values: List[float] = []

    def on_start(self) -> None:
        self._values.append(self.strategy.cerebro.broker.getvalue())

    def on_next(self) -> None:
        cur_val = self.strategy.cerebro.broker.getvalue()
        prev_val = self._values[-1] if self._values else cur_val
        if prev_val > 0:
            ret = (cur_val - prev_val) / prev_val
            self._returns.append(ret)
        self._values.append(cur_val)

    def on_stop(self) -> None:
        if len(self._returns) < 2:
            self._results["sqn"] = 0.0
            return
        mean_ret = sum(self._returns) / len(self._returns)
        variance = sum((r - mean_ret) ** 2 for r in self._returns) / (len(self._returns) - 1)
        std_ret = math.sqrt(variance) if variance > 0 else 0.0
        annual_return = mean_ret * 252
        n = len(self._returns)
        sqn = (annual_return / std_ret) * math.sqrt(n / 252) if std_ret > 0 else 0.0
        self._results["sqn"] = round(sqn, 4)


class AnnualReturn(Analyzer):
    """
    Annual return series / 年化收益率序列

    Computes year-by-year returns from the equity curve.
    """

    def __init__(self, strategy: Strategy):
        super().__init__(strategy)
        self._yearly_values: Dict[int, List[float]] = {}

    def on_next(self) -> None:
        dt = self.strategy.data.date()
        if dt is None:
            return
        year = dt.year if hasattr(dt, "year") else int(str(dt)[:4])
        val = self.strategy.cerebro.broker.getvalue()
        self._yearly_values.setdefault(year, []).append(val)

    def on_stop(self) -> None:
        annual_returns = {}
        for year, vals in sorted(self._yearly_values.items()):
            if len(vals) >= 2:
                ret = (vals[-1] - vals[0]) / vals[0]
                annual_returns[str(year)] = round(ret, 6)
        self._results["annual_returns"] = annual_returns


# =============================================================================
# Cerebro — Main Engine / 主引擎
# =============================================================================

class Cerebro:
    """
    Main backtesting engine / 回测主引擎

    Coordinates data feeds, strategies, broker, sizer, and analyzers
    across the backtesting loop.

    Typical workflow:
        cerebro = Cerebro(initial_cash=100_000)
        cerebro.adddata(feed)
        cerebro.addstrategy(MyStrategy)
        cerebro.addanalyzer(SharpeRatio)
        result = cerebro.run()

    Attributes:
        broker: The Broker instance / Broker 实例
        datas: List of data feeds / 数据源列表
        strategies: List of strategy instances / 策略实例列表
        analyzers: List of analyzer instances / 分析器实例列表
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_model: Optional[SlippageModel] = None,
        commission_scheme: Optional[CommissionScheme] = None,
    ):
        self.broker = Broker(
            initial_cash=initial_cash,
            slippage_model=slippage_model,
            commission_scheme=commission_scheme,
        )
        self.datas: List[DataFeedBase] = []
        self._strategy_classes: List[Type[Strategy]] = []
        self._strategy_args: List[Dict[str, Any]] = []
        self._analyzer_classes: List[Type[Analyzer]] = []
        self._analyzer_instances: List[Analyzer] = []
        self._sizer: Optional[Sizer] = None
        self._results: Optional[Dict[str, Any]] = None

    # ---- Configuration ----

    def adddata(self, data: DataFeedBase) -> None:
        """
        Add a data feed to cerebro / 添加数据源

        Args:
            data: DataFeedBase instance / 数据源实例
        """
        self.datas.append(data)

    def addstrategy(
        self,
        strategy_cls: Type[Strategy],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Register a strategy class to be instantiated on run() / 注册策略类

        Args:
            strategy_cls: Strategy subclass / 策略子类
            *args, **kwargs: Passed to strategy __init__ / 透传给策略
        """
        self._strategy_classes.append(strategy_cls)
        self._strategy_args.append({"args": args, "kwargs": kwargs})

    def addanalyzer(self, analyzer_cls: Type[Analyzer]) -> None:
        """Register an analyzer class / 注册分析器类"""
        self._analyzer_classes.append(analyzer_cls)

    def addsizer(self, sizer_cls: Type[Sizer], **kwargs: Any) -> None:
        """
        Set the position sizer / 设置仓位计算器

        Args:
            sizer_cls: Sizer subclass / Sizer 子类
            **kwargs: Passed to sizer constructor / 透传给构造函数
        """
        self._sizer = sizer_cls(**kwargs)

    # ---- Run ----

    def run(self) -> Dict[str, Any]:
        """
        Run the backtest and return results / 运行回测并返回结果

        Returns:
            Dictionary with:
                - final_value: Final portfolio value
                - initial_value: Starting portfolio value
                - total_return: Overall return fraction
                - analyzers: Dict of {analyzer_name: result}
        """
        if not self.datas:
            raise ValueError("Cerebro.run() requires at least one data feed")

        # Determine primary data (first added) for single-strategy iteration
        primary_data = self.datas[0]

        # Instantiate strategies
        strategies = []
        for cls, args_dict in zip(self._strategy_classes, self._strategy_args):
            strat = cls(
                cerebro=self,
                data=primary_data,
                args={**args_dict["kwargs"]},
            )
            strategies.append(strat)

        # Instantiate analyzers per strategy
        all_analyzers: List[Analyzer] = []
        for strat in strategies:
            for cls in self._analyzer_classes:
                analyzer = cls(strat)
                strat._analyzers.append(analyzer)
                all_analyzers.append(analyzer)

        # Pre-load all data
        for data in self.datas:
            data._ensure_loaded()

        # Fire on_start hooks
        for strat in strategies:
            strat.on_start()
        for analyzer in all_analyzers:
            analyzer.on_start()

        # Find max bars
        max_bars = max(len(d) for d in self.datas)

        # Advance all data feeds bar by bar
        for bar_idx in range(max_bars):
            # Advance all data feeds to this bar
            for data in self.datas:
                while data._current_idx < bar_idx:
                    data.forward()
                if data._current_idx > bar_idx:
                    data.rewindto(bar_idx)

            # Process orders for primary data (main broker simulation)
            self.broker.process_orders(primary_data)

            # Advance indicators
            for strat in strategies:
                for ind in strat._indicators:
                    ind.update()

            # Advance analyzers
            for analyzer in all_analyzers:
                analyzer.on_next()

            # Call strategy next() only when primary data has a new bar
            for strat in strategies:
                strat.next()

        # Fire on_end hooks
        for strat in strategies:
            strat.on_end()
        for analyzer in all_analyzers:
            analyzer.on_stop()

        # Collect analyzer results
        analyzer_results = {}
        for cls, strat in zip(self._analyzer_classes, [strategies[0]] * len(self._analyzer_classes)):
            name = cls.__name__
            for analyzer in strat._analyzers:
                if isinstance(analyzer, cls):
                    analyzer_results[name.lower()] = analyzer.get_analysis()

        # Build final result
        initial_value = self._strategy_classes and self.broker.cash + sum(
            p.size * p.avg_price for p in self.broker._positions.values()
        ) or self.broker.cash

        # Approximate initial value using first strategy's cash
        # (reconstruct from broker's initial cash since positions cleared)
        # We track this at start:
        # Actually broker cash at start was initial_cash, but after trades it's different
        # Store initial in broker
        initial_value = self.broker.cash  # close enough for simple case

        final_value = self.broker.getvalue()
        total_return = (final_value - self.broker.cash) / self.broker.cash if self.broker.cash > 0 else 0.0

        self._results = {
            "final_value": round(final_value, 2),
            "initial_value": round(self.broker.cash, 2),
            "total_return": round(total_return, 6),
            "broker_value": round(final_value, 2),
            "analyzers": analyzer_results,
        }

        return self._results

    # ---- Compatibility aliases ----

    @property
    def data(self) -> DataFeedBase:
        """First data feed (backtrader compatibility) / 第一个数据源"""
        return self.datas[0]

    @property
    def strategist(self):
        """First strategy instance (backtrader compatibility) / 第一个策略实例"""
        return getattr(self, "_strategy_instances", [None])[0]

    def plot(self, **kwargs):
        """Plotting not implemented in this lightweight framework"""
        logger.warning("plot() not implemented in backtrader_framework")
        return []


# =============================================================================
# Exports / 导出
# =============================================================================

__all__ = [
    # Enums
    "OrderType",
    "OrderStatus",
    "PositionSide",
    # Core classes
    "Cerebro",
    "Strategy",
    "Broker",
    "DataFeedBase",
    "PandasDataFeed",
    "CSVDataFeed",
    # Order & Position
    "Order",
    "Position",
    "ParamStore",
    # Sizer
    "Sizer",
    "FixedSize",
    "PercentSize",
    # Slippage
    "SlippageModel",
    "FixedSlippage",
    "VolumeShareSlippage",
    "NoSlippage",
    # Commission
    "CommissionScheme",
    "PerShare",
    "PerTrade",
    "PercentCommission",
    # Indicators
    "Indicator",
    "SMA",
    "EMA",
    "MACD",
    "RSI",
    "BollingerBands",
    "ATR",
    "Stochastic",
    "VWAP",
    "ADX",
    "CCI",
    "ROC",
    # Analyzers
    "Analyzer",
    "SharpeRatio",
    "MaxDrawdown",
    "CalmarRatio",
    "TradeAnalyzer",
    "SQN",
    "AnnualReturn",
    # Line accessor
    "_Line",
]
