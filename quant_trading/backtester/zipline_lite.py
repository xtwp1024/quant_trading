# -*- coding: utf-8 -*-
"""
Zipline-Lite: Lightweight Event-Driven Backtester
zipline_lite: 轻量级事件驱动回测引擎

Inspired by Quantopian's Zipline (https://github.com/quantopian/zipline).
Pure Python + pandas implementation. No Cython dependency.

Design philosophy:
    Event-driven architecture: BAR, ORDER, CANCEL, FILL, DIVIDEND events flow
    through a central EventDrivenBacktester. Every bar the strategy's
    handle_data() is called, orders are processed through slippage + commission
    models, portfolio is updated, and performance metrics are recorded.

Classes
-------
EventDrivenBacktester  — Core event loop and simulation runner
SlippageModel           — Base class for slippage / market impact models
VolumeShareSlippage     — Quadratic volume-share price impact (Zipline default)
FixedSlippage           — Fixed spread per trade
NoSlippage              — Fill at close price with no impact (testing)
CommissionModel         — Base class for commission models
PerShareCommission       — Per-share commission (Zipline default: $0.001/sh)
PerTradeCommission      — Flat fee per trade
PercentCommission       — Percentage of trade value
NoCommission            — Free trading (testing)
Portfolio               — Tracks positions, cash, NAV, daily returns
DataPortal              — Historical data access with LRU caching
PipelineEngine          — Batch factor computation across asset/time grids
TradingAlgorithm        — User strategy interface (initialize / handle_data)

Usage
-----
```python
from quant_trading.backtester.zipline_lite import (
    EventDrivenBacktester, TradingAlgorithm, SlippageModel, Portfolio,
    VolumeShareSlippage, PerShareCommission,
)

class MyStrategy(TradingAlgorithm):
    def initialize(self, algo):
        algo.slippage = VolumeShareSlippage(volume_limit=0.025, price_impact=0.1)
        algo.commission = PerShareCommission(cost=0.001)
        algo.set_benchmark(lambda dt: 0.0)

    def handle_data(self, algo, data):
        for symbol in data.symbols:
            if data.current(symbol, 'close') < data.current(symbol, 'short_mavg'):
                algo.order(symbol, 100)

bt = EventDrivenBacktester(
    capital_base=100_000,
    data_frequency='daily',       # 'daily' or 'minute'
    emission_rate='daily',        # 'daily' or 'minute' performance emission
)
result = bt.run(MyStrategy(), history)
# result: { 'orders', 'transactions', 'dividends', 'portfolio',
#           'daily_returns', 'metrics' }
```
"""

from __future__ import annotations

import math
import uuid
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "EventDrivenBacktester",
    "SlippageModel",
    "VolumeShareSlippage",
    "FixedSlippage",
    "NoSlippage",
    "CommissionModel",
    "PerShareCommission",
    "PerTradeCommission",
    "PercentCommission",
    "NoCommission",
    "Portfolio",
    "DataPortal",
    "PipelineEngine",
    "TradingAlgorithm",
    # events
    "EventType",
    "Event",
    "BarEvent",
    "OrderEvent",
    "FillEvent",
    "CancelEvent",
    "DividendEvent",
]


# ---------------------------------------------------------------------------
# Event System / 事件系统
# ---------------------------------------------------------------------------

class EventType(Enum):
    """Event type enumeration / 事件类型枚举"""
    BAR      = auto()   # Price bar arrived / 价格 bar 到达
    ORDER    = auto()   # Order submitted / 订单提交
    CANCEL   = auto()   # Order cancelled / 订单取消
    FILL     = auto()   # Order filled / 订单成交
    DIVIDEND = auto()   # Dividend paid / 分红发放
    SESSION_START = auto()
    SESSION_END   = auto()


class Event:
    """Lightweight event object / 轻量级事件对象"""
    dt: datetime
    event_type: EventType
    payload: Dict[str, Any]

    def __init__(self, dt: datetime, event_type: EventType, payload: Optional[Dict[str, Any]] = None):
        self.dt = dt
        self.event_type = event_type
        self.payload = payload if payload is not None else {}

    def __repr__(self):
        return f"Event({self.dt:%Y-%m-%d %H:%M}, {self.event_type.name})"


class BarEvent(Event):
    """Price bar event / 价格 bar 事件"""

    def __init__(
        self,
        dt: datetime,
        symbol: str = "",
        open_: float = float("nan"),
        high: float = float("nan"),
        low: float = float("nan"),
        close: float = float("nan"),
        volume: int = 0,
    ):
        super().__init__(dt, EventType.BAR, {
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        self.symbol = symbol
        self.open_ = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class OrderEvent(Event):
    """Order submission event / 订单提交事件"""

    def __init__(
        self,
        dt: datetime,
        order_id: str = "",
        symbol: str = "",
        amount: int = 0,
        direction: int = 0,
        style: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ):
        super().__init__(dt, EventType.ORDER, {
            "order_id": order_id,
            "symbol": symbol,
            "amount": amount,
            "direction": direction,
            "style": style,
            "limit_price": limit_price,
            "stop_price": stop_price,
        })
        self.order_id = order_id
        self.symbol = symbol
        self.amount = amount
        self.direction = direction
        self.style = style
        self.limit_price = limit_price
        self.stop_price = stop_price


class FillEvent(Event):
    """Order fill event / 订单成交事件"""

    def __init__(
        self,
        dt: datetime,
        order_id: str = "",
        symbol: str = "",
        amount: int = 0,
        price: float = 0.0,
        commission: float = 0.0,
        slippage_cost: float = 0.0,
    ):
        super().__init__(dt, EventType.FILL, {
            "order_id": order_id,
            "symbol": symbol,
            "amount": amount,
            "price": price,
            "commission": commission,
            "slippage_cost": slippage_cost,
        })
        self.order_id = order_id
        self.symbol = symbol
        self.amount = amount
        self.price = price
        self.commission = commission
        self.slippage_cost = slippage_cost


class CancelEvent(Event):
    """Order cancellation event / 订单取消事件"""

    def __init__(self, dt: datetime, order_id: str = "", reason: str = ""):
        super().__init__(dt, EventType.CANCEL, {"order_id": order_id, "reason": reason})
        self.order_id = order_id
        self.reason = reason


class DividendEvent(Event):
    """Dividend event / 分红事件"""

    def __init__(
        self,
        dt: datetime,
        symbol: str = "",
        amount: float = 0.0,
        share_count: int = 0,
    ):
        super().__init__(dt, EventType.DIVIDEND, {
            "symbol": symbol,
            "amount": amount,
            "share_count": share_count,
        })
        self.symbol = symbol
        self.amount = amount
        self.share_count = share_count


# ---------------------------------------------------------------------------
# Slippage Models / 滑点模型
# ---------------------------------------------------------------------------

class SlippageModel(metaclass=ABCMeta):
    """
    Abstract base class for slippage models.
    滑点模型抽象基类。

    Slippage models determine the price at which an order is filled and the
    number of shares filled in the current bar.

    Subclasses must implement :meth:`process_order`.
    """

    @abstractmethod
    def process_order(
        self,
        bar: BarEvent,
        order: "Order",
        volume_for_bar: int,
    ) -> Tuple[Optional[float], Optional[int]]:
        """
        Compute execution price and volume for an order in the current bar.

        Parameters
        ----------
        bar : BarEvent
            Current price bar.
        order : Order
            The order to simulate.
        volume_for_bar : int
            Shares already filled for this asset in the current bar.

        Returns
        -------
        (execution_price, execution_volume)
            execution_price is None means no fill this bar.
            execution_volume is signed (positive=buy, negative=sell).
        """
        raise NotImplementedError("process_order")

    def simulate(
        self,
        bar: BarEvent,
        orders: List["Order"],
        volume_for_bar: int = 0,
    ) -> List[Tuple["Order", FillEvent]]:
        """
        Simulate fills for all open orders of a single asset in a single bar.

        Yields (order, fill_event) tuples for orders that were filled.
        Raises ``LiquidityExceeded`` if volume limit is hit.
        """
        self._volume_for_bar = volume_for_bar
        fills: List[Tuple["Order", FillEvent]] = []

        for order in orders:
            if order.open_amount == 0:
                continue

            price = bar.close
            if price != price:   # NaN check
                continue

            # Check stop/limit triggers
            if not order.check_triggers(price, bar.dt):
                continue

            try:
                exec_price, exec_vol = self.process_order(bar, order, self._volume_for_bar)
            except LiquidityExceeded:
                break

            if exec_price is None or exec_vol is None or exec_vol == 0:
                continue

            fill = FillEvent(
                dt=bar.dt,
                order_id=order.id,
                symbol=order.symbol,
                amount=int(exec_vol),
                price=float(exec_price),
            )
            fills.append((order, fill))
            self._volume_for_bar += abs(exec_vol)

        return fills


class LiquidityExceeded(Exception):
    """Raised when a slippage model exhausts available volume in a bar."""
    pass


class NoSlippage(SlippageModel):
    """
    Fill immediately at close price with no market impact.
    无滑点模型：以收盘价即时成交，无市场影响。

    Primarily used for testing.
    """

    def process_order(
        self,
        bar: BarEvent,
        order: "Order",
        volume_for_bar: int,
    ) -> Tuple[float, int]:
        return bar.close, order.open_amount


class FixedSlippage(SlippageModel):
    """
    Apply a fixed half-spread to all fills.
    固定滑点模型：对所有成交应用固定价差（买入加半差，卖出减半差）。

    Parameters
    ----------
    spread : float
        Total bid-ask spread in price units. Buy orders add spread/2,
        sell orders subtract spread/2. Default 0.0.
    """

    def __init__(self, spread: float = 0.0):
        self.spread = spread

    def process_order(
        self,
        bar: BarEvent,
        order: "Order",
        volume_for_bar: int,
    ) -> Tuple[float, int]:
        direction = 1 if order.amount > 0 else -1
        exec_price = bar.close + direction * (self.spread / 2.0)
        return exec_price, order.open_amount


class VolumeShareSlippage(SlippageModel):
    """
    Model slippage as a quadratic function of volume share.
    成交量比例滑点模型：以成交量占比的二次函数建模滑点。

    Buy orders fill at ``close * (1 + price_impact * volume_share**2)``.
    Sell orders fill at ``close * (1 - price_impact * volume_share**2)``.

    Parameters
    ----------
    volume_limit : float
        Maximum fraction of bar volume that can be filled per bar.
        Default 0.025 (2.5%). 0.5 means 50%.
    price_impact : float
        Scaling coefficient for price impact. Default 0.1.
        Larger values = more price impact.
    """

    DEFAULT_VOLUME_LIMIT = 0.025
    DEFAULT_PRICE_IMPACT = 0.1

    def __init__(
        self,
        volume_limit: float = DEFAULT_VOLUME_LIMIT,
        price_impact: float = DEFAULT_PRICE_IMPACT,
    ):
        self.volume_limit = volume_limit
        self.price_impact = price_impact

    def process_order(
        self,
        bar: BarEvent,
        order: "Order",
        volume_for_bar: int,
    ) -> Tuple[Optional[float], Optional[int]]:
        if bar.volume <= 0:
            return None, None

        max_volume = self.volume_limit * bar.volume
        remaining = max_volume - volume_for_bar
        if remaining < 1:
            raise LiquidityExceeded()

        shares_to_fill = min(abs(order.open_amount), int(remaining))
        if shares_to_fill < 1:
            return None, None

        total_volume = volume_for_bar + shares_to_fill
        volume_share = min(total_volume / bar.volume, self.volume_limit)

        price = bar.close
        direction = 1 if order.amount > 0 else -1

        # Check limit price
        if order.limit_price is not None:
            impacted = price + direction * self.price_impact * (volume_share**2) * price
            if direction > 0 and impacted > order.limit_price:
                return None, None
            if direction < 0 and impacted < order.limit_price:
                return None, None

        impacted_price = price + direction * self.price_impact * (volume_share**2) * price
        return impacted_price, int(math.copysign(shares_to_fill, order.amount))


# ---------------------------------------------------------------------------
# Commission Models / 佣金模型
# ---------------------------------------------------------------------------

class CommissionModel(metaclass=ABCMeta):
    """
    Abstract base class for commission models.
    佣金模型抽象基类。

    Commission models calculate the cost of executing a transaction.
    """

    @abstractmethod
    def calculate(
        self,
        order: "Order",
        fill: FillEvent,
    ) -> float:
        """
        Calculate the commission for a fill.
        计算成交的佣金费用。

        Parameters
        ----------
        order : Order
            The order being filled.
        fill : FillEvent
            The fill event.

        Returns
        -------
        float
            Commission charged in currency units.
        """
        raise NotImplementedError("calculate")


class NoCommission(CommissionModel):
    """Commission-free trading / 免佣金模型（测试用）"""

    def calculate(self, order: "Order", fill: FillEvent) -> float:
        return 0.0


class PerShareCommission(CommissionModel):
    """
    Commission per share traded.
    按股数佣金模型：每股收取固定佣金。

    Zipline default for equities.

    Parameters
    ----------
    cost : float
        Commission per share. Default $0.001 (0.1 cent/sh).
    min_cost : float
        Minimum commission per trade. Default $0.0.
    """

    DEFAULT_COST = 0.001

    def __init__(self, cost: float = DEFAULT_COST, min_cost: float = 0.0):
        self.cost_per_share = float(cost)
        self.min_cost = float(min_cost)

    def calculate(self, order: "Order", fill: FillEvent) -> float:
        shares = abs(fill.amount)
        cost = shares * self.cost_per_share
        if order.commission_paid == 0:
            cost = max(cost, self.min_cost)
        return max(0.0, cost)


class PerTradeCommission(CommissionModel):
    """
    Flat fee per trade, regardless of share count.
    按笔佣金模型：每笔交易收取固定费用。

    Parameters
    ----------
    cost : float
        Fixed commission per trade. Default $0.0.
    """

    def __init__(self, cost: float = 0.0):
        self.cost = float(cost)

    def calculate(self, order: "Order", fill: FillEvent) -> float:
        if order.commission_paid == 0:
            return self.cost
        return 0.0


class PercentCommission(CommissionModel):
    """
    Commission as a percentage of trade value.
    比例佣金模型：按成交金额的百分比收取佣金。

    Parameters
    ----------
    rate : float
        Commission rate (e.g., 0.001 = 0.1% of trade value).
    """

    def __init__(self, rate: float = 0.0):
        self.rate = float(rate)

    def calculate(self, order: "Order", fill: FillEvent) -> float:
        return abs(fill.amount * fill.price * self.rate)


# ---------------------------------------------------------------------------
# Order / 订单
# ---------------------------------------------------------------------------

class OrderStatus(Enum):
    OPEN     = auto()
    FILLED   = auto()
    CANCELLED= auto()
    REJECTED = auto()
    HELD     = auto()


class Order:
    """
    Represents a trading order.
    交易订单。

    Attributes
    ----------
    id : str
        Unique order identifier.
    dt : datetime
        Order creation timestamp.
    symbol : str
        Ticker symbol.
    amount : int
        Signed quantity: positive=buy, negative=sell.
    filled : int
        Number of shares already filled.
    commission_paid : float
        Total commission paid so far.
    status : OrderStatus
        Current order status.
    limit_price : float, optional
        Limit price for limit orders.
    stop_price : float, optional
        Stop price for stop orders.
    """

    # Direction constants
    BUY  = +1
    SELL = -1

    def __init__(
        self,
        dt: datetime,
        symbol: str,
        amount: int,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        id: Optional[str] = None,
    ):
        self.id = id or uuid.uuid4().hex[:12]
        self.dt = dt
        self.symbol = symbol
        self.amount = amount
        self.filled = 0
        self.commission_paid = 0.0
        self.status = OrderStatus.OPEN
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.stop_reached = False
        self.limit_reached = False

    @property
    def direction(self) -> int:
        """+1 for buy (positive amount), -1 for sell (negative amount)."""
        return 1 if self.amount > 0 else -1

    @property
    def open_amount(self) -> int:
        """Shares remaining to fill."""
        return self.amount - self.filled

    @property
    def open(self) -> bool:
        return self.status == OrderStatus.OPEN

    def check_triggers(self, price: float, dt: datetime) -> bool:
        """
        Evaluate stop and limit price triggers.
        检查止损和限价触发器。

        Returns True if the order is now triggered and should fill.
        """
        if self.open_amount == 0:
            return False

        if self.stop_price is not None and not self.stop_reached:
            if self.direction > 0 and price >= self.stop_price:
                self.stop_reached = True
            elif self.direction < 0 and price <= self.stop_price:
                self.stop_reached = True

        if self.limit_price is not None and not self.limit_reached:
            if self.direction > 0 and price <= self.limit_price:
                self.limit_reached = True
            elif self.direction < 0 and price >= self.limit_price:
                self.limit_reached = True

        triggered = True
        if self.stop_price is not None and not self.stop_reached:
            triggered = False
        if self.limit_price is not None and not self.limit_reached:
            triggered = False

        if triggered:
            self.dt = dt
        return triggered

    def fill(self, amount: int, price: float, commission: float) -> None:
        """Record a fill for this order."""
        self.filled += amount
        self.commission_paid += commission
        if self.open_amount == 0:
            self.status = OrderStatus.FILLED

    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED

    def reject(self, reason: str = "") -> None:
        self.status = OrderStatus.REJECTED

    def __repr__(self):
        return (
            f"Order(id={self.id}, symbol={self.symbol}, amount={self.amount}, "
            f"filled={self.filled}, status={self.status.name})"
        )


# ---------------------------------------------------------------------------
# Portfolio / 组合
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    Represents a single asset position.
    单个资产持仓。

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    amount : int
        Number of shares held (can be negative for short).
    cost_basis : float
        Average cost per share.
    last_sale_price : float
        Most recent known price.
    """
    symbol: str
    amount: int = 0
    cost_basis: float = 0.0
    last_sale_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.amount * self.last_sale_price

    @property
    def unrealized_pnl(self) -> float:
        return self.amount * (self.last_sale_price - self.cost_basis)

    @property
    def realized_pnl(self) -> float:
        return 0.0   # simplified; full PnL tracking needs trade history

    def update(self, amount: int, price: float) -> None:
        """Apply a fill to this position."""
        if self.amount == 0 and amount != 0:
            self.cost_basis = price
        elif amount != 0:
            total_cost = self.amount * self.cost_basis + amount * price
            new_amount = self.amount + amount
            if new_amount != 0:
                self.cost_basis = total_cost / new_amount
            else:
                self.cost_basis = 0.0
        self.amount += amount
        self.last_sale_price = price
        if self.amount == 0:
            self.cost_basis = 0.0


class Portfolio:
    """
    Tracks the current state of a trading portfolio.
    跟踪交易组合的当前状态。

    Attributes
    ----------
    capital_base : float
        Initial starting capital.
    cash : float
        Available cash.
    positions : Dict[str, Position]
        Current positions keyed by symbol.
    starting_cash : float
        Initial capital (for benchmarking).
    cumulative_pnl : float
        Cumulative realized + unrealized PnL.
    cumulative_returns : float
        Cumulative return (fraction).
    daily_returns : List[float]
        Daily return series.
    nav_history : List[float]
        Net asset value over time.
    """

    def __init__(self, capital_base: float = 100_000.0):
        self.capital_base = capital_base
        self.starting_cash = capital_base
        self.cash = capital_base
        self.positions: Dict[str, Position] = {}
        self.cumulative_pnl = 0.0
        self.cumulative_returns = 0.0
        self.daily_returns: List[float] = []
        self.nav_history: List[float] = []
        self._pnl_from_opens: Dict[str, float] = defaultdict(float)

    @property
    def positions_value(self) -> float:
        """Total market value of all positions."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def net_asset_value(self) -> float:
        """Total portfolio value = cash + positions value."""
        return self.cash + self.positions_value

    @property
    def total_liquidity_value(self) -> float:
        """Alias for NAV (net asset value)."""
        return self.net_asset_value

    def total_liquidation_value(self) -> float:
        """Estimated total liquidation value including short positions."""
        return self.net_asset_value

    def update_position_price(self, symbol: str, price: float) -> None:
        """Update last sale price for a position."""
        if symbol in self.positions:
            self.positions[symbol].last_sale_price = price

    def execute_fill(self, symbol: str, amount: int, price: float, commission: float) -> None:
        """
        Apply a fill to the portfolio.
        执行成交：更新持仓和现金。

        For long positions (amount > 0): cash decreases.
        For short positions (amount < 0): cash increases.
        """
        # Update cost-basis tracking for realized PnL
        pos = self.positions.get(symbol)
        if pos is not None and pos.amount != 0 and (pos.amount + amount) == 0:
            # Closing a position
            pnl_per_share = price - pos.cost_basis
            realized_pnl = pnl_per_share * min(abs(pos.amount), abs(amount))
            self.cumulative_pnl += realized_pnl
            self._pnl_from_opens[symbol] = 0.0

        # Cash flow: buying reduces cash, selling increases cash
        # Commission is always deducted from cash
        trade_value = -amount * price  # negative for buys, positive for sells
        self.cash += trade_value - commission

        # Update or create position
        if symbol in self.positions:
            pos = self.positions[symbol]
        else:
            pos = Position(symbol=symbol)
            self.positions[symbol] = pos

        pos.update(amount, price)

        # Remove zero positions
        if pos.amount == 0:
            del self.positions[symbol]

    def process_dividend(self, symbol: str, cash_amount: float) -> None:
        """Credit cash dividend to portfolio."""
        self.cash += cash_amount

    def record_daily_return(self) -> None:
        """Calculate and store daily return."""
        nav = self.net_asset_value
        prev_nav = self.nav_history[-1] if self.nav_history else self.starting_cash
        if prev_nav != 0:
            ret = (nav - prev_nav) / prev_nav
        else:
            ret = 0.0
        self.daily_returns.append(ret)
        self.nav_history.append(nav)

    def get_state(self, dt: datetime) -> Dict[str, Any]:
        """Return a snapshot of the current portfolio state."""
        return {
            "dt": dt,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "nav": self.net_asset_value,
            "cumulative_pnl": self.cumulative_pnl,
            "positions": {
                sym: {
                    "amount": pos.amount,
                    "cost_basis": pos.cost_basis,
                    "last_sale_price": pos.last_sale_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for sym, pos in self.positions.items()
            },
        }

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of positions."""
        if not self.positions:
            return pd.DataFrame()
        records = []
        for sym, pos in self.positions.items():
            records.append({
                "symbol": sym,
                "amount": pos.amount,
                "cost_basis": pos.cost_basis,
                "last_price": pos.last_sale_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
            })
        return pd.DataFrame(records).set_index("symbol")


# ---------------------------------------------------------------------------
# DataPortal / 数据端口
# ---------------------------------------------------------------------------

class DataPortal:
    """
    Historical data access layer with LRU caching.
    历史数据访问层，带 LRU 缓存。

    Provides a clean interface to bar history and spot values,
    abstracting the underlying data source (DataFrame, HDF5, etc.).

    Parameters
    ----------
    history : pd.DataFrame or pd.Panel
        Price/volume data. Expected columns: open, high, low, close, volume.
        Index: DatetimeIndex (for single-asset) or MultiIndex (dt, symbol).
    """

    def __init__(
        self,
        history: Union[pd.DataFrame, pd.Panel],
        dividends: Optional[pd.DataFrame] = None,
        splits: Optional[pd.DataFrame] = None,
    ):
        self._history = history
        self._dividends = dividends
        self._splits = splits
        self._cache: Dict[str, Any] = {}
        self._cache_size = 200

    @staticmethod
    def from_ohlcv(
        df: pd.DataFrame,
        symbol: str = "asset",
        dividends: Optional[pd.DataFrame] = None,
        splits: Optional[pd.DataFrame] = None,
    ) -> "DataPortal":
        """
        Construct a DataPortal from a standard OHLCV DataFrame.
        从标准 OHLCV DataFrame 构造 DataPortal。

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: open, high, low, close, volume.
            Index: DatetimeIndex.
        symbol : str
            Symbol name to use as the asset identifier.
        """
        if df.index.name != "dt" and "dt" not in df.columns:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "dt"
        if symbol != "asset" and "symbol" not in df.columns:
            df = df.copy()
            df["symbol"] = symbol
        return DataPortal(history=df, dividends=dividends, splits=splits)

    @staticmethod
    def from_multi(
        df: pd.DataFrame,
        dividends: Optional[pd.DataFrame] = None,
    ) -> "DataPortal":
        """
        Construct a DataPortal from a multi-asset DataFrame.
        MultiIndex (dt, symbol), columns: open, high, low, close, volume.
        """
        return DataPortal(history=df, dividends=dividends)

    def get_spot_value(
        self,
        symbol: str,
        field: str,
        dt: datetime,
    ) -> float:
        """
        Get the most recent value for ``field`` up to ``dt``.
        获取截至 ``dt`` 的最新字段值。

        Parameters
        ----------
        symbol : str
            Asset symbol.
        field : str
            Field name: open, high, low, close, volume, price.
        dt : datetime
            Timestamp.

        Returns
        -------
        float
        """
        bar = self.get_bar_at(dt, symbol)
        if bar is None:
            return float("nan")
        val = bar.get(field, bar.get("close", float("nan")))
        return float(val) if val == val else float("nan")

    def get_bar_at(self, dt: datetime, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the bar for a symbol at (or most recently before) a datetime."""
        idx = pd.Timestamp(dt)
        h = self._history

        if isinstance(h, pd.DataFrame):
            if h.empty:
                return None
            if isinstance(h.index, pd.MultiIndex):
                try:
                    row = h.loc[idx, symbol]
                    return row.to_dict() if isinstance(row, pd.Series) else None
                except KeyError:
                    # Find most recent bar before idx
                    available = h.index.get_level_values(0)
                    before = available[available <= idx]
                    if len(before) == 0:
                        return None
                    nearest = before[-1]
                    try:
                        row = h.loc[nearest, symbol]
                        return row.to_dict() if isinstance(row, pd.Series) else None
                    except KeyError:
                        return None
            else:
                available = h.index
                before = available[available <= idx]
                if len(before) == 0:
                    return None
                nearest = before[-1]
                return h.loc[nearest].to_dict()
        return None

    def history(
        self,
        symbol: str,
        field: str,
        bar_count: int,
        frequency: str = "1d",
        end_dt: Optional[datetime] = None,
    ) -> pd.Series:
        """
        Get a trailing window of historical values.
        获取历史滚动窗口数据。

        Parameters
        ----------
        symbol : str
            Asset symbol.
        field : str
            Field name.
        bar_count : int
            Number of bars.
        frequency : str
            '1d' or '1m'.
        end_dt : datetime, optional
            End timestamp (inclusive). Defaults to last available.

        Returns
        -------
        pd.Series
            Index: DatetimeIndex, Values: field values.
        """
        cache_key = f"{symbol}:{field}:{bar_count}:{frequency}:{end_dt}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        h = self._history
        if isinstance(h, pd.DataFrame):
            if isinstance(h.index, pd.MultiIndex):
                if end_dt is None:
                    end_dt = h.index.get_level_values(0).max()
                idx = pd.Timestamp(end_dt)
                # Get bars up to and including idx
                slice_ = h.loc[:idx]
                try:
                    asset_bars = slice_.xs(symbol, level=1)
                except KeyError:
                    return pd.Series(dtype=float)
                series = asset_bars[field]
                result = series.iloc[-bar_count:]
            else:
                series = h[field]
                if end_dt is not None:
                    series = series.loc[:pd.Timestamp(end_dt)]
                result = series.iloc[-bar_count:]
        else:
            return pd.Series(dtype=float)

        if len(self._cache) >= self._cache_size:
            # Simple FIFO eviction
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = result
        return result

    def get_dividends(self, symbol: str, dt: datetime) -> List[DividendEvent]:
        """Get dividend events for a symbol occurring at a given datetime."""
        if self._dividends is None:
            return []
        divs = self._dividends
        if isinstance(divs, pd.DataFrame):
            try:
                row = divs.loc[pd.Timestamp(dt), symbol]
                amount = float(row.get("amount", 0))
                if amount > 0:
                    return [DividendEvent(
                        dt=dt,
                        symbol=symbol,
                        amount=amount,
                        share_count=1,
                    )]
            except (KeyError, TypeError):
                pass
        return []

    def get_splits(self, symbols: List[str], dt: datetime) -> List[Tuple[str, float]]:
        """Get splits occurring at a given datetime for given symbols."""
        if self._splits is None:
            return []
        splits = self._splits
        if isinstance(splits, pd.DataFrame):
            try:
                row = splits.loc[pd.Timestamp(dt)]
                return [
                    (sym, float(row[sym]))
                    for sym in symbols
                    if sym in row and pd.notna(row[sym]) and row[sym] != 1.0
                ]
            except (KeyError, TypeError):
                pass
        return []

    def symbols(self) -> List[str]:
        """Return list of available symbols."""
        h = self._history
        if isinstance(h, pd.DataFrame):
            if isinstance(h.index, pd.MultiIndex):
                return list(h.index.get_level_values(1).unique())
            elif "symbol" in h.columns:
                return list(h["symbol"].unique())
            else:
                return ["asset"]
        return []

    def date_range(self) -> Tuple[datetime, datetime]:
        """Return the start and end datetime of the data."""
        h = self._history
        if isinstance(h, pd.DataFrame) and not h.empty:
            if isinstance(h.index, pd.MultiIndex):
                dates = h.index.get_level_values(0)
            else:
                dates = h.index
            return dates.min(), dates.max()
        return datetime(2000, 1, 1), datetime(2000, 1, 1)


# ---------------------------------------------------------------------------
# Pipeline Engine / 因子计算引擎
# ---------------------------------------------------------------------------

class PipelineEngine:
    """
    Batch factor computation engine for multi-asset time-series.
    批量因子计算引擎，用于多资产时间序列。

    Works with DataPortal history to compute factor values across
    a grid of (date, symbol) pairs, then applies screens (filters).

    Inspired by Zipline's SimplePipelineEngine.

    Parameters
    ----------
    data_portal : DataPortal
        Source of historical price/volume data.
    """

    def __init__(self, data_portal: DataPortal):
        self.data_portal = data_portal

    def compute_chunk(
        self,
        terms: Dict[str, Callable],
        start_dt: datetime,
        end_dt: datetime,
        screen_symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute factor terms over a date range for all symbols.
        在日期范围内计算所有标的的因子。

        Parameters
        ----------
        terms : Dict[str, Callable]
            Dictionary mapping factor names to functions.
            Each function signature: (history: pd.DataFrame, symbols: List[str]) -> pd.Series/DataFrame
            The function receives a MultiIndex (dt, symbol) DataFrame of close prices
            and returns a Series or DataFrame indexed by (dt, symbol).
        start_dt : datetime
            Start of computation window.
        end_dt : datetime
            End of computation window.
        screen_symbols : List[str], optional
            Universe of symbols to compute for.

        Returns
        -------
        pd.DataFrame
            MultiIndex (dt, symbol), columns = factor names.
        """
        symbols = screen_symbols or self.data_portal.symbols()
        h = self.data_portal._history

        if isinstance(h, pd.DataFrame) and isinstance(h.index, pd.MultiIndex):
            # Multi-asset history
            window = h.loc[start_dt:end_dt]
            results: Dict[str, List] = {name: [] for name in terms}
            dates = sorted(window.index.get_level_values(0).unique())

            for dt in dates:
                bar_data = window.loc[:dt]
                for name, func in terms.items():
                    try:
                        result = func(bar_data, symbols)
                        if isinstance(result, pd.Series):
                            results[name].append(result)
                        elif isinstance(result, pd.DataFrame):
                            for sym in symbols:
                                if sym in result.columns:
                                    results[name].append((dt, sym, result[sym].iloc[-1] if len(result) > 0 else float('nan')))
                    except Exception:
                        results[name].append(None)

            # Build output DataFrame
            if all(isinstance(v[0], tuple) for v in results.values() if v):
                rows = []
                for name, vals in results.items():
                    for dt, sym, val in vals:
                        rows.append({"dt": dt, "symbol": sym, name: val})
                df = pd.DataFrame(rows).set_index(["dt", "symbol"])
            else:
                df = pd.DataFrame()
            return df
        else:
            # Single-asset
            return pd.DataFrame()

    def run_pipeline(
        self,
        pipeline: "PipelineLite",
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """
        Execute a PipelineLite over a date range.
        执行 PipelineLite。

        Parameters
        ----------
        pipeline : PipelineLite
            Pipeline with columns and screen.
        start_dt : datetime
        end_dt : datetime

        Returns
        -------
        pd.DataFrame
            MultiIndex (dt, symbol) with pipeline output columns.
        """
        terms = pipeline._terms
        screen = pipeline._screen

        df = self.compute_chunk(terms, start_dt, end_dt)

        if screen is not None:
            try:
                df = df[df.index.map(lambda x: screen(x[0], x[1]))]
            except Exception:
                pass

        return df


class PipelineLite:
    """
    Lightweight Pipeline API for declaring factor computations.
    轻量级 Pipeline API，用于声明因子计算。

    Mirrors Zipline's Pipeline API:
        pipe = PipelineLite()
        pipe.add_factor(MyFactor(), 'my_factor')
        pipe.set_screen(MyFilter())

    Example
    -------
    ```python
    from quant_trading.backtester.zipline_lite import PipelineLite, Returns

    pipe = PipelineLite()
    pipe.add_factor(Returns(window_length=20), 'returns_20d')
    pipe.set_screen(lambda dt, sym: True)
    results = engine.run_pipeline(pipe, start_dt, end_dt)
    ```
    """

    def __init__(self):
        self._terms: Dict[str, Callable] = {}
        self._screen: Optional[Callable] = None
        self._columns: Dict[str, str] = {}

    def add_factor(
        self,
        factor_func: Callable[[pd.DataFrame, List[str]], pd.Series],
        name: str,
    ) -> "PipelineLite":
        """
        Add a factor computation to the pipeline.

        Parameters
        ----------
        factor_func : Callable
            Function(history, symbols) -> pd.Series indexed by (dt, symbol).
        name : str
            Column name for the factor output.
        """
        self._terms[name] = factor_func
        self._columns[name] = name
        return self

    def set_screen(self, screen_func: Callable[[datetime, str], bool]) -> "PipelineLite":
        """
        Set a filter that controls which (dt, symbol) rows are returned.

        Parameters
        ----------
        screen_func : Callable
            Function(dt, symbol) -> bool (True = include).
        """
        self._screen = screen_func
        return self

    def evaluate(self, history: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the pipeline on a full history DataFrame (offline batch)."""
        engine = PipelineEngine(DataPortal(history))
        all_dates = history.index.get_level_values(0)
        return engine.compute_chunk(
            self._terms,
            all_dates.min(),
            all_dates.max(),
        )


# Built-in factor helpers
def Returns(window_length: int = 20) -> Callable:
    """Return a function that computes rolling returns over ``window_length`` bars."""
    def compute(history: pd.DataFrame, symbols: List[str]) -> pd.Series:
        close = history["close"] if "close" in history.columns else history
        if isinstance(close.index, pd.MultiIndex):
            result = {}
            for sym in symbols:
                try:
                    sym_close = close.xs(sym, level=1)
                    ret = sym_close.pct_change(window_length)
                    for dt, val in ret.items():
                        result[(dt, sym)] = val
                except KeyError:
                    pass
            return pd.Series(result)
        return close.pct_change(window_length)
    return compute


def Rank(factor_func: Callable) -> Callable:
    """Wrap a factor function to return cross-sectional ranks (0-1)."""
    def ranked(history: pd.DataFrame, symbols: List[str]) -> pd.Series:
        raw = factor_func(history, symbols)
        if isinstance(raw, pd.Series):
            dt_level = raw.index.get_level_values(0) if isinstance(raw.index, pd.MultiIndex) else raw.index
            ranked_vals = raw.groupby(level=0, group_keys=False).rank(pct=True)
            return ranked_vals
        return raw
    return ranked


def ZScore(factor_func: Callable, window: int = 20) -> Callable:
    """Wrap a factor function to return rolling z-scores."""
    def zscored(history: pd.DataFrame, symbols: List[str]) -> pd.Series:
        raw = factor_func(history, symbols)
        if isinstance(raw, pd.Series) and isinstance(raw.index, pd.MultiIndex):
            return raw.groupby(level=1, group_keys=False).transform(
                lambda x: (x - x.rolling(window, min_periods=1).mean()) /
                          x.rolling(window, min_periods=1).std().replace(0, 1)
            )
        return raw
    return zscored


# ---------------------------------------------------------------------------
# TradingAlgorithm / 交易算法接口
# ---------------------------------------------------------------------------

class TradingAlgorithm(metaclass=ABCMeta):
    """
    User strategy interface.
    用户策略接口。

    Subclass this and implement:
        initialize(algo)  — called once at start
        handle_data(algo, data)  — called every bar

    Inside the strategy you have access to:
        algo.order(symbol, amount)       — place an order
        algo.cancel(order_id)            — cancel an order
        algo.data_portal                  — historical data access
        algo.portfolio                    — current portfolio
        algo.current(symbol, field)      — spot value for current bar
        algo.history(symbol, field, n)    — trailing window
        algo.record(**kwargs)             — record custom metrics

    Parameters
    ----------
    capital_base : float
        Initial capital. Default 100_000.
    """

    def __init__(self, capital_base: float = 100_000.0):
        self.capital_base = capital_base

    @abstractmethod
    def initialize(self, algo: "AlgoAPI") -> None:
        """Called once before simulation starts."""
        raise NotImplementedError("initialize")

    @abstractmethod
    def handle_data(self, algo: "AlgoAPI", data: "BarData") -> None:
        """Called every bar to process market data and place orders."""
        raise NotImplementedError("handle_data")


class AlgoAPI:
    """
    TradingAlgorithm's view into the backtester environment.
    策略算法运行时 API。

    Provides order placement, data access, and recording.
    Created by EventDrivenBacktester and passed to the user's strategy.
    """

    def __init__(
        self,
        backtester: "EventDrivenBacktester",
    ):
        self._bt = backtester
        self.portfolio = backtester.portfolio
        self.data_portal = backtester.data_portal
        self._records: Dict[str, List[Any]] = defaultdict(list)
        self._benchmark_fn: Optional[Callable[[datetime], float]] = None

    @property
    def symbols(self) -> List[str]:
        """List of tradable symbols in current bar."""
        return self._bt._current_symbols

    def order(
        self,
        symbol: str,
        amount: int,
        style: Optional[str] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Place an order.
        下单。

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        amount : int
            Positive = buy, negative = sell.
        style : str, optional
            'market' (default), 'limit', 'stop'.
        limit_price : float, optional
            Limit price for limit orders.
        stop_price : float, optional
            Stop price for stop orders.

        Returns
        -------
        Order or None
            The created order, or None if rejected.
        """
        return self._bt.place_order(
            symbol=symbol,
            amount=amount,
            style=style,
            limit_price=limit_price,
            stop_price=stop_price,
        )

    def cancel(self, order_id: str) -> None:
        """Cancel an open order."""
        self._bt.cancel_order(order_id)

    def current(self, symbol: str, field: str = "close") -> float:
        """
        Get the most recent value for a field.
        获取当前 bar 的最新字段值。

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        field : str
            Field name: open, high, low, close, volume, price.

        Returns
        -------
        float
        """
        if symbol not in self._bt._current_bar:
            return float("nan")
        bar = self._bt._current_bar[symbol]
        val = bar.get(field, bar.get("close", float("nan")))
        return float(val) if val == val else float("nan")

    def history(
        self,
        symbol: str,
        field: str,
        bar_count: int,
        frequency: str = "1d",
    ) -> pd.Series:
        """
        Get trailing window of historical values.
        获取历史滚动窗口。

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        field : str
            Field name.
        bar_count : int
            Number of bars.
        frequency : str
            '1d' or '1m'.

        Returns
        -------
        pd.Series
        """
        dt = self._bt._current_dt
        return self.data_portal.history(symbol, field, bar_count, frequency, end_dt=dt)

    def set_benchmark(self, fn: Callable[[datetime], float]) -> None:
        """Set a benchmark returns function (daily returns)."""
        self._benchmark_fn = fn

    def record(self, **kwargs: Any) -> None:
        """Record custom metrics at the current bar."""
        self._records["dt"].append(self._bt._current_dt)
        for k, v in kwargs.items():
            self._records[k].append(v)

    def get_records(self) -> pd.DataFrame:
        """Return recorded metrics as a DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame(self._records).set_index("dt")


class BarData:
    """
    Snapshot of current bar data accessible to the strategy.
    当前 bar 数据快照，策略可访问。

    Parameters
    ----------
    bars : Dict[str, Dict[str, Any]]
        Maps symbol -> {open, high, low, close, volume}.
    current_dt : datetime
        Current simulation timestamp.
    data_portal : DataPortal, optional
        DataPortal for history() calls.
    """

    def __init__(
        self,
        bars: Dict[str, Dict[str, Any]],
        current_dt: datetime,
        data_portal: Optional[DataPortal] = None,
    ):
        self._bars = bars
        self.current_dt = current_dt
        self._data_portal = data_portal

    def current(self, symbol: str, field: str = "close") -> float:
        """Get current value for a symbol and field."""
        bar = self._bars.get(symbol, {})
        val = bar.get(field, bar.get("close", float("nan")))
        return float(val) if val == val else float("nan")

    def symbols(self) -> List[str]:
        """All symbols with data in current bar."""
        return list(self._bars.keys())

    def history(
        self,
        symbol: str,
        field: str,
        bar_count: int,
        frequency: str = "1d",
    ) -> pd.Series:
        """Proxy to DataPortal.history for trailing window access."""
        if self._data_portal is not None:
            return self._data_portal.history(
                symbol, field, bar_count, frequency, end_dt=self.current_dt
            )
        raise NotImplementedError(
            "BarData.history requires DataPortal. Use algo.history() instead."
        )

    def can_trade(self, symbol: str) -> bool:
        """Check if an asset is currently tradeable."""
        return symbol in self._bars and self._bars[symbol].get("volume", 0) > 0


# ---------------------------------------------------------------------------
# EventDrivenBacktester / 事件驱动回测引擎
# ---------------------------------------------------------------------------

class EventDrivenBacktester:
    """
    Core event-driven backtester.
    核心事件驱动回测引擎。

    Event loop
    ----------
    The backtester iterates over the history DataFrame, emitting BAR events
    for each timestamp. For each bar:

        1. Process any scheduled functions (schedule_function).
        2. Call the strategy's handle_data(data).
        3. Process all open orders:
           a. Evaluate stop/limit triggers.
           b. Apply slippage model → FillEvent.
           c. Apply commission model → commission cost.
           d. Update portfolio (cash, positions).
        4. Process dividends.
        5. Update portfolio NAV and record performance.
        6. Emit performance snapshot (daily or minute).

    Parameters
    ----------
    capital_base : float
        Initial capital. Default 100_000.
    data_frequency : str
        'daily' or 'minute'. Default 'daily'.
    emission_rate : str
        Performance emission rate: 'daily' or 'minute'. Default 'daily'.
    slippage : SlippageModel, optional
        Slippage model instance. Default: VolumeShareSlippage.
    commission : CommissionModel, optional
        Commission model instance. Default: PerShareCommission.
    benchmark_fn : Callable, optional
        Function(dt) -> float daily return for benchmark. Used in perf metrics.
    """

    # Direction constants
    BUY  = +1
    SELL = -1

    def __init__(
        self,
        capital_base: float = 100_000.0,
        data_frequency: str = "daily",
        emission_rate: str = "daily",
        slippage: Optional[SlippageModel] = None,
        commission: Optional[CommissionModel] = None,
        benchmark_fn: Optional[Callable[[datetime], float]] = None,
    ):
        self.capital_base = capital_base
        self.data_frequency = data_frequency
        self.emission_rate = emission_rate
        self.slippage = slippage or VolumeShareSlippage()
        self.commission = commission or PerShareCommission()
        self.benchmark_fn = benchmark_fn

        # Internal state (reset on each run)
        self.portfolio: Optional[Portfolio] = None
        self.data_portal: Optional[DataPortal] = None
        self.orders: OrderedDict[str, Order] = OrderedDict()
        self._orders_by_symbol: Dict[str, List[Order]] = defaultdict(list)
        self.perf_buffer: List[Dict[str, Any]] = []
        self.transaction_log: List[FillEvent] = []
        self.order_log: List[Order] = []
        self.dividend_log: List[DividendEvent] = []
        self._current_dt: Optional[datetime] = None
        self._current_bar: Dict[str, Dict[str, Any]] = {}
        self._current_symbols: List[str] = []
        self._schedule: List[Tuple[Callable, str]] = []  # (func, rule_str)
        self._session_starts: List[datetime] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_slippage(self, model: SlippageModel) -> None:
        """Set the slippage model (also configurable via constructor)."""
        self.slippage = model

    def set_commission(self, model: CommissionModel) -> None:
        """Set the commission model (also configurable via constructor)."""
        self.commission = model

    def schedule_function(
        self,
        func: Callable[["AlgoAPI", BarData], None],
        rule: str = "end_of_day",
    ) -> None:
        """
        Schedule a function to run at a specific point in each session.
        调度函数在每个交易日特定时点运行。

        Parameters
        ----------
        func : Callable
            Function(algo, data) to call.
        rule : str
            'start_of_day' | 'end_of_day' | 'every_bar'
        """
        self._schedule.append((func, rule))

    def run(
        self,
        algorithm: TradingAlgorithm,
        history: Union[pd.DataFrame, pd.Panel],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Run the backtest.
        运行回测。

        Parameters
        ----------
        algorithm : TradingAlgorithm
            Subclass of TradingAlgorithm with initialize() and handle_data().
        history : pd.DataFrame or pd.Panel
            Price/volume data.
            pd.DataFrame with DatetimeIndex: single-asset.
            pd.DataFrame with MultiIndex (dt, symbol): multi-asset.
            Must have columns: open, high, low, close, volume.
        start : datetime, optional
            Simulation start. Defaults to first bar.
        end : datetime, optional
            Simulation end. Defaults to last bar.

        Returns
        -------
        Dict[str, Any]
            {
                'portfolio': Portfolio,
                'orders': List[Order],
                'transactions': List[FillEvent],
                'dividends': List[DividendEvent],
                'daily_returns': List[float],
                'nav_series': pd.Series,
                'perf_buffer': List[Dict],
            }
        """
        # Reset state
        self.portfolio = Portfolio(self.capital_base)
        self.data_portal = DataPortal(history)
        self.orders = OrderedDict()
        self._orders_by_symbol = defaultdict(list)
        self.perf_buffer = []
        self.transaction_log = []
        self.order_log = []
        self.dividend_log = []
        self._current_dt = None
        self._current_bar = {}

        # Initialize the algorithm
        algo_api = AlgoAPI(self)
        try:
            algorithm.initialize(algo_api)
        except TypeError:
            # Support initialize() with no args for convenience
            algorithm.initialize()

        # Normalize history to DataFrame with MultiIndex if needed
        h = history
        # pd.Panel was removed in pandas 2.0; ignore it gracefully
        try:
            if isinstance(h, pd.Panel):
                h = h.to_frame()
        except AttributeError:
            pass
        h = self._normalize_history(h)

        # Determine date range
        if isinstance(h.index, pd.MultiIndex):
            all_dts = sorted(h.index.get_level_values(0).unique())
        else:
            all_dts = sorted(h.index)

        if start is not None:
            all_dts = [dt for dt in all_dts if dt >= pd.Timestamp(start)]
        if end is not None:
            all_dts = [dt for dt in all_dts if dt <= pd.Timestamp(end)]

        if not all_dts:
            warnings.warn("No data in the specified date range.")
            return self._empty_result()

        # Set initial NAV for return calculation
        self.portfolio.nav_history.append(self.portfolio.net_asset_value)

        # Main event loop
        for dt in all_dts:
            self._current_dt = dt.to_pydatetime()
            bar_data = self._build_bar_data(h, dt)
            self._current_bar = bar_data
            self._current_symbols = list(bar_data.keys())

            # Update position last sale prices
            for sym, bar in bar_data.items():
                self.portfolio.update_position_price(sym, bar.get("close", 0.0))

            # Run scheduled functions at start of day
            for func, rule in self._schedule:
                if rule == "start_of_day" or rule == "every_bar":
                    func(algo_api, BarData(bar_data, self._current_dt, self.data_portal))

            # Run strategy handle_data
            algo_api2 = AlgoAPI(self)  # fresh each bar to reflect latest state
            try:
                algorithm.handle_data(algo_api2, BarData(bar_data, self._current_dt, self.data_portal))
            except TypeError:
                algorithm.handle_data(algo_api2)

            # Process orders for each symbol
            for sym, bar in bar_data.items():
                self._process_bar_orders(sym, bar, dt)

            # Process dividends
            self._process_dividends(dt)

            # Process scheduled end-of-day functions
            for func, rule in self._schedule:
                if rule == "end_of_day":
                    func(algo_api, BarData(bar_data, self._current_dt, self.data_portal))

            # Record daily return
            self.portfolio.record_daily_return()

            # Emit performance snapshot
            if self.emission_rate == "daily":
                self._emit_perf_snapshot()

        # Finalize
        self._finalize()

        return self._build_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_history(self, h: pd.DataFrame) -> pd.DataFrame:
        """Ensure history is a DataFrame with proper index."""
        if h.empty:
            return h

        # If simple DatetimeIndex, convert to MultiIndex with default symbol
        if isinstance(h.index, pd.DatetimeIndex) and isinstance(h.columns, str):
            # Single asset with column strings — already simple OHLCV
            if "symbol" not in h.columns.str.lower():
                # Wrap in MultiIndex (dt, symbol)
                h = h.copy()
                h["symbol"] = "asset"
                h = h.set_index(["symbol"], append=True)
                h = h.swaplevel(0, 1)
        elif isinstance(h.index, pd.MultiIndex):
            pass  # Already MultiIndex
        return h

    def _build_bar_data(
        self,
        h: pd.DataFrame,
        dt: pd.Timestamp,
    ) -> Dict[str, Dict[str, Any]]:
        """Build a dict of {symbol: {open, high, low, close, volume}} for this dt."""
        bars: Dict[str, Dict[str, Any]] = {}

        if isinstance(h.index, pd.MultiIndex):
            try:
                dt_bars = h.loc[dt]
                if isinstance(dt_bars, pd.DataFrame):
                    for sym in dt_bars.index:
                        row = dt_bars.loc[sym]
                        bars[str(sym)] = {
                            "open": row.get("open", row.get("close")),
                            "high": row.get("high", row.get("close")),
                            "low": row.get("low", row.get("close")),
                            "close": row.get("close"),
                            "volume": row.get("volume", 0),
                        }
                elif isinstance(dt_bars, pd.Series):
                    bars["asset"] = {
                        "open": dt_bars.get("open", dt_bars.get("close")),
                        "high": dt_bars.get("high", dt_bars.get("close")),
                        "low": dt_bars.get("low", dt_bars.get("close")),
                        "close": dt_bars.get("close"),
                        "volume": dt_bars.get("volume", 0),
                    }
            except KeyError:
                # No data for this exact timestamp — find nearest prior
                available = h.index.get_level_values(0)
                before = available[available <= dt]
                if len(before) > 0:
                    nearest = before[-1]
                    return self._build_bar_data(h, nearest)
        else:
            try:
                row = h.loc[dt]
                bars["asset"] = {
                    "open": row.get("open", row.get("close")),
                    "high": row.get("high", row.get("close")),
                    "low": row.get("low", row.get("close")),
                    "close": row.get("close"),
                    "volume": row.get("volume", 0),
                }
            except (KeyError, TypeError):
                pass

        return bars

    def place_order(
        self,
        symbol: str,
        amount: int,
        style: Optional[str] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[Order]:
        """Place a new order and return it."""
        if amount == 0:
            return None
        order = Order(
            dt=self._current_dt,
            symbol=symbol,
            amount=amount,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        self.orders[order.id] = order
        self._orders_by_symbol[symbol].append(order)
        self.order_log.append(order)
        return order

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        order = self.orders.get(order_id)
        if order is not None:
            order.cancel()

    def _process_bar_orders(
        self,
        symbol: str,
        bar: Dict[str, Any],
        dt: pd.Timestamp,
    ) -> None:
        """Process all open orders for a symbol in a single bar."""
        orders = self._orders_by_symbol.get(symbol, [])
        if not orders:
            return

        # Build a synthetic BarEvent
        bar_event = BarEvent(
            dt=dt.to_pydatetime(),
            symbol=symbol,
            open_=bar.get("open", bar.get("close")),
            high=bar.get("high", bar.get("close")),
            low=bar.get("low", bar.get("close")),
            close=bar.get("close"),
            volume=int(bar.get("volume", 0)),
        )

        volume_filled = 0
        still_open: List[Order] = []

        for order in orders:
            if not order.open:
                continue

            # Simulate fill through slippage model
            try:
                exec_price, exec_vol = self.slippage.process_order(
                    bar_event, order, volume_filled
                )
            except LiquidityExceeded:
                exec_price, exec_vol = None, None

            if exec_price is None or exec_vol is None or exec_vol == 0:
                still_open.append(order)
                continue

            # Limit price check
            if order.limit_price is not None:
                if order.direction > 0 and exec_price > order.limit_price:
                    still_open.append(order)
                    continue
                if order.direction < 0 and exec_price < order.limit_price:
                    still_open.append(order)
                    continue

            # Commission
            fill = FillEvent(
                dt=dt.to_pydatetime(),
                order_id=order.id,
                symbol=symbol,
                amount=int(exec_vol),
                price=float(exec_price),
            )
            comm = self.commission.calculate(order, fill)
            fill.commission = comm

            # Update order
            order.fill(abs(exec_vol), exec_price, comm)

            # Update portfolio
            self.portfolio.execute_fill(symbol, int(exec_vol), exec_price, comm)

            # Log
            self.transaction_log.append(fill)
            volume_filled += abs(exec_vol)

        # Remove filled/cancelled orders from active list
        self._orders_by_symbol[symbol] = still_open

    def _process_dividends(self, dt: pd.Timestamp) -> None:
        """Credit dividends for all positions held on ex-date."""
        if self.data_portal is None:
            return
        for sym in list(self.portfolio.positions.keys()):
            divs = self.data_portal.get_dividends(sym, dt.to_pydatetime())
            for div in divs:
                pos = self.portfolio.positions.get(sym)
                if pos is not None and pos.amount > 0:
                    cash_amount = div.amount * pos.amount
                    self.portfolio.process_dividend(sym, cash_amount)
                    self.dividend_log.append(div)

    def _emit_perf_snapshot(self) -> None:
        """Capture current performance state into perf_buffer."""
        self.perf_buffer.append({
            "dt": self._current_dt,
            "cash": self.portfolio.cash,
            "positions_value": self.portfolio.positions_value,
            "nav": self.portfolio.net_asset_value,
            "cumulative_pnl": self.portfolio.cumulative_pnl,
            "positions": dict(self.portfolio.positions),
        })

    def _finalize(self) -> None:
        """Called at end of simulation."""
        pass

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result when no data."""
        return {
            "portfolio": self.portfolio,
            "orders": [],
            "transactions": [],
            "dividends": [],
            "daily_returns": [],
            "nav_series": pd.Series([], dtype=float),
            "perf_buffer": [],
        }

    def _build_result(self) -> Dict[str, Any]:
        """Package results into a dict."""
        nav_series = pd.Series(
            self.portfolio.nav_history,
            index=pd.to_datetime(self.portfolio.nav_history),
        ) if self.portfolio.nav_history else pd.Series(dtype=float)

        return {
            "portfolio": self.portfolio,
            "orders": list(self.orders.values()),
            "transactions": self.transaction_log,
            "dividends": self.dividend_log,
            "daily_returns": self.portfolio.daily_returns,
            "nav_series": nav_series,
            "perf_buffer": self.perf_buffer,
        }
