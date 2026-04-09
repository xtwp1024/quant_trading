# -*- coding: utf-8 -*-
"""
Event-driven Backtesting Engine
事件驱动回测引擎 — Zipline风格的initialize/handle_data范式

Architecture:
    EventEngine → BarEvent/OrderEvent/PortfolioEvent
    生命周期钩子: before_trading_start / handle_data / after_trading_end

Built on PyData stack: pandas, numpy, scipy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from enum import Enum, auto

__all__ = [
    "Event",
    "BarEvent",
    "OrderEvent",
    "PortfolioEvent",
    "EventType",
    "EventEngine",
]


# ------------------------------------------------------------------
# Event Definitions / 事件定义
# ------------------------------------------------------------------


class EventType(Enum):
    """事件类型枚举."""
    BAR = auto()
    ORDER = auto()
    FILL = auto()
    PORTFOLIO = auto()
    BEFORE_TRADING = auto()
    AFTER_TRADING = auto()
    SCHEDULED = auto()


@dataclass
class Event:
    """
    事件基类 / Event base class.

    Attributes:
        dt (pd.Timestamp): 事件时间戳
        symbol (str): 标的代码
    """
    dt: pd.Timestamp
    symbol: str


@dataclass
class BarEvent(Event):
    """
    K线事件 / OHLCV Bar Event.

    Attributes:
        open (float): 开盘价
        high (float): 最高价
        low (float): 最低价
        close (float): 收盘价
        volume (float): 成交量
    """
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_dict(cls, dt: pd.Timestamp, symbol: str, row: pd.Series) -> BarEvent:
        """从Series行数据构造BarEvent."""
        return cls(
            dt=dt,
            symbol=symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )


@dataclass
class OrderEvent(Event):
    """
    订单提交事件 / Order Submission Event.

    Attributes:
        amount (int): 成交数量（正=买入，负=卖出）
        price (float): 订单价格
        order_id (str): 订单ID
    """
    amount: int
    price: float
    order_id: str = ""


@dataclass
class FillEvent(Event):
    """
    订单成交事件 / Order Fill Event.

    Attributes:
        order_id (str): 关联订单ID
        amount (int): 成交数量
        price (float): 成交价格
        commission (float): 手续费
    """
    order_id: str
    amount: int
    price: float
    commission: float = 0.0


@dataclass
class PortfolioEvent(Event):
    """
    组合状态更新事件 / Portfolio Update Event.
    """
    cash: float
    equity: float
    positions: dict  # {symbol: {"amount": int, "cost": float}}
    daily_returns: float = 0.0


# ------------------------------------------------------------------
# Context — 算法运行时状态容器
# ------------------------------------------------------------------


@dataclass
class Context:
    """
    算法运行时状态上下文 / Algorithm runtime context.

    用户在 initialize() 中设置 context 变量，在 handle_data() 中访问。

    Attributes:
        portfolio (dict): 组合账户信息
       持仓 (dict): 持仓数据 {symbol: amount}
        current_dt (pd.Timestamp): 当前时间
        event_engine (EventEngine): 事件引擎引用
    """
    portfolio: dict = field(default_factory=lambda: {
        "cash": 100000.0,
        "starting_cash": 100000.0,
        "equity": 100000.0,
        "positions": {},
        "daily_returns": 0.0,
        "total_returns": 0.0,
    })
    _positions: dict = field(default_factory=dict)  # {symbol: {"amount": int, "cost": float}}
    current_dt: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2024-01-01"))
    event_engine: Optional[EventEngine] = None

    @property
    def positions(self) -> dict:
        """获取当前持仓 {symbol: amount}."""
        return {s: v["amount"] for s, v in self._positions.items()}

    def get_position(self, symbol: str) -> int:
        """获取指定标的的持仓数量."""
        return self._positions.get(symbol, {}).get("amount", 0)

    def get_position_cost(self, symbol: str) -> float:
        """获取指定标的的平均持仓成本."""
        return self._positions.get(symbol, {}).get("cost", 0.0)

    def order(self, symbol: str, amount: int, price: Optional[float] = None) -> str:
        """
        提交订单 / Submit order.

        Args:
            symbol: 标的代码
            amount: 成交数量（正=买入，负=卖出）
            price: 限价（None=市价单）

        Returns:
            str: 订单ID
        """
        if self.event_engine is None:
            raise RuntimeError("Context not connected to EventEngine")
        return self.event_engine.order(self, symbol, amount, price)


# ------------------------------------------------------------------
# EventEngine — 核心回测引擎
# ------------------------------------------------------------------


class EventEngine:
    """
    事件驱动回测引擎 / Event-Driven Backtesting Engine.

    Zipline风格的 initialize/handle_data 范式，支持生命周期钩子。

    Example:
        >>> def initialize(context):
        ...     context.set_universe(["000001.XSHE", "600000.XSHG"])
        ...     context.count = 0
        ...
        >>> def handle_data(context, data):
        ...     context.count += 1
        ...     if context.count > 10:
        ...         context.order("000001.XSHE", 100)
        ...
        >>> engine = EventEngine()
        >>> results = engine.run(data, initialize, handle_data)
    """

    def __init__(self) -> None:
        """初始化事件引擎."""
        self._universe: list[str] = []
        self._factors: list[Callable] = []
        self._transforms: list[Callable] = []
        self._before_trading_start: Optional[Callable] = None
        self._after_trading_end: Optional[Callable] = None
        self._schedule: list[tuple[pd.Timestamp, Callable]] = []
        self._order_id_counter: int = 0
        self._pending_orders: dict[str, OrderEvent] = {}
        self._event_history: list[Event] = []

    # ------------------------------------------------------------------
    # Configuration / 配置方法
    # ------------------------------------------------------------------

    def set_universe(self, symbols: list[str]) -> None:
        """
        设置交易标的 universe.

        Args:
            symbols: 标的代码列表，如 ["000001.XSHE", "600000.XSHG"]
        """
        self._universe = list(symbols)

    def add_factor(self, factor: callable) -> None:
        """
        添加因子计算函数.

        Args:
            factor: func(data: pd.DataFrame) -> pd.Series|pd.DataFrame
        """
        self._factors.append(factor)

    def add_transform(self, transform: callable) -> None:
        """
        添加数据转换函数.

        Args:
            transform: func(data: pd.DataFrame) -> pd.DataFrame
        """
        self._transforms.append(transform)

    def before_trading_start(self, callback: Callable[[Context], None]) -> None:
        """
        注册盘前回调.

        Args:
            callback: func(context) — 每日交易开始前调用
        """
        self._before_trading_start = callback

    def after_trading_end(self, callback: Callable[[Context], None]) -> None:
        """
        注册盘后回调.

        Args:
            callback: func(context) — 每日交易结束后调用
        """
        self._after_trading_end = callback

    # ------------------------------------------------------------------
    # Core run loop / 核心回测循环
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        initialize: Callable[[Context], None],
        handle_data: Callable[[Context, pd.DataFrame], None],
        capital_base: float = 100000.0,
    ) -> pd.DataFrame:
        """
        运行回测 / Run backtest.

        Args:
            data: MultiIndex DataFrame (dt, symbol) with OHLCV  columns
                  Required columns: open, high, low, close, volume
            initialize: func(context) — 初始化回调，仅在开始时调用一次
            handle_data: func(context, data) — 每个bar调用一次
            capital_base: 初始资金，默认 100000.0

        Returns:
            pd.DataFrame: 回测结果，包含：
                - dt: 时间戳
                - symbol: 标的代码
                - equity: 组合权益
                - cash: 现金
                - returns: 收益率
        """
        # --- 初始化 context ---
        context = Context()
        context.portfolio = {
            "cash": capital_base,
            "starting_cash": capital_base,
            "equity": capital_base,
            "positions": {},
            "daily_returns": 0.0,
            "total_returns": 0.0,
        }
        context.event_engine = self

        # 应用初始化回调
        initialize(context)

        # 如果未设置 universe，尝试从数据中推断
        if not self._universe:
            if isinstance(data.columns, pd.MultiIndex):
                self._universe = list(data.columns.get_level_values(1).unique())
            else:
                self._universe = list(data.columns)

        # --- 数据预处理 ---
        # 应用 transforms
        for tf in self._transforms:
            data = tf(data)

        # 收集所有时间戳
        if isinstance(data.index, pd.MultiIndex):
            timestamps = sorted(data.index.get_level_values(0).unique())
        else:
            timestamps = sorted(data.index.unique())

        # --- 每日统计收集 ---
        daily_stats = []
        prev_equity = capital_base

        # --- 主循环：逐 bar 处理 ---
        for dt in timestamps:
            context.current_dt = pd.Timestamp(dt)

            # 获取当前 bar 数据
            if isinstance(data.index, pd.MultiIndex):
                bar_data = data.xs(dt, level=0)
            else:
                bar_data = data.loc[[dt]].droplevel(0) if isinstance(data.index, pd.MultiIndex) else data.loc[dt]

            # 安全获取 bar_data
            if isinstance(bar_data, pd.DataFrame):
                bar_dict = {s: bar_data.loc[s] if s in bar_data.index else None
                            for s in self._universe}
            elif isinstance(bar_data, pd.Series):
                bar_dict = bar_data.to_dict()
            else:
                bar_dict = {}

            # --- before_trading_start 钩子 ---
            if self._before_trading_start is not None:
                self._before_trading_start(context)

            # --- 计算因子 ---
            factor_values = {}
            for factor_fn in self._factors:
                try:
                    factor_val = factor_fn(bar_data)
                    if isinstance(factor_val, pd.Series):
                        factor_values[factor_fn.__name__] = factor_val
                except Exception:
                    pass

            # --- handle_data 钩子 ---
            handle_data(context, bar_data)

            # --- 模拟订单成交（简化：市价单立即成交） ---
            self._process_pending_orders(context, bar_dict, dt)

            # --- 更新组合权益 ---
            equity = self._calculate_equity(context, bar_dict)
            context.portfolio["equity"] = equity
            daily_returns = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            context.portfolio["daily_returns"] = daily_returns
            context.portfolio["total_returns"] = (equity - capital_base) / capital_base
            prev_equity = equity

            # --- 记录每日统计 ---
            for symbol in self._universe:
                if symbol in bar_dict:
                    stat = {
                        "dt": dt,
                        "symbol": symbol,
                        "equity": equity,
                        "cash": context.portfolio["cash"],
                        "returns": daily_returns,
                        "position": context.get_position(symbol),
                    }
                    # 添加因子值
                    for fname, fval in factor_values.items():
                        if symbol in fval.index:
                            stat[fname] = fval[symbol]
                    daily_stats.append(stat)

            # --- after_trading_end 钩子 ---
            if self._after_trading_end is not None:
                self._after_trading_end(context)

        # --- 转换为 DataFrame ---
        result = pd.DataFrame(daily_stats)
        if not result.empty:
            result = result.set_index(["dt", "symbol"])

        return result

    # ------------------------------------------------------------------
    # Order Management / 订单管理
    # ------------------------------------------------------------------

    def order(
        self,
        context: Context,
        symbol: str,
        amount: int,
        price: Optional[float] = None,
    ) -> str:
        """
        提交订单 / Submit order.

        Args:
            context: 运行时上下文
            symbol: 标的代码
            amount: 成交数量（正=买入，负=卖出）
            price: 限价（None=市价单）

        Returns:
            str: 订单ID
        """
        self._order_id_counter += 1
        order_id = f"ORDER_{self._order_id_counter:06d}"

        order = OrderEvent(
            dt=context.current_dt,
            symbol=symbol,
            amount=amount,
            price=price if price is not None else 0.0,
            order_id=order_id,
        )
        self._pending_orders[order_id] = order
        return order_id

    def _process_pending_orders(
        self,
        context: Context,
        bar_dict: dict,
        dt: pd.Timestamp,
    ) -> None:
        """处理待成交订单（简化：市价单按收盘价成交）."""
        filled_orders = []

        for order_id, order in self._pending_orders.items():
            if order.symbol not in bar_dict or bar_dict[order.symbol] is None:
                continue

            bar = bar_dict[order.symbol]
            # 市价单：用收盘价成交
            fill_price = bar.get("close", order.price) if order.price == 0 else order.price
            commission = abs(fill_price * order.amount) * 0.0003  # 默认万三手续费

            # 更新持仓
            if order.symbol not in context._positions:
                context._positions[order.symbol] = {"amount": 0, "cost": 0.0}

            pos = context._positions[order.symbol]
            old_amount = pos["amount"]
            new_amount = old_amount + order.amount

            if new_amount == 0:
                # 平仓
                pos["amount"] = 0
                pos["cost"] = 0.0
                context.portfolio["cash"] += (
                    old_amount * fill_price - commission
                    - (pos.get("cost", 0.0) * old_amount if old_amount > 0 else 0)
                )
            elif old_amount == 0:
                # 新开仓
                pos["amount"] = new_amount
                pos["cost"] = fill_price
                context.portfolio["cash"] -= fill_price * new_amount + commission
            elif (old_amount > 0 and new_amount > 0) or (old_amount < 0 and new_amount < 0):
                # 同方向加仓
                total_cost = pos.get("cost", 0.0) * old_amount + fill_price * order.amount
                pos["amount"] = new_amount
                pos["cost"] = total_cost / new_amount if new_amount != 0 else 0.0
                context.portfolio["cash"] -= fill_price * order.amount + commission
            else:
                # 反方向：先平后开
                close_amount = min(abs(old_amount), abs(order.amount))
                if old_amount > 0:
                    # 平多
                    context.portfolio["cash"] += fill_price * close_amount - commission
                else:
                    # 平空
                    context.portfolio["cash"] -= fill_price * close_amount + commission

                remaining = new_amount
                if remaining != 0:
                    pos["amount"] = remaining
                    pos["cost"] = fill_price
                else:
                    pos["amount"] = 0
                    pos["cost"] = 0.0

            filled_orders.append(order_id)

        # 清除已成交订单
        for oid in filled_orders:
            del self._pending_orders[oid]

    def _calculate_equity(self, context: Context, bar_dict: dict) -> float:
        """计算当前权益 = 现金 + 持仓市值."""
        cash = context.portfolio["cash"]
        position_value = 0.0

        for symbol, pos in context._positions.items():
            amount = pos.get("amount", 0)
            if amount == 0:
                continue
            if symbol in bar_dict and bar_dict[symbol] is not None:
                price = bar_dict[symbol].get("close", pos.get("cost", 0.0))
                position_value += amount * price
            else:
                position_value += amount * pos.get("cost", 0.0)

        return cash + position_value

    # ------------------------------------------------------------------
    # Scheduling / 调度
    # ------------------------------------------------------------------

    def schedule(
        self,
        func: Callable[[Context], None],
        dt: pd.Timestamp,
    ) -> None:
        """
        注册定时任务 / Schedule a function call at specific time.

        Args:
            func: 待执行的回调函数
            dt: 执行时间
        """
        self._schedule.append((dt, func))
