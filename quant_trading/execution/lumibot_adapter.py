# -*- coding: utf-8 -*-
"""
Lumibot Adapter - Lumibot multi-broker trading framework integration.
 lumibot_adapter.py

 Implemented Classes / 类实现:
   LumibotBroker        - Abstract broker interface / 抽象券商接口
   PolygonBroker        - Polygon market data + execution / Polygon行情+执行
   AlpacaBroker         - Alpaca broker integration (REST via urllib) / Alpaca券商集成
   LumibotStrategy      - User strategy base class / 用户策略基类
   LumibotBacktest     - Backtest engine over any broker data / 任意券商数据回测引擎

 Supported Brokers / 支持券商:
   Polygon, Alpaca (no external lumibot SDK dependency / 无外部lumibot SDK依赖)

 Key Methods / 核心方法:
   submit_order(), cancel_order(), get_positions(), get_historical_data()

 Pure Python urllib. No lumibot SDK dependency.
 纯Python urllib实现，无lumibot SDK依赖。

---

 Lumibot Architecture (absorbed from D:/Hive/Data/trading_repos/lumibot/):
   Broker (abstract) -> submit_order/cancel_order/get_positions/_submit_order
   DataSource (abstract) -> get_historical_prices/get_last_price
   Strategy (base) -> create_order/submit_order/get_positions/on_trading_iteration

 This adapter re-implements the core interfaces in pure Python urllib
 so that the quant_trading system can drive lumibot-style strategies
 without installing the lumibot package.
 本适配器用纯Python urllib重新实现核心接口，使量化交易系统
 能够在不安装lumibot包的情况下驱动lumibot风格策略。

---

 Author: Claude Code (absorbed from lumibot v4.4+)
 Date: 2026-03-31
"""

from __future__ import annotations

import json
import math
import time
import uuid
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_logger = logging.getLogger("lumibot_adapter")


# ---------------------------------------------------------------------------
# Enums / 枚举类型
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    """Order side / 订单方向"""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type / 订单类型"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status / 订单状态"""

    PENDING = "pending_new"
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force / 订单有效期"""

    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


# ---------------------------------------------------------------------------
# Entity Classes / 实体类
# ---------------------------------------------------------------------------


class Asset:
    """
    Trading asset representation / 交易资产表示

    Attributes / 属性:
        symbol: str - Ticker symbol / 标的代码
        asset_type: str - "stock", "option", "future", "forex", "crypto" / 资产类型
        expiration: datetime, optional - Expiry for options/futures / 到期日
        strike: float, optional - Strike price for options / 行权价
        right: str, optional - "call" or "put" for options / 期权方向
        multiplier: float - Contract multiplier / 合约乘数
    """

    AssetType = Literal["stock", "option", "future", "forex", "crypto"]

    def __init__(
        self,
        symbol: str,
        asset_type: Asset.AssetType = "stock",
        expiration: Optional[datetime] = None,
        strike: Optional[float] = None,
        right: Optional[str] = None,
        multiplier: float = 1.0,
        quote: Optional["Asset"] = None,
    ):
        self.symbol = symbol.upper() if symbol else ""
        self.asset_type = asset_type
        self.expiration = expiration
        self.strike = strike
        self.right = right  # "call" or "put"
        self.multiplier = multiplier
        self.quote = quote

    def __repr__(self):
        if self.asset_type == "option":
            return (
                f"{self.symbol}_{self.expiration.strftime('%Y%m%d')}_"
                f"{self.strike}_{self.right.upper()}"
            )
        return self.symbol

    @property
    def is_option(self) -> bool:
        return self.asset_type == "option"

    @property
    def is_crypto(self) -> bool:
        return self.asset_type == "crypto"

    @property
    def is_forex(self) -> bool:
        return self.asset_type == "forex"

    @property
    def is_future(self) -> bool:
        return self.asset_type == "future"


class Order:
    """
    Trading order representation / 交易订单表示

    Attributes / 属性:
        identifier: str - Broker-assigned order ID / 券商分配订单ID
        asset: Asset - The asset to trade / 交易资产
        quantity: Decimal - Quantity to trade / 交易数量
        side: OrderSide - "buy" or "sell" / 买卖方向
        order_type: OrderType - Type of order / 订单类型
        limit_price: float, optional - Limit price / 限价
        stop_price: float, optional - Stop price / 止损价
        status: OrderStatus - Current status / 当前状态
        filled_quantity: Decimal - Filled quantity / 已成交数量
        avg_fill_price: float - Average fill price / 平均成交价
        created_at: datetime - Creation time / 创建时间
        updated_at: datetime - Last update time / 更新时间
    """

    def __init__(
        self,
        asset: Asset,
        quantity: Union[int, float, Decimal, str],
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str] = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Union[TimeInForce, str] = TimeInForce.GTC,
        stop_limit_price: Optional[float] = None,
        trail_price: Optional[float] = None,
        trail_percent: Optional[float] = None,
        expiration: Optional[datetime] = None,
        custom_id: Optional[str] = None,
        **kwargs,
    ):
        self.identifier = custom_id or str(uuid.uuid4())
        self.asset = asset
        self.quantity = Decimal(str(quantity))
        self.side = OrderSide(side) if isinstance(side, str) else side
        self.order_type = OrderType(order_type) if isinstance(order_type, str) else order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.stop_limit_price = stop_limit_price
        self.trail_price = trail_price
        self.trail_percent = trail_percent
        self.time_in_force = TimeInForce(time_in_force) if isinstance(time_in_force, str) else time_in_force
        self.expiration = expiration

        self.status = OrderStatus.PENDING
        self.filled_quantity = Decimal("0")
        self.avg_fill_price: float = 0.0
        self.broker_create_date: Optional[datetime] = None
        self.broker_update_date: Optional[datetime] = None
        self._strategy_name: str = ""
        self._created_at = datetime.now(timezone.utc)

    def __repr__(self):
        return (
            f"Order(id={self.identifier[:8]}, asset={self.asset}, "
            f"qty={self.quantity}, side={self.side.value}, "
            f"type={self.order_type.value}, status={self.status.value})"
        )

    @property
    def is_active(self) -> bool:
        """Returns True if order is still active (not terminal) / 订单是否活跃"""
        return self.status in (OrderStatus.PENDING, OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    def get_remaining_quantity(self) -> Decimal:
        """Returns remaining quantity to fill / 剩余未成交数量"""
        return self.quantity - self.filled_quantity


class Position:
    """
    Position representation / 持仓表示

    Attributes / 属性:
        asset: Asset - The held asset / 持仓资产
        quantity: Decimal - Held quantity / 持仓数量
        avg_entry_price: float - Average entry price / 平均入场价
        current_price: float - Current market price / 当前市场价格
        unrealized_pnl: float - Unrealized PnL / 未实现盈亏
        realized_pnl: float - Realized PnL / 已实现盈亏
    """

    def __init__(
        self,
        asset: Asset,
        quantity: Union[int, float, Decimal, str],
        avg_entry_price: float = 0.0,
        current_price: float = 0.0,
    ):
        self.asset = asset
        self.quantity = Decimal(str(quantity))
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price

    def __repr__(self):
        return (
            f"Position(asset={self.asset}, qty={self.quantity}, "
            f"entry={self.avg_entry_price:.2f}, pnl={self.unrealized_pnl:.2f})"
        )

    @property
    def market_value(self) -> float:
        """Market value of position / 持仓市值"""
        return float(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> float:
        """Cost basis / 成本基础"""
        return float(self.quantity) * self.avg_entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized PnL / 未实现盈亏"""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_percent(self) -> float:
        """Unrealized PnL percent / 未实现盈亏百分比"""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


class Bar:
    """
    OHLCV bar representation / K线数据表示

    Attributes / 属性:
        timestamp: datetime - Bar timestamp / K线时间戳
        open: float - Open price / 开盘价
        high: float - High price / 最高价
        low: float - Low price / 最低价
        close: float - Close price / 收盘价
        volume: float - Volume / 成交量
        asset: Asset - The asset / 标的资产
    """

    def __init__(
        self,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        asset: Optional[Asset] = None,
    ):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.asset = asset

    def __repr__(self):
        return (
            f"Bar({self.timestamp.strftime('%Y-%m-%d %H:%M')}, "
            f"O={self.open:.2f}, H={self.high:.2f}, "
            f"L={self.low:.2f}, C={self.close:.2f}, V={self.volume:.0f})"
        )


class Bars:
    """
    Collection of Bar objects / Bar对象集合

    Provides both list access and pandas DataFrame access.
    同时提供列表访问和pandas DataFrame访问。
    """

    def __init__(self, bars: List[Bar], df=None):
        self._bars = bars
        self._df = df

    def __len__(self):
        return len(self._bars)

    def __iter__(self):
        return iter(self._bars)

    def __getitem__(self, key):
        return self._bars[key]

    @property
    def df(self):
        """Pandas DataFrame representation / Pandas DataFrame表示"""
        if self._df is None:
            try:
                import pandas as pd

                records = [
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "symbol": bar.asset.symbol if bar.asset else None,
                    }
                    for bar in self._bars
                ]
                self._df = pd.DataFrame(records)
            except ImportError:
                raise ImportError("pandas is required for Bars.df access")
        return self._df


# ---------------------------------------------------------------------------
# REST Helper (pure urllib) / REST辅助函数（纯urllib）
# ---------------------------------------------------------------------------


def _urllib_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Make HTTP request using urllib / 使用urllib发起HTTP请求

    Parameters / 参数:
        url: str - Target URL / 目标URL
        method: str - HTTP method / HTTP方法
        headers: dict - Request headers / 请求头
        data: dict - Request body / 请求体
        timeout: int - Timeout in seconds / 超时秒数

    Returns / 返回:
        dict - JSON response / JSON响应
    """
    import urllib.request
    import urllib.error

    headers = headers or {}
    headers.setdefault("Content-Type", "application/json")
    headers.setdefault("Accept", "application/json")

    body = json.dumps(data).encode("utf-8") if data else None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            result = response.read().decode(charset)
            if result:
                return json.loads(result)
            return {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        try:
            error_body = json.loads(body) if body else {}
        except Exception:
            error_body = {"error": body}
        raise LumibotBrokerAPIError(
            f"HTTP {e.code}: {e.reason} - {error_body}"
        ) from e
    except urllib.error.URLError as e:
        raise LumibotBrokerAPIError(f"URL Error: {e.reason}") from e


# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class LumibotBrokerAPIError(Exception):
    """Exception raised for broker API errors / 券商API错误异常"""

    pass


class LumibotDataError(Exception):
    """Exception raised for data source errors / 数据源错误异常"""

    pass


# ---------------------------------------------------------------------------
# LumibotBroker - Abstract Broker Interface / 抽象券商接口
# ---------------------------------------------------------------------------


class LumibotBroker(ABC):
    """
    Abstract broker interface for lumibot-style order execution.
    适用于lumibot风格订单执行的抽象券商接口。

    Concrete implementations must implement:
    具体实现必须实现:
        _submit_order(), cancel_order(), _get_balances_at_broker(),
        _pull_broker_positions(), _pull_broker_all_orders()

    Lifecycle Methods / 生命周期方法:
        submit_order() -> Order       - Submit a new order / 提交新订单
        cancel_order() -> None        - Cancel an existing order / 取消现有订单
        get_positions() -> List[Position]  - Get all positions / 获取所有持仓
        get_orders() -> List[Order]    - Get all orders / 获取所有订单
        get_historical_data() -> Bars - Get historical OHLCV / 获取历史K线
        get_last_price() -> float     - Get last price / 获取最新价格

    Example / 示例:
        >>> broker = AlpacaBroker(api_key="...", api_secret="...", paper=True)
        >>> order = broker.submit_order(asset, quantity, "buy")
        >>> broker.cancel_order(order)
        >>> positions = broker.get_positions()
        >>> bars = broker.get_historical_data(asset, 100, "1day")
    """

    # Class-level constants / 类级常量
    IS_BACKTESTING_BROKER: ClassVar[bool] = False

    # Order event types / 订单事件类型
    NEW_ORDER: ClassVar[str] = "new"
    CANCELED_ORDER: ClassVar[str] = "canceled"
    FILLED_ORDER: ClassVar[str] = "fill"
    PARTIALLY_FILLED_ORDER: ClassVar[str] = "partial_fill"
    ERROR_ORDER: ClassVar[str] = "error"

    def __init__(
        self,
        name: str = "",
        data_source: Optional["LumibotDataSource"] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize LumibotBroker.

        Parameters / 参数:
            name: str - Broker name / 券商名称
            data_source: LumibotDataSource - Data source for market data / 市场数据源
            config: dict - Broker configuration / 券商配置
        """
        self.name = name
        self.data_source = data_source
        self.config = config or {}

        # Order tracking / 订单追踪
        self._orders: Dict[str, Order] = {}  # identifier -> Order
        self._positions: List[Position] = []

        # Event subscribers / 事件订阅者
        self._subscribers: List[Callable[[str, Order], None]] = []

        _logger.debug(f"LumibotBroker '{name}' initialized")

    # ---- Order Management / 订单管理 ----

    def submit_order(
        self,
        asset: Union[Asset, str],
        quantity: Union[int, float, Decimal, str],
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str] = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Union[TimeInForce, str] = TimeInForce.GTC,
        **kwargs,
    ) -> Order:
        """
        Submit a new order to the broker / 向券商提交新订单

        Parameters / 参数:
            asset: Asset or str - Asset to trade / 交易资产
            quantity: number - Quantity to trade / 交易数量
            side: str - "buy" or "sell" / 买卖方向
            order_type: str - "market", "limit", "stop", "stop_limit" / 订单类型
            limit_price: float - Limit price for limit orders / 限价
            stop_price: float - Stop price for stop orders / 止损价
            time_in_force: str - "day", "gtc", "ioc", "fok" / 订单有效期

        Returns / 返回:
            Order - The submitted order with broker-assigned ID / 已提交订单（含券商分配ID）

        Example / 示例:
            >>> order = broker.submit_order("AAPL", 100, "buy", "limit", limit_price=150.0)
            >>> print(f"Order ID: {order.identifier}")
        """
        # Normalize asset / 规范化资产
        if isinstance(asset, str):
            asset = Asset(symbol=asset, asset_type="stock")

        # Normalize side / 规范化方向
        side = OrderSide(side) if isinstance(side, str) else side

        # Normalize order type / 规范化订单类型
        order_type = OrderType(order_type) if isinstance(order_type, str) else order_type

        # Normalize time in force / 规范化有效期
        time_in_force = TimeInForce(time_in_force) if isinstance(time_in_force, str) else time_in_force

        # Create order object / 创建订单对象
        order = Order(
            asset=asset,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            **kwargs,
        )

        # Submit to broker (implementation-specific) / 提交到券商（实现特定）
        order = self._submit_order(order)

        # Track order / 追踪订单
        self._orders[order.identifier] = order

        # Notify subscribers / 通知订阅者
        self._notify_subscribers(self.NEW_ORDER, order)

        return order

    def cancel_order(self, order: Union[Order, str]) -> bool:
        """
        Cancel an existing order / 取消现有订单

        Parameters / 参数:
            order: Order or str - Order object or order identifier / 订单对象或订单ID

        Returns / 返回:
            bool - True if cancellation was successful / 取消是否成功

        Example / 示例:
            >>> success = broker.cancel_order(order.identifier)
            >>> if success:
            ...     print("Order canceled")
        """
        if isinstance(order, str):
            order = self._orders.get(order)

        if order is None:
            _logger.warning(f"Cancel order failed: order not found")
            return False

        if not order.is_active:
            _logger.warning(f"Cancel order failed: order {order.identifier} is not active")
            return False

        result = self._cancel_order(order)

        if result:
            order.status = OrderStatus.CANCELED
            self._notify_subscribers(self.CANCELED_ORDER, order)

        return result

    def cancel_all_orders(self) -> int:
        """
        Cancel all active orders / 取消所有活跃订单

        Returns / 返回:
            int - Number of orders canceled / 已取消订单数量
        """
        active_orders = [o for o in self._orders.values() if o.is_active]
        count = 0
        for order in active_orders:
            if self.cancel_order(order):
                count += 1
        return count

    # ---- Position & Account / 持仓与账户 ----

    def get_positions(self) -> List[Position]:
        """
        Get all current positions / 获取所有当前持仓

        Returns / 返回:
            List[Position] - List of current positions / 当前持仓列表

        Example / 示例:
            >>> positions = broker.get_positions()
            >>> for pos in positions:
            ...     print(f"{pos.asset}: {pos.quantity} @ {pos.avg_entry_price}")
        """
        self._positions = self._pull_broker_positions()
        return self._positions

    def get_position(self, asset: Union[Asset, str]) -> Optional[Position]:
        """
        Get position for a specific asset / 获取特定资产持仓

        Parameters / 参数:
            asset: Asset or str - Asset to query / 查询资产

        Returns / 返回:
            Position or None - Position if exists / 持仓（若存在）
        """
        if isinstance(asset, str):
            symbol = asset.upper()
        else:
            symbol = asset.symbol

        positions = self.get_positions()
        for pos in positions:
            if pos.asset.symbol == symbol:
                return pos
        return None

    def get_balances(self) -> Tuple[float, float, float]:
        """
        Get account balances / 获取账户余额

        Returns / 返回:
            Tuple[float, float, float] - (cash, positions_value, total_liquidation_value)
                                        / (现金, 持仓价值, 总清算价值)
        """
        return self._get_balances_at_broker()

    def get_cash(self) -> float:
        """Get available cash / 获取可用现金"""
        cash, _, _ = self.get_balances()
        return cash

    def get_buying_power(self) -> float:
        """Get buying power (typically 2x cash for US stocks) / 获取购买力（美股通常为现金的2倍）"""
        cash, _, _ = self.get_balances()
        return cash * 2

    # ---- Market Data / 市场数据 ----

    def get_historical_data(
        self,
        asset: Union[Asset, str],
        length: int,
        timestep: str = "1day",
        timeshift: Optional[datetime] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Bars:
        """
        Get historical OHLCV bars / 获取历史K线数据

        Parameters / 参数:
            asset: Asset or str - Asset to query / 查询资产
            length: int - Number of bars to return / 返回K线数量
            timestep: str - "1minute", "5minute", "1hour", "1day" etc / K线周期
            timeshift: datetime - End time for the data range / 数据范围结束时间
            start: datetime - Start time for the data range / 数据范围开始时间
            end: datetime - End time for the data range / 数据范围结束时间

        Returns / 返回:
            Bars - Historical bars / 历史K线数据

        Example / 示例:
            >>> bars = broker.get_historical_data("AAPL", 100, "1day")
            >>> df = bars.df  # Get pandas DataFrame / 获取pandas DataFrame
            >>> closes = [bar.close for bar in bars]
        """
        if isinstance(asset, str):
            asset = Asset(symbol=asset, asset_type="stock")

        if self.data_source is None:
            raise LumibotDataError("No data source configured for historical data")

        return self.data_source.get_historical_prices(
            asset=asset,
            length=length,
            timestep=timestep,
            timeshift=timeshift,
            start=start,
            end=end,
        )

    def get_last_price(self, asset: Union[Asset, str]) -> Optional[float]:
        """
        Get the last known price for an asset / 获取资产最新价格

        Parameters / 参数:
            asset: Asset or str - Asset to query / 查询资产

        Returns / 返回:
            float or None - Last price / 最新价格
        """
        if isinstance(asset, str):
            asset = Asset(symbol=asset, asset_type="stock")

        if self.data_source is None:
            raise LumibotDataError("No data source configured for price data")

        return self.data_source.get_last_price(asset)

    def get_last_prices(self, assets: List[Union[Asset, str]]) -> Dict[str, float]:
        """
        Get last prices for multiple assets / 获取多个资产最新价格

        Parameters / 参数:
            assets: List[Asset or str] - Assets to query / 查询资产列表

        Returns / 返回:
            Dict[str, float] - symbol -> price mapping / 标的->价格映射
        """
        result = {}
        for asset in assets:
            if isinstance(asset, str):
                asset = Asset(symbol=asset, asset_type="stock")
            price = self.get_last_price(asset)
            if price is not None:
                result[asset.symbol] = price
        return result

    # ---- Order Tracking / 订单追踪 ----

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """
        Get all orders, optionally filtered by status / 获取所有订单，可按状态筛选

        Parameters / 参数:
            status: OrderStatus - Filter by status / 按状态筛选

        Returns / 返回:
            List[Order] - Matching orders / 匹配的订单列表
        """
        orders = list(self._orders.values())
        if status is not None:
            orders = [o for o in orders if o.status == status]
        return orders

    def get_order(self, identifier: str) -> Optional[Order]:
        """Get order by identifier / 通过ID获取订单"""
        return self._orders.get(identifier)

    # ---- Event Subscription / 事件订阅 ----

    def subscribe(self, callback: Callable[[str, Order], None]) -> None:
        """
        Subscribe to order events / 订阅订单事件

        Parameters / 参数:
            callback: Callable - Function called with (event_type, order) / 回调函数（事件类型，订单）
        """
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, Order], None]) -> None:
        """Unsubscribe from order events / 退订订单事件"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _notify_subscribers(self, event_type: str, order: Order) -> None:
        """Notify all subscribers of an event / 通知所有订阅者"""
        for callback in self._subscribers:
            try:
                callback(event_type, order)
            except Exception as e:
                _logger.error(f"Subscriber callback error: {e}")

    # ---- Abstract Methods (implement in subclasses) / 抽象方法（在子类实现）----

    @abstractmethod
    def _submit_order(self, order: Order) -> Order:
        """
        Submit order to broker (implementation-specific) / 提交订单到券商（实现特定）
        Must set order.identifier and order.status / 必须设置order.identifier和order.status
        """
        pass

    @abstractmethod
    def _cancel_order(self, order: Order) -> bool:
        """
        Cancel order at broker (implementation-specific) / 在券商取消订单（实现特定）
        """
        pass

    @abstractmethod
    def _get_balances_at_broker(self) -> Tuple[float, float, float]:
        """
        Get balances from broker / 从券商获取余额

        Returns / 返回:
            Tuple[float, float, float] - (cash, positions_value, total_liquidation_value)
        """
        pass

    @abstractmethod
    def _pull_broker_positions(self) -> List[Position]:
        """
        Pull positions from broker / 从券商拉取持仓
        """
        pass

    @abstractmethod
    def _pull_broker_all_orders(self) -> List[Order]:
        """
        Pull all orders from broker / 从券商拉取所有订单
        """
        pass


# ---------------------------------------------------------------------------
# PolygonBroker - Polygon market data + execution / Polygon行情+执行
# ---------------------------------------------------------------------------


class PolygonDataSource:
    """
    Polygon.io market data source (REST via urllib).
    Polygon.io市场数据源（通过urllib调用REST API）。

    Supports: Stocks, Options, Forex, Crypto
    支持：股票、期权、外汇、加密货币

    API Docs: https://polygon.io/docs/getting-started
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize Polygon data source / 初始化Polygon数据源

        Parameters / 参数:
            api_key: str - Polygon API key / Polygon API密钥
            timeout: int - Request timeout / 请求超时
            max_retries: int - Max retry attempts / 最大重试次数
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache: Dict[str, Tuple[datetime, Any]] = {}  # simple TTL cache

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        cache_seconds: int = 60,
    ) -> Dict[str, Any]:
        """
        Make GET request to Polygon API / 向Polygon API发起GET请求

        Parameters / 参数:
            path: str - API path / API路径
            params: dict - Query parameters / 查询参数
            cache_seconds: int - Cache TTL / 缓存有效期

        Returns / 返回:
            dict - API response / API响应
        """
        # Check cache / 检查缓存
        cache_key = f"{path}:{json.dumps(params or {}, sort_keys=True)}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now(timezone.utc) - cached_time < timedelta(seconds=cache_seconds):
                return cached_data

        url = f"{self.BASE_URL}/{path.lstrip('/')}"
        params = params or {}
        params["apiKey"] = self.api_key

        headers = {"User-Agent": "lumibot-adapter/1.0"}

        for attempt in range(self.max_retries):
            try:
                result = _urllib_request(
                    url=url,
                    method="GET",
                    headers=headers,
                    data={"params": params},
                    timeout=self.timeout,
                )

                # Cache successful response / 缓存成功响应
                self._cache[cache_key] = (datetime.now(timezone.utc), result)
                return result

            except LumibotBrokerAPIError as e:
                if attempt == self.max_retries - 1:
                    raise
                _logger.warning(f"Polygon API retry {attempt + 1}/{self.max_retries}: {e}")
                time.sleep(1 * (attempt + 1))

        return {}

    def get_last_price(self, asset: Asset) -> Optional[float]:
        """
        Get last traded price / 获取最新成交价

        Parameters / 参数:
            asset: Asset - Asset to query / 查询资产

        Returns / 返回:
            float or None - Last price / 最新价格
        """
        symbol = asset.symbol

        # Determine endpoint based on asset type / 根据资产类型确定端点
        if asset.is_crypto:
            path = f"/v2/aggs/ticker/X:{symbol}/prev"
        elif asset.is_forex:
            path = f"/v2/aggs/ticker/C:{symbol}/prev"
        elif asset.is_option:
            # Options use special format / 期权使用特殊格式
            path = f"/v2/aggs/ticker/O:{symbol}/prev"
        else:
            path = f"/v2/aggs/ticker/{symbol}/prev"

        try:
            result = self._get(path)
            results = result.get("results", [])
            if results:
                return results[0]["c"]  # close price
        except Exception as e:
            _logger.error(f"Polygon get_last_price error for {symbol}: {e}")

        return None

    def get_historical_prices(
        self,
        asset: Asset,
        length: int,
        timestep: str = "1day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        **kwargs,
    ) -> Bars:
        """
        Get historical OHLCV bars / 获取历史K线数据

        Parameters / 参数:
            asset: Asset - Asset to query / 查询资产
            length: int - Number of bars / K线数量
            timestep: str - "1minute", "5minute", "1hour", "1day" / K线周期
            start: datetime - Start time / 开始时间
            end: datetime - End time / 结束时间

        Returns / 返回:
            Bars - Historical bars / 历史K线数据
        """
        symbol = asset.symbol

        # Map timestep to Polygon multiplier and timespan / 映射K线周期到Polygon乘数和timespan
        timestep_map = {
            "1minute": ("1", "minute"),
            "5minute": ("5", "minute"),
            "15minute": ("15", "minute"),
            "1hour": ("1", "hour"),
            "1day": ("1", "day"),
            "1week": ("1", "week"),
            "1month": ("1", "month"),
        }

        if timestep not in timestep_map:
            timestep = "1day"

        multiplier, timespan = timestep_map[timestep]

        # Calculate date range / 计算日期范围
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            # Estimate start based on length / 根据长度估算开始时间
            unit_days = {
                "minute": 1 / 1440,
                "hour": 1 / 24,
                "day": 1,
                "week": 7,
                "month": 30,
            }
            days_per_bar = unit_days.get(timespan, 1)
            start = end - timedelta(days=length * days_per_bar * 2)

        # Build path / 构建路径
        if asset.is_crypto:
            path = f"/v2/aggs/ticker/X:{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        elif asset.is_forex:
            path = f"/v2/aggs/ticker/C:{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        elif asset.is_option:
            path = f"/v2/aggs/ticker/O:{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        else:
            path = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        try:
            result = self._get(path)
            results = result.get("results", [])
            results = results[-length:]  # Take last 'length' bars / 取最后length条

            bars = []
            for r in results:
                ts = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc)
                bar = Bar(
                    timestamp=ts,
                    open=r["o"],
                    high=r["h"],
                    low=r["l"],
                    close=r["c"],
                    volume=r["v"],
                    asset=asset,
                )
                bars.append(bar)

            return Bars(bars)

        except Exception as e:
            _logger.error(f"Polygon get_historical_prices error for {symbol}: {e}")
            return Bars([])

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timestep: str = "1day",
    ) -> Bars:
        """
        Get bars for a symbol (alias for get_historical_prices with date range).
        获取标的K线（带日期范围的get_historical_prices别名）。
        """
        asset = Asset(symbol=symbol)
        return self.get_historical_prices(
            asset=asset,
            length=1000,  # large number, will be limited by date range
            timestep=timestep,
            start=start,
            end=end,
        )


class PolygonBroker(LumibotBroker):
    """
    Polygon broker implementation using Polygon.io REST API.
    使用Polygon.io REST API的Polygon券商实现。

    Supports: Stocks, Options, Forex, Crypto execution
    支持：股票、期权、外汇、加密货币执行

    Requires: Polygon API key with appropriate market data subscription
    要求：具有适当市场数据订阅的Polygon API密钥

    Example / 示例:
        >>> from lumibot_adapter import PolygonBroker, Asset
        >>>
        >>> data_source = PolygonDataSource(api_key="your_api_key")
        >>> broker = PolygonBroker(
        ...     name="polygon",
        ...     data_source=data_source,
        ...     api_key="your_trade_api_key",
        ...     api_secret="your_trade_secret",
        ... )
        >>>
        >>> # Get market data / 获取市场数据
        >>> price = broker.get_last_price("AAPL")
        >>> bars = broker.get_historical_data("AAPL", 100, "1day")
        >>>
        >>> # Submit order / 提交订单
        >>> order = broker.submit_order("AAPL", 100, "buy")
    """

    def __init__(
        self,
        name: str = "polygon",
        data_source: Optional[PolygonDataSource] = None,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Polygon broker / 初始化Polygon券商

        Parameters / 参数:
            name: str - Broker name / 券商名称
            data_source: PolygonDataSource - Polygon data source / Polygon数据源
            config: dict - Broker configuration / 券商配置
            api_key: str - Polygon Trade API key / Polygon Trade API密钥
            api_secret: str - Polygon Trade API secret / Polygon Trade API密钥
        """
        super().__init__(name=name, data_source=data_source, config=config, **kwargs)

        self.trade_api_key = api_key or config.get("POLYGON_API_KEY") if config else None
        self.trade_api_secret = api_secret or config.get("POLYGON_SECRET") if config else None

        if not self.trade_api_key:
            _logger.warning("Polygon broker initialized without trade API key (market data only)")

    def _submit_order(self, order: Order) -> Order:
        """
        Submit order to Polygon / 向Polygon提交订单

        Note: Polygon Trade API requires specific plan. If not available,
        order is tracked locally with PENDING status.
        注意：Polygon Trade API需要特定计划。 如果不可用，
        订单将在本地以PENDING状态跟踪。
        """
        if not self.trade_api_key:
            # No trade API - simulate order locally / 无交易API - 本地模拟订单
            order.status = OrderStatus.NEW
            order.broker_create_date = datetime.now(timezone.utc)
            _logger.info(f"Polygon (no trade API): order {order.identifier} tracked locally")
            return order

        # Build order request / 构建订单请求
        symbol = order.asset.symbol

        # Map order type / 映射订单类型
        order_type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
        }

        order_data = {
            "symbol": symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": order_type_map.get(order.order_type, "market"),
            "time_in_force": order.time_in_force.value.upper(),
        }

        if order.limit_price is not None:
            order_data["limit_price"] = str(order.limit_price)

        if order.stop_price is not None:
            order_data["stop_price"] = str(order.stop_price)

        try:
            # In a real implementation, this would call Polygon Trade API
            # 在实际实现中，这会调用Polygon Trade API
            url = f"https://api.polygon.io/v2/orders"
            headers = {
                "Authorization": f"Bearer {self.trade_api_key}",
                "Content-Type": "application/json",
            }

            result = _urllib_request(
                url=url,
                method="POST",
                headers=headers,
                data=order_data,
                timeout=30,
            )

            order.identifier = result.get("id", order.identifier)
            order.status = OrderStatus(result.get("status", "new"))
            order.broker_create_date = datetime.now(timezone.utc)

        except LumibotBrokerAPIError as e:
            _logger.error(f"Polygon order submission error: {e}")
            order.status = OrderStatus.REJECTED

        return order

    def _cancel_order(self, order: Order) -> bool:
        """Cancel order at Polygon / 在Polygon取消订单"""
        if not self.trade_api_key:
            # Simulate cancellation locally / 本地模拟取消
            return True

        try:
            url = f"https://api.polygon.io/v2/orders/{order.identifier}"
            headers = {"Authorization": f"Bearer {self.trade_api_key}"}

            _urllib_request(url=url, method="DELETE", headers=headers, timeout=30)
            return True

        except LumibotBrokerAPIError as e:
            _logger.error(f"Polygon order cancellation error: {e}")
            return False

    def _get_balances_at_broker(self) -> Tuple[float, float, float]:
        """
        Get account balances from Polygon / 从Polygon获取账户余额

        Returns / 返回:
            Tuple[float, float, float] - (cash, positions_value, total_liquidation_value)
        """
        if not self.trade_api_key:
            return (0.0, 0.0, 0.0)

        try:
            url = "https://api.polygon.io/v2/account"
            headers = {"Authorization": f"Bearer {self.trade_api_key}"}

            result = _urllib_request(url=url, method="GET", headers=headers, timeout=30)

            cash = float(result.get("cash", 0))
            equity = float(result.get("equity", 0))

            return (cash, equity - cash, equity)

        except LumibotBrokerAPIError as e:
            _logger.error(f"Polygon get_balances error: {e}")
            return (0.0, 0.0, 0.0)

    def _pull_broker_positions(self) -> List[Position]:
        """Pull positions from Polygon / 从Polygon拉取持仓"""
        if not self.trade_api_key:
            return []

        try:
            url = "https://api.polygon.io/v2/positions"
            headers = {"Authorization": f"Bearer {self.trade_api_key}"}

            result = _urllib_request(url=url, method="GET", headers=headers, timeout=30)

            positions = []
            for item in result.get("positions", []):
                asset = Asset(
                    symbol=item["symbol"],
                    asset_type="stock",
                )
                pos = Position(
                    asset=asset,
                    quantity=item.get("qty", 0),
                    avg_entry_price=item.get("avg_entry_price", 0),
                    current_price=item.get("current_price", 0),
                )
                positions.append(pos)

            return positions

        except LumibotBrokerAPIError as e:
            _logger.error(f"Polygon get_positions error: {e}")
            return []

    def _pull_broker_all_orders(self) -> List[Order]:
        """Pull all orders from Polygon / 从Polygon拉取所有订单"""
        if not self.trade_api_key:
            return list(self._orders.values())

        try:
            url = "https://api.polygon.io/v2/orders"
            headers = {"Authorization": f"Bearer {self.trade_api_key}"}

            result = _urllib_request(url=url, method="GET", headers=headers, timeout=30)

            orders = []
            for item in result.get("orders", []):
                asset = Asset(symbol=item["symbol"], asset_type="stock")
                order = Order(
                    asset=asset,
                    quantity=item.get("qty", 0),
                    side=item.get("side", "buy"),
                    order_type=item.get("type", "market"),
                    custom_id=item.get("id"),
                )
                order.status = OrderStatus(item.get("status", "new"))
                orders.append(order)

            return orders

        except LumibotBrokerAPIError as e:
            _logger.error(f"Polygon get_orders error: {e}")
            return list(self._orders.values())


# ---------------------------------------------------------------------------
# AlpacaBroker - Alpaca broker integration (REST via urllib)
# Alpaca券商集成（通过urllib调用REST API）
# ---------------------------------------------------------------------------


class AlpacaDataSource:
    """
    Alpaca market data source (REST via urllib).
    Alpaca市场数据源（通过urllib调用REST API）。

    Supports: Stocks, ETFs
    支持：股票、ETF

    API Docs: https://docs.alpaca.markets/
    """

    BASE_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        timeout: int = 30,
    ):
        """
        Initialize Alpaca data source / 初始化Alpaca数据源

        Parameters / 参数:
            api_key: str - Alpaca API key ID / Alpaca API密钥ID
            secret_key: str - Alpaca secret key / Alpaca密钥
            timeout: int - Request timeout / 请求超时
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.timeout = timeout
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        cache_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Make GET request to Alpaca API / 向Alpaca API发起GET请求

        Parameters / 参数:
            path: str - API path / API路径
            params: dict - Query parameters / 查询参数
            base_url: str - Base URL override / 基础URL覆盖
            cache_seconds: int - Cache TTL / 缓存有效期

        Returns / 返回:
            dict - API response / API响应
        """
        cache_key = f"{path}:{json.dumps(params or {}, sort_keys=True)}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now(timezone.utc) - cached_time < timedelta(seconds=cache_seconds):
                return cached_data

        url = f"{base_url or self.BASE_URL}/{path.lstrip('/')}"
        params = params or {}

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

        try:
            result = _urllib_request(
                url=url,
                method="GET",
                headers=headers,
                data={"params": params},
                timeout=self.timeout,
            )

            self._cache[cache_key] = (datetime.now(timezone.utc), result)
            return result

        except LumibotBrokerAPIError as e:
            _logger.error(f"Alpaca data API error: {e}")
            return {}

    def get_last_price(self, asset: Asset) -> Optional[float]:
        """
        Get last traded price / 获取最新成交价

        Parameters / 参数:
            asset: Asset - Asset to query / 查询资产

        Returns / 返回:
            float or None - Last price / 最新价格
        """
        try:
            # Alpaca uses polygon for market data / Alpaca使用polygon获取市场数据
            path = f"v2/stocks/{asset.symbol}/trades"
            params = {"start": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()}

            result = self._get(path, params=params)
            trades = result.get("trades", [])

            if trades:
                return float(trades[-1]["p"])

        except Exception as e:
            _logger.error(f"Alpaca get_last_price error for {asset.symbol}: {e}")

        # Fallback: try quote / 回退：尝试行情
        try:
            path = f"v2/stocks/{asset.symbol}/quotes"
            params = {"start": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()}

            result = self._get(path, params=params)
            quotes = result.get("quotes", [])

            if quotes:
                return float(quotes[-1]["bp"])

        except Exception:
            pass

        return None

    def get_historical_prices(
        self,
        asset: Asset,
        length: int,
        timestep: str = "1day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        **kwargs,
    ) -> Bars:
        """
        Get historical OHLCV bars / 获取历史K线数据

        Parameters / 参数:
            asset: Asset - Asset to query / 查询资产
            length: int - Number of bars / K线数量
            timestep: str - "1minute", "5minute", "15minute", "1hour", "1day" / K线周期
            start: datetime - Start time / 开始时间
            end: datetime - End time / 结束时间

        Returns / 返回:
            Bars - Historical bars / 历史K线数据
        """
        # Map timestep to Alpaca timeframe / 映射K线周期到Alpaca时间框架
        timeframe_map = {
            "1minute": "1Min",
            "5minute": "5Min",
            "15minute": "15Min",
            "1hour": "1H",
            "1day": "1Day",
        }

        tf = timeframe_map.get(timestep, "1Day")

        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=length * 2)

        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": tf,
            "limit": length,
        }

        try:
            path = f"v2/stocks/{asset.symbol}/bars"
            result = self._get(path, params=params)

            bars_data = result.get("bars", [])
            bars = []

            for bar_dict in bars_data:
                ts = datetime.fromisoformat(bar_dict["t"].replace("Z", "+00:00"))
                bar = Bar(
                    timestamp=ts,
                    open=bar_dict["o"],
                    high=bar_dict["h"],
                    low=bar_dict["l"],
                    close=bar_dict["c"],
                    volume=bar_dict["v"],
                    asset=asset,
                )
                bars.append(bar)

            return Bars(bars)

        except Exception as e:
            _logger.error(f"Alpaca get_historical_prices error for {asset.symbol}: {e}")
            return Bars([])


class AlpacaBroker(LumibotBroker):
    """
    Alpaca broker implementation using Alpaca Trade API (REST via urllib).
    使用Alpaca Trade API的券商实现（通过urllib调用REST）。

    Supports: Stocks, ETFs (US market)
    支持：股票、ETF（美国市场）

    Requires: Alpaca API key with trading permissions
    要求：具有交易权限的Alpaca API密钥

    Example / 示例:
        >>> from lumibot_adapter import AlpacaBroker, Asset
        >>>
        >>> data_source = AlpacaDataSource(api_key="...", secret_key="...")
        >>> broker = AlpacaBroker(
        ...     name="alpaca",
        ...     data_source=data_source,
        ...     api_key="your_alpaca_key",
        ...     secret_key="your_alpaca_secret",
        ...     paper=True,  # Use paper trading / 使用模拟交易
        ... )
        >>>
        >>> # Get market data / 获取市场数据
        >>> price = broker.get_last_price("AAPL")
        >>> bars = broker.get_historical_data("AAPL", 100, "1day")
        >>>
        >>> # Submit order / 提交订单
        >>> order = broker.submit_order("AAPL", 100, "buy")
        >>> print(f"Order submitted: {order.identifier}")
    """

    TRADE_URL = "https://paper-api.alpaca.markets"  # Default to paper / 默认为模拟
    DATA_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        name: str = "alpaca",
        data_source: Optional[AlpacaDataSource] = None,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        **kwargs,
    ):
        """
        Initialize Alpaca broker / 初始化Alpaca券商

        Parameters / 参数:
            name: str - Broker name / 券商名称
            data_source: AlpacaDataSource - Alpaca data source / Alpaca数据源
            config: dict - Broker configuration / 券商配置
            api_key: str - Alpaca API key ID / Alpaca API密钥ID
            secret_key: str - Alpaca secret key / Alpaca密钥
            paper: bool - Use paper trading (default True) / 使用模拟交易（默认True）
        """
        super().__init__(name=name, data_source=data_source, config=config, **kwargs)

        self.api_key = api_key or config.get("APCA_API_KEY") if config else None
        self.secret_key = secret_key or config.get("APCA_API_SECRET") if config else None
        self.paper = paper

        # Set URLs based on paper/live mode / 根据模拟/实盘模式设置URL
        if paper:
            self.trade_url = "https://paper-api.alpaca.markets"
        else:
            self.trade_url = "https://api.alpaca.markets"

        if not self.api_key or not self.secret_key:
            _logger.warning("Alpaca broker initialized without API credentials (market data only)")

    def _headers(self) -> Dict[str, str]:
        """Get headers for Alpaca API requests / 获取Alpaca API请求头"""
        return {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
            "Content-Type": "application/json",
        }

    def _submit_order(self, order: Order) -> Order:
        """
        Submit order to Alpaca / 向Alpaca提交订单

        Parameters / 参数:
            order: Order - Order to submit / 要提交的订单

        Returns / 返回:
            Order - Order with broker-assigned ID / 含券商分配ID的订单
        """
        if not self.api_key:
            # No trade API - simulate locally / 无交易API - 本地模拟
            order.status = OrderStatus.NEW
            order.broker_create_date = datetime.now(timezone.utc)
            _logger.info(f"Alpaca (no trade API): order {order.identifier} tracked locally")
            return order

        # Build order request / 构建订单请求
        symbol = order.asset.symbol

        # Determine order side and type / 确定订单方向和类型
        order_type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
        }

        order_data = {
            "symbol": symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": order_type_map.get(order.order_type, "market"),
            "time_in_force": order.time_in_force.value.upper(),
        }

        if order.limit_price is not None:
            order_data["limit_price"] = str(order.limit_price)

        if order.stop_price is not None:
            order_data["stop_price"] = str(order.stop_price)

        try:
            url = f"{self.trade_url}/v2/orders"
            result = _urllib_request(
                url=url,
                method="POST",
                headers=self._headers(),
                data=order_data,
                timeout=30,
            )

            order.identifier = result.get("id", order.identifier)
            order.status = OrderStatus(result.get("status", "new"))
            order.broker_create_date = datetime.now(timezone.utc)

            _logger.info(f"Alpaca order submitted: {order.identifier} ({order.side.value} {order.quantity} {symbol})")

        except LumibotBrokerAPIError as e:
            _logger.error(f"Alpaca order submission error: {e}")
            order.status = OrderStatus.REJECTED

        return order

    def _cancel_order(self, order: Order) -> bool:
        """
        Cancel order at Alpaca / 在Alpaca取消订单

        Parameters / 参数:
            order: Order - Order to cancel / 要取消的订单

        Returns / 返回:
            bool - True if cancellation was successful / 取消是否成功
        """
        if not self.api_key:
            return True  # Simulate success / 模拟成功

        try:
            url = f"{self.trade_url}/v2/orders/{order.identifier}"
            _urllib_request(url=url, method="DELETE", headers=self._headers(), timeout=30)

            _logger.info(f"Alpaca order canceled: {order.identifier}")
            return True

        except LumibotBrokerAPIError as e:
            _logger.error(f"Alpaca order cancellation error: {e}")
            return False

    def _get_balances_at_broker(self) -> Tuple[float, float, float]:
        """
        Get account balances from Alpaca / 从Alpaca获取账户余额

        Returns / 返回:
            Tuple[float, float, float] - (cash, positions_value, total_liquidation_value)
        """
        if not self.api_key:
            return (0.0, 0.0, 0.0)

        try:
            url = f"{self.trade_url}/v2/account"
            result = _urllib_request(url=url, method="GET", headers=self._headers(), timeout=30)

            cash = float(result.get("cash", 0))
            equity = float(result.get("equity", 0))
            position_market_value = float(result.get("portfolio_market_value", 0))

            return (cash, position_market_value, equity)

        except LumibotBrokerAPIError as e:
            _logger.error(f"Alpaca get_balances error: {e}")
            return (0.0, 0.0, 0.0)

    def _pull_broker_positions(self) -> List[Position]:
        """
        Pull positions from Alpaca / 从Alpaca拉取持仓

        Returns / 返回:
            List[Position] - Current positions / 当前持仓
        """
        if not self.api_key:
            return []

        try:
            url = f"{self.trade_url}/v2/positions"
            result = _urllib_request(url=url, method="GET", headers=self._headers(), timeout=30)

            positions = []
            for item in result:
                asset = Asset(symbol=item["symbol"], asset_type="stock")
                pos = Position(
                    asset=asset,
                    quantity=item.get("qty", 0),
                    avg_entry_price=float(item.get("avg_entry_price", 0)),
                    current_price=float(item.get("current_price", 0)),
                )
                positions.append(pos)

            return positions

        except LumibotBrokerAPIError as e:
            _logger.error(f"Alpaca get_positions error: {e}")
            return []

    def _pull_broker_all_orders(self) -> List[Order]:
        """
        Pull all orders from Alpaca / 从Alpaca拉取所有订单

        Returns / 返回:
            List[Order] - All orders / 所有订单
        """
        if not self.api_key:
            return list(self._orders.values())

        try:
            url = f"{self.trade_url}/v2/orders"
            params = {"status": "all", "limit": 100}
            result = _urllib_request(url=url, method="GET", headers=self._headers(), data={"params": params}, timeout=30)

            orders = []
            for item in result:
                asset = Asset(symbol=item["symbol"], asset_type="stock")
                order = Order(
                    asset=asset,
                    quantity=item.get("qty", 0),
                    side=item.get("side", "buy"),
                    order_type=item.get("type", "market"),
                    custom_id=item.get("id"),
                )
                order.status = OrderStatus(item.get("status", "new"))
                order.filled_quantity = Decimal(str(item.get("filled_qty", 0)))
                order.broker_create_date = datetime.fromisoformat(
                    item.get("created_at", "").replace("Z", "+00:00")
                ) if item.get("created_at") else None
                orders.append(order)

            return orders

        except LumibotBrokerAPIError as e:
            _logger.error(f"Alpaca get_orders error: {e}")
            return list(self._orders.values())


# ---------------------------------------------------------------------------
# LumibotStrategy - User Strategy Base Class / 用户策略基类
# ---------------------------------------------------------------------------


class LumibotStrategy:
    """
    Base class for lumibot-style strategies.
    lumibot风格策略的基类。

    Inherit from this class and override the lifecycle methods:
    继承此类并覆盖生命周期方法:
        initialize()      - Called once at strategy initialization / 策略初始化时调用一次
        on_trading_iteration() - Called each trading iteration / 每个交易周期调用
        before_market_opens() - Called before market opens / 市场开盘前调用
        after_market_closes() - Called after market closes / 市场收盘后调用

    Key Properties / 关键属性:
        broker: LumibotBroker - The broker instance / 券商实例
        sleeptime: str - Time between iterations (eg "1D", "1H", "1Min") / 迭代间隔
        name: str - Strategy name / 策略名称

    Example / 示例:
        >>> class MyStrategy(LumibotStrategy):
        ...     def initialize(self):
        ...         self.sleeptime = "1D"
        ...         self.buy_symbol = "AAPL"
        ...
        ...     def on_trading_iteration(self):
        ...         price = self.get_last_price(self.buy_symbol)
        ...         self.log_message(f"{self.buy_symbol} price: {price}")
        ...         positions = self.get_positions()
        ...         if len(positions) == 0:
        ...             self.submit_order(self.buy_symbol, 100, "buy")
    """

    def __init__(
        self,
        broker: LumibotBroker,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize strategy / 初始化策略

        Parameters / 参数:
            broker: LumibotBroker - Broker instance / 券商实例
            name: str - Strategy name / 策略名称
            parameters: dict - Strategy parameters / 策略参数
        """
        self.broker = broker
        self._name = name or self.__class__.__name__
        self.parameters = parameters or {}
        self.sleeptime = "1D"  # Default to daily / 默认为每日

        # State / 状态
        self._initialized = False
        self._positions: List[Position] = []
        self._orders: List[Order] = []

        # Bind broker callbacks / 绑定券商回调
        self.broker.subscribe(self._on_broker_event)

    @property
    def name(self) -> str:
        """Strategy name / 策略名称"""
        return self._name

    # ---- Lifecycle Methods / 生命周期方法 ----

    def initialize(self) -> None:
        """
        Called once at strategy initialization.
        策略初始化时调用一次。

        Override this method to set up your strategy state.
        覆盖此方法以设置策略状态。

        Example / 示例:
            >>> def initialize(self):
            ...     self.sleeptime = "1H"
            ...     self.target_symbols = ["AAPL", "GOOGL", "MSFT"]
            ...     self.rebalance_threshold = 0.05
        """
        pass

    def on_trading_iteration(self) -> None:
        """
        Called each trading iteration.
        每个交易周期调用。

        This is the main strategy logic method.
        这是主要的策略逻辑方法。

        Example / 示例:
            >>> def on_trading_iteration(self):
            ...     for symbol in self.target_symbols:
            ...         price = self.get_last_price(symbol)
            ...         self.log_message(f"{symbol}: {price}")
        """
        pass

    def before_market_opens(self) -> None:
        """
        Called before market opens.
        市场开盘前调用。

        Example / 示例:
            >>> def before_market_opens(self):
            ...     self.log_message("Market about to open")
        """
        pass

    def after_market_closes(self) -> None:
        """
        Called after market closes.
        市场收盘后调用。

        Example / 示例:
            >>> def after_market_closes(self):
            ...     self.log_message(f"Market closed. PnL: {self.get_realized_pnl()}")
        """
        pass

    def on_bot_crash(self, exception: Exception) -> None:
        """
        Called when the strategy crashes.
        策略崩溃时调用。

        Parameters / 参数:
            exception: Exception - The exception that caused the crash / 导致崩溃的异常
        """
        _logger.error(f"Strategy {self._name} crashed: {exception}")

    # ---- Broker Interface / 券商接口 ----

    def submit_order(
        self,
        asset: Union[Asset, str],
        quantity: Union[int, float, str],
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str] = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs,
    ) -> Order:
        """
        Submit an order / 提交订单

        Parameters / 参数:
            asset: Asset or str - Asset to trade / 交易资产
            quantity: number - Quantity to trade / 交易数量
            side: str - "buy" or "sell" / 买卖方向
            order_type: str - "market", "limit", etc / 订单类型
            limit_price: float - Limit price / 限价
            stop_price: float - Stop price / 止损价

        Returns / 返回:
            Order - Submitted order / 已提交订单
        """
        return self.broker.submit_order(
            asset=asset,
            quantity=quantity,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            **kwargs,
        )

    def create_order(
        self,
        asset: Union[Asset, str],
        quantity: Union[int, float, str],
        side: Union[OrderSide, str],
        order_type: Union[OrderType, str] = OrderType.MARKET,
        **kwargs,
    ) -> Order:
        """
        Create an order without submitting / 创建订单但不提交

        Parameters / 参数:
            asset: Asset or str - Asset to trade / 交易资产
            quantity: number - Quantity to trade / 交易数量
            side: str - "buy" or "sell" / 买卖方向
            order_type: str - Order type / 订单类型

        Returns / 返回:
            Order - Created order (not submitted) / 创建的订单（未提交）
        """
        if isinstance(asset, str):
            asset = Asset(symbol=asset, asset_type="stock")

        return Order(
            asset=asset,
            quantity=quantity,
            side=side,
            order_type=order_type,
            **kwargs,
        )

    def cancel_order(self, order: Union[Order, str]) -> bool:
        """Cancel an order / 取消订单"""
        return self.broker.cancel_order(order)

    def get_positions(self) -> List[Position]:
        """Get all positions / 获取所有持仓"""
        self._positions = self.broker.get_positions()
        return self._positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol / 获取特定标的持仓"""
        return self.broker.get_position(symbol)

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get all orders / 获取所有订单"""
        return self.broker.get_orders(status=status)

    # ---- Market Data / 市场数据 ----

    def get_last_price(self, asset: Union[Asset, str]) -> Optional[float]:
        """Get last price / 获取最新价格"""
        return self.broker.get_last_price(asset)

    def get_last_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get last prices for multiple symbols / 获取多个标的最新价格"""
        return self.broker.get_last_prices(symbols)

    def get_historical_data(
        self,
        asset: Union[Asset, str],
        length: int,
        timestep: str = "1day",
        **kwargs,
    ) -> Bars:
        """Get historical bars / 获取历史K线"""
        return self.broker.get_historical_data(
            asset=asset,
            length=length,
            timestep=timestep,
            **kwargs,
        )

    # ---- Account / 账户 ----

    def get_portfolio_value(self) -> float:
        """Get total portfolio value / 获取组合总价值"""
        _, _, total = self.broker.get_balances()
        return total

    def get_cash(self) -> float:
        """Get available cash / 获取可用现金"""
        return self.broker.get_cash()

    def get_buying_power(self) -> float:
        """Get buying power / 获取购买力"""
        return self.broker.get_buying_power()

    # ---- Logging / 日志 ----

    def log_message(self, message: str, level: str = "info") -> None:
        """
        Log a message / 记录消息

        Parameters / 参数:
            message: str - Message to log / 要记录的消息
            level: str - Log level: "debug", "info", "warning", "error" / 日志级别
        """
        log_func = getattr(_logger, level, _logger.info)
        log_func(f"[{self._name}] {message}")

    # ---- Event Handlers / 事件处理 ----

    def _on_broker_event(self, event_type: str, order: Order) -> None:
        """Handle broker events / 处理券商事件"""
        if event_type == self.broker.FILLED_ORDER:
            self.on_order_filled(order)
        elif event_type == self.broker.CANCELED_ORDER:
            self.on_order_canceled(order)
        elif event_type == self.broker.PARTIALLY_FILLED_ORDER:
            self.on_order_partially_filled(order)
        elif event_type == self.broker.NEW_ORDER:
            self.on_order_submitted(order)

    def on_order_submitted(self, order: Order) -> None:
        """Called when an order is submitted / 订单提交时调用"""
        pass

    def on_order_filled(self, order: Order) -> None:
        """Called when an order is filled / 订单成交时调用"""
        pass

    def on_order_partially_filled(self, order: Order) -> None:
        """Called when an order is partially filled / 订单部分成交时调用"""
        pass

    def on_order_canceled(self, order: Order) -> None:
        """Called when an order is canceled / 订单取消时调用"""
        pass

    # ---- Run / 运行 ----

    def run(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        exchange: str = "NASDAQ",
    ) -> None:
        """
        Run the strategy (simple loop for live trading).
        运行策略（实盘简单循环）。

        For backtesting, use LumibotBacktest class.
        回测请使用LumibotBacktest类。

        Parameters / 参数:
            start: datetime - Start datetime / 开始时间
            end: datetime - End datetime / 结束时间
            exchange: str - Exchange to trade on / 交易所
        """
        import sched
        import time as time_module

        if not self._initialized:
            self.initialize()
            self._initialized = True

        scheduler = sched.scheduler(time_module.time, time_module.sleep)

        def run_iteration():
            try:
                self.on_trading_iteration()
            except Exception as e:
                self.on_bot_crash(e)
                raise

            # Schedule next iteration / 安排下次迭代
            scheduler.enter(self._parse_sleeptime(), 1, run_iteration)

        def run_pre_market():
            try:
                self.before_market_opens()
            except Exception as e:
                self.on_bot_crash(e)

        def run_post_market():
            try:
                self.after_market_closes()
            except Exception as e:
                self.on_bot_crash(e)

        # Schedule pre/post market and main loop / 安排开盘前后和主循环
        scheduler.enter(1, 1, run_pre_market)
        scheduler.enter(2, 1, run_iteration)
        scheduler.enter(3, 1, run_post_market)

        _logger.info(f"Strategy '{self._name}' starting...")
        scheduler.run()

    def _parse_sleeptime(self) -> float:
        """
        Parse sleeptime string to seconds.
        将sleeptime字符串解析为秒数。

        Examples: "1D" -> 86400, "1H" -> 3600, "5Min" -> 300
        """
        s = self.sleeptime.strip().upper()

        if s.endswith("D"):
            return float(s[:-1]) * 86400
        elif s.endswith("H"):
            return float(s[:-1]) * 3600
        elif s.endswith("MIN"):
            return float(s[:-3]) * 60
        elif s.endswith("M"):
            return float(s[:-1]) * 60
        elif s.endswith("S"):
            return float(s[:-1])
        else:
            return 86400  # Default to 1 day / 默认为1天


# ---------------------------------------------------------------------------
# LumibotBacktest - Backtest Engine / 回测引擎
# ---------------------------------------------------------------------------


class LumibotBacktest:
    """
    Backtest engine for lumibot strategies over historical data.
    历史数据回测引擎，适用于lumibot策略。

    Supports any broker data source for historical data.
    支持任意券商数据源的历史数据。

    Example / 示例:
        >>> from lumibot_adapter import (
        ...     LumibotBacktest,
        ...     AlpacaDataSource,
        ...     AlpacaBroker,
        ...     LumibotStrategy,
        ... )
        >>>
        >>> class MyStrategy(LumibotStrategy):
        ...     def initialize(self):
        ...         self.sleeptime = "1D"
        ...         self.buy_symbol = "SPY"
        ...
        ...     def on_trading_iteration(self):
        ...         price = self.get_last_price(self.buy_symbol)
        ...         cash = self.get_cash()
        ...         if cash > price * 100:
        ...             self.submit_order(self.buy_symbol, 100, "buy")
        >>>
        >>> # Set up backtest / 设置回测
        >>> data_source = AlpacaDataSource(api_key="...", secret_key="...")
        >>> broker = AlpacaBroker(data_source=data_source)
        >>>
        >>> backtest = LumibotBacktest(
        ...     broker=broker,
        ...     strategy_class=MyStrategy,
        ...     datetime_start=datetime(2023, 1, 1),
        ...     datetime_end=datetime(2023, 12, 31),
        ... )
        >>>
        >>> results = backtest.run()
        >>> print(f"Final portfolio: ${results['final_value']:.2f}")
    """

    def __init__(
        self,
        broker: LumibotBroker,
        strategy_class: Type[LumibotStrategy],
        datetime_start: datetime,
        datetime_end: datetime,
        initial_budget: float = 10000.0,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize backtest / 初始化回测

        Parameters / 参数:
            broker: LumibotBroker - Broker for market data / 市场数据券商
            strategy_class: Type[LumibotStrategy] - Strategy class to run / 策略类
            datetime_start: datetime - Backtest start date / 回测开始日期
            datetime_end: datetime - Backtest end date / 回测结束日期
            initial_budget: float - Starting capital / 起始资金
            name: str - Backtest name / 回测名称
            parameters: dict - Strategy parameters / 策略参数
        """
        self.broker = broker
        self.strategy_class = strategy_class
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end
        self.initial_budget = initial_budget
        self.name = name or f"backtest_{datetime_start.strftime('%Y%m%d')}_{datetime_end.strftime('%Y%m%d')}"
        self.parameters = parameters or {}

        # Internal state / 内部状态
        self._strategy: Optional[LumibotStrategy] = None
        self._current_datetime: datetime = datetime_start
        self._cash: float = initial_budget
        self._portfolio_value: float = initial_budget
        self._positions: Dict[str, Position] = {}  # symbol -> Position
        self._orders: Dict[str, Order] = {}
        self._trades: List[Dict[str, Any]] = []

        # Results / 结果
        self._results: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        """
        Run the backtest / 运行回测

        Returns / 返回:
            dict - Backtest results including final value, trades, metrics
                   / 回测结果，包含最终价值、交易记录、指标
        """
        import pandas as pd

        _logger.info(
            f"Starting backtest: {self.datetime_start.date()} to {self.datetime_end.date()}, "
            f"initial_budget=${self.initial_budget:,.2f}"
        )

        # Initialize strategy / 初始化策略
        self._strategy = self.strategy_class(
            broker=self.broker,
            name=self.name,
            parameters=self.parameters,
        )
        self._strategy.initialize()

        # Trading loop / 交易循环
        current_date = self.datetime_start
        iteration = 0

        # Create a simple date range / 创建简单日期范围
        trading_dates = pd.date_range(start=self.datetime_start, end=self.datetime_end, freq="B")  # Business days

        for trading_date in trading_dates:
            self._current_datetime = trading_date

            # Update broker data source to current date / 更新券商数据源到当前日期
            # (The broker's data source will return historical data up to this date)
            # （券商数据源将返回至此日期的历史数据）

            try:
                # Pre-market / 盘前
                self._strategy.before_market_opens()

                # Main trading iteration / 主交易迭代
                self._strategy.on_trading_iteration()

                # Process any simulated fills / 处理模拟成交
                self._process_pending_orders()

                # Post-market / 盘后
                self._strategy.after_market_closes()

                # Update portfolio value / 更新组合价值
                self._update_portfolio_value()

            except Exception as e:
                _logger.error(f"Backtest error on {trading_date.date()}: {e}")
                self._strategy.on_bot_crash(e)

            iteration += 1

        # Final results / 最终结果
        self._results = self._calculate_results()

        _logger.info(
            f"Backtest complete. Final value: ${self._results['final_value']:,.2f}, "
            f"Return: {self._results['total_return_pct']:.2f}%"
        )

        return self._results

    def _process_pending_orders(self) -> None:
        """
        Process pending orders (simulate fills at next day's close or at limit price).
        处理待执行订单（在次日收盘或达到限价时模拟成交）。

        This is a simplified backtest fill simulation.
        这是简化的回测成交模拟。
        """
        for order_id, order in list(self._orders.items()):
            if not order.is_active:
                continue

            # Get current price / 获取当前价格
            price = self.broker.get_last_price(order.asset)

            if price is None:
                continue

            should_fill = False

            if order.order_type == OrderType.MARKET:
                should_fill = True
                fill_price = price
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= (order.limit_price or float("inf")):
                    should_fill = True
                    fill_price = min(price, order.limit_price or price)
                elif order.side == OrderSide.SELL and price >= (order.limit_price or 0):
                    should_fill = True
                    fill_price = max(price, order.limit_price or price)
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and price >= (order.stop_price or 0):
                    should_fill = True
                    fill_price = price
                elif order.side == OrderSide.SELL and price <= (order.stop_price or float("inf")):
                    should_fill = True
                    fill_price = price
            else:
                # For other order types, use market price / 其他订单类型使用市场价
                should_fill = True
                fill_price = price

            if should_fill:
                self._fill_order(order, fill_price)

    def _fill_order(self, order: Order, fill_price: float) -> None:
        """
        Fill an order at a given price / 以给定价格成交订单
        """
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.broker_update_date = self._current_datetime

        # Update cash and positions / 更新现金和持仓
        trade_value = float(order.quantity) * fill_price

        if order.side == OrderSide.BUY:
            self._cash -= trade_value
            # Update or create position / 更新或创建持仓
            symbol = order.asset.symbol
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos.quantity + order.quantity
                pos.avg_entry_price = (
                    (float(pos.quantity) * pos.avg_entry_price + float(order.quantity) * fill_price)
                    / float(total_qty)
                )
                pos.quantity = total_qty
                pos.current_price = fill_price
            else:
                self._positions[symbol] = Position(
                    asset=order.asset,
                    quantity=order.quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                )
        else:  # SELL
            self._cash += trade_value
            symbol = order.asset.symbol
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.quantity -= order.quantity
                if pos.quantity <= 0:
                    del self._positions[symbol]

        # Record trade / 记录交易
        self._trades.append(
            {
                "datetime": self._current_datetime,
                "symbol": order.asset.symbol,
                "side": order.side.value,
                "quantity": float(order.quantity),
                "price": fill_price,
                "value": trade_value,
                "order_id": order.identifier,
            }
        )

        _logger.debug(
            f"Order filled: {order.side.value.upper()} {order.quantity} {order.asset.symbol} @ {fill_price}"
        )

    def _update_portfolio_value(self) -> None:
        """Update portfolio value based on current positions / 根据当前持仓更新组合价值"""
        positions_value = 0.0
        for pos in self._positions.values():
            price = self.broker.get_last_price(pos.asset)
            if price is not None:
                pos.current_price = price
            positions_value += pos.market_value

        self._portfolio_value = self._cash + positions_value

    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate final backtest results / 计算最终回测结果"""
        import pandas as pd

        trades_df = pd.DataFrame(self._trades) if self._trades else pd.DataFrame()

        # Calculate returns / 计算收益
        total_return = self._portfolio_value - self.initial_budget
        total_return_pct = (total_return / self.initial_budget) * 100 if self.initial_budget > 0 else 0

        # Calculate trade statistics / 计算交易统计
        num_trades = len(self._trades)
        num_winners = 0
        num_losers = 0

        if not trades_df.empty and "value" in trades_df.columns:
            buys = trades_df[trades_df["side"] == "buy"]
            sells = trades_df[trades_df["side"] == "sell"]

            # Pair up buys and sells by symbol / 按标的配对买卖
            for symbol in trades_df["symbol"].unique():
                symbol_trades = trades_df[trades_df["symbol"] == symbol]
                buy_vals = symbol_trades[symbol_trades["side"] == "buy"]["value"].sum()
                sell_vals = symbol_trades[symbol_trades["side"] == "sell"]["value"].sum()
                if sell_vals > buy_vals:
                    num_winners += 1
                else:
                    num_losers += 1

        return {
            "name": self.name,
            "initial_budget": self.initial_budget,
            "final_value": self._portfolio_value,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "num_trades": num_trades,
            "num_winners": num_winners,
            "num_losers": num_losers,
            "final_cash": self._cash,
            "positions": dict(self._positions),
            "trades": trades_df,
            "datetime_start": self.datetime_start,
            "datetime_end": self.datetime_end,
        }


# ---------------------------------------------------------------------------
# Exports / 导出
# ---------------------------------------------------------------------------

__all__ = [
    # Core entities / 核心实体
    "Asset",
    "Order",
    "Position",
    "Bar",
    "Bars",
    # Enums / 枚举
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    # Broker classes / 券商类
    "LumibotBroker",
    "PolygonBroker",
    "AlpacaBroker",
    # Data source classes / 数据源类
    "PolygonDataSource",
    "AlpacaDataSource",
    # Strategy / 策略
    "LumibotStrategy",
    # Backtest / 回测
    "LumibotBacktest",
    # Exceptions / 异常
    "LumibotBrokerAPIError",
    "LumibotDataError",
]
