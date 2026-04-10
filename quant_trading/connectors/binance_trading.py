"""
Binance Trading Adapter - Production Trading Interface

Binance现货交易适配器 - 生产级交易接口

Provides unified trading interface for Binance spot trading with:
- Account info (balance, positions)
- Order placement (market, limit, stop-loss)
- Order cancellation and status tracking
- Proper error handling and retry logic

Usage:
    >>> from quant_trading.connectors.binance_trading import BinanceTradingAdapter
    >>> adapter = BinanceTradingAdapter(api_key="...", api_secret="...", testnet=True)
    >>> await adapter.connect()
    >>> balance = await adapter.get_account_balance()
    >>> order = await adapter.place_order("BTCUSDT", "BUY", "LIMIT", 0.001, 50000)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

try:
    import aiohttp
    import json
    import hashlib
    import hmac
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

from quant_trading.connectors.binance_rest import BinanceRESTClient


class TradingMode(Enum):
    """交易模式"""
    LIVE = "live"           # 实盘交易
    PAPER = "paper"         # 模拟交易
    TESTNET = "testnet"     # 测试网


class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """订单状态"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class TimeInForce(Enum):
    """订单有效期"""
    GTC = "GTC"  # Good Till Cancel - 成交为止
    IOC = "IOC"  # Immediate Or Cancel - 即刻成交或取消
    FOK = "FOK"  # Fill Or Kill - 全部成交或取消


@dataclass
class TradingOrder:
    """交易订单"""
    client_order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    order_id: Optional[int] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: int = 0
    updated_at: int = 0
    error_message: Optional[str] = None


@dataclass
class AccountBalance:
    """账户余额"""
    asset: str
    free: float
    locked: float
    total: float = 0.0

    def __post_init__(self):
        self.total = self.free + self.locked


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    margin: float = 0.0


class BinanceAPIException(Exception):
    """Binance API异常"""
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"Binance API Error {code}: {msg}")


class TradingError(Exception):
    """交易异常"""
    pass


class RateLimitError(TradingError):
    """速率限制"""
    pass


class InsufficientBalanceError(TradingError):
    """余额不足"""
    pass


class InvalidOrderError(TradingError):
    """无效订单"""
    pass


class BinanceTradingAdapter:
    """
    Binance现货交易适配器

    提供完整的交易功能，包括：
    - 账户信息查询（余额、持仓）
    - 订单下单（市价、限价、止损）
    - 订单取消和状态查询
    - 错误处理和重试机制

    Args:
        api_key: Binance API密钥
        api_secret: Binance API密钥密码
        testnet: 是否使用测试网
        recv_window: 接收窗口时间（毫秒）
        timeout: 请求超时时间（秒）
    """

    # API URLs
    BASE_URL = "https://api.binance.com"
    TESTNET_URL = "https://testnet.binance.vision"

    # Rate limits
    MAX_REQUESTS_PER_MINUTE = 1200
    MAX_ORDERS_PER_SECOND = 10
    MAX_ORDERS_PER_10_SECONDS = 200

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        recv_window: int = 5000,
        timeout: int = 30,
        rate_limit_callback: Optional[Callable] = None,
    ):
        if not _HAS_AIOHTTP:
            raise ImportError("aiohttp is required: pip install aiohttp")

        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._recv_window = recv_window
        self._timeout = timeout
        self._rate_limit_callback = rate_limit_callback

        self._base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._logger = logging.getLogger("BinanceTradingAdapter")

        # Rate limiting
        self._request_timestamps: List[float] = []
        self._order_timestamps: List[float] = []
        self._last_rate_limit_check = 0

        # Sync client for simple operations
        self._sync_client = BinanceRESTClient(api_key, api_secret, self._base_url, timeout)

        # State
        self._connected = False
        self._listen_key: Optional[str] = None
        self._user_data_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_order_update: Optional[Callable[[TradingOrder], None]] = None
        self._on_balance_update: Optional[Callable[[List[AccountBalance]], None]] = None
        self._on_trade_update: Optional[Callable[[Dict], None]] = None

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected

    @property
    def mode(self) -> TradingMode:
        """当前交易模式"""
        if self._testnet:
            return TradingMode.TESTNET
        return TradingMode.LIVE

    # ====================
    # Connection / 连接
    # ====================

    async def connect(self) -> None:
        """
        连接到Binance API

        Example:
            >>> adapter = BinanceTradingAdapter(api_key="...", api_secret="...")
            >>> await adapter.connect()
        """
        if self._connected:
            return

        self._session = aiohttp.ClientSession()
        self._connected = True
        self._logger.info(f"Connected to Binance (testnet={self._testnet})")

        # Start user data stream if API keys provided
        if self._api_key and self._api_secret:
            await self._start_user_data_stream()

    async def disconnect(self) -> None:
        """断开连接"""
        if not self._connected:
            return

        # Stop user data stream
        if self._listen_key:
            await self._stop_user_data_stream()

        # Cancel user data task
        if self._user_data_task:
            self._user_data_task.cancel()
            try:
                await self._user_data_task
            except asyncio.CancelledError:
                pass

        # Close session
        if self._session:
            await self._session.close()
            self._session = None

        self._connected = False
        self._logger.info("Disconnected from Binance")

    async def _start_user_data_stream(self) -> None:
        """启动用户数据流"""
        try:
            data = await self._request("POST", "/api/v3/userDataStream", signed=False)
            self._listen_key = data.get("listenKey")
            if self._listen_key:
                self._user_data_task = asyncio.create_task(self._listen_user_data())
                self._logger.info("User data stream started")
        except Exception as e:
            self._logger.warning(f"Failed to start user data stream: {e}")

    async def _stop_user_data_stream(self) -> None:
        """停止用户数据流"""
        if self._listen_key:
            try:
                await self._request(
                    "DELETE",
                    "/api/v3/userDataStream",
                    signed=False,
                    params={"listenKey": self._listen_key}
                )
            except Exception as e:
                self._logger.warning(f"Error stopping user data stream: {e}")

    async def _listen_user_data(self) -> None:
        """监听用户数据流"""
        while self._connected and self._listen_key:
            try:
                # Keep alive every 30 minutes
                await asyncio.sleep(1800)
                if self._listen_key:
                    await self._request(
                        "PUT",
                        "/api/v3/userDataStream",
                        signed=False,
                        params={"listenKey": self._listen_key}
                    )
                    self._logger.debug("User data stream keepalive sent")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"User data stream error: {e}")
                await asyncio.sleep(5)

    # ====================
    # Rate Limiting / 速率限制
    # ====================

    async def _check_rate_limit(self, signed: bool = False) -> None:
        """检查速率限制"""
        current_time = time.time()

        # Clean old timestamps (older than 1 minute)
        self._request_timestamps = [t for t in self._request_timestamps if current_time - t < 60]

        if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
            sleep_time = 60 - (current_time - self._request_timestamps[0])
            if sleep_time > 0:
                self._logger.warning(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                if self._rate_limit_callback:
                    self._rate_limit_callback(sleep_time)
                await asyncio.sleep(sleep_time)
                self._request_timestamps = self._request_timestamps[1:]

        # Order-specific rate limiting
        if signed:
            self._order_timestamps = [t for t in self._order_timestamps if current_time - t < 1]
            if len(self._order_timestamps) >= self.MAX_ORDERS_PER_SECOND:
                await asyncio.sleep(1)
                self._order_timestamps = []

            self._order_timestamps = [t for t in self._order_timestamps if current_time - t < 10]
            if len(self._order_timestamps) >= self.MAX_ORDERS_PER_10_SECONDS:
                await asyncio.sleep(10)
                self._order_timestamps = []

        self._request_timestamps.append(current_time)

    # ====================
    # HTTP Requests / HTTP请求
    # ====================

    async def _request(
        self,
        method: str,
        endpoint: str,
        signed: bool = False,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """发送API请求"""
        if not self._session:
            raise TradingError("Not connected. Call connect() first.")

        await self._check_rate_limit(signed)

        url = f"{self._base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self._api_key} if self._api_key else {}

        # Add signature for signed requests
        if signed and self._api_key and self._api_secret:
            params = params or {}
            timestamp = int(time.time() * 1000)
            params["timestamp"] = timestamp
            params["recvWindow"] = self._recv_window

            query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            signature = hmac.new(
                self._api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            params["signature"] = signature

        try:
            async with self._session.request(
                method,
                url,
                params=params,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    raise RateLimitError("Rate limit exceeded")
                elif response.status == 400 or response.status == 403 or response.status == 404:
                    error_data = await response.json()
                    code = error_data.get("code", response.status)
                    msg = error_data.get("msg", "Unknown error")
                    raise BinanceAPIException(code, msg)
                else:
                    text = await response.text()
                    raise TradingError(f"HTTP {response.status}: {text}")

        except aiohttp.ClientError as e:
            raise TradingError(f"Connection error: {e}")

    # ====================
    # Account Info / 账户信息
    # ====================

    async def get_account_info(self) -> Dict[str, Any]:
        """
        获取完整账户信息

        Returns:
            包含所有余额和账户状态的字典

        Example:
            >>> info = await adapter.get_account_info()
            >>> print(info["balances"])
        """
        return await self._request("GET", "/api/v3/account", signed=True)

    async def get_account_balance(self) -> List[AccountBalance]:
        """
        获取账户余额（仅非零余额）

        Returns:
            账户余额列表

        Example:
            >>> balances = await adapter.get_account_balance()
            >>> for b in balances:
            ...     print(f"{b.asset}: {b.free} available, {b.locked} locked")
        """
        account = await self.get_account_info()
        balances = []

        for bal in account.get("balances", []):
            free = float(bal.get("free", 0))
            locked = float(bal.get("locked", 0))
            if free > 0 or locked > 0:
                balances.append(AccountBalance(
                    asset=bal["asset"],
                    free=free,
                    locked=locked,
                ))

        return balances

    async def get_balance(self, asset: str) -> Optional[AccountBalance]:
        """
        获取指定资产余额

        Args:
            asset: 资产符号（如 "USDT", "BTC"）

        Returns:
            账户余额，如果没有则返回None
        """
        balances = await self.get_account_balance()
        for bal in balances:
            if bal.asset == asset.upper():
                return bal
        return AccountBalance(asset=asset.upper(), free=0, locked=0)

    # ====================
    # Order Placement / 下单
    # ====================

    def _generate_client_order_id(self) -> str:
        """生成客户端订单ID"""
        return f"CLT{int(time.time() * 1000)}"

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
    ) -> TradingOrder:
        """
        下单

        Args:
            symbol: 交易对（如 "BTCUSDT"）
            side: 订单方向（BUY/SELL）
            order_type: 订单类型（MARKET/LIMIT/STOP_LOSS等）
            quantity: 数量
            price: 价格（限价单必需）
            stop_price: 止损价格
            time_in_force: 有效期
            client_order_id: 客户端订单ID（可选，自动生成）

        Returns:
            TradingOrder对象

        Example:
            >>> # 市价买单
            >>> order = await adapter.place_order("BTCUSDT", OrderSide.BUY, OrderType.MARKET, 0.001)
            >>>
            >>> # 限价卖单
            >>> order = await adapter.place_order(
            ...     "BTCUSDT", OrderSide.SELL, OrderType.LIMIT, 0.001,
            ...     price=50000, time_in_force=TimeInForce.GTC
            ... )
        """
        # Validation
        symbol = symbol.upper()
        if order_type in [OrderType.LIMIT, OrderType.LIMIT_MAKER] and price is None:
            raise InvalidOrderError(f"Price required for {order_type} orders")

        if order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT,
                          OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT]:
            if stop_price is None:
                raise InvalidOrderError(f"Stop price required for {order_type} orders")

        # Check balance for BUY orders
        if side == OrderSide.BUY:
            quote_asset = symbol.replace("USDT", "").replace("BTC", "").replace("ETH", "")
            if quote_asset != symbol:  # Has quote asset
                balance = await self.get_balance(quote_asset if quote_asset else "USDT")
                required = quantity * (price or 0)
                if balance and balance.free < required * 1.001:  # 0.1% buffer
                    raise InsufficientBalanceError(
                        f"Insufficient {quote_asset or 'USDT'} balance: "
                        f"required {required:.2f}, available {balance.free if balance else 0:.2f}"
                    )

        # Build params
        client_id = client_order_id or self._generate_client_order_id()
        params = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": str(quantity),
            "newClientOrderId": client_id,
        }

        if price is not None:
            params["price"] = str(price)
            params["timeInForce"] = time_in_force.value

        if stop_price is not None:
            params["stopPrice"] = str(stop_price)

        # Place order
        try:
            response = await self._request("POST", "/api/v3/order", signed=True, params=params)

            order = TradingOrder(
                client_order_id=response.get("clientOrderId", client_id),
                symbol=response["symbol"],
                side=OrderSide(side.value),  # Keep original
                type=OrderType(response["type"]),
                quantity=float(response["origQty"]),
                price=float(response["price"]) if response.get("price") else None,
                stop_price=float(response["stopPrice"]) if response.get("stopPrice") else None,
                order_id=response.get("orderId"),
                status=OrderStatus(response["status"]),
                filled_quantity=float(response.get("executedQty", 0)),
                avg_fill_price=float(response.get("avgPrice", 0)),
                created_at=response.get("transactTime", 0),
                updated_at=response.get("updateTime", response.get("transactTime", 0)),
            )

            self._logger.info(f"Order placed: {order.client_order_id} {order.status.value}")
            return order

        except BinanceAPIException as e:
            self._logger.error(f"Order rejected: {e}")
            raise InvalidOrderError(f"Order rejected: {e.msg}")

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> TradingOrder:
        """
        市价单

        Args:
            symbol: 交易对
            side: 方向
            quantity: 数量

        Returns:
            TradingOrder对象
        """
        return await self.place_order(symbol, side, OrderType.MARKET, quantity)

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> TradingOrder:
        """
        限价单

        Args:
            symbol: 交易对
            side: 方向
            quantity: 数量
            price: 价格
            time_in_force: 有效期

        Returns:
            TradingOrder对象
        """
        return await self.place_order(symbol, side, OrderType.LIMIT, quantity, price=price, time_in_force=time_in_force)

    async def place_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
    ) -> TradingOrder:
        """
        止损单

        Args:
            symbol: 交易对
            side: 方向
            quantity: 数量
            stop_price: 触发价格

        Returns:
            TradingOrder对象
        """
        return await self.place_order(symbol, side, OrderType.STOP_LOSS, quantity, stop_price=stop_price)

    # ====================
    # Order Management / 订单管理
    # ====================

    async def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Optional[TradingOrder]:
        """
        查询订单状态

        Args:
            symbol: 交易对
            order_id: 交易所订单ID
            client_order_id: 客户端订单ID

        Returns:
            TradingOrder对象，如果未找到则返回None
        """
        if order_id is None and client_order_id is None:
            raise InvalidOrderError("order_id or client_order_id required")

        params = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id

        try:
            response = await self._request("GET", "/api/v3/order", signed=True, params=params)

            return TradingOrder(
                client_order_id=response.get("clientOrderId", ""),
                symbol=response["symbol"],
                side=OrderSide(response["side"]),
                type=OrderType(response["type"]),
                quantity=float(response["origQty"]),
                price=float(response["price"]) if response.get("price") else None,
                stop_price=float(response["stopPrice"]) if response.get("stopPrice") else None,
                order_id=response.get("orderId"),
                status=OrderStatus(response["status"]),
                filled_quantity=float(response.get("executedQty", 0)),
                avg_fill_price=float(response.get("avgPrice", 0)),
                created_at=response.get("time", 0),
                updated_at=response.get("updateTime", 0),
            )

        except BinanceAPIException as e:
            if e.code == -2013:  # Order not found
                return None
            raise

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[TradingOrder]:
        """
        获取所有未成交订单

        Args:
            symbol: 可选的交易对过滤

        Returns:
            未成交订单列表
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()

        try:
            response = await self._request("GET", "/api/v3/openOrders", signed=True, params=params)

            orders = []
            for o in response:
                orders.append(TradingOrder(
                    client_order_id=o.get("clientOrderId", ""),
                    symbol=o["symbol"],
                    side=OrderSide(o["side"]),
                    type=OrderType(o["type"]),
                    quantity=float(o["origQty"]),
                    price=float(o["price"]) if o.get("price") else None,
                    stop_price=float(o["stopPrice"]) if o.get("stopPrice") else None,
                    order_id=o.get("orderId"),
                    status=OrderStatus(o["status"]),
                    filled_quantity=float(o.get("executedQty", 0)),
                    avg_fill_price=float(o.get("avgPrice", 0)),
                    created_at=o.get("time", 0),
                    updated_at=o.get("updateTime", 0),
                ))

            return orders

        except BinanceAPIException as e:
            if e.code == -2013:
                return []
            raise

    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> bool:
        """
        取消订单

        Args:
            symbol: 交易对
            order_id: 交易所订单ID
            client_order_id: 客户端订单ID

        Returns:
            成功返回True
        """
        params = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id

        try:
            await self._request("DELETE", "/api/v3/order", signed=True, params=params)
            self._logger.info(f"Order cancelled: {client_order_id or order_id}")
            return True

        except BinanceAPIException as e:
            if e.code == -2013:  # Order already cancelled or not found
                return True
            self._logger.error(f"Cancel order failed: {e}")
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        取消所有未成交订单

        Args:
            symbol: 可选的交易对过滤

        Returns:
            取消的订单数量
        """
        open_orders = await self.get_open_orders(symbol)
        cancelled = 0

        for order in open_orders:
            if await self.cancel_order(order.symbol, order_id=order.order_id):
                cancelled += 1

        return cancelled

    # ====================
    # Trade / 交易历史
    # ====================

    async def get_my_trades(
        self,
        symbol: str,
        limit: int = 500,
        from_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取成交历史

        Args:
            symbol: 交易对
            limit: 返回数量（默认500）
            from_id: 从指定ID开始

        Returns:
            成交列表
        """
        params = {"symbol": symbol.upper(), "limit": limit}
        if from_id:
            params["fromId"] = from_id

        return await self._request("GET", "/api/v3/myTrades", signed=True, params=params)

    # ====================
    # Market Data / 市场数据（代理到REST client）
    # ====================

    async def get_ticker_price(self, symbol: str) -> float:
        """获取最新价格"""
        data = await self._request(
            "GET",
            "/api/v3/ticker/price",
            params={"symbol": symbol.upper()}
        )
        return float(data["price"])

    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, List]:
        """获取订单簿"""
        return await self._request(
            "GET",
            "/api/v3/depth",
            params={"symbol": symbol.upper(), "limit": limit}
        )

    # ====================
    # Sync Methods / 同步方法
    # ====================

    def ping(self) -> bool:
        """连通性测试（同步）"""
        try:
            self._sync_client.ping()
            return True
        except Exception:
            return False

    def get_server_time(self) -> int:
        """获取服务器时间（同步）"""
        account = self._sync_client.get_account()
        return int(time.time() * 1000)  # Fallback to local time


# ====================
# Convenience Factory / 便捷工厂函数
# ====================

def create_live_trading_adapter(api_key: str, api_secret: str) -> BinanceTradingAdapter:
    """创建实盘交易适配器"""
    return BinanceTradingAdapter(api_key=api_key, api_secret=api_secret, testnet=False)


def create_paper_trading_adapter(api_key: str = "", api_secret: str = "") -> BinanceTradingAdapter:
    """创建模拟交易适配器"""
    return BinanceTradingAdapter(api_key=api_key, api_secret=api_secret, testnet=True)


def create_testnet_adapter(api_key: str = "", api_secret: str = "") -> BinanceTradingAdapter:
    """创建测试网适配器（与模拟交易相同）"""
    return BinanceTradingAdapter(api_key=api_key, api_secret=api_secret, testnet=True)


__all__ = [
    "BinanceTradingAdapter",
    "TradingMode",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    "TradingOrder",
    "AccountBalance",
    "Position",
    "TradingError",
    "BinanceAPIException",
    "RateLimitError",
    "InsufficientBalanceError",
    "InvalidOrderError",
    "create_live_trading_adapter",
    "create_paper_trading_adapter",
    "create_testnet_adapter",
]
