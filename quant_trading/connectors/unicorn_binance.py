"""
UnicornBinance - Binance API Suite (Inspired by unicorn-binance-suite)
======================================================================

UnicornBinance 是一套轻量级 Binance API 封装，参考 UNICORN Binance Suite 设计理念，
提供纯 Python 标准库实现的 REST API 和 WebSocket 客户端。

核心组件 (Core Components):
    - UnicornBinanceREST  : REST API 封装 (现货/杠杆)
    - UnicornBinanceWebSocket : WebSocket 实时数据客户端

特性 (Features):
    - urllib.request 纯标准库实现 REST 调用，无重依赖
    - HMAC-SHA256 签名认证
    - K线/深度/成交/账户/订单管理
    - WebSocket 实时行情流 (K线/深度/成交/Ticker)
    - 异步与同步双接口支持

使用方法 (Usage):
    >>> from quant_trading.connectors.unicorn_binance import UnicornBinanceREST, UnicornBinanceWebSocket
    >>>
    >>> # REST 示例
    >>> client = UnicornBinanceREST(api_key="your_key", api_secret="your_secret")
    >>> klines = client.fetch_klines("BTCUSDT", "1m", limit=100)
    >>> order_book = client.fetch_order_book("BTCUSDT", limit=20)
    >>>
    >>> # WebSocket 示例
    >>> def on_kline(data): print(data)
    >>> ws = UnicornBinanceWebSocket()
    >>> ws.stream_klines("BTCUSDT", "1m", on_kline)
    >>> ws.start()

This module is inspired by the unicorn-binance-suite project:
https://github.com/oliver-zehentleitner/unicorn-binance-suite
All licensing and rights belong to the original authors.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import logging
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen, Request

__all__ = [
    "UnicornBinanceREST",
    "UnicornBinanceWebSocket",
    "UnicornBinanceAPIError",
    "UnicornBinanceConnectionError",
    "UnicornBinanceRateLimitError",
]

logger = logging.getLogger("UnicornBinance")

# ===================
# Constants / 常量
# ===================

BINANCE_REST_URL = "https://api.binance.com"
BINANCE_WS_URL = "https://stream.binance.com:9443/ws"
BINANCE_WS_COMBINED_URL = "https://stream.binance.com:9443/stream?streams="

DEFAULT_RECV_WINDOW = 5000
DEFAULT_TIMEOUT = 30


# ===================
# Exceptions / 异常类
# ===================


class UnicornBinanceAPIError(Exception):
    """
    Binance API 错误 / Binance API Error.

    Attributes:
        code: 错误码 (Error code)
        msg: 错误信息 (Error message)
    """

    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"[{code}] {msg}")

    def __repr__(self) -> str:
        return f"UnicornBinanceAPIError(code={self.code}, msg='{self.msg}')"


class UnicornBinanceConnectionError(Exception):
    """
    Binance 连接错误 / Binance Connection Error.
    """

    pass


class UnicornBinanceRateLimitError(UnicornBinanceAPIError):
    """
    Binance 速率限制错误 / Binance Rate Limit Error.

    Raised when Binance API rate limit is exceeded (HTTP 429).
    """

    def __init__(self, msg: str = "Rate limit exceeded"):
        super().__init__(-1003, msg)


# ===================
# UnicornBinanceREST
# ===================


class UnicornBinanceREST:
    """
    Binance REST API 客户端 (同步版) / Binance REST API Client (Sync).

    基于 urllib.request 的纯标准库实现，支持 Binance 现货/杠杆 API。
    无需安装额外依赖，适合嵌入式量化交易系统。

    支持功能 (Supported Features):
        - 市场数据: K线、深度、成交、Ticker、价格
        - 账户信息: 余额、账户详情
        - 订单管理: 下单、撤单、查询订单

    参数 (Parameters):
        api_key: Binance API Key (可选，公开数据接口不需要)
        api_secret: Binance API Secret (可选，签名接口需要)
        base_url: API 基础 URL，默认 https://api.binance.com
        timeout: 请求超时时间(秒)，默认 30
        recv_window: 接收窗口时间(毫秒)，默认 5000

    示例 (Example):
        >>> client = UnicornBinanceREST()
        >>> client.fetch_klines("BTCUSDT", "1m", limit=100)
        >>> client.fetch_order_book("BTCUSDT", limit=20)
        >>>
        >>> # 认证接口
        >>> auth_client = UnicornBinanceREST(api_key="key", api_secret="secret")
        >>> auth_client.fetch_account()
        >>> auth_client.place_order("BTCUSDT", "BUY", "LIMIT", quantity=0.001, price=50000)
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = BINANCE_REST_URL,
        timeout: int = DEFAULT_TIMEOUT,
        recv_window: int = DEFAULT_RECV_WINDOW,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._recv_window = recv_window
        self._logger = logger.getChild("REST")

    # ===================
    # Internal Methods / 内部方法
    # ===================

    def _sign(self, params: Dict[str, Any]) -> str:
        """
        使用 HMAC-SHA256 对参数进行签名 / Sign parameters with HMAC-SHA256.

        Args:
            params: 待签名的参数字典

        Returns:
            签名字符串 (signature string)
        """
        timestamp = int(time.time() * 1000)
        params["timestamp"] = timestamp
        params["recvWindow"] = self._recv_window
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        signed: bool = False,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        发起 HTTP 请求到 Binance API / Make HTTP request to Binance API.

        Args:
            method: HTTP 方法 (GET/POST/DELETE/PUT)
            endpoint: API 端点路径
            signed: 是否需要签名认证
            params: 查询参数
            headers: 额外请求头

        Returns:
            响应 JSON 数据

        Raises:
            UnicornBinanceAPIError: API 返回错误
            UnicornBinanceConnectionError: 连接错误
            UnicornBinanceRateLimitError: 速率限制
        """
        url = f"{self._base_url}{endpoint}"
        params = params or {}
        headers = headers or {}

        # 添加 API Key Header
        if self._api_key:
            headers["X-MBX-APIKEY"] = self._api_key

        # 签名
        if signed:
            if not self._api_key or not self._api_secret:
                raise UnicornBinanceAPIError(
                    -1, "API key and secret required for signed requests"
                )
            params["signature"] = self._sign(params)

        # 构建完整 URL
        if params:
            url_with_params = f"{url}?{urlencode(params)}"
        else:
            url_with_params = url

        try:
            req = Request(url_with_params, method=method, headers=headers)
            with urlopen(req, timeout=self._timeout) as resp:
                data = resp.read()
                # Handle gzip compression
                if resp.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return json.loads(data.decode("utf-8"))

        except UnicornBinanceAPIError:
            raise
        except Exception as e:
            raise UnicornBinanceConnectionError(f"Request failed: {e}") from e

    # ===================
    # Market Data / 市场数据
    # ===================

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[List[Any]]:
        """
        获取 K线数据 / Fetch candlestick/kline data.

        获取指定交易对的K线数据，支持时间范围过滤。

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"
            interval: K线周期: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
            limit: 返回数量 (1-1000)，默认 500
            start_time: 起始时间戳(毫秒)
            end_time: 结束时间戳(毫秒)

        返回 (Returns):
            List[List]: K线数组列表，每条K线格式:
                [
                    1499040000000,      # 开盘时间 (Open time)
                    "0.00100000",       # 开盘价 (Open)
                    "0.00250000",       # 最高价 (High)
                    "0.00100000",       # 最低价 (Low)
                    "0.00200000",       # 收盘价 (Close)
                    "1480.00000000",    # 成交量 (Volume)
                    1499644799999,      # 收盘时间 (Close time)
                    "3080.00000000",    # 成交额 (Quote asset volume)
                    0,                  # 成交笔数 (Trades)
                    "1756.87472397",    # 主动买入成交量 (Taker buy volume)
                    "28.46694368",      # 主动买入成交额 (Taker buy quote volume)
                    "0"                 # Ignore (是否跳过)
                ]

        示例 (Example):
            >>> client.fetch_klines("BTCUSDT", "1h", limit=100)
            >>> # 获取特定时间段
            >>> client.fetch_klines("ETHUSDT", "4h", start_time=1609459200000, end_time=1612137600000)
        """
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        return self._request("GET", "/api/v3/klines", params=params)

    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 100,
    ) -> Dict[str, List[List[str]]]:
        """
        获取订单簿深度 / Fetch order book depth.

        获取指定交易对的当前挂单深度。

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"
            limit: 深度数量，可选: 5, 10, 20, 50, 100, 500, 1000, 5000

        返回 (Returns):
            Dict containing:
                lastUpdateId: 最后更新ID
                bids: 买单深度 [[price, qty], ...]
                asks: 卖单深度 [[price, qty], ...]

        示例 (Example):
            >>> client.fetch_order_book("BTCUSDT", limit=20)
            {'lastUpdateId': 123456789, 'bids': [['50000.00', '1.5']], 'asks': [['50001.00', '2.0']]}
        """
        params = {"symbol": symbol.upper(), "limit": limit}
        return self._request("GET", "/api/v3/depth", params=params)

    def fetch_ticker(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        获取 24hr Ticker 统计 / Fetch 24hr ticker statistics.

        获取指定交易对的24小时价格统计信息。

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"

        返回 (Returns):
            Dict with ticker data:
                {
                    "symbol": "BTCUSDT",
                    "priceChange": "0.00123000",
                    "priceChangePercent": "0.123",
                    "lastPrice": "50000.00000000",
                    "highPrice": "51000.00000000",
                    "lowPrice": "49000.00000000",
                    "volume": "12345.67890000",
                    "quoteVolume": "617283456.78900000",
                    ...
                }

        示例 (Example):
            >>> client.fetch_ticker("BTCUSDT")
        """
        params = {"symbol": symbol.upper()}
        return self._request("GET", "/api/v3/ticker/24hr", params=params)

    def fetch_trades(
        self,
        symbol: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        获取近期成交记录 / Fetch recent trades.

        获取指定交易对的近期成交历史。

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"
            limit: 返回数量 (1-1000)，默认 500

        返回 (Returns):
            List of trade objects:
                [
                    {
                        "id": 12345,
                        "price": "50000.00000000",
                        "qty": "0.00100000",
                        "time": 1234567890123,
                        "isBuyerMaker": true,
                        "isBestMatch": true
                    },
                    ...
                ]

        示例 (Example):
            >>> client.fetch_trades("BTCUSDT", limit=100)
        """
        params = {"symbol": symbol.upper(), "limit": limit}
        return self._request("GET", "/api/v3/trades", params=params)

    def fetch_ticker_price(
        self,
        symbol: str,
    ) -> Dict[str, str]:
        """
        获取最新价格 / Fetch latest price.

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"

        返回 (Returns):
            {"symbol": "BTCUSDT", "price": "50000.00000000"}

        示例 (Example):
            >>> client.fetch_ticker_price("BTCUSDT")
        """
        params = {"symbol": symbol.upper()}
        return self._request("GET", "/api/v3/ticker/price", params=params)

    def fetch_book_ticker(
        self,
        symbol: str,
    ) -> Dict[str, str]:
        """
        获取深度窗口最优挂单 / Fetch best order book ticker.

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"

        返回 (Returns):
            {
                "symbol": "BTCUSDT",
                "bidPrice": "50000.00",
                "bidQty": "1.50000000",
                "askPrice": "50001.00",
                "askQty": "2.00000000"
            }

        示例 (Example):
            >>> client.fetch_book_ticker("BTCUSDT")
        """
        params = {"symbol": symbol.upper()}
        return self._request("GET", "/api/v3/ticker/bookTicker", params=params)

    def fetch_exchange_info(self) -> Dict[str, Any]:
        """
        获取交易所交易规则和交易对信息 / Fetch exchange trading rules and symbol info.

        返回 (Returns):
            包含交易对信息的字典:
                {
                    "timezone": "UTC",
                    "serverTime": 1234567890123,
                    "symbols": [
                        {
                            "symbol": "BTCUSDT",
                            "status": "TRADING",
                            "baseAsset": "BTC",
                            "quoteAsset": "USDT",
                            ...
                        },
                        ...
                    ],
                    ...
                }

        示例 (Example):
            >>> info = client.fetch_exchange_info()
            >>> symbols = [s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING']
        """
        return self._request("GET", "/api/v3/exchangeInfo")

    def ping(self) -> Dict[str, Any]:
        """
        测试服务器连通性 / Ping Binance server.

        返回 (Returns):
            {"msg": "success", "code": 0}

        示例 (Example):
            >>> client.ping()
        """
        return self._request("GET", "/api/v3/ping")

    # ===================
    # Account / 账户
    # ===================

    def fetch_account(self) -> Dict[str, Any]:
        """
        获取账户信息 / Fetch account information.

        需要 API Key 和 Secret 签名认证。

        返回 (Returns):
            {
                "makerCommission": 10,
                "takerCommission": 10,
                "buyerCommission": 0,
                "sellerCommission": 0,
                "canTrade": true,
                "canWithdraw": true,
                "canDeposit": true,
                "balances": [
                    {"asset": "BTC", "free": "1.00000000", "locked": "0.00000000"},
                    {"asset": "USDT", "free": "1000.00000000", "locked": "0.00000000"},
                    ...
                ],
                ...
            }

        示例 (Example):
            >>> client.fetch_account()
        """
        return self._request("GET", "/api/v3/account", signed=True)

    def fetch_balance(self) -> Dict[str, Any]:
        """
        获取账户余额 / Fetch account balance.

        获取所有余额大于0的资产信息。

        返回 (Returns):
            {"balances": [{"asset": "BTC", "free": "1.0", "locked": "0.0"}, ...]}

        示例 (Example):
            >>> client.fetch_balance()
        """
        account = self.fetch_account()
        return {"balances": account.get("balances", [])}

    def fetch_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取当前挂单 / Fetch open orders.

        参数 (Parameters):
            symbol: 可选，指定交易对过滤，例如 "BTCUSDT"

        返回 (Returns):
            当前所有挂单列表

        示例 (Example):
            >>> client.fetch_open_orders()
            >>> client.fetch_open_orders("BTCUSDT")
        """
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/api/v3/openOrders", signed=True, params=params)

    def fetch_order(
        self,
        symbol: str,
        order_id: int,
    ) -> Dict[str, Any]:
        """
        查询订单详情 / Fetch order details.

        参数 (Parameters):
            symbol: 交易对符号
            order_id: 订单ID

        返回 (Returns):
            订单详细信息字典

        示例 (Example):
            >>> client.fetch_order("BTCUSDT", 123456789)
        """
        params = {"symbol": symbol.upper(), "orderId": order_id}
        return self._request("GET", "/api/v3/order", signed=True, params=params)

    # ===================
    # Order Management / 订单管理
    # ===================

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        下单 / Place an order.

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"
            side: 订单方向 "BUY" 或 "SELL"
            order_type: 订单类型: LIMIT, MARKET, LIMIT_MAKER, STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT
            quantity: 订单数量
            price: 订单价格 (LIMIT类订单必需)
            time_in_force: 有效期限: GTC(成交为止), IOC(立即取消或部分成交), FOK(全部成交或取消)
            **kwargs: 额外参数:
                - client_order_id: 自定义订单ID
                - stop_price: 止损/止盈价格
                - iceberg_qty: 冰山订单数量

        返回 (Returns):
            订单响应:
                {
                    "symbol": "BTCUSDT",
                    "orderId": 123456789,
                    "clientOrderId": "...",
                    "price": "50000.00000000",
                    "origQty": "0.00100000",
                    "status": "NEW",
                    "type": "LIMIT",
                    "side": "BUY",
                    "transactTime": 1234567890123,
                    ...
                }

        示例 (Example):
            >>> # LIMIT 买单
            >>> client.place_order("BTCUSDT", "BUY", "LIMIT", quantity=0.001, price=50000)
            >>> # MARKET 卖单
            >>> client.place_order("BTCUSDT", "SELL", "MARKET", quantity=0.001)
        """
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }

        if price is not None:
            params["price"] = str(price)
            if order_type.upper() in ("LIMIT", "LIMIT_MAKER", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"):
                params["timeInForce"] = time_in_force

        for key, value in kwargs.items():
            params[key] = value

        return self._request("POST", "/api/v3/order", signed=True, params=params)

    def cancel_order(
        self,
        symbol: str,
        order_id: int,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        撤销订单 / Cancel an order.

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"
            order_id: 要撤销的订单ID
            client_order_id: 可选，自定义订单ID

        返回 (Returns):
            撤销确认响应

        示例 (Example):
            >>> client.cancel_order("BTCUSDT", 123456789)
        """
        params: Dict[str, Any] = {"symbol": symbol.upper(), "orderId": order_id}
        if client_order_id:
            params["clientOrderId"] = client_order_id
        return self._request("DELETE", "/api/v3/order", signed=True, params=params)

    def cancel_all_orders(
        self,
        symbol: str,
    ) -> List[Dict[str, Any]]:
        """
        撤销指定交易对所有挂单 / Cancel all open orders for a symbol.

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"

        返回 (Returns):
            撤销结果列表

        示例 (Example):
            >>> client.cancel_all_orders("BTCUSDT")
        """
        params = {"symbol": symbol.upper()}
        return self._request("DELETE", "/api/v3/openOrders", signed=True, params=params)

    # ===================
    # User Data Stream / 用户数据流
    # ===================

    def start_user_stream(self) -> str:
        """
        开启用户数据流 / Start user data stream.

        返回 (Returns):
            listen_key: 用于WebSocket订阅用户数据的密钥

        示例 (Example):
            >>> listen_key = client.start_user_stream()
        """
        data = self._request("POST", "/api/v3/userDataStream", signed=False)
        return data.get("listenKey")

    def keepalive_user_stream(self, listen_key: str) -> Dict[str, Any]:
        """
        延长用户数据流有效期 / Keep user data stream alive.

        用户数据流每60分钟需要调用一次延长有效期。

        参数 (Parameters):
            listen_key: start_user_stream 返回的 listen_key

        返回 (Returns):
            {"listenKey": "...", "expires": 3600000}

        示例 (Example):
            >>> client.keepalive_user_stream(listen_key)
        """
        params = {"listenKey": listen_key}
        return self._request("PUT", "/api/v3/userDataStream", params=params)

    def close_user_stream(self, listen_key: str) -> Dict[str, Any]:
        """
        关闭用户数据流 / Close user data stream.

        参数 (Parameters):
            listen_key: 要关闭的 listen_key

        返回 (Returns):
            {}

        示例 (Example):
            >>> client.close_user_stream(listen_key)
        """
        params = {"listenKey": listen_key}
        return self._request("DELETE", "/api/v3/userDataStream", params=params)


# ===================
# UnicornBinanceWebSocket
# ===================


class UnicornBinanceWebSocket:
    """
    Binance WebSocket 实时数据客户端 (同步版) / Binance WebSocket Real-time Client (Sync).

    基于 websocket-client 库的轻量级实现，支持 Binance 公开数据流和用户数据流。

    支持的数据流 (Supported Streams):
        - K线流: <symbol>@kline_<interval>
        - 深度流: <symbol>@depth<level>
        - 成交流: <symbol>@trade
        - Ticker流: <symbol>@ticker
        - 全市场Ticker: !miniTicker@arr
        - 用户数据流: <listen_key>

    参数 (Parameters):
        streams: 初始化时订阅的流列表
        callback: 默认消息回调函数
        on_error: 错误回调函数

    示例 (Example):
        >>> def on_kline(data):
        ...     print(f"Kline: {data['k']}")
        >>>
        >>> ws = UnicornBinanceWebSocket()
        >>> ws.stream_klines("BTCUSDT", "1m", on_kline)
        >>> ws.start()
        >>> ws.stop()
    """

    def __init__(
        self,
        streams: Optional[List[str]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self._streams = streams or []
        self._callback = callback
        self._on_error = on_error
        self._ws: Any = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._subscriptions: Dict[str, Callable] = {}
        self._logger = logger.getChild("WebSocket")
        self._ws_lock = threading.Lock()

        # Lazy import websocket
        self._ws_module: Any = None
        self._ws_class: Any = None

    def _ensure_websocket(self) -> None:
        """Lazy load websocket module."""
        if self._ws_module is not None:
            return
        try:
            import websocket

            self._ws_module = websocket
            self._ws_class = websocket.WebSocketApp
        except ImportError:
            raise ImportError(
                "websocket-client is required: pip install websocket-client\n"
                "Or use: from quant_trading.connectors.binance_ws import BinanceWebSocketClient"
            )

    # ===================
    # Subscription Methods / 订阅方法
    # ===================

    def subscribe(
        self,
        streams: List[str],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        订阅 WebSocket 数据流 / Subscribe to WebSocket streams.

        参数 (Parameters):
            streams: 流名称列表，例如 ["btcusdt@kline_1m", "btcusdt@depth20"]
            callback: 消息回调函数

        示例 (Example):
            >>> def on_data(data): print(data)
            >>> ws.subscribe(["btcusdt@kline_1m", "btcusdt@trade"], on_data)
        """
        for stream in streams:
            self._subscriptions[stream] = callback or self._callback
            self._send({"method": "SUBSCRIBE", "params": [stream], "id": self._next_id()})

    def unsubscribe(self, streams: List[str]) -> None:
        """
        取消订阅 WebSocket 数据流 / Unsubscribe from WebSocket streams.

        参数 (Parameters):
            streams: 要取消订阅的流名称列表

        示例 (Example):
            >>> ws.unsubscribe(["btcusdt@kline_1m"])
        """
        for stream in streams:
            if stream in self._subscriptions:
                del self._subscriptions[stream]
            self._send({"method": "UNSUBSCRIBE", "params": [stream], "id": self._next_id()})

    def _next_id(self) -> int:
        """生成唯一消息ID / Generate unique message ID."""
        return int(time.time() * 1000)

    def _send(self, msg: Dict[str, Any]) -> None:
        """发送 WebSocket 消息 / Send WebSocket message."""
        if self._ws and self._running:
            try:
                self._ws.send(json.dumps(msg))
            except Exception as e:
                self._logger.error(f"Failed to send message: {e}")

    # ===================
    # Stream Helpers / 便捷流订阅方法
    # ===================

    def stream_klines(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """
        订阅 K线数据流 / Subscribe to kline/candlestick stream.

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"
            interval: K线周期，例如 "1m", "5m", "1h", "4h", "1d"
            callback: K线数据回调函数

        返回 (Returns):
            流名称字符串

        示例 (Example):
            >>> def on_kline(k):
            ...     kline = k['k']
            ...     print(f"OHLC: {kline['o']}, {kline['h']}, {kline['l']}, {kline['c']}")
            >>> ws.stream_klines("BTCUSDT", "1m", on_kline)
        """
        stream = f"{symbol.lower()}@kline_{interval}"
        self._subscriptions[stream] = callback
        if self._running:
            self._send({"method": "SUBSCRIBE", "params": [stream], "id": self._next_id()})
        return stream

    def stream_depth(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        level: int = 20,
    ) -> str:
        """
        订阅订单簿深度流 / Subscribe to depth order book stream.

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"
            callback: 深度数据回调函数
            level: 深度级别，可选 5, 10, 20, 100, 500, 1000

        返回 (Returns):
            流名称字符串

        示例 (Example):
            >>> def on_depth(d):
            ...     print(f"Bids: {d['bids'][:5]}, Asks: {d['asks'][:5]}")
            >>> ws.stream_depth("BTCUSDT", on_depth, level=20)
        """
        stream = f"{symbol.lower()}@depth{level}"
        self._subscriptions[stream] = callback
        if self._running:
            self._send({"method": "SUBSCRIBE", "params": [stream], "id": self._next_id()})
        return stream

    def stream_trades(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """
        订阅实时成交流 / Subscribe to trade stream.

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"
            callback: 成交数据回调函数

        返回 (Returns):
            流名称字符串

        示例 (Example):
            >>> def on_trade(t):
            ...     print(f"Trade: {t['p']} x {t['q']} @ {t['T']}")
            >>> ws.stream_trades("BTCUSDT", on_trade)
        """
        stream = f"{symbol.lower()}@trade"
        self._subscriptions[stream] = callback
        if self._running:
            self._send({"method": "SUBSCRIBE", "params": [stream], "id": self._next_id()})
        return stream

    def stream_ticker(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """
        订阅 24hr Ticker 流 / Subscribe to 24hr ticker stream.

        参数 (Parameters):
            symbol: 交易对符号，例如 "BTCUSDT"
            callback: Ticker数据回调函数

        返回 (Returns):
            流名称字符串

        示例 (Example):
            >>> def on_ticker(t):
            ...     print(f"{t['s']}: Last={t['c']}, High={t['h']}, Low={t['l']}")
            >>> ws.stream_ticker("BTCUSDT", on_ticker)
        """
        stream = f"{symbol.lower()}@ticker"
        self._subscriptions[stream] = callback
        if self._running:
            self._send({"method": "SUBSCRIBE", "params": [stream], "id": self._next_id()})
        return stream

    def stream_all_mini_ticker(
        self,
        callback: Callable[[List[Dict[str, Any]]], None],
    ) -> str:
        """
        订阅所有交易对 Mini Ticker 流 / Subscribe to all mini ticker stream.

        参数 (Parameters):
            callback: 所有Ticker数据回调函数

        返回 (Returns):
            流名称字符串 "!miniTicker@arr"

        示例 (Example):
            >>> def on_tickers(tickers):
            ...     for t in tickers:
            ...         print(f"{t['s']}: {t['c']}")
            >>> ws.stream_all_mini_ticker(on_tickers)
        """
        stream = "!miniTicker@arr"
        self._subscriptions[stream] = callback
        if self._running:
            self._send({"method": "SUBSCRIBE", "params": [stream], "id": self._next_id()})
        return stream

    def stream_user_data(
        self,
        listen_key: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> str:
        """
        订阅用户数据流 (需要有效的 listen_key) / Subscribe to user data stream.

        通过 UnicornBinanceREST.start_user_stream() 获取 listen_key。

        参数 (Parameters):
            listen_key: 用户数据流密钥
            callback: 用户数据回调函数 (订单/余额更新等)

        返回 (Returns):
            流名称字符串

        示例 (Example):
            >>> rest = UnicornBinanceREST(api_key="key", api_secret="secret")
            >>> listen_key = rest.start_user_stream()
            >>> def on_user_data(d): print(d)
            >>> ws.stream_user_data(listen_key, on_user_data)
        """
        self._subscriptions[listen_key] = callback
        if self._running:
            self._send({"method": "SUBSCRIBE", "params": [listen_key], "id": self._next_id()})
        return listen_key

    # ===================
    # Connection Management / 连接管理
    # ===================

    def start(self) -> None:
        """
        启动 WebSocket 连接 (后台线程) / Start WebSocket connection (background thread).

        示例 (Example):
            >>> ws = UnicornBinanceWebSocket()
            >>> ws.stream_klines("BTCUSDT", "1m", callback)
            >>> ws.start()
            >>> # ... later ...
            >>> ws.stop()
        """
        if self._running:
            self._logger.warning("WebSocket already running")
            return

        self._running = True
        self._ensure_websocket()

        # Build combined URL if streams provided at init
        if self._streams:
            stream_str = "/".join(self._streams)
            url = f"{BINANCE_WS_COMBINED_URL}{stream_str}"
        else:
            url = BINANCE_WS_URL

        self._thread = threading.Thread(
            target=self._run,
            args=(url,),
            name="UnicornBinanceWebSocket",
            daemon=True,
        )
        self._thread.start()
        self._logger.info(f"WebSocket started: {url}")

    def _run(self, url: str) -> None:
        """WebSocket 运行循环 / WebSocket run loop."""
        self._ensure_websocket()

        self._ws = self._ws_class(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )

        while self._running:
            try:
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                self._logger.error(f"WebSocket error: {e}")

            if self._running:
                time.sleep(1)

    def _on_open(self, ws: Any) -> None:
        """WebSocket 连接打开回调 / WebSocket open callback."""
        self._logger.info("WebSocket connection opened")
        # Resubscribe to all streams on reconnect
        for stream, callback in list(self._subscriptions.items()):
            self._send(
                {"method": "SUBSCRIBE", "params": [stream], "id": self._next_id()}
            )

    def _on_close(self, ws: Any, close_status_code: int, close_msg: str) -> None:
        """WebSocket 连接关闭回调 / WebSocket close callback."""
        self._logger.info(
            f"WebSocket closed: {close_status_code} - {close_msg}"
        )

    def _on_error(self, ws: Any, error: Exception) -> None:
        """WebSocket 错误回调 / WebSocket error callback."""
        self._logger.error(f"WebSocket error: {error}")
        if self._on_error:
            self._on_error(error)

    def _on_message(self, ws: Any, message: str) -> None:
        """
        处理 WebSocket 消息 / Handle WebSocket message.

        自动路由消息到对应流的回调函数。

        参数 (Parameters):
            ws: WebSocket 连接
            message: 原始消息字符串
        """
        try:
            data = json.loads(message)

            # Handle combined stream format (with stream/data wrapper)
            if isinstance(data, dict) and "stream" in data and "data" in data:
                stream = data["stream"]
                event_data = data["data"]
            else:
                stream = ""
                event_data = data

            # Get appropriate callback
            callback = self._subscriptions.get(stream, self._callback)

            if callback:
                try:
                    callback(event_data)
                except Exception as e:
                    self._logger.error(f"Callback error for {stream}: {e}")

        except json.JSONDecodeError as e:
            self._logger.warning(f"Failed to decode message: {e}")
        except Exception as e:
            self._logger.error(f"Error processing message: {e}")

    def stop(self) -> None:
        """
        停止 WebSocket 连接 / Stop WebSocket connection.

        示例 (Example):
            >>> ws.stop()
        """
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception as e:
                self._logger.warning(f"Error closing WebSocket: {e}")
        if self._thread:
            self._thread.join(timeout=5)
        self._logger.info("WebSocket stopped")

    @property
    def is_running(self) -> bool:
        """检查 WebSocket 是否运行中 / Check if WebSocket is running."""
        return self._running
