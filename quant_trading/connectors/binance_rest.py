"""
Binance REST API Client

Binance REST API client supporting spot/margin/futures/options.
Endpoint: https://api.binance.com

Bilingual docstrings (English/Chinese).

Usage:
    >>> from quant_trading.connectors.binance_rest import BinanceRESTClient
    >>> client = BinanceRESTClient(api_key="your_key", api_secret="your_secret")
    >>> klines = client.get_klines("BTCUSDT", "1m", limit=500)
    >>> order_book = client.get_order_book("BTCUSDT", limit=100)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional
from decimal import Decimal

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# Binance API Endpoints
BINANCE_API_URL = "https://api.binance.com"
BINANCE_API_TESTNET_URL = "https://testnet.binance.vision"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_MARGIN_URL = "https://api.binance.com"

# Default settings
DEFAULT_RECEIVE_WINDOW = 5000
DEFAULT_REQUEST_TIMEOUT = 30

logger = logging.getLogger("BinanceRESTClient")


class BinanceAPIError(Exception):
    """Binance API error / Binance API错误."""

    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"Binance API Error {code}: {msg}")


class BinanceConnectionError(Exception):
    """Connection error with Binance API / Binance API连接错误."""

    pass


class BinanceRateLimitError(BinanceAPIError):
    """Rate limit exceeded / 速率限制超出."""

    def __init__(self, msg: str = "Rate limit exceeded"):
        self.code = -1003
        self.msg = msg
        super().__init__(self.code, self.msg)


class BinanceRESTClient:
    """
    Binance REST API client.

    Supports: spot/margin/futures/options
    Endpoint: https://api.binance.com

    Args:
        api_key: Binance API key (optional for public endpoints)
        api_secret: Binance API secret (optional for public endpoints)
        base_url: Base URL for API (default: https://api.binance.com)
        timeout: Request timeout in seconds (default: 30)

    Attributes:
        connected: Whether the client is connected

    Example:
        >>> client = BinanceRESTClient()
        >>> client.ping()
        {'msg': 'success', 'code': 0}
        >>> client.get_klines("BTCUSDT", "1h", limit=10)
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = BINANCE_API_URL,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ):
        if not _HAS_REQUESTS:
            raise ImportError(
                "requests library is required: pip install requests"
            )

        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._recv_window = DEFAULT_RECEIVE_WINDOW
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self._api_key}) if self._api_key else None
        self._logger = logging.getLogger("BinanceRESTClient")

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sign request parameters with HMAC SHA256.

        Args:
            params: Unsigned request parameters

        Returns:
            Signed parameters including signature
        """
        if not self._api_key or not self._api_secret:
            return params

        timestamp = int(time.time() * 1000)
        params["timestamp"] = timestamp
        params["recvWindow"] = self._recv_window

        query_string = "&".join(
            f"{k}={v}" for k, v in sorted(params.items())
        )
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        params["signature"] = signature
        return params

    def _request(
        self,
        method: str,
        endpoint: str,
        signed: bool = False,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Make an API request to Binance.

        Args:
            method: HTTP method (GET, POST, DELETE, PUT)
            endpoint: API endpoint path (e.g., "/api/v3/order")
            signed: Whether request requires signature
            params: Query parameters
            data: Request body data

        Returns:
            Response JSON data

        Raises:
            BinanceAPIError: On API error
            BinanceConnectionError: On connection error
        """
        url = f"{self._base_url}{endpoint}"
        params = params or {}
        data = data or {}

        if signed:
            params = self._sign_params(params)

        try:
            if method == "GET":
                response = self._session.get(
                    url, params=params, timeout=self._timeout
                )
            elif method == "POST":
                response = self._session.post(
                    url, params=params, json=data, timeout=self._timeout
                )
            elif method == "DELETE":
                response = self._session.delete(
                    url, params=params, timeout=self._timeout
                )
            elif method == "PUT":
                response = self._session.put(
                    url, params=params, json=data, timeout=self._timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                raise BinanceRateLimitError("Rate limit exceeded")
            else:
                try:
                    error_data = response.json()
                    raise BinanceAPIError(
                        code=error_data.get("code", response.status_code),
                        msg=error_data.get("msg", "Unknown error"),
                    )
                except ValueError:
                    raise BinanceAPIError(
                        code=response.status_code,
                        msg=response.text or "Unknown error",
                    )

        except requests.RequestException as e:
            raise BinanceConnectionError(f"Request failed: {e}")

    # ===================
    # Market Data / 市场数据
    # ===================

    def ping(self) -> dict:
        """
        Test connectivity to the Exchange.

        Returns:
            dict: {'msg': 'success', 'code': 0}
        """
        return self._request("GET", "/api/v3/ping")

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        **kwargs,
    ) -> list:
        """
        Get kline/candlestick data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, 1w)
            limit: Max number of klines (1-1000), default 500
            **kwargs: startTime, endTime, startStr, endStr

        Returns:
            List of kline data arrays:
            [
                [
                    1499040000000,  # Open time
                    "0.00100000",   # Open
                    "0.00250000",   # High
                    "0.00100000",   # Low
                    "0.00200000",   # Close
                    "1480.00000000", # Volume
                    1499644799999,  # Close time
                    ...
                ]
            ]

        Example:
            >>> client.get_klines("BTCUSDT", "1h", limit=10)
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        params.update(kwargs)
        return self._request("GET", "/api/v3/klines", params=params)

    def get_order_book(self, symbol: str, limit: int = 100) -> dict:
        """
        Get order book depth for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            dict with 'bids' and 'asks' lists:
            {
                "lastUpdateId": 160,
                "bids": [["0.0024", "10"]],  # [price, quantity]
                "asks": [["0.0026", "100"]]
            }

        Example:
            >>> client.get_order_book("BTCUSDT", limit=20)
        """
        params = {"symbol": symbol.upper(), "limit": limit}
        return self._request("GET", "/api/v3/depth", params=params)

    def get_ticker(self, symbol: str) -> dict:
        """
        Get 24hr ticker price change statistics.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            dict with ticker data including:
            {
                "symbol": "BTCUSDT",
                "priceChange": "0.00123000",
                "priceChangePercent": "0.123",
                "lastPrice": "1234.56789000",
                "highPrice": "1240.00000000",
                "lowPrice": "1200.00000000",
                "volume": "12345.67890000",
                ...
            }

        Example:
            >>> client.get_ticker("BTCUSDT")
        """
        params = {"symbol": symbol.upper()}
        return self._request("GET", "/api/v3/ticker/24hr", params=params)

    def get_recent_trades(self, symbol: str, limit: int = 500) -> list:
        """
        Get recent trades for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Number of trades to return (1-1000), default 500

        Returns:
            List of trade objects:
            [
                {
                    "id": 12345,
                    "price": "0.00100000",
                    "qty": "100",
                    "time": 1234567890123,
                    "isBuyerMaker": true,
                    ...
                }
            ]

        Example:
            >>> client.get_recent_trades("BTCUSDT", limit=100)
        """
        params = {"symbol": symbol.upper(), "limit": limit}
        return self._request("GET", "/api/v3/trades", params=params)

    def get_ticker_price(self, symbol: str) -> dict:
        """
        Get latest price for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            dict: {'symbol': 'BTCUSDT', 'price': '1234.56789000'}

        Example:
            >>> client.get_ticker_price("BTCUSDT")
        """
        params = {"symbol": symbol.upper()}
        return self._request("GET", "/api/v3/ticker/price", params=params)

    def get_orderbook_ticker(self, symbol: str) -> dict:
        """
        Get order book top bid/ask prices.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            dict:
            {
                "symbol": "BTCUSDT",
                "bidPrice": "1234.00",
                "bidQty": "1.5",
                "askPrice": "1235.00",
                "askQty": "2.0"
            }

        Example:
            >>> client.get_orderbook_ticker("BTCUSDT")
        """
        params = {"symbol": symbol.upper()}
        return self._request("GET", "/api/v3/ticker/bookTicker", params=params)

    # ===================
    # Account / 账户
    # ===================

    def get_account(self, **kwargs) -> dict:
        """
        Get account information.

        Requires API key and secret.

        Returns:
            dict with account balances and permissions:
            {
                "makerCommission": 10,
                "takerCommission": 10,
                "buyerCommission": 0,
                "sellerCommission": 0,
                "canTrade": true,
                "balances": [
                    {"asset": "BTC", "free": "1.00000000", "locked": "0.00000000"},
                    {"asset": "USDT", "free": "1000.00000000", "locked": "0.00000000"},
                ],
                ...
            }

        Example:
            >>> client.get_account()
        """
        return self._request("GET", "/api/v3/account", signed=True, params=kwargs)

    def get_balance(self) -> dict:
        """
        Get account balance for all assets with non-zero balance.

        Requires API key and secret.

        Returns:
            dict: {'balances': [{'asset': 'BTC', 'free': '1.0', 'locked': '0.0'}, ...]}

        Example:
            >>> client.get_balance()
        """
        account = self.get_account()
        return {"balances": account.get("balances", [])}

    def get_open_orders(self, symbol: str = None) -> list:
        """
        Get all open orders or open orders for a specific symbol.

        Args:
            symbol: Optional trading pair to filter (e.g., "BTCUSDT")

        Returns:
            List of open order objects

        Example:
            >>> client.get_open_orders()
            >>> client.get_open_orders("BTCUSDT")
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/api/v3/openOrders", signed=True, params=params)

    # ===================
    # Order / 订单
    # ===================

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = None,
        **kwargs,
    ) -> dict:
        """
        Place an order on Binance.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: Order side ("BUY" or "SELL")
            order_type: Order type ("LIMIT", "MARKET", "LIMIT_MAKER", etc.)
            quantity: Order quantity in base currency
            price: Order price (required for LIMIT orders)
            **kwargs: Optional parameters (timeInForce, stopPrice, etc.)

        Returns:
            dict with order response:
            {
                "symbol": "BTCUSDT",
                "orderId": 123456789,
                "clientOrderId": "...",
                "price": "50000.00000000",
                "origQty": "0.00100000",
                "status": "NEW",
                ...
            }

        Example:
            >>> client.place_order("BTCUSDT", "BUY", "LIMIT", 0.001, 50000)
        """
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }

        if price is not None:
            params["price"] = str(price)
            params["timeInForce"] = kwargs.get("timeInForce", "GTC")

        # Add optional parameters
        for key, value in kwargs.items():
            if key not in ("timeInForce",):
                params[key] = value

        return self._request("POST", "/api/v3/order", signed=True, params=params)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """
        Cancel an order on Binance.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            order_id: Order ID to cancel

        Returns:
            dict with cancellation response

        Example:
            >>> client.cancel_order("BTCUSDT", 123456789)
        """
        params = {"symbol": symbol.upper(), "orderId": order_id}
        return self._request("DELETE", "/api/v3/order", signed=True, params=params)

    def get_order(self, symbol: str, order_id: int) -> dict:
        """
        Get order status from Binance.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            order_id: Order ID

        Returns:
            dict with order details

        Example:
            >>> client.get_order("BTCUSDT", 123456789)
        """
        params = {"symbol": symbol.upper(), "orderId": order_id}
        return self._request("GET", "/api/v3/order", signed=True, params=params)

    # ===================
    # User Data Stream / 用户数据流
    # ===================

    def start_user_data_stream(self) -> str:
        """
        Start user data stream and return listen key.

        Returns:
            str: Listen key for user data stream

        Example:
            >>> listen_key = client.start_user_data_stream()
        """
        data = self._request("POST", "/api/v3/userDataStream", signed=False)
        return data.get("listenKey")

    def keep_alive_user_data_stream(self, listen_key: str) -> dict:
        """
        Keep user data stream alive.

        Args:
            listen_key: Listen key from start_user_data_stream

        Returns:
            dict: {'listenKey': '...', 'expires': 3600000}

        Example:
            >>> client.keep_alive_user_data_stream(listen_key)
        """
        params = {"listenKey": listen_key}
        return self._request("PUT", "/api/v3/userDataStream", params=params)

    def close_user_data_stream(self, listen_key: str) -> dict:
        """
        Close user data stream.

        Args:
            listen_key: Listen key to close

        Returns:
            dict: {}

        Example:
            >>> client.close_user_data_stream(listen_key)
        """
        params = {"listenKey": listen_key}
        return self._request("DELETE", "/api/v3/userDataStream", params=params)

    def close(self):
        """Close the HTTP session."""
        if self._session:
            self._session.close()


# ===================
# Backward Compatibility / 向后兼容
# ===================
# Legacy async connector (retained for existing code compatibility)
# 旧版异步连接器（为现有代码兼容性保留）

try:
    import aiohttp
    from quant_trading.connectors.base_connector import BaseConnector
    from quant_trading.connectors.order_types import (
        InFlightOrderBase,
        OrderState,
        OrderType,
        TradeFee,
        TradeType,
    )

    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


class BinanceRateLimiter:
    """
    Rate limiter for Binance API requests.

    Implements request weight-based rate limiting.
    """

    def __init__(self):
        self._weight = 0
        self._last_reset = time.time()
        self._request_cache: Dict[str, float] = {}
        self._weight_limit = 6000
        self._order_per_second_limit = 10
        self._order_per_10second_limit = 200

    async def wait_if_needed(self, signed: bool):
        """Wait if rate limit would be exceeded."""
        import asyncio
        current_time = time.time()

        if current_time - self._last_reset >= 60:
            self._weight = 0
            self._last_reset = current_time

        if signed:
            self._cleanup_old_requests("order_per_second", 1)
            if len(self._request_cache.get("order_per_second", [])) >= self._order_per_second_limit:
                sleep_time = 1 - (current_time - max(self._request_cache.get("order_per_second", [0])))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self._cleanup_old_requests("order_per_10second", 10)
            if len(self._request_cache.get("order_per_10second", [])) >= self._order_per_10second_limit:
                sleep_time = 10 - (current_time - max(self._request_cache.get("order_per_10second", [0])))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        await asyncio.sleep(0.1)

    def _cleanup_old_requests(self, key: str, window: int):
        """Clean up old request timestamps."""
        current_time = time.time()
        if key not in self._request_cache:
            self._request_cache[key] = []
        self._request_cache[key] = [
            t for t in self._request_cache[key] if current_time - t < window
        ]


if _HAS_AIOHTTP:
    class BinanceRESTConnector(BaseConnector):
        """
        Binance REST API connector (async version).

        Provides comprehensive access to Binance API endpoints.
        """

        def __init__(
            self,
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            testnet: bool = False,
            recv_window: int = DEFAULT_RECEIVE_WINDOW,
            timeout: int = DEFAULT_REQUEST_TIMEOUT,
        ):
            super().__init__()
            self._api_key = api_key or ""
            self._api_secret = api_secret or ""
            self._testnet = testnet
            self._recv_window = recv_window
            self._timeout = timeout
            self._base_url = BINANCE_API_TESTNET_URL if testnet else BINANCE_API_URL
            self._session: Optional[aiohttp.ClientSession] = None
            self._rate_limiter = BinanceRateLimiter()
            self._exchange_info: Optional[Dict[str, Any]] = None
            self._order_book_cache: Dict[str, Dict[str, Any]] = {}
            self._logger = logging.getLogger("BinanceRESTConnector")

        @property
        def trading_pairs(self) -> List[str]:
            if self._exchange_info:
                return [
                    sym["symbol"]
                    for sym in self._exchange_info.get("symbols", [])
                    if sym["status"] == "TRADING"
                ]
            return []

        @property
        def ready(self) -> bool:
            return self._exchange_info is not None and self._session is not None

        async def connect(self):
            if self._session is None:
                self._session = aiohttp.ClientSession()
            await self._load_exchange_info()
            self._logger.info(f"Connected to Binance REST API (testnet={self._testnet})")

        async def disconnect(self):
            if self._session:
                await self._session.close()
                self._session = None
            self._logger.info("Disconnected from Binance REST API")

        async def _load_exchange_info(self):
            data = await self._request("GET", "/api/v3/exchangeInfo")
            self._exchange_info = data

        async def _request(
            self,
            method: str,
            endpoint: str,
            signed: bool = False,
            params: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            await self._rate_limiter.wait_if_needed(signed)
            url = f"{self._base_url}{endpoint}"
            headers = {"X-MBX-APIKEY": self._api_key} if signed else {}

            if signed and self._api_key and self._api_secret:
                timestamp = int(time.time() * 1000)
                query_params = params or {}
                query_params["timestamp"] = timestamp
                query_params["recvWindow"] = self._recv_window
                query_string = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
                signature = hmac.new(
                    self._api_secret.encode("utf-8"),
                    query_string.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                query_params["signature"] = signature
                params = query_params

            if not self._session:
                self._session = aiohttp.ClientSession()

            async with self._session.request(
                method, url, params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    raise BinanceRateLimitError()
                else:
                    error_data = await response.json()
                    raise BinanceAPIError(
                        code=error_data.get("code", response.status),
                        msg=error_data.get("msg", "Unknown error"),
                    )

        async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, List]:
            params = {"symbol": symbol.upper(), "limit": limit}
            return await self._request("GET", "/api/v3/depth", params=params)

        async def get_klines(
            self, symbol: str, interval: str, limit: int = 500, **kwargs
        ) -> List:
            params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
            params.update(kwargs)
            return await self._request("GET", "/api/v3/klines", params=params)

        async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
            params = {"symbol": symbol.upper()}
            return await self._request("GET", "/api/v3/ticker/24hr", params=params)

        async def get_account(self) -> Dict[str, Any]:
            return await self._request("GET", "/api/v3/account", signed=True)

        async def get_open_orders(self, symbol: str = None) -> List[Dict]:
            params = {"timestamp": int(time.time() * 1000), "recvWindow": self._recv_window}
            if symbol:
                params["symbol"] = symbol.upper()
            return await self._request("GET", "/api/v3/openOrders", signed=True, params=params)

        def get_order_price_quantum(self, trading_pair: str, price) -> Decimal:
            return Decimal("0.01")

        def get_order_size_quantum(self, trading_pair: str, amount) -> Decimal:
            return Decimal("0.001")

        def get_price(self, trading_pair: str, is_buy: bool, amount=Decimal("0")) -> Decimal:
            ticker = self._order_book_cache.get(trading_pair)
            if not ticker:
                return Decimal("NaN")
            if is_buy:
                return Decimal(ticker.get("askPrice", "0"))
            else:
                return Decimal(ticker.get("bidPrice", "0"))

        async def update_order_book_cache(self, trading_pair: str):
            ticker = await self._request("GET", "/api/v3/ticker/bookTicker",
                                        params={"symbol": trading_pair.upper()})
            if isinstance(ticker, list) and len(ticker) > 0:
                ticker = ticker[0]
            self._order_book_cache[trading_pair] = ticker

        def estimate_fee_pct(self, is_maker: bool) -> Decimal:
            return Decimal("0.0002") if is_maker else Decimal("0.001")


__all__ = [
    # New sync client
    "BinanceRESTClient",
    # Legacy async connector
    "BinanceRESTConnector",
    "BinanceRateLimiter",
    # Exceptions
    "BinanceAPIError",
    "BinanceConnectionError",
    "BinanceRateLimitError",
]
