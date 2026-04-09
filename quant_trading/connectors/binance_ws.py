"""
Binance WebSocket Client

Binance WebSocket real-time client for market data streams.
Supports: kline, depth, trade, ticker, miniTicker streams.

Supported streams:
- kline_<interval>       : Candlestick/kline streams
- <symbol>@depth         : Partial book depth stream
- <symbol>@trade         : Trade streams
- <symbol>@ticker        : 24hr ticker streams
- !miniTicker@arr        : All mini ticker streams

Bilingual docstrings (English/Chinese).

Usage:
    >>> from quant_trading.connectors.binance_ws import BinanceWebSocketClient
    >>> ws = BinanceWebSocketClient()
    >>> def on_kline(data): print(data)
    >>> ws.stream_klines("BTCUSDT", "1m", on_kline)
    >>> ws.start()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, List, Optional

try:
    import websocket

    _HAS_WEBSOCKET = True
except ImportError:
    _HAS_WEBSOCKET = False

# Binance WebSocket URLs
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
BINANCE_WS_COMBINED_URL = "wss://stream.binance.com:9443/stream?streams="

logger = logging.getLogger("BinanceWebSocketClient")


class BinanceWSError(Exception):
    """Binance WebSocket error."""

    pass


class BinanceWebSocketClient:
    """
    Binance WebSocket real-time client.

    Supports streams:
    - kline_<interval>     : Candlestick/kline streams
    - <symbol>@depth       : Partial book depth stream
    - <symbol>@trade       : Trade streams
    - <symbol>@ticker      : 24hr ticker streams
    - !miniTicker@arr       : All mini ticker streams

    Args:
        streams: Optional list of streams to subscribe to on start
        callback: Default callback for all messages

    Example:
        >>> ws = BinanceWebSocketClient()
        >>> ws.stream_klines("BTCUSDT", "1m", callback)
        >>> ws.start()
    """

    def __init__(
        self,
        streams: Optional[List[str]] = None,
        callback: Optional[Callable[[dict], None]] = None,
    ):
        if not _HAS_WEBSOCKET:
            raise ImportError(
                "websocket library is required: pip install websocket-client"
            )

        self._streams = streams or []
        self._callback = callback
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._subscriptions: dict[str, Callable] = {}
        self._logger = logging.getLogger("BinanceWebSocketClient")

    def subscribe(self, streams: List[str], callback: Callable) -> None:
        """
        Subscribe to WebSocket streams.

        Args:
            streams: List of stream names to subscribe to
                    e.g., ["btcusdt@kline_1m", "btcusdt@depth20"]
            callback: Callback function to handle messages

        Example:
            >>> ws.subscribe(["btcusdt@kline_1m", "btcusdt@trade"], my_callback)
        """
        for stream in streams:
            self._subscriptions[stream] = callback
            self._send_subscription(stream, subscribe=True)

    def unsubscribe(self, streams: List[str]) -> None:
        """
        Unsubscribe from WebSocket streams.

        Args:
            streams: List of stream names to unsubscribe from

        Example:
            >>> ws.unsubscribe(["btcusdt@kline_1m"])
        """
        for stream in streams:
            if stream in self._subscriptions:
                del self._subscriptions[stream]
            self._send_subscription(stream, subscribe=False)

    def _send_subscription(self, stream: str, subscribe: bool = True) -> None:
        """Send subscription/unsubscription message."""
        if self._ws and self._running:
            msg = {
                "method": "SUBSCRIBE" if subscribe else "UNSUBSCRIBE",
                "params": [stream],
                "id": int(time.time() * 1000),
            }
            self._ws.send(json.dumps(msg))

    def start(self) -> None:
        """
        Start the WebSocket connection in a background thread.

        Example:
            >>> ws = BinanceWebSocketClient()
            >>> ws.stream_klines("BTCUSDT", "1m", callback)
            >>> ws.start()
            >>> # ... later ...
            >>> ws.stop()
        """
        if self._running:
            return

        self._running = True

        # Build stream URL
        if self._streams:
            stream_str = "/".join(self._streams)
            url = f"{BINANCE_WS_COMBINED_URL}{stream_str}"
        else:
            url = BINANCE_WS_URL

        self._thread = threading.Thread(
            target=self._run_websocket,
            args=(url,),
            daemon=True,
        )
        self._thread.start()
        self._logger.info(f"WebSocket started: {url}")

    def _run_websocket(self, url: str) -> None:
        """Run WebSocket in background thread."""
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )

        while self._running:
            try:
                self._ws.run_forever(
                    ping_interval=30,
                    ping_timeout=10,
                )
            except Exception as e:
                self._logger.error(f"WebSocket error: {e}")

            if self._running:
                time.sleep(1)

    def _on_open(self, ws) -> None:
        """Handle WebSocket open."""
        self._logger.info("WebSocket connection opened")

    def _on_close(self, ws, close_status_code: int, close_msg: str) -> None:
        """Handle WebSocket close."""
        self._logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")

    def _on_error(self, ws, error: Exception) -> None:
        """Handle WebSocket error."""
        self._logger.error(f"WebSocket error: {error}")

    def _on_message(self, ws, message: str) -> None:
        """
        Handle incoming WebSocket message.

        Routes messages to appropriate callbacks based on stream name.
        """
        try:
            data = json.loads(message)

            # Handle combined stream format (with stream name)
            if isinstance(data, dict) and "stream" in data and "data" in data:
                stream = data["stream"]
                event_data = data["data"]
                event_type = event_data.get("e", "")
            else:
                # Single value or unknown format
                stream = ""
                event_data = data
                event_type = data.get("e", "") if isinstance(data, dict) else ""

            # Find callback for this stream
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
        Stop the WebSocket connection.

        Example:
            >>> ws.stop()
        """
        self._running = False
        if self._ws:
            self._ws.close()
        if self._thread:
            self._thread.join(timeout=5)
        self._logger.info("WebSocket stopped")

    # ===================
    # Stream Methods / 流方法
    # ===================

    def stream_klines(
        self,
        symbol: str,
        interval: str,
        callback: Callable,
    ) -> None:
        """
        Subscribe to kline/candlestick stream.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            callback: Callback function to handle kline data

        Example:
            >>> def on_kline(kline):
            ...     print(f"OHLC: {kline['o']}, {kline['h']}, {kline['l']}, {kline['c']}")
            >>> ws.stream_klines("BTCUSDT", "1m", on_kline)
        """
        stream = f"{symbol.lower()}@kline_{interval}"
        self._subscriptions[stream] = callback
        if self._running:
            self._send_subscription(stream, subscribe=True)

    def stream_depth(
        self,
        symbol: str,
        callback: Callable,
        level: int = 20,
    ) -> None:
        """
        Subscribe to order book depth stream.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            callback: Callback function to handle depth data
            level: Depth level (5, 10, 20, 100, 500, 1000)

        Example:
            >>> def on_depth(depth):
            ...     print(f"Bids: {depth['bids'][:5]}, Asks: {depth['asks'][:5]}")
            >>> ws.stream_depth("BTCUSDT", on_depth, level=20)
        """
        stream = f"{symbol.lower()}@depth{level}"
        self._subscriptions[stream] = callback
        if self._running:
            self._send_subscription(stream, subscribe=True)

    def stream_trade(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to trade stream.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            callback: Callback function to handle trade data

        Example:
            >>> def on_trade(trade):
            ...     print(f"Trade: {trade['p']} x {trade['q']}")
            >>> ws.stream_trade("BTCUSDT", on_trade)
        """
        stream = f"{symbol.lower()}@trade"
        self._subscriptions[stream] = callback
        if self._running:
            self._send_subscription(stream, subscribe=True)

    def stream_ticker(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to 24hr ticker stream.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            callback: Callback function to handle ticker data

        Example:
            >>> def on_ticker(ticker):
            ...     print(f"Last: {ticker['c']}, High: {ticker['h']}, Low: {ticker['l']}")
            >>> ws.stream_ticker("BTCUSDT", on_ticker)
        """
        stream = f"{symbol.lower()}@ticker"
        self._subscriptions[stream] = callback
        if self._running:
            self._send_subscription(stream, subscribe=True)

    def stream_all_mini_ticker(self, callback: Callable) -> None:
        """
        Subscribe to all mini ticker streams.

        Args:
            callback: Callback function to handle mini ticker data

        Example:
            >>> def on_mini_ticker(tickers):
            ...     for t in tickers:
            ...         print(f"{t['s']}: {t['c']}")
            >>> ws.stream_all_mini_ticker(on_mini_ticker)
        """
        stream = "!miniTicker@arr"
        self._subscriptions[stream] = callback
        if self._running:
            self._send_subscription(stream, subscribe=True)

    def stream_combined(
        self,
        streams: List[str],
        callback: Callable,
    ) -> None:
        """
        Subscribe to multiple streams at once.

        Args:
            streams: List of stream names (e.g., ["btcusdt@kline_1m", "btcusdt@depth20"])
            callback: Callback function to handle all stream data

        Example:
            >>> def on_data(data): print(data)
            >>> ws.stream_combined(["btcusdt@kline_1m", "btcusdt@trade"], on_data)
        """
        for stream in streams:
            self._subscriptions[stream] = callback

        if self._running:
            for stream in streams:
                self._send_subscription(stream, subscribe=True)


# ===================
# Backward Compatibility / 向后兼容
# ===================
# Legacy async WebSocket connector (retained for existing code compatibility)
# 旧版异步WebSocket连接器（为现有代码兼容性保留）

try:
    import asyncio
    import json as _json
    from dataclasses import dataclass, field
    from decimal import Decimal
    from typing import Any, Callable, Dict, List, Optional

    _HAS_AIOHTTP_WS = True
except ImportError:
    _HAS_AIOHTTP_WS = False


if _HAS_AIOHTTP_WS:
    # WebSocket URLs
    _BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
    _BINANCE_WS_COMBINED_URL = "wss://stream.binance.com:9443/stream?streams="


    @dataclass
    class WSSubscription:
        """Represents a WebSocket stream subscription."""

        stream_name: str
        stream_type: str
        symbol: Optional[str] = None
        interval: Optional[str] = None
        callback: Optional[Callable] = None
        subscribed: bool = False


    @dataclass
    class BinanceWSMessage:
        """Parsed WebSocket message."""

        stream: str
        event_type: str
        data: Dict[str, Any]
        timestamp: float


    class BinanceWSConnector:
        """
        Binance WebSocket API connector (async version).

        Manages WebSocket connections to Binance for real-time data streams.
        """

        def __init__(
            self,
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            testnet: bool = False,
            on_user_data: Optional[Callable[[Dict], None]] = None,
            on_order_book_update: Optional[Callable[[str, Dict], None]] = None,
            on_trade: Optional[Callable[[Dict], None]] = None,
            on_ticker: Optional[Callable[[Dict], None]] = None,
        ):
            self._api_key = api_key
            self._api_secret = api_secret
            self._testnet = testnet
            self._base_url = _BINANCE_WS_URL
            self._on_user_data = on_user_data
            self._on_order_book_update = on_order_book_update
            self._on_trade = on_trade
            self._on_ticker = on_ticker
            self._ws: Optional[Any] = None
            self._session: Optional[Any] = None
            self._connected = False
            self._subscribed_streams: Dict[str, WSSubscription] = {}
            self._logger = logging.getLogger("BinanceWSConnector")
            self._tasks: List[asyncio.Task] = []

        @property
        def is_connected(self) -> bool:
            return self._connected

        async def connect(self):
            import aiohttp
            if self._session is None:
                self._session = aiohttp.ClientSession()
            self._connected = True

        async def disconnect(self):
            self._connected = False
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            self._tasks.clear()
            if self._ws:
                await self._ws.close()
                self._ws = None
            if self._session:
                await self._session.close()
                self._session = None

        async def subscribe_depth(self, symbol: str, levels: int = 10,
                                  callback: Optional[Callable] = None):
            stream_name = f"{symbol.lower()}@depth{levels}@100ms"
            sub = WSSubscription(
                stream_name=stream_name,
                stream_type="depth",
                symbol=symbol.upper(),
                callback=callback or self._on_order_book_update,
            )
            self._subscribed_streams[stream_name] = sub
            return stream_name

        async def subscribe_trade(self, symbol: str, callback: Optional[Callable] = None):
            stream_name = f"{symbol.lower()}@trade"
            sub = WSSubscription(
                stream_name=stream_name,
                stream_type="trade",
                symbol=symbol.upper(),
                callback=callback or self._on_trade,
            )
            self._subscribed_streams[stream_name] = sub
            return stream_name

        async def subscribe_ticker(self, symbol: str, callback: Optional[Callable] = None):
            stream_name = f"{symbol.lower()}@ticker"
            sub = WSSubscription(
                stream_name=stream_name,
                stream_type="ticker",
                symbol=symbol.upper(),
                callback=callback or self._on_ticker,
            )
            self._subscribed_streams[stream_name] = sub
            return stream_name

        async def subscribe_kline(self, symbol: str, interval: str = "1m",
                                 callback: Optional[Callable] = None):
            stream_name = f"{symbol.lower()}@kline_{interval}"
            sub = WSSubscription(
                stream_name=stream_name,
                stream_type="kline",
                symbol=symbol.upper(),
                interval=interval,
                callback=callback,
            )
            self._subscribed_streams[stream_name] = sub
            return stream_name

        async def subscribe_user_data(self, listen_key: str):
            stream_name = listen_key
            sub = WSSubscription(
                stream_name=stream_name,
                stream_type="user_data",
                callback=self._on_user_data,
            )
            self._subscribed_streams[stream_name] = sub
            return stream_name


    class BinanceWSManager:
        """
        WebSocket manager for multiple Binance connections.

        Provides high-level interface for managing multiple WebSocket streams.
        """

        def __init__(
            self,
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            testnet: bool = False,
        ):
            self._api_key = api_key
            self._api_secret = api_secret
            self._testnet = testnet
            self._public_ws: Optional[BinanceWSConnector] = None
            self._user_ws: Optional[BinanceWSConnector] = None
            self._order_book_callbacks: Dict[str, Callable] = {}
            self._trade_callbacks: Dict[str, Callable] = {}
            self._ticker_callbacks: Dict[str, Callable] = {}
            self._user_data_callback: Optional[Callable] = None
            self._logger = logging.getLogger("BinanceWSManager")

        async def start(self):
            self._public_ws = BinanceWSConnector(
                on_order_book_update=self._handle_order_book_update,
                on_trade=self._handle_trade,
                on_ticker=self._handle_ticker,
            )
            await self._public_ws.connect()
            if self._api_key:
                self._user_ws = BinanceWSConnector(
                    api_key=self._api_key,
                    api_secret=self._api_secret,
                    on_user_data=self._handle_user_data,
                )
                await self._user_ws.connect()

        async def stop(self):
            if self._public_ws:
                await self._public_ws.disconnect()
            if self._user_ws:
                await self._user_ws.disconnect()

        async def subscribe_order_book(self, symbol: str, levels: int = 10,
                                       callback: Optional[Callable] = None):
            if self._public_ws:
                stream_name = await self._public_ws.subscribe_depth(symbol, levels)
                if callback:
                    self._order_book_callbacks[symbol] = callback
                return stream_name
            return None

        async def subscribe_trades(self, symbol: str, callback: Optional[Callable] = None):
            if self._public_ws:
                stream_name = await self._public_ws.subscribe_trade(symbol)
                if callback:
                    self._trade_callbacks[symbol] = callback
                return stream_name
            return None

        async def subscribe_ticker(self, symbol: str, callback: Optional[Callable] = None):
            if self._public_ws:
                stream_name = await self._public_ws.subscribe_ticker(symbol)
                if callback:
                    self._ticker_callbacks[symbol] = callback
                return stream_name
            return None

        def _handle_order_book_update(self, symbol: str, data: Dict):
            if symbol in self._order_book_callbacks:
                self._order_book_callbacks[symbol](data)

        def _handle_trade(self, data: Dict):
            symbol = data.get("symbol", "")
            if symbol in self._trade_callbacks:
                self._trade_callbacks[symbol](data)

        def _handle_ticker(self, data: Dict):
            symbol = data.get("symbol", "")
            if symbol in self._ticker_callbacks:
                self._ticker_callbacks[symbol](data)

        def _handle_user_data(self, data: Dict):
            if self._user_data_callback:
                self._user_data_callback(data)


__all__ = [
    # New sync client
    "BinanceWebSocketClient",
    "BinanceWSError",
    # Legacy async connector
    "BinanceWSConnector",
    "BinanceWSManager",
    "WSSubscription",
    "BinanceWSMessage",
]
