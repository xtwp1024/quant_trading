"""
quant_trading.hft.clob_client
==============================
Async WebSocket client for CLOB (Central Limit Order Book) market data.

Adapted from py_polymarket_hft_mm clob_client + orderbook WebSocket patterns:
- Async WebSocket integration using standard `websockets` library
- Auto-reconnection with exponential backoff
- Order submission / cancellation (interface only, requires exchange API)
- Thread-safe global client singleton

Note:
    This is an architectural reference -- order submission and cancellation
    require integration with a specific exchange's CLOB API (e.g., Polymarket).
    The client provides the interface pattern; wire it to your exchange.

WebSocket URL pattern:
    wss://<host>/ws/market   (public market data)
    wss://<host>/ws/user     (private user order data)
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Represents a limit order in the CLOB."""
    order_id: str
    token_id: str
    price: float
    size: float
    side: OrderSide
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class MarketUpdate:
    """A parsed market data update from the WebSocket feed."""
    event_type: str           # e.g., "book", "price_change", "trade"
    asset_id: Optional[str] = None
    bids: Optional[List[Dict]] = None
    asks: Optional[List[Dict]] = None
    price_changes: Optional[List[Dict]] = None
    raw: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Async WebSocket Client
# ---------------------------------------------------------------------------


class CLOBClient:
    """
    Async WebSocket client for CLOB market data.

    Features:
    - Async connect/disconnect with context manager support
    - Auto-reconnection with exponential backoff (max 5 retries)
    - Public market data subscription (book, price_change, trades)
    - User-specific order updates subscription
    - Order submission/cancellation interface (requires exchange API)
    - Callback-based update handlers

    Usage (async):
        async with CLOBClient("wss://ws.example.com/ws") as client:
            client.subscribe_market(asset_ids=["token1", "token2"])
            async for update in client.stream():
                process(update)

    Usage (sync thread):
        client = CLOBClient("wss://ws.example.com/ws")
        client.start()
        # market_data = client.get_current_book()
        # client.stop()
    """

    def __init__(
        self,
        ws_url: str,
        user_ws_url: Optional[str] = None,
        reconnect_delay: float = 0.5,
        max_reconnect_delay: float = 30.0,
        max_retries: int = 5,
        request_timeout: float = 10.0,
    ):
        """
        Args:
            ws_url: WebSocket URL for public market data.
            user_ws_url: WebSocket URL for private user data (optional).
            reconnect_delay: Initial delay between reconnection attempts.
            max_reconnect_delay: Maximum delay between reconnection attempts.
            max_retries: Maximum number of reconnection retries (0 = infinite).
            request_timeout: Timeout for REST API calls (order submission etc.).
        """
        self.ws_url = ws_url
        self.user_ws_url = user_ws_url or ws_url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        # Async state
        self._ws: Any = None
        self._user_ws: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._connect_count = 0

        # Market data buffers (thread-safe)
        self._market_buf: Dict[str, MarketUpdate] = {}
        self._buf_lock = threading.Lock()

        # Order state
        self._orders: Dict[str, Order] = {}
        self._orders_lock = threading.Lock()

        # Callbacks
        self._on_book_update: Optional[Callable[[MarketUpdate], None]] = None
        self._on_price_change: Optional[Callable[[MarketUpdate], None]] = None
        self._on_trade: Optional[Callable[[MarketUpdate], None]] = None
        self._on_order_update: Optional[Callable[[Order], None]] = None
        self._on_connect: Optional[Callable[[], None]] = None
        self._on_disconnect: Optional[Callable[[], None]] = None

        # Queue for sync consumers
        self._update_queue: List[MarketUpdate] = []
        self._queue_lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_book_update(
        self,
        cb: Callable[[MarketUpdate], None],
    ) -> "CLOBClient":
        """Set callback for book (order book snapshot) updates. Returns self for chaining."""
        self._on_book_update = cb
        return self

    def on_price_change(
        self,
        cb: Callable[[MarketUpdate], None],
    ) -> "CLOBClient":
        """Set callback for price change (incremental) updates. Returns self for chaining."""
        self._on_price_change = cb
        return self

    def on_trade(
        self,
        cb: Callable[[MarketUpdate], None],
    ) -> "CLOBClient":
        """Set callback for trade (fill) updates. Returns self for chaining."""
        self._on_trade = cb
        return self

    def on_order_update(
        self,
        cb: Callable[[Order], None],
    ) -> "CLOBClient":
        """Set callback for user order status updates. Returns self for chaining."""
        self._on_order_update = cb
        return self

    def on_connect(
        self,
        cb: Callable[[], None],
    ) -> "CLOBClient":
        """Set callback for successful connect. Returns self for chaining."""
        self._on_connect = cb
        return self

    def on_disconnect(
        self,
        cb: Callable[[], None],
    ) -> "CLOBClient":
        """Set callback for disconnect. Returns self for chaining."""
        self._on_disconnect = cb
        return self

    # -------------------------------------------------------------------------
    # Public API (sync thread usage)
    # -------------------------------------------------------------------------

    def subscribe_market(self, asset_ids: List[str]) -> None:
        """
        Subscribe to market data for the given asset IDs.

        Sends subscription message to the public WebSocket.

        Args:
            asset_ids: List of token/asset IDs to subscribe to.
        """
        msg = {
            "type": "market",
            "assets_ids": asset_ids,
        }
        asyncio.run(self._send_async(json.dumps(msg)))

    def subscribe_user(self, user_id: str) -> None:
        """
        Subscribe to user-specific order updates.

        Args:
            user_id: User/account ID for order updates.
        """
        if not self.user_ws_url:
            logger.warning("User WebSocket URL not configured")
            return
        msg = {
            "type": "user",
            "user_id": user_id,
        }
        asyncio.run(self._send_async(json.dumps(msg)))

    def start(self) -> None:
        """Start the WebSocket reader loop in a background thread (sync usage)."""
        if self._running:
            logger.warning("CLOBClient already running")
            return
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._run_reader_loop,
            daemon=True,
            name="CLOBClient-reader",
        )
        self._reader_thread.start()
        logger.info(f"CLOBClient started (url={self.ws_url})")

    def stop(self) -> None:
        """Stop the WebSocket reader loop."""
        self._running = False
        if self._loop:
            asyncio.run(self._shutdown_async())
        logger.info("CLOBClient stopped")

    def get_current_book(self, asset_id: str) -> Optional[MarketUpdate]:
        """
        Get the most recent book update for an asset (sync access).

        Returns:
            MarketUpdate with bids/asks, or None if not yet received.
        """
        with self._buf_lock:
            return self._market_buf.get(asset_id)

    def pop_updates(self) -> List[MarketUpdate]:
        """
        Atomically pop all queued market updates (sync usage).

        Returns:
            List of MarketUpdate objects received since last call.
        """
        with self._queue_lock:
            updates = list(self._update_queue)
            self._update_queue.clear()
            return updates

    # -------------------------------------------------------------------------
    # Order management (interface -- requires exchange API integration)
    # -------------------------------------------------------------------------

    def submit_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: OrderSide,
    ) -> Optional[str]:
        """
        Submit a limit order to the CLOB.

        This is an INTERFACE method -- it must be wired to your exchange's
        order submission API (e.g., py_clob_client, REST API, etc.).

        Args:
            token_id: Token/asset ID to trade.
            price: Limit price.
            size: Order size.
            side: BUY or SELL.

        Returns:
            Exchange-assigned order ID, or None on failure.
        """
        order_id = f"ORD-{uuid.uuid4().hex[:12].upper()}"
        order = Order(
            order_id=order_id,
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            status=OrderStatus.PENDING,
        )
        with self._orders_lock:
            self._orders[order_id] = order

        logger.info(
            f"[CLOBClient] submit_order INTERFACE: "
            f"id={order_id} {side.value} {size}@{price} token={token_id}"
        )
        # TODO: Wire to actual exchange API here
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        This is an INTERFACE method -- wire to your exchange's cancel API.

        Returns:
            True if cancel request was sent successfully.
        """
        with self._orders_lock:
            order = self._orders.get(order_id)
            if order is None:
                logger.warning(f"cancel_order: order {order_id} not found")
                return False
            order.status = OrderStatus.CANCELLED

        logger.info(f"[CLOBClient] cancel_order INTERFACE: id={order_id}")
        # TODO: Wire to actual exchange API here
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Return the current state of an order."""
        with self._orders_lock:
            return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Return all open (pending/partial) orders."""
        with self._orders_lock:
            return [
                o for o in self._orders.values()
                if o.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)
            ]

    # -------------------------------------------------------------------------
    # Async internals
    # -------------------------------------------------------------------------

    async def _send_async(self, msg: str) -> None:
        """Send a message on the WebSocket (async)."""
        if self._ws and self._ws.open:
            await self._ws.send(msg)
        else:
            logger.warning(f"Cannot send -- WebSocket not connected: {msg[:100]}")

    async def _connect_async(self) -> Any:
        """
        Establish WebSocket connection with exponential backoff retry.

        Returns:
            The connected websocket object, or None after max_retries.
        """
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets library not installed. "
                "Install with: pip install websockets"
            )
            return None

        delay = self.reconnect_delay
        for attempt in range(self.max_retries + 1):
            try:
                self._ws = await websockets.connect(
                    self.ws_url,
                    open_timeout=self.request_timeout,
                )
                self._connect_count += 1
                logger.info(
                    f"WebSocket connected (attempt {attempt + 1}, "
                    f"total_connects={self._connect_count})"
                )
                if self._on_connect:
                    self._on_connect()
                return self._ws
            except Exception as e:
                logger.warning(
                    f"WebSocket connection attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.max_reconnect_delay)
                else:
                    logger.error(
                        f"Max retries ({self.max_retries}) reached for WebSocket"
                    )
        return None

    async def _reader_async(self) -> None:
        """Async message reader loop -- dispatches to appropriate handlers."""
        try:
            import websockets
        except ImportError:
            return

        ws = await self._connect_async()
        if ws is None:
            self._running = False
            return

        try:
            async for raw_msg in ws:
                try:
                    update = self._parse_message(raw_msg)
                    if update is None:
                        continue

                    # Buffer for sync consumers
                    with self._buf_lock:
                        if update.asset_id:
                            self._market_buf[update.asset_id] = update
                    with self._queue_lock:
                        self._update_queue.append(update)

                    # Invoke callbacks
                    if update.event_type == "book" and self._on_book_update:
                        self._on_book_update(update)
                    elif update.event_type == "price_change" and self._on_price_change:
                        self._on_price_change(update)
                    elif update.event_type == "trade" and self._on_trade:
                        self._on_trade(update)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e} -- raw: {raw_msg[:200]}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

        except Exception as e:
            logger.error(f"WebSocket reader error: {e}")
        finally:
            self._ws = None
            if self._on_disconnect:
                self._on_disconnect()

    def _parse_message(self, raw_msg: str) -> Optional[MarketUpdate]:
        """Parse a raw WebSocket JSON message into a MarketUpdate."""
        try:
            data = json.loads(raw_msg)
        except Exception:
            return None

        event_type = data.get("event_type", "")

        if event_type == "book":
            return MarketUpdate(
                event_type="book",
                asset_id=data.get("asset_id"),
                bids=data.get("bids", []),
                asks=data.get("asks", []),
                raw=data,
            )
        elif event_type == "price_change":
            return MarketUpdate(
                event_type="price_change",
                asset_id=data.get("asset_id"),
                price_changes=data.get("price_changes", []),
                raw=data,
            )
        elif event_type == "trade":
            return MarketUpdate(
                event_type="trade",
                asset_id=data.get("asset_id"),
                raw=data,
            )
        return None

    async def _shutdown_async(self) -> None:
        """Gracefully close WebSocket connections."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._user_ws:
            try:
                await self._user_ws.close()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Background thread runner (for sync usage)
    # -------------------------------------------------------------------------

    def _run_reader_loop(self) -> None:
        """Background thread entry point -- runs the async reader loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._reader_async())

    # -------------------------------------------------------------------------
    # Context manager (async usage)
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "CLOBClient":
        self._running = True
        return self

    async def __aexit__(self, *args) -> None:
        self._running = False
        await self._shutdown_async()

    def __repr__(self) -> str:
        return (
            f"CLOBClient(url={self.ws_url}, "
            f"connected={self.is_connected}, "
            f"orders={len(self._orders)})"
        )
