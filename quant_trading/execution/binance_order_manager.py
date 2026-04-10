"""
Binance Order Manager - Order Tracking and Fill Monitoring

订单管理器 - 订单跟踪与成交监控

Manages order lifecycle with:
- Order tracking and state management
- Fill monitoring with polling
- Partial fill handling
- Order timeout management
- Automatic retry on transient failures

Usage:
    >>> from quant_trading.execution.binance_order_manager import BinanceOrderManager
    >>> from quant_trading.connectors.binance_trading import BinanceTradingAdapter
    >>> adapter = BinanceTradingAdapter(api_key="...", api_secret="...")
    >>> manager = BinanceOrderManager(adapter)
    >>> order = await manager.place_order("BTCUSDT", "BUY", "LIMIT", 0.001, 50000)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Optional
from enum import Enum

from quant_trading.connectors.binance_trading import (
    BinanceTradingAdapter,
    OrderStatus,
    OrderSide,
    OrderType,
    TradingOrder,
)


class OrderState(Enum):
    """订单状态机"""
    PENDING = "pending"           # 待发送
    SUBMITTED = "submitted"       # 已提交
    PARTIAL = "partial"           # 部分成交
    FILLED = "filled"             # 完全成交
    CANCELLED = "cancelled"      # 已取消
    TIMEOUT = "timeout"           # 超时
    REJECTED = "rejected"        # 被拒绝
    FAILED = "failed"            # 失败


@dataclass
class TrackedOrder:
    """追踪订单"""
    order: TradingOrder
    state: OrderState = OrderState.PENDING
    submit_time: float = 0.0
    last_update_time: float = 0.0
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fill_history: List[Dict] = field(default_factory=list)
    retry_count: int = 0
    error_message: Optional[str] = None
    callbacks: List[Callable] = field(default_factory=list)


@dataclass
class Fill:
    """成交记录"""
    order_id: str
    fill_id: int
    quantity: float
    price: float
    commission: float
    commission_asset: str
    timestamp: int


class OrderTimeoutError(Exception):
    """订单超时"""
    pass


class MaxRetriesExceededError(Exception):
    """超过最大重试次数"""
    pass


class BinanceOrderManager:
    """
    Binance订单管理器

    提供完整的订单生命周期管理：
    - 自动状态跟踪
    - 成交监控（轮询/WebSocket）
    - 部分成交处理
    - 订单超时管理
    - 失败自动重试

    Args:
        adapter: BinanceTradingAdapter实例
        poll_interval: 订单状态轮询间隔（秒）
        order_timeout: 订单超时时间（秒），0表示不超时
        max_retries: 最大重试次数
    """

    def __init__(
        self,
        adapter: BinanceTradingAdapter,
        poll_interval: float = 1.0,
        order_timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self._adapter = adapter
        self._poll_interval = poll_interval
        self._order_timeout = order_timeout
        self._max_retries = max_retries

        self._orders: Dict[str, TrackedOrder] = {}  # client_order_id -> TrackedOrder
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._logger = logging.getLogger("BinanceOrderManager")

        # Callbacks
        self._on_fill_callback: Optional[Callable[[str, Fill], None]] = None
        self._on_order_update_callback: Optional[Callable[[TrackedOrder], None]] = None

    @property
    def tracked_orders(self) -> List[TrackedOrder]:
        """所有追踪中的订单"""
        return list(self._orders.values())

    @property
    def pending_orders(self) -> List[TrackedOrder]:
        """待成交订单"""
        return [
            o for o in self._orders.values()
            if o.state in [OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIAL]
        ]

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running

    # ====================
    # Callbacks / 回调设置
    # ====================

    def set_fill_callback(self, callback: Callable[[str, Fill], None]) -> None:
        """设置成交回调"""
        self._on_fill_callback = callback

    def set_order_update_callback(self, callback: Callable[[TrackedOrder], None]) -> None:
        """设置订单更新回调"""
        self._on_order_update_callback = callback

    def add_order_callback(self, client_order_id: str, callback: Callable[[TrackedOrder], None]) -> None:
        """为特定订单添加回调"""
        if client_order_id in self._orders:
            self._orders[client_order_id].callbacks.append(callback)

    # ====================
    # Order Placement / 下单
    # ====================

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        wait_for_fill: bool = True,
        timeout: Optional[float] = None,
    ) -> TrackedOrder:
        """
        下单并追踪

        Args:
            symbol: 交易对
            side: 方向
            order_type: 订单类型
            quantity: 数量
            price: 价格
            stop_price: 止损价格
            client_order_id: 客户端订单ID
            wait_for_fill: 是否等待成交
            timeout: 等待成交超时时间

        Returns:
            TrackedOrder对象

        Example:
            >>> order = await manager.place_order(
            ...     "BTCUSDT", OrderSide.BUY, OrderType.LIMIT,
            ...     0.001, price=50000
            ... )
            >>> # 或等待成交
            >>> order = await manager.place_order(
            ...     "BTCUSDT", OrderSide.BUY, OrderType.MARKET,
            ...     0.001, wait_for_fill=True, timeout=30
            ... )
        """
        # Generate order ID if not provided
        if client_order_id is None:
            client_order_id = f"ORD_{int(time.time() * 1000)}"

        # Create tracked order
        tracked = TrackedOrder(
            order=TradingOrder(
                client_order_id=client_order_id,
                symbol=symbol.upper(),
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
            ),
            state=OrderState.PENDING,
            submit_time=time.time(),
            last_update_time=time.time(),
        )

        self._orders[client_order_id] = tracked
        tracked.state = OrderState.SUBMITTED

        # Submit order with retry
        try:
            result = await self._submit_with_retry(tracked, price, stop_price)
            tracked.order = result
            tracked.last_update_time = time.time()

            self._logger.info(f"Order submitted: {client_order_id}")

            # Start monitoring if not running
            if not self._running:
                await self.start()

            # Wait for fill if requested
            if wait_for_fill:
                await self.wait_for_fill(client_order_id, timeout or self._order_timeout)

        except Exception as e:
            tracked.state = OrderState.FAILED
            tracked.error_message = str(e)
            tracked.last_update_time = time.time()
            self._logger.error(f"Order failed: {client_order_id} - {e}")
            raise

        # Trigger callbacks
        self._notify_order_update(tracked)
        return tracked

    async def _submit_with_retry(
        self,
        tracked: TrackedOrder,
        price: Optional[float],
        stop_price: Optional[float],
    ) -> TradingOrder:
        """带重试的订单提交"""
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                tracked.retry_count = attempt
                result = await self._adapter.place_order(
                    symbol=tracked.order.symbol,
                    side=tracked.order.side,
                    order_type=tracked.order.type,
                    quantity=tracked.order.quantity,
                    price=price,
                    stop_price=stop_price,
                    client_order_id=tracked.order.client_order_id,
                )
                return result

            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self._logger.warning(
                        f"Order retry {attempt + 1}/{self._max_retries} "
                        f"for {tracked.order.client_order_id} after {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self._logger.error(f"Order failed after {self._max_retries} retries: {e}")

        raise last_error

    # ====================
    # Market Orders / 市价单
    # ====================

    async def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        wait_for_fill: bool = True,
        timeout: Optional[float] = None,
    ) -> TrackedOrder:
        """市价单"""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            wait_for_fill=wait_for_fill,
            timeout=timeout,
        )

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        wait_for_fill: bool = True,
        timeout: Optional[float] = None,
    ) -> TrackedOrder:
        """限价单"""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            wait_for_fill=wait_for_fill,
            timeout=timeout,
        )

    async def place_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        wait_for_fill: bool = False,
    ) -> TrackedOrder:
        """止损单"""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LOSS,
            quantity=quantity,
            stop_price=stop_price,
            wait_for_fill=wait_for_fill,
        )

    # ====================
    # Order Management / 订单管理
    # ====================

    async def get_order(self, client_order_id: str) -> Optional[TrackedOrder]:
        """获取追踪订单"""
        return self._orders.get(client_order_id)

    async def cancel_order(self, client_order_id: str) -> bool:
        """
        取消订单

        Returns:
            成功返回True
        """
        tracked = self._orders.get(client_order_id)
        if not tracked:
            return False

        if tracked.state in [OrderState.FILLED, OrderState.CANCELLED]:
            return True  # Already in terminal state

        try:
            success = await self._adapter.cancel_order(
                symbol=tracked.order.symbol,
                order_id=tracked.order.order_id,
                client_order_id=client_order_id,
            )

            if success:
                tracked.state = OrderState.CANCELLED
                tracked.last_update_time = time.time()
                self._notify_order_update(tracked)
                self._logger.info(f"Order cancelled: {client_order_id}")

            return success

        except Exception as e:
            self._logger.error(f"Cancel order failed: {client_order_id} - {e}")
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """取消所有待成交订单"""
        cancelled = 0
        for tracked in self.pending_orders:
            if symbol is None or tracked.order.symbol == symbol.upper():
                if await self.cancel_order(tracked.order.client_order_id):
                    cancelled += 1
        return cancelled

    # ====================
    # Fill Monitoring / 成交监控
    # ====================

    async def wait_for_fill(
        self,
        client_order_id: str,
        timeout: float = 60.0,
    ) -> TrackedOrder:
        """
        等待订单成交

        Args:
            client_order_id: 客户端订单ID
            timeout: 超时时间（秒）

        Returns:
            TrackedOrder对象

        Raises:
            OrderTimeoutError: 超时
        """
        tracked = self._orders.get(client_order_id)
        if not tracked:
            raise ValueError(f"Order not found: {client_order_id}")

        start_time = time.time()

        while tracked.state in [OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIAL]:
            if time.time() - start_time > timeout:
                tracked.state = OrderState.TIMEOUT
                tracked.last_update_time = time.time()
                self._notify_order_update(tracked)
                raise OrderTimeoutError(
                    f"Order {client_order_id} timed out after {timeout}s "
                    f"(filled: {tracked.filled_quantity}/{tracked.order.quantity})"
                )

            await asyncio.sleep(self._poll_interval)
            await self._check_order_status(client_order_id)

        return tracked

    async def _check_order_status(self, client_order_id: str) -> Optional[TrackedOrder]:
        """检查订单状态"""
        tracked = self._orders.get(client_order_id)
        if not tracked:
            return None

        # Skip terminal states
        if tracked.state in [OrderState.FILLED, OrderState.CANCELLED, OrderState.TIMEOUT]:
            return tracked

        try:
            order = await self._adapter.get_order(
                symbol=tracked.order.symbol,
                client_order_id=client_order_id,
            )

            if order is None:
                return tracked

            # Update tracked order
            old_filled_qty = tracked.filled_quantity
            tracked.order.status = order.status
            tracked.filled_quantity = order.filled_quantity
            tracked.avg_fill_price = order.avg_fill_price
            tracked.last_update_time = time.time()

            # Detect partial fill
            if order.filled_quantity > old_filled_qty and tracked.state != OrderState.PARTIAL:
                tracked.state = OrderState.PARTIAL
                fill_qty = order.filled_quantity - old_filled_qty
                fill = Fill(
                    order_id=client_order_id,
                    fill_id=int(time.time() * 1000),
                    quantity=fill_qty,
                    price=order.avg_fill_price,
                    commission=0.0,
                    commission_asset="BNB",
                    timestamp=int(time.time() * 1000),
                )
                tracked.fill_history.append(fill.__dict__)

                # Trigger fill callback
                if self._on_fill_callback:
                    self._on_fill_callback(client_order_id, fill)

                self._logger.info(
                    f"Partial fill: {client_order_id} - "
                    f"{fill_qty}/{tracked.order.quantity} @ {order.avg_fill_price}"
                )

            # Update state based on status
            if order.status == OrderStatus.FILLED:
                tracked.state = OrderState.FILLED
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                tracked.state = OrderState.PARTIAL

            self._notify_order_update(tracked)

        except Exception as e:
            self._logger.warning(f"Check order status failed: {client_order_id} - {e}")

        return tracked

    async def _monitor_loop(self) -> None:
        """监控循环"""
        self._logger.info("Order monitor started")

        while self._running:
            try:
                # Check all pending orders
                for tracked in list(self.pending_orders):
                    await self._check_order_status(tracked.order.client_order_id)

                    # Check timeout
                    if self._order_timeout > 0:
                        elapsed = time.time() - tracked.submit_time
                        if elapsed > self._order_timeout:
                            if tracked.state in [OrderState.PENDING, OrderState.SUBMITTED]:
                                tracked.state = OrderState.TIMEOUT
                                tracked.last_update_time = time.time()
                                self._logger.warning(
                                    f"Order timeout: {tracked.order.client_order_id} "
                                    f"after {elapsed:.1f}s"
                                )
                                self._notify_order_update(tracked)

                await asyncio.sleep(self._poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)

        self._logger.info("Order monitor stopped")

    # ====================
    # Lifecycle / 生命周期
    # ====================

    async def start(self) -> None:
        """启动订单管理器"""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """停止订单管理器"""
        if not self._running:
            return

        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    # ====================
    # Helpers / 辅助方法
    # ====================

    def _notify_order_update(self, tracked: TrackedOrder) -> None:
        """通知订单更新"""
        if self._on_order_update_callback:
            try:
                self._on_order_update_callback(tracked)
            except Exception as e:
                self._logger.error(f"Order update callback error: {e}")

        # Notify order-specific callbacks
        for callback in tracked.callbacks:
            try:
                callback(tracked)
            except Exception as e:
                self._logger.error(f"Order callback error: {e}")

    def get_order_summary(self) -> Dict[str, Any]:
        """获取订单摘要"""
        return {
            "total": len(self._orders),
            "pending": len(self.pending_orders),
            "filled": len([o for o in self._orders.values() if o.state == OrderState.FILLED]),
            "cancelled": len([o for o in self._orders.values() if o.state == OrderState.CANCELLED]),
            "failed": len([o for o in self._orders.values() if o.state == OrderState.FAILED]),
            "by_state": {
                s.value: len([o for o in self._orders.values() if o.state == s])
                for s in OrderState
            },
        }


__all__ = [
    "BinanceOrderManager",
    "OrderState",
    "TrackedOrder",
    "Fill",
    "OrderTimeoutError",
    "MaxRetriesExceededError",
]
