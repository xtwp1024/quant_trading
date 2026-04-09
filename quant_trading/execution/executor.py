"""Executor - Trade execution module"""

import time
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from quant_trading.connectors.binance_rest import BinanceRESTClient


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """订单"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: int = 0
    updated_at: int = 0
    exchange_order_id: Optional[int] = None  # 交易所订单ID


@dataclass
class Trade:
    """成交"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float = 0.0
    timestamp: int = 0


class Executor:
    """交易执行器

    统一执行层：实盘连接 Binance，模拟盘走本地撮合。
    管道: place_order → _send_order → 轮询状态 → _on_fill 成交回报
    可选接入 RiskManager，形成 risk ↔ execution 闭环。
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        test_mode: bool = True,
        poll_interval: float = 1.0,
        risk_manager: Any = None,
    ):
        """
        初始化执行器

        Args:
            api_key: Binance API 密钥
            api_secret: Binance API 密钥密码
            test_mode: 是否为测试模式（True=模拟撮合，False=实盘 Binance）
            poll_interval: 订单状态轮询间隔（秒）
            risk_manager: 可选 RiskManager 实例，接入后成交自动触发风控更新
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_mode = test_mode
        self.poll_interval = poll_interval
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self._order_counter = 0
        self.risk_manager = risk_manager  # 可选的风控管理器

        # 实盘 Binance 连接（test_mode=False 时激活）
        self._binance: Optional[BinanceRESTClient] = None
        if not test_mode and api_key and api_secret:
            self._binance = BinanceRESTClient(api_key, api_secret)

    def _generate_order_id(self) -> str:
        """生成订单ID"""
        self._order_counter += 1
        return f"ORD_{int(time.time() * 1000)}_{self._order_counter}"

    def _generate_trade_id(self) -> str:
        """生成成交ID"""
        return f"TRD_{int(time.time() * 1000)}_{len(self.trades) + 1}"

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None
    ) -> Order:
        """
        下单（如果接入了 risk_manager，则在提交前进行风控检查）

        Args:
            symbol: 交易对
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单必需）

        Returns:
            订单对象

        Raises:
            PermissionError: 风控检查未通过
        """
        # gap #3 核心修复：风控检查 → 订单放行前拦截
        if self.risk_manager is not None:
            allowed = self.risk_manager.check_trade_allowed(symbol)
            if not allowed["allowed"]:
                raise PermissionError(f"风控拦截: {allowed['reason']}")

        order_id = self._generate_order_id()
        now = int(time.time() * 1000)

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=now,
            updated_at=now
        )

        self.orders[order_id] = order

        if self.test_mode:
            # 测试模式：模拟立即成交
            await self._simulate_fill(order)
        else:
            # 实盘模式：调用 Binance API
            await self._send_order(order)
            # 后台轮询订单状态直到成交/取消
            asyncio.create_task(self._poll_order_status(order))

        return order

    async def _simulate_fill(self, order: Order) -> None:
        """模拟成交（同时触发风控更新）"""
        now = int(time.time() * 1000)
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = order.price or 100.0  # 假设价格
        order.updated_at = now

        trade = Trade(
            id=self._generate_trade_id(),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.avg_fill_price,
            timestamp=now
        )
        self.trades.append(trade)

        # gap #3 修复：模拟成交同样触发风控更新
        if self.risk_manager is not None:
            side_sign = 1 if order.side == OrderSide.BUY else -1
            trade_record = {
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.filled_quantity,
                "price": order.avg_fill_price,
                "value": order.filled_quantity * order.avg_fill_price * side_sign,
                "pnl": 0.0,
            }
            self.risk_manager.record_trade(trade_record)

    async def _send_order(self, order: Order) -> None:
        """发送真实订单到 Binance"""
        if self._binance is None:
            raise RuntimeError("Binance 客户端未初始化，请确认 api_key/api_secret 已传入且 test_mode=False")

        # 映射 OrderType → Binance order type
        type_map = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP",
            OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
        }
        binance_type = type_map.get(order.type, "LIMIT")
        side_map = {OrderSide.BUY: "BUY", OrderSide.SELL: "SELL"}

        resp = self._binance.place_order(
            symbol=order.symbol.upper(),
            side=side_map[order.side],
            order_type=binance_type,
            quantity=order.quantity,
            price=order.price,
        )

        # 保存交易所返回的 orderId，用于后续查询
        order.exchange_order_id = resp.get("orderId")
        order.status = OrderStatus.PENDING
        order.updated_at = int(time.time() * 1000)

    async def _poll_order_status(self, order: Order) -> None:
        """后台轮询订单状态，直到 FILLED / CANCELLED / REJECTED"""
        if self._binance is None or order.exchange_order_id is None:
            return

        while True:
            await asyncio.sleep(self.poll_interval)

            if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                break

            try:
                resp = self._binance.get_order(order.symbol.upper(), order.exchange_order_id)
            except Exception:
                continue

            status_map = {
                "NEW": OrderStatus.PENDING,
                "PARTIALLY_FILLED": OrderStatus.PARTIAL,
                "FILLED": OrderStatus.FILLED,
                "CANCELED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "EXPIRED": OrderStatus.CANCELLED,
            }
            new_status = status_map.get(resp.get("status", ""), OrderStatus.PENDING)

            if new_status != order.status:
                order.status = new_status
                order.updated_at = int(time.time() * 1000)

                # 处理成交
                if new_status == OrderStatus.FILLED:
                    self._on_fill(order, resp)
                elif new_status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                    break

    def _on_fill(self, order: Order, resp: dict) -> None:
        """处理成交回报，同时通知 RiskManager 更新暴露度"""
        executed_qty = float(resp.get("executedQty", 0))
        avg_price = float(resp.get("price", order.price or 0))

        order.filled_quantity = executed_qty
        order.avg_fill_price = avg_price

        trade = Trade(
            id=self._generate_trade_id(),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=executed_qty,
            price=avg_price,
            timestamp=int(time.time() * 1000),
        )
        self.trades.append(trade)

        # gap #3 修复：成交回报 → RiskManager.record_trade() → 更新暴露度
        if self.risk_manager is not None:
            side_sign = 1 if order.side == OrderSide.BUY else -1
            trade_record = {
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": executed_qty,
                "price": avg_price,
                "value": executed_qty * avg_price * side_sign,
                "pnl": 0.0,  # 平仓时由策略层计算
            }
            self.risk_manager.record_trade(trade_record)

    async def cancel_order(self, order_id: str) -> bool:
        """
        取消订单

        Args:
            order_id: 订单ID

        Returns:
            是否成功
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return False

        if self.test_mode:
            order.status = OrderStatus.CANCELLED
            order.updated_at = int(time.time() * 1000)
            return True
        else:
            return await self._cancel_order(order)

    async def _cancel_order(self, order: Order) -> bool:
        """通过 Binance 取消订单"""
        if self._binance is None or order.exchange_order_id is None:
            return False
        try:
            self._binance.cancel_order(order.symbol.upper(), order.exchange_order_id)
            order.status = OrderStatus.CANCELLED
            order.updated_at = int(time.time() * 1000)
            return True
        except Exception:
            return False

    async def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单信息"""
        return self.orders.get(order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未成交订单"""
        orders = [o for o in self.orders.values() if o.status == OrderStatus.PENDING]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """获取成交历史"""
        if symbol:
            return [t for t in self.trades if t.symbol == symbol]
        return self.trades

    def get_account_balance(self) -> Dict[str, float]:
        """获取账户余额（模拟）"""
        return {
            "USDT": 10000.0,
            "BTC": 0.0,
            "ETH": 0.0
        }


# 导出主要类
__all__ = ["Executor", "Order", "Trade", "OrderType", "OrderSide", "OrderStatus"]
