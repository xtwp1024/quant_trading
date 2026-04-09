"""
PolymarketMarketMaker — Polymarket CLOB Market Maker
=====================================================
Absorbed from: D:/Hive/Data/trading_repos/polymarket-market-maker-bot/

生产级Polymarket CLOB做市商:
- 实时WebSocket订单簿
- 库存风险管理
- 动态价差调整
- Prometheus监控

Note: Internal HTTP/WebSocket calls are async (httpx/aiohttp).
Public API is synchronous for ease of integration with the quant framework.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import websockets

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Prometheus Metrics (singleton, thread-safe)
# ---------------------------------------------------------------------------

_pm_orders_placed = Counter(
    "pm_mm_orders_placed_total", "Total orders placed", ["side", "outcome"]
)
_pm_orders_filled = Counter(
    "pm_mm_orders_filled_total", "Total orders filled", ["side", "outcome"]
)
_pm_orders_cancelled = Counter("pm_mm_orders_cancelled_total", "Total orders cancelled")
_pm_inventory = Gauge("pm_mm_inventory", "Current inventory positions", ["type"])
_pm_exposure = Gauge("pm_mm_exposure_usd", "Current net exposure in USD")
_pm_spread = Gauge("pm_mm_spread_bps", "Current spread in basis points")
_pm_profit = Gauge("pm_mm_profit_usd", "Cumulative profit in USD")
_pm_quote_latency = Histogram(
    "pm_mm_quote_latency_ms",
    "Quote generation latency (ms)",
    buckets=[10, 50, 100, 250, 500, 1000],
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Inventory:
    """Inventory positions for YES/NO outcomes.

    仓位数据类，用于追踪YES/NO仓位的盈亏和净暴露。
    """
    yes_position: float = 0.0
    no_position: float = 0.0
    net_exposure_usd: float = 0.0
    total_value_usd: float = 0.0

    def update(self, yes_delta: float, no_delta: float, price: float) -> None:
        """更新YES/NO仓位及美元价值。"""
        self.yes_position += yes_delta
        self.no_position += no_delta
        yes_value = self.yes_position * price
        no_value = self.no_position * (1.0 - price)
        self.net_exposure_usd = yes_value - no_value
        self.total_value_usd = yes_value + no_value

    def get_skew(self) -> float:
        """返回当前仓位偏斜度 (0-1)。"""
        total = abs(self.yes_position) + abs(self.no_position)
        if total == 0:
            return 0.0
        return abs(self.net_exposure_usd) / self.total_value_usd if self.total_value_usd > 0 else 0.0

    def is_balanced(self, max_skew: float = 0.3) -> bool:
        return self.get_skew() <= max_skew


@dataclass
class Quote:
    """单一报价结构 (bid 或 ask)。"""
    side: str          # "BUY" or "SELL"
    price: float
    size: float
    market: str
    token_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "side": self.side,
            "size": str(self.size),
            "price": str(self.price),
            "token_id": self.token_id,
        }


# ---------------------------------------------------------------------------
# Internal Async Helpers
# ---------------------------------------------------------------------------

async def _async_get_orderbook(base_url: str, market_id: str) -> dict[str, Any]:
    """异步获取Polymarket订单簿 REST。"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{base_url}/book", params={"market": market_id})
        resp.raise_for_status()
        return resp.json()


async def _async_get_market_info(base_url: str, market_id: str) -> dict[str, Any]:
    """异步获取市场信息。"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{base_url}/markets/{market_id}")
        resp.raise_for_status()
        return resp.json()


async def _async_ws_connect(ws_url: str, market_id: str, orderbook_cache: dict) -> None:
    """异步WebSocket连接 + 订阅订单簿更新到本地缓存。"""
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({
            "type": "subscribe",
            "channel": "l2_book",
            "market": market_id,
        }))
        async for msg in ws:
            data = json.loads(msg)
            if data.get("type") == "l2_book_update" and data.get("market") == market_id:
                book = data.get("book", {})
                orderbook_cache["bids"] = book.get("bids", [])
                orderbook_cache["asks"] = book.get("asks", [])


# ---------------------------------------------------------------------------
# PolymarketMarketMaker
# ---------------------------------------------------------------------------

class PolymarketMarketMaker:
    """Polymarket CLOB Market Maker.

    Polymarket CLOB做市商.

    特点:
    - 实时WebSocket订单簿 (l2_book_update)
    - 库存风险管理 (InventoryManager)
    - 动态价差调整 (QuoteEngine)
    - Prometheus监控指标

    Args:
        api_key: API密钥 (目前仅作占位，Polymarket使用ETH签名认证)
        min_spread: 最小价差 (比例，如0.01=1%)
        max_position: 最大仓位 (USD)
        target_inventory: 目标库存偏向 (0=完全中性)

    Attributes:
        order_book: 当前订单簿缓存 {"bids": [...], "asks": [...]}
        open_orders: 当前挂单 {order_id: order_dict}
        inventory: Inventory实例
    """

    def __init__(
        self,
        api_key: str = "",
        min_spread: float = 0.01,
        max_position: float = 1000.0,
        target_inventory: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.min_spread = min_spread          # 最小价差 (比例)
        self.max_position = max_position     # 最大仓位 USD
        self.target_inventory = target_inventory

        # REST/WebSocket URLs
        self._api_url = "https://clob.polymarket.com"
        self._ws_url = "wss://clob-ws.polymarket.com"

        # Internal state
        self._order_book: dict[str, list] = {"bids": [], "asks": []}
        self._open_orders: dict[str, dict[str, Any]] = {}
        self._inventory = Inventory()
        self._running = False
        self._ws_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Config
        self._cancel_replace_interval_ms = 500
        self._order_lifetime_ms = 3000
        self._default_size = 100.0
        self._max_exposure_usd = 10000.0
        self._min_exposure_usd = -10000.0
        self._max_position_size_usd = 5000.0
        self._inventory_skew_limit = 0.3
        self._stop_loss_pct = 0.10
        self._auto_redeem_enabled = True
        self._redeem_threshold_usd = 1.0
        self._auto_close_enabled = False
        self._close_spread_threshold_bps = 50
        self._quote_refresh_rate_ms = 1000
        self._batch_cancellations = True

        # Stats
        self._last_quote_time = 0.0
        self._cumulative_profit_usd = 0.0

    # ------------------------------------------------------------------
    # Connection management / 连接管理
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """连接Polymarket WebSocket.

        在独立线程中启动asyncio事件循环来处理WebSocket订阅。
        Connect to the Polymarket WebSocket feed in a background thread.
        """
        if self._running:
            return
        self._running = True

        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._ws_main())

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    async def _ws_main(self) -> None:
        """WebSocket主循环 (async)。"""
        market_id = getattr(self, "_market_id", "unknown")
        try:
            async with websockets.connect(self._ws_url) as ws:
                await ws.send(json.dumps({
                    "type": "subscribe",
                    "channel": "l2_book",
                    "market": market_id,
                }))
                while self._running:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(msg)
                        await self._handle_ws_message(data)
                    except asyncio.TimeoutError:
                        continue
        except Exception:
            pass

    async def _handle_ws_message(self, data: dict[str, Any]) -> None:
        """处理WebSocket消息并更新本地订单簿缓存。"""
        msg_type = data.get("type", "")
        if msg_type == "l2_book_update":
            book = data.get("book", {})
            with self._lock:
                self._order_book["bids"] = book.get("bids", self._order_book["bids"])
                self._order_book["asks"] = book.get("asks", self._order_book["asks"])

    def disconnect(self) -> None:
        """断开连接，停止WebSocket并清理资源。"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None

    # ------------------------------------------------------------------
    # Order book / 订单簿
    # ------------------------------------------------------------------

    def update_order_book(self, bids: list, asks: list) -> None:
        """更新订单簿缓存 (手动模式，不走WebSocket时使用).

        Args:
            bids: 买单列表 [[price, size], ...]
            asks: 卖单列表 [[price, size], ...]
        """
        with self._lock:
            self._order_book["bids"] = bids
            self._order_book["asks"] = asks

    async def _fetch_orderbook_via_rest(self, market_id: str) -> dict[str, Any]:
        """通过REST API获取订单簿并更新缓存。"""
        book = await _async_get_orderbook(self._api_url, market_id)
        with self._lock:
            self._order_book["bids"] = book.get("bids", [])
            self._order_book["asks"] = book.get("asks", [])
        return book

    # ------------------------------------------------------------------
    # Quote computation / 报价引擎
    # ------------------------------------------------------------------

    def compute_quotes(self, market_id: str) -> tuple[dict, dict]:
        """计算当前报价 (bid, ask).

        根据订单簿最佳买卖价和库存状态计算双边报价。

        Args:
            market_id: Polymarket市场ID

        Returns:
            (bid_order, ask_order): 买方报价和卖方报价的字典，
            如无可行报价则对应位置为空dict {}

        Example:
            >>> bid, ask = mm.compute_quotes("0x1234...")
            >>> print(bid)  # {"side": "BUY", "price": 0.495, "size": 50.0, ...}
        """
        t0 = time.perf_counter()

        with self._lock:
            bids = self._order_book.get("bids", [])
            asks = self._order_book.get("asks", [])

        if not bids or not asks:
            return {}, {}

        try:
            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 1.0
        except (ValueError, IndexError):
            return {}, {}

        if best_bid <= 0 or best_ask <= 1:
            return {}, {}

        mid_price = (best_bid + best_ask) / 2.0
        spread_bps = int(self.min_spread * 10000)

        bid_price = round(mid_price * (1 - spread_bps / 10000), 4)
        ask_price = round(mid_price * (1 + spread_bps / 10000), 4)

        yes_size = self._get_quote_size_yes(mid_price)
        no_size = self._get_quote_size_no(mid_price)

        bid_order: dict[str, Any] = {}
        ask_order: dict[str, Any] = {}

        if self._can_quote_yes(yes_size):
            bid_order = {
                "side": "BUY",
                "price": bid_price,
                "size": yes_size,
                "market": market_id,
                "token_id": "",   # 需通过market_info获取yes_token_id
            }

        if self._can_quote_no(no_size):
            ask_order = {
                "side": "BUY",    # Polymarket SELL = 卖出NO
                "price": 1.0 - ask_price,
                "size": no_size,
                "market": market_id,
                "token_id": "",
            }

        # 记录延迟
        latency_ms = (time.perf_counter() - t0) * 1000
        _pm_quote_latency.observe(latency_ms)

        return bid_order, ask_order

    # ------------------------------------------------------------------
    # Inventory management / 库存管理
    # ------------------------------------------------------------------

    def manage_inventory(self) -> list[dict]:
        """库存管理 — 返回需要调整的订单列表.

        检查当前库存偏斜度，如果超过阈值则返回需要平仓的订单。

        Returns:
            需平仓订单列表，每项包含订单ID和方向
        """
        adjustments: list[dict] = []
        skew = self._inventory.get_skew()

        if skew > self._inventory_skew_limit:
            for oid, order in list(self._open_orders.items()):
                side = order.get("side", "")
                if skew > 0 and side == "BUY" and self._inventory.yes_position > 0:
                    adjustments.append({"order_id": oid, "action": "cancel", "reason": "inventory_skew"})
                elif skew < 0 and side == "BUY" and self._inventory.no_position > 0:
                    adjustments.append({"order_id": oid, "action": "cancel", "reason": "inventory_skew"})

        # 止损检查
        if abs(self._inventory.net_exposure_usd) > self.max_position * (1 + self._stop_loss_pct):
            for oid, order in list(self._open_orders.items()):
                adjustments.append({"order_id": oid, "action": "cancel", "reason": "stop_loss"})

        return adjustments

    def _get_quote_size_yes(self, base_size: float, price: float = 0.5) -> float:
        """计算YES仓位报价大小。"""
        if not self._can_quote_yes(base_size):
            max_size = max(0.0, self._max_exposure_usd - self._inventory.net_exposure_usd)
            return min(base_size, max_size / price if price > 0 else 0)
        if self._inventory.net_exposure_usd > self.target_inventory:
            return base_size * 0.5
        return base_size

    def _get_quote_size_no(self, base_size: float, price: float = 0.5) -> float:
        """计算NO仓位报价大小。"""
        if not self._can_quote_no(base_size):
            max_size = max(0.0, abs(self._min_exposure_usd - self._inventory.net_exposure_usd))
            return min(base_size, max_size / (1.0 - price) if price < 1 else 0)
        if self._inventory.net_exposure_usd < self.target_inventory:
            return base_size * 0.5
        return base_size

    def _can_quote_yes(self, size_usd: float) -> bool:
        return (self._inventory.net_exposure_usd + size_usd) <= self._max_exposure_usd

    def _can_quote_no(self, size_usd: float) -> bool:
        return (self._inventory.net_exposure_usd - size_usd) >= self._min_exposure_usd

    # ------------------------------------------------------------------
    # Risk limits / 风险控制
    # ------------------------------------------------------------------

    def check_risk_limits(self) -> tuple[bool, str]:
        """检查风险限制.

        验证:
        1. 仓位大小限制
        2. 暴露度限制
        3. 库存偏斜限制

        Returns:
            (is_ok, reason): 是否通过风险检查及原因
        """
        # 仓位大小
        if abs(self._inventory.net_exposure_usd) > self._max_position_size_usd:
            return False, f"Position size {abs(self._inventory.net_exposure_usd):.2f} exceeds limit"

        # 暴露度
        if self._inventory.net_exposure_usd > self._max_exposure_usd:
            return False, f"Exposure {self._inventory.net_exposure_usd:.2f} above max"
        if self._inventory.net_exposure_usd < self._min_exposure_usd:
            return False, f"Exposure {self._inventory.net_exposure_usd:.2f} below min"

        # 库存偏斜
        skew = self._inventory.get_skew()
        if skew > self._inventory_skew_limit:
            return False, f"Inventory skew {skew:.3f} exceeds limit {self._inventory_skew_limit}"

        return True, "OK"

    def should_stop_trading(self) -> bool:
        """是否应停止交易 (接近风险上限)。"""
        return abs(self._inventory.net_exposure_usd) > abs(self._max_exposure_usd) * 0.9

    # ------------------------------------------------------------------
    # Main loop / 主循环
    # ------------------------------------------------------------------

    def run(self) -> None:
        """主循环 — 在当前线程中同步运行 (仅用于简单场景).

        推荐使用 connect()+disconnect() 在后台线程运行，
        并通过 compute_quotes() 和 manage_inventory() 手动驱动。

        This is a synchronous convenience wrapper.
        For production use, call connect() separately and drive quotes manually.
        """
        self.connect()
        try:
            while self._running:
                time.sleep(self._quote_refresh_rate_ms / 1000.0)
        except KeyboardInterrupt:
            self.disconnect()

    # ------------------------------------------------------------------
    # Metrics / Prometheus监控
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """返回Prometheus格式的当前指标字典.

        Returns:
            包含各监控指标的字典:
            - orders_placed_total (Counter, keyed by side/outcome)
            - orders_filled_total
            - orders_cancelled_total
            - inventory_yes, inventory_no (Gauge)
            - exposure_usd (Gauge)
            - spread_bps (Gauge)
            - profit_usd (Gauge)
        """
        skew = self._inventory.get_skew()
        spread_bps = 0.0
        with self._lock:
            bids = self._order_book.get("bids", [])
            asks = self._order_book.get("asks", [])
        if bids and asks:
            try:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                if best_bid > 0:
                    spread_bps = (best_ask - best_bid) / best_bid * 10000
            except (ValueError, IndexError):
                pass

        return {
            "inventory_yes": self._inventory.yes_position,
            "inventory_no": self._inventory.no_position,
            "net_exposure_usd": self._inventory.net_exposure_usd,
            "total_value_usd": self._inventory.total_value_usd,
            "inventory_skew": skew,
            "spread_bps": spread_bps,
            "cumulative_profit_usd": self._cumulative_profit_usd,
            "open_orders_count": len(self._open_orders),
        }

    def record_inventory_change(self, yes_delta: float, no_delta: float, price: float) -> None:
        """记录仓位变化并更新Prometheus指标。

        Args:
            yes_delta: YES仓位变化量
            no_delta: NO仓位变化量
            price: 当前市场价格
        """
        self._inventory.update(yes_delta, no_delta, price)
        _pm_inventory.labels(type="yes").set(self._inventory.yes_position)
        _pm_inventory.labels(type="no").set(self._inventory.no_position)
        _pm_exposure.set(self._inventory.net_exposure_usd)

    def record_order_filled(self, side: str, outcome: str) -> None:
        """记录订单成交事件。"""
        _pm_orders_filled.labels(side=side, outcome=outcome).inc()

    def record_order_placed(self, side: str, outcome: str) -> None:
        """记录订单提交事件。"""
        _pm_orders_placed.labels(side=side, outcome=outcome).inc()

    def record_order_cancelled(self) -> None:
        """记录订单取消事件。"""
        _pm_orders_cancelled.inc()

    def record_profit(self, profit_usd: float) -> None:
        """记录已实现盈亏。"""
        self._cumulative_profit_usd += profit_usd
        _pm_profit.set(self._cumulative_profit_usd)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def set_market_id(self, market_id: str) -> None:
        """设置当前交易的市场ID。"""
        self._market_id = market_id

    def configure(
        self,
        max_exposure_usd: float = 10000.0,
        min_exposure_usd: float = -10000.0,
        default_size: float = 100.0,
        min_spread_bps: int = 10,
        cancel_replace_interval_ms: int = 500,
        order_lifetime_ms: int = 3000,
        max_position_size_usd: float = 5000.0,
        inventory_skew_limit: float = 0.3,
        stop_loss_pct: float = 0.10,
        auto_redeem_enabled: bool = True,
        redeem_threshold_usd: float = 1.0,
        auto_close_enabled: bool = False,
        close_spread_threshold_bps: int = 50,
        quote_refresh_rate_ms: int = 1000,
        batch_cancellations: bool = True,
        api_url: str | None = None,
        ws_url: str | None = None,
    ) -> None:
        """批量配置做市商参数。

        Args:
            max_exposure_usd: 最大净暴露 (多头方向)
            min_exposure_usd: 最小净暴露 (空头方向)
            default_size: 默认报价大小 (USD)
            min_spread_bps: 最小价差 (basis points)
            cancel_replace_interval_ms: 取消/刷新周期 (ms)
            order_lifetime_ms: 订单最大生命周期 (ms)
            max_position_size_usd: 单笔最大仓位
            inventory_skew_limit: 库存偏斜上限 (0-1)
            stop_loss_pct: 止损百分比
            auto_redeem_enabled: 是否自动赎回
            redeem_threshold_usd: 自动赎回门槛
            auto_close_enabled: 是否自动关闭
            close_spread_threshold_bps: 自动关闭价差阈值
            quote_refresh_rate_ms: 报价刷新间隔 (ms)
            batch_cancellations: 是否批量取消订单
            api_url: Polymarket API URL (覆盖默认)
            ws_url: Polymarket WebSocket URL (覆盖默认)
        """
        self._max_exposure_usd = max_exposure_usd
        self._min_exposure_usd = min_exposure_usd
        self._default_size = default_size
        self.min_spread = min_spread_bps / 10000.0
        self._cancel_replace_interval_ms = cancel_replace_interval_ms
        self._order_lifetime_ms = order_lifetime_ms
        self._max_position_size_usd = max_position_size_usd
        self._inventory_skew_limit = inventory_skew_limit
        self._stop_loss_pct = stop_loss_pct
        self._auto_redeem_enabled = auto_redeem_enabled
        self._redeem_threshold_usd = redeem_threshold_usd
        self._auto_close_enabled = auto_close_enabled
        self._close_spread_threshold_bps = close_spread_threshold_bps
        self._quote_refresh_rate_ms = quote_refresh_rate_ms
        self._batch_cancellations = batch_cancellations
        if api_url:
            self._api_url = api_url
        if ws_url:
            self._ws_url = ws_url

    def __repr__(self) -> str:
        return (
            f"PolymarketMarketMaker("
            f"min_spread={self.min_spread:.4f}, "
            f"max_position={self.max_position}, "
            f"inventory_skew={self._inventory.get_skew():.3f})"
        )


__all__ = [
    "PolymarketMarketMaker",
    "Inventory",
    "Quote",
]
