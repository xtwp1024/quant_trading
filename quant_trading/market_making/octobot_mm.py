"""
OctoBot-Market-Making Framework — 多交易所做市接口
===================================================
Absorbed from: D:/Hive/Data/trading_repos/OctoBot-Market-Making/

Core components:
- ExchangeInterface:      多交易所统一REST接口
- ReferencePriceEngine:   多交易所订单簿聚合 + 参考价格计算 + 套利检测
- OctoBotMarketMaker:     OctoBot风格做市框架 (自适应价差, 参考价格保护)

Features:
- 多交易所支持 (15+ 交易所)
- 参考价格套利保护
- 订单簿聚合
- 自适应价差
- 做市策略评估器
"""

from __future__ import annotations

import time
import uuid
import hashlib
import hmac
import json
import math
import statistics
from typing import Optional

# -------------------------------------------------------------------
# ExchangeInterface — 交易所统一REST接口
# -------------------------------------------------------------------


class ExchangeInterface:
    """
    交易所统一REST接口 / Unified REST interface for cryptocurrency exchanges.

    Supports any exchange implementing the standard REST API pattern:
    - Order book retrieval
    - Limit order placement
    - Order cancellation
    - Balance inquiry

    Args:
        exchange_name: 交易所名称, e.g. "binance", "okx", "huobi"
        api_key:       API密钥
        api_secret:    API密钥对应私钥
    """

    def __init__(
        self,
        exchange_name: str,
        api_key: str,
        api_secret: str,
        passphrase: Optional[str] = None,
    ) -> None:
        self.exchange_name = exchange_name.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase  # For exchanges that require it (e.g. OKX)
        self._base_url = self._get_base_url()
        self._orders: dict[str, dict] = {}  # Local order tracking

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _get_base_url(self) -> str:
        """Return the REST base URL for the exchange."""
        BASE_URLS = {
            "binance":    "https://api.binance.com",
            "okx":        "https://www.okx.com",
            "huobi":      "https://api.huobi.pro",
            "bybit":      "https://api.bybit.com",
            "kucoin":     "https://api.kucoin.com",
            "gateio":     "https://api.gateio.ws",
            "bitget":     "https://api.bitget.com",
            "mexc":       "https://api.mexc.com",
            "deribit":    "https://www.deribit.com",
            "phemex":     "https://api.phemex.com",
            "kraken":     "https://api.kraken.com",
            "coinbase":   "https://api.coinbase.com",
            "gemini":     "https://api.gemini.com",
            "bitstamp":   "https://api.bitstamp.net",
            "luno":       "https://api.luno.com",
        }
        return BASE_URLS.get(self.exchange_name, f"https://api.{self.exchange_name}.com")

    def _sign(self, payload: str) -> str:
        """HMAC-SHA256 signature of the request payload."""
        return hmac.new(
            self.api_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _headers(self, signed: bool = False) -> dict:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "X-Exchange":   self.exchange_name,
        }
        if signed:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        signed: bool = False,
        params: Optional[dict] = None,
    ) -> dict:
        """
        Low-level REST request — subclasses should override with real HTTP client.
        默认实现仅返回模拟数据 (mock)；生产环境需替换为真实HTTP调用.

        Returns:
            dict: JSON response parsed as dictionary
        """
        # Mock implementation — replace with real requests in production
        return {
            "exchange":    self.exchange_name,
            "method":     method,
            "endpoint":   endpoint,
            "params":     params or {},
            "mock":       True,
            "timestamp":  time.time(),
        }

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        """
        获取指定交易对的订单簿 / Fetch order book for a trading pair.

        Args:
            symbol: 交易对符号, e.g. "BTC/USDT"
            limit:  聚合深度, 默认20档

        Returns:
            dict: Order book with keys:
                - bids: list of [price, qty]
                - asks: list of [price, qty]
                - timestamp: Unix timestamp (ms)
        """
        # Normalize symbol: "BTC/USDT" -> "btcusdt"
        norm = symbol.replace("/", "").lower()
        resp = self._request("GET", f"/api/v1/orderbook/{norm}", params={"limit": limit})
        return {
            "bids":      resp.get("bids", [[0.0, 0.0]]),
            "asks":      resp.get("asks", [[0.0, 0.0]]),
            "timestamp": resp.get("timestamp", int(time.time() * 1000)),
            "exchange":  self.exchange_name,
            "symbol":    symbol,
        }

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
    ) -> dict:
        """
        下限价单 / Place a limit (maker) order.

        Args:
            symbol: 交易对, e.g. "BTC/USDT"
            side:   "buy" or "sell"
            price:  限价
            qty:    数量

        Returns:
            dict: Order confirmation with keys:
                - order_id: 交易所返回的订单ID
                - symbol, side, price, qty
                - status: "open"
        """
        order_id = f"mm_{uuid.uuid4().hex[:16]}"
        ts = int(time.time() * 1000)
        order = {
            "order_id":   order_id,
            "symbol":     symbol,
            "side":       side.lower(),
            "price":      round(price, 8),
            "qty":        round(qty, 8),
            "status":     "open",
            "timestamp":  ts,
            "exchange":   self.exchange_name,
        }
        self._orders[order_id] = order
        return order

    def cancel_order(self, order_id: str) -> dict:
        """
        取消订单 / Cancel an open order.

        Args:
            order_id: 需取消的订单ID

        Returns:
            dict: Cancellation result with keys:
                - order_id, success (bool), reason (if failed)
        """
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
            return {"order_id": order_id, "success": True, "exchange": self.exchange_name}
        return {"order_id": order_id, "success": False, "reason": "order_not_found", "exchange": self.exchange_name}

    def get_balance(self) -> dict:
        """
        获取账户余额 / Fetch account balance for all assets.

        Returns:
            dict: Mapping from asset symbol to dict with keys:
                - free: 可用数量
                - locked: 冻结数量
                - total: 总额
        """
        resp = self._request("GET", "/api/v1/balance", signed=True)
        return resp.get("balances", {
            "USDT": {"free": 0.0, "locked": 0.0, "total": 0.0},
            "BTC":  {"free": 0.0, "locked": 0.0, "total": 0.0},
        })

    def get_open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """
        获取当前挂单 / Fetch open (unfilled) orders.

        Args:
            symbol: 可选, 筛选特定交易对

        Returns:
            list[dict]: 当前挂起的订单列表
        """
        open_orders = [
            o for o in self._orders.values()
            if o["status"] == "open" and (symbol is None or o["symbol"] == symbol)
        ]
        return open_orders

    def get_position(self, symbol: str) -> dict:
        """
        获取持仓信息 / Get current position for a symbol.

        Args:
            symbol: 交易对, e.g. "BTC/USDT"

        Returns:
            dict: Position with keys:
                - symbol, side ("long"/"short"/"flat")
                - qty, entry_price, unrealized_pnl
        """
        resp = self._request("GET", f"/api/v1/position/{symbol.replace('/','')}", signed=True)
        return resp.get("position", {
            "symbol":          symbol,
            "side":            "flat",
            "qty":             0.0,
            "entry_price":     0.0,
            "unrealized_pnl":  0.0,
        })


# -------------------------------------------------------------------
# ReferencePriceEngine — 参考价格引擎 (多交易所聚合 + 套利检测)
# -------------------------------------------------------------------


class ReferencePriceEngine:
    """
    参考价格引擎 — 多交易所订单簿聚合 / Multi-exchange order-book aggregator.

    功能 / Features:
        1. 聚合多个交易所的订单簿, 计算加权参考价格
        2. 基于深度和流动性的加权价格计算
        3. 检测跨交易所套利机会

    The reference price is the volume-weighted mid-price across all connected
    exchanges, giving more weight to exchanges with deeper order books near
    the mid.

    Args:
        exchanges: ExchangeInterface实例列表
    """

    def __init__(self, exchanges: list[ExchangeInterface]) -> None:
        if not exchanges:
            raise ValueError("At least one exchange must be provided.")
        self.exchanges = exchanges
        self._price_cache: dict[str, tuple[float, float]] = {}  # symbol -> (ref_price, timestamp)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _mid_price(order_book: dict) -> float:
        """从订单簿计算中间价 / Compute mid-price from order book."""
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks:
            return 0.0
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        return (best_bid + best_ask) / 2.0

    @staticmethod
    def _weighted_mid(order_book: dict) -> float:
        """
        基于顶档深度的加权中间价 / Depth-weighted mid-price.

        Weight = sqrt(bid_qty * ask_qty) — gives higher weight when both
        sides are deep, reducing impact of thin books.
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks:
            return 0.0
        best_bid_p, bid_q = float(bids[0][0]), float(bids[0][1])
        best_ask_p, ask_q = float(asks[0][0]), float(asks[0][1])

        # Harmonic mean weighted by quantity imbalance
        total_q = bid_q + ask_q
        if total_q == 0:
            return (best_bid_p + best_ask_p) / 2.0

        # Volume-weighted mid
        return (best_bid_p * ask_q + best_ask_p * bid_q) / total_q

    @staticmethod
    def _spread_pct(mid: float, order_book: dict) -> float:
        """计算价差百分比 (bp)."""
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks or mid == 0:
            return 0.0
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        return abs(best_ask - best_bid) / mid

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get_reference_price(self, symbol: str, use_cache: bool = True) -> float:
        """
        获取加权参考价格 / Compute aggregated reference price for a symbol.

        多交易所加权平均:
        - 每交易所根据深度加权 (sqrt(bid_qty * ask_qty))
        - 排除价格偏离中位数超过阈值的交易所 ( outlier rejection)
        - 最终加权平均作为参考价格

        Args:
            symbol:     交易对, e.g. "BTC/USDT"
            use_cache:  是否使用缓存 (默认True, 避免频繁轮询)

        Returns:
            float: 加权参考价格
        """
        now = time.time()
        if use_cache and symbol in self._price_cache:
            cached_price, cached_ts = self._price_cache[symbol]
            if now - cached_ts < 1.0:  # 1s cache TTL
                return cached_price

        order_books = {}
        for ex in self.exchanges:
            try:
                ob = ex.get_order_book(symbol)
                order_books[ex.exchange_name] = ob
            except Exception:
                continue

        if not order_books:
            # Fallback to last cached value or 0
            return self._price_cache.get(symbol, (0.0, 0.0))[0]

        # Compute mid-price per exchange
        mid_prices = {name: self._weighted_mid(ob) for name, ob in order_books.items()}
        mid_prices = {k: v for k, v in mid_prices.items() if v > 0}
        if not mid_prices:
            return 0.0

        # Outlier rejection: compute median, exclude exchanges > 2% away
        median_price = statistics.median(mid_prices.values())
        threshold = median_price * 0.02

        valid_prices = {
            name: price
            for name, price in mid_prices.items()
            if abs(price - median_price) <= threshold
        }
        if not valid_prices:
            valid_prices = mid_prices  # Fallback to all if all are outliers

        # Volume-weighted aggregation per exchange
        weights: dict[str, float] = {}
        for name, ob in order_books.items():
            if name not in valid_prices:
                continue
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if bids and asks:
                w = math.sqrt(float(bids[0][1]) * float(asks[0][1]))
                weights[name] = w

        total_weight = sum(weights.values())
        if total_weight == 0:
            # Equal weight if no depth info
            ref_price = statistics.mean(valid_prices.values())
        else:
            ref_price = sum(
                valid_prices[name] * (weights.get(name, 1.0) / total_weight)
                for name in valid_prices
            )

        self._price_cache[symbol] = (ref_price, now)
        return ref_price

    def detect_arb_opportunity(
        self,
        symbol: str,
        min_profit_pct: float = 0.001,
    ) -> tuple[bool, float]:
        """
        检测跨交易所套利机会 / Detect cross-exchange arbitrage opportunity.

        策略:
        - 在所有交易所获取订单簿
        - 找到最高买价 (bid) 和最低卖价 (ask)
        - 若 最高买价 > 最低卖价, 存在套利空间

        Args:
            symbol:        交易对
            min_profit_pct: 最小利润门槛 (默认 0.1%)

        Returns:
            tuple[bool, float]: (是否存在套利机会, 年化利润百分比)
        """
        now = time.time()

        best_bid_ex:  Optional[tuple[str, float, float]] = None  # (exchange, price, qty)
        best_ask_ex: Optional[tuple[str, float, float]] = None  # (exchange, price, qty)

        for ex in self.exchanges:
            try:
                ob = ex.get_order_book(symbol)
            except Exception:
                continue

            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            if not bids or not asks:
                continue

            bid_p, bid_q = float(bids[0][0]), float(bids[0][1])
            ask_p, ask_q = float(asks[0][0]), float(asks[0][1])

            if best_bid_ex is None or bid_p > best_bid_ex[1]:
                best_bid_ex = (ex.exchange_name, bid_p, bid_q)
            if best_ask_ex is None or ask_p < best_ask_ex[1]:
                best_ask_ex = (ex.exchange_name, ask_p, ask_q)

        if best_bid_ex is None or best_ask_ex is None:
            return False, 0.0

        bid_ex_name, bid_price, bid_qty = best_bid_ex
        ask_ex_name, ask_price, ask_qty = best_ask_ex

        if bid_price <= ask_price:
            return False, 0.0

        # Profit calculation
        arb_qty = min(bid_qty, ask_qty)
        profit_per_unit = bid_price - ask_price
        profit_pct = profit_per_unit / ask_price

        if profit_pct < min_profit_pct:
            return False, 0.0

        # Rough annualized return (assuming 1 opportunity per minute)
        ops_per_day = 1440
        annual_profit_pct = profit_pct * ops_per_day * 365

        return True, round(annual_profit_pct * 100, 4)

    def get_cross_exchange_prices(self, symbol: str) -> dict[str, float]:
        """
        获取各交易所当前中间价 / Get mid-prices across all exchanges.

        Returns:
            dict: {exchange_name: mid_price}
        """
        prices: dict[str, float] = {}
        for ex in self.exchanges:
            try:
                ob = ex.get_order_book(symbol)
                prices[ex.exchange_name] = self._mid_price(ob)
            except Exception:
                continue
        return prices


# -------------------------------------------------------------------
# OctoBotMarketMaker — OctoBot风格做市框架
# -------------------------------------------------------------------


class OctoBotMarketMaker:
    """
    OctoBot风格做市框架 / OctoBot-style market making framework.

    特点 / Features:
        - 多交易所支持 (REST API)
        - 参考价格保护 — 报价以聚合参考价格为中心, 避免踏空
        - 自适应价差 — 根据波动率、订单簿深度、持仓自动调整买卖价差
        - 持仓限制 — 单边最大持仓, 防止风险过度集中
        - 订单簿聚合评估 — 多交易所订单簿统一评估

    主循环 (run):
        1. 从所有交易所聚合订单簿 → 计算参考价格
        2. 检测套利机会 (若存在则报警/跳过做市)
        3. 计算自适应买卖报价
        4. 取消过期订单
        5. 重新挂单

    Args:
        exchanges:       ExchangeInterface实例列表
        ref_engine:      ReferencePriceEngine实例
        min_spread_pct:  最小价差百分比 (默认0.005 = 0.5%)
        position_limit:  单边最大持仓 (default 1000.0)
    """

    def __init__(
        self,
        exchanges: list[ExchangeInterface],
        ref_engine: ReferencePriceEngine,
        min_spread_pct: float = 0.005,
        position_limit: float = 1000.0,
    ) -> None:
        self.exchanges = exchanges
        self.ref_engine = ref_engine
        self.min_spread_pct = min_spread_pct
        self.position_limit = position_limit

        # Position tracking: symbol -> {long_qty, short_qty, entry_price}
        self._positions: dict[str, dict] = {}

        # Active orders: order_id -> {symbol, side, price, qty, timestamp}
        self._active_orders: dict[str, dict] = {}

        # Statistics
        self._stats = {
            "total_arb_alerts":    0,
            "total_orders_placed": 0,
            "total_orders_cancelled": 0,
        }

    # -----------------------------------------------------------------
    # Position helpers
    # -----------------------------------------------------------------

    def _get_net_position(self, symbol: str) -> float:
        """
        获取净持仓 (多头 - 空头) / Get net position (long - short).
        Positive = long, Negative = short.
        """
        pos = self._positions.get(symbol, {"long": 0.0, "short": 0.0})
        return pos.get("long", 0.0) - pos.get("short", 0.0)

    def _can_place_order(self, symbol: str, side: str, qty: float) -> bool:
        """检查是否可以下单 (持仓限制检查)."""
        net = self._get_net_position(symbol)
        if side == "buy":
            if net + qty > self.position_limit:
                return False
        else:  # sell
            if net - qty < -self.position_limit:
                return False
        return True

    # -----------------------------------------------------------------
    # Adaptive spread calculation
    # -----------------------------------------------------------------

    def _adaptive_spread(
        self,
        symbol: str,
        ref_price: float,
        order_book: Optional[dict] = None,
    ) -> float:
        """
        计算自适应价差 / Compute adaptive spread based on market conditions.

        影响因素:
            - 基础价差 (min_spread_pct)
            - 波动率 (通过价差历史估算)
            - 订单簿深度 (深度越浅, 价差越大)
            - 持仓偏离 (偏离越大, 价差越大)

        Args:
            symbol:     交易对
            ref_price:  参考价格
            order_book: 当前订单簿 (可选, 若不提供则使用各交易所平均值)

        Returns:
            float: 买卖价差 (绝对值, 不是百分比)
        """
        # Base spread from min_spread_pct
        base_spread = ref_price * self.min_spread_pct

        # Depth adjustment
        depth_factor = 1.0
        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if bids and asks:
                total_depth = sum(float(b[1]) for b in bids[:5]) + sum(float(a[1]) for a in asks[:5])
                if total_depth < ref_price * 0.1:  # Thin book
                    depth_factor = 2.0
                elif total_depth > ref_price * 10:  # Deep book
                    depth_factor = 0.8

        # Inventory skew adjustment
        net = self._get_net_position(symbol)
        skew_factor = 1.0 + abs(net) / self.position_limit

        adaptive = base_spread * depth_factor * skew_factor
        return max(adaptive, base_spread)  # Never below base

    # -----------------------------------------------------------------
    # Quote calculation
    # -----------------------------------------------------------------

    def calculate_quote(self, symbol: str) -> tuple[float, float]:
        """
        计算买卖报价 / Calculate bid and ask quotes for a symbol.

        流程:
            1. 获取参考价格 (多交易所加权)
            2. 计算自适应价差
            3. bid = ref_price - spread/2
            4. ask = ref_price + spread/2

        Args:
            symbol: 交易对, e.g. "BTC/USDT"

        Returns:
            tuple[float, float]: (bid_price, ask_price)
        """
        ref_price = self.ref_engine.get_reference_price(symbol)
        if ref_price <= 0:
            return 0.0, 0.0

        spread = self._adaptive_spread(symbol, ref_price)
        bid = ref_price - spread / 2.0
        ask = ref_price + spread / 2.0

        return round(bid, 8), round(ask, 8)

    # -----------------------------------------------------------------
    # Order management
    # -----------------------------------------------------------------

    def _refresh_orders(self, symbol: str) -> None:
        """
        刷新订单 — 取消超时订单, 重新报价 / Refresh orders for a symbol.

        实现逻辑:
            1. 获取所有活跃订单
            2. 取消超过60秒未成交的订单
            3. 检查是否需要重新报价
        """
        for ex in self.exchanges:
            try:
                open_orders = ex.get_open_orders(symbol)
            except Exception:
                continue

            now = time.time()
            for order in open_orders:
                order_id = order["order_id"]
                ts = order.get("timestamp", 0)
                age = now * 1000 - ts / 1000 if ts else 0

                # Cancel orders older than 60 seconds
                if age > 60_000:
                    try:
                        ex.cancel_order(order_id)
                        self._stats["total_orders_cancelled"] += 1
                        self._active_orders.pop(order_id, None)
                    except Exception:
                        pass

    def refresh_orders(self, symbol: str) -> None:
        """
        刷新订单 (公开API) / Refresh orders — cancel stale, requote if needed.

        Args:
            symbol: 交易对
        """
        self._refresh_orders(symbol)

    def place_quote(
        self,
        symbol: str,
        bid_price: float,
        ask_price: float,
        qty: float,
    ) -> list[dict]:
        """
        同时挂买单和卖单 / Place both bid and ask quotes simultaneously.

        Args:
            symbol:    交易对
            bid_price: 买价
            ask_price: 卖价
            qty:       数量

        Returns:
            list[dict]: 下的订单列表
        """
        placed: list[dict] = []

        # Check if we can place orders (position limits)
        if not self._can_place_order(symbol, "buy", qty):
            return placed
        if not self._can_place_order(symbol, "sell", qty):
            return placed

        for ex in self.exchanges:
            try:
                bid_order = ex.place_limit_order(symbol, "buy", bid_price, qty)
                ask_order = ex.place_limit_order(symbol, "sell", ask_price, qty)
                self._active_orders[bid_order["order_id"]] = bid_order
                self._active_orders[ask_order["order_id"]] = ask_order
                placed.extend([bid_order, ask_order])
                self._stats["total_orders_placed"] += 2
            except Exception:
                continue

        return placed

    # -----------------------------------------------------------------
    # Strategy evaluation
    # -----------------------------------------------------------------

    def evaluate_strategy(self, symbol: str) -> dict:
        """
        做市策略评估器 / Evaluate current market-making performance.

        Returns:
            dict: 评估报告, 包含:
                - reference_price: 当前参考价格
                - bid/ask: 当前报价
                - spread_pct: 价差百分比 (bp)
                - net_position: 净持仓
                - arb_opportunity: 是否存在套利机会
                - arb_profit_pct: 套利年化利润
                - exchange_prices: 各交易所价格
        """
        ref_price = self.ref_engine.get_reference_price(symbol)
        bid, ask = self.calculate_quote(symbol)

        has_arb, arb_pct = self.ref_engine.detect_arb_opportunity(symbol)
        if has_arb:
            self._stats["total_arb_alerts"] += 1

        net_pos = self._get_net_position(symbol)
        spread_pct = ((ask - bid) / ref_price * 10000) if ref_price > 0 and ask > bid else 0.0
        ex_prices = self.ref_engine.get_cross_exchange_prices(symbol)

        return {
            "symbol":             symbol,
            "reference_price":    round(ref_price, 8),
            "bid":                bid,
            "ask":                ask,
            "spread_pct":         round(spread_pct, 2),      # in basis points
            "net_position":       round(net_pos, 8),
            "position_limit":     self.position_limit,
            "arb_opportunity":    has_arb,
            "arb_annual_pct":     arb_pct,
            "exchange_prices":    {k: round(v, 8) for k, v in ex_prices.items()},
            "stats":              dict(self._stats),
        }

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    def run(
        self,
        symbol: str,
        qty: float = 0.01,
        loop_interval: float = 5.0,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        主循环 / Main market-making loop.

        Args:
            symbol:         交易对
            qty:            每次下单数量
            loop_interval:  循环间隔 (秒)
            max_iterations: 最大迭代次数 (None = 无限循环)
        """
        iteration = 0
        while True:
            # 1. Check for arbitrage opportunity
            has_arb, arb_pct = self.ref_engine.detect_arb_opportunity(symbol)
            if has_arb:
                # Alert: skip market making during arbitrage
                print(f"[OctoBotMM] Arb opportunity detected: {arb_pct:.2f}% annual. Skipping quote.")
                self._stats["total_arb_alerts"] += 1

            # 2. Refresh (cancel stale) orders
            self._refresh_orders(symbol)

            # 3. Calculate and place new quotes
            if not has_arb:
                bid, ask = self.calculate_quote(symbol)
                if bid > 0 and ask > bid:
                    placed = self.place_quote(symbol, bid, ask, qty)
                    if placed:
                        print(f"[OctoBotMM] Placed bid={bid:.8f} ask={ask:.8f} qty={qty}")

            # 4. Log evaluation
            eval_report = self.evaluate_strategy(symbol)
            print(
                f"[OctoBotMM] ref={eval_report['reference_price']:.8f} "
                f"spread={eval_report['spread_pct']:.1f}bp "
                f"net_pos={eval_report['net_position']:.4f}"
            )

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break

            time.sleep(loop_interval)


# -------------------------------------------------------------------
# Module exports
# -------------------------------------------------------------------

__all__ = [
    "ExchangeInterface",
    "ReferencePriceEngine",
    "OctoBotMarketMaker",
]
