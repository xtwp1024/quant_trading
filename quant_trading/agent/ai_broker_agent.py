"""
AI Broker Agent — Multi-Agent Stock Analysis & Trading System
AI Broker 代理 — 多智能体股票分析与交易系统

Absorbs logic from ai-broker-investing-agent TypeScript/Next.js app:
- BullBearDebateEngine  (multi-agent debate via Groq LLM)
- AlpacaConnector       (Alpaca Broker REST API, pure urllib)
- Backtester            (buy-and-hold, momentum strategies)
- StockQuoteService     (Yahoo Finance / Finnhub quote aggregation)
- PolymarketAgent       (prediction market data, price change tracking)
- AIInvestAdvisor       (unified advisor facade)

所有HTTP客户端仅使用urllib（无外部SDK依赖）。
All HTTP clients use urllib only (no external SDK dependencies).

Classes:
    BullBearDebateEngine  — 多智能体辩论引擎（牛市vs熊市）
    AlpacaConnector       — Alpaca Broker REST API 适配器（纯 urllib）
    Backtester            — 历史回测引擎（买入持有、动量策略）
    StockQuoteService     — 股票行情服务（Yahoo Finance + Finnhub）
    PolymarketAgent       — 预测市场连接器（Polymarket API）
    AIInvestAdvisor       — 统一投资顾问门面（整合所有Agent）

__all__ = [
    "BullBearDebateEngine",
    "AlpacaConnector",
    "Backtester",
    "StockQuoteService",
    "PolymarketAgent",
    "AIInvestAdvisor",
    "TradeDecision",
    "BacktestResult",
    "StockQuote",
    "PolymarketMarket",
]
"""

from __future__ import annotations

import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Lazy-import helpers for graceful degradation
# ---------------------------------------------------------------------------

def _lazy_json_loads(s, **kw):
    """Lazy JSON decode (placeholder — unused but kept for compatibility)."""
    import json as _json
    return _json.loads(s, **kw)


# ---------------------------------------------------------------------------
# Dataclasses / Types
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    """
    交易决策数据结构 / Trade decision data structure.

    Attributes:
        action (str): 交易动作 — BUY | SELL | HOLD
        confidence (float): 置信度 0.0–1.0
        reasoning (str): 决策理由
        timestamp (str): ISO 格式时间戳
        symbol (str): 股票代码
    """
    action: str
    confidence: float
    reasoning: str
    timestamp: str
    symbol: str = ""

    def __post_init__(self):
        self.action = self.action.upper()
        if self.action not in ("BUY", "SELL", "HOLD"):
            self.action = "HOLD"


@dataclass
class BacktestTrade:
    """单笔交易记录 / Single trade record."""
    date: str
    action: str  # BUY | SELL
    price: float
    shares: int
    value: float


@dataclass
class BacktestMetrics:
    """回测绩效指标 / Backtest performance metrics."""
    sharpe_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0


@dataclass
class BacktestResult:
    """
    回测结果 / Backtest result.

    Attributes:
        success (bool): 是否成功
        symbol (str): 股票代码
        initial_capital (float): 初始资金
        final_value (float): 最终价值
        total_return (float): 绝对收益
        total_return_percent (float): 收益率百分比
        trades (list[BacktestTrade]): 交易列表
        metrics (BacktestMetrics): 绩效指标
    """
    success: bool
    symbol: str
    initial_capital: float
    final_value: float
    total_return: float
    total_return_percent: float
    trades: list[BacktestTrade]
    metrics: BacktestMetrics


@dataclass
class StockQuote:
    """
    股票行情数据 / Stock quote data.

    Attributes:
        symbol (str): 股票代码
        price (float): 当前价格
        change (float): 价格变化
        change_percent (float): 变化百分比
        open (float): 开盘价
        high (float): 最高价
        low (float): 最低价
        previous_close (float): 前收盘价
        volume (int): 成交量
        market_cap (float): 市值
        currency (str): 货币
        name (str): 公司名称
        exchange (str): 交易所
        sector (str): 行业板块
        industry (str): 细分行业
        timestamp (int): Unix 时间戳
        source (str): 数据来源
    """
    symbol: str = ""
    price: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    previous_close: float = 0.0
    volume: int = 0
    market_cap: float = 0.0
    currency: str = "USD"
    name: str = ""
    exchange: str = ""
    sector: str = ""
    industry: str = ""
    timestamp: int = 0
    source: str = "unknown"


@dataclass
class PolymarketMarket:
    """
    Polymarket 预测市场数据 / Polymarket prediction market data.

    Attributes:
        id (str): 市场 ID
        question (str): 市场问题
        slug (str): URL slug
        volume_24hr (int): 24小时成交量
        volume_total (int): 总成交量
        active (bool): 是否活跃
        closed (bool): 是否已关闭
        outcomes (list[str]): 结果列表
        outcome_prices (list[str]): 各结果赔率
        clob_token_ids (list[str]): CLOB 代币 ID
        description (str): 描述
        end_date (str): 结束日期
        category (str): 类别
        price_changes (dict): 价格变化 (daily/weekly/monthly)
    """
    id: str = ""
    question: str = ""
    slug: str = ""
    volume_24hr: int = 0
    volume_total: int = 0
    active: bool = True
    closed: bool = False
    outcomes: list[str] = field(default_factory=list)
    outcome_prices: list[str] = field(default_factory=list)
    clob_token_ids: list[str] = field(default_factory=list)
    description: str = ""
    end_date: str = ""
    category: str = ""
    price_changes: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# REST Client Helper
# ---------------------------------------------------------------------------

class _RestClient:
    """
    轻量级 REST 客户端（仅使用 urllib）。/ Lightweight REST client using urllib only.

    支持 GET/POST/DELETE 请求，自动处理 JSON 编码/解码，
    支持 API Key 认证和自定义请求头。
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        secret_key: str = "",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.secret_key = secret_key
        self.timeout = timeout
        self._ctx = ssl.create_default_context()

    def _headers(self, extra: Optional[dict] = None) -> dict:
        """构建请求头。"""
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            h["APCA-API-KEY-ID"] = self.api_key
        if self.secret_key:
            h["APCA-API-SECRET-KEY"] = self.secret_key
        if extra:
            h.update(extra)
        return h

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}" if self.base_url else path

    def request(
        self,
        method: str,
        path: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """
        发送 REST 请求。/ Send a REST request.

        Args:
            method: GET | POST | DELETE | PUT
            path: API 路径
            data: 请求体字典
            params: URL 查询参数

        Returns:
            dict: 解析后的 JSON 响应
        """
        url = self._url(path)
        if params:
            url += "?" + urllib.parse.urlencode(params)

        body = json.dumps(data).encode("utf-8") if data else None

        req = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            body_raw = e.read().decode("utf-8") if e.fp else ""
            try:
                err_body = json.loads(body_raw) if body_raw else {}
            except Exception:
                err_body = {"raw": body_raw}
            raise RuntimeError(
                f"HTTP {e.code} {e.reason}: {err_body}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"URL error: {e.reason}") from e

    def get(self, path: str, params: Optional[dict] = None) -> dict:
        return self.request("GET", path, params=params)

    def post(self, path: str, data: dict) -> dict:
        return self.request("POST", path, data=data)

    def delete(self, path: str, params: Optional[dict] = None) -> dict:
        return self.request("DELETE", path, params=params)


# ---------------------------------------------------------------------------
# AlpacaConnector
# ---------------------------------------------------------------------------

class AlpacaConnector:
    """
    Alpaca Broker REST API 连接器（纯 urllib）/ Alpaca Broker REST API connector.

    支持现货/期货订单管理、账户查询、持仓查询。
    Alpaca 文档: https://docs.alpaca.markets/

    Attributes:
        key_id (str): API Key ID
        secret_key (str): API Secret Key
        paper (bool): 是否使用 Paper 交易模式（默认 True）
        base_url (str): API Base URL

    Example:
        >>> conn = AlpacaConnector(
        ...     key_id="PKXXXX",
        ...     secret_key="SECXXXX",
        ...     paper=True,
        ... )
        >>> orders = conn.get_orders(status="all", limit=50)
        >>> print(orders)
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    BROKER_URL = "https://broker-api.alpaca.markets"

    def __init__(
        self,
        key_id: str = "",
        secret_key: str = "",
        paper: bool = True,
        base_url: str = "",
    ):
        # 从环境变量兜底 / Fallback to env vars
        self.key_id = key_id or os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or ""
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET") or os.getenv("APCA_API_SECRET_KEY") or ""
        self.paper = paper
        self.base_url = (
            base_url
            or os.getenv("ALPACA_BASE_URL")
            or os.getenv("APCA_API_BASE_URL")
            or (self.PAPER_URL if paper else self.LIVE_URL)
        )
        self._client = _RestClient(
            base_url=self.base_url,
            api_key=self.key_id,
            secret_key=self.secret_key,
        )

    # ---- Account ---------------------------------------------------------

    def get_account(self) -> dict:
        """
        获取账户信息。/ Get account information.

        Returns:
            dict: 账户信息字典
        """
        return self._client.get("/v2/account")

    # ---- Orders ----------------------------------------------------------

    def get_orders(
        self,
        status: str = "all",
        limit: int = 50,
        direction: str = "desc",
        symbols: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        查询订单列表。/ Get list of orders.

        Args:
            status: all | open | closed | filled | canceled
            limit: 返回数量上限
            direction: asc | desc（按时间排序方向）
            symbols: 仅返回涉及这些symbol的订单

        Returns:
            list[dict]: 订单列表
        """
        params: dict[str, Any] = {"status": status, "limit": limit, "direction": direction}
        if symbols:
            params["symbols"] = ",".join(symbols)
        return self._client.get("/v2/orders", params=params)

    def get_order(self, order_id: str) -> dict:
        """获取单个订单。/ Get a single order by ID."""
        return self._client.get(f"/v2/orders/{order_id}")

    def create_order(
        self,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_price: Optional[float] = None,
        trail_percent: Optional[float] = None,
        extended_hours: bool = False,
        client_order_id: Optional[str] = None,
        order_class: Optional[str] = None,
        take_profit: Optional[dict] = None,
        stop_loss: Optional[dict] = None,
    ) -> dict:
        """
        创建新订单。/ Create a new order.

        Args:
            symbol: 股票代码（如 "AAPL"）
            qty: 购买数量（与 notional 二选一）
            notional: 购买金额（与 qty 二选一）
            side: buy | sell
            order_type: market | limit | stop | stop_limit | trailing_stop
            time_in_force: day | gtc | opg | cls | ioc | fok
            limit_price: 限价价格
            stop_price: 止损价格
            trail_price: 跟踪止损价格
            trail_percent: 跟踪止损百分比
            extended_hours: 是否允许盘前盘后交易
            client_order_id: 客户端自定义订单ID
            order_class: simple | bracket | one_triggers_other | two_triggers_other
            take_profit: 止盈参数 {"limit_price": float}
            stop_loss: 止损参数 {"stop_price": float}

        Returns:
            dict: 创建的订单对象
        """
        data: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force.lower(),
            "extended_hours": extended_hours,
        }
        if qty is not None:
            data["qty"] = str(qty)
        if notional is not None:
            data["notional"] = str(notional)
        if limit_price is not None:
            data["limit_price"] = str(limit_price)
        if stop_price is not None:
            data["stop_price"] = str(stop_price)
        if trail_price is not None:
            data["trail_price"] = str(trail_price)
        if trail_percent is not None:
            data["trail_percent"] = str(trail_percent)
        if client_order_id:
            data["client_order_id"] = client_order_id
        if order_class:
            data["order_class"] = order_class
        if take_profit:
            data["take_profit"] = take_profit
        if stop_loss:
            data["stop_loss"] = stop_loss

        return self._client.post("/v2/orders", data)

    def cancel_order(self, order_id: str) -> dict:
        """
        取消指定订单。/ Cancel a specific order.

        Args:
            order_id: 订单 ID

        Returns:
            dict: 通常为空响应
        """
        return self._client.delete(f"/v2/orders/{order_id}")

    def cancel_all_orders(self) -> dict:
        """取消所有未完成订单。/ Cancel all open orders."""
        return self._client.delete("/v2/orders")

    # ---- Positions -------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """获取所有当前持仓。/ Get all open positions."""
        return self._client.get("/v2/positions")

    def get_position(self, symbol: str) -> dict:
        """获取指定symbol的持仓。/ Get position for a specific symbol."""
        return self._client.get(f"/v2/positions/{symbol.upper()}")

    # ---- Assets ----------------------------------------------------------

    def list_assets(self, status: str = "active") -> list[dict]:
        """
        列出可交易资产。/ List tradable assets.

        Args:
            status: active | inactive | paused

        Returns:
            list[dict]: 资产列表
        """
        return self._client.get("/v2/assets", params={"status": status})

    # ---- Clock -----------------------------------------------------------

    def get_clock(self) -> dict:
        """获取市场时间。/ Get market clock."""
        return self._client.get("/v2/clock")


# ---------------------------------------------------------------------------
# StockQuoteService
# ---------------------------------------------------------------------------

class StockQuoteService:
    """
    股票行情服务（Yahoo Finance + Finnhub）/ Stock quote service.

    纯 urllib 实现，无 SDK 依赖。
    数据来源优先级：Finnhub（详细） > Yahoo Finance（备用）

    Attributes:
        finnhub_key (str): Finnhub API Key
        cache_ttl (int): 缓存有效期（秒），默认 60s
    """

    YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance"
    FINNHUB_BASE = "https://finnhub.io/api/v1"

    def __init__(self, finnhub_key: str = "", cache_ttl: int = 60):
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY") or ""
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, StockQuote]] = {}

    # ---- Yahoo Finance ---------------------------------------------------

    def _yahoo_quote(self, symbol: str) -> Optional[StockQuote]:
        """通过 Yahoo Finance 获取行情。"""
        url = f"{self.YAHOO_BASE}/chart/{symbol.upper()}"
        params = {
            "range": "1d",
            "interval": "1d",
            "includePrePost": "false",
        }
        try:
            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
            return StockQuote(
                symbol=symbol.upper(),
                price=meta.get("regularMarketPrice", 0.0),
                change=meta.get("regularMarketChange", 0.0),
                change_percent=meta.get("regularMarketChangePercent", 0.0),
                open=meta.get("regularMarketOpen", 0.0),
                high=meta.get("regularMarketDayHigh", 0.0),
                low=meta.get("regularMarketDayLow", 0.0),
                previous_close=meta.get("previousClose", 0.0),
                volume=int(meta.get("regularMarketVolume", 0)),
                market_cap=float(meta.get("marketCap", 0)),
                currency=meta.get("currency", "USD"),
                name=meta.get("shortName", meta.get("symbol", symbol)),
                exchange=meta.get("exchange", ""),
                timestamp=int(meta.get("regularMarketTime", 0)),
                source="yahoo",
            )
        except Exception:
            return None

    def _finnhub_quote(self, symbol: str) -> Optional[StockQuote]:
        """通过 Finnhub 获取行情。"""
        if not self.finnhub_key:
            return None
        url = f"{self.FINNHUB_BASE}/quote"
        params = {"symbol": symbol.upper(), "token": self.finnhub_key}
        try:
            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("c") and len(data["c"]) >= 6:
                return StockQuote(
                    symbol=symbol.upper(),
                    price=float(data["c"][0]),
                    change=float(data["c"][1]),
                    change_percent=float(data["c"][2]) if data["c"][2] else 0.0,
                    open=float(data["o"][0]) if isinstance(data["o"], list) else float(data["o"]),
                    high=float(data["h"][0]) if isinstance(data["h"], list) else float(data["h"]),
                    low=float(data["l"][0]) if isinstance(data["l"], list) else float(data["l"]),
                    previous_close=float(data["pc"][0]) if isinstance(data["pc"], list) else float(data["pc"]),
                    volume=int(data["v"][0]) if isinstance(data["v"], list) else int(data["v"]),
                    timestamp=int(data["t"][0]) if isinstance(data["t"], list) else int(data.get("t", 0)),
                    source="finnhub",
                )
        except Exception:
            pass
        return None

    def _finnhub_profile(self, symbol: str) -> dict:
        """获取 Finnhub 股票详情（板块、行业）。"""
        if not self.finnhub_key:
            return {}
        url = f"{self.FINNHUB_BASE}/stock/profile2"
        params = {"symbol": symbol.upper(), "token": self.finnhub_key}
        try:
            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {}

    def get_quote(self, symbol: str, use_cache: bool = True) -> StockQuote:
        """
        获取股票行情（优先 Finnhub，失败则 Yahoo Finance）。/ Get stock quote.

        Args:
            symbol: 股票代码（如 "AAPL"）
            use_cache: 是否使用本地缓存（默认 True）

        Returns:
            StockQuote: 行情对象
        """
        now = time.time()
        if use_cache and symbol.upper() in self._cache:
            ts, cached = self._cache[symbol.upper()]
            if now - ts < self.cache_ttl:
                return cached

        # 尝试 Finnhub
        quote = self._finnhub_quote(symbol)
        if quote is None:
            quote = self._yahoo_quote(symbol)
        if quote is None:
            quote = StockQuote(symbol=symbol.upper(), source="none")

        # 补充 Finnhub 板块/行业信息
        profile = self._finnhub_profile(symbol)
        if profile:
            quote.sector = profile.get("finnhubIndustry", "")
            quote.industry = profile.get("industry", "")
            if not quote.name:
                quote.name = profile.get("name", "")

        self._cache[symbol.upper()] = (now, quote)
        return quote

    def get_historical(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> list[dict]:
        """
        获取历史K线数据（Yahoo Finance）。/ Get historical OHLCV data.

        Args:
            symbol: 股票代码
            start_date: 开始日期 "YYYY-MM-DD"
            end_date: 结束日期 "YYYY-MM-DD"
            interval: K线周期 1d | 1wk | 1mo

        Returns:
            list[dict]: 包含 date, open, high, low, close, volume 的字典列表
        """
        try:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        except Exception:
            return []

        url = f"{self.YAHOO_BASE}/chart/{symbol.upper()}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": interval,
            "includePrePost": "false",
        }
        try:
            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            result = data.get("chart", {}).get("result", [{}])
            if not result:
                return []
            quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
            timestamps = result[0].get("timestamp", [])
            closes = quotes.get("close", [])
            opens = quotes.get("open", [])
            highs = quotes.get("high", [])
            lows = quotes.get("low", [])
            volumes = quotes.get("volume", [])

            out = []
            for i, ts in enumerate(timestamps):
                out.append({
                    "date": datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                    "open": opens[i] if i < len(opens) and opens[i] is not None else 0.0,
                    "high": highs[i] if i < len(highs) and highs[i] is not None else 0.0,
                    "low": lows[i] if i < len(lows) and lows[i] is not None else 0.0,
                    "close": closes[i] if i < len(closes) and closes[i] is not None else 0.0,
                    "volume": volumes[i] if i < len(volumes) and volumes[i] is not None else 0,
                })
            return out
        except Exception:
            return []

    def get_peers(self, symbol: str) -> list[str]:
        """
        获取相关股票列表（Finnhub）。/ Get peer stocks.
        """
        if not self.finnhub_key:
            return []
        url = f"{self.FINNHUB_BASE}/stock/peers"
        params = {"symbol": symbol.upper(), "token": self.finnhub_key}
        try:
            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                return [p for p in data if p != symbol.upper()]
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    历史回测引擎 / Historical backtesting engine.

    支持策略:
    - buy-and-hold   : 买入持有策略
    - momentum       : 动量策略（20日均线 crossover）

    Attributes:
        initial_capital (float): 初始资金（默认 100,000）
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital

    def _calc_momentum_signals(
        self, prices: list[dict]
    ) -> list[dict]:
        """
        计算动量信号（20日均线 crossover）。/ Calculate momentum signals.

        Args:
            prices: K线数据列表，每项包含 close 和 date

        Returns:
            list[dict]: 每项含 date, signal (BUY|SELL|HOLD)
        """
        ma_window = 20
        signals = []
        for i in range(ma_window, len(prices)):
            window_prices = [p["close"] for p in prices[i - ma_window:i]]
            prev_prices = [p["close"] for p in prices[i - ma_window - 1:i - 1]]

            ma = sum(window_prices) / ma_window
            prev_ma = sum(prev_prices) / ma_window

            current_close = prices[i]["close"]
            prev_close = prices[i - 1]["close"]

            if current_close > ma and prev_close <= prev_ma:
                signals.append({"date": prices[i]["date"], "signal": "BUY"})
            elif current_close < ma and prev_close >= prev_ma:
                signals.append({"date": prices[i]["date"], "signal": "SELL"})
            else:
                signals.append({"date": prices[i]["date"], "signal": "HOLD"})
        return signals

    def _metrics_from_trades(
        self,
        trades: list[BacktestTrade],
        final_value: float,
        max_value: float,
        min_value: float,
    ) -> BacktestMetrics:
        """计算绩效指标。"""
        total = len(trades)
        winning = 0
        # 统计盈利交易（每对 BUY-SELL）
        buy_price = 0.0
        for t in trades:
            if t.action == "BUY":
                buy_price = t.price
            elif t.action == "SELL" and buy_price > 0:
                if t.price > buy_price:
                    winning += 1
                buy_price = 0.0

        pairs = total // 2 if total >= 2 else 1
        win_rate = (winning / pairs * 100) if pairs > 0 else 0.0
        max_dd = ((max_value - min_value) / max_value * 100) if max_value > 0 else 0.0

        return BacktestMetrics(
            max_drawdown=max_dd,
            win_rate=round(win_rate, 2),
            total_trades=total,
            sharpe_ratio=None,  # 简化版未实现
        )

    def run(
        self,
        symbol: str,
        prices: list[dict],
        strategy: str = "buy-and-hold",
    ) -> BacktestResult:
        """
        运行回测。/ Run backtest.

        Args:
            symbol: 股票代码
            prices: K线数据列表，每项需含 close, date
            strategy: buy-and-hold | momentum

        Returns:
            BacktestResult: 回测结果
        """
        if not prices:
            return BacktestResult(
                success=False,
                symbol=symbol,
                initial_capital=self.initial_capital,
                final_value=self.initial_capital,
                total_return=0.0,
                total_return_percent=0.0,
                trades=[],
                metrics=BacktestMetrics(),
            )

        try:
            if strategy == "buy-and-hold":
                return self._run_buy_and_hold(symbol, prices)
            elif strategy == "momentum":
                return self._run_momentum(symbol, prices)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        except Exception as e:
            return BacktestResult(
                success=False,
                symbol=symbol,
                initial_capital=self.initial_capital,
                final_value=self.initial_capital,
                total_return=0.0,
                total_return_percent=0.0,
                trades=[],
                metrics=BacktestMetrics(),
            )

    def _run_buy_and_hold(self, symbol: str, prices: list[dict]) -> BacktestResult:
        """买入持有策略回测。"""
        cash = self.initial_capital
        shares = 0
        trades = []

        buy_price = prices[0]["close"]
        shares = math.floor(self.initial_capital / buy_price)
        cost = shares * buy_price
        cash -= cost

        trades.append(BacktestTrade(
            date=prices[0]["date"],
            action="BUY",
            price=buy_price,
            shares=shares,
            value=cost,
        ))

        sell_price = prices[-1]["close"]
        revenue = shares * sell_price
        cash += revenue

        trades.append(BacktestTrade(
            date=prices[-1]["date"],
            action="SELL",
            price=sell_price,
            shares=shares,
            value=revenue,
        ))

        final_value = cash
        total_return = final_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital * 100)

        metrics = BacktestMetrics(
            max_drawdown=0.0,
            win_rate=100.0 if final_value > self.initial_capital else 0.0,
            total_trades=2,
        )

        return BacktestResult(
            success=True,
            symbol=symbol,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_return_percent=total_return_pct,
            trades=trades,
            metrics=metrics,
        )

    def _run_momentum(self, symbol: str, prices: list[dict]) -> BacktestResult:
        """动量策略回测（20日 MA crossover）。"""
        signals = self._calc_momentum_signals(prices)
        cash = self.initial_capital
        shares = 0
        trades = []
        max_value = self.initial_capital
        min_value = self.initial_capital

        # offset by MA window (20)
        for i, signal in enumerate(signals):
            price_data = prices[i + 20]
            price_close = price_data.get("close")
            if price_close is None or price_close <= 0:
                continue

            if signal["signal"] == "BUY" and shares == 0:
                buy_shares = math.floor(cash / price_close)
                if buy_shares > 0:
                    cost = buy_shares * price_close
                    cash -= cost
                    trades.append(BacktestTrade(
                        date=signal["date"],
                        action="BUY",
                        price=price_close,
                        shares=buy_shares,
                        value=cost,
                    ))
                    shares = buy_shares

            elif signal["signal"] == "SELL" and shares > 0:
                revenue = shares * price_close
                cash += revenue
                trades.append(BacktestTrade(
                    date=signal["date"],
                    action="SELL",
                    price=price_close,
                    shares=shares,
                    value=revenue,
                ))
                shares = 0

            current_value = cash + (shares * price_close)
            max_value = max(max_value, current_value)
            min_value = min(min_value, current_value)

        # 平仓
        if shares > 0 and prices:
            last_price = prices[-1]["close"]
            cash += shares * last_price
            trades.append(BacktestTrade(
                date=prices[-1]["date"],
                action="SELL",
                price=last_price,
                shares=shares,
                value=shares * last_price,
            ))
            shares = 0

        final_value = cash
        total_return = final_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital * 100) if self.initial_capital > 0 else 0.0
        metrics = self._metrics_from_trades(trades, final_value, max_value, min_value)

        return BacktestResult(
            success=True,
            symbol=symbol,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_return_percent=total_return_pct,
            trades=trades,
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# PolymarketAgent
# ---------------------------------------------------------------------------

class PolymarketAgent:
    """
    Polymarket 预测市场连接器 / Polymarket prediction market connector.

    API 文档: https://docs.polymarket.com/
    纯 urllib 实现，无 SDK 依赖。

    Attributes:
        api_base (str): API Base URL
    """

    POLYMARKET_API = "https://clob.polymarket.com"
    MARKETS_BASE = "https://gamma-api.polymarket.com"

    def __init__(self, api_base: str = ""):
        self.api_base = api_base or self.MARKETS_BASE
        self._client = _RestClient(base_url=self.api_base)

    def _safe_parse_array(self, value: Any, splitter: Optional[str] = None) -> list:
        """安全解析可能为数组或 JSON 字符串的字段。"""
        if isinstance(value, list):
            # 处理双重序列化
            if (
                len(value) == 1
                and isinstance(value[0], str)
                and value[0].strip().startswith("[")
            ):
                try:
                    parsed = json.loads(value[0])
                    return parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    pass
            return value
        if not value:
            return []
        if isinstance(value, str):
            if value.strip().startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, str):
                        try:
                            return [json.loads(parsed)]
                        except Exception:
                            return [parsed]
                    return parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    pass
            if splitter:
                return [s.strip() for s in value.split(splitter) if s.strip()]
        return []

    def _calc_price_changes(
        self,
        token_id: str,
        current_price: float,
    ) -> dict:
        """
        计算价格变化（简化版，实际应调用 Polymarket CLOB API）。
        Calculate price changes (simplified; real impl should call CLOB API).
        """
        # 此为占位实现，实际应通过 CLOB API 获取历史价格
        return {"daily": None, "weekly": None, "monthly": None}

    def get_markets(
        self,
        limit: int = 50,
        offset: int = 0,
        window: str = "24h",
        category: Optional[str] = None,
    ) -> dict:
        """
        获取市场列表（从 Gamma API）。/ Get list of markets.

        Args:
            limit: 返回数量
            offset: 偏移量
            window: volume 排序窗口 24h | total
            category: 可选类别过滤

        Returns:
            dict: 含 markets, count, source, timestamp
        """
        sort_by = "volume24hr" if window != "total" else "volumeTotal"
        # Gamma API markets endpoint
        url = f"{self.api_base}/markets"
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "sortBy": sort_by,
            "active": "true",
        }
        if category:
            params["category"] = category

        try:
            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "markets": [],
                "count": 0,
                "source": "error",
                "timestamp": datetime.utcnow().isoformat(),
            }

        markets = data if isinstance(data, list) else data.get("markets", [])
        formatted = []

        for m in markets:
            clob_token_ids = self._safe_parse_array(m.get("clobTokenIds"))
            outcome_prices = self._safe_parse_array(m.get("outcomePrices"), ",")

            price_changes = {"daily": None, "weekly": None, "monthly": None}
            if clob_token_ids and outcome_prices:
                try:
                    current_p = float(outcome_prices[0])
                    token_id = clob_token_ids[0]
                    if token_id and re.match(r"^[a-zA-Z0-9_-]+$", str(token_id)) and len(str(token_id)) >= 10:
                        price_changes = self._calc_price_changes(str(token_id), current_p)
                except Exception:
                    pass

            formatted.append({
                "id": m.get("id", ""),
                "question": m.get("question", ""),
                "slug": m.get("slug", ""),
                "volume24hr": int(m.get("volume24hr", 0) or 0),
                "volumeTotal": int(m.get("volumeTotal", 0) or 0),
                "active": m.get("active", True),
                "closed": m.get("closed", False),
                "outcomes": self._safe_parse_array(m.get("outcomes"), ","),
                "outcomePrices": outcome_prices,
                "clobTokenIds": clob_token_ids,
                "image": m.get("image", ""),
                "description": m.get("description", ""),
                "endDate": m.get("endDate", ""),
                "groupItemTitle": m.get("groupItemTitle", ""),
                "enableOrderBook": m.get("enableOrderBook", False),
                "tags": self._safe_parse_array(m.get("tags")),
                "category": m.get("category", ""),
                "subcategory": m.get("subcategory", ""),
                "priceChanges": price_changes,
            })

        return {
            "success": True,
            "markets": formatted,
            "count": len(formatted),
            "source": "api",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def search_markets(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """
        搜索市场（通过 Gamma API）。/ Search markets.

        Args:
            query: 搜索关键词
            limit: 返回数量
            offset: 偏移量

        Returns:
            dict: 含 markets, count, source
        """
        # Polymarket gamma search endpoint
        url = f"{self.api_base}/markets"
        params: dict[str, Any] = {
            "search": query,
            "limit": limit,
            "offset": offset,
            "active": "true",
        }
        try:
            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "markets": [],
                "count": 0,
                "source": "error",
            }

        markets = data if isinstance(data, list) else data.get("markets", [])
        # 应用搜索过滤
        filtered = [
            m for m in markets
            if query.lower() in m.get("question", "").lower()
            or query.lower() in m.get("description", "").lower()
        ]

        return {
            "success": True,
            "markets": filtered[offset:offset + limit],
            "count": len(filtered),
            "source": "search",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_market_price(self, clob_token_id: str) -> Optional[float]:
        """
        获取指定 CLOB Token 的当前价格。/ Get current price for a CLOB token.
        """
        url = f"{self.POLYMARKET_API}/prices"
        params = {"token_ids": clob_token_id}
        try:
            req = urllib.request.Request(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict) and "data" in data:
                prices = data["data"].get(clob_token_id)
                if prices and len(prices) > 0:
                    return float(prices[0])
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# BullBearDebateEngine
# ---------------------------------------------------------------------------

class BullBearDebateEngine:
    """
    多智能体牛熊辩论引擎 / Multi-agent bull vs bear debate engine.

    模拟多方辩论：
    1. 基础分析师（市场/新闻/情绪/基本面）
    2. 牛市倡导者（Bull Researcher）
    3. 熊市倡导者（Bear Researcher）
    4. 投资裁判（Investment Judge）
    5. 交易员（Trader）

    接 Groq API 进行 LLM 分析（通过 urllib）。

    Attributes:
        groq_api_key (str): Groq API Key
        deep_think_model (str): 深度思考模型
        quick_think_model (str): 快速思考模型
        temperature (float): LLM temperature（默认 0.3）
    """

    GROQ_BASE = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        groq_api_key: str = "",
        deep_think_model: str = "llama-3.3-70b-versatile",
        quick_think_model: str = "llama-3.1-8b-instant",
        temperature: float = 0.3,
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY") or ""
        self.deep_think_model = deep_think_model
        self.quick_think_model = quick_think_model
        self.temperature = temperature
        self._client = _RestClient(
            base_url=self.GROQ_BASE,
            api_key=self.groq_api_key,
        )

    def _llm_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        调用 LLM（Groq API）。/ Call LLM via Groq API.

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
            model: 模型名（默认 self.deep_think_model）
            temperature: 温度参数

        Returns:
            str: LLM 输出的文本
        """
        if not self.groq_api_key:
            raise RuntimeError("GROQ_API_KEY not configured")
        model = model or self.deep_think_model
        temp = temperature if temperature is not None else self.temperature

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
        }
        try:
            data = self._client.post("/chat/completions", payload)
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}") from e

    def _system_prompt(self, role: str) -> str:
        """构建各角色的系统提示。"""
        prompts = {
            "market_analyst": (
                "You are a Market Analyst. Analyze current market conditions, "
                "price trends, volume, and technical indicators for the given stock. "
                "Provide a detailed market report."
            ),
            "news_analyst": (
                "You are a News Analyst. Analyze recent news, headlines, and "
                "sentiment surrounding the given stock. Provide a news impact report."
            ),
            "bull_researcher": (
                "You are a Bullish Researcher. Argue WHY the stock is a GOOD investment. "
                "Present bullish arguments, positive catalysts, growth potential, "
                "and favorable market conditions. Be thorough and persuasive."
            ),
            "bear_researcher": (
                "You are a Bearish Researcher. Argue WHY the stock is a RISKY investment. "
                "Present bearish arguments, risks, headwinds, valuation concerns, "
                "and negative catalysts. Be thorough and persuasive."
            ),
            "judge": (
                "You are an Investment Judge. After hearing bull and bear arguments, "
                "make a impartial decision: BUY, SELL, or HOLD. "
                "Provide clear reasoning based on the evidence presented."
            ),
            "trader": (
                "You are a Professional Trader. Based on all analyst reports and the judge's "
                "decision, produce a final trading recommendation: BUY, SELL, or HOLD. "
                "Include your reasoning. Be decisive."
            ),
        }
        return prompts.get(role, f"You are a {role} agent.")

    def analyze(
        self,
        symbol: str,
        trade_date: Optional[str] = None,
        quick_mode: bool = False,
        analysts: Optional[list[str]] = None,
    ) -> dict:
        """
        运行完整的多智能体辩论分析。/ Run full multi-agent debate analysis.

        Args:
            symbol: 股票代码（如 "AAPL"）
            trade_date: 交易日期（YYYY-MM-DD），默认当天
            quick_mode: 是否使用快速模式（使用 8b 模型）
            analysts: 启用的分析师类型，默认 ["market", "news"]

        Returns:
            dict: 包含 decision, confidence, market_report, news_report,
                 bull_arguments, bear_arguments, judge_decision, trader_decision, timestamp
        """
        trade_date = trade_date or datetime.now().strftime("%Y-%m-%d")
        model = self.quick_think_model if quick_mode else self.deep_think_model
        analysts = analysts or ["market", "news"]

        # Step 1: Market analysis
        market_report = ""
        if "market" in analysts:
            prompt = (
                f"Analyze the stock {symbol} for trading date {trade_date}. "
                "Provide a comprehensive market analysis including price trends, "
                "volume, technical indicators, and key support/resistance levels."
            )
            try:
                messages = [
                    {"role": "system", "content": self._system_prompt("market_analyst")},
                    {"role": "user", "content": prompt},
                ]
                market_report = self._llm_chat(messages, model=model)
            except Exception as e:
                market_report = f"Market analysis unavailable: {e}"

        # Step 2: News analysis
        news_report = ""
        if "news" in analysts:
            prompt = (
                f"Analyze recent news and sentiment for {symbol} as of {trade_date}. "
                "Identify key news events, sentiment drivers, and potential price impacts."
            )
            try:
                messages = [
                    {"role": "system", "content": self._system_prompt("news_analyst")},
                    {"role": "user", "content": prompt},
                ]
                news_report = self._llm_chat(messages, model=model)
            except Exception as e:
                news_report = f"News analysis unavailable: {e}"

        # Step 3: Bull vs Bear debate (3 rounds)
        bull_history = ""
        bear_history = ""

        for round_num in range(1, 4):
            # Bull argument
            bull_prompt = (
                f"[Round {round_num}] Stock: {symbol}, Date: {trade_date}\n"
                f"Previous context:\nBull case so far: {bull_history}\n"
                f"Bear case so far: {bear_history}\n"
                f"Market analysis: {market_report}\nNews: {news_report}\n"
                f"Present your strongest BULLISH arguments for {symbol}. "
                f"Focus on positive catalysts, growth drivers, and favorable conditions."
            )
            try:
                messages = [
                    {"role": "system", "content": self._system_prompt("bull_researcher")},
                    {"role": "user", "content": bull_prompt},
                ]
                bull_response = self._llm_chat(messages, model=model)
                bull_history += f"\n[Round {round_num}] {bull_response}"
            except Exception as e:
                bull_history += f"\n[Round {round_num}] Error: {e}"

            # Bear argument
            bear_prompt = (
                f"[Round {round_num}] Stock: {symbol}, Date: {trade_date}\n"
                f"Previous context:\nBull case so far: {bull_history}\n"
                f"Bear case so far: {bear_history}\n"
                f"Market analysis: {market_report}\nNews: {news_report}\n"
                f"Present your strongest BEARISH arguments against {symbol}. "
                f"Focus on risks, headwinds, valuation concerns, and negative catalysts."
            )
            try:
                messages = [
                    {"role": "system", "content": self._system_prompt("bear_researcher")},
                    {"role": "user", "content": bear_prompt},
                ]
                bear_response = self._llm_chat(messages, model=model)
                bear_history += f"\n[Round {round_num}] {bear_response}"
            except Exception as e:
                bear_history += f"\n[Round {round_num}] Error: {e}"

        # Step 4: Judge decision
        judge_prompt = (
            f"Stock: {symbol}, Date: {trade_date}\n\n"
            f"BULL ARGUMENTS:\n{bull_history}\n\n"
            f"BEAR ARGUMENTS:\n{bear_history}\n\n"
            f"Based on the above debate, make your impartial investment decision. "
            f"Should an investor BUY, SELL, or HOLD {symbol}? Provide clear reasoning."
        )
        try:
            messages = [
                {"role": "system", "content": self._system_prompt("judge")},
                {"role": "user", "content": judge_prompt},
            ]
            judge_decision = self._llm_chat(messages, model=self.quick_think_model)
        except Exception as e:
            judge_decision = f"Judge decision unavailable: {e}"

        # Step 5: Trader final decision
        trader_prompt = (
            f"Stock: {symbol}, Date: {trade_date}\n\n"
            f"MARKET REPORT:\n{market_report}\n\n"
            f"NEWS REPORT:\n{news_report}\n\n"
            f"BULL CASE:\n{bull_history}\n\n"
            f"BEAR CASE:\n{bear_history}\n\n"
            f"JUDGE'S DECISION:\n{judge_decision}\n\n"
            f"You are the final decision-maker. Produce a clear BUY, SELL, or HOLD "
            f"recommendation for {symbol} with specific reasoning. Be decisive."
        )
        try:
            messages = [
                {"role": "system", "content": self._system_prompt("trader")},
                {"role": "user", "content": trader_prompt},
            ]
            trader_decision = self._llm_chat(messages, model=self.quick_think_model)
        except Exception as e:
            trader_decision = f"Trader decision unavailable: {e}"

        # Extract action
        action = "HOLD"
        confidence = 0.5
        upper_trader = trader_decision.upper()
        if "BUY" in upper_trader and "NOT BUY" not in upper_trader:
            action = "BUY"
            confidence = 0.75
        elif "SELL" in upper_trader and "NOT SELL" not in upper_trader:
            action = "SELL"
            confidence = 0.75

        return {
            "success": True,
            "ticker": symbol.upper(),
            "result": {
                "company": symbol.upper(),
                "decision": action,
                "confidence": confidence,
                "market_report": market_report or "No market analysis performed.",
                "news_report": news_report or "No news analysis performed.",
                "sentiment_report": news_report,  # 简化版用 news 代替
                "fundamentals_report": "No fundamentals analysis performed.",
                "bull_arguments": bull_history.strip(),
                "bear_arguments": bear_history.strip(),
                "judge_decision": judge_decision.strip(),
                "trader_decision": trader_decision.strip(),
                "timestamp": datetime.utcnow().isoformat(),
            },
        }


# ---------------------------------------------------------------------------
# AIInvestAdvisor (Facade)
# ---------------------------------------------------------------------------

class AIInvestAdvisor:
    """
    统一投资顾问门面（整合所有 Agent）/ Unified investment advisor facade.

    整合 BullBearDebateEngine、StockQuoteService、Backtester、
    AlpacaConnector 和 PolymarketAgent，提供一站式投资分析服务。

    Attributes:
        groq_api_key (str): Groq API Key（用于辩论引擎）
        finnhub_key (str): Finnhub API Key（用于股票数据）
        alpaca_key (str): Alpaca API Key
        alpaca_secret (str): Alpaca Secret Key
        initial_capital (float): 回测初始资金
    """

    def __init__(
        self,
        groq_api_key: str = "",
        finnhub_key: str = "",
        alpaca_key: str = "",
        alpaca_secret: str = "",
        initial_capital: float = 100_000.0,
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY") or ""
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY") or ""
        self.alpaca_key = alpaca_key or os.getenv("ALPACA_API_KEY") or ""
        self.alpaca_secret = alpaca_secret or os.getenv("ALPACA_SECRET") or ""
        self.initial_capital = initial_capital

        # 延迟初始化各子 Agent（按需创建）
        self._debate_engine: Optional[BullBearDebateEngine] = None
        self._quote_service: Optional[StockQuoteService] = None
        self._backtester: Optional[Backtester] = None
        self._alpaca: Optional[AlpacaConnector] = None
        self._polymarket: Optional[PolymarketAgent] = None

    @property
    def debate_engine(self) -> BullBearDebateEngine:
        if self._debate_engine is None:
            self._debate_engine = BullBearDebateEngine(groq_api_key=self.groq_api_key)
        return self._debate_engine

    @property
    def quote_service(self) -> StockQuoteService:
        if self._quote_service is None:
            self._quote_service = StockQuoteService(finnhub_key=self.finnhub_key)
        return self._quote_service

    @property
    def backtester(self) -> Backtester:
        if self._backtester is None:
            self._backtester = Backtester(initial_capital=self.initial_capital)
        return self._backtester

    @property
    def alpaca(self) -> AlpacaConnector:
        if self._alpaca is None:
            self._alpaca = AlpacaConnector(
                key_id=self.alpaca_key,
                secret_key=self.alpaca_secret,
                paper=True,
            )
        return self._alpaca

    @property
    def polymarket(self) -> PolymarketAgent:
        if self._polymarket is None:
            self._polymarket = PolymarketAgent()
        return self._polymarket

    def analyze(self, symbol: str, quick_mode: bool = False) -> dict:
        """
        综合分析股票。/ Comprehensive stock analysis.

        整合行情获取 + 辩论分析 + 回测。
        """
        # 1. 获取行情
        quote = self.quote_service.get_quote(symbol)
        # 2. 运行辩论
        debate_result = {}
        try:
            debate_result = self.debate_engine.analyze(
                symbol,
                quick_mode=quick_mode,
            )
        except Exception as e:
            debate_result = {"success": False, "error": str(e)}

        # 3. 运行回测（近1年）
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y-%m-%d")
        prices = self.quote_service.get_historical(symbol, start_date, end_date)
        backtest_result = self.backtester.run(symbol, prices, strategy="buy-and-hold")

        return {
            "success": True,
            "symbol": symbol.upper(),
            "quote": {
                "price": quote.price,
                "change": quote.change,
                "change_percent": quote.change_percent,
                "volume": quote.volume,
                "market_cap": quote.market_cap,
                "sector": quote.sector,
                "industry": quote.industry,
                "source": quote.source,
            },
            "analysis": debate_result.get("result", {}),
            "backtest": {
                "success": backtest_result.success,
                "total_return_percent": backtest_result.total_return_percent,
                "final_value": backtest_result.final_value,
                "total_trades": backtest_result.metrics.total_trades,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_quote(self, symbol: str) -> StockQuote:
        """获取股票行情。"""
        return self.quote_service.get_quote(symbol)

    def backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        strategy: str = "buy-and-hold",
    ) -> BacktestResult:
        """
        运行回测。/ Run backtest.
        """
        prices = self.quote_service.get_historical(symbol, start_date, end_date)
        return self.backtester.run(symbol, prices, strategy=strategy)

    def get_polymarket_markets(
        self,
        limit: int = 50,
        category: Optional[str] = None,
    ) -> dict:
        """获取 Polymarket 市场列表。"""
        return self.polymarket.get_markets(limit=limit, category=category)

    def search_polymarket(self, query: str, limit: int = 50) -> dict:
        """搜索 Polymarket 市场。"""
        return self.polymarket.search_markets(query, limit=limit)

    # ---- Alpaca 代理方法（委托给 AlpacaConnector） ----

    def get_account(self) -> dict:
        """获取 Alpaca 账户信息。"""
        return self.alpaca.get_account()

    def get_orders(
        self,
        status: str = "all",
        limit: int = 50,
    ) -> list[dict]:
        """获取 Alpaca 订单列表。"""
        return self.alpaca.get_orders(status=status, limit=limit)

    def place_order(
        self,
        symbol: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        **kwargs,
    ) -> dict:
        """
        下单。/ Place an order.
        """
        return self.alpaca.create_order(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            **kwargs,
        )

    def cancel_order(self, order_id: str) -> dict:
        """取消订单。"""
        return self.alpaca.cancel_order(order_id)

    def get_positions(self) -> list[dict]:
        """获取持仓。"""
        return self.alpaca.get_positions()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "BullBearDebateEngine",
    "AlpacaConnector",
    "Backtester",
    "StockQuoteService",
    "PolymarketAgent",
    "AIInvestAdvisor",
    "TradeDecision",
    "BacktestResult",
    "BacktestTrade",
    "BacktestMetrics",
    "StockQuote",
    "PolymarketMarket",
    "_RestClient",
]
