"""
Kalshi Prediction Market Trading Module / Kalshi 预测市场交易模块

Provides pure urllib-based REST API client and trading logic for Kalshi prediction markets.
Kalshi 是一个经济事件预测市场平台，提供基于概率的交易市场。

Classes:
    - KalshiAPI: REST API client using urllib (synchronous)
    - KalshiMarketScanner: Scan markets for trading opportunities
    - KalshiEventAnalyzer: Analyze event probability and edge
    - KalshiTrader: Execute trades on Kalshi prediction market

Example / 示例:
    >>> api = KalshiAPI(api_key="key", private_key="pem")
    >>> scanner = KalshiMarketScanner(api)
    >>> opportunities = scanner.scan_opportunities()
"""

import hashlib
import json
import time
import base64
import ssl
import urllib.request
import urllib.error
import urllib.parse
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
import math

# Optional logging - fail silently if loguru not available
try:
    from loguru import logger
    _has_logger = True
except ImportError:
    _has_logger = False
    import sys

    class _DummyLogger:
        def info(self, *args, **kwargs): print(*args, file=sys.stdout)
        def error(self, *args, **kwargs): print(*args, file=sys.stderr)
        def warning(self, *args, **kwargs): print(*args, file=sys.stdout)
        def debug(self, *args, **kwargs): pass
    logger = _DummyLogger()


# =============================================================================
# Data Classes / 数据类
# =============================================================================

@dataclass
class MarketData:
    """单市场数据 / Single market data."""
    ticker: str
    title: str
    subtitle: str = ""
    yes_bid: float = 0.0      # 买Yes的最高价 / Highest bid for YES
    yes_ask: float = 0.0      # 卖Yes的最低价 / Lowest ask for YES
    no_bid: float = 0.0       # 买No的最高价 / Highest bid for NO
    no_ask: float = 0.0       # 卖No的最低价 / Lowest ask for NO
    volume: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    open_interest: float = 0.0
    open_time: str = ""
    close_time: str = ""
    status: str = ""

    @property
    def yes_price(self) -> float:
        """买入Yes的价格 (mid price) / Mid price for YES position."""
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.yes_bid or self.yes_ask or 0.0

    @property
    def no_price(self) -> float:
        """买入No的价格 (mid price) / Mid price for NO position."""
        if self.no_bid and self.no_ask:
            return (self.no_bid + self.no_ask) / 2.0
        return self.no_bid or self.no_ask or 0.0

    @property
    def yes_probability(self) -> float:
        """隐含Yes概率 / Implied YES probability (0-1)."""
        return self.yes_price  # Kalshi价格已经是概率形式 / Price is already in probability form

    @property
    def no_probability(self) -> float:
        """隐含No概率 / Implied NO probability (0-1)."""
        return self.no_price


@dataclass
class EventData:
    """事件数据 / Event data containing multiple markets."""
    event_ticker: str
    title: str
    subtitle: str = ""
    category: str = ""
    strike_date: str = ""
    strike_period: str = ""
    volume: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    open_interest: float = 0.0
    mutually_exclusive: bool = False
    markets: List[MarketData] = field(default_factory=list)
    time_remaining_hours: Optional[float] = None


@dataclass
class TradingOpportunity:
    """交易机会 / Trading opportunity with analysis."""
    event: EventData
    market: MarketData
    research_probability: float       # 研究预测概率 (0-1) / Research predicted probability
    market_probability: float         # 市场隐含概率 (0-1) / Market implied probability
    edge: float                       # 边缘 = 研究概率 - 市场概率 / Edge = research - market
    expected_value: float             # 期望值 EV / Expected Value
    r_score: float                   # R分数 (z-score风格) / Risk-adjusted score
    kelly_fraction: float             # Kelly fraction for position sizing
    action: str                       # "buy_yes", "buy_no", or "skip"
    confidence: float                 # 置信度 (0-1) / Confidence (0-1)
    amount: float                     # 建议投注金额 / Suggested bet amount
    reasoning: str = ""


# =============================================================================
# KalshiAPI - Pure urllib REST Client / 纯urllib REST客户端
# =============================================================================

class KalshiAPI:
    """
    Kalshi REST API客户端 (纯urllib同步版本) / Kalshi REST API client using pure urllib.

    支持RSA签名认证，支持demo和正式环境。
    Supports RSA signature authentication, demo and production environments.

    Args:
        api_key: Kalshi API访问密钥 / API access key
        private_key: RSA私钥 (PEM格式) / RSA private key in PEM format
        use_demo: 是否使用demo环境 / Whether to use demo environment
        timeout: 请求超时秒数 / Request timeout in seconds

    Example / 示例:
        >>> api = KalshiAPI(api_key="your_key", private_key="-----BEGIN RSA...")
        >>> events = api.get_events()
    """

    DEMO_BASE_URL = "https://demo-api.kalshi.co"
    PROD_BASE_URL = "https://api.kalshi.co"

    def __init__(
        self,
        api_key: str,
        private_key: str,
        use_demo: bool = True,
        timeout: int = 30
    ):
        self.api_key = api_key
        self.private_key = private_key
        self.base_url = self.DEMO_BASE_URL if use_demo else self.PROD_BASE_URL
        self.timeout = timeout
        self._session_token: Optional[str] = None

    # -------------------------------------------------------------------------
    # RSA Signing / RSA签名
    # -------------------------------------------------------------------------

    def _sign_message(self, message: str) -> str:
        """
        使用RSA私钥签名消息 / Sign message using RSA private key.

        Args:
            message: 待签名的消息字符串 / Message string to sign

        Returns:
            Base64编码的签名 / Base64-encoded signature
        """
        try:
            # Use cryptography library for RSA signing
            from cryptography.hazmat.primitives import serialization, hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.hazmat.backends import default_backend

            private_key = serialization.load_pem_private_key(
                self.private_key.encode(),
                password=None,
                backend=default_backend()
            )

            signature = private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode()

        except ImportError:
            # Fallback: use hashlib for simple HMAC (not real RSA, for testing only)
            logger.warning("cryptography not available, using HMAC fallback (NOT secure for production)")
            return base64.b64encode(
                hashlib.sha256((message + self.private_key).encode()).digest()
            ).decode()

    def _get_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        生成API请求头 (含RSA签名) / Generate request headers with RSA signature.

        Args:
            method: HTTP方法 / HTTP method (GET, POST, etc.)
            path: API路径 / API path

        Returns:
            包含认证头的字典 / Dict with auth headers
        """
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method}{path}"
        signature = self._sign_message(message)

        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json"
        }

    # -------------------------------------------------------------------------
    # HTTP Requests / HTTP请求
    # -------------------------------------------------------------------------

    def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        发起HTTP请求 / Make HTTP request.

        Args:
            method: HTTP方法 / HTTP method
            path: API路径 / API path
            data: POST请求体 / POST request body
            params: URL查询参数 / URL query parameters

        Returns:
            API响应JSON / API response JSON
        """
        headers = self._get_headers(method, path)

        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        request_data = json.dumps(data).encode() if data else None

        req = urllib.request.Request(
            url,
            data=request_data,
            headers=headers,
            method=method
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode())
                return result
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            logger.error(f"HTTP Error {e.code}: {error_body}")
            raise APIError(f"HTTP {e.code}: {error_body}", e.code)
        except urllib.error.URLError as e:
            logger.error(f"URL Error: {e.reason}")
            raise APIError(f"Network error: {e.reason}", None)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET请求 / GET request."""
        return self._make_request("GET", path, params=params)

    def post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """POST请求 / POST request."""
        return self._make_request("POST", path, data=data)

    # -------------------------------------------------------------------------
    # Events API / 事件API
    # -------------------------------------------------------------------------

    def get_events(
        self,
        limit: int = 50,
        status: str = "open",
        cursor: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        获取事件列表 / Get list of events.

        Args:
            limit: 每页数量 / Items per page (max 100)
            status: 事件状态过滤 / Event status filter ("open", "closed")
            cursor: 分页游标 / Pagination cursor

        Returns:
            (事件列表, 下一页游标) / (events list, next cursor)
        """
        params = {
            "limit": min(limit, 100),
            "status": status,
            "with_nested_markets": "true"
        }
        if cursor:
            params["cursor"] = cursor

        data = self.get("/trade-api/v2/events", params=params)
        events = data.get("events", [])
        next_cursor = data.get("cursor")

        return events, next_cursor

    def get_all_events(
        self,
        limit: int = 50,
        max_close_ts: Optional[int] = None,
        min_time_remaining_hours: float = 1.0
    ) -> List[EventData]:
        """
        获取所有事件 (带分页) / Get all events with pagination.

        Args:
            limit: 每页数量 / Items per page
            max_close_ts: 最大关闭时间戳 / Maximum close timestamp
            min_time_remaining_hours: 最小剩余时间(小时) / Minimum time remaining in hours

        Returns:
            事件数据列表 / List of EventData
        """
        all_events = []
        cursor = None
        page = 1

        while True:
            events, cursor = self.get_events(limit=limit, cursor=cursor)
            if not events:
                break

            for event in events:
                enriched = self._enrich_event(event, max_close_ts, min_time_remaining_hours)
                if enriched is not None:
                    all_events.append(enriched)

            logger.info(f"Page {page}: fetched {len(events)} events, total enriched: {len(all_events)}")

            if not cursor:
                break
            page += 1

        # Sort by 24h volume
        all_events.sort(key=lambda x: x.volume_24h, reverse=True)
        return all_events

    def _enrich_event(
        self,
        event: Dict[str, Any],
        max_close_ts: Optional[int],
        min_time_remaining_hours: float
    ) -> Optional[EventData]:
        """Enrich event with market data and filters."""
        now = datetime.now(timezone.utc)
        all_markets = event.get("markets", [])

        # Filter by close time if specified
        if max_close_ts is not None and all_markets:
            filtered = []
            for m in all_markets:
                close_str = m.get("close_time", "")
                if not close_str:
                    continue
                try:
                    close_dt = self._parse_iso_time(close_str)
                    close_ts = int(close_dt.timestamp())
                    if close_ts <= max_close_ts:
                        filtered.append(m)
                except Exception:
                    continue
            all_markets = filtered

        if not all_markets:
            return None

        # Sort by volume and take top N
        sorted_markets = sorted(all_markets, key=lambda m: m.get("volume", 0), reverse=True)
        top_markets = sorted_markets[:10]

        # Calculate totals from top markets
        total_volume = sum(m.get("volume", 0) for m in top_markets)
        total_volume_24h = sum(m.get("volume_24h", 0) for m in top_markets)
        total_liquidity = sum(m.get("liquidity", 0) for m in top_markets)
        total_oi = sum(m.get("open_interest", 0) for m in top_markets)

        # Time remaining
        time_remaining_hours = None
        strike_date_str = event.get("strike_date", "")
        if strike_date_str:
            try:
                strike_date = self._parse_iso_time(strike_date_str)
                remaining = (strike_date - now).total_seconds()
                time_remaining_hours = remaining / 3600

                if remaining > 0 and remaining < min_time_remaining_hours * 3600:
                    logger.debug(f"Event {event.get('event_ticker')} expires soon, skipping")
                    return None
            except Exception:
                pass

        # Convert markets
        markets = [
            MarketData(
                ticker=m["ticker"],
                title=m.get("title", ""),
                subtitle=m.get("subtitle", ""),
                yes_bid=m.get("yes_bid", 0),
                yes_ask=m.get("yes_ask", 0),
                no_bid=m.get("no_bid", 0),
                no_ask=m.get("no_ask", 0),
                volume=m.get("volume", 0),
                volume_24h=m.get("volume_24h", 0),
                liquidity=m.get("liquidity", 0),
                open_interest=m.get("open_interest", 0),
                open_time=m.get("open_time", ""),
                close_time=m.get("close_time", ""),
                status=m.get("status", "")
            )
            for m in top_markets
        ]

        return EventData(
            event_ticker=event.get("event_ticker", ""),
            title=event.get("title", ""),
            subtitle=event.get("sub_title", ""),
            category=event.get("category", ""),
            strike_date=strike_date_str,
            strike_period=event.get("strike_period", ""),
            volume=total_volume,
            volume_24h=total_volume_24h,
            liquidity=total_liquidity,
            open_interest=total_oi,
            mutually_exclusive=event.get("mutually_exclusive", False),
            markets=markets,
            time_remaining_hours=time_remaining_hours
        )

    # -------------------------------------------------------------------------
    # Markets API / 市场API
    # -------------------------------------------------------------------------

    def get_market(self, ticker: str) -> Optional[MarketData]:
        """
        获取指定市场数据 / Get specific market data.

        Args:
            ticker: 市场代码 / Market ticker

        Returns:
            MarketData或None / MarketData or None
        """
        try:
            data = self.get(f"/trade-api/v2/markets/{ticker}")
            m = data.get("market", {})
            return MarketData(
                ticker=m.get("ticker", ""),
                title=m.get("title", ""),
                subtitle=m.get("subtitle", ""),
                yes_bid=m.get("yes_bid", 0),
                yes_ask=m.get("yes_ask", 0),
                no_bid=m.get("no_bid", 0),
                no_ask=m.get("no_ask", 0),
                volume=m.get("volume", 0),
                volume_24h=m.get("volume_24h", 0),
                liquidity=m.get("liquidity", 0),
                open_interest=m.get("open_interest", 0),
                open_time=m.get("open_time", ""),
                close_time=m.get("close_time", ""),
                status=m.get("status", "")
            )
        except Exception as e:
            logger.error(f"Error getting market {ticker}: {e}")
            return None

    def get_markets_for_event(self, event_ticker: str, limit: int = 50) -> List[MarketData]:
        """
        获取事件的所有市场 / Get all markets for an event.

        Args:
            event_ticker: 事件代码 / Event ticker
            limit: 限制数量 / Limit count

        Returns:
            MarketData列表 / List of MarketData
        """
        try:
            data = self.get("/trade-api/v2/markets", params={
                "event_ticker": event_ticker,
                "status": "open",
                "limit": min(limit, 100)
            })
            markets = data.get("markets", [])
            return [
                MarketData(
                    ticker=m.get("ticker", ""),
                    title=m.get("title", ""),
                    subtitle=m.get("subtitle", ""),
                    yes_bid=m.get("yes_bid", 0),
                    yes_ask=m.get("yes_ask", 0),
                    no_bid=m.get("no_bid", 0),
                    no_ask=m.get("no_ask", 0),
                    volume=m.get("volume", 0),
                    volume_24h=m.get("volume_24h", 0),
                    liquidity=m.get("liquidity", 0),
                    open_interest=m.get("open_interest", 0),
                    open_time=m.get("open_time", ""),
                    close_time=m.get("close_time", ""),
                    status=m.get("status", "")
                )
                for m in markets
            ]
        except Exception as e:
            logger.error(f"Error getting markets for event {event_ticker}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Portfolio API / 投资组合API
    # -------------------------------------------------------------------------

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        获取用户持仓 / Get user positions.

        Returns:
            持仓列表 / List of positions
        """
        try:
            data = self.get("/trade-api/v2/portfolio/positions")
            return data.get("market_positions", [])
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def has_position(self, ticker: str) -> bool:
        """
        检查是否持有指定市场仓位 / Check if user has position in market.

        Args:
            ticker: 市场代码 / Market ticker

        Returns:
            是否有持仓 / Whether position exists
        """
        positions = self.get_positions()
        for pos in positions:
            if pos.get("ticker") == ticker:
                size = pos.get("position", 0)
                if size != 0:
                    return True
        return False

    def place_order(
        self,
        ticker: str,
        side: str,
        amount: float
    ) -> Dict[str, Any]:
        """
        下单 / Place an order.

        Args:
            ticker: 市场代码 / Market ticker
            side: "yes" 或 "no"
            amount: 金额(美元) / Amount in USD

        Returns:
            订单结果 / Order result dict
        """
        import uuid

        client_order_id = str(uuid.uuid4())
        buy_max_cost_cents = int(amount * 100)
        max_contracts = 1000

        order_data = {
            "ticker": ticker,
            "side": side,
            "action": "buy",
            "type": "market",
            "client_order_id": client_order_id,
            "count": max_contracts,
            "buy_max_cost": buy_max_cost_cents
        }

        try:
            result = self.post("/trade-api/v2/portfolio/orders", data=order_data)
            order_id = result.get("order_id", "")
            logger.info(f"Order placed: {ticker} {side} ${amount} -> order_id={order_id}")
            return {"success": True, "order_id": order_id, "client_order_id": client_order_id}
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # Utility / 工具
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_iso_time(time_str: str) -> datetime:
        """解析ISO8601时间字符串 / Parse ISO8601 time string."""
        if time_str.endswith('Z'):
            time_str = time_str[:-1] + '+00:00'
        dt = datetime.fromisoformat(time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


class APIError(Exception):
    """API错误异常 / API error exception."""

    def __init__(self, message: str, code: Optional[int]):
        super().__init__(message)
        self.code = code


# =============================================================================
# KalshiMarketScanner - Scan Prediction Markets / 扫描预测市场
# =============================================================================

class KalshiMarketScanner:
    """
    扫描Kalshi预测市场寻找交易机会 / Scan Kalshi prediction markets for trading opportunities.

    扫描事件市场，收集价格数据，识别潜在边缘。
    Scans events and markets, collects pricing data, identifies potential edges.

    Args:
        api: KalshiAPI实例 / KalshiAPI instance
        min_volume_24h: 最小24小时交易量过滤 / Minimum 24h volume filter
        min_liquidity: 最小流动性过滤 / Minimum liquidity filter

    Example / 示例:
        >>> scanner = KalshiMarketScanner(api)
        >>> opportunities = scanner.scan_opportunities()
    """

    def __init__(
        self,
        api: KalshiAPI,
        min_volume_24h: float = 1000.0,
        min_liquidity: float = 500.0
    ):
        self.api = api
        self.min_volume_24h = min_volume_24h
        self.min_liquidity = min_liquidity

    def scan_opportunities(
        self,
        max_events: int = 50,
        max_close_ts: Optional[int] = None,
        min_time_remaining_hours: float = 1.0
    ) -> List[EventData]:
        """
        扫描市场寻找机会 / Scan markets for opportunities.

        Args:
            max_events: 最大分析事件数 / Maximum number of events to analyze
            max_close_ts: 最大关闭时间戳 / Maximum close timestamp
            min_time_remaining_hours: 最小剩余时间 / Minimum time remaining

        Returns:
            EventData列表 / List of EventData with market opportunities
        """
        logger.info("Starting market scan...")

        events = self.api.get_all_events(
            limit=max_events,
            max_close_ts=max_close_ts,
            min_time_remaining_hours=min_time_remaining_hours
        )

        # Filter by volume and liquidity
        filtered_events = []
        for event in events:
            if event.volume_24h < self.min_volume_24h:
                continue
            if event.liquidity < self.min_liquidity:
                continue

            # Filter markets with insufficient pricing
            valid_markets = [
                m for m in event.markets
                if m.yes_bid > 0 and m.yes_ask > 0
            ]

            if not valid_markets:
                continue

            event.markets = valid_markets
            filtered_events.append(event)

        logger.info(f"Scan complete: {len(filtered_events)} events passed filters")
        return filtered_events

    def get_market_opportunities(
        self,
        event: EventData,
        top_n: int = 5
    ) -> List[MarketData]:
        """
        获取事件中的最佳机会市场 / Get best opportunity markets from an event.

        Args:
            event: 事件数据 / Event data
            top_n: 返回前N个 / Return top N

        Returns:
            MarketData列表 (按交易量排序) / List sorted by volume
        """
        # Sort by volume * liquidity as proxy for opportunity quality
        scored = [
            (m, m.volume * (m.yes_bid + m.no_bid) / 2)
            for m in event.markets
            if m.yes_bid > 0 and m.yes_ask > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_n]]


# =============================================================================
# KalshiEventAnalyzer - Analyze Event Probability and Edge / 分析事件概率和边缘
# =============================================================================

class KalshiEventAnalyzer:
    """
    分析事件概率和边缘 / Analyze event probability and edge.

    基于研究概率和市场隐含概率计算交易边缘和期望值。
    Calculate trading edge and expected value based on research vs market probabilities.

    Args:
        z_threshold: R分数阈值 (默认1.5) / R-score threshold (default 1.5)
        min_edge: 最小边缘阈值 / Minimum edge threshold

    Example / 示例:
        >>> analyzer = KalshiEventAnalyzer(z_threshold=1.5)
        >>> analysis = analyzer.analyze_market(market, research_prob=0.65)
    """

    def __init__(
        self,
        z_threshold: float = 1.5,
        min_edge: float = 0.05
    ):
        self.z_threshold = z_threshold
        self.min_edge = min_edge

    def calculate_ev(
        self,
        research_prob: float,
        market_prob: float,
        bet_amount: float = 1.0,
        side: str = "yes"
    ) -> float:
        """
        计算期望值 / Calculate expected value.

        EV = P(win) * payoff - P(lose) * stake

        Args:
            research_prob: 研究预测概率 (0-1) / Research predicted probability
            market_prob: 市场隐含概率 (0-1) / Market implied probability
            bet_amount: 投注金额 / Bet amount
            side: 投注方向 "yes" 或 "no" / Bet side "yes" or "no"

        Returns:
            期望值 / Expected value
        """
        if side == "yes":
            # Buy YES: win if outcome is YES
            # Cost: market_prob (you pay market_prob to buy YES)
            # Payout: 1.0 (you get $1 if YES)
            net_payout = 1.0 - market_prob  # profit on winning
            p_win = research_prob
            p_lose = 1 - research_prob
            stake = market_prob  # what you pay
        else:
            # Buy NO: win if outcome is NO
            net_payout = 1.0 - market_prob
            p_win = 1 - research_prob
            p_lose = research_prob
            stake = market_prob

        ev = p_win * net_payout * bet_amount - p_lose * stake * bet_amount
        return ev

    def calculate_r_score(
        self,
        research_prob: float,
        market_prob: float
    ) -> float:
        """
        计算R分数 (风险调整边缘) / Calculate R-score (risk-adjusted edge).

        R = (p - y) / sqrt(p * (1-p))

        这类似于z-score，用于衡量边缘的统计显著性。
        Similar to z-score, measures statistical significance of edge.

        Args:
            research_prob: 研究预测概率 (0-1) / Research predicted probability
            market_prob: 市场隐含概率 (0-1) / Market implied probability

        Returns:
            R分数 / R-score
        """
        edge = research_prob - market_prob
        std_dev = math.sqrt(market_prob * (1 - market_prob))

        if std_dev < 1e-10:
            return 0.0

        return edge / std_dev

    def calculate_kelly_fraction(
        self,
        research_prob: float,
        market_prob: float,
        kelly_fraction: float = 0.5
    ) -> float:
        """
        计算Kelly fraction用于仓位大小 / Calculate Kelly fraction for position sizing.

        f* = (bp - q) / b
        where b = net payout odds, p = win probability, q = 1-p

        Args:
            research_prob: 研究预测概率 (0-1) / Research predicted probability
            market_prob: 市场隐含概率 (0-1) / Market implied probability
            kelly_fraction: Kelly分数折扣 (0.5 = 半Kelly) / Kelly fraction discount (0.5 = half-Kelly)

        Returns:
            Kelly fraction (0-1) / Kelly fraction
        """
        if market_prob <= 0 or market_prob >= 1:
            return 0.0

        # For YES bet
        b = (1.0 - market_prob) / market_prob  # net odds
        p = research_prob
        q = 1 - p

        if b * p - q <= 0:
            return 0.0

        kelly = (b * p - q) / b
        kelly = max(0.0, min(1.0, kelly))  # clamp to [0, 1]

        return kelly * kelly_fraction

    def analyze_market(
        self,
        market: MarketData,
        research_prob: float,
        confidence: float = 0.5,
        kelly_fraction: float = 0.5,
        max_bet: float = 100.0,
        bankroll: float = 1000.0
    ) -> Optional[TradingOpportunity]:
        """
        分析单个市场的交易机会 / Analyze trading opportunity for a single market.

        Args:
            market: 市场数据 / Market data
            research_prob: 研究预测概率 (0-1) / Research predicted probability
            confidence: 置信度 (0-1) / Confidence (0-1)
            kelly_fraction: Kelly fraction to apply
            max_bet: 最大投注金额 / Maximum bet amount
            bankroll: 总资金用于Kelly计算 / Total bankroll for Kelly sizing

        Returns:
            TradingOpportunity或None / TradingOpportunity or None
        """
        market_prob = market.yes_probability
        edge = research_prob - market_prob

        # Calculate R-score
        r_score = self.calculate_r_score(research_prob, market_prob)

        # Determine action
        if edge > 0 and r_score >= self.z_threshold:
            action = "buy_yes"
        elif edge < 0 and r_score <= -self.z_threshold:
            # For NO bet, edge is reversed
            action = "buy_no"
            research_prob = 1 - research_prob
            market_prob = market.no_probability
            edge = research_prob - market_prob
            r_score = self.calculate_r_score(research_prob, market_prob)
        else:
            return None

        # Calculate Kelly fraction
        kelly = self.calculate_kelly_fraction(
            research_prob, market_prob, kelly_fraction
        )

        # Calculate position size
        max_kelly_bet = bankroll * kelly
        amount = min(max_bet, max_kelly_bet)

        if amount < 1.0:
            return None

        # Calculate EV
        ev = self.calculate_ev(
            research_prob, market_prob, amount, "yes" if action == "buy_yes" else "no"
        )

        return TradingOpportunity(
            event=EventData(event_ticker="", title=""),  # Placeholder
            market=market,
            research_prob=research_prob,
            market_prob=market_prob,
            edge=abs(edge),
            expected_value=ev,
            r_score=abs(r_score),
            kelly_fraction=kelly,
            action=action,
            confidence=confidence,
            amount=amount,
            reasoning=f"R={r_score:.2f}, edge={edge:.2%}, kelly={kelly:.2%}"
        )

    def rank_opportunities(
        self,
        opportunities: List[TradingOpportunity],
        top_n: int = 10
    ) -> List[TradingOpportunity]:
        """
        对机会进行排序 / Rank opportunities.

        按R分数和EV综合排序。
        Sort by combined R-score and EV.

        Args:
            opportunities: 机会列表 / List of opportunities
            top_n: 返回前N个 / Return top N

        Returns:
            排序后的机会列表 / Sorted opportunity list
        """
        scored = [
            (opp, opp.r_score * 0.6 + abs(opp.expected_value) * 100 * 0.4)
            for opp in opportunities
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [opp for opp, _ in scored[:top_n]]


# =============================================================================
# KalshiTrader - Execute Trades / 执行交易
# =============================================================================

class KalshiTrader:
    """
    Kalshi预测市场交易器 / Kalshi prediction market trader.

    整合扫描、分析和交易功能。
    Integrates scanning, analysis, and trading functionality.

    Args:
        api: KalshiAPI实例 / KalshiAPI instance
        scanner: KalshiMarketScanner实例 / KalshiMarketScanner instance
        analyzer: KalshiEventAnalyzer实例 / KalshiEventAnalyzer instance
        dry_run: 是否模拟交易 / Whether to simulate trading
        max_bet: 单笔最大投注 / Maximum bet per trade
        bankroll: 总资金 / Total bankroll

    Example / 示例:
        >>> trader = KalshiTrader(api, dry_run=True)
        >>> results = trader.find_odds_edge()
        >>> trader.place_bet(results[0])
    """

    def __init__(
        self,
        api: KalshiAPI,
        scanner: Optional[KalshiMarketScanner] = None,
        analyzer: Optional[KalshiEventAnalyzer] = None,
        dry_run: bool = True,
        max_bet: float = 100.0,
        bankroll: float = 1000.0,
        kelly_fraction: float = 0.5
    ):
        self.api = api
        self.scanner = scanner or KalshiMarketScanner(api)
        self.analyzer = analyzer or KalshiEventAnalyzer()
        self.dry_run = dry_run
        self.max_bet = max_bet
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction

        # Track placed trades
        self._traded_tickers: set = set()

    def find_odds_edge(
        self,
        research_data: Dict[str, float],
        max_events: int = 50,
        skip_existing: bool = True,
        max_close_ts: Optional[int] = None
    ) -> List[TradingOpportunity]:
        """
        寻找存在边缘的Odds机会 / Find odds edge opportunities.

        Args:
            research_data: 研究数据字典 {ticker: probability} / Research data dict
            max_events: 最大扫描事件数 / Max events to scan
            skip_existing: 是否跳过已有仓位 / Whether to skip existing positions
            max_close_ts: 最大关闭时间戳 / Max close timestamp

        Returns:
            TradingOpportunity列表 / List of TradingOpportunity
        """
        # Scan markets
        events = self.scanner.scan_opportunities(
            max_events=max_events,
            max_close_ts=max_close_ts
        )

        opportunities = []
        for event in events:
            for market in event.markets:
                # Skip if no research data
                if market.ticker not in research_data:
                    continue

                # Skip if already traded in this session
                if skip_existing and market.ticker in self._traded_tickers:
                    logger.info(f"Skipping {market.ticker} - already traded")
                    continue

                # Check for existing position
                if skip_existing and self.api.has_position(market.ticker):
                    logger.info(f"Skipping {market.ticker} - has existing position")
                    continue

                research_prob = research_data[market.ticker]

                opp = self.analyzer.analyze_market(
                    market=market,
                    research_prob=research_prob,
                    confidence=0.5,
                    kelly_fraction=self.kelly_fraction,
                    max_bet=self.max_bet,
                    bankroll=self.bankroll
                )

                if opp is not None:
                    opp.event = event
                    opportunities.append(opp)

        # Rank and return
        ranked = self.analyzer.rank_opportunities(opportunities)
        logger.info(f"Found {len(ranked)} opportunities from {len(opportunities)} analyzed")
        return ranked

    def calculate_ev(
        self,
        market: MarketData,
        research_prob: float,
        side: str = "yes",
        amount: Optional[float] = None
    ) -> float:
        """
        计算期望值 / Calculate expected value.

        Args:
            market: 市场数据 / Market data
            research_prob: 研究概率 (0-1) / Research probability
            side: 投注方向 / Bet side ("yes" or "no")
            amount: 投注金额 / Bet amount

        Returns:
            期望值 / Expected value
        """
        market_prob = market.yes_probability if side == "yes" else market.no_probability
        amount = amount or self.max_bet
        return self.analyzer.calculate_ev(research_prob, market_prob, amount, side)

    def place_bet(
        self,
        opportunity: TradingOpportunity,
        amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        执行投注 / Place a bet.

        Args:
            opportunity: 交易机会 / Trading opportunity
            amount: 投注金额 (默认使用机会建议金额) / Bet amount (default uses opportunity amount)

        Returns:
            交易结果 / Trade result
        """
        amount = amount or opportunity.amount

        if self.dry_run:
            logger.info(f"[DRY RUN] Would place: {opportunity.market.ticker} {opportunity.action} ${amount:.2f}")
            return {
                "success": True,
                "dry_run": True,
                "ticker": opportunity.market.ticker,
                "action": opportunity.action,
                "amount": amount,
                "expected_value": opportunity.expected_value,
                "r_score": opportunity.r_score
            }

        # Extract side from action
        side = "yes" if opportunity.action == "buy_yes" else "no"

        # Place the order
        result = self.api.place_order(opportunity.market.ticker, side, amount)

        if result.get("success"):
            self._traded_tickers.add(opportunity.market.ticker)

        result["opportunity"] = {
            "ticker": opportunity.market.ticker,
            "action": opportunity.action,
            "amount": amount,
            "expected_value": opportunity.expected_value,
            "r_score": opportunity.r_score
        }

        return result

    def batch_place_bets(
        self,
        opportunities: List[TradingOpportunity],
        total_budget: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        批量投注 / Place multiple bets.

        Args:
            opportunities: 机会列表 / List of opportunities
            total_budget: 总预算 (默认使用bankroll) / Total budget (default uses bankroll)

        Returns:
            结果列表 / List of results
        """
        total_budget = total_budget or self.bankroll
        results = []

        # Calculate position sizes proportionally
        total_r_score = sum(opp.r_score for opp in opportunities)
        if total_r_score <= 0:
            return results

        for opp in opportunities:
            # Allocate proportionally by R-score
            allocation = total_budget * (opp.r_score / total_r_score)
            allocation = min(allocation, self.max_bet)  # Cap at max bet

            if allocation < 1.0:
                continue

            result = self.place_bet(opp, amount=allocation)
            results.append(result)

            # Update budget
            total_budget -= allocation if result.get("success") else 0

            if total_budget <= 0:
                break

        return results


# =============================================================================
# Module Exports / 模块导出
# =============================================================================

__all__ = [
    "KalshiAPI",
    "KalshiMarketScanner",
    "KalshiEventAnalyzer",
    "KalshiTrader",
    "TradingOpportunity",
    "MarketData",
    "EventData",
    "APIError"
]
