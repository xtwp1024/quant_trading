"""
Polymarket API Client
=====================

Polymarket CLOB API / Gamma API 客户端，提供市场数据获取和订单操作。

端点: https://clob.polymarket.com / https://gamma-api.polymarket.com

典型用法:
    from quant_trading.arbitrage.polymarket_api import PolymarketClient

    client = PolymarketClient()
    markets = await client.get_markets()
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("polymarket_arb")


# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

CLOB_API_URL = "https://clob.polymarket.com"
GAMMA_API_URL = "https://gamma-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CHAIN_ID = 137  # Polygon Mainnet

POLYMARKET_API_KEY = ""
POLYMARKET_SECRET = ""
POLYMARKET_PASSPHRASE = ""

# Cross-platform feeds
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

CRYPTO_ASSETS = ["BTC", "ETH", "XRP", "SOL"]

MARKET_SEARCH_KEYWORDS = {
    "BTC": ["Bitcoin", "BTC", "bitcoin price"],
    "ETH": ["Ethereum", "ETH", "ethereum price"],
    "XRP": ["XRP", "Ripple", "xrp price", "XRPL"],
    "SOL": ["Solana", "SOL", "solana price"],
}

ORDER_TYPE = "FOK"
SLIPPAGE_TOLERANCE = 0.5
DRY_RUN = True


# ---------------------------------------------------------------------------
# PolymarketClient
# ---------------------------------------------------------------------------

class PolymarketClient:
    """Polymarket CLOB API 客户端.

    端点: https://clob.polymarket.com

    API Reference:
        https://docs.polymarket.com/
    """

    def __init__(self, api_key: str = ""):
        self.base_url = CLOB_API_URL
        self.gamma_url = GAMMA_API_URL
        self.api_key = api_key or POLYMARKET_API_KEY
        self.secret = POLYMARKET_SECRET
        self.passphrase = POLYMARKET_PASSPHRASE
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 100
        self._last_request_time = 0.0
        self._min_request_interval = 0.6  # ~100 req/min

    # -----------------------------------------------------------------------
    # Session & Auth
    # -----------------------------------------------------------------------

    async def _ensure_session(self) -> None:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers=self._auth_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            )

    def _auth_headers(self) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        message = timestamp + "GET" + "/auth"
        if self.secret:
            signature = hmac.new(
                base64.b64decode(self.secret),
                message.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        else:
            signature = ""
        return {
            "POLY-API-KEY": self.api_key,
            "POLY-SIGNATURE": signature,
            "POLY-TIMESTAMP": timestamp,
            "POLY-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

    async def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    async def _get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> Optional[Any]:
        await self._ensure_session()
        await self._rate_limit()
        url = f"{self.base_url}{endpoint}"
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    logger.warning("Rate limited, backing off 60s...")
                    await asyncio.sleep(60)
                    return await self._get(endpoint, params)
                else:
                    text = await resp.text()
                    logger.error(f"API error {resp.status}: {text}")
                    return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    async def _post(self, endpoint: str, data: Dict) -> Optional[Any]:
        await self._ensure_session()
        await self._rate_limit()
        url = f"{self.base_url}{endpoint}"
        try:
            async with self.session.post(url, json=data) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                else:
                    text = await resp.text()
                    logger.error(f"POST error {resp.status}: {text}")
                    return None
        except Exception as e:
            logger.error(f"POST failed: {e}")
            return None

    # -----------------------------------------------------------------------
    # Market Data
    # -----------------------------------------------------------------------

    async def get_markets(
        self,
        filter_active: bool = True,
        min_volume: float = 1000.0,
        next_cursor: str = "",
    ) -> List[Dict]:
        """获取活跃市场列表 / Fetch active markets.

        Args:
            filter_active: 仅返回活跃市场.
            min_volume: 最小成交量过滤 (USD).
            next_cursor: 分页游标.

        Returns:
            市场列表.
        """
        params: Dict[str, Any] = {}
        if filter_active:
            params["active"] = "true"
        if next_cursor:
            params["next_cursor"] = next_cursor

        data = await self._get("/markets", params)
        if data is None:
            return []

        markets = data if isinstance(data, list) else data.get("data", [])
        if min_volume > 0:
            markets = [
                m for m in markets
                if float(m.get("volume", 0) or 0) >= min_volume
            ]
        return markets

    async def get_market_by_id(self, condition_id: str) -> Optional[Dict]:
        """通过 condition_id 获取市场详情."""
        return await self._get(f"/markets/{condition_id}")

    async def search_gamma_markets(self, query: str) -> List[Dict]:
        """在 Gamma API 搜索市场."""
        await self._ensure_session()
        url = f"{self.gamma_url}/markets"
        params = {"closed": "false", "limit": 50}
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    markets = await resp.json()
                    return [
                        m for m in markets
                        if query.lower() in m.get("question", "").lower()
                        or query.lower() in m.get("description", "").lower()
                        or query.lower() in str(m.get("tags", [])).lower()
                    ]
                return []
        except Exception as e:
            logger.error(f"Gamma search failed: {e}")
            return []

    async def discover_crypto_markets(self) -> Dict[str, List[Dict]]:
        """按资产类别发现市场 / Discover markets by crypto asset.

        Returns:
            Dict[asset, List[market_dict]]
        """
        crypto_markets = {asset: [] for asset in CRYPTO_ASSETS}
        for asset, keywords in MARKET_SEARCH_KEYWORDS.items():
            for keyword in keywords:
                markets = await self.search_gamma_markets(keyword)
                for m in markets:
                    if m not in crypto_markets[asset]:
                        crypto_markets[asset].append(m)
                await asyncio.sleep(0.5)
        total = sum(len(v) for v in crypto_markets.values())
        logger.info(
            f"Discovered {total} crypto markets: "
            + ", ".join(f"{k}={len(v)}" for k, v in crypto_markets.items())
        )
        return crypto_markets

    # -----------------------------------------------------------------------
    # Price & Order Book
    # -----------------------------------------------------------------------

    async def get_order_book(self, market_id: str) -> Dict:
        """获取订单簿 / Fetch order book for a market.

        Args:
            market_id: Polymarket condition_id 或 token_id.

        Returns:
            订单簿字典，包含 bids/asks.
        """
        data = await self._get("/book", {"token_id": market_id})
        return data if data else {}

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """获取中间价 / Get midpoint price."""
        data = await self._get("/midpoint", {"token_id": token_id})
        if data and "mid" in data:
            return float(data["mid"])
        return None

    async def get_price(self, token_id: str) -> Optional[Dict]:
        """获取单个token价格 / Get price for a token."""
        return await self._get("/price", {"token_id": token_id})

    async def get_spread(self, token_id: str) -> Optional[Dict]:
        """获取买卖价差 / Get bid-ask spread."""
        return await self._get("/spread", {"token_id": token_id})

    async def get_prices_for_market(self, condition_id: str) -> Dict[str, Dict]:
        """获取市场中所有outcome的价格.

        Returns:
            Dict[outcome_name, Dict[token_id, price, bid, ask]]
        """
        market = await self._get(f"/markets/{condition_id}")
        if not market:
            return {}
        tokens = market.get("tokens", [])
        result = {}
        for token in tokens:
            outcome = token.get("outcome", "").upper()
            token_id = token.get("token_id", "")
            price_data = await self.get_price(token_id)
            if price_data:
                result[outcome] = {
                    "token_id": token_id,
                    "price": float(price_data.get("price", 0)),
                    "bid": float(price_data.get("bid", 0)),
                    "ask": float(price_data.get("ask", 0)),
                }
        return result

    # -----------------------------------------------------------------------
    # Trading
    # -----------------------------------------------------------------------

    async def place_order(
        self,
        market_id: str,
        side: str,
        amount: float,
        price: float,
        order_type: str = ORDER_TYPE,
    ) -> Optional[Dict]:
        """下单 / Place an order.

        Args:
            market_id: Token ID on Polymarket.
            side: 'yes' or 'no' (case-insensitive).
            amount: 数量 (shares).
            price: 价格 (0-1).
            order_type: 订单类型，默认 FOK.

        Returns:
            订单结果字典.
        """
        side_upper = side.upper()
        if DRY_RUN:
            logger.info(
                f"[DRY RUN] Order: {side_upper} {amount:.2f} @ ${price:.4f} "
                f"token={market_id[:16]}... type={order_type}"
            )
            return {
                "id": f"dry_run_{int(time.time() * 1000)}",
                "status": "SIMULATED",
                "side": side_upper,
                "price": price,
                "size": amount,
                "token_id": market_id,
            }

        order_data = {
            "tokenID": market_id,
            "price": price,
            "size": amount,
            "side": side_upper,
            "type": order_type,
            "feeRateBps": 0,
        }
        return await self._post("/order", order_data)

    async def place_market_order(
        self,
        market_id: str,
        side: str,
        amount_usd: float,
    ) -> Optional[Dict]:
        """市价单 / Place a market order.

        Args:
            market_id: Token ID.
            side: 'yes' or 'no'.
            amount_usd: 投入金额 (USD).

        Returns:
            订单结果.
        """
        book = await self.get_order_book(market_id)
        if not book:
            return None

        side_upper = side.upper()
        if side_upper == "BUY":
            asks = book.get("asks", [])
            if not asks:
                return None
            best_ask = float(asks[0]["price"])
            price = best_ask * (1 + SLIPPAGE_TOLERANCE / 100)
            size = amount_usd / best_ask
        else:
            bids = book.get("bids", [])
            if not bids:
                return None
            best_bid = float(bids[0]["price"])
            price = best_bid * (1 - SLIPPAGE_TOLERANCE / 100)
            size = amount_usd / best_bid

        return await self.place_order(market_id, side_upper, size, price, "FOK")

    async def cancel_order(self, order_id: str) -> Optional[Dict]:
        """取消订单 / Cancel an order."""
        if DRY_RUN:
            logger.info(f"[DRY RUN] Cancel order: {order_id}")
            return {"status": "CANCELLED"}
        return await self._post("/cancel", {"orderID": order_id})

    async def get_open_orders(self) -> List[Dict]:
        """获取活跃订单 / Get open orders."""
        data = await self._get("/orders/active")
        return data if data else []

    async def get_balance(self) -> float:
        """获取余额 / Get account balance (USDC)."""
        data = await self._get("/balances")
        if data and "balance" in data:
            return float(data["balance"])
        return 0.0

    # -----------------------------------------------------------------------
    # WebSocket
    # -----------------------------------------------------------------------

    async def subscribe_orderbook(
        self,
        token_ids: List[str],
        callback,
    ):
        """订阅订单簿 WebSocket / Subscribe to order book WebSocket."""
        import websockets
        while True:
            try:
                async with websockets.connect(WS_URL) as ws:
                    subscribe_msg = {
                        "type": "subscribe",
                        "channel": "book",
                        "assets": token_ids,
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"WebSocket subscribed to {len(token_ids)} tokens")

                    async for message in ws:
                        data = json.loads(message)
                        await callback(data)

            except Exception as e:
                logger.warning(f"WebSocket disconnected: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    async def close(self) -> None:
        """关闭 session / Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()


# ---------------------------------------------------------------------------
# CrossPlatformFeeds
# ---------------------------------------------------------------------------

class CrossPlatformFeeds:
    """跨平台价格获取 (Binance / CoinGecko 备用)."""

    COINGECKO_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "XRP": "ripple",
        "SOL": "solana",
    }
    BINANCE_SYMBOLS = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "XRP": "XRPUSDT",
        "SOL": "SOLUSDT",
    }

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self._price_cache: Dict[str, Dict] = {}

    async def _ensure_session(self) -> None:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
            )

    async def get_binance_price(self, asset: str) -> Optional[float]:
        """从 Binance 获取当前价格 / Fetch spot price from Binance."""
        symbol = self.BINANCE_SYMBOLS.get(asset)
        if not symbol:
            return None
        await self._ensure_session()
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = float(data["price"])
                    self._price_cache[asset] = {
                        "price": price,
                        "timestamp": time.time(),
                        "source": "binance",
                    }
                    return price
        except Exception as e:
            logger.error(f"Binance price fetch failed for {asset}: {e}")
        return None

    async def get_coingecko_price(self, asset: str) -> Optional[float]:
        """从 CoinGecko 获取当前价格 / Fetch spot price from CoinGecko."""
        cg_id = self.COINGECKO_IDS.get(asset)
        if not cg_id:
            return None
        await self._ensure_session()
        try:
            url = f"{COINGECKO_API_URL}/simple/price?ids={cg_id}&vs_currencies=usd"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = data[cg_id]["usd"]
                    self._price_cache[asset] = {
                        "price": price,
                        "timestamp": time.time(),
                        "source": "coingecko",
                    }
                    return float(price)
        except Exception as e:
            logger.error(f"CoinGecko price fetch failed for {asset}: {e}")
        return None

    async def get_all_prices(self) -> Dict[str, Dict]:
        """获取所有资产价格 (优先 Binance，备用 CoinGecko)."""
        results = {}
        for asset in CRYPTO_ASSETS:
            price = await self.get_binance_price(asset)
            if price is None:
                price = await self.get_coingecko_price(asset)
            if price is not None:
                results[asset] = {
                    "price": price,
                    "source": self._price_cache.get(asset, {}).get("source", "unknown"),
                    "timestamp": time.time(),
                }
        return results

    def get_cached_price(
        self,
        asset: str,
        max_age_sec: int = 30,
    ) -> Optional[float]:
        """从缓存获取价格 (30秒内有效)."""
        cached = self._price_cache.get(asset)
        if cached and (time.time() - cached["timestamp"]) < max_age_sec:
            return cached["price"]
        return None

    async def close(self) -> None:
        """关闭 session."""
        if self.session and not self.session.closed:
            await self.session.close()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "PolymarketClient",
    "CrossPlatformFeeds",
    "CLOB_API_URL",
    "GAMMA_API_URL",
    "WS_URL",
    "CHAIN_ID",
    "CRYPTO_ASSETS",
    "MARKET_SEARCH_KEYWORDS",
    "ORDER_TYPE",
    "SLIPPAGE_TOLERANCE",
    "DRY_RUN",
]
