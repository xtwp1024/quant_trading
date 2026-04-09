"""Market tools for the skill-first agent.

Domain-driven market services layer providing:
- Quote data (real-time and historical)
- News aggregation
- Macro economic data
- Market sentiment
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Quote:
    """A single asset quote."""

    symbol: str
    price: float
    change: float = 0.0
    change_pct: float = 0.0
    volume: int = 0
    timestamp: str = ""


@dataclass
class MarketNews:
    """A market news item."""

    title: str
    url: str
    source: str
    published_at: str
    summary: str = ""
    sentiment: str = "neutral"  # positive, negative, neutral


@dataclass
class MacroData:
    """Macro economic indicator."""

    indicator: str
    value: float
    unit: str
    country: str = ""
    date: str = ""


# ---------------------------------------------------------------------------
# Market tools
# ---------------------------------------------------------------------------


class MarketTools:
    """
    Market tools providing quote, news, and macro data.

    This is the domain-driven services layer for market data.
    Skills use these tools to gather information for their execution.

    Example skills:
    - Quote lookup: "what is the price of BTC"
    - News aggregation: "latest news about AAPL"
    - Macro data: "latest CPI data"
    - Sentiment: "social sentiment for NVDA"
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = self._config.get("cache_ttl", 60.0)

    # -----------------------------------------------------------------------
    # Quote tools
    # -----------------------------------------------------------------------

    async def quote(self, symbol: str) -> Quote:
        """
        Get real-time quote for a symbol.

        Supports:
        - US stocks: AAPL, NVDA, TSLA, etc.
        - Crypto: BTC, ETH, SOL
        - A-share: 000001.SZ, 600000.SH
        - HK: 0700.HK, 9988.HK
        - Metals: XAU, XAG, GLD
        """
        # Check cache
        cache_key = f"quote:{symbol}"
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if datetime.now(UTC).timestamp() - ts < self._cache_ttl:
                return data

        # Normalize symbol
        normalized = self._normalize_symbol(symbol)
        quote = await self._fetch_quote(normalized)

        # Update cache
        self._cache[cache_key] = (datetime.now(UTC).timestamp(), quote)
        return quote

    async def batch_quotes(self, symbols: list[str]) -> list[Quote]:
        """Get quotes for multiple symbols concurrently."""
        results = await asyncio.gather(*[self.quote(s) for s in symbols], return_exceptions=True)
        return [r for r in results if isinstance(r, Quote)]

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize a symbol to standard format."""
        text = str(symbol or "").strip().upper()
        # Already normalized
        if re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,3})?", text):
            return text
        # A-share 6-digit
        if re.fullmatch(r"\d{6}", text):
            if text.startswith(("0", "3")):
                return f"{text}.SZ"
            return f"{text}.SH"
        # A-share with prefix
        if re.fullmatch(r"(SH|SZ)\d{6}", text):
            return text[2:] + "." + text[:2]
        return text

    async def _fetch_quote(self, symbol: str) -> Quote:
        """Fetch quote from data sources (placeholder - integrate with real sources)."""
        # Placeholder implementation
        # In production, this would call tushare, akshare, yfinance, etc.
        return Quote(
            symbol=symbol,
            price=0.0,
            change=0.0,
            change_pct=0.0,
            volume=0,
            timestamp=_utc_now_iso(),
        )

    # -----------------------------------------------------------------------
    # News tools
    # -----------------------------------------------------------------------

    async def news(
        self,
        query: str | None = None,
        symbols: list[str] | None = None,
        limit: int = 10,
    ) -> list[MarketNews]:
        """
        Get market news filtered by query or symbols.

        Args:
            query: Text search query.
            symbols: Filter by mentioned symbols.
            limit: Maximum number of results.

        Returns:
            List of market news items.
        """
        all_news = await self._fetch_news(query=query, symbols=symbols or [], limit=limit)
        return all_news[:limit]

    async def _fetch_news(
        self,
        query: str | None = None,
        symbols: list[str] | None = None,
        limit: int = 10,
    ) -> list[MarketNews]:
        """Fetch news from sources (placeholder - integrate with news APIs)."""
        # Placeholder - in production, integrate with news APIs
        return []

    # -----------------------------------------------------------------------
    # Macro data tools
    # -----------------------------------------------------------------------

    async def macro(
        self,
        indicator: str,
        country: str = "US",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[MacroData]:
        """
        Get macro economic indicator data.

        Args:
            indicator: Indicator name (e.g. "CPI", "NFP", "PMI", "GDP")
            country: Country code (e.g. "US", "CN", "EU")
            start_date: Start date in ISO format.
            end_date: End date in ISO format.

        Returns:
            List of macro data points.
        """
        return await self._fetch_macro(indicator=indicator, country=country, start_date=start_date, end_date=end_date)

    async def _fetch_macro(
        self,
        indicator: str,
        country: str,
        start_date: str | None,
        end_date: str | None,
    ) -> list[MacroData]:
        """Fetch macro data (placeholder - integrate with macro data sources)."""
        # Placeholder - in production, integrate with FRED, wind, tushare macro
        return []

    # -----------------------------------------------------------------------
    # Market sentiment
    # -----------------------------------------------------------------------

    async def social_sentiment(self, symbol: str, platform: str = "all") -> dict[str, Any]:
        """
        Get social media sentiment for a symbol.

        Args:
            symbol: Ticker symbol.
            platform: Platform filter ("twitter", "reddit", "all").

        Returns:
            Sentiment scores and metrics.
        """
        return await self._fetch_sentiment(symbol=symbol, platform=platform)

    async def _fetch_sentiment(self, symbol: str, platform: str) -> dict[str, Any]:
        """Fetch sentiment data (placeholder)."""
        return {
            "symbol": symbol,
            "platform": platform,
            "sentiment": "neutral",
            "score": 0.0,
            "volume": 0,
            "timestamp": _utc_now_iso(),
        }

    # -----------------------------------------------------------------------
    # Market profile
    # -----------------------------------------------------------------------

    async def market_snapshot(self, market: str = "us") -> dict[str, Any]:
        """
        Get a snapshot of market conditions.

        Args:
            market: Market identifier ("us", "a-share", "hk", "crypto", "metals")

        Returns:
            Market snapshot with key metrics.
        """
        snapshots = {
            "us": await self._us_market_snapshot(),
            "a-share": await self._ashare_market_snapshot(),
            "hk": await self._hk_market_snapshot(),
            "crypto": await self._crypto_market_snapshot(),
            "metals": await self._metals_market_snapshot(),
        }
        return snapshots.get(market, snapshots["us"])

    async def _us_market_snapshot(self) -> dict[str, Any]:
        """US market snapshot (placeholder)."""
        return {
            "market": "us",
            "timestamp": _utc_now_iso(),
            "indices": {},
            "fear_greed": 50,
            "vix": 0.0,
        }

    async def _ashare_market_snapshot(self) -> dict[str, Any]:
        """A-share market snapshot (placeholder)."""
        return {
            "market": "a-share",
            "timestamp": _utc_now_iso(),
            "indices": {},
            "turnover": 0.0,
        }

    async def _hk_market_snapshot(self) -> dict[str, Any]:
        """HK market snapshot (placeholder)."""
        return {
            "market": "hk",
            "timestamp": _utc_now_iso(),
            "indices": {},
        }

    async def _crypto_market_snapshot(self) -> dict[str, Any]:
        """Crypto market snapshot (placeholder)."""
        return {
            "market": "crypto",
            "timestamp": _utc_now_iso(),
            "btc_dominance": 0.0,
            "total_mcap": 0.0,
        }

    async def _metals_market_snapshot(self) -> dict[str, Any]:
        """Metals market snapshot (placeholder)."""
        return {
            "market": "metals",
            "timestamp": _utc_now_iso(),
            "xau_usd": 0.0,
            "xag_usd": 0.0,
        }

    # -----------------------------------------------------------------------
    # Event extraction
    # -----------------------------------------------------------------------

    async def extract_events(self, text: str) -> list[dict[str, Any]]:
        """
        Extract market events from text.

        Looks for earnings dates, macro events, splits, dividends, etc.
        """
        events: list[dict[str, Any]] = []

        # Earnings patterns
        earnings_pattern = r"(\b[A-Z]{1,5}\b).*?(?:earnings|earnings? report|earnings? release|财报|业绩)"
        for match in re.finditer(earnings_pattern, text, re.IGNORECASE):
            events.append({
                "type": "earnings",
                "symbol": match.group(1),
                "matched_text": match.group(0),
            })

        # FOMC patterns
        fomc_pattern = r"(?:FOMC|fed meeting|美联储|议息会议).*?(\d{4})"
        for match in re.finditer(fomc_pattern, text, re.IGNORECASE):
            events.append({
                "type": "fomc",
                "year": match.group(1),
                "matched_text": match.group(0),
            })

        # CPI patterns
        cpi_pattern = r"(?:CPI|inflation|通胀).*?(?:data|报告|report)?"
        for match in re.finditer(cpi_pattern, text, re.IGNORECASE):
            events.append({
                "type": "macro",
                "indicator": "CPI",
                "matched_text": match.group(0),
            })

        return events

    # -----------------------------------------------------------------------
    # Source planning
    # -----------------------------------------------------------------------

    def get_source_plan(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Determine optimal data sources for a request.

        Returns a ranked list of source configurations.
        """
        markets = request.get("markets", [])
        assets = request.get("asset_classes", [])
        freshness = request.get("freshness", "realtime")

        plan: list[dict[str, Any]] = []

        # A-share sources
        if "a-share" in markets or "equity" in assets:
            plan.append({"source": "tushare", "priority": 1, "coverage": "a-share"})
            plan.append({"source": "akshare", "priority": 2, "coverage": "a-share"})

        # US sources
        if "us" in markets:
            plan.append({"source": "yfinance", "priority": 1, "coverage": "us"})

        # Crypto sources
        if "crypto" in assets:
            plan.append({"source": "binance", "priority": 1, "coverage": "crypto"})

        # Macro sources
        if "macro" in assets:
            plan.append({"source": "fred", "priority": 1, "coverage": "macro"})

        return plan


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
