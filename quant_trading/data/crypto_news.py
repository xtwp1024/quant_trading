"""Crypto exchange news crawler for 量化之神.

Adapted from crypto_exchange_news_crawler (https://github.com/lowweihong/crypto-exchange-news-crawler).

Architecture:
- ExchangeNewsSpider: base spider class for exchange-specific scrapers
- BinanceNewsSpider, BybitNewsSpider, OKXNewsSpider, BitgetNewsSpider, etc.: exchange-specific stubs
- CryptoNewsAggregator: aggregate and deduplicate news across all exchanges
- CryptoNewsCrawler: high-level facade for the crawler subsystem

Supported exchanges: Binance, Bybit, OKX, Bitget, Kraken, Kucoin, Mexc, Upbit,
                    Bitfinex, BingX, XT, Crypto.com, Deepcoin

Key features:
- Structured news data: news_id, title, desc, url, category_str, exchange, announced_at_timestamp, timestamp
- Scrapy-based crawler architecture with optional Playwright for dynamic content
- JSON/CSV/XML output support
- Rate limiting (DOWNLOAD_DELAY) and proxy support
- Multi-exchange aggregation with deduplication

Note: This module provides the crawler architecture and stub spiders. To enable live
scraping, install dependencies: scrapy, scrapy-playwright, playwright.
"""

from __future__ import annotations

import csv
import json
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlencode

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CryptoNewsItem:
    """Structured news item from a crypto exchange announcement feed."""

    news_id: str
    title: str
    desc: str
    url: str
    category_str: str
    exchange: str
    announced_at_timestamp: int
    timestamp: int = field(default_factory=lambda: int(time.time()))
    sentiment_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CryptoNewsItem":
        return cls(**d)


# ---------------------------------------------------------------------------
# Base Spider
# ---------------------------------------------------------------------------

# Default user-agent list (mirrors the original project)
DEFAULT_USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.166 Safari/537.36",
    "Mozilla/5.0 (Windows 7 Enterprise; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6099.71 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
]

DEFAULT_MAX_PAGE = 2
DEFAULT_DOWNLOAD_DELAY = 3.0


class ExchangeNewsSpider(ABC):
    """Abstract base spider for exchange-specific news scrapers.

    Subclasses must implement the ``start_requests`` and ``parse`` methods.
    The crawler architecture mirrors Scrapy semantics but without requiring
    the full Scrapy runtime -- spiders yield ``CryptoNewsItem`` dicts.
    """

    name: str = "exchange_news_spider"

    # Configuration (can be overridden per-spider or via settings dict)
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "USER_AGENT": DEFAULT_USER_AGENTS,
        "MAX_PAGE": DEFAULT_MAX_PAGE,
        "DOWNLOAD_DELAY": DEFAULT_DOWNLOAD_DELAY,
        "PROXY_LIST": [],
    })

    headers: Dict[str, str] = field(default_factory=dict)

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        if settings:
            self.settings = {**self.settings, **settings}

    def random_ua(self) -> str:
        return random.choice(self.settings.get("USER_AGENT", DEFAULT_USER_AGENTS))

    def random_proxy(self) -> Optional[str]:
        proxies = self.settings.get("PROXY_LIST", [])
        return random.choice(proxies) if proxies else None

    @abstractmethod
    def start_requests(self) -> Iterator[Any]:
        """Yield initial requests for the spider. Yields scrapy.Request-like objects."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, response: Any) -> Iterator[Dict[str, Any]]:
        """Parse a response and yield CryptoNewsItem dicts."""
        raise NotImplementedError

    def make_item(
        self,
        news_id: str,
        title: str,
        url: str,
        category_str: str,
        desc: str = "",
        announced_at_timestamp: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Helper to build a standardised news item dict."""
        return {
            "news_id": news_id,
            "title": title,
            "desc": desc,
            "url": url,
            "category_str": category_str,
            "exchange": self.name,
            "announced_at_timestamp": announced_at_timestamp or int(time.time()),
            "timestamp": int(time.time()),
        }

    def output_json(self, items: List[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def output_csv(self, items: List[Dict[str, Any]], path: str) -> None:
        if not items:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0].keys()))
            writer.writeheader()
            writer.writerows(items)

    def output_xml(self, items: List[Dict[str, Any]], path: str) -> None:
        root = ET.Element("news")
        for item in items:
            el = ET.SubElement(root, "item")
            for k, v in item.items():
                child = ET.SubElement(el, k)
                child.text = str(v)
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Exchange-specific spider stubs
# ---------------------------------------------------------------------------


class BinanceNewsSpider(ExchangeNewsSpider):
    """Spider stub for Binance support announcements.

    Live endpoint: ``https://www.binance.com/bapi/apex/v1/public/apex/cms/article/list/query``
    """

    name = "binance"

    headers = {
        "accept": "*/*",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "clienttype": "web",
        "content-type": "application/json",
        "lang": "en",
        "priority": "u=1, i",
        "referer": "https://www.binance.com/en/support/announcement/",
        "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    params = {"type": "1", "pageNo": "1", "pageSize": "50"}
    url = "https://www.binance.com/bapi/apex/v1/public/apex/cms/article/list/query?"

    def start_requests(self) -> Iterator[Any]:
        for page in range(int(self.settings.get("MAX_PAGE", DEFAULT_MAX_PAGE))):
            params = {**self.params, "pageNo": str(page + 1)}
            yield _make_request(
                self.url + urlencode(params),
                headers={**self.headers, "user-agent": self.random_ua()},
                callback=self.parse,
            )

    def parse(self, response: Any) -> Iterator[Dict[str, Any]]:
        data = response.get("data", {})
        for catalog in data.get("catalogs", []):
            for article in catalog.get("articles", []):
                yield self.make_item(
                    news_id=article["code"],
                    title=article["title"],
                    url=f"https://www.binance.com/en/support/announcement/detail/{article['code']}",
                    category_str=catalog.get("catalogName", ""),
                    announced_at_timestamp=article.get("releaseDate", 0) // 1000,
                )


class BybitNewsSpider(ExchangeNewsSpider):
    """Spider stub for Bybit announcements.

    Live endpoint: ``https://api2.bybit.com/announcements/api/search/v1/index/announcement-posts_en``
    """

    name = "bybit"

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "content-type": "application/json;charset=UTF-8",
        "origin": "https://announcements.bybit.com",
        "priority": "u=1, i",
        "referer": "https://announcements.bybit.com/",
        "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    url = "https://api2.bybit.com/announcements/api/search/v1/index/announcement-posts_en"

    def start_requests(self) -> Iterator[Any]:
        for page in range(self.settings.get("MAX_PAGE", DEFAULT_MAX_PAGE)):
            body = json.dumps({"data": {"query": "", "page": page, "hitsPerPage": 8}})
            yield _make_request(
                self.url,
                headers={**self.headers, "user-agent": self.random_ua()},
                method="POST",
                body=body,
                callback=self.parse,
            )

    def parse(self, response: Any) -> Iterator[Dict[str, Any]]:
        for r in response.get("result", {}).get("hits", []):
            yield self.make_item(
                news_id=r["url"].split("-")[-1].replace("/", ""),
                title=r["title"],
                desc=r.get("description", ""),
                url=f"https://announcements.bybit.com{r['url']}",
                category_str=r.get("category", {}).get("key", ""),
                announced_at_timestamp=int(r.get("date_timestamp", 0)),
            )


class OKXNewsSpider(ExchangeNewsSpider):
    """Spider stub for OKX help centre announcements.

    Live endpoint: ``https://www.okx.com/help/category/announcements``
    """

    name = "okx"

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "priority": "u=0, i",
        "referer": "https://www.okx.com/en-sg/help/section/announcements-trading-updates/",
        "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "upgrade-insecure-requests": "1",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    url_section = "https://www.okx.com/help/section/"

    def start_requests(self) -> Iterator[Any]:
        yield _make_request(
            "https://www.okx.com/help/category/announcements",
            headers={**self.headers, "user-agent": self.random_ua()},
            callback=self._get_section,
        )

    def _get_section(self, response: Any) -> Iterator[Dict[str, Any]]:
        # Extracts section list from __NEXT_DATA__ JSON embedded in the page.
        try:
            next_data = _extract_next_data(response)
            sections = next_data.get("appContext", {}).get("initialProps", {}).get("sectionData", {}).get("allSections", [])
        except Exception:
            return

        for section in sections:
            cat_str = section.get("title", "")
            slug = section.get("slug", "")
            yield _make_request(
                self.url_section + slug,
                headers={**self.headers, "user-agent": self.random_ua()},
                callback=self.parse,
                cb_kwargs={"params": {"cat_str": cat_str, "page": 1, "section_slug": slug}},
            )

    def parse(self, response: Any, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        try:
            next_data = _extract_next_data(response)
            arc_ls = next_data.get("appContext", {}).get("initialProps", {}).get("sectionData", {}).get("articleList", {})
        except Exception:
            return

        for item in arc_ls.get("items", []):
            yield self.make_item(
                news_id=str(item.get("id", "")),
                title=item.get("title", ""),
                url=f"https://www.okx.com/help/{item.get('slug', '')}",
                category_str=(params or {}).get("cat_str", ""),
                desc="",
            )


class BitgetNewsSpider(ExchangeNewsSpider):
    """Spider stub for Bitget support centre announcements.

    Uses Scrapy + Playwright for dynamic content rendering.
    Live URL: ``https://www.bitget.com/en/support/sections/``
    """

    name = "bitget"
    # Playwright is required for Bitget; this stub shows the intended architecture.
    # To enable: pip install scrapy-playwright playwright && playwright install chromium
    use_playwright = True

    section_map: Dict[str, str] = {
        "360007868532": "Latest News.Bitget News",
        "5955813039257": "New Listings.Spot",
        "12508313405000": "New Listings.Futures",
        "12508313443168": "New Listings.Margin",
        "12508313405075": "New Listings.Copy Trading",
        "12508313443194": "New Listings.Bots",
        "4413154768537": "Competitions and promotions.Ongoing competitions and promotions",
        "4413127530649": "Competitions and promotions.Previous competitions & events",
        "4411481755417": "Competitions and promotions.Reward Distribution",
        "6483596785177": "Competitions and promotions.KCGI",
        "12508313446623": "Maintenance or system updates.Asset maintenance",
        "12508313404850": "Maintenance or system updates.Spot Maintenance",
        "12508313405050": "Maintenance or system updates.System Updates",
        "12508313404950": "Maintenance or system updates.Futures Maintenance",
    }

    def start_requests(self) -> Iterator[Any]:
        for section_id, cat in self.section_map.items():
            for page in range(int(self.settings.get("MAX_PAGE", DEFAULT_MAX_PAGE))):
                url = f"https://www.bitget.com/en/support/sections/{section_id}"
                if page > 0:
                    url += f"/{page + 1}"
                yield _make_request(
                    url,
                    headers={"user-agent": self.random_ua()},
                    callback=self.parse,
                    cb_kwargs={"cat": cat},
                    meta={"playwright": True},
                )

    def parse(self, response: Any, cat: str = "") -> Iterator[Dict[str, Any]]:
        # The actual implementation extracts __NEXT_DATA__ JSON and parses
        # props["pageProps"]["sectionArticle"]["items"].
        # Stub returns the intended field mapping:
        try:
            containers = response.get("items", [])
            for content in containers:
                yield self.make_item(
                    news_id=content.get("contentId", ""),
                    title=content.get("title", ""),
                    url=f"https://www.bitget.com/support/articles/{content.get('contentId', '')}",
                    category_str=cat,
                    announced_at_timestamp=int(content.get("showTime", 0)) // 1000,
                )
        except Exception:
            pass


class KrakenNewsSpider(ExchangeNewsSpider):
    """Spider stub for Kraken blog posts (WordPress REST API).

    Live endpoint: ``https://blog.kraken.com/wp-json/wp/v2/posts``
    """

    name = "kraken"

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "priority": "u=0, i",
        "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "upgrade-insecure-requests": "1",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    def start_requests(self) -> Iterator[Any]:
        # First fetch categories, then posts
        yield _make_request(
            "https://blog.kraken.com/wp-json/wp/v2/categories?per_page=100",
            headers={**self.headers, "user-agent": self.random_ua()},
            callback=self._parse_category,
        )

    def _parse_category(self, response: Any) -> Iterator[Dict[str, Any]]:
        self.asset_dict: Dict[int, str] = {r["id"]: r["name"] for r in response}
        params = {"per_page": 50, "page": 1, "order": "desc", "orderby": "date"}
        for page in range(self.settings.get("MAX_PAGE", DEFAULT_MAX_PAGE)):
            params["page"] = page + 1
            yield _make_request(
                "https://blog.kraken.com/wp-json/wp/v2/posts?" + urlencode(params),
                headers={**self.headers, "user-agent": self.random_ua()},
                callback=self.parse,
            )

    asset_dict: Dict[int, str] = field(default_factory=dict)

    def parse(self, response: Any) -> Iterator[Dict[str, Any]]:
        for item in response:
            cats = ",".join(self.asset_dict.get(i, "") for i in item.get("categories", []))
            yield self.make_item(
                news_id=str(item.get("id", "")),
                title=item.get("title", {}).get("rendered", ""),
                desc=item.get("excerpt", {}).get("rendered", ""),
                url=f"https://blog.kraken.com/?p={item.get('id', '')}",
                category_str=cats,
                announced_at_timestamp=int(
                    datetime.strptime(item.get("date", ""), "%Y-%m-%dT%H:%M:%S").timestamp()
                ) if item.get("date") else 0,
            )


class KucoinNewsSpider(ExchangeNewsSpider):
    """Spider stub for KuCoin announcements.

    Live endpoint: ``https://www.kucoin.com/_api/cms/articles``
    """

    name = "kucoin"

    category_dict = {
        "latest-announcements": "Latest Announcements",
        "activities": "Latest Events",
        "new-listings": "New Listings",
        "product-updates": "Product Updates",
        "vip": "Institutions and VIPs",
        "maintenance-updates": "System Maintenance",
        "delistings": "Delisting",
        "others": "Others",
    }

    headers = {
        "accept": "application/json",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "priority": "u=1, i",
        "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-site": "global",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    params = {"category": "", "lang": "en_US", "page": "1", "pageSize": "10"}

    def start_requests(self) -> Iterator[Any]:
        for cat, cat_name in self.category_dict.items():
            p = {**self.params, "category": cat}
            h = {**self.headers, "referer": f"https://www.kucoin.com/announcement/{cat}", "user-agent": self.random_ua()}
            yield _make_request(
                "https://www.kucoin.com/_api/cms/articles?" + urlencode(p),
                headers=h,
                callback=self.parse,
                cb_kwargs={"params": {"cat_name": cat_name, "cat_slug": cat, "page": 1}},
            )

    def parse(self, response: Any, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        p = params or {}
        for item in response.get("items", []):
            yield self.make_item(
                news_id=item.get("id", ""),
                title=item.get("title", ""),
                desc=item.get("summary", ""),
                url=f"https://www.kucoin.com/announcement{item.get('path', '')}",
                category_str=p.get("cat_name", ""),
                announced_at_timestamp=item.get("publish_ts", 0),
            )


class MexcNewsSpider(ExchangeNewsSpider):
    """Spider stub for MEXC announcements.

    Live endpoint: ``https://www.mexc.com/help/announce/api/en-US/section/{id}/articles``
    """

    name = "mexc"

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "priority": "u=0, i",
        "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "upgrade-insecure-requests": "1",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    def start_requests(self) -> Iterator[Any]:
        yield _make_request(
            "https://www.mexc.com/help/announce/api/en-US/section/360000254192/sections",
            headers={**self.headers, "user-agent": self.random_ua()},
            callback=self._get_section,
        )

    def _get_section(self, response: Any) -> Iterator[Dict[str, Any]]:
        for r in response.get("data", []):
            sid = r.get("id", "")
            yield _make_request(
                f"https://www.mexc.com/help/announce/api/en-US/section/{sid}/articles?"
                + urlencode({"page": "1", "perPage": "10"}),
                headers={**self.headers, "user-agent": self.random_ua()},
                callback=self.parse,
                meta={"section_id": sid, "page": "1"},
            )

    def parse(self, response: Any) -> Iterator[Dict[str, Any]]:
        data = response.get("data", {})
        for r in data.get("results", []):
            yield self.make_item(
                news_id=str(r.get("id", "")),
                title=r.get("title", ""),
                url=f"https://www.mexc.com/support/articles/{r.get('id', '')}",
                category_str=", ".join(i.get("name", "") for i in r.get("parentSections", [])),
                announced_at_timestamp=int(
                    datetime.strptime(r.get("createdAt", ""), "%Y-%m-%dT%H:%M:%SZ").timestamp()
                ) if r.get("createdAt") else 0,
            )


class UpbitNewsSpider(ExchangeNewsSpider):
    """Spider stub for Upbit announcements.

    Supports: sg (Singapore), id (Indonesia), th (Thailand), kr (Korea).
    Live endpoint: ``https://{country}-api-manager.upbit.com/api/v1/announcements``
    """

    name = "upbit"

    def __init__(self, country: str = "sg", settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings)
        if country.lower() not in ("sg", "id", "th", "kr"):
            raise ValueError("Country must be sg, id, th, or kr")
        self.country = country.lower()
        self._setup_headers()

    def _setup_headers(self) -> None:
        lang_map = {"sg": "en-SG, en;q=1, en-GB;q=0.1", "id": "id-ID, id;q=1, en-GB;q=0.1", "th": "th-TH, th;q=1, en-GB;q=0.1", "kr": "ko-KR, ko;q=1, en-GB;q=0.1"}
        self.headers = {
            "accept": "application/json",
            "accept-language": lang_map.get(self.country, "en-SG, en;q=1"),
            "origin": f"https://{'kr' if self.country == 'kr' else self.country}.upbit.com",
            "priority": "u=1, i",
            "referer": f"https://{'kr' if self.country == 'kr' else self.country}.upbit.com/",
            "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": self.random_ua(),
        }
        if self.country == "kr":
            self.url = "https://api-manager.upbit.com/api/v1/announcements?"
        else:
            self.url = f"https://{self.country}-api-manager.upbit.com/api/v1/announcements?"

    def start_requests(self) -> Iterator[Any]:
        params = {"os": "web", "page": "1", "per_page": "20", "category": "all"}
        yield _make_request(
            self.url + urlencode(params),
            headers={**self.headers, "user-agent": self.random_ua()},
            callback=self.parse,
            cb_kwargs={"params": params},
        )

    def parse(self, response: Any, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        data = response.get("data", {})
        url_prefix = f"https://{'kr' if self.country == 'kr' else self.country}.upbit.com/service_center/notice?id="
        for r in data.get("notices", []):
            yield self.make_item(
                news_id=str(r.get("id", "")),
                title=r.get("title", ""),
                url=url_prefix + str(r.get("id", "")),
                category_str=r.get("category", ""),
                announced_at_timestamp=int(
                    datetime.strptime(r.get("listed_at", ""), "%Y-%m-%dT%H:%M:%S%z").timestamp()
                ) if r.get("listed_at") else 0,
            )


class BitfinexNewsSpider(ExchangeNewsSpider):
    """Spider stub for Bitfinex posts.

    Live endpoint: ``https://api-pub.bitfinex.com/v2/posts/hist/``
    """

    name = "bitfinex"

    headers = {
        "accept": "*/*",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "origin": "https://www.bitfinex.com",
        "priority": "u=1, i",
        "referer": "https://www.bitfinex.com/",
        "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    params = {"limit": 20, "type": 1}
    url = "https://api-pub.bitfinex.com/v2/posts/hist/?"

    def start_requests(self) -> Iterator[Any]:
        yield _make_request(
            self.url + urlencode(self.params),
            headers={**self.headers, "user-agent": self.random_ua()},
            callback=self.parse,
            cb_kwargs={"params": self.params},
        )

    def parse(self, response: Any, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        for item in response:
            yield self.make_item(
                news_id=str(item[0]),
                title=item[3],
                desc=item[4] if len(item) > 4 else "",
                url=f"https://www.bitfinex.com/posts/{item[0]}",
                category_str="",
                announced_at_timestamp=item[1] // 1000 if len(item) > 1 else 0,
            )


class BingXNewsSpider(ExchangeNewsSpider):
    """Spider stub for BingX support notices.

    Uses Scrapy + Playwright for dynamic content.
    Live URL: ``https://bingx.com/en/support/notice-center/``
    """

    name = "bingx"
    use_playwright = True

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    def start_requests(self) -> Iterator[Any]:
        yield _make_request(
            "https://bingx.com/en/support/notice-center/",
            headers={"user-agent": self.random_ua()},
            callback=self._get_sections,
            meta={"playwright": True},
        )

    def _get_sections(self, response: Any) -> Iterator[Dict[str, Any]]:
        # Intercept AJAX call to listSections endpoint
        for section in response.get("sections", []):
            sid = section.get("sectionId", "")
            sname = section.get("sectionName", "")
            yield _make_request(
                f"https://bingx.com/en/support/notice-center/{sid}/",
                headers={"user-agent": self.random_ua()},
                callback=self.parse,
                cb_kwargs={"section_name": sname},
                meta={"playwright": True},
            )

    def parse(self, response: Any, section_name: str = "") -> Iterator[Dict[str, Any]]:
        for article in response.get("articles", []):
            yield self.make_item(
                news_id=str(article.get("articleId", "")),
                title=article.get("title", ""),
                url=f"https://bingx.com/en/support/articles/{article.get('articleId', '')}",
                category_str=section_name,
                announced_at_timestamp=int(
                    datetime.fromisoformat(article.get("createTime", "2000-01-01T00:00:00Z").replace("Z", "+00:00")).timestamp()
                ),
            )


class XTNewsSpider(ExchangeNewsSpider):
    """Spider stub for XT.com support announcements.

    Uses Scrapy + Playwright for dynamic Zendesk content.
    Live URL: ``https://xtsupport.zendesk.com/hc/en-us/categories/10304894611993-Important-Announcements``
    """

    name = "xt"
    use_playwright = True

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    def start_requests(self) -> Iterator[Any]:
        yield _make_request(
            "https://xtsupport.zendesk.com/hc/en-us/categories/10304894611993-Important-Announcements",
            headers={"user-agent": self.random_ua()},
            callback=self._get_sections,
            meta={"playwright": True},
        )

    def _get_sections(self, response: Any) -> Iterator[Dict[str, Any]]:
        for h2 in response.get("section_titles", []):
            cat = h2.get("text", "").strip()
            href = h2.get("href", "")
            for page in range(int(self.settings.get("MAX_PAGE", DEFAULT_MAX_PAGE))):
                url = href if page == 0 else f"{href}?page={page + 1}#articles"
                yield _make_request(
                    url,
                    headers={"user-agent": self.random_ua()},
                    callback=self.parse,
                    cb_kwargs={"category_str": cat},
                    meta={"playwright": True},
                )

    def parse(self, response: Any, category_str: str = "") -> Iterator[Dict[str, Any]]:
        for article in response.get("articles", []):
            yield self.make_item(
                news_id=article.get("id", ""),
                title=article.get("title", ""),
                desc=article.get("body", ""),
                url=article.get("url", ""),
                category_str=category_str,
                announced_at_timestamp=int(
                    datetime.fromisoformat(article.get("created_at", "2000-01-01T00:00:00Z").replace("Z", "+00:00")).timestamp()
                ),
            )


class CryptocomNewsSpider(ExchangeNewsSpider):
    """Spider stub for Crypto.com exchange announcements.

    Live endpoint: ``https://static2.crypto.com/exchange/announcements_en.json``
    """

    name = "cryptocom"

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "origin": "https://crypto.com",
        "priority": "u=1, i",
        "referer": "https://crypto.com/",
        "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    def start_requests(self) -> Iterator[Any]:
        yield _make_request(
            "https://static2.crypto.com/exchange/announcements_en.json",
            headers={**self.headers, "user-agent": self.random_ua()},
            callback=self.parse,
        )

    def parse(self, response: Any) -> Iterator[Dict[str, Any]]:
        for r in response:
            yield self.make_item(
                news_id=str(r.get("id", "")),
                title=r.get("title", ""),
                desc=r.get("content", ""),
                url=f"https://crypto.com/exchange/announcements/{r.get('category', '')}/{r.get('id', '')}",
                category_str=f"{r.get('category', '')}.{r.get('productType', '')}",
                announced_at_timestamp=int(r.get("announcedAt", 0)) // 1000,
            )


class DeepcoinNewsSpider(ExchangeNewsSpider):
    """Spider stub for Deepcoin support centre announcements.

    Live URL: ``https://support.deepcoin.online/hc/en-001/categories/360003875752-Important-Announcements``
    """

    name = "deepcoin"

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "priority": "u=0, i",
        "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "upgrade-insecure-requests": "1",
        "user-agent": random.choice(DEFAULT_USER_AGENTS),
    }

    def start_requests(self) -> Iterator[Any]:
        yield _make_request(
            "https://support.deepcoin.online/hc/en-001/categories/360003875752-Important-Announcements",
            headers={**self.headers, "user-agent": self.random_ua()},
            callback=self._get_sections,
        )

    def _get_sections(self, response: Any) -> Iterator[Dict[str, Any]]:
        for section in response.get("sections", []):
            sname = section.get("name", "")
            surl = section.get("url", "")
            yield _make_request(
                surl,
                headers={**self.headers, "user-agent": self.random_ua()},
                callback=self._parse_sections,
                cb_kwargs={"section_name": sname, "section_url": surl, "page": 1},
            )

    def _parse_sections(self, response: Any, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        for article in response.get("article_list", []):
            yield _make_request(
                article.get("url", ""),
                headers={**self.headers, "user-agent": self.random_ua()},
                callback=self.parse,
                cb_kwargs={"params": params},
            )

    def parse(self, response: Any, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        yield self.make_item(
            news_id=str(response.get("id", "")),
            title=response.get("title", ""),
            desc=response.get("body", ""),
            url=response.get("url", ""),
            category_str=(params or {}).get("section_name", ""),
            announced_at_timestamp=int(
                datetime.strptime(response.get("created_at", "2000-01-01T00:00:00Z"), "%Y-%m-%dT%H:%M:%SZ").timestamp()
            ) if response.get("created_at") else 0,
        )


# ---------------------------------------------------------------------------
# Spider registry
# ---------------------------------------------------------------------------

SPIDER_REGISTRY: Dict[str, type] = {
    "binance": BinanceNewsSpider,
    "bybit": BybitNewsSpider,
    "okx": OKXNewsSpider,
    "bitget": BitgetNewsSpider,
    "kraken": KrakenNewsSpider,
    "kucoin": KucoinNewsSpider,
    "mexc": MexcNewsSpider,
    "upbit": UpbitNewsSpider,
    "bitfinex": BitfinexNewsSpider,
    "bingx": BingXNewsSpider,
    "xt": XTNewsSpider,
    "cryptocom": CryptocomNewsSpider,
    "deepcoin": DeepcoinNewsSpider,
}


def get_spider(name: str, **kwargs) -> ExchangeNewsSpider:
    """Factory: instantiate a spider by exchange name."""
    cls = SPIDER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown spider: {name}. Available: {list(SPIDER_REGISTRY.keys())}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# CryptoNewsAggregator
# ---------------------------------------------------------------------------


class CryptoNewsAggregator:
    """Aggregate and deduplicate news items across multiple exchanges.

    Usage:
        aggregator = CryptoNewsAggregator()
        aggregator.add_spider("binance")
        aggregator.add_spider("bybit")
        items = aggregator.fetch_all()
        aggregator.save("news.json", format="json")
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = settings or {}
        self.spiders: List[ExchangeNewsSpider] = []
        self._all_items: List[Dict[str, Any]] = []

    def add_spider(self, name: str, **kwargs) -> None:
        self.spiders.append(get_spider(name, settings={**self.settings, **kwargs}))

    def add_spider_instance(self, spider: ExchangeNewsSpider) -> None:
        self.spiders.append(spider)

    def fetch_all(self) -> List[CryptoNewsItem]:
        """Run all registered spiders and collect items."""
        self._all_items = []
        for spider in self.spiders:
            try:
                for req in spider.start_requests():
                    # In a live Scrapy runtime, the engine would schedule the request.
                    # Here we return a stub response for demonstration.
                    response = _stub_fetch(req)
                    if response is not None:
                        for item in spider.parse(response):
                            self._all_items.append(item)
            except Exception:
                # Broad catch to prevent one spider from breaking the aggregator.
                pass
        return [CryptoNewsItem.from_dict(it) for it in self._all_items]

    def deduplicate(self, items: Optional[List[CryptoNewsItem]] = None) -> List[CryptoNewsItem]:
        """Remove duplicate news items by (exchange, news_id) pair."""
        seen: set = set()
        result: List[CryptoNewsItem] = []
        for item in (items or [CryptoNewsItem.from_dict(it) for it in self._all_items]):
            key = (item.exchange, item.news_id)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    def save(self, path: str, format: str = "json", items: Optional[List[CryptoNewsItem]] = None) -> None:
        """Save items to a file in the specified format (json, csv, xml)."""
        data = [(items or [CryptoNewsItem.from_dict(it) for it in self._all_items])]
        data_list = [it.to_dict() for it in (items or self.deduplicate())]
        p = Path(path)
        if format == "json":
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
        elif format == "csv":
            if data_list:
                with open(p, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(data_list[0].keys()))
                    writer.writeheader()
                    writer.writerows(data_list)
        elif format == "xml":
            try:
                import xml.etree.ElementTree as ET
                root = ET.Element("news")
                for item in data_list:
                    el = ET.SubElement(root, "item")
                    for k, v in item.items():
                        child = ET.SubElement(el, k)
                        child.text = str(v)
                ET.indent(root)
                ET.ElementTree(root).write(str(p), encoding="utf-8", xml_declaration=True)
            except ImportError:
                raise RuntimeError("xml.etree.ElementTree is not available")
        else:
            raise ValueError(f"Unknown format: {format}. Use json, csv, or xml.")


# ---------------------------------------------------------------------------
# CryptoNewsCrawler -- high-level facade
# ---------------------------------------------------------------------------


class CryptoNewsCrawler:
    """High-level facade for the crypto news crawler subsystem.

    Provides a simple interface to:
    - Crawl news from one or more exchanges
    - Aggregate and deduplicate results
    - Save to JSON / CSV / XML
    - Retrieve raw CryptoNewsItem list for use in trading strategies

    Example:
        crawler = CryptoNewsCrawler()
        crawler.crawl("binance", "bybit", "okx")
        crawler.save("news.json", format="json")
        items = crawler.items  # List[CryptoNewsItem] ready for strategy use
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = settings or {}
        self.aggregator = CryptoNewsAggregator(settings=self.settings)
        self._items: List[CryptoNewsItem] = []

    @property
    def items(self) -> List[CryptoNewsItem]:
        return self._items

    def crawl(self, *exchange_names: str) -> "CryptoNewsCrawler":
        """Crawl news from the specified exchanges."""
        for name in exchange_names:
            self.aggregator.add_spider(name)
        self._items = self.aggregator.fetch_all()
        return self

    def deduplicate(self) -> "CryptoNewsCrawler":
        """Deduplicate the current items list."""
        self._items = self.aggregator.deduplicate(self._items)
        return self

    def save(self, path: str, format: str = "json") -> "CryptoNewsCrawler":
        """Save current items to a file."""
        self.aggregator.save(path, format, items=self._items)
        return self

    def filter_by_category(self, keyword: str) -> List[CryptoNewsItem]:
        """Return items whose category_str contains the keyword (case-insensitive)."""
        kw = keyword.lower()
        return [it for it in self._items if kw in it.category_str.lower()]

    def filter_by_exchange(self, exchange: str) -> List[CryptoNewsItem]:
        """Return items from a specific exchange."""
        return [it for it in self._items if it.exchange == exchange.lower()]

    def filter_by_time_range(self, since_timestamp: int, until_timestamp: Optional[int] = None) -> List[CryptoNewsItem]:
        """Return items within a unix timestamp range."""
        until = until_timestamp or int(time.time())
        return [it for it in self._items if since_timestamp <= it.announced_at_timestamp <= until]

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [it.to_dict() for it in self._items]


# ---------------------------------------------------------------------------
# Internal helpers (Scrapy-compatible stubs for use outside full Scrapy runtime)
# ---------------------------------------------------------------------------

# xml.etree import deferred to avoid top-level dependency on Python stdlib only
import xml.etree.ElementTree as ET


class _StubRequest:
    """Minimal request-like object used when running outside Scrapy."""

    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        callback: Optional[Any] = None,
        cb_kwargs: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.url = url
        self.method = method
        self.headers = headers or {}
        self.body = body
        self.callback = callback
        self.cb_kwargs = cb_kwargs or {}
        self.meta = meta or {}


class _StubResponse:
    """Minimal response-like object used when running outside Scrapy."""

    def __init__(self, url: str, body: Any, headers: Optional[Dict[str, str]] = None, status: int = 200):
        self.url = url
        self.body = body
        self.headers = headers or {}
        self.status = status

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self.body, dict):
            return self.body.get(key, default)
        return default


def _make_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    callback: Optional[Any] = None,
    cb_kwargs: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> _StubRequest:
    return _StubRequest(
        url=url,
        method=method,
        headers=headers,
        body=body,
        callback=callback,
        cb_kwargs=cb_kwargs,
        meta=meta,
    )


def _extract_next_data(response: Any) -> Dict[str, Any]:
    """Extract __NEXT_DATA__ JSON from a response dict (used by OKX spider)."""
    return response.get("__NEXT_DATA__", {})


async def _stub_fetch(req: _StubRequest) -> Optional[_StubResponse]:
    """Stub fetcher that returns a placeholder empty response.

    In a live environment, replace with an actual HTTP client
    (e.g. httpx, aiohttp) or use the full Scrapy runtime.
    """
    # Return a no-op empty response so stubs can iterate without crashing
    return _StubResponse(url=req.url, body={})


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "CryptoNewsCrawler",
    "CryptoNewsAggregator",
    "CryptoNewsItem",
    "ExchangeNewsSpider",
    "BinanceNewsSpider",
    "BybitNewsSpider",
    "OKXNewsSpider",
    "BitgetNewsSpider",
    "KrakenNewsSpider",
    "KucoinNewsSpider",
    "MexcNewsSpider",
    "UpbitNewsSpider",
    "BitfinexNewsSpider",
    "BingXNewsSpider",
    "XTNewsSpider",
    "CryptocomNewsSpider",
    "DeepcoinNewsSpider",
    "SPIDER_REGISTRY",
    "get_spider",
]
