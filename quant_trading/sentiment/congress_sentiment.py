# -*- coding: utf-8 -*-
"""
Congress Sentiment Module / 国会情绪模块
========================================
Collects and scores sentiment from US Congress stock trading data.
收集并评分美国国会股票交易数据情绪.

Data Sources / 数据来源:
    - Senate: https://efdsearch.senate.gov/search/ (Periodic Transaction Reports)
    - House:  https://disclosures-clerk.house.gov/FinancialDisclosure

Core Classes / 核心类:
    CongressDataCollector      — 国会披露网页抓取 (urllib)
    SenateTradeScraper         — 参议院交易爬虫
    HouseTradeScraper         — 众议院交易爬虫
    Politician                — 政治家数据模型
    Stock                     — 股票数据模型
    Trade                     — 交易数据模型
    CongressSentimentScorer   — 基于买卖比率的情绪打分
    SentimentFeatures         — 情绪特征容器

Key API / 关键接口:
    get_congress_sentiment()  → sentiment score (-1..+1) + confidence (0..1)

Author: Quant Trading System / 量化交易系统
Date:   2026-03-31
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import ssl
import time
import urllib.parse
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "CongressDataCollector",
    "SenateTradeScraper",
    "HouseTradeScraper",
    "Politician",
    "Stock",
    "Trade",
    "CongressSentimentScorer",
    "SentimentFeatures",
    "get_congress_sentiment",
    "load_trades_from_dicts",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Lazy Import Helpers / 延迟导入助手
# =============================================================================

def _urllib():
    """Lazy import urllib — pure Python stdlib only."""
    try:
        import urllib.request
        import urllib.error
        return urllib.request, urllib.error
    except ImportError:
        raise ImportError("urllib not available — pure Python 3 required")


def _numpy():
    """Lazy import NumPy for scoring calculations."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("numpy is required for CongressSentimentScorer")


def _pdfplumber():
    """Lazy import pdfplumber with graceful degradation."""
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        return None


# =============================================================================
# HTTP / SSL helpers
# =============================================================================

# 忽略 SSL 证书错误 (国会网站偶有证书问题)
_SSL_ctx = ssl.create_default_context()
_SSL_ctx.check_hostname = False
_SSL_ctx.verify_mode = ssl.CERT_NONE

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36 CongressSentiment/1.0"
)


def _urlopen(url: str, data: Optional[bytes] = None, timeout: int = 20) -> io.BytesIO:
    """用 urllib.urlopen 抓取页面，忽略 SSL，返回 BytesIO."""
    req_obj, err_obj = _urllib()
    req = req_obj.Request(url, data=data, headers={"User-Agent": _UA})
    try:
        resp = req_obj.urlopen(req, context=_SSL_ctx, timeout=timeout)
    except Exception:
        resp = req_obj.urlopen(req, timeout=timeout)
    return io.BytesIO(resp.read())


def _decode(b: io.BytesIO, encoding: str = "utf-8", errors: str = "replace") -> str:
    """BytesIO → str，自动处理不同编码."""
    b.seek(0)
    raw = b.read()
    for enc in (encoding, "latin-1", "cp1252"):
        try:
            return raw.decode(enc, errors=errors)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


# =============================================================================
# Data Models / 数据模型
# =============================================================================

class Politician:
    """
    US Congress politician record / 美国国会政治家记录.

    Attributes:
        name (str): Full name / 全名
        part_of_congress (str): 'House' or 'Senate' / 国会分支
        state (str): State represented / 代表州
        political_party (str): Party affiliation / 政党
        office (str): Office title / 职位
    """

    __slots__ = ('name', 'part_of_congress', 'state', 'political_party', 'office')

    def __init__(
        self,
        name: str,
        part_of_congress: str,
        state: str = "",
        political_party: str = "",
        office: str = "",
    ):
        self.name = name
        self.part_of_congress = part_of_congress
        self.state = state
        self.political_party = political_party
        self.office = office

    def __repr__(self) -> str:
        return f"<Politician {self.name} ({self.part_of_congress})>"

    def to_dict(self) -> Dict[str, str]:
        return {s: getattr(self, s) for s in self.__slots__}


class Stock:
    """
    Stock / ETF / asset record / 股票/ETF/资产记录.

    Attributes:
        ticker (str): Stock ticker symbol / 股票代码
        company_name (str): Company name / 公司名称
        asset_type (str): Type (stock, ETF, option, etc.) / 类型
    """

    __slots__ = ('ticker', 'company_name', 'asset_type')

    def __init__(
        self,
        ticker: str,
        company_name: str = "",
        asset_type: str = "stock",
    ):
        self.ticker = ticker.upper().strip()
        self.company_name = company_name
        self.asset_type = asset_type

    def __repr__(self) -> str:
        return f"<Stock {self.ticker}>"

    def to_dict(self) -> Dict[str, str]:
        return {s: getattr(self, s) for s in self.__slots__}


class Trade:
    """
    Single stock trade by a politician / 政治家单笔股票交易.

    Attributes:
        politician_name (str): Name of the politician / 政治家姓名
        stock_ticker (str): Ticker symbol / 股票代码
        purchased_or_sold (str): 'purchase' or 'sold' / 买入或卖出
        transaction_date (str): Date string YYYY-MM-DD / 交易日期
        amount (str): Amount range string / 金额范围
        chamber (str): 'House' or 'Senate' / 议院
    """

    __slots__ = (
        'politician_name',
        'stock_ticker',
        'purchased_or_sold',
        'transaction_date',
        'amount',
        'chamber',
    )

    # 常见金额区间 → 估算中值
    AMOUNT_MAP = {
        "$1,000 - $15,000": 8_000,
        "$1,001 - $15,000": 8_000,
        "$15,001 - $50,000": 32_500,
        "$50,001 - $100,000": 75_000,
        "$100,001 - $250,000": 175_000,
        "$250,001 - $500,000": 375_000,
        "$500,001 - $1,000,000": 750_000,
        "$1,000,001 - $5,000,000": 3_000_000,
        "$5,000,001 - $25,000,000": 15_000_000,
        "Over $25,000,000": 30_000_000,
    }

    def __init__(
        self,
        politician_name: str,
        stock_ticker: str,
        purchased_or_sold: str,
        transaction_date: str,
        amount: str = "",
        chamber: str = "House",
    ):
        self.politician_name = politician_name
        self.stock_ticker = stock_ticker.upper().strip()
        self.purchased_or_sold = purchased_or_sold.lower()
        self.transaction_date = transaction_date
        self.amount = amount
        self.chamber = chamber

    def __repr__(self) -> str:
        return (
            f"<Trade {self.politician_name} {self.purchased_or_sold} "
            f"{self.stock_ticker} on {self.transaction_date}>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}

    def estimated_value(self) -> float:
        """返回估算交易金额（美元），用于加权情绪计算."""
        if not self.amount:
            return 0.0
        # 直接数值
        m = re.search(r"\$?([\d,]+)", self.amount)
        if m:
            return float(m.group(1).replace(",", ""))
        # 区间匹配
        for key, val in self.AMOUNT_MAP.items():
            if key.lower() in self.amount.lower():
                return val
        return 0.0

    def sentiment_score(self) -> float:
        """
        Return numeric sentiment: +1 for purchase, -1 for sale.
        返回数值情绪：+1 买入，-1 卖出。
        """
        if self.purchased_or_sold in ('purchase', 'bought', 'acquired', 'buy'):
            return 1.0
        elif self.purchased_or_sold in ('sold', 'sale', 'disposed', 'sell'):
            return -1.0
        return 0.0


# =============================================================================
# CongressDataCollector — 抓取国会披露 (纯 urllib)
# =============================================================================

class CongressDataCollector:
    """
    国会信息披露数据采集器 / Congressional disclosure scraper.

    纯 urllib 实现，不依赖 Selenium/BeautifulSoup。

    Senate 源: https://efdsearch.senate.gov/search/
        → 搜索 "Periodic Transaction Report" 获取近期交易列表
        → 访问每条记录的详情页获取交易表格

    House 源: https://disclosures-clerk.house.gov/FinancialDisclosure
        → 搜索 PTR 记录 → 访问 PDF 链接 → 解析 PDF 内容

    Usage:
        collector = CongressDataCollector()
        trades = collector.collect_recent(days=30)
        for t in trades:
            print(t.politician_name, t.purchased_or_sold, t.stock_ticker, t.transaction_date)
    """

    SENATE_SEARCH_URL = "https://efdsearch.senate.gov/search/"
    HOUSE_SEARCH_URL = "https://disclosures-clerk.house.gov/FinancialDisclosure"

    # 交易类型关键词
    BUY_KW = {"purchase", "bought", "p", "buy"}
    SELL_KW = {"sale", "sold", "s", "sell", "short"}
    EX_KW = {"exchange", "exchanged", "e", "ex"}
    OPTION_KW = {"option", "call", "put"}

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.trades: List[Trade] = []

    # --------------------------------------------------------------------------
    # Senate scraping
    # --------------------------------------------------------------------------

    def collect_senate_recent(self, days: int = 90) -> List[Trade]:
        """抓取 Senate 最近 N 天的周期性交易报告.

        Args:
            days: 回溯天数，默认 90 天

        Returns:
            Trade 列表
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        # 1. 先访问同意页面获取 cookie
        try:
            self._senate_agree()
        except Exception as e:
            logger.error("Senate consent failed: %s", e)
            return []

        # 2. 提交搜索表单
        params = {
            "agree": "on",
            "start": from_date.strftime("%m/%d/%Y"),
            "end": to_date.strftime("%m/%d/%Y"),
            "form_id": "senate_ethics",
            "filer": "",
            "category": "periodic_transaction",
        }
        records = self._senate_search(params)

        # 3. 遍历每条记录抓取详情
        for rec in records:
            try:
                detail = self._senate_fetch_detail(rec["href"])
                self.trades.extend(detail)
            except Exception as e:
                logger.debug("Senate detail error %s: %s", rec["name"], e)
            time.sleep(0.3)

        return self.trades

    def _senate_agree(self) -> None:
        """接受 Senate EFD 使用条款，获取初始 cookie."""
        req_obj, _ = _urllib()
        data = urllib.parse.urlencode({"agree": "on"}).encode()
        req = req_obj.Request(
            self.SENATE_SEARCH_URL + "agree/",
            data=data,
            headers={"User-Agent": _UA, "Referer": self.SENATE_SEARCH_URL},
        )
        req_obj.urlopen(req, context=_SSL_ctx, timeout=self.timeout)

    def _senate_search(self, params: dict) -> List[dict]:
        """提交 Senate 搜索表单，返回记录列表 [{name, href}]."""
        post_data = {
            "agree": "on",
            "start": params["start"],
            "end": params["end"],
            "form_type": "periodic",
        }
        encoded = urllib.parse.urlencode(post_data).encode()
        req_obj, _ = _urllib()
        req = req_obj.Request(
            self.SENATE_SEARCH_URL,
            data=encoded,
            headers={
                "User-Agent": _UA,
                "Referer": self.SENATE_SEARCH_URL,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        body = _decode(_urlopen(req, timeout=self.timeout))

        # 解析记录列表 (纯 Python 正则)
        records = []
        row_pattern = re.compile(
            r'<tr[^>]*>.*?<td[^>]*>(.*?)</td>.*?'
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        for m in row_pattern.finditer(body):
            name = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            href = m.group(2).strip()
            if name and href and "ptr" in href.lower():
                records.append({"name": name, "href": href})
        return records

    def _senate_fetch_detail(self, url: str) -> List[Trade]:
        """访问 Senate PTR 详情页，解析交易表格."""
        body = _decode(_urlopen(url, timeout=self.timeout))
        trades = []

        # 尝试解析 JSON 数据
        json_match = re.search(
            r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});\s*</script>",
            body,
            re.DOTALL,
        )
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                trades = self._parse_senate_json(data)
                if trades:
                    return trades
            except Exception:
                pass

        # 回退：解析 HTML 表格
        return self._parse_senate_html_table(body)

    def _parse_senate_json(self, data: Any) -> List[Trade]:
        """从 Senate 页面的 JSON 数据中提取交易."""
        trades = []
        try:
            transactions = (
                data.get("filings", {})
                .get("currentFiling", {})
                .get("transactions", [])
            )
            for t in transactions:
                action = self._classify_action(str(t.get("type", "")))
                ticker = str(t.get("ticker", "") or "").strip()
                company = str(t.get("companyName", "") or "").strip()
                amount = str(t.get("amount", "") or "").strip()
                date = str(t.get("date", "") or "").strip()
                name = (
                    data.get("filings", {})
                    .get("currentFiling", {})
                    .get("member", {})
                    .get("name", "")
                )
                trades.append(
                    Trade(
                        politician_name=name,
                        stock_ticker=ticker,
                        purchased_or_sold=action,
                        transaction_date=self._parse_date(date),
                        amount=amount,
                        chamber="Senate",
                    )
                )
        except Exception:
            pass
        return trades

    def _parse_senate_html_table(self, body: str) -> List[Trade]:
        """从 Senate HTML 页面解析交易表格 (纯正则)."""
        trades = []

        # 提取议员姓名
        name_match = re.search(
            r'<h1[^>]*class=["\'][^"\']*member[^"\']*["\'][^>]*>([^<]+)</h1>',
            body,
            re.IGNORECASE,
        )
        politician = ""
        if name_match:
            politician = re.sub(r"<[^>]+>", "", name_match.group(1)).strip()

        # 解析表格行
        row_re = re.compile(
            r"<tr[^>]*>.*?"
            r"<td[^>]*>(.*?)</td>" * 7 +
            r".*?</tr>",
            re.DOTALL | re.IGNORECASE,
        )
        for m in row_re.finditer(body):
            cells = [
                re.sub(r"<[^>]+>", "", c).strip()
                for c in m.groups()
            ]
            if len(cells) < 7:
                continue
            date = cells[0]
            ticker = cells[1].split()[0] if cells[1] else ""
            company = " ".join(cells[1].split()[1:]) if " " in cells[1] else cells[1]
            action = self._classify_action(cells[3])
            amount = cells[4]

            if ticker or company:
                trades.append(
                    Trade(
                        politician_name=politician,
                        stock_ticker=ticker,
                        purchased_or_sold=action,
                        transaction_date=self._parse_date(date),
                        amount=amount,
                        chamber="Senate",
                    )
                )
        return trades

    # --------------------------------------------------------------------------
    # House scraping
    # --------------------------------------------------------------------------

    def collect_house_recent(self, days: int = 90) -> List[Trade]:
        """抓取 House 最近 N 天的 PTR 记录.

        流程:
            1. POST 搜索表单 → 获取 PDF 链接列表
            2. 下载每个 PDF → 提取表格数据

        注意: 需要 pdfplumber，否则跳过 PDF 解析
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        try:
            records = self._house_search(from_date, to_date)
        except Exception as e:
            logger.error("House search failed: %s", e)
            return []

        pdfplumber = _pdfplumber()

        for rec in records:
            try:
                if pdfplumber is not None:
                    trades = self._house_fetch_pdf(rec["href"])
                    self.trades.extend(trades)
            except Exception as e:
                logger.debug("House PDF error %s: %s", rec.get("name", ""), e)
            time.sleep(0.3)

        return self.trades

    def _house_search(
        self, from_date: datetime, to_date: datetime
    ) -> List[dict]:
        """POST House 搜索表单，返回 PTR 记录列表."""
        params = {
            "FilingYear": to_date.year,
            "FormType": "PTR",
            "submit": "Search",
        }
        url = self.HOUSE_SEARCH_URL + "?" + urllib.parse.urlencode(params)
        req_obj, _ = _urllib()
        req = req_obj.Request(url, headers={"User-Agent": _UA})
        body = _decode(_urlopen(req, timeout=self.timeout))

        records = []
        row_re = re.compile(
            r'<tr[^>]*>.*?'
            r'<td[^>]*>(.*?)</td>.*?'
            r'<a[^>]+href=["\']([^"\']+pdf[^"\']+)["\'][^>]*>.*?</a>',
            re.DOTALL | re.IGNORECASE,
        )
        for m in row_re.finditer(body):
            name = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            href = m.group(2).strip()
            if href and name:
                records.append({"name": name, "href": href})
        return records

    def _house_fetch_pdf(self, url: str) -> List[Trade]:
        """下载 House PDF，解析交易表格."""
        pdfplumber = _pdfplumber()
        if pdfplumber is None:
            return []

        if not url.startswith("http"):
            url = urllib.parse.urljoin(self.HOUSE_SEARCH_URL, url)

        raw = _urlopen(url, timeout=self.timeout)
        trades = []

        try:
            with pdfplumber.open(raw) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if not row or len(row) < 6:
                                continue
                            raw_date = str(row[0] or "").strip()
                            raw_ticker = str(row[1] or "").strip()
                            raw_company = str(row[2] or "").strip()
                            raw_type = str(row[3] or "").strip()
                            raw_amount = str(row[4] or "").strip()

                            action = self._classify_action(raw_type)
                            if action:
                                trades.append(
                                    Trade(
                                        politician_name="",
                                        stock_ticker=raw_ticker,
                                        purchased_or_sold=action,
                                        transaction_date=self._parse_date(raw_date),
                                        amount=raw_amount,
                                        chamber="House",
                                    )
                                )
        except Exception as e:
            logger.debug("PDF parse error %s: %s", url, e)

        return trades

    # --------------------------------------------------------------------------
    # Unified collector
    # --------------------------------------------------------------------------

    def collect_recent(
        self,
        days: int = 90,
        chambers: Tuple[str, ...] = ("senate", "house"),
    ) -> List[Trade]:
        """统一采集接口.

        Args:
            days:     回溯天数
            chambers: 采集议院 ('senate', 'house', 或两者)

        Returns:
            所有 Trade 列表
        """
        self.trades = []
        if "senate" in chambers:
            self.trades.extend(self.collect_senate_recent(days))
        if "house" in chambers:
            self.trades.extend(self.collect_house_recent(days))
        return self.trades

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _classify_action(self, raw: str) -> str:
        """根据原始字符串判断交易类型."""
        low = raw.lower()
        if any(k in low for k in self.BUY_KW):
            return "buy"
        if any(k in low for k in self.SELL_KW):
            return "sell"
        if any(k in low for k in self.EX_KW):
            return "exchange"
        if any(k in low for k in self.OPTION_KW):
            return "option"
        return ""

    def _parse_date(self, raw: str) -> str:
        """解析各种日期格式 → YYYY-MM-DD."""
        raw = raw.strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y", "%B %d, %Y", "%m-%d-%Y"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return raw


# =============================================================================
# SenateTradeScraper / 参议院爬虫
# =============================================================================

class SenateTradeScraper:
    """
    Scrape Senate financial disclosure trades (standalone).

    爬取参议院财务披露交易数据（独立使用）.

    Note: Uses urllib instead of Selenium for pure-Python compatibility.
    注意：使用 urllib 而非 Selenium 以保持纯 Python 兼容性。

    The Senate EFD search endpoint is:
    https://efdsearch.senate.gov/search/
    """

    BASE_URL = "https://efdsearch.senate.gov/search/"
    PDF_RE = re.compile(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', re.IGNORECASE)

    def __init__(self):
        self.reports: List[Dict[str, str]] = []

    def fetch_page(self, url: str, referrer: Optional[str] = None) -> bytes:
        """Fetch a URL using urllib (pure Python)."""
        req_obj, err_obj = _urllib()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 CongressSentiment/1.0"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml,*/*",
        }
        if referrer:
            headers["Referer"] = referrer
        req = req_obj.Request(url, headers=headers)
        try:
            with req_obj.urlopen(req, timeout=30) as resp:
                return resp.read()
        except err_obj.URLError as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}")

    def scrape(self, days_back: int = 365) -> List[Dict[str, str]]:
        """
        Scrape Senate trades from the last `days_back` days.

        爬取过去 `days_back` 天的参议院交易。

        Returns:
            List of dicts with keys: name, report_link
        """
        self.reports = []
        to_date = datetime.now().strftime("%m/%d/%Y")
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%m/%d/%Y")

        params = (
            f"?date_range={from_date}+-+{to_date}"
            "&category=trade"
            "&sort=date"
            "&order=desc"
        )
        search_url = self.BASE_URL + params

        try:
            html = self.fetch_page(search_url).decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"SenateTradeScraper: could not fetch search page: {exc}")
            return []

        for match in self.PDF_RE.finditer(html):
            href = match.group(1)
            name = self._extract_name_from_link(href, html, match.start())
            if name and "pdf" in href.lower():
                self.reports.append({"name": name, "report_link": href})
            time.sleep(0.5)

        return self.reports

    def _extract_name_from_link(
        self, href: str, html: str, pos: int
    ) -> Optional[str]:
        """Try to extract a human name near the PDF link."""
        window = html[max(0, pos - 300): pos + 300]
        parts = window.split()
        for i, part in enumerate(parts):
            if ".pdf" in part.lower():
                candidates = []
                for j in range(max(0, i - 4), i):
                    w = parts[j].strip(",.>\"")
                    if w and w[0].isupper() and len(w) > 1:
                        candidates.append(w)
                if candidates:
                    return " ".join(candidates[:3])
        return None


# =============================================================================
# HouseTradeScraper / 众议院爬虫
# =============================================================================

class HouseTradeScraper:
    """
    Scrape House of Representatives financial disclosure trades (standalone).

    爬取众议院财务披露交易数据（独立使用）.

    Note: Uses urllib instead of Selenium for pure-Python compatibility.
    注意：使用 urllib 而非 Selenium 以保持纯 Python 兼容性。

    The House disclosure portal is:
    https://disclosures-clerk.house.gov/FinancialDisclosure
    """

    BASE_URL = "https://disclosures-clerk.house.gov/FinancialDisclosure"
    PTR_RE = re.compile(r'href=["\']([^"\']*PTR[^"\']*)["\']', re.IGNORECASE)
    NAME_RE = re.compile(r'([A-Z][a-z]+ [A-Z][a-z]+)')

    def __init__(self):
        self.reports: List[Dict[str, str]] = []

    def fetch_page(self, url: str, referrer: Optional[str] = None) -> bytes:
        """Fetch a URL using urllib (pure Python)."""
        req_obj, err_obj = _urllib()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 CongressSentiment/1.0"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml,*/*",
        }
        if referrer:
            headers["Referer"] = referrer
        req = req_obj.Request(url, headers=headers)
        try:
            with req_obj.urlopen(req, timeout=30) as resp:
                return resp.read()
        except err_obj.URLError as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}")

    def scrape(self, year: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Scrape House trades for a given year (defaults to current year).

        爬取指定年份的众议院交易（默认为今年）。

        Returns:
            List of dicts with keys: name, office, filing_year, report_link
        """
        self.reports = []
        year = year or datetime.now().year
        search_url = f"{self.BASE_URL}/Search?FilingYear={year}"

        try:
            html = self.fetch_page(search_url).decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"HouseTradeScraper: could not fetch search page: {exc}")
            return []

        seen = set()
        for match in self.PTR_RE.finditer(html):
            href = match.group(1)
            if href in seen:
                continue
            seen.add(href)
            name = self._extract_name_from_context(html, match.start())
            self.reports.append({
                "name": name or "Unknown",
                "office": "",
                "filing_year": str(year),
                "report_link": href,
            })
            time.sleep(0.5)

        return self.reports

    def _extract_name_from_context(self, html: str, pos: int) -> Optional[str]:
        """Extract a name near the matched position."""
        window = html[max(0, pos - 400): pos + 100]
        match = self.NAME_RE.search(window)
        return match.group(1) if match else None


# =============================================================================
# SentimentFeatures / 情绪特征
# =============================================================================

class SentimentFeatures:
    """
    Container for computed sentiment features.

    计算得出的情绪特征容器。

    Attributes:
        ticker (str): Stock ticker / 股票代码
        net_sentiment (float): Raw buy-sell difference / 净买入-卖出差
        sentiment_ratio (float): Buy / (Buy + Sell) ratio / 买入比率
        purchase_count (int): Number of purchases / 买入次数
        sale_count (int): Number of sales / 卖出次数
        congress_pressure (float): Composite pressure score / 综合压力分数
        bullish_signals (List[str]): Detected bullish signals / 看涨信号
        bearish_signals (List[str]): Detected bearish signals / 看跌信号
        raw_features (Dict): All raw numpy features / 原始 NumPy 特征
    """

    __slots__ = (
        'ticker',
        'net_sentiment',
        'sentiment_ratio',
        'purchase_count',
        'sale_count',
        'congress_pressure',
        'bullish_signals',
        'bearish_signals',
        'raw_features',
    )

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.net_sentiment: float = 0.0
        self.sentiment_ratio: float = 0.5
        self.purchase_count: int = 0
        self.sale_count: int = 0
        self.congress_pressure: float = 0.0
        self.bullish_signals: List[str] = []
        self.bearish_signals: List[str] = []
        self.raw_features: Dict[str, float] = {}

    def __repr__(self) -> str:
        return (
            f"<SentimentFeatures {self.ticker} "
            f"net={self.net_sentiment:.3f} ratio={self.sentiment_ratio:.3f}>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


# =============================================================================
# CongressSentimentScorer / 情绪评分器 (Pure NumPy)
# =============================================================================

class CongressSentimentScorer:
    """
    Score stock sentiment from congressional trading activity.

    基于国会交易活动对股票进行情绪评分。

    Uses pure NumPy for all numerical calculations.

    Attributes:
        trades (List[Trade]): List of trades to score / 要评分的交易列表
        decay_half_life_days (float): Exponential decay half-life for older trades
                                      旧交易的指数衰减半衰期（天）
        weights (Dict[str, float]): Component weights for scoring / 评分权重
    """

    DEFAULT_WEIGHTS = {
        "global_ratio": 0.40,
        "politician_weighted": 0.30,
        "recent_bias": 0.20,
        "confidence": 0.10,
    }

    def __init__(
        self,
        trades: Optional[List[Trade]] = None,
        decay_half_life_days: float = 90.0,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.trades: List[Trade] = trades or []
        self.decay_half_life_days = decay_half_life_days
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._cached_score: Optional[Dict[str, Any]] = None

    def load(self, trades: List[Trade]) -> "CongressSentimentScorer":
        """Load trade data for scoring."""
        self.trades = trades
        self._cached_score = None
        return self

    def _time_decay(self, transaction_date: str) -> float:
        """
        Compute exponential decay weight for a trade date (Pure Python).

        计算交易日期的指数衰减权重。
        """
        try:
            trade_dt = datetime.strptime(transaction_date[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            return 0.0
        days_old = (datetime.now() - trade_dt).days
        if days_old < 0:
            days_old = 0
        return 0.5 ** (days_old / self.decay_half_life_days)

    def _compute_arrays(self) -> Tuple[Any, Any, Any]:
        """
        Build NumPy arrays from trade list for vectorized scoring.

        从交易列表构建 NumPy 数组以进行向量化评分。

        Returns:
            (scores, weights, amounts) — all as np.ndarray
        """
        np = _numpy()
        scores = np.array([t.sentiment_score() for t in self.trades], dtype=np.float64)
        weights = np.array(
            [self._time_decay(t.transaction_date) for t in self.trades],
            dtype=np.float64,
        )
        amounts = np.array(
            [t.estimated_value() for t in self.trades],
            dtype=np.float64,
        )
        return scores, weights, amounts

    def score(self) -> Dict[str, Any]:
        """
        Compute composite sentiment score using pure NumPy.

        使用纯 NumPy 计算综合情绪分数 (-1 =极度看空, +1 =极度看多).

        Returns:
            {
                'sentiment'  : float,  -1.0 到 +1.0，0=中性
                'confidence' : float,   0.0 到 1.0
                'components' : {...},
                'summary'    : {...},
            }
        """
        if self._cached_score:
            return self._cached_score

        n = len(self.trades)
        if n == 0:
            return {
                "sentiment": 0.0,
                "confidence": 0.0,
                "components": {},
                "summary": {},
            }

        np = _numpy()
        scores, weights, amounts = self._compute_arrays()

        # Buy/sell classification using NumPy boolean indexing
        buys_mask = scores > 0
        sells_mask = scores < 0

        buy_count = int(np.sum(buys_mask))
        sell_count = int(np.sum(sells_mask))
        total_directional = buy_count + sell_count

        # Global ratio → sentiment (-1 to +1)
        if total_directional > 0:
            global_ratio = buy_count / total_directional
        else:
            global_ratio = 0.5
        global_sentiment = float(global_ratio * 2 - 1)

        # Weighted sum: scores × time decay
        weighted_sum = float(np.dot(scores, weights))

        # Amount-weighted sentiment (value-weighted)
        total_amount = float(np.sum(amounts))
        if total_amount > 0:
            amount_weighted = float(np.dot(scores * amounts, weights)) / total_amount
        else:
            amount_weighted = 0.0

        # Recent bias: last 30 days
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        recent_trades = [t for t in self.trades if t.transaction_date >= cutoff]
        if recent_trades:
            r_scores = np.array(
                [t.sentiment_score() for t in recent_trades], dtype=np.float64
            )
            r_buys = int(np.sum(r_scores > 0))
            r_sells = int(np.sum(r_scores < 0))
            r_total = r_buys + r_sells
            recent_ratio = r_buys / r_total if r_total > 0 else 0.5
            recent_bias = float(recent_ratio * 2 - 1)
        else:
            recent_bias = 0.0

        # Confidence: sample size
        confidence_raw = float(np.clip(n / 100.0, 0.0, 1.0))

        # Weighted combination
        sentiment = (
            global_sentiment * self.weights["global_ratio"]
            + amount_weighted * self.weights["politician_weighted"]
            + recent_bias * self.weights["recent_bias"]
            + confidence_raw * self.weights["confidence"] *
              (global_sentiment if global_sentiment > 0 else 0)
        )
        sentiment = float(np.clip(sentiment, -1.0, 1.0))

        self._cached_score = {
            "sentiment": round(float(sentiment), 4),
            "confidence": round(float(confidence_raw), 4),
            "components": {
                "global_ratio": round(float(global_sentiment), 4),
                "amount_weighted": round(float(amount_weighted), 4),
                "recent_bias": round(float(recent_bias), 4),
                "confidence_raw": round(float(confidence_raw), 4),
                "net_weighted_sum": round(float(weighted_sum), 4),
            },
            "summary": {
                "total_trades": n,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "buy_ratio": round(float(global_ratio), 4),
                "total_amount": round(float(total_amount), 2),
            },
        }
        return self._cached_score

    def score_by_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Score sentiment for a single ticker.

        对单个股票进行情绪评分。

        Uses NumPy vectorized operations for efficiency.
        """
        ticker = ticker.upper()
        ticker_trades = [
            t for t in self.trades
            if t.stock_ticker.upper() == ticker
        ]
        if not ticker_trades:
            return {"sentiment": 0.0, "confidence": 0.0, "ticker": ticker}

        scorer = CongressSentimentScorer(
            ticker_trades,
            decay_half_life_days=self.decay_half_life_days,
            weights=self.weights,
        )
        result = scorer.score()
        result["ticker"] = ticker
        return result

    def score_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Score all tickers in the loaded trade list.

        对所有股票进行评分。
        """
        np = _numpy()
        tickers = list({t.stock_ticker for t in self.trades if t.stock_ticker})
        return {ticker.upper(): self.score_by_ticker(ticker) for ticker in tickers}


# =============================================================================
# Top-level convenience API
# =============================================================================

def get_congress_sentiment(
    days: int = 90,
    chambers: Tuple[str, ...] = ("senate", "house"),
    weights: Optional[Dict[str, float]] = None,
    return_trades: bool = False,
) -> Dict[str, Any]:
    """获取国会内幕交易综合情绪分（主入口）.

    Args:
        days:        回溯天数（默认 90）
        chambers:    采集议院，默认两院均采
        weights:     自定义权重 dict
        return_trades: 是否在返回值中包含原始交易列表

    Returns:
        {
            'sentiment'  : float,  -1 到 +1
            'confidence' : float,   0 到 1
            'components' : {...},
            'summary'    : {...},
            'trades'     : [Trade, ...]   # 仅当 return_trades=True
        }

    Usage:
        >>> result = get_congress_sentiment(days=30)
        >>> print(result['sentiment'], result['confidence'])
    """
    collector = CongressDataCollector()
    trades = collector.collect_recent(days=days, chambers=chambers)

    scorer = CongressSentimentScorer(trades, weights=weights)
    result = scorer.score()

    if return_trades:
        result["trades"] = trades

    return result


# =============================================================================
# Offline / mock support (for testing without network)
# =============================================================================

def load_trades_from_dicts(
    records: List[Dict[str, Any]],
) -> List[Trade]:
    """将 dict 列表转为 Trade 列表（用于测试/离线注入）.

    Args:
        records: [{politician_name, chamber, stock_ticker, purchased_or_sold, amount, transaction_date}, ...]

    Usage:
        data = [
            {"politician_name": "Nancy Pelosi", "chamber": "House",
             "stock_ticker": "AAPL", "purchased_or_sold": "buy",
             "amount": "$50,001 - $100,000", "transaction_date": "2024-03-15"},
            ...
        ]
        trades = load_trades_from_dicts(data)
    """
    return [
        Trade(
            politician_name=r.get("politician_name", ""),
            stock_ticker=r.get("stock_ticker", ""),
            purchased_or_sold=r.get("purchased_or_sold", ""),
            transaction_date=r.get("transaction_date", ""),
            amount=r.get("amount", ""),
            chamber=r.get("chamber", "House"),
        )
        for r in records
    ]
