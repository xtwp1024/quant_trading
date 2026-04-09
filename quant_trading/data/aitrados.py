"""
Aitrados Unified Multi-Exchange Data Client — quant-trading integration.

Abstraction layer built on top of the aitrados-api (D:/Hive/Data/trading_repos/aitrados-api/).
Provides a single interface for fetching OHLCV, order book, ticker, and funding rate data
across 10+ exchanges: Binance, OKX, Bybit, Deribit, Kraken, Gate.io, Huobi, KuCoin, Bitget, MEXC.

Pure Python (urllib + websocket-client). No heavy exchange SDK dependencies.

Classes
-------
AitradosClient
    Unified REST client with authentication, retry, and rate-limiting.
ExchangeDataFetcher
    High-level fetcher that translates exchange-specific API responses into
    a normalised dict / list-of-dicts format.
UnifiedDataFrame
    Lightweight DataFrame-like container (pure-Python, no pandas required)
    that normalises column names, types, and timestamps from any exchange.
WebSocketDataStream
    Async WebSocket consumer for real-time ticker / trade / order-book deltas.

Bilingual docstrings: English primary, Chinese secondary in parentheses.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import threading
import urllib.request
import urllib.error
import urllib.parse
import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_logger = logging.getLogger("aitrados")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AitradosError(Exception):
    """Base exception for all aitrados errors."""
    pass

class AitradosAuthError(AitradosError):
    """Authentication or API-key error."""
    pass

class AitradosRateLimitError(AitradosError):
    """Rate limit exceeded."""
    def __init__(self, message: str, retry_after: float = 5.0):
        super().__init__(message)
        self.retry_after = retry_after

class AitradosValidationError(AitradosError):
    """Invalid parameters or exchange symbol."""
    pass

class AitradosConnectionError(AitradosError):
    """Network / connection failure."""
    pass


# ---------------------------------------------------------------------------
# Constants — supported exchanges and REST API base
# ---------------------------------------------------------------------------

SUPPORTED_EXCHANGES: List[str] = [
    "binance", "okx", "bybit", "deribit",
    "kraken", "gateio", "huobi", "kucoin",
    "bitget", "mexc",
]

# aitrados-api REST base (from source repo constants)
_AITRADOS_REST_BASE = "https://default.dataset-api.aitrados.com/api"
_AITRADOS_WS_BASE   = "wss://realtime.dataset-sub.aitrados.com/ws"

# Interval name mapping (from source IntervalName)
INTERVAL_MAP: Dict[str, str] = {
    "1m": "1M", "3m": "3M", "5m": "5M", "10m": "10M",
    "15m": "15M", "30m": "30M",
    "1h": "60M", "2h": "120M", "4h": "240M",
    "1d": "DAY", "1w": "WEEK", "1mon": "MON",
}
_REVERSE_INTERVAL_MAP: Dict[str, str] = {v: k for k, v in INTERVAL_MAP.items()}


# ---------------------------------------------------------------------------
# Rate Limiter  (thread-safe sliding window, SQLite-free)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """
    Thread-safe sliding-window rate limiter.
    Keeps requests within per-second and per-minute budgets.
    """

    def __init__(self, requests_per_second: float = 10.0, requests_per_minute: float = 120.0):
        self.rps = requests_per_second
        self.rpm = requests_per_minute
        self._history: List[float] = []
        self._lock = threading.Lock()

    def _clean_history(self, now: float) -> None:
        """Drop timestamps older than 60 s."""
        cutoff = now - 60.0
        self._history = [t for t in self._history if t > cutoff]

    def should_allow(self) -> Tuple[bool, str]:
        now = time.time()
        with self._lock:
            self._clean_history(now)
            rps_ok = len([t for t in self._history if t > now - 1.0]) < self.rps
            rpm_ok = len(self._history) < self.rpm
            if rps_ok and rpm_ok:
                return True, ""
            if not rps_ok:
                return False, "per-second"
            return False, "per-minute"

    def wait_time(self) -> float:
        now = time.time()
        with self._lock:
            self._clean_history(now)
            if not self._history:
                return 0.0
            oldest_1s = [t for t in self._history if t > now - 1.0]
            if len(oldest_1s) < self.rps:
                return 0.0
            # time until oldest 1-s request expires
            return max(0.0, 1.0 - (now - self._history[0]) + 0.01)

    def record(self) -> None:
        with self._lock:
            self._history.append(time.time())

    def handle_429(self, response_text: str) -> float:
        # Simple back-off: 1 s initial, exponential on repeated 429s
        try:
            data = json.loads(response_text)
            wait = float(data.get("retry_after", 1.0))
        except Exception:
            wait = 1.0
        return max(wait, 0.5)


# ---------------------------------------------------------------------------
# HTTP Client — pure urllib, no httpx
# ---------------------------------------------------------------------------

class _HttpClient:
    """
    Thin HTTP wrapper using urllib.request.
    Handles JSON encode/decode, query-params, headers, retry, and timeout.
    """

    DEFAULT_HEADERS: Dict[str, str] = {
        "User-Agent": "Aitrados-QuantTrader/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    def __init__(
        self,
        secret_key: str,
        base_url: str = _AITRADOS_REST_BASE,
        timeout: int = 30,
        max_retries: int = 5,
        rate_limiter: Optional[_RateLimiter] = None,
        debug: bool = False,
    ):
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = rate_limiter or _RateLimiter()
        self.debug = debug

    # ------------------------------------------------------------------
    # Low-level request
    # ------------------------------------------------------------------

    def _build_url(self, path: str, params: Dict[str, Any]) -> str:
        """Merge path + query-string params."""
        if params:
            qs = urllib.parse.urlencode(self._signed_params(params))
            return f"{self.base_url}/{path.lstrip('/')}?{qs}"
        return f"{self.base_url}/{path.lstrip('/')}"

    def _signed_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Append secret_key (used by aitrados-api auth)."""
        p = dict(params)
        p["secret_key"] = self.secret_key
        return p

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a synchronous HTTP request with retry and rate-limit handling.
        Returns parsed JSON dict.
        """
        params = params or {}
        attempt = 0

        while attempt < self.max_retries:
            # Rate-limit check
            allowed, limit_type = self.rate_limiter.should_allow()
            if not allowed:
                wait = self.rate_limiter.wait_time()
                _logger.warning(
                    f"Aitrados rate-limit ({limit_type}). "
                    f"Waiting {wait:.1f}s before retry."
                )
                time.sleep(math.ceil(wait))

            allowed, _ = self.rate_limiter.should_allow()
            if not allowed:
                time.sleep(1.0)

            attempt += 1
            try:
                url = self._build_url(path, params)
                if self.debug:
                    _logger.debug(f"[{method}] {url}")

                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode() if data else None,
                    headers=self.DEFAULT_HEADERS,
                    method=method,
                )

                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    result = json.loads(raw.decode("utf-8"))

                self.rate_limiter.record()

                # Check aitrados-api code field
                if result.get("code") != 200:
                    msg = result.get("message", str(result))
                    raise AitradosError(f"API error {result.get('code')}: {msg}")

                return result.get("result", result)

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = self.rate_limiter.handle_429(e.read().decode())
                    _logger.warning(f"HTTP 429 — backing off {wait:.1f}s (attempt {attempt})")
                    time.sleep(wait)
                    continue
                body = e.read().decode("utf-8", errors="replace")
                if self.debug:
                    _logger.debug(f"HTTP error body: {body}")
                # 4xx auth errors — don't retry
                if 400 <= e.code < 500 and e.code != 429:
                    raise AitradosAuthError(f"HTTP {e.code}: {body}") from e
                # 5xx — retry
                _logger.warning(f"HTTP {e.code}, retrying (attempt {attempt})")
                time.sleep(min(2 ** attempt, 30))

            except (urllib.error.URLError, TimeoutError) as e:
                _logger.warning(f"Connection error: {e}, retrying (attempt {attempt})")
                time.sleep(min(2 ** attempt, 30))

            except AitradosAuthError:
                raise
            except AitradosError:
                raise

        raise AitradosConnectionError(
            f"Request failed after {self.max_retries} retries: {path}"
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", path, params=params)

    def post(
        self, path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._request("POST", path, params=params, data=data)


# ---------------------------------------------------------------------------
# Symbol / Interval normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_symbol(exchange: str, symbol: str) -> str:
    """
    Convert exchange-native symbol to aitrados-api 'country_symbol'.
    e.g. 'BTC/USDT' on 'binance' -> 'CRYPTO:BINANCE:BTCUSDT'
    """
    # Strip common separators, upper-case
    clean = symbol.upper().replace("/", "").replace("-", "").replace("_", "")

    exchange_upper = exchange.upper()
    if exchange == "binance":
        return f"CRYPTO:{exchange_upper}:{clean}"
    elif exchange == "okx":
        return f"CRYPTO:{exchange_upper}:{clean}"
    elif exchange == "bybit":
        return f"CRYPTO:{exchange_upper}:{clean}"
    elif exchange == "deribit":
        # Deribit uses BTC-xxxx / ETH-xxxx format
        return f"CRYPTO:{exchange_upper}:{clean}"
    elif exchange == "kraken":
        return f"CRYPTO:{exchange_upper}:{clean}"
    elif exchange == "gateio":
        return f"CRYPTO:{exchange_upper}:{clean}"
    elif exchange == "huobi":
        return f"CRYPTO:{exchange_upper}:{clean}"
    elif exchange == "kucoin":
        return f"CRYPTO:{exchange_upper}:{clean}"
    elif exchange == "bitget":
        return f"CRYPTO:{exchange_upper}:{clean}"
    elif exchange == "mexc":
        return f"CRYPTO:{exchange_upper}:{clean}"
    else:
        return f"CRYPTO:{exchange_upper}:{clean}"


def _to_aitrados_interval(interval: str) -> str:
    """Convert standard interval (1m, 5m, 1h, 1d) to aitrados interval name."""
    return INTERVAL_MAP.get(interval.lower(), interval.upper())

def _from_aitrados_interval(aitrados_interval: str) -> str:
    """Reverse: aitrados interval name -> standard interval."""
    return _REVERSE_INTERVAL_MAP.get(aitrados_interval, aitrados_interval.lower())


# ---------------------------------------------------------------------------
# ExchangeDataFetcher — fetches & normalises data from each exchange
# ---------------------------------------------------------------------------

class ExchangeDataFetcher:
    """
    High-level data fetcher for a single exchange.
    Translates exchange-native API responses into normalised dict / list format.

    实例化时传入 AitradosClient，在其内部调用 REST / WebSocket 接口获取数据。
    """

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_ohlcv(
        exchange: str,
        raw: Union[List, Dict],
        interval: str,
    ) -> List[Dict[str, Any]]:
        """
        Normalise exchange OHLCV data to a common dict format.

        参数
        ----
        exchange : str   — 交易所名称 (binance, okx, bybit, ...)
        raw      : list/dict — 交易所原生响应
        interval : str   — K线周期 (1m, 5m, 1h, 1d, ...)

        返回
        ----
        List[Dict] ，每个字典包含:
            timestamp (int, ms), open (float), high (float),
            low (float), close (float), volume (float)
        """
        results: List[Dict[str, Any]] = []

        # aitrados-api format: list of [timestamp, open, high, low, close, volume]
        if isinstance(raw, dict):
            raw = raw.get("data", raw.get("result", []))
        if not isinstance(raw, list):
            return results

        for item in raw:
            if not item or len(item) < 6:
                continue
            # item may be list or dict with numeric keys
            if isinstance(item, list):
                ts, o, h, l, c, v = item[0], item[1], item[2], item[3], item[4], item[5]
            else:
                try:
                    ts = int(item.get("t", item.get("ts", item.get("timestamp", 0))))
                    o  = float(item.get("o", item.get("open", 0)))
                    h  = float(item.get("h", item.get("high", 0)))
                    l  = float(item.get("l", item.get("low", 0)))
                    c  = float(item.get("c", item.get("close", 0)))
                    v  = float(item.get("v", item.get("volume", 0)))
                except (TypeError, ValueError):
                    continue

            results.append({
                "exchange": exchange,
                "symbol": item[6] if isinstance(item, list) and len(item) > 6 else "",
                "timestamp": ts,
                "interval": interval,
                "open": o, "high": h, "low": l, "close": c, "volume": v,
            })

        return results

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_orderbook(
        exchange: str,
        raw: Union[Dict, List],
        symbol: str,
        depth: int = 20,
    ) -> Dict[str, Any]:
        """
        Normalise order-book / market-depth response.

        返回
        ----
        Dict with keys: exchange, symbol, timestamp (ms),
                        bids [(price, qty), ...], asks [(price, qty), ...]
        """
        ts = int(time.time() * 1000)
        bids, asks = [], []

        if isinstance(raw, dict):
            # Common nested structures
            bids = raw.get("bids", raw.get("data", {}).get("bids", []))
            asks = raw.get("asks", raw.get("data", {}).get("asks", []))
            ts = raw.get("ts", raw.get("data", {}).get("ts", ts))
        elif isinstance(raw, list):
            # Some exchanges return [[price, qty], ...]
            half = len(raw) // 2
            bids = raw[:half]
            asks = raw[half:]

        # Ensure (price, qty) tuple pairs
        def _parse_entries(items):
            out = []
            for entry in items[:depth]:
                if isinstance(entry, list) and len(entry) >= 2:
                    out.append((float(entry[0]), float(entry[1])))
                elif isinstance(entry, dict):
                    out.append((float(entry.get("price", 0)), float(entry.get("qty", entry.get("size", 0)))))
            return out

        return {
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": ts,
            "bids": _parse_entries(bids),
            "asks": _parse_entries(asks),
        }

    # ------------------------------------------------------------------
    # Ticker
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_ticker(
        exchange: str,
        raw: Union[Dict, List],
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Normalise ticker / 24hr stats response.

        返回
        ----
        Dict with: exchange, symbol, timestamp (ms), last_price,
                   high_24h, low_24h, volume_24h, quote_volume_24h,
                   price_change, price_change_pct, ...
        """
        ts = int(time.time() * 1000)
        d: Dict[str, Any] = {}

        if isinstance(raw, dict):
            d = raw.get("data", raw)
            ts = d.get("ts", d.get("timestamp", ts))
        elif isinstance(raw, list) and raw:
            d = raw[0] if isinstance(raw[0], dict) else {"last_price": raw[0]}

        return {
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": ts,
            "last_price":  float(d.get("last_price",   d.get("c",        d.get("close",        0)))),
            "high_24h":    float(d.get("high_24h",       d.get("h",        d.get("high",         0)))),
            "low_24h":     float(d.get("low_24h",        d.get("l",        d.get("low",          0)))),
            "volume_24h":  float(d.get("volume_24h",     d.get("v",        d.get("base_volume",  0)))),
            "quote_volume_24h": float(d.get("quote_volume_24h", d.get("q", d.get("quote_volume", 0)))),
            "price_change":      float(d.get("price_change",    d.get("p", 0))),
            "price_change_pct":  float(d.get("price_change_pct", d.get("P", 0))),
        }

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_funding_rate(
        exchange: str,
        raw: Union[Dict, List],
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Normalise funding-rate response (perpetual futures).

        返回
        ----
        Dict with: exchange, symbol, timestamp (ms),
                   funding_rate, next_funding_time (ms), ...
        """
        ts = int(time.time() * 1000)
        d: Dict[str, Any] = {}

        if isinstance(raw, dict):
            d = raw.get("data", raw)
            ts = d.get("ts", d.get("timestamp", ts))
        elif isinstance(raw, list) and raw:
            d = raw[0] if isinstance(raw[0], dict) else {}

        return {
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": ts,
            "funding_rate": float(d.get("funding_rate", d.get("rate", 0))),
            "next_funding_time": int(d.get("next_funding_time", d.get("nextFundingTime", 0))),
        }


# ---------------------------------------------------------------------------
# UnifiedDataFrame — pure-Python DataFrame without pandas dependency
# ---------------------------------------------------------------------------

class UnifiedDataFrame:
    """
    Lightweight DataFrame-like container (pure-Python, no pandas required).

    Organises normalised market-data rows into typed columns.
    Supports conversion to list-of-dicts and to pandas DataFrame (optional).

    轻量级 DataFrame 容器（纯 Python，无需 pandas）。
    支持转换为 list[dict]，以及按需转换为 pandas DataFrame。
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        data: Optional[List[List[Any]]] = None,
        index: Optional[List[Any]] = None,
    ):
        """
        参数
        ----
        columns : list of str — 列名列表
        data    : list of list — 行数据 [[val, val, ...], ...]
        index   : list of Any — 行索引（可选）
        """
        self._columns: List[str] = columns or []
        self._data: List[List[Any]] = data or []
        self._index: List[Any] = index or list(range(len(self._data)))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_list(cls, rows: List[Dict[str, Any]]) -> "UnifiedDataFrame":
        """
        Build from a list of normalised dicts (e.g. from ExchangeDataFetcher).

        从 dict 列表构建（适用于 fetch_ohlcv / fetch_ticker 等返回值）。
        """
        if not rows:
            return cls()
        columns = list(rows[0].keys())
        data = [[row.get(c) for c in columns] for row in rows]
        return cls(columns=columns, data=data)

    @classmethod
    def from_ohlcv(cls, rows: List[Dict[str, Any]]) -> "UnifiedDataFrame":
        """Build from normalised OHLCV list."""
        return cls.from_list(rows)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def dtypes(self) -> Dict[str, str]:
        """Infer column types from first non-null value."""
        types = {}
        for col in self._columns:
            vals = [row[i] for i, row in enumerate(self._data) if i < len(self._data) and row[i] is not None]
            if not vals:
                types[col] = "object"
            elif all(isinstance(v, bool) for v in vals):
                types[col] = "bool"
            elif all(isinstance(v, int) for v in vals):
                types[col] = "int64"
            elif all(isinstance(v, float) for v in vals):
                types[col] = "float64"
            elif all(isinstance(v, str) for v in vals):
                types[col] = "str"
            else:
                types[col] = "object"
        return types

    def shape(self) -> Tuple[int, int]:
        return len(self._data), len(self._columns)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Union[int, str, slice]) -> Any:
        if isinstance(key, int):
            return dict(zip(self._columns, self._data[key]))
        if isinstance(key, str):
            idx = self._columns.index(key)
            return [row[idx] for row in self._data]
        if isinstance(key, slice):
            new_data = self._data[key]
            return UnifiedDataFrame(columns=self._columns, data=new_data)
        raise TypeError(f"Invalid key type: {type(key)}")

    def head(self, n: int = 5) -> List[Dict[str, Any]]:
        return [dict(zip(self._columns, row)) for row in self._data[:n]]

    def tail(self, n: int = 5) -> List[Dict[str, Any]]:
        return [dict(zip(self._columns, row)) for row in self._data[-n:]]

    def to_dict(self, orient: str = "list") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Convert to Python dict / list-of-dicts.

        参数
        ----
        orient : "list" (default) — {col: [val, ...]}
                 "records"         — [{col: val, ...}, ...]
        """
        if orient == "records":
            return [dict(zip(self._columns, row)) for row in self._data]
        # orient == "list"
        return {col: [row[i] for row in self._data] for i, col in enumerate(self._columns)}

    def to_pandas(self):
        """
        Convert to pandas DataFrame.
        Requires pandas to be installed.
        """
        try:
            import pandas as pd  # type: ignore
            return pd.DataFrame(self._data, columns=self._columns, index=self._index)
        except ImportError:
            raise ImportError("pandas is not installed. Run: pip install pandas")

    def __repr__(self) -> str:
        rows = [dict(zip(self._columns, r)) for r in self._data[:5]]
        meta = f"UnifiedDataFrame({self.shape()[0]} rows x {self.shape()[1]} cols)"
        return meta + "\n" + json.dumps(rows[:3], default=str, indent=2)


# ---------------------------------------------------------------------------
# AitradosClient — unified REST client
# ---------------------------------------------------------------------------

class AitradosClient:
    """
    Unified REST client for the aitrados-api.

    Wraps a thread-safe urllib HTTP layer and provides high-level
    methods that map any supported exchange to the correct aitrados
    REST endpoint (OHLCV, order book, ticker, funding rate).

    Aitrados 统一 REST 客户端。
    支持 Binance, OKX, Bybit, Deribit, Kraken, Gate.io, Huobi, KuCoin, Bitget, MEXC
    共 10 家交易所的行情数据获取。

    参数
    ----
    secret_key : str  — aitrados-api secret key
    timeout    : int — HTTP timeout in seconds (default 30)
    debug      : bool — enable debug logging (default False)

    用法示例
    --------
    >>> client = AitradosClient(secret_key="your-key")
    >>> data = client.fetch_ohlcv("binance", "BTC/USDT", "1h",
    ...                            from_date="2024-01-01", to_date="2024-01-02")
    >>> print(data[:3])
    [{'timestamp': 1704067200000, 'open': ...}, ...]
    """

    def __init__(
        self,
        secret_key: str,
        timeout: int = 30,
        debug: bool = False,
    ):
        if not secret_key:
            raise AitradosValidationError("secret_key is required")
        self._http = _HttpClient(
            secret_key=secret_key,
            base_url=_AITRADOS_REST_BASE,
            timeout=timeout,
            max_retries=5,
            debug=debug,
        )
        self._fetcher = ExchangeDataFetcher()

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 150,
        is_eth: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV (candlestick) data.

        参数
        ----
        exchange  : str — 交易所名 (binance, okx, bybit, kraken, ...)
        symbol    : str — 交易对符号，支持 'BTC/USDT', 'BTC-USDT', 'BTCUSDT' 等格式
        interval  : str — K线周期，如 '1m', '5m', '1h', '1d'
        from_date : str — 起始日期，YYYY-MM-DD 格式（可选）
        to_date   : str — 结束日期，YYYY-MM-DD 格式（可选）
        limit     : int — 每页数据条数上限，默认 150
        is_eth    : bool — 是否包含美股盘前盘后数据（仅对美股有效，默认 False）

        返回
        ----
        List[Dict[str, Any]] — 标准化的 OHLCV 数据列表，每个元素包含:
            timestamp, open, high, low, close, volume, exchange, symbol, interval
        """
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            raise AitradosValidationError(
                f"Unsupported exchange: {exchange}. "
                f"Supported: {SUPPORTED_EXCHANGES}"
            )

        country_symbol = _normalise_symbol(exchange, symbol)
        aitrados_interval = _to_aitrados_interval(interval)

        # Build query params
        params: Dict[str, Any] = {
            "schema_asset": "crypto",
            "country_symbol": country_symbol,
            "interval": aitrados_interval,
            "format": "json",
            "limit": limit,
            "is_eth": 1 if is_eth else 0,
        }
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        path = (
            f"crypto/bars/{country_symbol}/{aitrados_interval}/from/{from_date or ''}"
            f"/to/{to_date or ''}"
        )
        # Simpler approach: use the latest endpoint when no date range
        if not from_date and not to_date:
            path = f"crypto/bars/{country_symbol}/{aitrados_interval}/latest"

        try:
            raw = self._http.get(path, params=params)
        except Exception as e:
            _logger.error(f"fetch_ohlcv failed for {exchange}:{symbol}: {e}")
            raise

        rows = self._fetcher.normalise_ohlcv(exchange, raw, interval)
        return rows

    def fetch_ohlcv_latest(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        limit: int = 150,
    ) -> List[Dict[str, Any]]:
        """
        Fetch the latest OHLCV bars (no date range needed).

        最新 K线数据获取，无需指定日期范围。
        """
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            raise AitradosValidationError(f"Unsupported exchange: {exchange}")
        country_symbol = _normalise_symbol(exchange, symbol)
        aitrados_interval = _to_aitrados_interval(interval)
        path = f"crypto/bars/{country_symbol}/{aitrados_interval}/latest"
        params = {
            "schema_asset": "crypto",
            "limit": limit,
        }
        raw = self._http.get(path, params=params)
        return self._fetcher.normalise_ohlcv(exchange, raw, interval)

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    def fetch_orderbook(
        self,
        exchange: str,
        symbol: str,
        depth: int = 20,
    ) -> Dict[str, Any]:
        """
        Fetch order book / market depth.

        参数
        ----
        exchange : str — 交易所名
        symbol   : str — 交易对符号
        depth    : int — 档位数，默认 20

        返回
        ----
        Dict — 标准化的订单簿，包含 bids, asks, timestamp, exchange, symbol
        """
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            raise AitradosValidationError(f"Unsupported exchange: {exchange}")
        country_symbol = _normalise_symbol(exchange, symbol)
        path = f"crypto/depth/{country_symbol}"
        params = {"limit": depth}
        raw = self._http.get(path, params=params)
        return self._fetcher.normalise_orderbook(exchange, raw, symbol, depth)

    # ------------------------------------------------------------------
    # Tickers
    # ------------------------------------------------------------------

    def fetch_tickers(
        self,
        exchange: str,
        symbol: Optional[str] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fetch 24hr ticker / statistics.

        参数
        ----
        exchange : str        — 交易所名
        symbol   : str | None — 交易对（None 表示全市场）

        返回
        ----
        单交易对时返回 Dict；全市场时返回 List[Dict]
        """
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            raise AitradosValidationError(f"Unsupported exchange: {exchange}")
        country_symbol = _normalise_symbol(exchange, symbol or "")
        path = f"crypto/ticker/{country_symbol}"
        params: Dict[str, Any] = {"format": "json"}
        raw = self._http.get(path, params=params)
        if symbol:
            return self._fetcher.normalise_ticker(exchange, raw, symbol)
        # Multi-ticker: normalise each
        if isinstance(raw, list):
            return [
                self._fetcher.normalise_ticker(exchange, item, item.get("symbol", ""))
                for item in raw
            ]
        return [self._fetcher.normalise_ticker(exchange, raw, symbol or "")]

    # ------------------------------------------------------------------
    # Funding rate (perpetuals)
    # ------------------------------------------------------------------

    def fetch_funding_rate(
        self,
        exchange: str,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Fetch current funding rate for a perpetual futures contract.

        参数
        ----
        exchange : str — 交易所名
        symbol   : str — 合约交易对，如 'BTC/USDT'

        返回
        ----
        Dict — 标准化资金费率数据
        """
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            raise AitradosValidationError(f"Unsupported exchange: {exchange}")
        country_symbol = _normalise_symbol(exchange, symbol)
        path = f"crypto/funding/{country_symbol}"
        raw = self._http.get(path)
        return self._fetcher.normalise_funding_rate(exchange, raw, symbol)

    # ------------------------------------------------------------------
    # UnifiedDataFrame helpers
    # ------------------------------------------------------------------

    def fetch_ohlcv_dataframe(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 150,
    ) -> UnifiedDataFrame:
        """Same as fetch_ohlcv but returns a UnifiedDataFrame."""
        rows = self.fetch_ohlcv(
            exchange=exchange, symbol=symbol, interval=interval,
            from_date=from_date, to_date=to_date, limit=limit,
        )
        return UnifiedDataFrame.from_ohlcv(rows)

    def __enter__(self) -> "AitradosClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client (no-op for urllib, here for API compatibility)."""
        _logger.debug("AitradosClient.close() called")


# ---------------------------------------------------------------------------
# WebSocketDataStream — async WebSocket consumer
# ---------------------------------------------------------------------------

class WebSocketDataStream:
    """
    Async WebSocket consumer for real-time market data.

    Connects to the aitrados-api WebSocket feed and subscribes to
    live ticker, trade, or OHLC updates for any supported exchange.

    异步 WebSocket 数据流消费者，连接 aitrados-api 实时行情。

    参数
    ----
    secret_key   : str            — aitrados-api secret key
    on_ticker    : Callable|None  — 收到 ticker 更新的回调函数
    on_ohlc      : Callable|None  — 收到 OHLC 更新的回调函数
    on_trade     : Callable|None  — 收到成交更新的回调函数
    on_error     : Callable|None  — 错误回调
    debug        : bool           — 启用调试日志

    用法示例
    --------
    >>> async def handle_ticker(client, msg):
    ...     print(msg)
    >>> stream = WebSocketDataStream("key", on_ticker=handle_ticker)
    >>> await stream.connect()
    >>> stream.subscribe_ticker("binance", "BTC/USDT")
    >>> await asyncio.sleep(60)
    >>> await stream.disconnect()
    """

    def __init__(
        self,
        secret_key: str,
        on_ticker: Optional[Callable[..., Any]] = None,
        on_ohlc: Optional[Callable[..., Any]] = None,
        on_trade: Optional[Callable[..., Any]] = None,
        on_error: Optional[Callable[..., Any]] = None,
        debug: bool = False,
    ):
        if not secret_key:
            raise AitradosValidationError("secret_key is required for WebSocket")
        self.secret_key = secret_key
        self.debug = debug

        # Callbacks
        self._on_ticker = on_ticker
        self._on_ohlc   = on_ohlc
        self._on_trade  = on_trade
        self._on_error  = on_error

        # Internal state
        self._ws = None          # websocket.WebSocket object (sync, used in async context)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._should_exit = threading.Event()
        self._subscribed_topics: Dict[str, List[str]] = {}
        self._connected = False
        self._lock = threading.Lock()

        # Dynamic import to avoid hard dependency on websocket-client
        try:
            import websocket  # type: ignore
            self._ws_module = websocket
        except ImportError:
            raise ImportError(
                "websocket-client is required for WebSocketDataStream. "
                "Install: pip install websocket-client"
            )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Connect to the aitrados-api WebSocket server.
        连接到 WebSocket 服务器并完成认证。
        """
        if self._connected:
            _logger.warning("WebSocket already connected")
            return

        self._should_exit.clear()
        self._loop = asyncio.get_running_loop()

        # Build ws URL with auth
        uri = f"{_AITRADOS_WS_BASE}?secret_key={self.secret_key}"

        if self.debug:
            _logger.debug(f"Connecting to WebSocket: {uri[:60]}...")

        try:
            self._ws = self._ws_module.WebSocket(
                timeout=30,
                enable_multithread=True,
            )
            self._ws.connect(uri)
            self._connected = True

            # Authenticate
            auth_payload = json.dumps({
                "message_type": "authenticate",
                "params": {"secret_key": self.secret_key},
            })
            self._ws.send(auth_payload)
            if self.debug:
                _logger.debug("--> Sent auth request")

            # Read auth response
            resp_raw = self._ws.recv()
            resp = json.loads(resp_raw)
            if self.debug:
                _logger.debug(f"<-- Auth response: {resp_raw[:200]}")

            authenticated = resp.get("result", {}).get("authenticated", False)
            if not authenticated:
                raise AitradosAuthError("WebSocket authentication failed")

            if self.debug:
                _logger.info("WebSocket authenticated successfully")

            # Start recv loop
            self._recv_task = self._loop.create_task(self._recv_loop())

        except Exception as e:
            self._connected = False
            _logger.error(f"WebSocket connect error: {e}")
            raise

    async def _recv_loop(self) -> None:
        """Background task that continuously receives and dispatches messages."""
        while not self._should_exit.is_set():
            try:
                # Use asyncio to wait on the sync socket with a timeout
                msg_raw = await asyncio.wait_for(
                    self._loop.run_in_executor(None, self._ws.recv),
                    timeout=5.0,
                )
                if msg_raw is None:
                    break
                await self._dispatch(json.loads(msg_raw))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self._should_exit.is_set():
                    break
                _logger.error(f"WebSocket recv error: {e}")
                if self._on_error:
                    await self._safe_callback(self._on_error, {"error": str(e)})
                break

    async def _dispatch(self, msg: Dict[str, Any]) -> None:
        """Route a received message to the appropriate callback."""
        msg_type = msg.get("message_type", "")
        results  = msg.get("result", [])

        if not isinstance(results, list):
            results = [results]

        if msg_type == "ticker":
            for item in results:
                await self._safe_callback(self._on_ticker, item)
        elif msg_type == "ohlc":
            for item in results:
                await self._safe_callback(self._on_ohlc, item)
        elif msg_type == "trade":
            for item in results:
                await self._safe_callback(self._on_trade, item)
        elif msg_type == "error":
            _logger.error(f"Server error: {msg.get('message')} — {msg.get('detail')}")
            await self._safe_callback(self._on_error, msg)
        elif self.debug:
            _logger.debug(f"<-- {msg_type}: {str(results)[:100]}")

    async def _safe_callback(self, cb: Optional[Callable[..., Any]], data: Any) -> None:
        """Execute a callback, handling both sync and async callables."""
        if not cb:
            return
        try:
            if asyncio.iscoroutinefunction(cb):
                await cb(self, data)
            else:
                cb(self, data)
        except Exception as e:
            _logger.error(f"Callback error: {e}")

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    def _subscribe(self, subscribe_type: str, *topics: str) -> None:
        """Send a subscribe WebSocket message synchronously (thread-safe)."""
        if not self._connected or self._ws is None:
            _logger.warning("Not connected — cannot subscribe")
            return
        payload = json.dumps({
            "message_type": "subscribe",
            "params": {
                "subscribe_type": subscribe_type,
                "topics": list(topics),
            },
        })
        try:
            self._ws.send(payload)
            if self.debug:
                _logger.debug(f"--> Subscribe {subscribe_type}: {topics}")
            with self._lock:
                key = subscribe_type
                if key not in self._subscribed_topics:
                    self._subscribed_topics[key] = []
                for t in topics:
                    if t not in self._subscribed_topics[key]:
                        self._subscribed_topics[key].append(t)
        except Exception as e:
            _logger.error(f"Subscribe error: {e}")

    def _unsubscribe(self, subscribe_type: str, *topics: str) -> None:
        """Send an unsubscribe WebSocket message."""
        if not self._connected or self._ws is None:
            return
        payload = json.dumps({
            "message_type": "unsubscribe",
            "params": {
                "subscribe_type": subscribe_type,
                "topics": list(topics),
            },
        })
        try:
            self._ws.send(payload)
            if self.debug:
                _logger.debug(f"--> Unsubscribe {subscribe_type}: {topics}")
        except Exception as e:
            _logger.error(f"Unsubscribe error: {e}")

    # ------------------------------------------------------------------
    # Public subscription API
    # ------------------------------------------------------------------

    def subscribe_ticker(self, exchange: str, symbol: str) -> None:
        """
        Subscribe to real-time ticker updates for a symbol.
        订阅实时 ticker 数据。

        参数
        ----
        exchange : str — 交易所名
        symbol   : str — 交易对，如 'BTC/USDT'
        """
        topic = _normalise_symbol(exchange, symbol)
        self._subscribe("ticker", topic)

    def unsubscribe_ticker(self, exchange: str, symbol: str) -> None:
        """取消订阅 ticker（Unsubscribe from ticker updates）。"""
        topic = _normalise_symbol(exchange, symbol)
        self._unsubscribe("ticker", topic)

    def subscribe_ohlc(self, exchange: str, symbol: str, interval: str = "1m") -> None:
        """
        Subscribe to real-time OHLC (candlestick) updates.
        订阅实时 K 线数据。

        参数
        ----
        exchange : str — 交易所名
        symbol   : str — 交易对
        interval : str — K线周期，如 '1m', '5m', '1h'
        """
        topic = _normalise_symbol(exchange, symbol)
        aitrados_interval = _to_aitrados_interval(interval)
        self._subscribe("ohlc", f"{topic}:{aitrados_interval}")

    def unsubscribe_ohlc(self, exchange: str, symbol: str, interval: str = "1m") -> None:
        """取消订阅 OHLC（Unsubscribe from OHLC updates）。"""
        topic = _normalise_symbol(exchange, symbol)
        aitrados_interval = _to_aitrados_interval(interval)
        self._unsubscribe("ohlc", f"{topic}:{aitrados_interval}")

    def subscribe_trades(self, exchange: str, symbol: str) -> None:
        """
        Subscribe to real-time public trade (tick) data.
        订阅实时成交数据。
        """
        topic = _normalise_symbol(exchange, symbol)
        self._subscribe("trade", topic)

    def unsubscribe_trades(self, exchange: str, symbol: str) -> None:
        """取消订阅成交（Unsubscribe from trade updates）。"""
        topic = _normalise_symbol(exchange, symbol)
        self._unsubscribe("trade", topic)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        """
        Gracefully disconnect from the WebSocket server.
        优雅地断开 WebSocket 连接。
        """
        self._should_exit.set()
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._connected = False
        if self.debug:
            _logger.info("WebSocket disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_subscribed_topics(self) -> Dict[str, List[str]]:
        """返回当前已订阅的主题列表（Returns currently subscribed topics）。"""
        with self._lock:
            return dict(self._subscribed_topics)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "WebSocketDataStream":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()
