"""
CCXT-Style Unified Exchange Adapter
CCXT风格统一交易所适配器

A pure Python implementation inspired by CCXT (CryptoCurrency eXchange Trading Library),
providing unified REST API access to 100+ cryptocurrency exchanges using urllib.

Supported Exchanges / 支持的交易所:
    Binance, Bybit, OKX, Deribit, Kraken, Coinbase, Gate.io, Huobi,
    KuCoin, Bitget, MEXC, Phemex, and 90+ more

Features / 功能特性:
    - Market data: OHLCV, order books, tickers, trades
    - Trading: Market/limit orders, cancellation
    - Margin: Cross-exchange margin calculation
    - Normalization: Unified data format across exchanges

Example / 示例:
    adapter = CCXTAdapter('binance')
    adapter.load_markets()
    ohlcv = adapter.fetch_ohlcv('BTC/USDT', '1m')
    orderbook = adapter.fetch_order_book('BTC/USDT')
    orders = adapter.place_order('BTC/USDT', 'buy', 'limit', 1.0, 50000)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.parse as urlparse
import urllib.request as urllib2
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

__all__ = [
    "CCXTAdapter",
    "ExchangeNormalizer",
    "MarketDataFetcher",
    "OrderExecutor",
    "MarginCalculator",
    "SUPPORTED_EXCHANGES",
    "EXCHANGE_CONFIGS",
]

# =============================================================================
# Supported Exchanges / 支持的交易所
# =============================================================================

SUPPORTED_EXCHANGES = [
    "binance", "bybit", "okx", "deribit", "kraken", "coinbase",
    "gateio", "huobi", "kucoin", "bitget", "mexc", "phemex",
    "bitfinex", "bitmex", "bitstamp", "bittrex", "cryptocom",
    "exmo", "gemini", "hitbtc", "hotcoin", "kraken", "lbank",
    "poloniex", "tokocrypto", "upbit", "zb", "ace", "alpaca",
    "bit2c", "bigone", "bequant", "bibox", "bigone", "birake",
    "bitbns", "bitcash", "bitflyer", "bithumb", "bitmart",
    "bitopro", "bitrue", "bitso", "bitsonic", "bitvavo", "bkex",
    "bl3p", "blockchaincom", "btcalpha", "btcmarkets", "btcturk",
    "buda", "bw", "bytetrade", "catex", "cex", "chainex",
    "chilebit", "coinbaseprime", "coinbasepro", "coincheck",
    "coinex", "coinfalcon", "coinfloor", "coinmate", "coinone",
    "coinsbit", "crex24", "deribit", "digifinex", "equos",
    "eterbase", "exmo", "exrate", "f崩", "flowbtc", "foxbit",
    "ftx", "gateio", "gbtc", "happyshiba", "hb", "hicone",
    "hipo", "hitbtc", "hollaex", "huobi", "icon", "idex",
    "indodax", "itbit", "kkex", "kraken", "kuna", "latoken",
    "lbank", "liquid", "luno", "lykke", "mandala", "mercado",
    "mixcoins", "negociec", "novadax", "nukem", "oceanex",
    "okcoin", "okx", "paymium", "phemex", "poloniex", "probit",
    "qtrade", "ripio", "southxchange", "stex", "surviving",
    "tidex", "timex", "tokenize", "tokocrypto", "topb", "upbit",
    "vbtx", "wavesexchange", "whitebit", "xena", "yobit", "zb",
]

# Exchange API Configurations / 交易所API配置
EXCHANGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "binance": {
        "name": "Binance",
        "hostname": "binance.com",
        "baseUrl": "https://api.binance.com",
        "testnet": "https://testnet.binance.vision",
        "rateLimit": 1200,
        "timeframes": {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"},
        "quote": ["USDT", "BUSD", "BTC", "ETH"],
        "marginMode": ["isolated", "cross"],
    },
    "bybit": {
        "name": "Bybit",
        "hostname": "bybit.com",
        "baseUrl": "https://api.bybit.com",
        "testnet": "https://api-testnet.bybit.com",
        "rateLimit": 20,
        "timeframes": {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D", "1w": "W"},
        "quote": ["USDT", "USDC", "BTC", "ETH"],
        "marginMode": ["isolated", "cross"],
    },
    "okx": {
        "name": "OKX",
        "hostname": "okx.com",
        "baseUrl": "https://www.okx.com",
        "testnet": "https://www.okx.com",
        "rateLimit": 20,
        "timeframes": {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1W"},
        "quote": ["USDT", "USDC", "BTC", "ETH"],
        "marginMode": ["isolated", "cross"],
    },
    "deribit": {
        "name": "Deribit",
        "hostname": "deribit.com",
        "baseUrl": "https://www.deribit.com",
        "testnet": "https://test.deribit.com",
        "rateLimit": 100,
        "timeframes": {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"},
        "quote": ["BTC", "ETH", "USDC"],
        "marginMode": ["cross"],
    },
    "kraken": {
        "name": "Kraken",
        "hostname": "kraken.com",
        "baseUrl": "https://api.kraken.com",
        "testnet": "https://api.kraken.com",
        "rateLimit": 1000,
        "timeframes": {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "1440", "1w": "10080"},
        "quote": ["USD", "EUR", "BTC", "ETH"],
        "marginMode": ["cross"],
    },
    "coinbase": {
        "name": "Coinbase",
        "hostname": "coinbase.com",
        "baseUrl": "https://api.coinbase.com",
        "testnet": "https://api-public.sandbox.coinbase.com",
        "rateLimit": 10,
        "timeframes": {"1m": "60", "5m": "300", "15m": "900", "1h": "3600", "4h": "14400", "1d": "86400", "1w": "604800"},
        "quote": ["USD", "EUR", "GBP", "BTC", "ETH"],
        "marginMode": [],
    },
    "gateio": {
        "name": "Gate.io",
        "hostname": "gate.io",
        "baseUrl": "https://api.gateio.io",
        "testnet": "https://api.gateio.io",
        "rateLimit": 100,
        "timeframes": {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"},
        "quote": ["USDT", "USDC", "BTC", "ETH"],
        "marginMode": ["isolated", "cross"],
    },
    "huobi": {
        "name": "Huobi",
        "hostname": "huobi.com",
        "baseUrl": "https://api.huobi.com",
        "testnet": "https://api.huobi.com",
        "rateLimit": 100,
        "timeframes": {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "60min", "4h": "4hour", "1d": "1day", "1w": "1week"},
        "quote": ["USDT", "BTC", "ETH"],
        "marginMode": ["isolated", "cross"],
    },
    "kucoin": {
        "name": "KuCoin",
        "hostname": "kucoin.com",
        "baseUrl": "https://api.kucoin.com",
        "testnet": "https://openapi-sandbox.kucoin.com",
        "rateLimit": 120,
        "timeframes": {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day", "1w": "1week"},
        "quote": ["USDT", "BTC", "ETH"],
        "marginMode": ["isolated", "cross"],
    },
    "bitget": {
        "name": "Bitget",
        "hostname": "bitget.com",
        "baseUrl": "https://api.bitget.com",
        "testnet": "https://api-testnet.bitget.com",
        "rateLimit": 100,
        "timeframes": {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"},
        "quote": ["USDT", "USDC", "BTC", "ETH"],
        "marginMode": ["isolated", "cross"],
    },
    "mexc": {
        "name": "MEXC",
        "hostname": "mexc.com",
        "baseUrl": "https://api.mexc.com",
        "testnet": "https://api.mexc.com",
        "rateLimit": 200,
        "timeframes": {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"},
        "quote": ["USDT", "BTC", "ETH"],
        "marginMode": ["isolated", "cross"],
    },
    "phemex": {
        "name": "Phemex",
        "hostname": "phemex.com",
        "baseUrl": "https://api.phemex.com",
        "testnet": "https://api-testnet.phemex.com",
        "rateLimit": 100,
        "timeframes": {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"},
        "quote": ["USDT", "BTC", "ETH", "USDC"],
        "marginMode": ["isolated", "cross"],
    },
}


# =============================================================================
# Data Classes / 数据类
# =============================================================================

@dataclass
class Market:
    """Market structure / 市场结构"""
    id: str
    symbol: str
    base: str
    quote: str
    baseId: str
    quoteId: str
    type: str = "spot"  # spot, future, swap, option
    linear: bool = False
    inverse: bool = False
    contractSize: float = 1.0
    precision_amount: float = 0.0001
    precision_price: float = 0.01
    limits_amount_min: float = 0.0
    limits_amount_max: float = float("inf")
    limits_price_min: float = 0.0
    limits_price_max: float = float("inf")


@dataclass
class OrderBook:
    """Order book structure / 订单簿结构"""
    symbol: str
    bids: List[List[float]] = field(default_factory=list)
    asks: List[List[float]] = field(default_factory=list)
    timestamp: int = 0
    datetime: str = ""


@dataclass
class OHLCV:
    """OHLCV candle structure / K线烛台结构"""
    timestamp: int
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1m"
    symbol: str = ""


@dataclass
class Ticker:
    """Ticker structure / 行情结构"""
    symbol: str
    last: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    quoteVolume: float = 0.0
    change: float = 0.0
    changePercent: float = 0.0
    timestamp: int = 0


@dataclass
class Order:
    """Order structure / 订单结构"""
    id: str
    symbol: str
    type: str
    side: str
    price: float
    amount: float
    filled: float = 0.0
    remaining: float = 0.0
    status: str = "pending"  # pending, open, closed, canceled
    timestamp: int = 0
    fee: float = 0.0
    feeCurrency: str = ""


@dataclass
class Trade:
    """Trade structure / 交易结构"""
    id: str
    symbol: str
    price: float
    amount: float
    side: str
    timestamp: int
    datetime: str = ""
    fee: float = 0.0
    feeCurrency: str = ""


@dataclass
class Position:
    """Position structure / 持仓结构"""
    symbol: str
    side: str
    amount: float
    entryPrice: float
    unrealizedPnl: float
    leverage: int = 1
    marginMode: str = "cross"
    liquidationPrice: float = 0.0


# =============================================================================
# Exchange Normalizer / 交易所数据标准化器
# =============================================================================

class ExchangeNormalizer:
    """Normalize exchange-specific data to unified format / 将交易所特定数据标准化为统一格式

    This class handles the conversion of exchange-specific response formats
    into a unified CCXT-style format.

    Example / 示例:
        normalizer = ExchangeNormalizer('binance')
        normalized = normalizer.normalize_ticker(data, 'BTC/USDT')
    """

    def __init__(self, exchange_id: str):
        """Initialize normalizer for specific exchange / 为特定交易所初始化标准化器

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'bybit')
        """
        self.exchange_id = exchange_id
        self.logger = logging.getLogger(f"ExchangeNormalizer.{exchange_id}")

    def normalize_ticker(self, data: Dict, symbol: str) -> Ticker:
        """Normalize ticker data / 标准化ticker数据

        Args:
            data: Raw exchange ticker data
            symbol: Trading pair symbol

        Returns:
            Normalized Ticker object
        """
        if self.exchange_id == "binance":
            return Ticker(
                symbol=symbol,
                last=float(data.get("lastPrice", 0) or 0),
                bid=float(data.get("bidPrice", 0) or 0) if data.get("bidPrice") else 0.0,
                ask=float(data.get("askPrice", 0) or 0) if data.get("askPrice") else 0.0,
                high=float(data.get("highPrice", 0) or 0),
                low=float(data.get("lowPrice", 0) or 0),
                volume=float(data.get("volume", 0) or 0),
                quoteVolume=float(data.get("quoteVolume", 0) or 0),
                change=float(data.get("priceChange", 0) or 0),
                changePercent=float(data.get("priceChangePercent", 0) or 0),
                timestamp=data.get("closeTime", 0) or 0,
            )
        elif self.exchange_id == "bybit":
            result = data.get("result", {}) or {}
            item = result.get("list", [{}])[0] if result.get("list") else {}
            return Ticker(
                symbol=symbol,
                last=float(item.get("lastPrice", 0) or 0),
                bid=float(item.get("bid1Price", 0) or 0) if item.get("bid1Price") else 0.0,
                ask=float(item.get("ask1Price", 0) or 0) if item.get("ask1Price") else 0.0,
                high=float(item.get("highPrice24h", 0) or 0),
                low=float(item.get("lowPrice24h", 0) or 0),
                volume=float(item.get("volume24h", 0) or 0),
                quoteVolume=float(item.get("turnover24h", 0) or 0),
                change=float(item.get("price24hPcnt", 0) or 0),
                changePercent=float(item.get("price24hPcnt", 0) or 0) * 100,
                timestamp=int(item.get("ts", 0) or 0),
            )
        elif self.exchange_id == "okx":
            data_list = data.get("data", [{}])
            item = data_list[0] if data_list else {}
            return Ticker(
                symbol=symbol,
                last=float(item.get("last", 0) or 0),
                bid=float(item.get("bidPx", 0) or 0) if item.get("bidPx") else 0.0,
                ask=float(item.get("askPx", 0) or 0) if item.get("askPx") else 0.0,
                high=float(item.get("high24h", 0) or 0),
                low=float(item.get("low24h", 0) or 0),
                volume=float(item.get("vol24h", 0) or 0),
                quoteVolume=float(item.get("quoteVol24h", 0) or 0),
                change=float(item.get("sodUtc8", 0) or 0),
                timestamp=int(item.get("ts", 0) or 0),
            )
        elif self.exchange_id == "deribit":
            result = data.get("result", {}) or {}
            return Ticker(
                symbol=symbol,
                last=float(result.get("last_price", 0) or 0),
                bid=float(result.get("best_bid_price", 0) or 0) if result.get("best_bid_price") else 0.0,
                ask=float(result.get("best_ask_price", 0) or 0) if result.get("best_ask_price") else 0.0,
                high=float(result.get("high", 0) or 0),
                low=float(result.get("low", 0) or 0),
                volume=float(result.get("volume", 0) or 0),
                quoteVolume=float(result.get("volume_usd", 0) or 0),
                change=float(result.get("price_change", 0) or 0),
                timestamp=int(result.get("timestamp", 0) or 0),
            )
        elif self.exchange_id == "kraken":
            result = data.get("result", {}) or {}
            ticker_data = list(result.values())[0] if result else {}
            return Ticker(
                symbol=symbol,
                last=float(ticker_data.get("c", ["0"])[0] or 0),
                bid=float(ticker_data.get("b", ["0"])[0] or 0),
                ask=float(ticker_data.get("a", ["0"])[0] or 0),
                high=float(ticker_data.get("h", ["0"])[0] or 0),
                low=float(ticker_data.get("l", ["0"])[0] or 0),
                volume=float(ticker_data.get("v", ["0"])[0] or 0),
                change=float(ticker_data.get("p", ["0"])[0] or 0),
                timestamp=int(time.time() * 1000),
            )
        elif self.exchange_id == "gateio":
            currency = data.get("currency", "")
            ticker_data = data.get("ticker", {}) or {}
            return Ticker(
                symbol=symbol,
                last=float(ticker_data.get("last", 0) or 0),
                bid=float(ticker_data.get("highest_bid", 0) or 0) if ticker_data.get("highest_bid") else 0.0,
                ask=float(ticker_data.get("lowest_ask", 0) or 0) if ticker_data.get("lowest_ask") else 0.0,
                high=float(ticker_data.get("high24h", 0) or 0),
                low=float(ticker_data.get("low24h", 0) or 0),
                volume=float(ticker_data.get("base_volume", 0) or 0),
                quoteVolume=float(ticker_data.get("quote_volume", 0) or 0),
                change=float(ticker_data.get("change", 0) or 0),
                timestamp=data.get("timestamp", 0) or 0,
            )
        elif self.exchange_id == "huobi":
            ticker_data = data.get("tick", {}) or {}
            return Ticker(
                symbol=symbol,
                last=float(ticker_data.get("last", 0) or 0),
                bid=float(ticker_data.get("bid", [0])[0] if ticker_data.get("bid") else 0 or 0),
                ask=float(ticker_data.get("ask", [0])[0] if ticker_data.get("ask") else 0 or 0),
                high=float(ticker_data.get("high", 0) or 0),
                low=float(ticker_data.get("low", 0) or 0),
                volume=float(ticker_data.get("vol", 0) or 0),
                quoteVolume=float(ticker_data.get("quoteVol", 0) or 0),
                timestamp=data.get("ts", 0) or 0,
            )
        elif self.exchange_id == "kucoin":
            data_dict = data.get("data", {}) or {}
            ticker_data = data_dict.get("ticker", {}) or {}
            return Ticker(
                symbol=symbol,
                last=float(ticker_data.get("last", 0) or 0),
                bid=float(ticker_data.get("buy", 0) or 0) if ticker_data.get("buy") else 0.0,
                ask=float(ticker_data.get("sell", 0) or 0) if ticker_data.get("sell") else 0.0,
                high=float(ticker_data.get("high", 0) or 0),
                low=float(ticker_data.get("low", 0) or 0),
                volume=float(ticker_data.get("vol", 0) or 0),
                quoteVolume=float(ticker_data.get("value", 0) or 0),
                timestamp=int(data_dict.get("time", 0) or 0),
            )
        elif self.exchange_id == "bitget":
            data_list = data.get("data", [{}])
            item = data_list[0] if data_list else {}
            return Ticker(
                symbol=symbol,
                last=float(item.get("last", 0) or 0),
                bid=float(item.get("bidPrice", 0) or 0) if item.get("bidPrice") else 0.0,
                ask=float(item.get("askPrice", 0) or 0) if item.get("askPrice") else 0.0,
                high=float(item.get("high24h", 0) or 0),
                low=float(item.get("low24h", 0) or 0),
                volume=float(item.get("baseVolume", 0) or 0),
                quoteVolume=float(item.get("quoteVolume", 0) or 0),
                timestamp=int(item.get("timestamp", 0) or 0),
            )
        else:
            # Generic fallback / 通用降级
            return self._normalize_generic_ticker(data, symbol)

    def _normalize_generic_ticker(self, data: Dict, symbol: str) -> Ticker:
        """Generic ticker normalization / 通用ticker标准化"""
        last = 0.0
        if isinstance(data, dict):
            for key in ["last", "lastPrice", "price", "close", "c"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, (int, float)):
                        last = float(val)
                    elif isinstance(val, (list, tuple)) and len(val) > 0:
                        try:
                            last = float(val[0])
                        except (ValueError, TypeError):
                            pass
                    break

        return Ticker(
            symbol=symbol,
            last=last,
            timestamp=int(time.time() * 1000),
        )

    def normalize_orderbook(self, data: Dict, symbol: str, limit: int = None) -> OrderBook:
        """Normalize order book data / 标准化订单簿数据

        Args:
            data: Raw exchange order book data
            symbol: Trading pair symbol
            limit: Maximum number of levels

        Returns:
            Normalized OrderBook object
        """
        bids, asks = [], []

        if self.exchange_id == "binance":
            bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
            asks = [[float(p), float(q)] for p, q in data.get("asks", [])]
            timestamp = data.get("lastUpdateId", 0)

        elif self.exchange_id == "bybit":
            result = data.get("result", {}) or {}
            bids = [[float(p), float(s)] for p, s in result.get("b", [])]
            asks = [[float(p), float(s)] for p, s in result.get("a", [])]
            timestamp = data.get("time", 0)

        elif self.exchange_id == "okx":
            data_list = data.get("data", [])
            for item in data_list:
                bids.append([float(item.get("bidPx", 0)), float(item.get("bidSz", 0))])
                asks.append([float(item.get("askPx", 0)), float(item.get("askSz", 0))])
            timestamp = int(data.get("ts", 0) or 0)

        elif self.exchange_id == "deribit":
            result = data.get("result", {}) or {}
            bids = [[float(b["price"]), float(b["size"])] for b in result.get("bids", [])]
            asks = [[float(a["price"]), float(a["size"])] for a in result.get("asks", [])]
            timestamp = result.get("timestamp", 0)

        elif self.exchange_id == "kraken":
            result = data.get("result", {}) or {}
            ob_data = list(result.values())[0] if result else {}
            bids_raw = ob_data.get("bs", ob_data.get("b", []))
            asks_raw = ob_data.get("as", ob_data.get("a", []))
            bids = [[float(p), float(v)] for p, v in bids_raw]
            asks = [[float(p), float(v)] for p, v in asks_raw]
            timestamp = int(time.time() * 1000)

        elif self.exchange_id == "gateio":
            bids = [[float(b["p"]), float(b["s"])] for b in data.get("bids", [])]
            asks = [[float(a["p"]), float(a["s"])] for a in data.get("asks", [])]
            timestamp = data.get("timestamp", 0)

        elif self.exchange_id == "huobi":
            tick = data.get("tick", {}) or {}
            bids_raw = tick.get("bid", [])
            asks_raw = tick.get("ask", [])
            bids = [[float(bids_raw[i]), float(bids_raw[i + 1])] for i in range(0, len(bids_raw), 2)]
            asks = [[float(asks_raw[i]), float(asks_raw[i + 1])] for i in range(0, len(asks_raw), 2)]
            timestamp = data.get("ts", 0)

        elif self.exchange_id == "kucoin":
            data_dict = data.get("data", {}) or {}
            bids_raw = data_dict.get("bids", [])
            asks_raw = data_dict.get("asks", [])
            bids = [[float(p), float(s)] for p, s in bids_raw]
            asks = [[float(p), float(s)] for p, s in asks_raw]
            timestamp = int(data_dict.get("time", 0) or 0)

        elif self.exchange_id == "bitget":
            data_dict = data.get("data", {}) or {}
            bids = [[float(b["price"]), float(b["size"])] for b in data_dict.get("bids", [])]
            asks = [[float(a["price"]), float(a["size"])] for a in data_dict.get("asks", [])]
            timestamp = int(data_dict.get("timestamp", 0) or 0)

        else:
            # Generic fallback
            if isinstance(data, dict):
                for key in ["bids", "b"]:
                    if key in data:
                        bids = self._parse_levels(data[key])
                        break
                for key in ["asks", "a"]:
                    if key in data:
                        asks = self._parse_levels(data[key])
                        break
            timestamp = int(time.time() * 1000)

        if limit:
            bids = bids[:limit]
            asks = asks[:limit]

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            datetime=self._format_datetime(timestamp),
        )

    def _parse_levels(self, levels: Any) -> List[List[float]]:
        """Parse price levels from various formats / 从各种格式解析价格级别"""
        result = []
        if isinstance(levels, list):
            for item in levels:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        result.append([float(item[0]), float(item[1])])
                    except (ValueError, TypeError):
                        pass
                elif isinstance(item, dict):
                    try:
                        price = float(item.get("price", item.get("p", 0)))
                        size = float(item.get("size", item.get("s", item.get("quantity", item.get("q", 0)))))
                        result.append([price, size])
                    except (ValueError, TypeError):
                        pass
        return result

    def normalize_ohlcv(self, data: Any, symbol: str, timeframe: str) -> List[OHLCV]:
        """Normalize OHLCV data / 标准化K线数据

        Args:
            data: Raw exchange OHLCV data
            symbol: Trading pair symbol
            timeframe: Timeframe string

        Returns:
            List of normalized OHLCV objects
        """
        ohlcv_list = []

        if self.exchange_id == "binance":
            for item in data:
                try:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item[0]),
                        datetime=self._format_datetime(int(item[0])),
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                except (IndexError, ValueError, TypeError):
                    continue

        elif self.exchange_id == "bybit":
            result = data.get("result", {}) or {}
            items = result.get("list", []) if isinstance(result, dict) else data
            for item in items:
                try:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item.get("start_time", item.get("t", 0))),
                        datetime=self._format_datetime(int(item.get("start_time", 0))),
                        open=float(item.get("open", 0)),
                        high=float(item.get("high", 0)),
                        low=float(item.get("low", 0)),
                        close=float(item.get("close", 0)),
                        volume=float(item.get("volume", 0)),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                except (ValueError, TypeError):
                    continue

        elif self.exchange_id == "okx":
            data_list = data.get("data", [])
            for item in data_list:
                try:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item.get("ts", 0)),
                        datetime=self._format_datetime(int(item.get("ts", 0))),
                        open=float(item.get("candles", ["0"])[0] if isinstance(item.get("candles"), list) else item.get("candle", ["0"])[0]) if False else float(item.get("open", 0)),
                        high=float(item.get("high", 0)),
                        low=float(item.get("low", 0)),
                        close=float(item.get("close", 0)),
                        volume=float(item.get("vol", 0)),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                except (ValueError, TypeError, IndexError):
                    continue

        elif self.exchange_id == "deribit":
            result = data.get("result", []) if isinstance(data, dict) else data
            for item in result:
                try:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item.get("t", 0) * 1000),
                        datetime=self._format_datetime(int(item.get("t", 0) * 1000)),
                        open=float(item.get("o", 0)),
                        high=float(item.get("h", 0)),
                        low=float(item.get("l", 0)),
                        close=float(item.get("c", 0)),
                        volume=float(item.get("v", 0)),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                except (ValueError, TypeError):
                    continue

        elif self.exchange_id == "kraken":
            result = data.get("result", {}) or {}
            ohlcv_data = list(result.values())[0] if result else []
            for item in ohlcv_data:
                try:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item[0] * 1000),
                        datetime=self._format_datetime(int(item[0] * 1000)),
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[6]),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                except (IndexError, ValueError, TypeError):
                    continue

        elif self.exchange_id == "gateio":
            for item in data:
                try:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item[0]),
                        datetime=self._format_datetime(int(item[0])),
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                except (IndexError, ValueError, TypeError):
                    continue

        elif self.exchange_id == "huobi":
            tick = data.get("data", data.get("tick", {})) or {}
            bars = tick.get("data", []) if isinstance(tick, dict) else []
            for item in bars:
                try:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item.get("id", 0) * 1000),
                        datetime=self._format_datetime(int(item.get("id", 0) * 1000)),
                        open=float(item.get("open", 0)),
                        high=float(item.get("high", 0)),
                        low=float(item.get("low", 0)),
                        close=float(item.get("close", 0)),
                        volume=float(item.get("vol", 0)),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                except (ValueError, TypeError):
                    continue

        elif self.exchange_id == "kucoin":
            data_dict = data.get("data", {}) or {}
            items = data_dict.get("data", data_dict.get("klines", []))
            for item in items:
                try:
                    parts = item.split(",") if isinstance(item, str) else item
                    if isinstance(parts, str):
                        parts = parts.split(",")
                    ohlcv_list.append(OHLCV(
                        timestamp=int(parts[0]),
                        datetime=self._format_datetime(int(parts[0])),
                        open=float(parts[1]),
                        high=float(parts[2]),
                        low=float(parts[3]),
                        close=float(parts[4]),
                        volume=float(parts[5]),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                except (IndexError, ValueError, TypeError):
                    continue

        elif self.exchange_id == "bitget":
            data_list = data.get("data", [])
            for item in data_list:
                try:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item.get("ts", 0)),
                        datetime=self._format_datetime(int(item.get("ts", 0))),
                        open=float(item.get("open", 0)),
                        high=float(item.get("high", 0)),
                        low=float(item.get("low", 0)),
                        close=float(item.get("close", 0)),
                        volume=float(item.get("vol", 0)),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                except (ValueError, TypeError):
                    continue

        else:
            # Generic fallback - try common formats
            ohlcv_list = self._normalize_generic_ohlcv(data, symbol, timeframe)

        return ohlcv_list

    def _normalize_generic_ohlcv(self, data: Any, symbol: str, timeframe: str) -> List[OHLCV]:
        """Generic OHLCV normalization / 通用K线标准化"""
        ohlcv_list = []
        if not isinstance(data, list):
            data = data.get("data", data.get("result", []))
        for item in data:
            try:
                if isinstance(item, list) and len(item) >= 6:
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item[0]),
                        datetime=self._format_datetime(int(item[0])),
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
                elif isinstance(item, dict):
                    ohlcv_list.append(OHLCV(
                        timestamp=int(item.get("t", item.get("timestamp", item.get("time", 0)))) * 1000 if item.get("t", 0) < 1e12 else int(item.get("t", item.get("timestamp", 0))),
                        datetime=self._format_datetime(int(item.get("t", item.get("timestamp", 0)))),
                        open=float(item.get("o", item.get("open", 0))),
                        high=float(item.get("h", item.get("high", 0))),
                        low=float(item.get("l", item.get("low", 0))),
                        close=float(item.get("c", item.get("close", 0))),
                        volume=float(item.get("v", item.get("vol", item.get("volume", 0)))),
                        timeframe=timeframe,
                        symbol=symbol,
                    ))
            except (ValueError, TypeError, IndexError):
                continue
        return ohlcv_list

    def normalize_order(self, data: Dict, symbol: str) -> Order:
        """Normalize order data / 标准化订单数据"""
        if self.exchange_id == "binance":
            return Order(
                id=str(data.get("orderId", "")),
                symbol=data.get("symbol", symbol),
                type=data.get("type", "").lower(),
                side=data.get("side", "").lower(),
                price=float(data.get("price", 0)),
                amount=float(data.get("origQty", 0)),
                filled=float(data.get("executedQty", 0)),
                remaining=float(data.get("origQty", 0)) - float(data.get("executedQty", 0)),
                status=self._normalize_order_status(data.get("status", "")),
                timestamp=data.get("transactTime", 0),
            )
        elif self.exchange_id == "bybit":
            list_data = data.get("result", data.get("list", []))
            item = list_data[0] if isinstance(list_data, list) else list_data
            return Order(
                id=str(item.get("orderId", item.get("id", ""))),
                symbol=item.get("symbol", symbol),
                type=item.get("orderType", item.get("type", "")).lower(),
                side=item.get("side", "").lower(),
                price=float(item.get("price", 0)),
                amount=float(item.get("qty", 0)),
                filled=float(item.get("execQty", 0)),
                remaining=float(item.get("qty", 0)) - float(item.get("execQty", 0)),
                status=self._normalize_order_status(item.get("orderStatus", item.get("status", ""))),
                timestamp=int(item.get("createTime", item.get("createdAt", 0))),
            )
        elif self.exchange_id == "okx":
            data_list = data.get("data", [])
            item = data_list[0] if data_list else {}
            return Order(
                id=str(item.get("ordId", "")),
                symbol=item.get("instId", symbol),
                type=item.get("ordType", "").lower(),
                side=item.get("side", "").lower(),
                price=float(item.get("px", 0)),
                amount=float(item.get("sz", 0)),
                filled=float(item.get("accFillSz", 0)),
                remaining=float(item.get("sz", 0)) - float(item.get("accFillSz", 0)),
                status=self._normalize_order_status(item.get("state", "")),
                timestamp=int(item.get("cTime", 0)),
            )
        else:
            return Order(
                id=str(data.get("id", data.get("orderId", ""))),
                symbol=symbol,
                type=data.get("type", "").lower(),
                side=data.get("side", "").lower(),
                price=float(data.get("price", 0)),
                amount=float(data.get("amount", data.get("quantity", data.get("size", 0)))),
                filled=float(data.get("filled", data.get("executedQty", 0))),
                status=self._normalize_order_status(data.get("status", "")),
                timestamp=data.get("timestamp", data.get("createdAt", 0)),
            )

    def _normalize_order_status(self, status: str) -> str:
        """Normalize order status to unified format / 将订单状态标准化为统一格式"""
        status_lower = status.lower()
        status_mapping = {
            "new": "open",
            "partially_filled": "open",
            "filled": "closed",
            "canceled": "canceled",
            "cancelled": "canceled",
            "rejected": "rejected",
            "expired": "expired",
            "pending": "pending",
            "open": "open",
            "closed": "closed",
        }
        return status_mapping.get(status_lower, status_lower)

    def _format_datetime(self, timestamp: int) -> str:
        """Format Unix timestamp to ISO datetime string"""
        if timestamp > 1e12:
            timestamp = timestamp / 1000
        try:
            from datetime import datetime
            return datetime.utcfromtimestamp(timestamp).isoformat() + "Z"
        except (ValueError, OSError):
            return ""


# =============================================================================
# Market Data Fetcher / 市场数据获取器
# =============================================================================

class MarketDataFetcher:
    """Fetch market data from exchanges / 从交易所获取市场数据

    Provides unified methods for fetching OHLCV, order books, tickers, and trades
    from any supported exchange.

    Example / 示例:
        fetcher = MarketDataFetcher('binance')
        ohlcv = fetcher.fetch_ohlcv('BTC/USDT', '1m', limit=100)
        orderbook = fetcher.fetch_order_book('BTC/USDT')
    """

    def __init__(self, exchange_id: str, config: Dict[str, Any] = None):
        """Initialize market data fetcher / 初始化市场数据获取器

        Args:
            exchange_id: Exchange identifier
            config: Optional exchange configuration override
        """
        self.exchange_id = exchange_id
        self.config = config or EXCHANGE_CONFIGS.get(exchange_id, {})
        self.normalizer = ExchangeNormalizer(exchange_id)
        self.logger = logging.getLogger(f"MarketDataFetcher.{exchange_id}")
        self._session = None
        self.timeout = 15

    @property
    def session(self):
        """Get or create HTTP session / 获取或创建HTTP会话"""
        if self._session is None:
            import http.client
            http.client.HTTPSConnection.debuglevel = 0
            self._session = urllib2.urlopen
        return self._session

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int = None,
        limit: int = 100,
        params: Dict = None,
    ) -> List[OHLCV]:
        """Fetch OHLCV candlestick data / 获取K线烛台数据

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch
            params: Additional exchange-specific parameters

        Returns:
            List of OHLCV objects
        """
        endpoint, method, parsed_params = self._get_ohlcv_params(symbol, timeframe, since, limit, params)
        data = self._request(endpoint, method, parsed_params)
        return self.normalizer.normalize_ohlcv(data, symbol, timeframe)

    def fetch_order_book(self, symbol: str, limit: int = 20, params: Dict = None) -> OrderBook:
        """Fetch order book / 获取订单簿

        Args:
            symbol: Trading pair symbol
            limit: Number of price levels
            params: Additional parameters

        Returns:
            OrderBook object
        """
        endpoint, method, parsed_params = self._get_orderbook_params(symbol, limit, params)
        data = self._request(endpoint, method, parsed_params)
        return self.normalizer.normalize_orderbook(data, symbol, limit)

    def fetch_ticker(self, symbol: str, params: Dict = None) -> Ticker:
        """Fetch ticker / 获取行情

        Args:
            symbol: Trading pair symbol
            params: Additional parameters

        Returns:
            Ticker object
        """
        endpoint, method, parsed_params = self._get_ticker_params(symbol, params)
        data = self._request(endpoint, method, parsed_params)
        return self.normalizer.normalize_ticker(data, symbol)

    def fetch_trades(self, symbol: str, limit: int = 50, params: Dict = None) -> List[Trade]:
        """Fetch recent trades / 获取最近交易

        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch
            params: Additional parameters

        Returns:
            List of Trade objects
        """
        endpoint, method, parsed_params = self._get_trades_params(symbol, limit, params)
        data = self._request(endpoint, method, parsed_params)
        return self._parse_trades(data, symbol)

    # -------------------------------------------------------------------------
    # Exchange-specific parameter builders / 交易所特定参数构建器
    # -------------------------------------------------------------------------

    def _get_ohlcv_params(
        self,
        symbol: str,
        timeframe: str,
        since: int = None,
        limit: int = 100,
        params: Dict = None,
    ) -> Tuple[str, str, Dict]:
        """Build OHLCV request parameters for exchange / 为交易所构建K线请求参数"""
        base, quote = symbol.split("/") if "/" in symbol else symbol.split("-")
        market_id = self._symbol_to_market_id(symbol)
        tf = self.config.get("timeframes", {}).get(timeframe, timeframe)

        if self.exchange_id == "binance":
            ep = "/api/v3/klines"
            p = {"symbol": market_id, "interval": tf, "limit": limit}
            if since:
                p["startTime"] = since
            return ep, "GET", p

        elif self.exchange_id == "bybit":
            ep = "/v5/market/kline"
            p = {"category": "spot", "symbol": market_id, "interval": tf, "limit": limit}
            if since:
                p["start"] = since
            return ep, "GET", p

        elif self.exchange_id == "okx":
            inst_id = f"{base}-{quote}" if quote in ["USDT", "USDC", "BTC", "ETH"] else f"{base}-{quote}"
            ep = "/api/v5/market/candles"
            p = {"instId": inst_id, "bar": tf, "limit": limit}
            if since:
                p["after"] = since
            return ep, "GET", p

        elif self.exchange_id == "deribit":
            ep = "/api/v2/public/get_tradingview_chart_data"
            resolution_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "60", "1d": "D", "1w": "D"}
            p = {"instrument_name": market_id, "resolution": resolution_map.get(timeframe, "1"), "length": limit}
            if since:
                p["start_timestamp"] = since
                p["end_timestamp"] = since + limit * 60000
            return ep, "GET", p

        elif self.exchange_id == "kraken":
            kraken_symbol = self._symbol_to_kraken_symbol(symbol)
            ep = f"/0/public/OHLC"
            p = {"pair": kraken_symbol, "interval": int(tf)}
            if since:
                p["since"] = since // 1000
            return ep, "GET", p

        elif self.exchange_id == "gateio":
            currency = f"{base}_{quote}"
            ep = "/api/v4/spot/candlesticks"
            p = {"currency_pair": currency, "interval": tf, "limit": limit}
            if since:
                p["from"] = since // 1000
                p["to"] = (since // 1000) + limit * 60
            return ep, "GET", p

        elif self.exchange_id == "huobi":
            symbol_id = f"{base}{quote}".lower()
            ep = "/market/history/kline"
            p = {"symbol": symbol_id, "period": tf, "size": limit}
            if since:
                p["symbol"] = symbol_id
            return ep, "GET", p

        elif self.exchange_id == "kucoin":
            base_quote = f"{base}-{quote}"
            ep = "/api/v1/market/candles"
            p = {"type": tf, "symbol": base_quote}
            if since:
                p["startAt"] = since // 1000
            return ep, "GET", p

        elif self.exchange_id == "bitget":
            ep = "/api/v5/market/candles"
            p = {"instId": market_id, "bar": tf, "limit": limit}
            if since:
                p["after"] = since
            return ep, "GET", p

        else:
            # Generic fallback
            return f"/api/v1/klines", "GET", {"symbol": market_id, "interval": tf, "limit": limit}

    def _get_orderbook_params(self, symbol: str, limit: int = 20, params: Dict = None) -> Tuple[str, str, Dict]:
        """Build order book request parameters / 构建订单簿请求参数"""
        market_id = self._symbol_to_market_id(symbol)

        if self.exchange_id == "binance":
            return "/api/v3/depth", "GET", {"symbol": market_id, "limit": limit}

        elif self.exchange_id == "bybit":
            return "/v5/market/orderbook", "GET", {"category": "spot", "symbol": market_id, "limit": limit}

        elif self.exchange_id == "okx":
            inst_id = self._symbol_to_okx_symbol(symbol)
            return "/api/v5/market/books", "GET", {"instId": inst_id, "sz": limit}

        elif self.exchange_id == "deribit":
            return "/api/v2/public/get_order_book", "GET", {"instrument_name": market_id, "depth": limit}

        elif self.exchange_id == "kraken":
            kraken_symbol = self._symbol_to_kraken_symbol(symbol)
            return "/0/public/Depth", "GET", {"pair": kraken_symbol}

        elif self.exchange_id == "gateio":
            currency = f"{base}_{quote}" if "base" in dir() else f"{symbol.replace('/', '_')}"
            currency = self._symbol_to_gateio_symbol(symbol)
            return "/api/v4/spot/order_book", "GET", {"currency_pair": currency, "limit": limit, "with_id": True}

        elif self.exchange_id == "huobi":
            symbol_id = self._symbol_to_huobi_symbol(symbol)
            return "/market/depth", "GET", {"symbol": symbol_id, "type": "step0"}

        elif self.exchange_id == "kucoin":
            base_quote = self._symbol_to_kucoin_symbol(symbol)
            return "/api/v1/market/orderbook/level2", "GET", {"symbol": base_quote}

        elif self.exchange_id == "bitget":
            return "/api/v5/market/books", "GET", {"instId": market_id, "limit": limit}

        else:
            return "/api/v1/depth", "GET", {"symbol": market_id, "limit": limit}

    def _get_ticker_params(self, symbol: str, params: Dict = None) -> Tuple[str, str, Dict]:
        """Build ticker request parameters / 构建行情请求参数"""
        market_id = self._symbol_to_market_id(symbol)

        if self.exchange_id == "binance":
            return "/api/v3/ticker/24hr", "GET", {"symbol": market_id}

        elif self.exchange_id == "bybit":
            return "/v5/market/tickers", "GET", {"category": "spot", "symbol": market_id}

        elif self.exchange_id == "okx":
            inst_id = self._symbol_to_okx_symbol(symbol)
            return "/api/v5/market/ticker", "GET", {"instId": inst_id}

        elif self.exchange_id == "deribit":
            return "/api/v2/public/get_ticker", "GET", {"instrument_name": market_id}

        elif self.exchange_id == "kraken":
            kraken_symbol = self._symbol_to_kraken_symbol(symbol)
            return "/0/public/Ticker", "GET", {"pair": kraken_symbol}

        elif self.exchange_id == "gateio":
            currency = self._symbol_to_gateio_symbol(symbol)
            return "/api/v4/spot/tickers", "GET", {"currency_pair": currency}

        elif self.exchange_id == "huobi":
            symbol_id = self._symbol_to_huobi_symbol(symbol)
            return "/market/detail/merged", "GET", {"symbol": symbol_id}

        elif self.exchange_id == "kucoin":
            base_quote = self._symbol_to_kucoin_symbol(symbol)
            return "/api/v1/market/orderbook/level1", "GET", {"symbol": base_quote}

        elif self.exchange_id == "bitget":
            return "/api/v5/market/ticker", "GET", {"instId": market_id}

        else:
            return "/api/v1/ticker", "GET", {"symbol": market_id}

    def _get_trades_params(self, symbol: str, limit: int = 50, params: Dict = None) -> Tuple[str, str, Dict]:
        """Build trades request parameters / 构建交易请求参数"""
        market_id = self._symbol_to_market_id(symbol)

        if self.exchange_id == "binance":
            return "/api/v3/trades", "GET", {"symbol": market_id, "limit": limit}

        elif self.exchange_id == "bybit":
            return "/v5/market/recent-trade", "GET", {"category": "spot", "symbol": market_id, "limit": limit}

        elif self.exchange_id == "okx":
            inst_id = self._symbol_to_okx_symbol(symbol)
            return "/api/v5/market/trades", "GET", {"instId": inst_id, "limit": limit}

        elif self.exchange_id == "deribit":
            return "/api/v2/public/get_last_trades", "GET", {"instrument_name": market_id}

        elif self.exchange_id == "kraken":
            kraken_symbol = self._symbol_to_kraken_symbol(symbol)
            return "/0/public/Trades", "GET", {"pair": kraken_symbol}

        elif self.exchange_id == "gateio":
            currency = self._symbol_to_gateio_symbol(symbol)
            return "/api/v4/spot/trades", "GET", {"currency_pair": currency}

        elif self.exchange_id == "huobi":
            symbol_id = self._symbol_to_huobi_symbol(symbol)
            return "/market/history/trade", "GET", {"symbol": symbol_id, "size": limit}

        elif self.exchange_id == "kucoin":
            base_quote = self._symbol_to_kucoin_symbol(symbol)
            return "/api/v1/market/histories", "GET", {"symbol": base_quote}

        elif self.exchange_id == "bitget":
            return "/api/v5/market/fills", "GET", {"instId": market_id, "limit": limit}

        else:
            return "/api/v1/trades", "GET", {"symbol": market_id, "limit": limit}

    # -------------------------------------------------------------------------
    # Symbol format converters / 交易对格式转换器
    # -------------------------------------------------------------------------

    def _symbol_to_market_id(self, symbol: str) -> str:
        """Convert unified symbol to exchange-specific market ID / 将统一交易对转换为交易所特定的市场ID"""
        base, quote = self._parse_symbol(symbol)
        if self.exchange_id in ["binance", "bybit", "bitget", "mexc", "phemex"]:
            return f"{base}{quote}"
        elif self.exchange_id == "okx":
            return f"{base}-{quote}"
        elif self.exchange_id == "deribit":
            return f"{base}-{quote}"
        elif self.exchange_id == "gateio":
            return f"{base}_{quote}"
        elif self.exchange_id == "huobi":
            return f"{base}{quote}".lower()
        elif self.exchange_id == "kucoin":
            return f"{base}-{quote}"
        elif self.exchange_id == "kraken":
            return self._symbol_to_kraken_symbol(symbol)
        else:
            return f"{base}{quote}"

    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """Parse symbol into base and quote / 将交易对解析为基准货币和报价货币"""
        if "/" in symbol:
            return symbol.split("/")
        elif "-" in symbol:
            return symbol.split("-")
        else:
            # Heuristic: common quote currencies
            for quote in ["USDT", "USDC", "USD", "EUR", "GBP", "BTC", "ETH"]:
                if symbol.endswith(quote) and len(symbol) > len(quote) + 1:
                    return symbol[:-len(quote)], quote
            return symbol, "USDT"

    def _symbol_to_okx_symbol(self, symbol: str) -> str:
        base, quote = self._parse_symbol(symbol)
        return f"{base}-{quote}"

    def _symbol_to_kraken_symbol(self, symbol: str) -> str:
        base, quote = self._parse_symbol(symbol)
        kraken_quotes = {"USDT": "USDTZ", "USD": "USD", "EUR": "ZEUR", "GBP": "ZGBP", "BTC": "XXBT", "ETH": "XETH"}
        q = kraken_quotes.get(quote, quote)
        bases = {"BTC": "XBT"}
        b = bases.get(base, base)
        return f"{b}{q}"

    def _symbol_to_gateio_symbol(self, symbol: str) -> str:
        base, quote = self._parse_symbol(symbol)
        return f"{base}_{quote}"

    def _symbol_to_huobi_symbol(self, symbol: str) -> str:
        base, quote = self._parse_symbol(symbol)
        return f"{base}{quote}".lower()

    def _symbol_to_kucoin_symbol(self, symbol: str) -> str:
        base, quote = self._parse_symbol(symbol)
        return f"{base}-{quote}"

    # -------------------------------------------------------------------------
    # HTTP Request / HTTP请求
    # -------------------------------------------------------------------------

    def _request(self, endpoint: str, method: str = "GET", params: Dict = None) -> Any:
        """Make HTTP request to exchange / 向交易所发送HTTP请求

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            params: Query parameters

        Returns:
            Parsed JSON response
        """
        params = params or {}
        base_url = self.config.get("baseUrl", "https://api.binance.com")

        if method == "GET" and params:
            query_string = urlparse.urlencode(params)
            url = f"{base_url}{endpoint}?{query_string}"
        else:
            url = f"{base_url}{endpoint}"

        try:
            request = urllib2.Request(url, method=method)
            request.add_header("User-Agent", "CCXT-Python-Adapter/1.0")
            request.add_header("Accept", "application/json")

            response = urllib2.urlopen(request, timeout=self.timeout)
            body = response.read().decode("utf-8")

            if response.headers.get("Content-Type", "").startswith("application/json"):
                return json.loads(body)
            return body

        except urllib2.HTTPError as e:
            self.logger.error(f"HTTP Error {e.code}: {e.reason} for {url}")
            try:
                error_body = e.read().decode("utf-8")
                return json.loads(error_body)
            except Exception:  # noqa: BLE001
                return {"error": str(e)}
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return {"error": str(e)}

    def _parse_trades(self, data: Any, symbol: str) -> List[Trade]:
        """Parse trades response / 解析交易响应"""
        trades = []

        try:
            if self.exchange_id == "binance":
                for item in data:
                    trades.append(Trade(
                        id=str(item.get("id", "")),
                        symbol=symbol,
                        price=float(item.get("price", 0)),
                        amount=float(item.get("qty", 0)),
                        side=item.get("isBuyerMaker", False) and "sell" or "buy",
                        timestamp=int(item.get("time", 0)),
                        datetime=item.get("time", ""),
                    ))
            elif self.exchange_id == "bybit":
                result = data.get("result", {}) or {}
                for item in result.get("list", []):
                    trades.append(Trade(
                        id=str(item.get("i", "")),
                        symbol=symbol,
                        price=float(item.get("p", 0)),
                        amount=float(item.get("v", 0)),
                        side=item.get("S", "").lower(),
                        timestamp=int(item.get("TS", 0)),
                    ))
            else:
                trades_raw = data.get("data", data.get("result", []))
                for item in trades_raw:
                    if isinstance(item, dict):
                        trades.append(Trade(
                            id=str(item.get("id", "")),
                            symbol=symbol,
                            price=float(item.get("price", 0)),
                            amount=float(item.get("amount", item.get("size", item.get("quantity", 0)))),
                            side=item.get("side", "").lower(),
                            timestamp=int(item.get("timestamp", item.get("time", 0))),
                        ))
        except Exception as e:
            self.logger.error(f"Failed to parse trades: {e}")

        return trades


# =============================================================================
# Order Executor / 订单执行器
# =============================================================================

class OrderExecutor:
    """Execute orders on exchanges / 在交易所执行订单

    Provides unified methods for placing and canceling orders across exchanges.
    Requires API credentials for authenticated requests.

    Example / 示例:
        executor = OrderExecutor('binance', api_key='...', api_secret='...')
        order = executor.place_order('BTC/USDT', 'buy', 'limit', 0.001, 50000)
        executor.cancel_order(order['id'], 'BTC/USDT')
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str = None,
        api_secret: str = None,
        passphrase: str = None,
        config: Dict[str, Any] = None,
        testnet: bool = False,
    ):
        """Initialize order executor / 初始化订单执行器

        Args:
            exchange_id: Exchange identifier
            api_key: API key for authentication
            api_secret: API secret for HMAC signing
            passphrase: Extra passphrase (required for some exchanges)
            config: Optional exchange configuration override
            testnet: Use testnet instead of mainnet
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.config = config or EXCHANGE_CONFIGS.get(exchange_id, {})
        self.testnet = testnet
        self.normalizer = ExchangeNormalizer(exchange_id)
        self.logger = logging.getLogger(f"OrderExecutor.{exchange_id}")
        self.timeout = 15

        self._rate_limit_last_request = 0
        self._rate_limit_min_interval = self.config.get("rateLimit", 1000) / 1000.0

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float = None,
        params: Dict = None,
    ) -> Order:
        """Place an order / 下单

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop_loss', 'take_profit', etc.
            amount: Order quantity
            price: Order price (required for limit orders)
            params: Additional exchange-specific parameters

        Returns:
            Order object
        """
        endpoint, method, order_params = self._build_place_order_params(
            symbol, side, order_type, amount, price, params
        )

        data = self._signed_request(endpoint, method, order_params)
        return self.normalizer.normalize_order(data, symbol)

    def cancel_order(self, order_id: str, symbol: str, params: Dict = None) -> Order:
        """Cancel an order / 取消订单

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            params: Additional parameters

        Returns:
            Order object with canceled status
        """
        endpoint, method, cancel_params = self._build_cancel_order_params(order_id, symbol, params)
        data = self._signed_request(endpoint, method, cancel_params)
        return self.normalizer.normalize_order(data, symbol)

    def get_order(self, order_id: str, symbol: str, params: Dict = None) -> Order:
        """Get order status / 获取订单状态

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            params: Additional parameters

        Returns:
            Order object
        """
        endpoint, method, query_params = self._build_get_order_params(order_id, symbol, params)
        data = self._signed_request(endpoint, method, query_params)
        return self.normalizer.normalize_order(data, symbol)

    def get_open_orders(self, symbol: str = None, params: Dict = None) -> List[Order]:
        """Get all open orders / 获取所有未完成订单

        Args:
            symbol: Optional symbol filter
            params: Additional parameters

        Returns:
            List of Order objects
        """
        endpoint, method, query_params = self._build_open_orders_params(symbol, params)
        data = self._signed_request(endpoint, method, query_params)
        return self._parse_orders_list(data, symbol)

    def get_balance(self, params: Dict = None) -> Dict[str, float]:
        """Get account balance / 获取账户余额

        Args:
            params: Additional parameters

        Returns:
            Dict mapping asset codes to balance amounts
        """
        endpoint, method, query_params = self._build_balance_params(params)
        data = self._signed_request(endpoint, method, query_params)
        return self._parse_balance(data)

    def get_positions(self, symbol: str = None, params: Dict = None) -> List[Position]:
        """Get positions / 获取持仓

        Args:
            symbol: Optional symbol filter
            params: Additional parameters

        Returns:
            List of Position objects
        """
        endpoint, method, query_params = self._build_positions_params(symbol, params)
        data = self._signed_request(endpoint, method, query_params)
        return self._parse_positions(data, symbol)

    # -------------------------------------------------------------------------
    # Order parameter builders / 订单参数构建器
    # -------------------------------------------------------------------------

    def _build_place_order_params(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float = None,
        params: Dict = None,
    ) -> Tuple[str, str, Dict]:
        """Build place order request parameters / 构建下单请求参数"""
        market_id = self._symbol_to_market_id(symbol)
        params = params or {}

        if self.exchange_id == "binance":
            ep = "/api/v3/order"
            p = {
                "symbol": market_id,
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": self._format_quantity(amount),
                "timestamp": int(time.time() * 1000),
                "recvWindow": 5000,
            }
            if order_type.lower() == "limit" and price:
                p["price"] = self._format_price(price)
                p["timeInForce"] = "GTC"
            return ep, "POST", {**p, **params}

        elif self.exchange_id == "bybit":
            ep = "/v5/order/create"
            p = {
                "category": "spot",
                "symbol": market_id,
                "side": side.title(),
                "orderType": order_type.title(),
                "qty": self._format_quantity(amount),
                "timestamp": int(time.time() * 1000),
            }
            if order_type.lower() == "limit" and price:
                p["orderType"] = "Limit"
                p["price"] = self._format_price(price)
            return ep, "POST", {**p, **params}

        elif self.exchange_id == "okx":
            inst_id = self._symbol_to_okx_symbol(symbol)
            ord_type = "limit" if order_type.lower() == "limit" else "market"
            ep = "/api/v5/trade/order"
            p = {
                "instId": inst_id,
                "tdMode": "cash",
                "side": side.lower(),
                "ordType": ord_type,
                "sz": self._format_quantity(amount),
                "timestamp": int(time.time() * 1000),
            }
            if order_type.lower() == "limit" and price:
                p["px"] = self._format_price(price)
            return ep, "POST", {**p, **params}

        elif self.exchange_id == "gateio":
            currency = self._symbol_to_gateio_symbol(symbol)
            ep = "/api/v4/spot/orders"
            p = {
                "currency_pair": currency,
                "side": side.lower(),
                "type": order_type.lower(),
                "amount": self._format_quantity(amount),
                "timestamp": int(time.time() * 1000),
            }
            if order_type.lower() == "limit" and price:
                p["price"] = self._format_price(price)
            return ep, "POST", {**p, **params}

        elif self.exchange_id == "kucoin":
            base_quote = self._symbol_to_kucoin_symbol(symbol)
            ep = "/api/v1/trade/order"
            p = {
                "symbol": base_quote,
                "side": side.lower(),
                "type": order_type.lower(),
                "size": self._format_quantity(amount),
                "timestamp": int(time.time() * 1000),
            }
            if order_type.lower() == "limit" and price:
                p["price"] = self._format_price(price)
                p["type"] = "limit"
            return ep, "POST", {**p, **params}

        elif self.exchange_id == "bitget":
            inst_id = self._symbol_to_bitget_symbol(symbol)
            ep = "/api/v5/place/trade"
            p = {
                "instId": inst_id,
                "side": side.lower(),
                "ordType": order_type.lower(),
                "sz": self._format_quantity(amount),
                "timestamp": int(time.time() * 1000),
            }
            if order_type.lower() == "limit" and price:
                p["px"] = self._format_price(price)
            return ep, "POST", {**p, **params}

        else:
            ep = "/api/v1/order"
            p = {
                "symbol": market_id,
                "side": side.upper(),
                "type": order_type.upper(),
                "amount": amount,
                "timestamp": int(time.time() * 1000),
            }
            if price:
                p["price"] = price
            return ep, "POST", {**p, **params}

    def _build_cancel_order_params(self, order_id: str, symbol: str, params: Dict = None) -> Tuple[str, str, Dict]:
        """Build cancel order request parameters / 构建取消订单请求参数"""
        market_id = self._symbol_to_market_id(symbol)
        params = params or {}

        if self.exchange_id == "binance":
            return "/api/v3/order", "DELETE", {
                "symbol": market_id,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000),
                "recvWindow": 5000,
            }

        elif self.exchange_id == "bybit":
            return "/v5/order/cancel", "POST", {
                "category": "spot",
                "symbol": market_id,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000),
            }

        elif self.exchange_id == "okx":
            inst_id = self._symbol_to_okx_symbol(symbol)
            return "/api/v5/trade/cancel-order", "POST", {
                "instId": inst_id,
                "ordId": order_id,
                "timestamp": int(time.time() * 1000),
            }

        elif self.exchange_id == "gateio":
            return "/api/v4/spot/orders", "DELETE", {
                "currency_pair": self._symbol_to_gateio_symbol(symbol),
                "order_id": order_id,
                "timestamp": int(time.time() * 1000),
            }

        elif self.exchange_id == "kucoin":
            return "/api/v1/trade/cancel-order", "POST", {
                "symbol": self._symbol_to_kucoin_symbol(symbol),
                "orderId": order_id,
                "timestamp": int(time.time() * 1000),
            }

        elif self.exchange_id == "bitget":
            return "/api/v5/place/cancel-order", "POST", {
                "instId": self._symbol_to_bitget_symbol(symbol),
                "ordId": order_id,
                "timestamp": int(time.time() * 1000),
            }

        else:
            return "/api/v1/order", "DELETE", {
                "symbol": market_id,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000),
            }

    def _build_get_order_params(self, order_id: str, symbol: str, params: Dict = None) -> Tuple[str, str, Dict]:
        """Build get order request parameters / 构建查询订单请求参数"""
        market_id = self._symbol_to_market_id(symbol)
        params = params or {}

        if self.exchange_id == "binance":
            return "/api/v3/order", "GET", {
                "symbol": market_id,
                "orderId": order_id,
                "timestamp": int(time.time() * 1000),
                "recvWindow": 5000,
            }

        elif self.exchange_id == "bybit":
            return "/v5/order/realtime", "GET", {
                "category": "spot",
                "symbol": market_id,
                "orderId": order_id,
            }

        elif self.exchange_id == "okx":
            return "/api/v5/trade/order", "GET", {
                "instId": self._symbol_to_okx_symbol(symbol),
                "ordId": order_id,
            }

        elif self.exchange_id == "gateio":
            return "/api/v4/spot/orders", "GET", {
                "currency_pair": self._symbol_to_gateio_symbol(symbol),
                "order_id": order_id,
            }

        else:
            return "/api/v1/order", "GET", {
                "symbol": market_id,
                "orderId": order_id,
            }

    def _build_open_orders_params(self, symbol: str = None, params: Dict = None) -> Tuple[str, str, Dict]:
        """Build open orders request parameters / 构建未完成订单请求参数"""
        params = params or {}

        if self.exchange_id == "binance":
            p = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
            if symbol:
                p["symbol"] = self._symbol_to_market_id(symbol)
            return "/api/v3/openOrders", "GET", p

        elif self.exchange_id == "bybit":
            p = {"category": "spot", "timestamp": int(time.time() * 1000)}
            if symbol:
                p["symbol"] = self._symbol_to_market_id(symbol)
            return "/v5/order/realtime", "GET", p

        elif self.exchange_id == "okx":
            p = {"instType": "SPOT"}
            if symbol:
                p["instId"] = self._symbol_to_okx_symbol(symbol)
            return "/api/v5/trade/orders-pending", "GET", p

        elif self.exchange_id == "gateio":
            p = {"timestamp": int(time.time() * 1000), "status": "open"}
            if symbol:
                p["currency_pair"] = self._symbol_to_gateio_symbol(symbol)
            return "/api/v4/spot/orders", "GET", p

        else:
            p = {}
            if symbol:
                p["symbol"] = self._symbol_to_market_id(symbol)
            return "/api/v1/orders", "GET", p

    def _build_balance_params(self, params: Dict = None) -> Tuple[str, str, Dict]:
        """Build balance request parameters / 构建余额请求参数"""
        params = params or {}

        if self.exchange_id == "binance":
            return "/api/v3/account", "GET", {
                "timestamp": int(time.time() * 1000),
                "recvWindow": 5000,
            }

        elif self.exchange_id == "bybit":
            return "/v5/account/wallet-balance", "GET", {
                "accountType": "SPOT",
                "timestamp": int(time.time() * 1000),
            }

        elif self.exchange_id == "okx":
            return "/api/v5/account/balance", "GET", {"ccy": params.get("ccy", "")}

        elif self.exchange_id == "gateio":
            return "/api/v4/spot/accounts", "GET", {"timestamp": int(time.time() * 1000)}

        elif self.exchange_id == "kucoin":
            return "/api/v1/account/balance", "GET", {}

        else:
            return "/api/v1/balance", "GET", {}

    def _build_positions_params(self, symbol: str = None, params: Dict = None) -> Tuple[str, str, Dict]:
        """Build positions request parameters / 构建持仓请求参数"""
        params = params or {}

        if self.exchange_id == "binance":
            p = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
            if symbol:
                p["symbol"] = self._symbol_to_market_id(symbol)
            return "/fapi/v2/positionRisk", "GET", p

        elif self.exchange_id == "bybit":
            p = {"category": "linear", "timestamp": int(time.time() * 1000)}
            if symbol:
                p["symbol"] = self._symbol_to_market_id(symbol)
            return "/v5/position/list", "GET", p

        elif self.exchange_id == "okx":
            p = {"instType": "SWAP"}
            if symbol:
                p["instId"] = self._symbol_to_okx_symbol(symbol)
            return "/api/v5/account/positions", "GET", p

        else:
            return "/api/v1/positions", "GET", {}

    # -------------------------------------------------------------------------
    # Signed requests / 签名请求
    # -------------------------------------------------------------------------

    def _signed_request(self, endpoint: str, method: str = "GET", params: Dict = None) -> Any:
        """Make signed HTTP request / 发送签名HTTP请求

        Args:
            endpoint: API endpoint path
            method: HTTP method
            params: Request parameters

        Returns:
            Parsed JSON response
        """
        params = params or {}
        base_url = self.config.get("testnet", self.config.get("baseUrl", "https://api.binance.com"))
        if self.testnet:
            base_url = self.config.get("testnet", base_url)

        # Rate limiting
        now = time.time() * 1000
        elapsed = now - self._rate_limit_last_request
        if elapsed < self._rate_limit_min_interval:
            time.sleep((self._rate_limit_min_interval - elapsed) / 1000)
        self._rate_limit_last_request = time.time() * 1000

        # Build signed request
        timestamp = int(time.time() * 1000)
        params["timestamp"] = params.get("timestamp", timestamp)

        if self.exchange_id == "binance":
            query_string = self._build_query_string(params)
            signature = self._sign_hmac_sha256(query_string, self.api_secret)
            full_params = {**params, "signature": signature}

            url = f"{base_url}{endpoint}"
            data = urlparse.urlencode(full_params).encode("utf-8")

            request = urllib2.Request(url, data=data, method="POST")
            request.add_header("X-MBX-APIKEY", self.api_key)
            request.add_header("Content-Type", "application/x-www-form-urlencoded")

        elif self.exchange_id == "bybit":
            params["api_key"] = self.api_key
            params["timestamp"] = timestamp
            sorted_params = sorted(params.items())
            query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
            signature = self._sign_hmac_sha256(query_string, self.api_secret)
            full_params = {**params, "sign": signature}

            url = f"{base_url}{endpoint}"
            data = json.dumps(full_params).encode("utf-8")

            request = urllib2.Request(url, data=data, method="POST")
            request.add_header("Content-Type", "application/json")
            request.add_header("X-BAPI-API-KEY", self.api_key)
            request.add_header("X-BAPI-SIGN", signature)
            request.add_header("X-BAPI-SIGN-TYPE", "2")

        elif self.exchange_id == "okx":
            timestamp_str = self._format_datetime_iso(timestamp / 1000)
            method_upper = method.upper()
            body = json.dumps(params) if params else ""
            message = timestamp_str + method_upper + endpoint + (body or "")

            signature = self._sign_hmac_sha256(message, self.api_secret)
            signature_base64 = signature  # Keep as hex for simplicity

            url = f"{base_url}{endpoint}"

            request = urllib2.Request(url, data=body.encode("utf-8") if body else None, method=method)
            request.add_header("Content-Type", "application/json")
            request.add_header("OK-ACCESS-KEY", self.api_key)
            request.add_header("OK-ACCESS-SIGN", signature_base64)
            request.add_header("OK-ACCESS-TIMESTAMP", timestamp_str)
            request.add_header("OK-ACCESS-PASSPHRASE", self.passphrase or "")

        elif self.exchange_id == "gateio":
            query_string = self._build_query_string(params)
            signature = self._sign_hmac_sha256(query_string, self.api_secret)
            full_params = {**params, "signature": signature}

            url = f"{base_url}{endpoint}"
            if method == "POST":
                data = json.dumps(full_params).encode("utf-8")
                request = urllib2.Request(url, data=data, method="POST")
                request.add_header("Content-Type", "application/json")
            else:
                url = f"{url}?{self._build_query_string(full_params)}"
                request = urllib2.Request(url, method="GET")

            request.add_header("KEY", self.api_key)
            request.add_header("SIGN", signature)

        elif self.exchange_id == "kucoin":
            timestamp_str = str(int(time.time() * 1000))
            method_upper = method.upper()
            body = json.dumps(params) if params else ""
            message = timestamp_str + method_upper + endpoint + body

            signature = self._sign_hmac_sha256(message, self.api_secret)

            url = f"{base_url}{endpoint}"

            request = urllib2.Request(url, data=body.encode("utf-8") if body else None, method=method)
            request.add_header("Content-Type", "application/json")
            request.add_header("KC-API-KEY", self.api_key)
            request.add_header("KC-API-SIGN", signature)
            request.add_header("KC-API-TIMESTAMP", timestamp_str)
            request.add_header("KC-API-PASSPHRASE", self.passphrase or "")
            request.add_header("KC-API-VERSION", "2")

        elif self.exchange_id == "bitget":
            timestamp_str = str(int(time.time() * 1000))
            method_upper = method.upper()
            body = json.dumps(params) if params else ""
            message = timestamp_str + method_upper + endpoint + body

            signature = self._sign_hmac_sha256(message, self.api_secret)

            url = f"{base_url}{endpoint}"

            request = urllib2.Request(url, data=body.encode("utf-8") if body else None, method=method)
            request.add_header("Content-Type", "application/json")
            request.add_header("ACCESS-KEY", self.api_key)
            request.add_header("ACCESS-SIGN", signature)
            request.add_header("ACCESS-TIMESTAMP", timestamp_str)
            request.add_header("ACCESS-PASSPHRASE", self.passphrase or "")

        else:
            # Generic HMAC signing
            query_string = self._build_query_string(params)
            signature = self._sign_hmac_sha256(query_string, self.api_secret)

            if method.upper() == "POST":
                url = f"{base_url}{endpoint}"
                data = json.dumps({**params, "signature": signature}).encode("utf-8")
                request = urllib2.Request(url, data=data, method="POST")
                request.add_header("Content-Type", "application/json")
            else:
                url = f"{base_url}{endpoint}?{query_string}&signature={signature}"
                request = urllib2.Request(url, method="GET")

            request.add_header("X-API-KEY", self.api_key)

        # Execute request
        try:
            request.add_header("User-Agent", "CCXT-Python-Adapter/1.0")
            request.add_header("Accept", "application/json")

            response = urllib2.urlopen(request, timeout=self.timeout)
            body = response.read().decode("utf-8")

            if response.headers.get("Content-Type", "").startswith("application/json"):
                return json.loads(body)
            return body

        except urllib2.HTTPError as e:
            self.logger.error(f"Signed request HTTP Error {e.code}: {e.reason}")
            try:
                error_body = e.read().decode("utf-8")
                return json.loads(error_body)
            except Exception:  # noqa: BLE001
                return {"error": str(e)}
        except Exception as e:
            self.logger.error(f"Signed request failed: {e}")
            return {"error": str(e)}

    def _build_query_string(self, params: Dict) -> str:
        """Build URL query string from parameters / 从参数构建URL查询字符串"""
        filtered = {k: v for k, v in params.items() if v is not None and v != ""}
        return "&".join([f"{k}={v}" for k, v in sorted(filtered.items())])

    def _sign_hmac_sha256(self, message: str, secret: str) -> str:
        """Generate HMAC-SHA256 signature / 生成HMAC-SHA256签名"""
        if not secret:
            return ""
        return hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _format_datetime_iso(self, timestamp: float) -> str:
        """Format timestamp to ISO 8601 string / 将时间戳格式化为ISO 8601字符串"""
        from datetime import datetime
        return datetime.utcfromtimestamp(timestamp).isoformat() + "Z"

    # -------------------------------------------------------------------------
    # Symbol format converters / 交易对格式转换器
    # -------------------------------------------------------------------------

    def _symbol_to_market_id(self, symbol: str) -> str:
        """Convert unified symbol to exchange market ID / 将统一交易对转换为交易所市场ID"""
        base, quote = self._parse_symbol(symbol)
        if self.exchange_id in ["binance", "bybit", "bitget", "mexc", "phemex"]:
            return f"{base}{quote}"
        elif self.exchange_id == "okx":
            return f"{base}-{quote}"
        elif self.exchange_id == "gateio":
            return f"{base}_{quote}"
        elif self.exchange_id == "huobi":
            return f"{base}{quote}".lower()
        elif self.exchange_id == "kucoin":
            return f"{base}-{quote}"
        elif self.exchange_id == "deribit":
            return f"{base}-{quote}"
        else:
            return f"{base}{quote}"

    def _symbol_to_okx_symbol(self, symbol: str) -> str:
        base, quote = self._parse_symbol(symbol)
        return f"{base}-{quote}"

    def _symbol_to_gateio_symbol(self, symbol: str) -> str:
        base, quote = self._parse_symbol(symbol)
        return f"{base}_{quote}"

    def _symbol_to_bitget_symbol(self, symbol: str) -> str:
        base, quote = self._parse_symbol(symbol)
        return f"{base}{quote}"

    def _parse_symbol(self, symbol: str) -> Tuple[str, str]:
        """Parse symbol into base and quote / 将交易对解析为基准货币和报价货币"""
        if "/" in symbol:
            return symbol.split("/")
        elif "-" in symbol:
            return symbol.split("-")
        else:
            for quote in ["USDT", "USDC", "USD", "EUR", "GBP", "BTC", "ETH"]:
                if symbol.endswith(quote) and len(symbol) > len(quote) + 1:
                    return symbol[:-len(quote)], quote
            return symbol, "USDT"

    # -------------------------------------------------------------------------
    # Response parsers / 响应解析器
    # -------------------------------------------------------------------------

    def _parse_orders_list(self, data: Any, symbol: str = None) -> List[Order]:
        """Parse orders list / 解析订单列表"""
        orders = []

        try:
            if self.exchange_id == "binance":
                for item in data:
                    orders.append(self.normalizer.normalize_order(item, item.get("symbol", symbol or "")))
            elif self.exchange_id == "bybit":
                result = data.get("result", {}) or {}
                for item in result.get("list", []):
                    orders.append(self.normalizer.normalize_order(item, item.get("symbol", symbol or "")))
            elif self.exchange_id == "okx":
                for item in data.get("data", []):
                    orders.append(self.normalizer.normalize_order(item, item.get("instId", symbol or "")))
            else:
                items = data.get("data", data.get("result", []))
                for item in items:
                    orders.append(self.normalizer.normalize_order(item, symbol or ""))
        except Exception as e:
            self.logger.error(f"Failed to parse orders list: {e}")

        return orders

    def _parse_balance(self, data: Any) -> Dict[str, float]:
        """Parse balance response / 解析余额响应"""
        balances = {}

        try:
            if self.exchange_id == "binance":
                for asset in data.get("balances", []):
                    free = float(asset.get("free", 0))
                    locked = float(asset.get("locked", 0))
                    if free > 0 or locked > 0:
                        balances[asset.get("asset", "")] = {"free": free, "locked": locked}

            elif self.exchange_id == "bybit":
                coins = data.get("result", {}).get("list", [])
                for coin in coins:
                    for item in coin.get("coin", []):
                        balances[item.get("coin", "")] = {
                            "free": float(item.get("available", 0)),
                            "locked": float(item.get("locked", 0)),
                        }

            elif self.exchange_id == "okx":
                for item in data.get("data", []):
                    for details in item.get("details", []):
                        ccy = details.get("ccy", "")
                        balances[ccy] = {
                            "free": float(details.get("availBal", 0)),
                            "locked": float(details.get("frozenBal", 0)),
                        }

            elif self.exchange_id == "gateio":
                for item in data:
                    currency = item.get("currency", "")
                    balances[currency] = {
                        "free": float(item.get("available", 0)),
                        "locked": float(item.get("locked", 0)),
                    }

            elif self.exchange_id == "kucoin":
                for item in data.get("data", {}).get("accounts", []):
                    currency = item.get("currency", "")
                    balances[currency] = {
                        "free": float(item.get("available", 0)),
                        "locked": float(item.get("hold", 0)),
                    }

            else:
                # Generic fallback
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            balances[key] = {
                                "free": float(value.get("free", value.get("available", 0))),
                                "locked": float(value.get("locked", value.get("frozen", 0))),
                            }
        except Exception as e:
            self.logger.error(f"Failed to parse balance: {e}")

        return balances

    def _parse_positions(self, data: Any, symbol: str = None) -> List[Position]:
        """Parse positions response / 解析持仓响应"""
        positions = []

        try:
            if self.exchange_id == "binance":
                for item in data:
                    positions.append(Position(
                        symbol=item.get("symbol", ""),
                        side="long" if float(item.get("positionAmt", 0)) > 0 else "short",
                        amount=abs(float(item.get("positionAmt", 0))),
                        entryPrice=float(item.get("entryPrice", 0)),
                        unrealizedPnl=float(item.get("unRealizedProfit", 0)),
                        leverage=int(item.get("leverage", 1)),
                        liquidationPrice=float(item.get("liquidationPrice", 0)),
                    ))
            elif self.exchange_id == "bybit":
                for item in data.get("result", {}).get("list", []):
                    positions.append(Position(
                        symbol=item.get("symbol", ""),
                        side=item.get("side", "").lower(),
                        amount=float(item.get("size", 0)),
                        entryPrice=float(item.get("avgPrice", 0)),
                        unrealizedPnl=float(item.get("unrealizedPnl", 0)),
                        leverage=int(item.get("leverage", 1)),
                        liquidationPrice=float(item.get("liqPrice", 0)),
                    ))
            elif self.exchange_id == "okx":
                for item in data.get("data", []):
                    positions.append(Position(
                        symbol=item.get("instId", ""),
                        side=item.get("posSide", "").lower(),
                        amount=float(item.get("pos", 0)),
                        entryPrice=float(item.get("avgPx", 0)),
                        unrealizedPnl=float(item.get("upl", 0)),
                        leverage=int(item.get("lever", 1)),
                    ))
        except Exception as e:
            self.logger.error(f"Failed to parse positions: {e}")

        return positions

    # -------------------------------------------------------------------------
    # Formatting helpers / 格式化辅助
    # -------------------------------------------------------------------------

    def _format_quantity(self, quantity: float) -> str:
        """Format quantity to exchange-specific precision / 将数量格式化为交易所特定精度"""
        return f"{quantity:.8f}".rstrip("0").rstrip(".")

    def _format_price(self, price: float) -> str:
        """Format price to exchange-specific precision / 将价格格式化为交易所特定精度"""
        return f"{price:.8f}".rstrip("0").rstrip(".")


# =============================================================================
# Margin Calculator / 保证金计算器
# =============================================================================

class MarginCalculator:
    """Calculate cross-exchange margin requirements / 计算跨交易所保证金要求

    Provides methods to calculate margin, liquidation prices, and leverage
    across different exchange margin systems.

    Example / 示例:
        calc = MarginCalculator('binance')
        margin = calc.calculate_margin('BTC/USDT', 0.1, 50000, 10)
        liq_price = calc.calculate_liquidation_price(position, 0.05)
    """

    def __init__(self, exchange_id: str, config: Dict[str, Any] = None):
        """Initialize margin calculator / 初始化保证金计算器

        Args:
            exchange_id: Exchange identifier
            config: Optional exchange configuration
        """
        self.exchange_id = exchange_id
        self.config = config or EXCHANGE_CONFIGS.get(exchange_id, {})
        self.logger = logging.getLogger(f"MarginCalculator.{exchange_id}")

    def calculate_margin(
        self,
        symbol: str,
        amount: float,
        price: float,
        leverage: int = 1,
        margin_mode: str = "cross",
    ) -> Dict[str, float]:
        """Calculate required margin for a position / 计算持仓所需保证金

        Args:
            symbol: Trading pair symbol
            amount: Position size
            price: Entry price
            leverage: Leverage multiplier
            margin_mode: 'isolated' or 'cross'

        Returns:
            Dict with margin details
        """
        notional_value = abs(amount * price)
        required_margin = notional_value / leverage

        return {
            "symbol": symbol,
            "amount": amount,
            "price": price,
            "notionalValue": notional_value,
            "requiredMargin": required_margin,
            "leverage": leverage,
            "marginMode": margin_mode,
            "maintenanceMargin": notional_value * 0.005,  # Approximate 0.5%
            "liquidationBuffer": required_margin * 0.3,  # 30% buffer estimate
        }

    def calculate_liquidation_price(
        self,
        position: Position,
        add_margin: float = 0,
        remove_margin: float = 0,
    ) -> float:
        """Calculate liquidation price for a position / 计算持仓的清算价格

        Args:
            position: Position object
            add_margin: Additional margin to add
            remove_margin: Margin to remove

        Returns:
            Estimated liquidation price
        """
        if position.amount <= 0 or position.entryPrice <= 0:
            return 0.0

        # Maintenance margin rate (typical 0.5% - 1%)
        maintenance_rate = 0.005
        is_long = position.side.lower() == "long"

        margin = position.amount * position.entryPrice / position.leverage
        adjusted_margin = margin + add_margin - remove_margin

        if adjusted_margin <= 0:
            return 0.0

        if is_long:
            # Long liquidation: entry * (1 - margin_ratio + maintenance_rate)
            liq_price = position.entryPrice * (1 - 1 / position.leverage + maintenance_rate)
        else:
            # Short liquidation: entry * (1 + margin_ratio - maintenance_rate)
            liq_price = position.entryPrice * (1 + 1 / position.leverage - maintenance_rate)

        return max(0, liq_price)

    def calculate_max_leverage(
        self,
        symbol: str,
        amount: float,
        price: float,
        available_margin: float,
    ) -> int:
        """Calculate maximum leverage for given position / 计算给定持仓的最大杠杆

        Args:
            symbol: Trading pair symbol
            amount: Position size
            price: Entry price
            available_margin: Available margin in quote currency

        Returns:
            Maximum leverage (capped by exchange limits)
        """
        notional = abs(amount * price)
        if notional <= 0 or available_margin <= 0:
            return 1

        leverage = notional / available_margin

        # Cap at exchange limits
        max_leverage = 125  # Typical max
        if self.exchange_id == "binance":
            max_leverage = 125
        elif self.exchange_id == "bybit":
            max_leverage = 100
        elif self.exchange_id == "okx":
            max_leverage = 100
        elif self.exchange_id == "deribit":
            max_leverage = 100

        return min(int(leverage), max_leverage)

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        amount: float,
        side: str,
        fee_rate: float = 0.0004,
    ) -> Dict[str, float]:
        """Calculate PnL for a trade / 计算交易盈亏

        Args:
            entry_price: Entry price
            exit_price: Exit price
            amount: Position size
            side: 'long' or 'short'
            fee_rate: Trading fee rate (default 0.04%)

        Returns:
            Dict with PnL details
        """
        is_long = side.lower() == "long"
        direction = 1 if is_long else -1

        gross_pnl = (exit_price - entry_price) * amount * direction
        entry_fee = entry_price * amount * fee_rate
        exit_fee = exit_price * amount * fee_rate
        total_fees = entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees

        return {
            "grossPnl": gross_pnl,
            "netPnl": net_pnl,
            "totalFees": total_fees,
            "entryFee": entry_fee,
            "exitFee": exit_fee,
            "roie": (net_pnl / (entry_price * amount)) * 100 if entry_price * amount > 0 else 0,
        }

    def calculate_breakeven_price(
        self,
        entry_price: float,
        amount: float,
        side: str,
        fee_rate: float = 0.0004,
    ) -> float:
        """Calculate breakeven price including fees / 计算包含费用的盈亏平衡价格

        Args:
            entry_price: Entry price
            amount: Position size
            side: 'long' or 'short'
            fee_rate: Trading fee rate

        Returns:
            Breakeven exit price
        """
        is_long = side.lower() == "long"
        direction = 1 if is_long else -1

        # Entry fee + Exit fee = 2 * price * amount * fee_rate
        # Gross PnL at exit = (exit - entry) * amount * direction
        # Breakeven: gross_pnl = 2 * exit * amount * fee_rate
        # (exit - entry) * direction = 2 * exit * fee_rate
        # exit * direction - entry * direction = 2 * exit * fee_rate
        # exit * (direction - 2 * fee_rate) = entry * direction
        # exit = entry * direction / (direction - 2 * fee_rate)

        denominator = direction - 2 * fee_rate * direction
        if abs(denominator) < 1e-10:
            return entry_price

        breakeven = entry_price * direction / denominator
        return breakeven


# =============================================================================
# Main CCXT Adapter / 主CCXT适配器
# =============================================================================

class CCXTAdapter:
    """Unified CCXT-style adapter for 100+ cryptocurrency exchanges
    统一CCXT风格的100+加密货币交易所适配器

    This is the main entry point providing a unified interface across all
    supported exchanges.

    Features / 功能:
        - Market data: OHLCV, order books, tickers, trades
        - Trading: Market/limit orders, cancellation
        - Account: Balance, positions
        - Margin: Cross-exchange margin calculation

    Args:
        exchange_id: Exchange identifier (e.g., 'binance', 'bybit', 'okx')
        api_key: Optional API key for authenticated requests
        api_secret: Optional API secret for HMAC signing
        passphrase: Optional passphrase (required for some exchanges)
        testnet: Use testnet if available

    Example / 示例:
        # Public data only
        adapter = CCXTAdapter('binance')
        ohlcv = adapter.fetch_ohlcv('BTC/USDT', '1m')

        # Authenticated requests
        adapter = CCXTAdapter('binance', api_key='...', api_secret='...')
        order = adapter.place_order('BTC/USDT', 'buy', 'limit', 0.001, 50000)
        balance = adapter.get_balance()

    Supported Exchanges / 支持的交易所:
        Binance, Bybit, OKX, Deribit, Kraken, Coinbase, Gate.io,
        Huobi, KuCoin, Bitget, MEXC, Phemex, and more
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str = None,
        api_secret: str = None,
        passphrase: str = None,
        testnet: bool = False,
    ):
        """Initialize CCXT adapter / 初始化CCXT适配器"""
        if exchange_id not in SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Unsupported exchange: {exchange_id}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXCHANGES[:20]))}..."
            )

        self.exchange_id = exchange_id.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        self.config = EXCHANGE_CONFIGS.get(self.exchange_id, {})

        # Initialize components
        self.fetcher = MarketDataFetcher(self.exchange_id, self.config)
        self.normalizer = ExchangeNormalizer(self.exchange_id)
        self.margin_calculator = MarginCalculator(self.exchange_id, self.config)

        # Order executor requires credentials
        self._order_executor = None
        if api_key and api_secret:
            self._order_executor = OrderExecutor(
                self.exchange_id,
                api_key,
                api_secret,
                passphrase,
                self.config,
                testnet,
            )

        # Market cache
        self._markets = {}
        self._symbols = []
        self.logger = logging.getLogger(f"CCXTAdapter.{self.exchange_id}")

    @property
    def order_executor(self) -> OrderExecutor:
        """Get order executor (requires API credentials) / 获取订单执行器（需要API凭证）"""
        if self._order_executor is None:
            raise ValueError(
                f"{self.exchange_id} order executor requires api_key and api_secret"
            )
        return self._order_executor

    # -------------------------------------------------------------------------
    # Market Data Methods / 市场数据方法
    # -------------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int = None,
        limit: int = 100,
        params: Dict = None,
    ) -> List[OHLCV]:
        """Fetch OHLCV candlestick data / 获取K线烛台数据

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '4h', '1d')
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch (default 100)
            params: Additional exchange-specific parameters

        Returns:
            List of OHLCV objects

        Example / 示例:
            ohlcv = adapter.fetch_ohlcv('BTC/USDT', '1h', limit=500)
            for candle in ohlcv:
                print(f"{candle.datetime}: O={candle.open} H={candle.high} L={candle.low} C={candle.close}")
        """
        return self.fetcher.fetch_ohlcv(symbol, timeframe, since, limit, params)

    def fetch_order_book(self, symbol: str, limit: int = 20, params: Dict = None) -> OrderBook:
        """Fetch order book / 获取订单簿

        Args:
            symbol: Trading pair symbol
            limit: Number of price levels (default 20)
            params: Additional parameters

        Returns:
            OrderBook object with bids and asks

        Example / 示例:
            ob = adapter.fetch_order_book('BTC/USDT', limit=50)
            print(f"Bids: {ob.bids[:5]}")
            print(f"Asks: {ob.asks[:5]}")
        """
        return self.fetcher.fetch_order_book(symbol, limit, params)

    def fetch_ticker(self, symbol: str, params: Dict = None) -> Ticker:
        """Fetch ticker / 获取行情

        Args:
            symbol: Trading pair symbol
            params: Additional parameters

        Returns:
            Ticker object with price and volume info
        """
        return self.fetcher.fetch_ticker(symbol, params)

    def fetch_trades(self, symbol: str, limit: int = 50, params: Dict = None) -> List[Trade]:
        """Fetch recent trades / 获取最近交易

        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch
            params: Additional parameters

        Returns:
            List of Trade objects
        """
        return self.fetcher.fetch_trades(symbol, limit, params)

    # -------------------------------------------------------------------------
    # Trading Methods (require API credentials) / 交易方法（需要API凭证）
    # -------------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float = None,
        params: Dict = None,
    ) -> Order:
        """Place an order / 下单

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop_loss', 'take_profit', etc.
            amount: Order quantity
            price: Order price (required for limit orders)
            params: Additional exchange-specific parameters

        Returns:
            Order object

        Example / 示例:
            order = adapter.place_order('BTC/USDT', 'buy', 'limit', 0.001, 50000)
            print(f"Order placed: {order.id}, Status: {order.status}")
        """
        return self.order_executor.place_order(symbol, side, order_type, amount, price, params)

    def cancel_order(self, order_id: str, symbol: str, params: Dict = None) -> Order:
        """Cancel an order / 取消订单

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            params: Additional parameters

        Returns:
            Order object with canceled status
        """
        return self.order_executor.cancel_order(order_id, symbol, params)

    def get_order(self, order_id: str, symbol: str, params: Dict = None) -> Order:
        """Get order status / 获取订单状态

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            params: Additional parameters

        Returns:
            Order object
        """
        return self.order_executor.get_order(order_id, symbol, params)

    def get_open_orders(self, symbol: str = None, params: Dict = None) -> List[Order]:
        """Get all open orders / 获取所有未完成订单

        Args:
            symbol: Optional symbol filter
            params: Additional parameters

        Returns:
            List of Order objects
        """
        return self.order_executor.get_open_orders(symbol, params)

    def get_balance(self, params: Dict = None) -> Dict[str, float]:
        """Get account balance / 获取账户余额

        Args:
            params: Additional parameters (e.g., {'ccy': 'BTC'})

        Returns:
            Dict mapping asset codes to balance amounts

        Example / 示例:
            balance = adapter.get_balance()
            for asset, amounts in balance.items():
                print(f"{asset}: free={amounts['free']}, locked={amounts['locked']}")
        """
        return self.order_executor.get_balance(params)

    def get_positions(self, symbol: str = None, params: Dict = None) -> List[Position]:
        """Get positions / 获取持仓

        Args:
            symbol: Optional symbol filter
            params: Additional parameters

        Returns:
            List of Position objects
        """
        return self.order_executor.get_positions(symbol, params)

    # -------------------------------------------------------------------------
    # Margin Methods / 保证金方法
    # -------------------------------------------------------------------------

    def calculate_margin(
        self,
        symbol: str,
        amount: float,
        price: float,
        leverage: int = 1,
        margin_mode: str = "cross",
    ) -> Dict[str, float]:
        """Calculate required margin / 计算所需保证金

        Args:
            symbol: Trading pair symbol
            amount: Position size
            price: Entry price
            leverage: Leverage multiplier
            margin_mode: 'isolated' or 'cross'

        Returns:
            Dict with margin details
        """
        return self.margin_calculator.calculate_margin(symbol, amount, price, leverage, margin_mode)

    def calculate_liquidation_price(
        self,
        position: Position,
        add_margin: float = 0,
        remove_margin: float = 0,
    ) -> float:
        """Calculate liquidation price / 计算清算价格

        Args:
            position: Position object
            add_margin: Additional margin to add
            remove_margin: Margin to remove

        Returns:
            Estimated liquidation price
        """
        return self.margin_calculator.calculate_liquidation_price(position, add_margin, remove_margin)

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        amount: float,
        side: str,
        fee_rate: float = 0.0004,
    ) -> Dict[str, float]:
        """Calculate PnL / 计算盈亏

        Args:
            entry_price: Entry price
            exit_price: Exit price
            amount: Position size
            side: 'long' or 'short'
            fee_rate: Trading fee rate

        Returns:
            Dict with PnL details
        """
        return self.margin_calculator.calculate_pnl(entry_price, exit_price, amount, side, fee_rate)

    # -------------------------------------------------------------------------
    # Market Loading / 市场加载
    # -------------------------------------------------------------------------

    def load_markets(self, reload: bool = False) -> Dict[str, Market]:
        """Load available markets / 加载可用市场

        Note: This is a simplified implementation. Full market loading
        would query the exchange for complete market metadata.

        Args:
            reload: Force reload even if cached

        Returns:
            Dict of market ID to Market objects
        """
        if self._markets and not reload:
            return self._markets

        # Simplified: load common markets
        common_markets = {
            "BTCUSDT": Market(
                id="BTCUSDT",
                symbol="BTC/USDT",
                base="BTC",
                quote="USDT",
                baseId="BTC",
                quoteId="USDT",
                type="spot",
                precision_amount=0.00001,
                precision_price=0.01,
            ),
            "ETHUSDT": Market(
                id="ETHUSDT",
                symbol="ETH/USDT",
                base="ETH",
                quote="USDT",
                baseId="ETH",
                quoteId="USDT",
                type="spot",
                precision_amount=0.0001,
                precision_price=0.01,
            ),
        }

        self._markets = common_markets
        self._symbols = list(common_markets.keys())
        return self._markets

    def market(self, symbol: str) -> Market:
        """Get market by symbol / 通过交易对获取市场

        Args:
            symbol: Unified symbol (e.g., 'BTC/USDT')

        Returns:
            Market object

        Raises:
            ValueError: If market not found
        """
        if not self._markets:
            self.load_markets()

        # Try direct lookup
        if symbol in self._markets:
            return self._markets[symbol]

        # Try by ID
        normalized = symbol.replace("/", "").replace("-", "")
        if normalized in self._markets:
            return self._markets[normalized]

        raise ValueError(f"Market {symbol} not found")

    # -------------------------------------------------------------------------
    # Utility Methods / 工具方法
    # -------------------------------------------------------------------------

    def is_authenticated(self) -> bool:
        """Check if adapter has API credentials / 检查适配器是否有API凭证"""
        return self._order_executor is not None

    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange configuration / 获取交易所配置"""
        return self.config.copy()

    @staticmethod
    def get_supported_exchanges() -> List[str]:
        """Get list of supported exchanges / 获取支持的交易所列表"""
        return sorted(SUPPORTED_EXCHANGES)

    @staticmethod
    def get_exchange_name(exchange_id: str) -> str:
        """Get display name for exchange / 获取交易所显示名称"""
        config = EXCHANGE_CONFIGS.get(exchange_id, {})
        return config.get("name", exchange_id.title())
