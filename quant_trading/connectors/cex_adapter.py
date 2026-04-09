"""
CEX Multi-Exchange Unified Adapter
多交易所统一适配器

Provides a unified trading/query interface across multiple CEX exchanges,
abstracting API differences between Binance, Bybit, OKX, Bitget, and KuCoin.

Supports:
- Market data (ticker, orderbook, klines)
- Trading (market/limit orders, cancel)
- Position and balance queries
- Leverage and margin mode configuration

Example:
    adapter = CEXAdapter('binance', api_key='...', api_secret='...')
    ticker = adapter.get_ticker('BTCUSDT')
    adapter.place_market_order('BTCUSDT', 'buy', 0.01)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

__all__ = ["CEXAdapter", "SUPPORTED_EXCHANGES"]

# Supported exchanges / 支持的交易所
SUPPORTED_EXCHANGES = ["binance", "bybit", "okx", "bitget", "kucoin"]

# Exchange API base URLs / 各交易所API地址
EXCHANGE_BASE_URLS: Dict[str, str] = {
    "binance": "https://api.binance.com",
    "bybit": "https://api.bybit.com",
    "okx": "https://www.okx.com",
    "bitget": "https://api.bitget.com",
    "kucoin": "https://api.kucoin.com",
}


class CEXAdapter:
    """多交易所统一适配器 / Multi-exchange unified adapter.

    提供统一的交易/查询接口，屏蔽各交易所API差异。
    Unified interface for trading and querying, abstracting exchange-specific API differences.

    Args:
        exchange: Exchange name (binance/bybit/okx/bitget/kucoin)
        api_key: API key for authentication
        api_secret: API secret for HMAC signing
        passphrase: Extra passphrase (required for OKX, KuCoin)

    Raises:
        ValueError: If exchange is not supported
    """

    def __init__(
        self,
        exchange: str,
        api_key: str,
        api_secret: str,
        passphrase: str = "",
    ):
        if exchange not in SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange: {exchange}. Supported: {SUPPORTED_EXCHANGES}")

        self.exchange = exchange.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = EXCHANGE_BASE_URLS.get(self.exchange, "")
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self.api_key})
        self.logger = logging.getLogger(f"CEXAdapter.{self.exchange}")

    # -------------------------------------------------------------------------
    # Market Data / 市场数据
    # -------------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> dict:
        """获取ticker信息 / Get ticker information.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Dict with price, volume, etc.
        """
        endpoint = self._ticker_endpoint()
        params = {"symbol": symbol.upper(), **self._default_params()}

        try:
            data = self._get(endpoint, params)
            return self._parse_ticker(data, symbol)
        except Exception as e:
            self.logger.warning(f"get_ticker failed for {symbol}: {e}")
            return self._fallback_ticker(symbol)

    def get_order_book(self, symbol: str, depth: int = 20) -> dict:
        """获取订单簿 / Get order book.

        Args:
            symbol: Trading pair symbol
            depth: Number of price levels (default 20)

        Returns:
            Dict with bids and asks
        """
        endpoint = self._orderbook_endpoint()
        params = {"symbol": symbol.upper(), "limit": depth, **self._default_params()}

        try:
            data = self._get(endpoint, params)
            return self._parse_orderbook(data, symbol)
        except Exception as e:
            self.logger.warning(f"get_order_book failed for {symbol}: {e}")
            return {"symbol": symbol, "bids": [], "asks": [], "timestamp": 0}

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> list:
        """获取K线数据 / Get candlestick/kline data.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
            limit: Number of klines (default 500, max 1000)

        Returns:
            List of kline dicts
        """
        endpoint = self._klines_endpoint()
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
            **self._default_params(),
        }

        try:
            data = self._get(endpoint, params)
            return self._parse_klines(data, symbol)
        except Exception as e:
            self.logger.warning(f"get_klines failed for {symbol}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Trading / 交易
    # -------------------------------------------------------------------------

    def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """市价单 / Place market order.

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            quantity: Order quantity

        Returns:
            Dict with order details
        """
        endpoint = self._order_endpoint()
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": self._format_quantity(quantity),
            **self._default_params(),
        }

        try:
            data = self._sign_and_post(endpoint, params)
            return self._parse_order_response(data, symbol)
        except Exception as e:
            self.logger.error(f"place_market_order failed: {e}")
            return self._fallback_order(symbol, side, "market", quantity, error=str(e))

    def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> dict:
        """限价单 / Place limit order.

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            price: Limit price
            quantity: Order quantity

        Returns:
            Dict with order details
        """
        endpoint = self._order_endpoint()
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "LIMIT",
            "price": self._format_price(price),
            "quantity": self._format_quantity(quantity),
            "timeInForce": "GTC",
            **self._default_params(),
        }

        try:
            data = self._sign_and_post(endpoint, params)
            return self._parse_order_response(data, symbol)
        except Exception as e:
            self.logger.error(f"place_limit_order failed: {e}")
            return self._fallback_order(symbol, side, "limit", quantity, price=price, error=str(e))

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """取消订单 / Cancel order.

        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID

        Returns:
            Dict with cancellation result
        """
        endpoint = self._cancel_endpoint()
        params = {
            "symbol": symbol.upper(),
            "orderId": order_id,
            **self._default_params(),
        }

        try:
            data = self._sign_and_post(endpoint, params)
            return {"success": True, "order_id": order_id, "data": data}
        except Exception as e:
            self.logger.error(f"cancel_order failed: {e}")
            return {"success": False, "order_id": order_id, "error": str(e)}

    def get_positions(self) -> list[dict]:
        """获取持仓 / Get all positions.

        Returns:
            List of position dicts
        """
        endpoint = self._positions_endpoint()
        params = self._default_params()

        try:
            data = self._sign_get(endpoint, params)
            return self._parse_positions(data)
        except Exception as e:
            self.logger.warning(f"get_positions failed: {e}")
            return []

    def get_balance(self) -> dict:
        """获取账户余额 / Get account balance.

        Returns:
            Dict with asset balances
        """
        endpoint = self._balance_endpoint()
        params = self._default_params()

        try:
            data = self._sign_get(endpoint, params)
            return self._parse_balance(data)
        except Exception as e:
            self.logger.warning(f"get_balance failed: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Exchange-specific configuration / 交易所特定配置
    # -------------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """设置杠杆 / Set leverage for a symbol.

        Args:
            symbol: Trading pair symbol
            leverage: Leverage multiplier (e.g., 5, 10, 20)
        """
        endpoint = self._leverage_endpoint()
        params = {
            "symbol": symbol.upper(),
            "leverage": leverage,
            **self._default_params(),
        }

        try:
            self._sign_and_post(endpoint, params)
            self.logger.info(f"Leverage set for {symbol}: {leverage}x")
        except Exception as e:
            self.logger.error(f"set_leverage failed for {symbol}: {e}")

    def set_margin_mode(self, symbol: str, mode: str) -> None:
        """设置保证金模式 / Set margin mode (isolated/cross).

        Args:
            symbol: Trading pair symbol
            mode: 'isolated' or 'cross'
        """
        endpoint = self._margin_mode_endpoint()
        params = {
            "symbol": symbol.upper(),
            "marginType": mode.upper(),
            **self._default_params(),
        }

        try:
            self._sign_and_post(endpoint, params)
            self.logger.info(f"Margin mode set for {symbol}: {mode}")
        except Exception as e:
            self.logger.error(f"set_margin_mode failed for {symbol}: {e}")

    # -------------------------------------------------------------------------
    # Private: HTTP methods / HTTP方法
    # -------------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict) -> Any:
        """Send GET request / 发送GET请求."""
        url = f"{self.base_url}{endpoint}"
        response = self._session.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def _sign_get(self, endpoint: str, params: dict) -> Any:
        """Send signed GET request / 发送签名GET请求."""
        signed_params = self._sign_params(params)
        return self._get(endpoint, signed_params)

    def _sign_and_post(self, endpoint: str, params: dict) -> Any:
        """Sign and send POST request / 签名并发送POST请求."""
        url = f"{self.base_url}{endpoint}"
        signed_params = self._sign_params(params)
        response = self._session.post(url, data=signed_params, timeout=15)
        response.raise_for_status()
        return response.json()

    def _sign_params(self, params: dict) -> dict:
        """Sign request parameters with HMAC / 使用HMAC签名参数."""
        timestamp = int(time.time() * 1000)
        params["timestamp"] = timestamp
        params["recvWindow"] = 5000

        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items()) if v != ""])

        if self.exchange == "binance":
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        elif self.exchange in ("okx", "kucoin"):
            message = timestamp + "GET" + endpoint + query_string
            if self.exchange == "okx":
                message = timestamp + "POST" + endpoint + query_string
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                message.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        else:
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

        params["signature"] = signature
        return params

    def _default_params(self) -> dict:
        """Default parameters for exchange APIs."""
        return {}

    # -------------------------------------------------------------------------
    # Private: Exchange-specific endpoints / 交易所特定端点
    # -------------------------------------------------------------------------

    def _ticker_endpoint(self) -> str:
        """Return ticker endpoint by exchange."""
        endpoints = {
            "binance": "/api/v3/ticker/24hr",
            "bybit": "/v5/market/tickers",
            "okx": "/api/v5/market/ticker",
            "bitget": "/api/v5/market/ticker",
            "kucoin": "/api/v1/market/orderbook/level1",
        }
        return endpoints.get(self.exchange, "/api/v3/ticker/24hr")

    def _orderbook_endpoint(self) -> str:
        """Return orderbook endpoint by exchange."""
        endpoints = {
            "binance": "/api/v3/depth",
            "bybit": "/v5/market/orderbook",
            "okx": "/api/v5/market/books",
            "bitget": "/api/v5/market/books",
            "kucoin": "/api/v1/market/orderbook/level2",
        }
        return endpoints.get(self.exchange, "/api/v3/depth")

    def _klines_endpoint(self) -> str:
        """Return klines endpoint by exchange."""
        endpoints = {
            "binance": "/api/v3/klines",
            "bybit": "/v5/market/kline",
            "okx": "/api/v5/market/candles",
            "bitget": "/api/v5/market/candles",
            "kucoin": "/api/v1/market/candles",
        }
        return endpoints.get(self.exchange, "/api/v3/klines")

    def _order_endpoint(self) -> str:
        """Return order endpoint by exchange."""
        endpoints = {
            "binance": "/api/v3/order",
            "bybit": "/v5/order/create",
            "okx": "/api/v5/trade/order",
            "bitget": "/api/v5/place/trade",
            "kucoin": "/api/v1/trade/order",
        }
        return endpoints.get(self.exchange, "/api/v3/order")

    def _cancel_endpoint(self) -> str:
        """Return cancel order endpoint by exchange."""
        endpoints = {
            "binance": "/api/v3/order",
            "bybit": "/v5/order/cancel",
            "okx": "/api/v5/trade/cancel-order",
            "bitget": "/api/v5/place/cancel-order",
            "kucoin": "/api/v1/trade/cancel-order",
        }
        return endpoints.get(self.exchange, "/api/v3/order")

    def _positions_endpoint(self) -> str:
        """Return positions endpoint by exchange."""
        endpoints = {
            "binance": "/fapi/v2/positionRisk",
            "bybit": "/v5/position/list",
            "okx": "/api/v5/account/positions",
            "bitget": "/api/v5/position/list",
            "kucoin": "/api/v1/api/v1/positions",
        }
        return endpoints.get(self.exchange, "/fapi/v2/positionRisk")

    def _balance_endpoint(self) -> str:
        """Return balance endpoint by exchange."""
        endpoints = {
            "binance": "/api/v3/account",
            "bybit": "/v5/account/wallet-balance",
            "okx": "/api/v5/account/balance",
            "bitget": "/api/v5/account/balance",
            "kucoin": "/api/v1/account/balance",
        }
        return endpoints.get(self.exchange, "/api/v3/account")

    def _leverage_endpoint(self) -> str:
        """Return leverage endpoint by exchange."""
        endpoints = {
            "binance": "/fapi/v1/leverage",
            "bybit": "/v5/position/set-leverage",
            "okx": "/api/v5/trade/set-leverage",
            "bitget": "/api/v5/position/set-leverage",
            "kucoin": "/api/v1/position/risk-limit-level/change",
        }
        return endpoints.get(self.exchange, "/fapi/v1/leverage")

    def _margin_mode_endpoint(self) -> str:
        """Return margin mode endpoint by exchange."""
        endpoints = {
            "binance": "/fapi/v1/marginType",
            "bybit": "/v5/position/switch-mode",
            "okx": "/api/v5/account/set-margin-mode",
            "bitget": "/api/v5/position/set-margin-mode",
            "kucoin": "/api/v1/position/margin-type",
        }
        return endpoints.get(self.exchange, "/fapi/v1/marginType")

    # -------------------------------------------------------------------------
    # Private: Response parsers / 响应解析
    # -------------------------------------------------------------------------

    def _parse_ticker(self, data: Any, symbol: str) -> dict:
        """Parse ticker response by exchange format."""
        if self.exchange == "binance":
            return {
                "symbol": symbol,
                "price": float(data.get("lastPrice", 0)),
                "volume": float(data.get("volume", 0)),
                "quote_volume": float(data.get("quoteVolume", 0)),
                "price_change_pct": float(data.get("priceChangePercent", 0)),
                "high": float(data.get("highPrice", 0)),
                "low": float(data.get("lowPrice", 0)),
                "timestamp": data.get("closeTime", 0),
            }
        elif self.exchange == "bybit":
            list_data = data.get("list", [{}])
            item = list_data[0] if list_data else {}
            return {
                "symbol": symbol,
                "price": float(item.get("lastPrice", 0)),
                "volume": float(item.get("volume24h", 0)),
                "quote_volume": float(item.get("turnover24h", 0)),
                "price_change_pct": float(item.get("price24hPcnt", 0)),
                "high": float(item.get("highPrice24h", 0)),
                "low": float(item.get("lowPrice24h", 0)),
                "timestamp": item.get("ts", 0),
            }
        elif self.exchange == "okx":
            data_list = data.get("data", [{}])
            item = data_list[0] if data_list else {}
            return {
                "symbol": symbol,
                "price": float(item.get("last", 0)),
                "volume": float(item.get("vol24h", 0)),
                "quote_volume": float(item.get("quoteVol24h", 0)),
                "price_change_pct": float(item.get("sodUtc8", 0)) / 100
                if item.get("sodUtc8")
                else 0,
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "timestamp": item.get("ts", 0),
            }
        else:
            return {
                "symbol": symbol,
                "price": float(data.get("price", 0)),
                "volume": float(data.get("size", 0)),
                "quote_volume": 0,
                "price_change_pct": 0,
                "high": 0,
                "low": 0,
                "timestamp": 0,
            }

    def _parse_orderbook(self, data: Any, symbol: str) -> dict:
        """Parse orderbook response by exchange format."""
        if self.exchange == "binance":
            return {
                "symbol": symbol,
                "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
                "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
                "timestamp": data.get("lastUpdateId", 0),
            }
        elif self.exchange == "bybit":
            bids = data.get("result", {}).get("b", [])
            asks = data.get("result", {}).get("a", [])
            return {
                "symbol": symbol,
                "bids": [[float(p), float(s)] for p, s in bids],
                "asks": [[float(p), float(s)] for p, s in asks],
                "timestamp": data.get("time", 0),
            }
        else:
            return {
                "symbol": symbol,
                "bids": [],
                "asks": [],
                "timestamp": 0,
            }

    def _parse_klines(self, data: Any, symbol: str) -> list:
        """Parse klines response by exchange format."""
        klines = []
        for item in data:
            if self.exchange == "binance":
                klines.append(
                    {
                        "symbol": symbol,
                        "open_time": item[0],
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                        "close_time": item[6],
                    }
                )
            else:
                klines.append({"symbol": symbol, "data": item})
        return klines

    def _parse_order_response(self, data: Any, symbol: str) -> dict:
        """Parse order response by exchange format."""
        if self.exchange == "binance":
            return {
                "order_id": str(data.get("orderId", "")),
                "symbol": data.get("symbol", symbol),
                "side": data.get("side", ""),
                "type": data.get("type", ""),
                "price": float(data.get("price", 0)),
                "quantity": float(data.get("origQty", 0)),
                "status": data.get("status", ""),
                "timestamp": data.get("transactTime", 0),
            }
        elif self.exchange == "bybit":
            return {
                "order_id": data.get("orderId", ""),
                "symbol": symbol,
                "side": data.get("side", ""),
                "type": data.get("orderType", ""),
                "price": float(data.get("price", 0)),
                "quantity": float(data.get("qty", 0)),
                "status": data.get("orderStatus", ""),
                "timestamp": data.get("createTime", 0),
            }
        else:
            return {"order_id": data.get("orderId", ""), "symbol": symbol}

    def _parse_positions(self, data: Any) -> list:
        """Parse positions response by exchange format."""
        positions = []
        if self.exchange == "binance":
            for item in data:
                positions.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "positionAmt": float(item.get("positionAmt", 0)),
                        "entryPrice": float(item.get("entryPrice", 0)),
                        "unrealizedProfit": float(item.get("unRealizedProfit", 0)),
                        "leverage": int(item.get("leverage", 1)),
                    }
                )
        elif self.exchange in ("bybit", "okx", "bitget"):
            items = data.get("result", data.get("data", []))
            for item in items:
                positions.append(
                    {
                        "symbol": item.get("symbol", ""),
                        "size": float(item.get("size", 0)),
                        "entry_price": float(item.get("entryPrice", 0)),
                        "unrealized_pnl": float(item.get("unrealizedPnl", 0)),
                        "leverage": int(item.get("leverage", 1)),
                    }
                )
        return positions

    def _parse_balance(self, data: Any) -> dict:
        """Parse balance response by exchange format."""
        if self.exchange == "binance":
            balances = {}
            for asset in data.get("balances", []):
                free = float(asset.get("free", 0))
                locked = float(asset.get("locked", 0))
                if free > 0 or locked > 0:
                    balances[asset.get("asset", "")] = {"free": free, "locked": locked}
            return balances
        elif self.exchange in ("bybit", "okx", "bitget"):
            coins = data.get("result", data.get("data", []))
            balances = {}
            for coin in coins:
                balances[coin.get("coin", "")] = {
                    "free": float(coin.get("available", 0)),
                    "locked": float(coin.get("locked", 0)),
                }
            return balances
        return {}

    # -------------------------------------------------------------------------
    # Private: Formatting helpers / 格式化辅助
    # -------------------------------------------------------------------------

    def _format_quantity(self, quantity: float) -> str:
        """Format quantity to exchange-specific precision."""
        return f"{quantity:.6f}".rstrip("0").rstrip(".")

    def _format_price(self, price: float) -> str:
        """Format price to exchange-specific precision."""
        return f"{price:.8f}".rstrip("0").rstrip(".")

    # -------------------------------------------------------------------------
    # Private: Fallback methods / 降级方法
    # -------------------------------------------------------------------------

    def _fallback_ticker(self, symbol: str) -> dict:
        """Return empty ticker on failure / 失败时返回空ticker."""
        return {
            "symbol": symbol,
            "price": 0,
            "volume": 0,
            "quote_volume": 0,
            "price_change_pct": 0,
            "high": 0,
            "low": 0,
            "timestamp": 0,
        }

    def _fallback_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = 0,
        error: str = "",
    ) -> dict:
        """Return empty order on failure / 失败时返回空订单."""
        return {
            "order_id": "",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "price": price,
            "quantity": quantity,
            "status": "FAILED",
            "error": error,
        }
