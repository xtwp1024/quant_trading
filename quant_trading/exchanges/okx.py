"""OKX exchange adapter"""
import ccxt.async_support as ccxt
from typing import Any, Dict, Optional


class OKXAdapter:
    """OKX exchange adapter using ccxt"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("apiKey", "")
        self.api_secret = config.get("secret", "")
        self.passphrase = config.get("password", "")
        self.testnet = config.get("testnet", False)
        self.exchange = None

    async def connect(self) -> None:
        """Connect to OKX"""
        self.exchange = ccxt.okx({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "password": self.passphrase,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if self.testnet:
            self.exchange.set_sandbox_mode(True)
        await self.exchange.load_markets()

    async def close(self) -> None:
        """Close connection"""
        if self.exchange:
            await self.exchange.close()

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        """Fetch OHLCV data"""
        return await self.exchange.fetch_ohlcv(symbol, timeframe, limit)

    async def fetch_order_book(self, symbol: str, limit: int = 20):
        """Fetch order book"""
        return await self.exchange.fetch_order_book(symbol, limit)

    async def fetch_ticker(self, symbol: str):
        """Fetch ticker"""
        return await self.exchange.fetch_ticker(symbol)

    async def fetch_balance(self):
        """Fetch balance"""
        return await self.exchange.fetch_balance()

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: Optional[float] = None):
        """Create order"""
        return await self.exchange.create_order(symbol, type, side, amount, price)

    async def cancel_order(self, order_id: str, symbol: str):
        """Cancel order"""
        return await self.exchange.cancel_order(order_id, symbol)

    async def fetch_funding_rate(self, symbol: str):
        """Fetch funding rate"""
        return await self.exchange.fetch_funding_rate(symbol)

    async def fetch_positions(self, symbol: str = None):
        """Fetch positions"""
        return await self.exchange.fetch_positions(symbol)
