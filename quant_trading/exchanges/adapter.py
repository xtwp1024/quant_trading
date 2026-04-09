"""Exchange adapter base class"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import ccxt.async_support as ccxt
import asyncio
import logging

logger = logging.getLogger(__name__)


class ExchangeAdapter(ABC):
    """Abstract base class for exchange adapters"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.name = self.__class__.__name__

    async def connect(self) -> None:
        """Connect to exchange"""
        if self.exchange is None:
            raise NotImplementedError("Subclass must set self.exchange")

        await self.exchange.load_markets()
        logger.info(f"Connected to {self.name}")

    async def close(self) -> None:
        """Close connection"""
        if self.exchange:
            await self.exchange.close()
            logger.info(f"Closed {self.name}")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> List:
        """Fetch OHLCV data"""
        return await self.exchange.fetch_ohlcv(symbol, timeframe, limit)

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Fetch order book"""
        return await self.exchange.fetch_order_book(symbol, limit)

    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker"""
        return await self.exchange.fetch_ticker(symbol)

    async def fetch_balance(self) -> Dict:
        """Fetch account balance"""
        return await self.exchange.fetch_balance()

    async def create_order(
        self, symbol: str, type: str, side: str, amount: float, price: Optional[float] = None
    ) -> Dict:
        """Create order"""
        return await self.exchange.create_order(symbol, type, side, amount, price)

    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel order"""
        return await self.exchange.cancel_order(order_id, symbol)


def get_exchange(name: str, config: Dict[str, Any]) -> ExchangeAdapter:
    """Factory function to get exchange adapter"""
    adapters = {
        "binance": BinanceAdapter,
        "okx": OKXAdapter,
        "bybit": BybitAdapter,
    }

    if name.lower() not in adapters:
        raise ValueError(f"Unknown exchange: {name}")

    return adapters[name.lower()](config)


# Import subclasses here to avoid circular imports
from .binance import BinanceAdapter
from .okx import OKXAdapter

try:
    from .bybit import BybitAdapter
except ImportError:
    pass
