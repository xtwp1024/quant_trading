"""
Base Connector Abstract Class

This module provides an abstract base class for exchange connectors, inspired by
the Hummingbot connector architecture. It defines the interface that all
exchange connectors must implement.

Key Features:
- Unified interface for multiple exchanges
- Order lifecycle management
- Balance tracking with real-time updates
- Order quantization for exchange requirements
- Event-driven architecture
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional

# Constants
s_decimal_0 = Decimal("0")
s_decimal_NaN = Decimal("NaN")


class BaseConnector(ABC):
    """
    Abstract base class for all exchange connectors.

    This class provides a standardized interface for interacting with
    cryptocurrency exchanges. Subclasses must implement the abstract
    methods for exchange-specific operations.

    Key Properties:
        name: Connector identifier
        trading_pairs: List of supported trading pairs
        ready: Whether connector is initialized and ready

    Key Methods:
        buy/sell: Place orders
        cancel: Cancel an order
        get_balance: Get asset balance
        quantize_order_price/amount: Adjust orders to exchange requirements
    """

    def __init__(self):
        self._account_balances: Dict[str, Decimal] = {}
        self._account_available_balances: Dict[str, Decimal] = {}
        self._real_time_balance_update: bool = True
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def name(self) -> str:
        """Return the connector name."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def trading_pairs(self) -> List[str]:
        """Return list of supported trading pairs."""
        pass

    @property
    def ready(self) -> bool:
        """Return whether connector is ready to trade."""
        raise NotImplementedError

    @property
    def real_time_balance_update(self) -> bool:
        return self._real_time_balance_update

    @real_time_balance_update.setter
    def real_time_balance_update(self, value: bool):
        self._real_time_balance_update = value

    # ===================
    # Balance Methods
    # ===================

    def get_balance(self, currency: str) -> Decimal:
        """
        Get total balance for a currency.

        Args:
            currency: Token symbol (e.g., "BTC", "USDT")

        Returns:
            Total balance as Decimal
        """
        return self._account_balances.get(currency, s_decimal_0)

    def get_available_balance(self, currency: str) -> Decimal:
        """
        Get available balance for trading.

        This accounts for in-flight orders (locked balance).

        Args:
            currency: Token symbol

        Returns:
            Available balance as Decimal
        """
        return self._account_available_balances.get(currency, s_decimal_0)

    # ===================
    # Order Methods
    # ===================

    @property
    def in_flight_orders(self) -> Dict[str, "InFlightOrderBase"]:
        """Return dictionary of in-flight orders."""
        raise NotImplementedError

    @abstractmethod
    async def place_order(
        self,
        trading_pair: str,
        side: "TradeType",
        amount: Decimal,
        order_type: "OrderType",
        price: Optional[Decimal] = None,
        **kwargs,
    ) -> str:
        """
        Place an order on the exchange.

        Args:
            trading_pair: Trading pair (e.g., "BTC-USDT")
            side: BUY or SELL
            amount: Order amount in base currency
            order_type: MARKET, LIMIT, etc.
            price: Order price (required for LIMIT orders)

        Returns:
            Client order ID
        """
        pass

    @abstractmethod
    async def cancel_order(self, trading_pair: str, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            trading_pair: Trading pair
            order_id: Client order ID

        Returns:
            True if cancellation was successful
        """
        pass

    # ===================
    # Quantization Methods
    # ===================

    @abstractmethod
    def get_order_price_quantum(self, trading_pair: str, price: Decimal) -> Decimal:
        """
        Get minimum price increment for a trading pair.

        Args:
            trading_pair: Trading pair
            price: Price to check

        Returns:
            Minimum price step
        """
        pass

    @abstractmethod
    def get_order_size_quantum(self, trading_pair: str, amount: Decimal) -> Decimal:
        """
        Get minimum size increment for a trading pair.

        Args:
            trading_pair: Trading pair
            amount: Amount to check

        Returns:
            Minimum size step
        """
        pass

    def quantize_order_price(
        self, trading_pair: str, price: Decimal
    ) -> Decimal:
        """
        Quantize order price to exchange requirements.

        Rounds down to the nearest valid price increment.

        Args:
            trading_pair: Trading pair
            price: Original price

        Returns:
            Quantized price
        """
        if price.is_nan():
            return price

        price_quantum = self.get_order_price_quantum(trading_pair, price)
        if price_quantum == 0:
            return price
        return (price // price_quantum) * price_quantum

    def quantize_order_amount(
        self, trading_pair: str, amount: Decimal
    ) -> Decimal:
        """
        Quantize order amount to exchange requirements.

        Rounds down to the nearest valid size increment.

        Args:
            trading_pair: Trading pair
            amount: Original amount

        Returns:
            Quantized amount
        """
        size_quantum = self.get_order_size_quantum(trading_pair, amount)
        if size_quantum == 0:
            return amount
        return (amount // size_quantum) * size_quantum

    # ===================
    # Fee Methods
    # ===================

    def estimate_fee_pct(self, is_maker: bool) -> Decimal:
        """
        Estimate trading fee percentage.

        Args:
            is_maker: Whether order is a maker order

        Returns:
            Fee as decimal (e.g., 0.001 for 0.1%)
        """
        return Decimal("0.001")  # 0.1% default

    # ===================
    # Market Data Methods
    # ===================

    @abstractmethod
    def get_price(
        self, trading_pair: str, is_buy: bool, amount: Decimal = s_decimal_NaN
    ) -> Decimal:
        """
        Get current market price.

        Args:
            trading_pair: Trading pair
            is_buy: True for ask price, False for bid price
            amount: Optional amount for volume-weighted price

        Returns:
            Current price
        """
        pass

    # ===================
    # Status Methods
    # ===================

    @property
    def status_dict(self) -> Dict[str, bool]:
        """
        Return status of various connector components.

        Should be overridden by subclasses to report
        connection status, API availability, etc.
        """
        raise NotImplementedError

    # ===================
    # Lifecycle Methods
    # ===================

    @abstractmethod
    async def connect(self):
        """Establish connection to exchange."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close connection to exchange."""
        pass


# Import at bottom to avoid circular dependency
from quant_trading.connectors.order_types import (
    InFlightOrderBase,
    OrderType,
    TradeType,
)
