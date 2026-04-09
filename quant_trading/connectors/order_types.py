"""
Order Types and Execution Enums

This module provides enumerations and data classes for order management,
inspired by the Hummingbot framework's order type system.

Key Components:
- OrderType: Market, Limit, LimitMaker order types
- TradeType: Buy and Sell directions
- OrderState: Order lifecycle states
- PositionAction: Open/Close position actions
- CloseType: Reasons for position closure
- PriceType: Different price sources for order placement
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


# ===================
# Order Type Enums
# ===================


class OrderType(Enum):
    """Order types supported by the trading system."""

    MARKET = 1
    LIMIT = 2
    LIMIT_MAKER = 3
    AMM_SWAP = 4  # For DEX swaps

    def is_limit_type(self) -> bool:
        """Return True if order type is a limit order."""
        return self in (OrderType.LIMIT, OrderType.LIMIT_MAKER)


class TradeType(Enum):
    """Trade direction."""

    BUY = 1
    SELL = 2
    RANGE = 3  # For liquidity provision


class PositionAction(Enum):
    """Action for position management."""

    OPEN = "OPEN"
    CLOSE = "CLOSE"
    NIL = "NIL"


class OrderState(Enum):
    """
    Order lifecycle states.

    State Transitions:
    PENDING_CREATE -> OPEN -> PARTIALLY_FILLED -> FILLED -> COMPLETED
                  -> PENDING_CANCEL -> CANCELED
                  -> FAILED
    """

    PENDING_CREATE = 0
    OPEN = 1
    PENDING_CANCEL = 2
    CANCELED = 3
    PARTIALLY_FILLED = 4
    FILLED = 5
    FAILED = 6
    PENDING_APPROVAL = 7
    APPROVED = 8
    CREATED = 9
    COMPLETED = 10


class PriceType(Enum):
    """Price sources for order placement."""

    MidPrice = 1  # (BestBid + BestAsk) / 2
    BestBid = 2  # Highest buy price
    BestAsk = 3  # Lowest sell price
    LastTrade = 4  # Last trade price
    LastOwnTrade = 5  # Last own trade price
    InventoryCost = 6  # Cost basis for inventory
    Custom = 7  # Custom price source


class CloseType(Enum):
    """
    Reasons for position/order closure.

    Used for tracking and analysis of trading outcomes.
    """

    TIME_LIMIT = 1  # Position expired due to time limit
    STOP_LOSS = 2  # Closed by stop loss
    TAKE_PROFIT = 3  # Closed by take profit
    EXPIRED = 4  # Order/position expired
    EARLY_STOP = 5  # Manually stopped early
    TRAILING_STOP = 6  # Closed by trailing stop
    INSUFFICIENT_BALANCE = 7  # Closed due to insufficient funds
    FAILED = 8  # Execution failed
    COMPLETED = 9  # Successfully completed
    POSITION_HOLD = 10  # Position held open


class PositionSide(Enum):
    """Position side for derivatives trading."""

    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


# ===================
# Data Classes
# ===================


@dataclass
class TokenAmount:
    """
    Represents an amount of a specific token.

    Used for fee calculations and balance tracking.
    """

    token: str
    amount: Decimal

    def __iter__(self):
        return iter((self.token, self.amount))

    def to_json(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "amount": str(self.amount),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "TokenAmount":
        return cls(token=data["token"], amount=Decimal(data["amount"]))


@dataclass
class TradeFee:
    """
    Trade fee information.

    Supports both percentage fees and flat fees.
    """

    percent: Decimal = Decimal("0")
    percent_token: Optional[str] = None
    flat_fees: List[TokenAmount] = field(default_factory=list)

    @property
    def fee_asset(self) -> str:
        """Get the token in which fees are paid."""
        if self.percent_token:
            return self.percent_token
        if self.flat_fees:
            return self.flat_fees[0].token
        return ""

    def total_fee_amount(
        self, trading_pair: str, price: Decimal, amount: Decimal
    ) -> Decimal:
        """
        Calculate total fee in quote currency.

        Args:
            trading_pair: Trading pair (e.g., "BTC-USDT")
            price: Execution price
            amount: Order amount

        Returns:
            Total fee amount
        """
        parts = trading_pair.split("-")
        base = parts[0]
        quote = parts[1] if len(parts) > 1 else parts[0]

        # Percentage fee
        percent_amount = price * amount * self.percent

        # Flat fees (converted to quote if needed)
        flat_amount = sum(
            fee.amount for fee in self.flat_fees if fee.token == quote
        )

        return percent_amount + flat_amount


@dataclass
class OrderUpdate:
    """
    Represents an update to an order's state.

    Used for REST/WebSocket order update processing.
    """

    trading_pair: str
    update_timestamp: float  # seconds
    new_state: OrderState
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    misc_updates: Optional[Dict[str, Any]] = None


@dataclass
class TradeUpdate:
    """
    Represents a trade/fill event.

    Tracks individual fills for an order.
    """

    trade_id: str
    client_order_id: str
    exchange_order_id: str
    trading_pair: str
    fill_timestamp: float  # seconds
    fill_price: Decimal
    fill_base_amount: Decimal
    fill_quote_amount: Decimal
    fee: TradeFee
    is_taker: bool = True

    @property
    def fee_asset(self) -> str:
        return self.fee.fee_asset

    def to_json(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "trading_pair": self.trading_pair,
            "fill_timestamp": self.fill_timestamp,
            "fill_price": str(self.fill_price),
            "fill_base_amount": str(self.fill_base_amount),
            "fill_quote_amount": str(self.fill_quote_amount),
            "fee": {
                "percent": str(self.fee.percent),
                "percent_token": self.fee.percent_token,
                "flat_fees": [f.to_json() for f in self.fee.flat_fees],
            },
            "is_taker": self.is_taker,
        }


@dataclass
class InFlightOrderBase:
    """
    Base class for tracking in-flight orders.

    Provides order lifecycle management with state transitions
    and event notifications.
    """

    client_order_id: str
    trading_pair: str
    order_type: OrderType
    trade_type: TradeType
    amount: Decimal
    creation_timestamp: float
    price: Optional[Decimal] = None
    exchange_order_id: Optional[str] = None
    initial_state: OrderState = OrderState.PENDING_CREATE

    # Execution tracking
    executed_amount_base: Decimal = Decimal("0")
    executed_amount_quote: Decimal = Decimal("0")
    last_update_timestamp: float = 0.0

    # Async events for state transitions
    exchange_order_id_update_event: asyncio.Event = field(
        default_factory=asyncio.Event
    )
    completely_filled_event: asyncio.Event = field(default_factory=asyncio.Event)
    processed_by_exchange_event: asyncio.Event = field(
        default_factory=asyncio.Event
    )

    # Internal state
    _current_state: OrderState = field(init=False, repr=False)

    def __post_init__(self):
        if self.exchange_order_id:
            self.exchange_order_id_update_event.set()
        self._trade_updates: Dict[str, TradeUpdate] = {}
        self._current_state = self.initial_state

    @property
    def current_state(self) -> OrderState:
        return self._current_state

    @current_state.setter
    def current_state(self, value: OrderState):
        self._current_state = value

    @property
    def base_asset(self) -> str:
        return self.trading_pair.split("-")[0]

    @property
    def quote_asset(self) -> str:
        parts = self.trading_pair.split("-")
        return parts[1] if len(parts) > 1 else parts[0]

    @property
    def is_pending_create(self) -> bool:
        return self.current_state == OrderState.PENDING_CREATE

    @property
    def is_open(self) -> bool:
        return self.current_state in {
            OrderState.PENDING_CREATE,
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.PENDING_CANCEL,
        }

    @property
    def is_done(self) -> bool:
        return self.current_state in {
            OrderState.CANCELED,
            OrderState.FILLED,
            OrderState.FAILED,
        } or self.executed_amount_base >= self.amount

    @property
    def is_filled(self) -> bool:
        return (
            self.current_state == OrderState.FILLED
            or self.executed_amount_base >= self.amount
        )

    @property
    def is_failure(self) -> bool:
        return self.current_state == OrderState.FAILED

    @property
    def is_cancelled(self) -> bool:
        return self.current_state == OrderState.CANCELED

    @property
    def average_executed_price(self) -> Optional[Decimal]:
        """Calculate volume-weighted average execution price."""
        if self.executed_amount_base == 0:
            return None
        return self.executed_amount_quote / self.executed_amount_base

    def update_with_order_update(self, order_update: OrderUpdate) -> bool:
        """
        Update order with an OrderUpdate.

        Returns True if the order was updated.
        """
        if (
            order_update.client_order_id != self.client_order_id
            and order_update.exchange_order_id != self.exchange_order_id
        ):
            return False

        if self.exchange_order_id is None and order_update.exchange_order_id:
            self.exchange_order_id = order_update.exchange_order_id
            self.exchange_order_id_update_event.set()

        self._current_state = order_update.new_state
        self.last_update_timestamp = order_update.update_timestamp

        return True

    def update_with_trade_update(self, trade_update: TradeUpdate) -> bool:
        """
        Update order with a TradeUpdate.

        Returns True if the order was updated.
        """
        if trade_update.trade_id in self._trade_updates:
            return False

        self._trade_updates[trade_update.trade_id] = trade_update
        self.executed_amount_base += trade_update.fill_base_amount
        self.executed_amount_quote += trade_update.fill_quote_amount
        self.last_update_timestamp = trade_update.fill_timestamp

        if self.executed_amount_base >= self.amount:
            self.completely_filled_event.set()

        return True

    def to_json(self) -> Dict[str, Any]:
        """Serialize order to JSON for persistence."""
        return {
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "trading_pair": self.trading_pair,
            "order_type": self.order_type.name,
            "trade_type": self.trade_type.name,
            "price": str(self.price) if self.price else None,
            "amount": str(self.amount),
            "executed_amount_base": str(self.executed_amount_base),
            "executed_amount_quote": str(self.executed_amount_quote),
            "last_state": str(self.current_state.value),
            "creation_timestamp": self.creation_timestamp,
            "last_update_timestamp": self.last_update_timestamp,
        }
