"""Data models for trading"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order"""
    symbol: str
    side: OrderSide
    amount: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: Optional[str] = None
    filled_amount: float = 0.0
    filled_price: Optional[float] = None


@dataclass
class Position:
    """Trading position"""
    symbol: str
    amount: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    def update_price(self, new_price: float) -> None:
        """Update current price and recalculate PnL"""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.entry_price) * self.amount
