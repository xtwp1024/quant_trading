"""Signal type definitions for quant_trading.

This module unifies two signal paradigms:
1. Direction-based (SignalDirection.LONG/SHORT/NEUTRAL) — used by BaseStrategy
2. Type-based   (SignalType.BUY/SELL/EXIT_LONG/CLOSE_ALL) — used by classic strategies

Signal lifecycle:
    BUY → HOLD → EXIT_LONG/SELL/CLOSE_ALL

The recommended interface is the Signal dataclass with both `direction`
(from BaseStrategy contract) and `type` (from classic strategy usage).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SignalDirection(Enum):
    """Signal direction (compatible with BaseStrategy)."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class SignalType(Enum):
    """Signal type (used by classic strategies).

    BUY       — Open / add long position
    SELL      — Open / add short position
    EXIT_LONG — Close long position (take profit or stop)
    EXIT_SHORT — Close short position
    CLOSE_ALL — Flatten all positions
    HOLD      — No action (explicit hold signal)
    """
    BUY = "buy"
    SELL = "sell"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    CLOSE_ALL = "close_all"
    HOLD = "hold"

    # Aliases matching BaseStrategy convention
    @property
    def direction(self) -> SignalDirection:
        if self in (SignalType.BUY, SignalType.EXIT_SHORT):
            return SignalDirection.LONG
        if self in (SignalType.SELL, SignalType.EXIT_LONG):
            return SignalDirection.SHORT
        return SignalDirection.NEUTRAL


@dataclass
class Signal:
    """Trading signal.

    Attributes
    ----------
    type : SignalType
        The signal type (buy/sell/exit/close).
    symbol : str
        Trading pair symbol (e.g. "BTC/USDT").
    timestamp : int
        Unix timestamp in milliseconds.
    price : float
        Price at which signal was generated.
    strength : float
        Signal strength [0.0, 1.0]. Default 1.0.
    reason : str, optional
        Human-readable description of why signal fired.
    stop_loss : float, optional
        Stop-loss price.
    take_profit : float, optional
        Take-profit price.
    metadata : dict, optional
        Additional signal data (indicator values, confidence, etc.).

    Compatibility fields (populated automatically from `type`):
    direction : SignalDirection
        Derived from SignalType for BaseStrategy compatibility.
    """
    type: SignalType
    symbol: str
    timestamp: int
    price: float
    strength: float = 1.0
    reason: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Populated on init for backward compatibility
    direction: SignalDirection = field(init=False)

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = SignalType(self.type)
        self.direction = self.type.direction
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": self.price,
            "strength": self.strength,
            "direction": self.direction.value,
            "reason": self.reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Signal":
        return cls(
            type=SignalType(d["type"]),
            symbol=d["symbol"],
            timestamp=d["timestamp"],
            price=d["price"],
            strength=d.get("strength", 1.0),
            reason=d.get("reason"),
            stop_loss=d.get("stop_loss"),
            take_profit=d.get("take_profit"),
            metadata=d.get("metadata", {}),
        )


# Backward-compatibility aliases — re-export SignalDirection at module root
SignalDirectionEnum = SignalDirection
SignalTypeEnum = SignalType
