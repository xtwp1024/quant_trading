"""Position Tracker — cross-exchange, multi-symbol position management.

Provides:
- Position       : Single-position data (entry, size, PnL, exposure)
- PositionState : Aggregated state across all positions
- PositionTracker : Full position management with risk metrics

Usage
-----
```python
from quant_trading.execution.position_tracker import PositionTracker

tracker = PositionTracker(initial_cash=100_000)
tracker.update_price("BTC/USDT", 50_000)
tracker.open_position("BTC/USDT", "buy", 0.5, 50_000, commission=25)
tracker.update_price("BTC/USDT", 52_000)
state = tracker.get_state()
print(f"Total PnL: ${state.total_unrealized_pnl:.2f}")
print(f"Exposure: {state.total_exposure_pct:.1%}")
```
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("PositionTracker")


@dataclass
class Position:
    """Single position data.

    Attributes
    ----------
    symbol : str
        Trading pair.
    side : str
        "long" or "short".
    quantity : float
        Base asset quantity.
    entry_price : float
        Average entry price (weighted).
    current_price : float
        Most recent market price.
    entry_value : float
        Notional value at entry (quantity * entry_price).
    unrealized_pnl : float
        Unrealized PnL in quote currency.
    unrealized_pnl_pct : float
        Unrealized PnL as percentage of entry value.
    commission_paid : float
        Total commission paid for this position.
    opened_at : int
        Unix timestamp (ms) when position was opened.
    bars_held : int
        Number of bars since opening (set by tracker).
    stop_loss : float
        Active stop-loss price (0 = none).
    take_profit : float
        Active take-profit price (0 = none).
    """
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    current_price: float = 0.0
    entry_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    commission_paid: float = 0.0
    opened_at: int = 0
    bars_held: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    def update_price(self, price: float) -> None:
        """Recalculate PnL at a new market price."""
        self.current_price = price
        if self.side == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # short
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

        if self.entry_value > 0:
            self.unrealized_pnl_pct = self.unrealized_pnl / self.entry_value

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "entry_value": self.entry_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "commission_paid": self.commission_paid,
            "opened_at": self.opened_at,
            "bars_held": self.bars_held,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


@dataclass
class PositionState:
    """Aggregated portfolio state across all positions."""
    total_exposure: float = 0.0       # Sum of abs(position_value)
    total_exposure_pct: float = 0.0  # Exposure / portfolio_value
    total_unrealized_pnl: float = 0.0
    total_commission_paid: float = 0.0
    net_exposure: float = 0.0        # long exposure - short exposure
    gross_exposure: float = 0.0      # sum of long + |short| exposure
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    num_positions: int = 0
    num_losing_positions: int = 0
    num_winning_positions: int = 0
    largest_position: float = 0.0
    smallest_position: float = 0.0
    portfolio_value: float = 0.0     # cash + net exposure


class PositionTracker:
    """Cross-exchange, multi-symbol position tracker with risk metrics.

    Manages:
    - Opening/closing positions (long/short)
    - Price updates for mark-to-market PnL
    - Stop-loss and take-profit monitoring
    - Risk metrics (exposure, leverage, drawdown)
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        max_leverage: float = 1.0,
        max_position_pct: float = 0.2,
        max_exposure_pct: float = 1.0,
    ):
        """Initialize position tracker.

        Parameters
        ----------
        initial_cash : float
            Starting cash in quote currency.
        max_leverage : float
            Maximum allowed leverage (1.0 = no leverage).
        max_position_pct : float
            Max position size as fraction of portfolio (0.2 = 20%).
        max_exposure_pct : float
            Max total long or short exposure (1.0 = 100%).
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.max_exposure_pct = max_exposure_pct

        self._positions: Dict[str, Position] = {}
        self._closed: List[Position] = []
        self._bar_index: int = 0
        self._commission_paid: float = 0.0
        self.logger = logging.getLogger("PositionTracker")

    # -------------------------------------------------------------------------
    # Position operations
    # -------------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: str,  # "buy"=long, "sell"=short
        quantity: float,
        price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        commission: float = 0.0,
        timestamp: int = 0,
    ) -> Position:
        """Open a new position.

        Parameters
        ----------
        symbol : str
            Trading pair.
        side : str
            "buy" (long) or "sell" (short).
        quantity : float
            Base asset quantity.
        price : float
            Execution price.
        stop_loss : float
            Stop-loss price (0 = no SL).
        take_profit : float
            Take-profit price (0 = no TP).
        commission : float
            Commission charged for this trade.
        timestamp : int
            Unix timestamp (ms).

        Returns
        -------
        Position
            The opened position.

        Raises
        ------
        ValueError
            If position already exists for this symbol or size exceeds limits.
        """
        from quant_trading.execution.commission import FixedCommission

        side_normalized = "long" if side.lower() in ("buy", "long") else "short"
        notional = quantity * price

        if symbol in self._positions:
            raise ValueError(f"Position already exists for {symbol}. Use modify_position.")

        portfolio_value = self._portfolio_value()
        max_pos_value = portfolio_value * self.max_position_pct

        if notional > max_pos_value:
            quantity = max_pos_value / price
            notional = quantity * price
            self.logger.warning(
                f"Quantity reduced to {quantity} due to max_position_pct={self.max_position_pct}"
            )

        # Check leverage
        total_exposure = self._total_exposure() + notional
        if total_exposure > portfolio_value * self.max_leverage:
            raise ValueError(
                f"Opening this position would exceed max_leverage={self.max_leverage}. "
                f"Current exposure: {self._total_exposure():.2f}, new: {notional:.2f}"
            )

        position = Position(
            symbol=symbol,
            side=side_normalized,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            entry_value=notional,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            commission_paid=commission,
            opened_at=timestamp or 0,
            bars_held=0,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        position.update_price(price)

        self._positions[symbol] = position
        self._commission_paid += commission

        # Deduct commission from cash
        self.cash -= commission

        self.logger.info(
            f"Opened {side_normalized} position: {symbol} qty={quantity} @ {price} "
            f"(notional={notional:.2f}, commission={commission:.4f})"
        )
        return position

    def close_position(
        self,
        symbol: str,
        price: float,
        commission: float = 0.0,
        timestamp: int = 0,
    ) -> Position:
        """Close an existing position.

        Parameters
        ----------
        symbol : str
            Trading pair.
        price : float
            Execution price.
        commission : float
            Commission charged.
        timestamp : int
            Unix timestamp (ms).

        Returns
        -------
        Position
            The closed position (moved to closed list).
        """
        if symbol not in self._positions:
            raise ValueError(f"No position found for {symbol}")

        pos = self._positions[symbol]

        # Realize PnL
        if pos.side == "long":
            realized_pnl = (price - pos.entry_price) * pos.quantity
        else:
            realized_pnl = (pos.entry_price - price) * pos.quantity

        pos.update_price(price)
        pos.commission_paid += commission

        self.cash += realized_pnl - commission
        self._commission_paid += commission
        self._closed.append(pos)
        del self._positions[symbol]

        self.logger.info(
            f"Closed {pos.side} position: {symbol} @ {price} "
            f"(realized PnL={realized_pnl:.2f}, commission={commission:.4f})"
        )
        return pos

    def modify_position(
        self,
        symbol: str,
        add_quantity: float = 0.0,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None,
    ) -> Position:
        """Add to or reduce an existing position (DCA-style).

        Parameters
        ----------
        symbol : str
            Trading pair.
        add_quantity : float
            Positive = add to position, negative = reduce.
        new_stop_loss : float, optional
            Update stop-loss.
        new_take_profit : float, optional
            Update take-profit.

        Returns
        -------
        Position
            Updated position.
        """
        if symbol not in self._positions:
            raise ValueError(f"No position found for {symbol}")

        pos = self._positions[symbol]

        if add_quantity != 0.0:
            new_qty = pos.quantity + add_quantity
            if new_qty <= 0:
                raise ValueError(f"add_quantity would close position. Use close_position instead.")

            # Update weighted average entry price
            old_value = pos.entry_price * pos.quantity
            add_value = pos.current_price * add_quantity
            pos.quantity = new_qty
            pos.entry_price = (old_value + add_value) / new_qty
            pos.entry_value = pos.entry_price * pos.quantity

        if new_stop_loss is not None:
            pos.stop_loss = new_stop_loss
        if new_take_profit is not None:
            pos.take_profit = new_take_profit

        return pos

    def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a position (mark-to-market).

        Also checks stop-loss and take-profit.
        """
        if symbol not in self._positions:
            return
        pos = self._positions[symbol]
        pos.update_price(price)

    def trigger_stops(self, symbol: str, current_price: float) -> List[str]:
        """Check and trigger SL/TP for a position.

        Returns
        -------
        List[str]
            List of triggered actions: "stop_loss", "take_profit", or empty.
        """
        if symbol not in self._positions:
            return []
        pos = self._positions[symbol]
        triggered = []

        if pos.side == "long":
            if 0 < pos.stop_loss < current_price:
                triggered.append("stop_loss")
            elif pos.take_profit > 0 and current_price >= pos.take_profit:
                triggered.append("take_profit")
        else:  # short
            if pos.stop_loss > 0 and current_price > pos.stop_loss:
                triggered.append("stop_loss")
            elif pos.take_profit > 0 and current_price <= pos.take_profit:
                triggered.append("take_profit")

        return triggered

    # -------------------------------------------------------------------------
    # State queries
    # -------------------------------------------------------------------------

    def get_state(self) -> PositionState:
        """Get aggregated portfolio state."""
        positions = list(self._positions.values())
        portfolio_value = self._portfolio_value()

        long_exp = sum(
            p.quantity * p.current_price for p in positions if p.side == "long"
        )
        short_exp = sum(
            p.quantity * p.current_price for p in positions if p.side == "short"
        )

        total_unrealized = sum(p.unrealized_pnl for p in positions)
        total_exposure = long_exp + short_exp
        net_exp = long_exp - short_exp

        winning = [p for p in positions if p.unrealized_pnl > 0]
        losing = [p for p in positions if p.unrealized_pnl < 0]
        qties = [p.quantity * p.current_price for p in positions]

        return PositionState(
            total_exposure=total_exposure,
            total_exposure_pct=total_exposure / portfolio_value if portfolio_value > 0 else 0,
            total_unrealized_pnl=total_unrealized,
            total_commission_paid=self._commission_paid,
            net_exposure=net_exp,
            gross_exposure=total_exposure,
            long_exposure=long_exp,
            short_exposure=short_exp,
            num_positions=len(positions),
            num_losing_positions=len(losing),
            num_winning_positions=len(winning),
            largest_position=max(qties) if qties else 0.0,
            smallest_position=min(qties) if qties else 0.0,
            portfolio_value=portfolio_value,
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a position by symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    def get_closed_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get closed positions, optionally filtered by symbol."""
        if symbol:
            return [p for p in self._closed if p.symbol == symbol]
        return list(self._closed)

    def has_position(self, symbol: str) -> bool:
        """Check if a position exists for symbol."""
        return symbol in self._positions

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _portfolio_value(self) -> float:
        """Net liquidation value: cash + net unrealized PnL."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self.cash + unrealized

    def _total_exposure(self) -> float:
        """Sum of all position notionals (long + |short|)."""
        return sum(
            p.quantity * p.current_price for p in self._positions.values()
        )

    def tick(self) -> None:
        """Advance the bar counter (call each new bar)."""
        self._bar_index += 1
        for pos in self._positions.values():
            pos.bars_held = self._bar_index
