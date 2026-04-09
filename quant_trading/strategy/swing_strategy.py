# -*- coding: utf-8 -*-
"""
SWING Strategy Implementation

The SWING (Sell Into Strength, Buy Into Weakness) strategy is a grid-based
trading algorithm that combines the best of both worlds:

1. DCA (Dollar Cost Averaging) on the way down - accumulate base currency
2. Grid selling on the way up - sell accumulated base at profit levels

Key Innovation:
- Unlike traditional grid strategies that only sell what they buy,
  SWING accumulates base currency over time and sells it at higher levels
- The strategy continues to buy on dips while selling on rallies
- Dynamic interval shifting based on price movements

How it works:
1. Places n buy orders below current price (DCA on dips)
2. When buy orders fill, places sell orders at profit levels
3. Additionally places extra sell orders to sell accumulated base
4. If price rises significantly, cancels lower buy orders and places new ones
5. If price drops, buy orders fill and process repeats

The SWING algorithm:
- Accumulates base on dips (DCA buys)
- Sells accumulated base at higher price levels
- Dynamic rebalancing between base and quote currency

Reference: kraken-infinity-grid (https://github.com/btschwertfeger/kraken-infinity-grid)
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_trading.signal import Signal, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams


@dataclass
class SWINGParams(StrategyParams):
    """SWING Strategy parameters"""

    interval: float = 0.01  # Grid interval as decimal (0.01 = 1%)
    amount_per_grid: float = 100.0  # Quote currency amount per grid
    max_investment: float = 10000.0  # Maximum investment limit
    n_open_buy_orders: int = 5  # Number of open buy orders to maintain
    fee: float = 0.0016  # Trading fee (Kraken maker fee ~0.16%)


class SWINGStrategy(BaseStrategy):
    """
    SWING Strategy Implementation

    Sells accumulated base at higher levels while continuing to buy on dips.

    This is the key innovation over standard grid strategies:
    - GridHODL/GridSell: Only sells what it buys
    - SWING: Accumulate base + sell accumulated base = more profits

    The strategy maintains:
    - n open buy orders at grid levels below current price
    - Sell orders for filled buy orders (at profit)
    - Extra sell orders to liquidate accumulated base
    """

    name: str = "swing"

    def __init__(
        self,
        symbol: str,
        params: Optional[SWINGParams] = None,
    ) -> None:
        super().__init__(symbol, params)
        self.params = params or SWINGParams()

        # Internal state
        self._buy_orders: List[Dict[str, Any]] = []
        self._sell_orders: List[Dict[str, Any]] = []
        self._highest_buy_price: float = 0.0
        self._accumulated_base: float = 0.0  # Base currency accumulated via DCA
        self._current_investment: float = 0.0
        self._last_price: float = 0.0

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals for SWING strategy"""
        signals = []

        if len(data) < 2:
            return signals

        current_price = float(data["close"].iloc[-1])
        self._last_price = current_price

        # Check buy orders
        for buy_order in self._buy_orders[:]:
            if current_price <= buy_order["price"]:
                signals.append(
                    Signal(
                        symbol=self.symbol,
                        direction=SignalDirection.LONG,
                        price=buy_order["price"],
                        strength=1.0,
                        metadata={
                            "order_type": "buy",
                            "grid_price": buy_order["price"],
                            "volume": buy_order["volume"],
                        },
                    )
                )
                self._buy_orders.remove(buy_order)
                self._on_buy_filled(buy_order)

        # Check sell orders
        for sell_order in self._sell_orders[:]:
            if current_price >= sell_order["price"]:
                signals.append(
                    Signal(
                        symbol=self.symbol,
                        direction=SignalDirection.SHORT,
                        price=sell_order["price"],
                        strength=1.0,
                        metadata={
                            "order_type": "sell",
                            "grid_price": sell_order["price"],
                            "volume": sell_order["volume"],
                            "is_extra_sell": sell_order.get("is_extra_sell", False),
                        },
                    )
                )
                self._sell_orders.remove(sell_order)
                self._on_sell_filled(sell_order)

        # Maintain the grid
        self._maintain_grid(current_price)

        return signals

    def _on_buy_filled(self, buy_order: Dict[str, Any]) -> None:
        """
        Handle filled buy order.

        When a buy order fills in SWING:
        1. Update highest buy price
        2. Accumulate the base currency
        3. Place sell order at profit level
        4. If extra sell conditions met, place additional sell
        """
        price = buy_order["price"]
        volume = buy_order["volume"]

        # Track highest buy price for dynamic sell placement
        if price > self._highest_buy_price:
            self._highest_buy_price = price

        # Update investment
        self._current_investment += price * volume

        # Accumulate base currency (key SWING feature)
        self._accumulated_base += volume

        # Place sell order at profit level
        sell_price = price * (1 + self.params.interval)
        sell_volume = float(
            Decimal(self.params.amount_per_grid)
            / (Decimal(sell_price) * (1 - (2 * Decimal(self.params.fee))))
        )

        self._sell_orders.append(
            {
                "price": sell_price,
                "volume": sell_volume,
                "txid": f"swing_sell_{buy_order.get('txid', 'unknown')}",
                "is_extra_sell": False,
            }
        )

    def _on_sell_filled(self, sell_order: Dict[str, Any]) -> None:
        """
        Handle filled sell order.

        When a sell order fills in SWING:
        1. Reduce accumulated base
        2. The quote currency increases (profit realized)
        3. Continue with normal grid maintenance
        """
        volume = sell_order["volume"]

        if not sell_order.get("is_extra_sell", False):
            # Regular sell from filled buy - reduce accumulated base
            self._accumulated_base -= volume
            if self._accumulated_base < 0:
                self._accumulated_base = 0

    def _maintain_grid(self, current_price: float) -> None:
        """
        Maintain the SWING grid.

        1. Shift buy orders up if price rises too high
        2. Ensure n_open_buy_orders are maintained
        3. Place extra sell orders if conditions met
        """
        # Step 1: Shift buy orders up if needed
        self._shift_buy_orders_up(current_price)

        # Step 2: Ensure n open buy orders
        while (
            len(self._buy_orders) < self.params.n_open_buy_orders
            and self._current_investment < self.params.max_investment
        ):
            last_buy_price = (
                self._buy_orders[-1]["price"] if self._buy_orders else current_price
            )
            buy_price = self._get_buy_price(last_buy_price)
            buy_volume = float(
                Decimal(self.params.amount_per_grid) / Decimal(buy_price)
            )

            self._buy_orders.append(
                {
                    "price": buy_price,
                    "volume": buy_volume,
                    "txid": f"swing_buy_{len(self._buy_orders)}",
                }
            )

        # Step 3: Place extra sell orders (SWING key feature)
        if len(self._sell_orders) == 0 and self._accumulated_base > 0:
            self._place_extra_sell(current_price)

    def _shift_buy_orders_up(self, current_price: float) -> None:
        """
        Shift buy orders up if price rises significantly.

        If current price > highest buy * (1 + interval)^2 * 1.001,
        cancel all buy orders and they'll be replaced at new levels.
        """
        if not self._buy_orders:
            return

        max_buy_price = max(o["price"] for o in self._buy_orders)
        threshold = (
            max_buy_price
            * (1 + self.params.interval)
            * (1 + self.params.interval)
            * 1.001
        )

        if current_price > threshold:
            self._buy_orders = []

    def _place_extra_sell(self, current_price: float) -> None:
        """
        Place extra sell order for accumulated base.

        This is the KEY SWING DIFFERENCE from other grid strategies.
        Extra sell is placed at 2x interval above reference price.

        Reference price = max(last_price, highest_buy_price)
        """
        if self._accumulated_base <= 0:
            return

        reference_price = max(current_price, self._highest_buy_price)
        extra_sell_price = (
            reference_price * (1 + self.params.interval) * (1 + self.params.interval)
        )

        sell_volume = float(
            Decimal(self.params.amount_per_grid)
            / (Decimal(extra_sell_price) * (1 - (2 * Decimal(self.params.fee))))
        )

        # Cap at accumulated base
        sell_volume = min(sell_volume, self._accumulated_base)

        if sell_volume > 0:
            self._sell_orders.append(
                {
                    "price": extra_sell_price,
                    "volume": sell_volume,
                    "txid": f"swing_extra_{len(self._sell_orders)}",
                    "is_extra_sell": True,
                }
            )

    def _get_buy_price(self, last_price: float) -> float:
        """
        Calculate buy price for next grid level.

        Buy price = last_price * 100 / (100 + 100 * interval)
        """
        return last_price * 100 / (100 + 100 * self.params.interval)

    def calculate_position_size(
        self,
        signal: Signal,
        context: Any,
    ) -> float:
        """Calculate position size for SWING signals"""
        if signal.direction == SignalDirection.LONG:
            return self.params.amount_per_grid / signal.price
        elif signal.direction == SignalDirection.SHORT:
            return signal.metadata.get("volume", 0.0)
        return 0.0

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Handle order filled callback"""
        pass

    def get_required_history(self) -> int:
        """Get required history length for SWING"""
        return self.params.n_open_buy_orders * 2

    def get_accumulated_base(self) -> float:
        """Get current accumulated base currency"""
        return self._accumulated_base

    def get_investment(self) -> float:
        """Get current investment"""
        return self._current_investment

    def reset(self) -> None:
        """Reset SWING strategy state"""
        self._buy_orders = []
        self._sell_orders = []
        self._highest_buy_price = 0.0
        self._accumulated_base = 0.0
        self._current_investment = 0.0
        self._last_price = 0.0
