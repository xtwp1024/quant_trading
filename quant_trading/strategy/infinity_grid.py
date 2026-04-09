# -*- coding: utf-8 -*-
"""
Infinity Grid Strategy Module

Adapted from kraken-infinity-grid trading bot for backtesting.
https://github.com/btschwertfeger/kraken-infinity-grid

Key strategies:
- GridHODL: Buy and hold with grid sells at profit levels
- GridSell: Sell into rallies at grid levels
- SWING: Sell accumulated base at higher levels while continuing to buy on dips (custom DCA)
- cDCA: custom Dollar Cost Averaging
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_trading.signal import Signal, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams


@dataclass
class InfinityGridParams(StrategyParams):
    """Infinity Grid strategy parameters"""

    strategy: str = "GridHODL"  # GridHODL, GridSell, SWING, cDCA
    interval: float = 0.01  # Grid interval as decimal (0.01 = 1%)
    amount_per_grid: float = 100.0  # Quote currency amount per grid
    max_investment: float = 10000.0  # Maximum investment limit
    n_open_buy_orders: int = 5  # Number of open buy orders to maintain
    fee: float = 0.0016  # Trading fee (Kraken maker fee ~0.16%)


class InfinityGridStrategy(BaseStrategy):
    """
    Infinity Grid Strategy base class.

    This implements a grid trading system where:
    - Buy orders are placed at regular intervals below the current price
    - When buy orders fill, sell orders are placed at profit levels
    - The strategy handles dynamic interval shifting based on price movements

    Strategies:
    - GridHODL: Buy and hold with grid sells at profit levels
    - GridSell: Sell into rallies at grid levels
    - SWING: Custom DCA that sells accumulated base at higher levels
    - cDCA: Custom Dollar Cost Averaging (buy only)
    """

    name: str = "infinity_grid"

    def __init__(
        self,
        symbol: str,
        params: Optional[InfinityGridParams] = None,
    ) -> None:
        super().__init__(symbol, params)
        self.params = params or InfinityGridParams()

        # Internal state
        self._buy_orders: List[Dict[str, Any]] = []
        self._sell_orders: List[Dict[str, Any]] = []
        self._highest_buy_price: float = 0.0
        self._vol_of_unfilled_remaining: float = 0.0
        self._vol_of_unfilled_remaining_max_price: float = 0.0
        self._last_price: float = 0.0
        self._current_investment: float = 0.0

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on grid strategy"""
        signals = []

        if len(data) < 2:
            return signals

        current_price = float(data["close"].iloc[-1])
        self._last_price = current_price

        # Check if any buy orders should be filled
        for buy_order in self._buy_orders[:]:
            if current_price <= buy_order["price"]:
                # Buy order filled
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
                self._handle_buy_filled(buy_order)

        # Check if any sell orders should be filled
        for sell_order in self._sell_orders[:]:
            if current_price >= sell_order["price"]:
                # Sell order filled
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
                        },
                    )
                )
                self._sell_orders.remove(sell_order)
                self._handle_sell_filled(sell_order)

        # Place new orders to maintain grid
        self._maintain_grid(current_price)

        return signals

    def _handle_buy_filled(self, buy_order: Dict[str, Any]) -> None:
        """Handle a filled buy order"""
        if buy_order["price"] > self._highest_buy_price:
            self._highest_buy_price = buy_order["price"]

        # Update investment
        self._current_investment += buy_order["price"] * buy_order["volume"]

        # For cDCA, we don't place sell orders
        if self.params.strategy == "cDCA":
            return

        # Place sell order at profit
        sell_price = self._get_sell_price(buy_order["price"])

        if self.params.strategy == "GridSell":
            # GridSell: volume equals executed buy volume
            sell_volume = buy_order["volume"]
        else:
            # GridHODL and SWING: volume respects fee to not reduce quote over time
            sell_volume = float(
                Decimal(self.params.amount_per_grid)
                / (Decimal(sell_price) * (1 - (2 * Decimal(self.params.fee))))
            )

        self._sell_orders.append(
            {
                "price": sell_price,
                "volume": sell_volume,
                "txid": f"sell_{buy_order.get('txid', 'unknown')}",
            }
        )

    def _handle_sell_filled(self, sell_order: Dict[str, Any]) -> None:
        """Handle a filled sell order"""
        # For SWING strategy, we might place extra sell orders
        if self.params.strategy == "SWING":
            self._handle_swing_sell_filled(sell_order)

    def _handle_swing_sell_filled(self, sell_order: Dict[str, Any]) -> None:
        """
        Handle SWING-specific sell logic.
        SWING sells accumulated base at higher levels while continuing to buy on dips.
        """
        # In SWING, after a sell is filled, we continue with normal operation
        # The key innovation is the extra sell order mechanism
        pass

    def _get_sell_price(self, buy_price: float) -> float:
        """Calculate sell price based on buy price and interval"""
        return buy_price * (1 + self.params.interval)

    def _get_buy_price(self, last_price: float) -> float:
        """Calculate buy price based on last price and interval"""
        return last_price * 100 / (100 + 100 * self.params.interval)

    def _maintain_grid(self, current_price: float) -> None:
        """Maintain the grid by placing/canceling orders"""
        # Cancel buy orders that are too far below current price
        self._shift_buy_orders_up(current_price)

        # Ensure we have n_open_buy_orders
        while len(self._buy_orders) < self.params.n_open_buy_orders:
            if self._current_investment >= self.params.max_investment:
                break

            buy_price = self._get_buy_price(
                self._buy_orders[-1]["price"] if self._buy_orders else current_price
            )
            buy_volume = float(
                Decimal(self.params.amount_per_grid) / Decimal(buy_price)
            )

            self._buy_orders.append(
                {
                    "price": buy_price,
                    "volume": buy_volume,
                    "txid": f"buy_{len(self._buy_orders)}",
                }
            )

        # For SWING, place extra sell orders if no sell orders exist
        if self.params.strategy == "SWING" and len(self._sell_orders) == 0:
            self._place_extra_swing_sell(current_price)

    def _shift_buy_orders_up(self, current_price: float) -> None:
        """Cancel buy orders that are too far below current price"""
        if not self._buy_orders:
            return

        max_buy_price = max(o["price"] for o in self._buy_orders)

        # If price has risen significantly, cancel and replace orders
        threshold = max_buy_price * (1 + self.params.interval) * (1 + self.params.interval) * 1.001
        if current_price > threshold:
            self._buy_orders = []

    def _place_extra_swing_sell(self, current_price: float) -> None:
        """
        Place extra sell order for SWING strategy.
        This sells accumulated base at higher price levels.
        """
        # Extra sell is 2x interval above last price or highest buy
        reference_price = max(current_price, self._highest_buy_price)
        extra_sell_price = reference_price * (1 + self.params.interval) * (1 + self.params.interval)

        # Calculate volume based on available balance and amount per grid
        sell_volume = float(
            Decimal(self.params.amount_per_grid)
            / (Decimal(extra_sell_price) * (1 - (2 * Decimal(self.params.fee))))
        )

        self._sell_orders.append(
            {
                "price": extra_sell_price,
                "volume": sell_volume,
                "txid": f"swing_extra_{len(self._sell_orders)}",
            }
        )

    def calculate_position_size(
        self,
        signal: Signal,
        context: Any,
    ) -> float:
        """Calculate position size for the signal"""
        if signal.direction == SignalDirection.LONG:
            return self.params.amount_per_grid / signal.price
        elif signal.direction == SignalDirection.SHORT:
            return signal.metadata.get("volume", 0.0)
        return 0.0

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """Handle order filled callback"""
        pass

    def get_required_history(self) -> int:
        """Get required history length"""
        return self.params.n_open_buy_orders * 2

    def reset(self) -> None:
        """Reset strategy state"""
        self._buy_orders = []
        self._sell_orders = []
        self._highest_buy_price = 0.0
        self._vol_of_unfilled_remaining = 0.0
        self._vol_of_unfilled_remaining_max_price = 0.0
        self._current_investment = 0.0


class GridHODLStrategy(InfinityGridStrategy):
    """
    GridHODL Strategy

    Buy and hold with grid sells at profit levels.
    Places n buy orders below current price. When a buy order fills,
    a sell order is placed at a profit level above the buy price.
    The sell volume is calculated to respect fees.
    """

    name: str = "grid_hodl"

    def __init__(
        self,
        symbol: str,
        params: Optional[InfinityGridParams] = None,
    ) -> None:
        super().__init__(symbol, params)
        self.params.strategy = "GridHODL"


class GridSellStrategy(InfinityGridStrategy):
    """
    GridSell Strategy

    Sell into rallies at grid levels.
    Similar to GridHODL but the sell volume is fixed to the executed
    volume of the buy order, rather than the amount per grid.
    """

    name: str = "grid_sell"

    def __init__(
        self,
        symbol: str,
        params: Optional[InfinityGridParams] = None,
    ) -> None:
        super().__init__(symbol, params)
        self.params.strategy = "GridSell"


class SWINGStrategy(InfinityGridStrategy):
    """
    SWING Strategy

    Custom DCA that sells accumulated base at higher levels while
    continuing to buy on dips.

    Key innovation:
    - Accumulates base on dips (DCA buys)
    - Sells accumulated base at higher price levels
    - Dynamic rebalancing

    The SWING algorithm differs from GridHODL by:
    1. Placing extra sell orders to sell accumulated base
    2. Continuing to buy on dips while selling on rallies
    3. Dynamic interval adjustment for sell orders
    """

    name: str = "swing"

    def __init__(
        self,
        symbol: str,
        params: Optional[InfinityGridParams] = None,
    ) -> None:
        super().__init__(symbol, params)
        self.params.strategy = "SWING"
        self._accumulated_base: float = 0.0

    def _handle_buy_filled(self, buy_order: Dict[str, Any]) -> None:
        """Handle buy filled for SWING - accumulate base"""
        super()._handle_buy_filled(buy_order)
        self._accumulated_base += buy_order["volume"]

    def _place_extra_swing_sell(self, current_price: float) -> None:
        """Place extra sell order for SWING strategy"""
        if self._accumulated_base <= 0:
            return

        # Extra sell is 2x interval above reference price
        reference_price = max(current_price, self._highest_buy_price)
        extra_sell_price = reference_price * (1 + self.params.interval) * (1 + self.params.interval)

        # Use accumulated base for the sell
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
                }
            )

    def _handle_sell_filled(self, sell_order: Dict[str, Any]) -> None:
        """Handle SWING sell filled - reduce accumulated base"""
        super()._handle_sell_filled(sell_order)
        # Validate before subtraction to prevent negative accumulation
        sell_volume = sell_order["volume"]
        if sell_volume > self._accumulated_base:
            self._accumulated_base = 0
        else:
            self._accumulated_base -= sell_volume

    def reset(self) -> None:
        """Reset SWING strategy state"""
        super().reset()
        self._accumulated_base = 0.0


class cDCAStrategy(InfinityGridStrategy):
    """
    cDCA Strategy (custom Dollar Cost Averaging)

    Only places buy orders at regular intervals without selling.
    Accumulates base currency over time through DCA buys.

    This is ideal for long-term accumulation of an asset
    without worrying about timing the market.
    """

    name: str = "cdca"

    def __init__(
        self,
        symbol: str,
        params: Optional[InfinityGridParams] = None,
    ) -> None:
        super().__init__(symbol, params)
        self.params.strategy = "cDCA"

    def _handle_buy_filled(self, buy_order: Dict[str, Any]) -> None:
        """Handle buy filled for cDCA - just accumulate, no sells"""
        if buy_order["price"] > self._highest_buy_price:
            self._highest_buy_price = buy_order["price"]

        self._current_investment += buy_order["price"] * buy_order["volume"]

        # cDCA does NOT place sell orders
        # Just accumulate base currency over time

    def calculate_position_size(
        self,
        signal: Signal,
        context: Any,
    ) -> float:
        """Calculate position size for cDCA - buy only"""
        if signal.direction == SignalDirection.LONG:
            return self.params.amount_per_grid / signal.price
        return 0.0
