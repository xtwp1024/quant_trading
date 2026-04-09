"""
Grid Trading Strategy Module

Absorbed from grid_trading_bot library with enhancements for the quant_trading framework.
Supports arithmetic/geometric spacing, simple and hedged grid strategies, and backtesting.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class GridSpacingType(Enum):
    """Grid spacing type - arithmetic or geometric progression."""
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class GridStrategyType(Enum):
    """Grid strategy type - simple or hedged."""
    SIMPLE_GRID = "simple_grid"
    HEDGED_GRID = "hedged_grid"


class GridCycleState(Enum):
    """State of a grid level in the trading cycle."""
    READY_TO_BUY_OR_SELL = "ready_to_buy_or_sell"
    READY_TO_BUY = "ready_to_buy"
    WAITING_FOR_BUY_FILL = "waiting_for_buy_fill"
    READY_TO_SELL = "ready_to_sell"
    WAITING_FOR_SELL_FILL = "waiting_for_sell_fill"


@dataclass
class GridLevel:
    """Represents a single grid level with price and state."""
    price: float
    state: GridCycleState = GridCycleState.READY_TO_BUY
    orders: List[Dict[str, Any]] = field(default_factory=list)
    paired_buy_level: Optional["GridLevel"] = None
    paired_sell_level: Optional["GridLevel"] = None

    def add_order(self, order: Dict[str, Any]) -> None:
        """Record an order at this level."""
        self.orders.append(order)

    def __str__(self) -> str:
        return (
            f"GridLevel(price={self.price:.4f}, "
            f"state={self.state.name}, "
            f"num_orders={len(self.orders)})"
        )


class GridSpacingStrategy:
    """Calculate grid price levels using arithmetic or geometric spacing."""

    @staticmethod
    def calculate_grids(
        bottom_range: float,
        top_range: float,
        num_grids: int,
        spacing_type: GridSpacingType,
    ) -> Tuple[List[float], float]:
        """
        Calculate price grids and central price based on spacing type.

        Args:
            bottom_range: Lower price boundary
            top_range: Upper price boundary
            num_grids: Number of grid levels
            spacing_type: ARITHMETIC or GEOMETRIC

        Returns:
            Tuple of (price_grids list, central_price)
        """
        if num_grids < 2:
            raise ValueError(f"num_grids must be at least 2 for arithmetic/geometric grids, got {num_grids}")

        if spacing_type == GridSpacingType.ARITHMETIC:
            grids = [
                bottom_range + i * (top_range - bottom_range) / (num_grids - 1)
                for i in range(num_grids)
            ]
            central_price = (top_range + bottom_range) / 2

        elif spacing_type == GridSpacingType.GEOMETRIC:
            grids = []
            ratio = (top_range / bottom_range) ** (1 / (num_grids - 1))
            current_price = bottom_range

            for _ in range(num_grids):
                grids.append(current_price)
                current_price *= ratio

            central_index = len(grids) // 2
            if num_grids % 2 == 0:
                central_price = (grids[central_index - 1] + grids[central_index]) / 2
            else:
                central_price = grids[central_index]
        else:
            raise ValueError(f"Unsupported spacing type: {spacing_type}")

        return grids, central_price


class BaseGridStrategy:
    """Abstract base class for grid strategy behavior."""

    def initialize_levels(
        self,
        price_grids: List[float],
        central_price: float,
    ) -> Tuple[List[float], List[float], Dict[float, GridLevel]]:
        """
        Initialize grid levels based on strategy type.

        Returns:
            Tuple of (buy_grids, sell_grids, grid_levels dict)
        """
        raise NotImplementedError

    def get_paired_sell_level(
        self,
        buy_grid_level: GridLevel,
        grid_levels: Dict[float, GridLevel],
        sorted_sell_grids: List[float],
    ) -> Optional[GridLevel]:
        """Determine the paired sell level for a buy grid level."""
        raise NotImplementedError

    def complete_order(
        self,
        grid_level: GridLevel,
        order_side: str,
    ) -> None:
        """Mark order completion and transition grid level state."""
        raise NotImplementedError

    def can_place_order(
        self,
        grid_level: GridLevel,
        order_side: str,
    ) -> bool:
        """Determine if an order can be placed on the grid level."""
        raise NotImplementedError


class SimpleGridStrategy(BaseGridStrategy):
    """
    Simple Grid Strategy:
    - Buy orders placed on grid levels below central price
    - Sell orders placed on grid levels above central price
    - After buy fills, level transitions to ready for sell
    - After sell fills, level transitions to ready for buy
    """

    def initialize_levels(
        self,
        price_grids: List[float],
        central_price: float,
    ) -> Tuple[List[float], List[float], Dict[float, GridLevel]]:
        buy_grids = [p for p in price_grids if p <= central_price]
        sell_grids = [p for p in price_grids if p > central_price]

        grid_levels = {
            price: GridLevel(
                price,
                GridCycleState.READY_TO_BUY if price <= central_price else GridCycleState.READY_TO_SELL,
            )
            for price in price_grids
        }
        return buy_grids, sell_grids, grid_levels

    def get_paired_sell_level(
        self,
        buy_grid_level: GridLevel,
        grid_levels: Dict[float, GridLevel],
        sorted_sell_grids: List[float],
    ) -> Optional[GridLevel]:
        for sell_price in sorted_sell_grids:
            sell_level = grid_levels[sell_price]
            if not self.can_place_order(sell_level, "sell"):
                continue
            if sell_price > buy_grid_level.price:
                return sell_level
        return None

    def complete_order(self, grid_level: GridLevel, order_side: str) -> None:
        if order_side == "buy":
            grid_level.state = GridCycleState.READY_TO_SELL
        elif order_side == "sell":
            grid_level.state = GridCycleState.READY_TO_BUY

    def can_place_order(self, grid_level: GridLevel, order_side: str) -> bool:
        if order_side == "buy":
            return grid_level.state == GridCycleState.READY_TO_BUY
        elif order_side == "sell":
            return grid_level.state == GridCycleState.READY_TO_SELL
        return False


class HedgedGridStrategy(BaseGridStrategy):
    """
    Hedged Grid Strategy:
    - Long and short positions simultaneously
    - Buy grids are all except top grid
    - Sell grids are all except bottom grid
    - Paired levels - when buy fills, paired sell becomes ready
    - When sell fills, paired buy becomes ready
    """

    def initialize_levels(
        self,
        price_grids: List[float],
        central_price: float,
    ) -> Tuple[List[float], List[float], Dict[float, GridLevel]]:
        buy_grids = price_grids[:-1]  # All except top grid
        sell_grids = price_grids[1:]  # All except bottom grid

        grid_levels = {
            price: GridLevel(
                price,
                GridCycleState.READY_TO_BUY_OR_SELL if price != price_grids[-1] else GridCycleState.READY_TO_SELL,
            )
            for price in price_grids
        }
        return buy_grids, sell_grids, grid_levels

    def get_paired_sell_level(
        self,
        buy_grid_level: GridLevel,
        grid_levels: Dict[float, GridLevel],
        sorted_sell_grids: List[float],
    ) -> Optional[GridLevel]:
        # In hedged mode, paired sell is the next level up
        sorted_prices = sorted(grid_levels.keys())
        current_index = sorted_prices.index(buy_grid_level.price)
        if current_index + 1 < len(sorted_prices):
            paired_sell_price = sorted_prices[current_index + 1]
            return grid_levels[paired_sell_price]
        return None

    def complete_order(self, grid_level: GridLevel, order_side: str) -> None:
        if order_side == "buy":
            grid_level.state = GridCycleState.READY_TO_BUY_OR_SELL
            if grid_level.paired_sell_level:
                grid_level.paired_sell_level.state = GridCycleState.READY_TO_SELL

        elif order_side == "sell":
            grid_level.state = GridCycleState.READY_TO_BUY_OR_SELL
            if grid_level.paired_buy_level:
                grid_level.paired_buy_level.state = GridCycleState.READY_TO_BUY

    def can_place_order(self, grid_level: GridLevel, order_side: str) -> bool:
        if order_side == "buy":
            return grid_level.state in {
                GridCycleState.READY_TO_BUY,
                GridCycleState.READY_TO_BUY_OR_SELL,
            }
        elif order_side == "sell":
            return grid_level.state in {
                GridCycleState.READY_TO_SELL,
                GridCycleState.READY_TO_BUY_OR_SELL,
            }
        return False


class GridManager:
    """
    Manages grid levels, spacing, and order placement for grid trading.

    Supports both arithmetic and geometric spacing, simple and hedged grid strategies.
    """

    def __init__(
        self,
        bottom_range: float,
        top_range: float,
        num_grids: int,
        spacing_type: GridSpacingType = GridSpacingType.ARITHMETIC,
        strategy_type: GridStrategyType = GridStrategyType.SIMPLE_GRID,
        buy_ratio: float = 1.0,
        sell_ratio: float = 1.0,
    ):
        self.bottom_range = bottom_range
        self.top_range = top_range
        self.num_grids = num_grids
        self.spacing_type = spacing_type
        self.strategy_type = strategy_type
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio

        self.grid_strategy: BaseGridStrategy = self._create_grid_strategy()
        self.price_grids: List[float] = []
        self.central_price: float = 0.0
        self.sorted_buy_grids: List[float] = []
        self.sorted_sell_grids: List[float] = []
        self.grid_levels: Dict[float, GridLevel] = {}
        self._sorted_prices: List[float] = []
        self._price_index_map: Dict[float, int] = {}

    def _create_grid_strategy(self) -> BaseGridStrategy:
        if self.strategy_type == GridStrategyType.SIMPLE_GRID:
            return SimpleGridStrategy()
        elif self.strategy_type == GridStrategyType.HEDGED_GRID:
            return HedgedGridStrategy()
        else:
            raise ValueError(f"Unsupported strategy type: {self.strategy_type}")

    def initialize_grids(self) -> None:
        """Initialize grid levels and assign states based on strategy."""
        self.price_grids, self.central_price = GridSpacingStrategy.calculate_grids(
            self.bottom_range,
            self.top_range,
            self.num_grids,
            self.spacing_type,
        )

        self.sorted_buy_grids, self.sorted_sell_grids, self.grid_levels = (
            self.grid_strategy.initialize_levels(self.price_grids, self.central_price)
        )

        self._sorted_prices = sorted(self.price_grids)
        self._price_index_map = {p: i for i, p in enumerate(self._sorted_prices)}

        # Pair levels for hedged grid
        if self.strategy_type == GridStrategyType.HEDGED_GRID:
            self._pair_grid_levels()

    def _pair_grid_levels(self) -> None:
        """Pair buy and sell levels for hedged grid strategy."""
        for price, level in self.grid_levels.items():
            if level.state == GridCycleState.READY_TO_BUY:
                paired_sell = self.get_paired_sell_level(level)
                if paired_sell:
                    level.paired_sell_level = paired_sell
                    paired_sell.paired_buy_level = level

    def get_trigger_price(self) -> float:
        """Get the trigger price for grid initialization."""
        return self.central_price

    def get_order_size_for_grid_level(
        self,
        total_balance: float,
        current_price: float,
        side: str,
    ) -> float:
        """
        Calculate order size for a grid level.

        Args:
            total_balance: Total portfolio value in quote currency
            current_price: Current price of the trading pair
            side: 'buy' or 'sell'

        Returns:
            Order size in base currency
        """
        total_grids = len(self.grid_levels)
        base_size = total_balance / total_grids / current_price
        ratio = self.buy_ratio if side == "buy" else self.sell_ratio
        return base_size * ratio

    def get_paired_sell_level(self, buy_grid_level: GridLevel) -> Optional[GridLevel]:
        """Get paired sell level for a buy level."""
        return self.grid_strategy.get_paired_sell_level(
            buy_grid_level,
            self.grid_levels,
            self.sorted_sell_grids,
        )

    def get_grid_level_below(self, grid_level: GridLevel) -> Optional[GridLevel]:
        """Get the grid level immediately below the given level."""
        current_index = self._price_index_map.get(grid_level.price)
        if current_index is not None and current_index > 0:
            lower_price = self._sorted_prices[current_index - 1]
            return self.grid_levels[lower_price]
        return None

    def mark_order_pending(
        self,
        grid_level: GridLevel,
        order_side: str,
    ) -> None:
        """Mark a grid level as having a pending order."""
        if order_side == "buy":
            grid_level.state = GridCycleState.WAITING_FOR_BUY_FILL
        elif order_side == "sell":
            grid_level.state = GridCycleState.WAITING_FOR_SELL_FILL

    def complete_order(
        self,
        grid_level: GridLevel,
        order_side: str,
    ) -> None:
        """Mark order completion and transition grid level state."""
        self.grid_strategy.complete_order(grid_level, order_side)

    def can_place_order(self, grid_level: GridLevel, order_side: str) -> bool:
        """Check if an order can be placed on the grid level."""
        return self.grid_strategy.can_place_order(grid_level, order_side)

    def get_grid_info(self) -> Dict[str, Any]:
        """Get comprehensive grid information."""
        return {
            "central_price": self.central_price,
            "num_grids": self.num_grids,
            "spacing_type": self.spacing_type.value,
            "strategy_type": self.strategy_type.value,
            "price_range": (self.bottom_range, self.top_range),
            "price_grids": self.price_grids,
            "buy_grids": self.sorted_buy_grids,
            "sell_grids": self.sorted_sell_grids,
        }


@dataclass
class GridTradingParams:
    """Parameters for grid trading strategy."""
    bottom_range: float = 0.0
    top_range: float = 0.0
    num_grids: int = 10
    spacing_type: GridSpacingType = GridSpacingType.ARITHMETIC
    strategy_type: GridStrategyType = GridStrategyType.SIMPLE_GRID
    buy_ratio: float = 1.0
    sell_ratio: float = 1.0
    take_profit_threshold: Optional[float] = None
    stop_loss_threshold: Optional[float] = None
    initial_balance: float = 10000.0
    trading_fee: float = 0.001


class GridTradingStrategy:
    """
    Grid Trading Strategy with support for:

    - Simple Grid: Buy below central price, sell above
    - Hedged Grid: Long and short simultaneously at paired levels
    - Arithmetic Spacing: Equal price intervals
    - Geometric Spacing: Equal percentage intervals
    - Backtesting with historical OHLCV data

    This is a self-contained strategy class that works with the quant_trading backtest engine.
    """

    name = "grid_trading"

    def __init__(
        self,
        symbol: str,
        params: Optional[GridTradingParams] = None,
    ):
        self.symbol = symbol
        self.params = params or GridTradingParams()

        # Initialize grid manager
        self.grid_manager: Optional[GridManager] = None
        self._initialized = False
        self._triggered = False

        # Balance tracking
        self.fiat_balance: float = self.params.initial_balance
        self.crypto_balance: float = 0.0
        self.total_fees: float = 0.0

        # Order tracking
        self.pending_orders: List[Dict[str, Any]] = []
        self.filled_orders: List[Dict[str, Any]] = []
        self.grid_orders_initialized = False

        # Backtest data
        self.data: Optional[pd.DataFrame] = None
        self.account_values: List[float] = []

    def initialize(self, current_price: float) -> None:
        """Initialize the strategy with current market price."""
        if self._initialized:
            return

        # Set price range based on current price if not provided
        if self.params.bottom_range == 0 and self.params.top_range == 0:
            range_percent = 0.1  # 10% range
            self.params.bottom_range = current_price * (1 - range_percent)
            self.params.top_range = current_price * (1 + range_percent)

        # Create and initialize grid manager
        self.grid_manager = GridManager(
            bottom_range=self.params.bottom_range,
            top_range=self.params.top_range,
            num_grids=self.params.num_grids,
            spacing_type=self.params.spacing_type,
            strategy_type=self.params.strategy_type,
            buy_ratio=self.params.buy_ratio,
            sell_ratio=self.params.sell_ratio,
        )
        self.grid_manager.initialize_grids()
        self._initialized = True

    def trigger_initialization(self, current_price: float) -> List[Dict[str, Any]]:
        """
        Trigger initial grid order placement when trigger price is crossed.

        Returns:
            List of initial orders to place
        """
        if self.grid_orders_initialized:
            return []

        orders = []
        trigger_price = self.grid_manager.get_trigger_price()

        # Initial buy at trigger price
        buy_quantity = self.fiat_balance / current_price * 0.5  # 50% of balance
        orders.append({
            "side": "buy",
            "price": current_price,
            "quantity": buy_quantity,
            "type": "market",
            "grid_level": trigger_price,
        })

        # Place initial grid orders
        for grid_price, level in self.grid_manager.grid_levels.items():
            if self.grid_manager.can_place_order(level, "buy"):
                size = self.grid_manager.get_order_size_for_grid_level(
                    self.fiat_balance, grid_price, "buy"
                )
                orders.append({
                    "side": "buy",
                    "price": grid_price,
                    "quantity": size,
                    "type": "limit",
                    "grid_level": grid_price,
                })
                level.state = GridCycleState.WAITING_FOR_BUY_FILL

            elif self.grid_manager.can_place_order(level, "sell"):
                size = self.grid_manager.get_order_size_for_grid_level(
                    self.fiat_balance, grid_price, "sell"
                )
                orders.append({
                    "side": "sell",
                    "price": grid_price,
                    "quantity": size,
                    "type": "limit",
                    "grid_level": grid_price,
                })
                level.state = GridCycleState.WAITING_FOR_SELL_FILL

        self.grid_orders_initialized = True
        return orders

    def simulate_order_fill(
        self,
        high_price: float,
        low_price: float,
        timestamp: Any,
    ) -> List[Dict[str, Any]]:
        """
        Simulate order fills based on high-low price range.

        Args:
            high_price: Highest price in the period
            low_price: Lowest price in the period
            timestamp: Current timestamp

        Returns:
            List of filled orders
        """
        if not self.grid_orders_initialized:
            return []

        filled = []

        for order in list(self.pending_orders):
            order_price = order["price"]
            order_side = order["side"]

            # Check if order price was crossed
            if order_side == "buy" and low_price <= order_price <= high_price:
                order["filled"] = True
                order["fill_price"] = order_price
                order["fill_time"] = timestamp
                self.pending_orders.remove(order)
                filled.append(order)
                self._process_filled_order(order)

            elif order_side == "sell" and low_price <= order_price <= high_price:
                order["filled"] = True
                order["fill_price"] = order_price
                order["fill_time"] = timestamp
                self.pending_orders.remove(order)
                filled.append(order)
                self._process_filled_order(order)

        return filled

    def _process_filled_order(self, order: Dict[str, Any]) -> None:
        """Process a filled order and update balances."""
        price = order["fill_price"]
        quantity = order["quantity"]
        side = order["side"]
        fee = price * quantity * self.params.trading_fee

        self.total_fees += fee

        if side == "buy":
            self.fiat_balance -= price * quantity + fee
            self.crypto_balance += quantity
            # Transition grid level
            for grid_price, level in self.grid_manager.grid_levels.items():
                if abs(grid_price - price) < 1e-8:
                    self.grid_manager.complete_order(level, "buy")
                    break

        elif side == "sell":
            self.fiat_balance += price * quantity - fee
            self.crypto_balance -= quantity
            # Transition grid level
            for grid_price, level in self.grid_manager.grid_levels.items():
                if abs(grid_price - price) < 1e-8:
                    self.grid_manager.complete_order(level, "sell")
                    break

        self.filled_orders.append(order)

    def place_order(self, order: Dict[str, Any]) -> None:
        """Place a new order."""
        order["filled"] = False
        self.pending_orders.append(order)

    def check_take_profit_stop_loss(self, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Check if take profit or stop loss is triggered.

        Returns:
            TP/SL order if triggered, None otherwise
        """
        if self.params.take_profit_threshold is not None:
            if current_price >= self.params.take_profit_threshold:
                return {
                    "side": "sell",
                    "price": current_price,
                    "quantity": self.crypto_balance,
                    "type": "market",
                    "reason": "take_profit",
                }

        if self.params.stop_loss_threshold is not None:
            if current_price <= self.params.stop_loss_threshold:
                return {
                    "side": "sell",
                    "price": current_price,
                    "quantity": self.crypto_balance,
                    "type": "market",
                    "reason": "stop_loss",
                }

        return None

    def get_total_value(self, current_price: float) -> float:
        """Get total portfolio value in quote currency."""
        return self.fiat_balance + self.crypto_balance * current_price

    def run_backtest(
        self,
        data: pd.DataFrame,
        skip_initial_purchase: bool = False,
    ) -> pd.DataFrame:
        """
        Run backtest on historical OHLCV data.

        Args:
            data: DataFrame with columns [timestamp, open, high, low, close, volume]
            skip_initial_purchase: Skip initial purchase on trigger

        Returns:
            DataFrame with account_value column added
        """
        self.data = data.copy()
        self.data["account_value"] = np.nan

        close_prices = self.data["close"].values
        high_prices = self.data["high"].values
        low_prices = self.data["low"].values
        timestamps = self.data.index

        initial_price = close_prices[0]
        self.initialize(initial_price)

        # Record initial account value
        self.data.loc[timestamps[0], "account_value"] = self.get_total_value(initial_price)

        last_price = None
        trigger_price = self.grid_manager.get_trigger_price()

        for i, (current_price, high_price, low_price, timestamp) in enumerate(
            zip(close_prices, high_prices, low_prices, timestamps, strict=False)
        ):
            # Check if trigger price is crossed on first candle
            if not self._triggered and last_price is not None:
                if (last_price <= trigger_price <= current_price) or (last_price == trigger_price):
                    if not skip_initial_purchase:
                        initial_buy = {
                            "side": "buy",
                            "price": current_price,
                            "quantity": self.fiat_balance / current_price * 0.5,
                            "type": "market",
                        }
                        self._process_filled_order(initial_buy)

                    orders = self.trigger_initialization(current_price)
                    for order in orders:
                        if order["type"] == "limit":
                            self.place_order(order)
                    self._triggered = True

            if not self._triggered:
                self.data.loc[timestamp, "account_value"] = self.get_total_value(current_price)
                last_price = current_price
                continue

            # Simulate order fills
            self.simulate_order_fill(high_price, low_price, timestamp)

            # Check TP/SL
            tp_sl = self.check_take_profit_stop_loss(current_price)
            if tp_sl is not None:
                self._process_filled_order(tp_sl)
                break

            self.data.loc[timestamp, "account_value"] = self.get_total_value(current_price)
            last_price = current_price

        return self.data

    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from backtest results."""
        if self.data is None:
            return {}

        initial_value = self.data["account_value"].iloc[0]
        final_value = self.data["account_value"].iloc[-1]
        roi = (final_value - initial_value) / initial_value * 100

        # Calculate max drawdown
        peak = self.data["account_value"].expanding(min_periods=1).max()
        drawdown = (peak - self.data["account_value"]) / peak * 100
        max_drawdown = drawdown.max()

        # Calculate max runup
        trough = self.data["account_value"].expanding(min_periods=1).min()
        runup = (self.data["account_value"] - trough) / trough * 100
        max_runup = runup.max()

        # Time in profit
        time_in_profit = (self.data["account_value"] > initial_value).mean() * 100

        return {
            "symbol": self.symbol,
            "strategy": self.name,
            "initial_value": initial_value,
            "final_value": final_value,
            "roi_percent": roi,
            "max_drawdown_percent": max_drawdown,
            "max_runup_percent": max_runup,
            "time_in_profit_percent": time_in_profit,
            "total_fees": self.total_fees,
            "num_trades": len(self.filled_orders),
            "final_fiat_balance": self.fiat_balance,
            "final_crypto_balance": self.crypto_balance,
        }

    def get_grid_info(self) -> Dict[str, Any]:
        """Get current grid information."""
        if self.grid_manager is None:
            return {}
        return self.grid_manager.get_grid_info()

    def reset(self) -> None:
        """Reset strategy state for new backtest."""
        self._initialized = False
        self._triggered = False
        self.grid_orders_initialized = False
        self.fiat_balance = self.params.initial_balance
        self.crypto_balance = 0.0
        self.total_fees = 0.0
        self.pending_orders = []
        self.filled_orders = []
        self.data = None
        self.account_values = []
        self.grid_manager = None
