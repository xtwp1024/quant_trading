"""
Dynamic Grid Trading Strategy / 动态网格交易策略

核心特性 / Core Features:
1. 价格区间和网格数量自适应 / Adaptive price range and grid count
2. 波动率感知网格调整 / Volatility-aware grid adjustment
3. 自动再平衡机制 / Automatic rebalancing mechanism
4. 多币种网格支持 / Multi-currency grid support

基于 Binance Grid Trader 核心算法重构 / Refactored from Binance Grid Trader core algorithms.
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
import math


class RebalanceAction(Enum):
    """Rebalance action types / 再平衡操作类型"""
    NONE = "none"
    REBALANCE = "rebalance"      # Full rebalance needed / 需要完全再平衡
    ADJUST = "adjust"            # Grid count adjustment / 网格数量调整


@dataclass
class GridLevel:
    """Single grid level / 单个网格层级"""
    price: float
    buy_orders: list[str] = field(default_factory=list)
    sell_orders: list[str] = field(default_factory=list)
    filled_buy: bool = False
    filled_sell: bool = False


@dataclass
class GridInfo:
    """Current grid state / 当前网格状态"""
    symbol: str
    lower_price: float
    upper_price: float
    grid_count: int
    step_price: float
    current_price: float
    grid_levels: list[float]
    total_trades: int
    unrealized_pnl: float
    realized_pnl: float


class DynamicGridStrategy:
    """Dynamic Grid Strategy — Adaptive price range and grid count.

    动态网格策略 — 价格区间和网格数量自适应.

    特点 / Features:
    1. 价格触及边界时自动再平衡 / Auto rebalance when price hits boundaries
    2. 波动率增大时扩大网格 / Expand grids when volatility increases
    3. 波动率减小时缩小网格 / Shrink grids when volatility decreases
    4. 支持网格数量动态调整 / Dynamic grid count adjustment
    5. 支持限价单推送模式 / Limit order push model

    算法来源 / Algorithm Source:
    - Binance Grid Trader (spot_grid_strategy.py, future_grid_strategy.py)
    - GridPositionCalculator for average price computation
    """

    def __init__(
        self,
        symbol: str,
        initial_lower: float,
        initial_upper: float,
        initial_grids: int = 10,
        order_size_pct: float = 0.01,
        auto_rebalance: bool = True,
        rebalance_threshold: float = 0.1,  # 10% deviation triggers rebalance
        vol_adaptive: bool = True,
        vol_window: int = 20,
        vol_expansion_factor: float = 1.5,
        vol_shrink_factor: float = 0.7,
        min_grids: int = 5,
        max_grids: int = 200,
        price_tick: float = 0.01,
    ):
        """Initialize dynamic grid strategy.

        参数 / Parameters:
            symbol: Trading pair symbol / 交易对符号
            initial_lower: Initial lower price boundary / 初始价格下界
            initial_upper: Initial upper price boundary / 初始价格上界
            initial_grids: Initial grid count / 初始网格数量
            order_size_pct: Order size as percentage of capital / 订单大小占资金比例
            auto_rebalance: Enable auto rebalance at boundaries / 是否在边界自动再平衡
            rebalance_threshold: Deviation ratio to trigger rebalance / 偏离比例触发再平衡
            vol_adaptive: Enable volatility-adaptive grid sizing / 启用波动率自适应网格
            vol_window: Rolling window for volatility calculation / 波动率计算窗口
            vol_expansion_factor: Grid expansion factor on high vol / 高波动时网格扩大系数
            vol_shrink_factor: Grid shrink factor on low vol / 低波动时网格缩小系数
            min_grids: Minimum grid count / 最小网格数
            max_grids: Maximum grid count / 最大网格数
            price_tick: Minimum price increment / 价格最小变动单位
        """
        self.symbol = symbol
        self.initial_lower = initial_lower
        self.initial_upper = initial_upper
        self.lower_price = initial_lower
        self.upper_price = initial_upper
        self.grid_count = initial_grids
        self.order_size_pct = order_size_pct
        self.auto_rebalance = auto_rebalance
        self.rebalance_threshold = rebalance_threshold
        self.vol_adaptive = vol_adaptive
        self.vol_window = vol_window
        self.vol_expansion_factor = vol_expansion_factor
        self.vol_shrink_factor = vol_shrink_factor
        self.min_grids = min_grids
        self.max_grids = max_grids
        self.price_tick = price_tick

        # Internal state / 内部状态
        self._current_price: float = (initial_lower + initial_upper) / 2
        self._step_price: float = (initial_upper - initial_lower) / initial_grids
        self._grid_levels: list[GridLevel] = []
        self._price_history: list[float] = []
        self._volatility: float = 0.0
        self._total_trades: int = 0
        self._realized_pnl: float = 0.0
        self._position: float = 0.0  # Net position / 净持仓
        self._avg_price: float = 0.0  # Average fill price / 平均成交价
        self._long_orders: dict[str, float] = {}  # order_id -> price
        self._short_orders: dict[str, float] = {}  # order_id -> price

        self._init_grid_levels()

    def _init_grid_levels(self) -> None:
        """Initialize grid levels / 初始化网格层级."""
        self._grid_levels = []
        step = (self.upper_price - self.lower_price) / self.grid_count
        for i in range(self.grid_count + 1):
            price = self._round_price(self.lower_price + i * step)
            self._grid_levels.append(GridLevel(price=price))

    def _round_price(self, price: float) -> float:
        """Round price to tick size / 价格取整到tick."""
        return round(price / self.price_tick) * self.price_tick

    def _compute_step_price(self) -> float:
        """Compute step price between grids / 计算网格间距."""
        return (self.upper_price - self.lower_price) / self.grid_count

    def _calculate_volatility(self, prices: list[float]) -> float:
        """Calculate rolling volatility (standard deviation) / 计算滚动波动率.

        Args:
            prices: Price series / 价格序列

        Returns:
            Volatility as coefficient of variation / 波动率（变异系数）
        """
        if len(prices) < 2:
            return 0.0
        mean = sum(prices) / len(prices)
        if mean == 0:
            return 0.0
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        return math.sqrt(variance) / mean

    def _estimate_new_grid_count(self) -> int:
        """Estimate new grid count based on volatility / 基于波动率估算网格数.

        Returns:
            Estimated grid count / 估算的网格数量
        """
        if not self.vol_adaptive or len(self._price_history) < self.vol_window:
            return self.grid_count

        vol = self._calculate_volatility(
            self._price_history[-self.vol_window:]
        )

        # Simple volatility-adaptive: higher vol -> fewer grids
        # 简化波动率自适应：波动率高时减少网格数
        current_range_pct = (self.upper_price - self.lower_price) / self._current_price

        if vol > self._volatility * self.vol_expansion_factor:
            # Volatility increased significantly -> expand grids (narrower spacing)
            # 波动率显著上升 -> 扩大网格（间距变小）
            new_count = int(self.grid_count * self.vol_expansion_factor)
        elif vol < self._volatility * self.vol_shrink_factor:
            # Volatility decreased -> shrink grids (wider spacing)
            # 波动率下降 -> 缩小网格（间距变大）
            new_count = int(self.grid_count * self.vol_shrink_factor)
        else:
            new_count = self.grid_count

        self._volatility = vol
        return max(self.min_grids, min(self.max_grids, new_count))

    def set_price_range(self, lower: float, upper: float) -> None:
        """Set new price range / 设置新的价格区间.

        Args:
            lower: New lower price / 新的价格下界
            upper: New upper price / 新的价格上界
        """
        if lower >= upper:
            raise ValueError(f"Lower price {lower} must be less than upper {upper}")

        self.lower_price = lower
        self.upper_price = upper
        self._step_price = self._compute_step_price()

        # Recompute grid levels / 重新计算网格层级
        self._init_grid_levels()

    def adjust_grids(self, new_count: int) -> list[dict]:
        """Adjust grid count — returns orders to cancel/place.

        调整网格数量 — 返回需要取消/放置的订单.

        Args:
            new_count: New grid count / 新的网格数量

        Returns:
            List of order actions: {action: 'cancel'|'place', side: 'buy'|'sell', price: float}
            订单操作列表
        """
        new_count = max(self.min_grids, min(self.max_grids, new_count))

        if new_count == self.grid_count:
            return []

        old_count = self.grid_count
        self.grid_count = new_count
        self._step_price = self._compute_step_price()

        actions: list[dict] = []

        # Cancel all existing orders / 取消所有现有订单
        for order_id in list(self._long_orders.keys()):
            actions.append({"action": "cancel", "order_id": order_id, "side": "buy"})
        for order_id in list(self._short_orders.keys()):
            actions.append({"action": "cancel", "order_id": order_id, "side": "sell"})

        self._long_orders.clear()
        self._short_orders.clear()

        # Recalculate current grid position / 重新计算当前网格位置
        mid_count = self._get_grid_index(self._current_price)

        # Place new orders around current price / 在当前价格周围放置新订单
        grids_to_place = min(5, new_count)  # Place 5 orders each side by default

        for i in range(1, grids_to_place + 1):
            # Buy orders below current price / 当前价格以下的买单
            buy_idx = mid_count - i
            if 0 <= buy_idx <= new_count:
                price = self._round_price(self.lower_price + buy_idx * self._step_price)
                actions.append({"action": "place", "side": "buy", "price": price})

            # Sell orders above current price / 当前价格以上的卖单
            sell_idx = mid_count + i
            if 0 <= sell_idx <= new_count:
                price = self._round_price(self.lower_price + sell_idx * self._step_price)
                actions.append({"action": "place", "side": "sell", "price": price})

        # Re-init grid levels / 重新初始化网格层级
        self._init_grid_levels()

        return actions

    def _get_grid_index(self, price: float) -> int:
        """Get grid index for a price / 获取价格对应的网格索引."""
        if price < self.lower_price or price > self.upper_price:
            return -1
        return int((price - self.lower_price) / self._step_price)

    def recompute_on_price_move(self, current_price: float) -> dict:
        """Recompute strategy on price move.

        价格变动时重新计算.

        This is the core algorithm from Binance Grid Trader's on_tick() method,
        adapted for REST-only architecture.

        Args:
            current_price: Current market price / 当前市场价格

        Returns:
            {action: 'rebalance' | 'adjust' | 'none', orders: list}
        """
        self._current_price = current_price

        # Update price history for volatility calculation
        self._price_history.append(current_price)
        if len(self._price_history) > self.vol_window * 2:
            self._price_history = self._price_history[-self.vol_window * 2:]

        # Check boundary conditions / 检查边界条件
        if current_price <= self.lower_price or current_price >= self.upper_price:
            if self.auto_rebalance:
                # Price hit boundary - need rebalance
                # 价格触及边界 - 需要再平衡
                return {
                    "action": RebalanceAction.REBALANCE.value,
                    "orders": self._generate_boundary_orders(current_price),
                }

        # Check deviation threshold / 检查偏离阈值
        mid_price = (self.lower_price + self.upper_price) / 2
        deviation = abs(current_price - mid_price) / mid_price

        if deviation > self.rebalance_threshold:
            if self.vol_adaptive:
                # Estimate new grid count / 估算新网格数
                new_count = self._estimate_new_grid_count()
                if new_count != self.grid_count:
                    return {
                        "action": RebalanceAction.ADJUST.value,
                        "orders": self.adjust_grids(new_count),
                    }

        # Generate normal grid orders / 生成普通网格订单
        orders = self._generate_grid_orders(current_price)

        return {
            "action": RebalanceAction.NONE.value,
            "orders": orders,
        }

    def _generate_boundary_orders(self, current_price: float) -> list[dict]:
        """Generate orders when price hits boundary / 边界触发时生成订单."""
        orders: list[dict] = []

        # Cancel all existing orders / 取消所有现有订单
        for order_id in list(self._long_orders.keys()):
            orders.append({"action": "cancel", "order_id": order_id, "side": "buy"})
        for order_id in list(self._short_orders.keys()):
            orders.append({"action": "cancel", "order_id": order_id, "side": "sell"})

        self._long_orders.clear()
        self._short_orders.clear()

        # Shift the price range to re-center on current price
        # 平移价格区间到以当前价格为中心
        range_half = (self.upper_price - self.lower_price) / 2
        new_lower = current_price - range_half
        new_upper = current_price + range_half

        self.set_price_range(new_lower, new_upper)

        # Place new orders around current price / 在当前价格周围放置新订单
        mid_count = self._get_grid_index(current_price)

        for i in range(1, 6):
            buy_idx = mid_count - i
            if 0 <= buy_idx <= self.grid_count:
                price = self._round_price(self.lower_price + buy_idx * self._step_price)
                orders.append({"action": "place", "side": "buy", "price": price})

            sell_idx = mid_count + i
            if 0 <= sell_idx <= self.grid_count:
                price = self._round_price(self.lower_price + sell_idx * self._step_price)
                orders.append({"action": "place", "side": "sell", "price": price})

        return orders

    def _generate_grid_orders(self, current_price: float) -> list[dict]:
        """Generate normal grid orders around current price / 在当前价格周围生成普通网格订单."""
        orders: list[dict] = []

        # If no long orders, place buy orders below current price
        # 如果没有买单，在当前价格下方放置买单
        if not self._long_orders:
            mid_count = self._get_grid_index(current_price)
            for i in range(1, 6):
                idx = mid_count - i
                if 0 <= idx <= self.grid_count:
                    price = self._round_price(self.lower_price + idx * self._step_price)
                    orders.append({"action": "place", "side": "buy", "price": price})

        # If no short orders, place sell orders above current price
        # 如果没有卖单，在当前价格上方放置卖单
        if not self._short_orders:
            mid_count = self._get_grid_index(current_price)
            for i in range(1, 6):
                idx = mid_count + i
                if 0 <= idx <= self.grid_count:
                    price = self._round_price(self.lower_price + idx * self._step_price)
                    orders.append({"action": "place", "side": "sell", "price": price})

        return orders

    def get_grid_info(self) -> dict:
        """Get current grid information / 获取当前网格信息.

        Returns:
            Grid state dict / 网格状态字典
        """
        grid_levels = [gl.price for gl in self._grid_levels]

        # Calculate unrealized PnL / 计算未实现盈亏
        unrealized_pnl = 0.0
        if self._position != 0 and self._avg_price > 0:
            unrealized_pnl = (self._current_price - self._avg_price) * self._position

        return {
            "symbol": self.symbol,
            "lower_price": self.lower_price,
            "upper_price": self.upper_price,
            "grid_count": self.grid_count,
            "step_price": self._step_price,
            "current_price": self._current_price,
            "grid_levels": grid_levels,
            "total_trades": self._total_trades,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "position": self._position,
            "avg_price": self._avg_price,
            "volatility": self._volatility,
            "long_orders_count": len(self._long_orders),
            "short_orders_count": len(self._short_orders),
        }

    def compute_profit(self, price_history: list[float]) -> dict:
        """Compute cumulative grid profit from price history.

        计算网格累计利润.

        Simulates grid trading on historical prices to compute total profit.
        基于历史价格模拟网格交易计算总利润.

        Args:
            price_history: Historical price series / 历史价格序列

        Returns:
            {total_profit, trade_count, buy_count, sell_count, max_drawdown}
        """
        if len(price_history) < 2:
            return {
                "total_profit": 0.0,
                "trade_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "max_drawdown": 0.0,
            }

        total_profit = 0.0
        trade_count = 0
        buy_count = 0
        sell_count = 0
        position = 0.0
        entry_price = 0.0
        max_drawdown = 0.0
        peak = 0.0

        step = (self.upper_price - self.lower_price) / self.grid_count

        for i, price in enumerate(price_history):
            # Determine which grid level we're at / 确定当前网格层级
            if price < self.lower_price or price > self.upper_price:
                continue

            grid_idx = int((price - self.lower_price) / step)

            # Check for buy trigger (price crosses below a grid line)
            # 检查买入触发（价格下穿网格线）
            if i > 0:
                prev_price = price_history[i - 1]
                prev_idx = int((prev_price - self.lower_price) / step)

                if grid_idx < prev_idx and position >= 0:
                    # Price moved down, buy at this grid / 价格下移，在此网格买入
                    position += 1
                    entry_price = (entry_price * (position - 1) + price) / position
                    buy_count += 1
                    trade_count += 1

                # Check for sell trigger (price crosses above a grid line)
                # 检查卖出触发（价格上穿网格线）
                elif grid_idx > prev_idx and position <= 0:
                    # Price moved up, sell at this grid / 价格上移，在此网格卖出
                    if position < 0:
                        profit = abs(position) * (price - abs(entry_price))
                        total_profit += profit
                    position -= 1
                    entry_price = price
                    sell_count += 1
                    trade_count += 1

            # Track peak and drawdown / 跟踪峰值和回撤
            peak = max(peak, total_profit)
            drawdown = peak - total_profit
            max_drawdown = max(max_drawdown, drawdown)

        # Close any remaining position / 平掉剩余仓位
        if position > 0:
            last_price = price_history[-1]
            total_profit += position * (last_price - entry_price)
        elif position < 0:
            last_price = price_history[-1]
            total_profit += abs(position) * (entry_price - last_price)

        return {
            "total_profit": total_profit,
            "trade_count": trade_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "max_drawdown": max_drawdown,
        }

    # Order management methods (for REST integration) / 订单管理方法（用于REST集成）
    def register_buy_order(self, order_id: str, price: float) -> None:
        """Register a buy order / 注册买单."""
        self._long_orders[order_id] = price

    def register_sell_order(self, order_id: str, price: float) -> None:
        """Register a sell order / 注册卖单."""
        self._short_orders[order_id] = price

    def on_order_filled(self, order_id: str, price: float, side: str, volume: float) -> dict:
        """Handle order fill event / 处理订单成交事件.

        Returns next orders to place / 返回后续需要放置的订单.

        Args:
            order_id: Filled order ID / 成交订单ID
            price: Fill price / 成交价格
            side: 'buy' or 'sell'
            volume: Fill volume / 成交数量

        Returns:
            {orders_to_place: list, pnl_update: float}
        """
        orders_to_place: list[dict] = []

        if side == "buy":
            if order_id in self._long_orders:
                del self._long_orders[order_id]

            # Update position / 更新持仓
            prev_pos = self._position
            self._position += volume

            if prev_pos <= 0 and self._position > 0:
                self._avg_price = price
            elif self._position > 0:
                self._avg_price = (self._avg_price * prev_pos + price * volume) / self._position

            self._total_trades += 1

            # Place opposite order at next grid up / 在上一个网格处放置反向订单
            next_price = self._round_price(price + self._step_price)
            if next_price <= self.upper_price:
                orders_to_place.append({"side": "sell", "price": next_price})

            # Place additional buy order below / 在下方放置额外买单
            count = len(self._long_orders) + 1
            buy_price = self._round_price(price - self._step_price * count)
            if buy_price >= self.lower_price:
                orders_to_place.append({"side": "buy", "price": buy_price})

        elif side == "sell":
            if order_id in self._short_orders:
                del self._short_orders[order_id]

            # Update position / 更新持仓
            prev_pos = self._position
            self._position -= volume

            if prev_pos >= 0 and self._position < 0:
                self._avg_price = price
            elif self._position < 0:
                self._avg_price = (self._avg_price * abs(prev_pos) + price * volume) / abs(self._position)

            self._total_trades += 1

            # Place opposite order at next grid down / 在下一个网格处放置反向订单
            next_price = self._round_price(price - self._step_price)
            if next_price >= self.lower_price:
                orders_to_place.append({"side": "buy", "price": next_price})

            # Place additional sell order above / 在上方放置额外卖单
            count = len(self._short_orders) + 1
            sell_price = self._round_price(price + self._step_price * count)
            if sell_price <= self.upper_price:
                orders_to_place.append({"side": "sell", "price": sell_price})

        return {"orders_to_place": orders_to_place}

    def on_order_cancelled(self, order_id: str) -> None:
        """Handle order cancellation / 处理订单取消."""
        if order_id in self._long_orders:
            del self._long_orders[order_id]
        elif order_id in self._short_orders:
            del self._short_orders[order_id]


__all__ = ["DynamicGridStrategy", "GridLevel", "GridInfo", "RebalanceAction"]
