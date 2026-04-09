"""
Multi-Currency Grid Manager / 多币种网格管理器

核心特性 / Core Features:
1. 多币种网格统一管理 / Unified multi-currency grid management
2. 资金分配与风控 / Capital allocation and risk control
3. 组合级别再平衡 / Portfolio-level rebalancing
4. 统一持仓与盈亏计算 / Unified position and PnL calculation

基于 Binance Grid Trader 架构扩展 / Extended from Binance Grid Trader architecture.
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field
import math


@dataclass
class GridCapitalAllocation:
    """Capital allocation for a single grid / 单个网格的资金分配"""
    symbol: str
    allocated_capital: float
    current_value: float = 0.0
    reserved_value: float = 0.0  # Reserved for pending orders / 预留用于待成交订单
    pnl: float = 0.0


@dataclass
class GridPortfolioStatus:
    """Overall portfolio status / 组合整体状态"""
    total_capital: float
    allocated_capital: float
    free_capital: float
    total_pnl: float
    total_trades: int
    grid_count: int
    allocations: dict[str, GridCapitalAllocation]


class GridManager:
    """Multi-currency grid manager / 多币种网格管理器.

    同时管理多个币种的网格策略，统一风控和资金分配.

    Features:
    1. Unified capital management / 统一资金管理
    2. Max concurrent grids limit / 最大并发网格限制
    3. Per-grid capital allocation / 单网格资金分配
    4. Portfolio-level rebalance / 组合级别再平衡
    5. Risk controls / 风控措施

    Usage Example / 使用示例:
        manager = GridManager(total_capital=10000, max_concurrent=5, capital_per_grid=1000)
        strategy = DynamicGridStrategy("BTCUSDT", lower=40000, upper=50000)
        manager.add_grid("BTCUSDT", strategy)
    """

    def __init__(
        self,
        total_capital: float,
        max_concurrent: int = 5,
        capital_per_grid: float = 1000.0,
        min_capital_reserve: float = 0.1,  # 10% reserve / 10%储备
    ):
        """Initialize grid manager / 初始化网格管理器.

        Args:
            total_capital: Total capital for all grids / 总资金
            max_concurrent: Maximum number of concurrent grids / 最大并发网格数
            capital_per_grid: Capital allocated per grid / 每个网格分配的资金
            min_capital_reserve: Minimum reserve ratio / 最小储备比例
        """
        self.total_capital = total_capital
        self.max_concurrent = max_concurrent
        self.capital_per_grid = capital_per_grid
        self.min_capital_reserve = min_capital_reserve

        # Grid strategies by symbol / 网格策略字典
        self.grids: dict[str, DynamicGridStrategy] = {}

        # Capital allocations / 资金分配
        self._allocations: dict[str, GridCapitalAllocation] = {}

        # Track reserved capital for pending orders / 跟踪待成交订单预留资金
        self._reserved_capital: float = 0.0

        # Portfolio statistics / 组合统计
        self._total_trades: int = 0
        self._realized_pnl: float = 0.0

    @property
    def concurrent_count(self) -> int:
        """Current number of active grids / 当前活跃网格数."""
        return len(self.grids)

    @property
    def free_capital(self) -> float:
        """Free capital available for new grids / 可用于新网格的自由资金."""
        return self.total_capital - self._get_allocated_capital() - self._reserved_capital

    def _get_allocated_capital(self) -> float:
        """Get total allocated capital / 获取已分配资金总额."""
        return sum(alloc.allocated_capital for alloc in self._allocations.values())

    def _can_add_grid(self, symbol: str) -> bool:
        """Check if a new grid can be added / 检查是否可以添加新网格."""
        if symbol in self.grids:
            return False
        if self.concurrent_count >= self.max_concurrent:
            return False
        if self.free_capital < self.capital_per_grid:
            return False
        # Check minimum reserve / 检查最低储备
        required = self.capital_per_grid + (self.capital_per_grid * self.min_capital_reserve)
        if self.total_capital - self._get_allocated_capital() < required:
            return False
        return True

    def add_grid(self, symbol: str, strategy: DynamicGridStrategy) -> bool:
        """Add a new symbol grid / 添加新币种网格.

        Args:
            symbol: Trading symbol / 交易对
            strategy: DynamicGridStrategy instance / 动态网格策略实例

        Returns:
            True if added successfully / 是否添加成功
        """
        if not self._can_add_grid(symbol):
            return False

        self.grids[symbol] = strategy
        self._allocations[symbol] = GridCapitalAllocation(
            symbol=symbol,
            allocated_capital=self.capital_per_grid,
        )
        return True

    def remove_grid(self, symbol: str) -> bool:
        """Remove a symbol grid / 移除币种网格.

        Args:
            symbol: Trading symbol / 交易对

        Returns:
            True if removed successfully / 是否移除成功
        """
        if symbol not in self.grids:
            return False

        strategy = self.grids[symbol]
        info = strategy.get_grid_info()

        # Update realized PnL / 更新已实现盈亏
        self._realized_pnl += info["realized_pnl"]
        self._total_trades += info["total_trades"]

        # Release reserved capital / 释放预留资金
        if symbol in self._allocations:
            alloc = self._allocations[symbol]
            self._reserved_capital -= alloc.reserved_value
            del self._allocations[symbol]

        del self.grids[symbol]
        return True

    def rebalance_all(self, prices: dict[str, float]) -> dict[str, list[dict]]:
        """Rebalance all grids based on current prices.

        对所有网格执行再平衡.

        Args:
            prices: Current prices for all symbols {symbol: price}
                    当前各币种价格

        Returns:
            {symbol: [orders]} - Orders needed for each symbol
            各币种需要的订单列表
        """
        all_orders: dict[str, list[dict]] = {}

        for symbol, strategy in self.grids.items():
            if symbol not in prices:
                continue

            current_price = prices[symbol]
            result = strategy.recompute_on_price_move(current_price)

            if result["orders"]:
                all_orders[symbol] = result["orders"]

            # Update allocation value / 更新分配价值
            if symbol in self._allocations:
                info = strategy.get_grid_info()
                self._allocations[symbol].current_value = (
                    abs(info["position"]) * current_price
                )
                self._allocations[symbol].pnl = (
                    info["realized_pnl"] + info["unrealized_pnl"]
                )

        return all_orders

    def get_portfolio_status(self) -> dict:
        """Get overall portfolio status / 获取组合状态.

        Returns:
            Portfolio status dict / 组合状态字典
        """
        allocations = {}
        total_pnl = 0.0
        total_trades = 0

        for symbol, strategy in self.grids.items():
            info = strategy.get_grid_info()
            alloc = self._allocations.get(symbol)

            total_pnl += info["realized_pnl"] + info["unrealized_pnl"]
            total_trades += info["total_trades"]

            allocations[symbol] = {
                "symbol": symbol,
                "allocated_capital": alloc.allocated_capital if alloc else 0,
                "current_value": abs(info["position"]) * info["current_price"],
                "position": info["position"],
                "realized_pnl": info["realized_pnl"],
                "unrealized_pnl": info["unrealized_pnl"],
                "total_pnl": info["realized_pnl"] + info["unrealized_pnl"],
                "grid_count": info["grid_count"],
                "volatility": info["volatility"],
                "long_orders": info["long_orders_count"],
                "short_orders": info["short_orders_count"],
            }

        return {
            "total_capital": self.total_capital,
            "allocated_capital": self._get_allocated_capital(),
            "free_capital": self.free_capital,
            "reserved_capital": self._reserved_capital,
            "total_pnl": total_pnl + self._realized_pnl,
            "total_trades": total_trades + self._total_trades,
            "grid_count": len(self.grids),
            "max_concurrent": self.max_concurrent,
            "concurrent_count": self.concurrent_count,
            "allocations": allocations,
        }

    def get_grid_info(self, symbol: str) -> Optional[dict]:
        """Get grid info for a specific symbol / 获取指定币种的网格信息.

        Args:
            symbol: Trading symbol / 交易对

        Returns:
            Grid info dict or None / 网格信息字典或None
        """
        if symbol not in self.grids:
            return None
        return self.grids[symbol].get_grid_info()

    def reserve_capital(self, symbol: str, amount: float) -> bool:
        """Reserve capital for pending orders / 预留资金用于待成交订单.

        Args:
            symbol: Trading symbol / 交易对
            amount: Amount to reserve / 预留金额

        Returns:
            True if successful / 是否成功
        """
        if symbol not in self._allocations:
            return False

        if amount > self.free_capital:
            return False

        self._allocations[symbol].reserved_value += amount
        self._reserved_capital += amount
        return True

    def release_capital(self, symbol: str, amount: float) -> bool:
        """Release reserved capital / 释放预留资金.

        Args:
            symbol: Trading symbol / 交易对
            amount: Amount to release / 释放金额

        Returns:
            True if successful / 是否成功
        """
        if symbol not in self._allocations:
            return False

        alloc = self._allocations[symbol]
        release_amt = min(amount, alloc.reserved_value)

        alloc.reserved_value -= release_amt
        self._reserved_capital -= release_amt
        return True

    def update_price(self, symbol: str, price: float) -> Optional[dict]:
        """Update price for a symbol and get recompute result.

        更新币种价格并获取重计算结果.

        Args:
            symbol: Trading symbol / 交易对
            price: Current price / 当前价格

        Returns:
            Recompute result dict or None / 重计算结果字典或None
        """
        if symbol not in self.grids:
            return None

        return self.grids[symbol].recompute_on_price_move(price)

    def handle_order_update(
        self,
        symbol: str,
        order_id: str,
        price: float,
        side: str,
        volume: float,
        status: str,  # 'filled', 'cancelled'
    ) -> Optional[list[dict]]:
        """Handle order update for a grid symbol.

        处理网格订单更新.

        Args:
            symbol: Trading symbol / 交易对
            order_id: Order ID / 订单ID
            price: Order price / 订单价格
            side: 'buy' or 'sell' / 买卖方向
            volume: Order volume / 订单数量
            status: Order status / 订单状态

        Returns:
            New orders to place or None / 新订单列表或None
        """
        if symbol not in self.grids:
            return None

        strategy = self.grids[symbol]

        if status == "filled":
            result = strategy.on_order_filled(order_id, price, side, volume)
            # Update reserved capital on fill / 成交时更新预留资金
            if side == "buy":
                self.release_capital(symbol, price * volume)
            return result.get("orders_to_place")

        elif status == "cancelled":
            strategy.on_order_cancelled(order_id)
            # Release reserved capital / 释放预留资金
            if side == "buy":
                self.release_capital(symbol, price * volume)
            return []

        return None

    def compute_portfolio_profit(
        self,
        price_histories: dict[str, list[float]],
    ) -> dict:
        """Compute profit for all grids from historical prices.

        从历史价格计算所有网格的利润.

        Args:
            price_histories: {symbol: [prices]} / 各币种历史价格

        Returns:
            {total_profit, by_symbol: {symbol: profit_info}}
        """
        total_profit = 0.0
        total_trades = 0
        by_symbol = {}

        for symbol, prices in price_histories.items():
            if symbol not in self.grids:
                continue

            strategy = self.grids[symbol]
            result = strategy.compute_profit(prices)

            total_profit += result["total_profit"]
            total_trades += result["trade_count"]

            by_symbol[symbol] = result

        return {
            "total_profit": total_profit,
            "total_trades": total_trades,
            "by_symbol": by_symbol,
        }

    def set_price_range(self, symbol: str, lower: float, upper: float) -> bool:
        """Set price range for a specific grid / 设置指定网格的价格区间.

        Args:
            symbol: Trading symbol / 交易对
            lower: Lower price / 价格下界
            upper: Upper price / 价格上界

        Returns:
            True if successful / 是否成功
        """
        if symbol not in self.grids:
            return False
        self.grids[symbol].set_price_range(lower, upper)
        return True

    def adjust_grids(self, symbol: str, new_count: int) -> Optional[list[dict]]:
        """Adjust grid count for a specific symbol / 调整指定网格的数量.

        Args:
            symbol: Trading symbol / 交易对
            new_count: New grid count / 新网格数量

        Returns:
            Orders to cancel/place or None / 取消/放置的订单或None
        """
        if symbol not in self.grids:
            return None
        return self.grids[symbol].adjust_grids(new_count)


__all__ = ["GridManager", "GridCapitalAllocation", "GridPortfolioStatus"]
