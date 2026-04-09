# -*- coding: utf-8 -*-
"""
GridHODL策略模块 / GridHODL Strategy Module

网格+Holder结合策略 — Grid + Holder combined strategy.

策略逻辑 / Strategy Logic:
1. 在预设价格区间内置若干网格 / Place grids within preset price range
2. 价格每触达一个网格, 进行低买高卖 / Buy low at each grid touch
3. 剩余仓位长期持有 (HODL) / Keep remaining position as long-term HODL
4. 波动越小, 网格收益越高 / Smaller volatility = higher grid profit

特点 / Features:
- 熊市: 网格持续低买积累仓位 / Bear: grid accumulates on dips
- 牛市: 网格高卖兑现利润, 底部仓位HODL / Bull: grid sells profits, HODL bottom

Ref: kraken-infinity-grid (GridHODL)
    https://github.com/btschwertfeger/kraken-infinity-grid
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_trading.signal import Signal, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams


__all__ = [
    "GridHODLParams",
    "GridHODLStrategy",
    "GridOrder",
]


@dataclass
class GridOrder:
    """单网格订单 / Single grid order."""
    price: float
    volume: float
    side: str  # "buy" or "sell"
    txid: Optional[str] = None
    is_hodl: bool = False  # 是否为HODL仓位 / Whether this is HODL position


@dataclass
class GridHODLParams(StrategyParams):
    """GridHODL策略参数 / GridHODL strategy parameters."""
    lower_price: float = 40000.0     # 网格价格下限 / Lower price boundary
    upper_price: float = 60000.0     # 网格价格上限 / Upper price boundary
    grid_count: int = 10            # 网格数量 / Number of grid levels
    order_size_pct: float = 0.01    # 每格1%资金 / 1% of funds per grid
    hodl_ratio: float = 0.3         # 30%仓位HODL / 30% position as HODL
    fee: float = 0.0016             # 交易费率 (Kraken maker ~0.16%) / Trading fee
    max_investment: float = 10000.0 # 最大投资额 / Maximum investment


class GridHODLStrategy(BaseStrategy):
    """
    GridHODL — 网格+Holder结合策略 / Grid + Holder combined strategy.

    在预设价格区间内置若干网格, 价格每触达一个网格进行低买高卖,
    剩余仓位长期持有 (HODL).

    Place grids within preset price range. Each time price reaches a grid,
    buy low and sell high. Remaining position is held long-term (HODL).

    策略特点 / Key Features:
    - 熊市环境: 网格持续低买积累仓位 / Bear market: grid accumulates on dips
    - 牛市环境: 网格高卖兑现利润, 底部仓位HODL / Bull: grid sells profits, HODL bottom
    - 波动越小, 网格收益越高 / Smaller volatility = higher grid profit
    - 部分仓位长期持有, 享受整体趋势 / Partial position HODL for overall trend

    与GridSell的区别 / vs GridSell:
    - GridSell: 只卖出已买入的仓位 / Only sells what it buys
    - GridHODL: 卖出部分, HODL部分享受趋势 / Sells part, HODLs part for trend

    与SWING的区别 / vs SWING:
    - SWING: 动态积累并卖出已积累的仓位 / Dynamically accumulates and sells accumulated
    - GridHODL: 固定网格静态仓位, HODL部分不参与网格 / Fixed grid static position, HODL doesn't trade
    """

    name: str = "grid_hodl"

    def __init__(
        self,
        symbol: str,
        params: Optional[GridHODLParams] = None,
    ) -> None:
        """
        初始化GridHODL策略 / Initialize GridHODL strategy.

        Args:
            symbol: 交易对符号 / Trading pair symbol (e.g., "BTCUSD")
            params: 策略参数 / Strategy parameters
        """
        super().__init__(symbol, params)
        p = params or GridHODLParams()
        self.params = p

        # 网格价格列表 / Grid price levels
        self._grid_prices: List[float] = []
        self._grid_orders: List[GridOrder] = []
        self._filled_buy_orders: List[GridOrder] = []
        self._filled_sell_orders: List[GridOrder] = []

        # HODL仓位 / HODL position
        self._hodl_position: float = 0.0
        self._hodl_cost: float = 0.0  # HODL仓位的平均成本 / Average cost of HODL position

        # 投资统计 / Investment stats
        self._total_invested: float = 0.0
        self._total_realized_profit: float = 0.0
        self._current_price: float = 0.0

        # 初始化网格 / Initialize grid
        self._init_grid()

    def _init_grid(self) -> None:
        """初始化网格价格 / Initialize grid price levels."""
        p = self.params
        if p.lower_price >= p.upper_price:
            raise ValueError(
                f"lower_price ({p.lower_price}) must be < upper_price ({p.upper_price})"
            )

        step = (p.upper_price - p.lower_price) / (p.grid_count - 1)
        self._grid_prices = [
            round(p.lower_price + i * step, 2)
            for i in range(p.grid_count)
        ]

    def generate_grid_orders(self) -> List[Dict[str, Any]]:
        """
        生成网格订单列表 / Generate grid orders list.

        Returns:
            List of order dicts with keys: price, volume, side, is_hodl
        """
        orders = []
        p = self.params

        for i, price in enumerate(self._grid_prices):
            # 每格订单金额 / Order amount per grid
            order_value = p.max_investment * p.order_size_pct

            # 买单价格 = 网格价格 * (1 - interval/2) 在网格下方
            # Buy price below grid
            buy_price = round(price * (1 - p.order_size_pct / 2), 2)

            # 卖单价格 = 网格价格 * (1 + interval/2) 在网格上方
            # Sell price above grid
            sell_price = round(price * (1 + p.order_size_pct / 2), 2)

            # 买单数量 / Buy volume
            buy_volume = round(order_value / buy_price, 8)

            # 卖单数量 (考虑费率) / Sell volume (considering fee)
            sell_volume = round(
                float(
                    Decimal(order_value)
                    / (Decimal(sell_price) * (1 - 2 * Decimal(p.fee)))
                ),
                8,
            )

            orders.append({
                "price": buy_price,
                "volume": buy_volume,
                "side": "buy",
                "is_hodl": False,
                "grid_index": i,
            })

            # HODL仓位只在中间格创建 / HODL position only at middle grids
            if p.hodl_ratio > 0 and i == len(self._grid_prices) // 2:
                hodl_volume = buy_volume * p.hodl_ratio
                orders.append({
                    "price": buy_price,
                    "volume": hodl_volume,
                    "side": "buy",
                    "is_hodl": True,
                    "grid_index": i,
                })
                # HODL仓位单独统计 / Track HODL separately
                self._hodl_position += hodl_volume
                self._hodl_cost += hodl_volume * buy_price

            # 卖单 (仅已买入仓位) / Sell orders only for filled buys
            orders.append({
                "price": sell_price,
                "volume": buy_volume * (1 - p.hodl_ratio),
                "side": "sell",
                "is_hodl": False,
                "grid_index": i,
            })

        return orders

    def on_price_update(self, current_price: float) -> List[Dict[str, Any]]:
        """
        价格更新时触发的操作 / Operations triggered by price update.

        Args:
            current_price: 当前价格 / Current price

        Returns:
            List of triggered actions: {"action": "buy"|"sell"|"hodl", "order": dict}
        """
        self._current_price = current_price
        triggered: List[Dict[str, Any]] = []

        for order in self._grid_orders[:]:
            if order.side == "buy" and current_price <= order.price:
                # 买单触发 / Buy triggered
                self._filled_buy_orders.append(order)
                self._total_invested += order.price * order.volume
                self._grid_orders.remove(order)

                triggered.append({
                    "action": "buy",
                    "order": {
                        "price": order.price,
                        "volume": order.volume,
                        "is_hodl": order.is_hodl,
                    },
                })

                # 如果非HODL仓位, 创建对应的卖单 / If non-HODL, create sell order
                if not order.is_hodl:
                    sell_price = round(
                        order.price * (1 + self.params.order_size_pct), 2
                    )
                    sell_volume = float(
                        Decimal(order.volume)
                        / (1 - 2 * Decimal(self.params.fee))
                    )
                    sell_order = GridOrder(
                        price=sell_price,
                        volume=round(sell_volume, 8),
                        side="sell",
                        txid=f"gh_sell_{len(self._filled_sell_orders)}",
                        is_hodl=False,
                    )
                    self._grid_orders.append(sell_order)

            elif order.side == "sell" and current_price >= order.price:
                # 卖单触发 / Sell triggered
                self._filled_sell_orders.append(order)
                profit = (order.price - self._get_buy_price_for_sell(order)) * order.volume
                self._total_realized_profit += profit
                self._grid_orders.remove(order)

                triggered.append({
                    "action": "sell",
                    "order": {
                        "price": order.price,
                        "volume": order.volume,
                    },
                })

        return triggered

    def _get_buy_price_for_sell(self, sell_order: GridOrder) -> float:
        """获取卖单对应的买入价格 / Get buy price corresponding to sell order."""
        for buy in reversed(self._filled_buy_orders):
            if not buy.is_hodl and buy.txid == sell_order.txid:
                return buy.price
        # 默认使用最近买入价格 / Default to most recent buy price
        for buy in reversed(self._filled_buy_orders):
            if not buy.is_hodl:
                return buy.price
        return sell_order.price / (1 + self.params.order_size_pct)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号 / Generate trading signals.

        Args:
            data: K线数据 / Candlestick data

        Returns:
            List of trading signals
        """
        signals = []
        if len(data) < 2:
            return signals

        current_price = float(data["close"].iloc[-1])
        self.on_price_update(current_price)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: Any,
    ) -> float:
        """计算仓位大小 / Calculate position size."""
        if signal.direction == SignalDirection.LONG:
            return self.params.max_investment * self.params.order_size_pct / signal.price
        return signal.metadata.get("volume", 0.0)

    def compute_profit(self) -> Dict[str, Any]:
        """
        计算当前累计利润 / Compute current cumulative profit.

        Returns:
            Dict with keys:
            - realized_profit: 已实现利润 / Realized profit
            - unrealized_profit: 未实现利润 / Unrealized profit
            - hodl_value: HODL仓位当前价值 / Current value of HODL position
            - hodl_profit: HODL利润 / HODL profit
            - total_profit: 总利润 / Total profit
            - grid_buy_count: 网格买入次数 / Grid buy count
            - grid_sell_count: 网格卖出次数 / Grid sell count
            - invested: 已投资金额 / Invested amount
        """
        unrealized = 0.0
        if self._hodl_position > 0 and self._current_price > 0:
            unrealized = (self._current_price * self._hodl_position) - self._hodl_cost

        hodl_value = self._current_price * self._hodl_position if self._current_price > 0 else 0.0
        hodl_profit = hodl_value - self._hodl_cost if self._hodl_cost > 0 else 0.0

        return {
            "realized_profit": round(self._total_realized_profit, 2),
            "unrealized_profit": round(unrealized, 2),
            "hodl_value": round(hodl_value, 2),
            "hodl_cost": round(self._hodl_cost, 2),
            "hodl_profit": round(hodl_profit, 2),
            "total_profit": round(
                self._total_realized_profit + unrealized + hodl_profit, 2
            ),
            "grid_buy_count": len(self._filled_buy_orders),
            "grid_sell_count": len(self._filled_sell_orders),
            "invested": round(self._total_invested, 2),
            "current_price": round(self._current_price, 2),
        }

    def get_grid_prices(self) -> List[float]:
        """获取网格价格列表 / Get grid price list."""
        return self._grid_prices.copy()

    def get_hodl_position(self) -> float:
        """获取HODL持仓量 / Get HODL position size."""
        return self._hodl_position

    def reset(self) -> None:
        """重置策略状态 / Reset strategy state."""
        self._grid_orders = []
        self._filled_buy_orders = []
        self._filled_sell_orders = []
        self._hodl_position = 0.0
        self._hodl_cost = 0.0
        self._total_invested = 0.0
        self._total_realized_profit = 0.0
        self._init_grid()

    def get_required_history(self) -> int:
        """获取所需历史数据长度 / Get required history length."""
        return self.params.grid_count * 2
