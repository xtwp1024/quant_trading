# -*- coding: utf-8 -*-
"""
SWING交易策略模块 / SWING Trading Strategy Module

区间震荡高卖低买 — Buy low and sell high within range.

策略逻辑 / Strategy Logic:
1. 识别关键支撑/阻力位 / Identify key support/resistance levels
2. 价格触及阻力卖, 触及支撑买 / Sell at resistance, buy at support
3. 网格间距根据波动率动态调整 / Dynamic grid spacing based on volatility
4. 止损设置在区间外 / Stop loss set outside range

特点 / Features:
- 区间震荡时表现最佳 / Best performance in range-bound markets
- 支撑位买入, 阻力位卖出 / Buy at support, sell at resistance
- 可动态调整网格间距 / Dynamic grid spacing adjustment
- 止损保护避免大幅亏损 / Stop loss protection

Ref: kraken-infinity-grid (SWING)
    https://github.com/btschwertfeger/kraken-infinity-grid
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams


__all__ = [
    "SWINGParams",
    "SWINGStrategy",
]


@dataclass
class SWINGParams(StrategyParams):
    """SWING策略参数 / SWING strategy parameters."""
    support: float = 45000.0       # 支撑位价格 / Support price level
    resistance: float = 55000.0   # 阻力位价格 / Resistance price level
    position_size: float = 100.0    # 每格投资金额(quote) / Investment per grid (quote)
    stop_loss_pct: float = 0.02    # 止损百分比 2% / Stop loss percentage
    interval: float = 0.01         # 网格间距 1% / Grid spacing
    fee: float = 0.0016            # 交易费率 / Trading fee
    n_open_buy_orders: int = 5     # 最大open买入单数 / Max open buy orders
    max_investment: float = 10000.0# 最大投资额 / Maximum investment
    use_dynamic_spacing: bool = True  # 是否动态调整网格 / Use dynamic grid spacing


class SWINGStrategy(BaseStrategy):
    """
    SWING交易策略 — 区间震荡高卖低买 / SWING Trading Strategy.

    在支撑位买入, 在阻力位卖出。价格区间内持续低买高卖循环,
    区间外止损保护。

    Buy at support, sell at resistance. Continuously buy low and sell high
    within the price range, with stop loss protection outside range.

    策略特点 / Key Features:
    - 区间震荡时表现最佳 / Best in range-bound markets
    - 支撑买入阻力卖出 / Support buy, resistance sell
    - 动态网格间距 / Dynamic grid spacing
    - 区间外止损 / Stop loss outside range

    与GridHODL的区别 / vs GridHODL:
    - GridHODL: 固定网格+HODL持仓, 不主动卖HODL / Fixed grid + HODL, doesn't actively sell HODL
    - SWING: 区间内高卖低买循环, 全部仓位参与交易 / Range trading, all positions trade

    与cDCA的区别 / vs cDCA:
    - cDCA: 定时定量买入, 不卖 / Regular DCA, no selling
    - SWING: 买卖双向, 有盈利有亏损 / Both buy and sell, has profit and loss
    """

    name: str = "swing"

    def __init__(
        self,
        symbol: str,
        params: Optional[SWINGParams] = None,
    ) -> None:
        """
        初始化SWING策略 / Initialize SWING strategy.

        Args:
            symbol: 交易对符号 / Trading pair symbol
            params: 策略参数 / Strategy parameters
        """
        super().__init__(symbol, params)
        p = params or SWINGParams()
        self.params = p

        # 内部状态 / Internal state
        self._buy_orders: List[Dict[str, Any]] = []
        self._sell_orders: List[Dict[str, Any]] = []
        self._highest_buy_price: float = 0.0
        self._accumulated_base: float = 0.0  # 已积累的基础货币 / Accumulated base currency
        self._current_investment: float = 0.0
        self._last_price: float = 0.0
        self._volatility: float = 0.0

        # 初始化网格 / Initialize grid
        self._dynamic_interval = p.interval

    def generate_signals(
        self,
        data: pd.DataFrame,
        volume: Optional[pd.Series] = None,
    ) -> List[Signal]:
        """
        生成交易信号 / Generate trading signals.

        Args:
            data: K线数据 (需要close列) / Candlestick data (needs close column)
            volume: 成交量数据 (可选) / Volume data (optional)

        Returns:
            List of trading signals
        """
        signals = []
        if len(data) < 2:
            return signals

        current_price = float(data["close"].iloc[-1])
        self._last_price = current_price

        # 计算波动率 / Calculate volatility
        if len(data) >= 20:
            returns = data["close"].pct_change().dropna()
            self._volatility = float(returns.std())

            # 动态调整网格间距 / Dynamic grid spacing adjustment
            if self.params.use_dynamic_spacing:
                self._dynamic_interval = self._compute_dynamic_interval()

        # 检查买单触发 / Check buy order triggers
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

        # 检查卖单触发 / Check sell order triggers
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

        # 检查止损 / Check stop loss
        stop_loss_price = self.params.support * (1 - self.params.stop_loss_pct)
        if current_price < stop_loss_price:
            signals.append(
                Signal(
                    symbol=self.symbol,
                    direction=SignalDirection.SHORT,
                    price=current_price,
                    strength=1.0,
                    metadata={
                        "order_type": "stop_loss",
                        "reason": "price_below_support",
                        "volume": self._accumulated_base,
                    },
                )
            )
            self._accumulated_base = 0.0

        # 维持网格 / Maintain grid
        self._maintain_grid(current_price)

        return signals

    def generate_signals_simple(
        self,
        price: float,
        volume: float = 0.0,
    ) -> Dict[str, Any]:
        """
        简化的信号生成 (REST接口) / Simplified signal generation (REST API).

        基于当前价格生成交易信号. / Generate trading signals based on current price.

        Args:
            price: 当前价格 / Current price
            volume: 当前成交量 / Current volume

        Returns:
            Dict with keys:
            - action: "buy"|"sell"|"stop_loss"|"hold"
            - price: trigger price
            - volume: order volume
            - is_extra_sell: bool
            - message: str description
        """
        self._last_price = price
        p = self.params

        # 计算动态波动率 (简化版) / Calculate dynamic volatility (simplified)
        if self._volatility == 0.0:
            self._volatility = abs(p.interval) / 2
        if p.use_dynamic_spacing:
            self._dynamic_interval = self._compute_dynamic_interval()

        # 检查止损 / Check stop loss
        stop_loss_price = p.support * (1 - p.stop_loss_pct)
        if price < stop_loss_price:
            return {
                "action": "stop_loss",
                "price": price,
                "volume": self._accumulated_base,
                "is_extra_sell": False,
                "message": f"Stop loss triggered @ {price:.2f} (below {stop_loss_price:.2f})",
            }

        # 检查卖单触发 / Check sell order triggers
        for sell_order in self._sell_orders[:]:
            if price >= sell_order["price"]:
                self._sell_orders.remove(sell_order)
                self._on_sell_filled(sell_order)
                return {
                    "action": "sell",
                    "price": sell_order["price"],
                    "volume": sell_order["volume"],
                    "is_extra_sell": sell_order.get("is_extra_sell", False),
                    "message": f"Sell triggered @ {sell_order['price']:.2f}",
                }

        # 检查买单触发 / Check buy order triggers
        for buy_order in self._buy_orders[:]:
            if price <= buy_order["price"]:
                self._buy_orders.remove(buy_order)
                self._on_buy_filled(buy_order)
                return {
                    "action": "buy",
                    "price": buy_order["price"],
                    "volume": buy_order["volume"],
                    "is_extra_sell": False,
                    "message": f"Buy triggered @ {buy_order['price']:.2f}",
                }

        # 维持网格 / Maintain grid
        self._maintain_grid(price)

        return {
            "action": "hold",
            "price": price,
            "volume": 0.0,
            "is_extra_sell": False,
            "message": f"Holding @ {price:.2f}",
        }

    def _compute_dynamic_interval(self) -> float:
        """
        根据波动率计算动态网格间距 / Compute dynamic grid spacing based on volatility.

        高波动 -> 大间距 / High volatility -> larger spacing
        低波动 -> 小间距 / Low volatility -> smaller spacing
        """
        p = self.params
        base = p.interval

        if self._volatility > 0.05:  # 高波动 / High volatility
            return min(base * 2, 0.03)
        elif self._volatility < 0.01:  # 低波动 / Low volatility
            return max(base / 2, 0.005)
        return base

    def _on_buy_filled(self, buy_order: Dict[str, Any]) -> None:
        """
        处理已成交买单 / Handle filled buy order.

        买单成交后:
        1. 更新最高买入价 / Update highest buy price
        2. 积累基础货币 / Accumulate base currency
        3. 在更高价格创建卖单 / Create sell order at higher price
        """
        price = buy_order["price"]
        volume = buy_order["volume"]

        if price > self._highest_buy_price:
            self._highest_buy_price = price

        self._current_investment += price * volume
        self._accumulated_base += volume

        # 创建卖单在间隔价格 / Create sell order at interval price
        sell_price = price * (1 + self._dynamic_interval)
        sell_volume = float(
            Decimal(str(volume))
            / (1 - 2 * Decimal(self.params.fee))
        )

        self._sell_orders.append({
            "price": round(sell_price, 2),
            "volume": round(sell_volume, 8),
            "txid": f"swing_sell_{buy_order.get('txid', 'unknown')}",
            "is_extra_sell": False,
        })

    def _on_sell_filled(self, sell_order: Dict[str, Any]) -> None:
        """
        处理已成交卖单 / Handle filled sell order.

        卖单成交后减少积累的基础货币. / Reduce accumulated base currency after sell fills.
        """
        if not sell_order.get("is_extra_sell", False):
            self._accumulated_base -= sell_order["volume"]
            if self._accumulated_base < 0:
                self._accumulated_base = 0.0

    def _maintain_grid(self, current_price: float) -> None:
        """
        维持SWING网格 / Maintain SWING grid.

        1. 如果价格超出阻力, 向上转移买单 / Shift buy orders up if price exceeds resistance
        2. 确保有n个open买单 / Ensure n open buy orders
        3. 在适当条件下放置额外卖单 / Place extra sell orders when appropriate
        """
        p = self.params

        # 如果价格超出阻力区间, 重置网格 / If price exits range, reset grid
        if current_price > p.resistance:
            self._shift_grid_up(current_price)
        elif current_price < p.support:
            self._shift_grid_down(current_price)

        # 确保有足够的open买单 / Ensure enough open buy orders
        self._ensure_n_open_buy_orders(current_price)

        # 放置额外卖单 (SWING特色) / Place extra sell orders (SWING feature)
        if (
            len(self._sell_orders) == 0
            and self._accumulated_base > 0
            and current_price > self._highest_buy_price
        ):
            self._place_extra_sell(current_price)

    def _shift_grid_up(self, current_price: float) -> None:
        """向上转移网格 (价格上涨超出阻力) / Shift grid up (price rose above resistance)."""
        if not self._buy_orders:
            return

        max_buy_price = max(o["price"] for o in self._buy_orders)
        threshold = (
            max_buy_price
            * (1 + self._dynamic_interval)
            * (1 + self._dynamic_interval)
            * 1.001
        )

        if current_price > threshold:
            self._buy_orders = []
            self._highest_buy_price = 0.0

    def _shift_grid_down(self, current_price: float) -> None:
        """向下转移网格 (价格跌破支撑) / Shift grid down (price fell below support)."""
        if not self._sell_orders and not self._buy_orders:
            # 价格跌破支撑, 在新位置重建网格 / Rebuild grid at new position
            self._highest_buy_price = 0.0
            self._accumulated_base = 0.0

    def _ensure_n_open_buy_orders(self, current_price: float) -> None:
        """确保有n个open买单 / Ensure n open buy orders exist."""
        p = self.params

        while (
            len(self._buy_orders) < p.n_open_buy_orders
            and self._current_investment < p.max_investment
        ):
            last_price = (
                self._buy_orders[-1]["price"] if self._buy_orders else current_price
            )
            buy_price = self._get_buy_price(last_price)
            buy_volume = float(
                Decimal(str(p.position_size)) / Decimal(str(buy_price))
            )

            self._buy_orders.append({
                "price": round(buy_price, 2),
                "volume": round(buy_volume, 8),
                "txid": f"swing_buy_{len(self._buy_orders)}",
            })

    def _place_extra_sell(self, current_price: float) -> None:
        """
        放置额外卖单 (SWING关键特性) / Place extra sell order (SWING key feature).

        额外卖单在2x间隔上方, 用于卖出积累的仓位. / Extra sell at 2x interval above, to sell accumulated position.
        """
        if self._accumulated_base <= 0:
            return

        reference_price = max(current_price, self._highest_buy_price)
        extra_sell_price = (
            reference_price
            * (1 + self._dynamic_interval)
            * (1 + self._dynamic_interval)
        )

        sell_volume = float(
            Decimal(str(self.params.position_size))
            / (Decimal(str(extra_sell_price)) * (1 - 2 * Decimal(str(self.params.fee))))
        )
        sell_volume = min(sell_volume, self._accumulated_base)

        if sell_volume > 0:
            self._sell_orders.append({
                "price": round(extra_sell_price, 2),
                "volume": round(sell_volume, 8),
                "txid": f"swing_extra_{len(self._sell_orders)}",
                "is_extra_sell": True,
            })

    def _get_buy_price(self, last_price: float) -> float:
        """
        计算下一个网格的买入价格 / Calculate buy price for next grid level.

        Buy price = last_price * 100 / (100 + 100 * interval)
        """
        return last_price * 100 / (100 + 100 * self._dynamic_interval)

    def calculate_position_size(
        self,
        signal: Signal,
        context: Any,
    ) -> float:
        """计算仓位大小 / Calculate position size."""
        if signal.direction == SignalDirection.LONG:
            return self.params.position_size / signal.price
        elif signal.direction == SignalDirection.SHORT:
            return signal.metadata.get("volume", 0.0)
        return 0.0

    def get_state(self) -> Dict[str, Any]:
        """
        获取当前策略状态 / Get current strategy state.

        Returns:
            Dict with current state info
        """
        return {
            "buy_orders_count": len(self._buy_orders),
            "sell_orders_count": len(self._sell_orders),
            "accumulated_base": round(self._accumulated_base, 8),
            "highest_buy_price": round(self._highest_buy_price, 2),
            "current_investment": round(self._current_investment, 2),
            "current_price": round(self._last_price, 2),
            "volatility": round(self._volatility, 6),
            "dynamic_interval": round(self._dynamic_interval, 6),
            "in_range": self.params.support <= self._last_price <= self.params.resistance,
        }

    def reset(self) -> None:
        """重置策略状态 / Reset strategy state."""
        self._buy_orders = []
        self._sell_orders = []
        self._highest_buy_price = 0.0
        self._accumulated_base = 0.0
        self._current_investment = 0.0
        self._last_price = 0.0
        self._volatility = 0.0
        self._dynamic_interval = self.params.interval

    def get_required_history(self) -> int:
        """获取所需历史数据长度 / Get required history length."""
        return max(self.params.n_open_buy_orders * 2, 20)
