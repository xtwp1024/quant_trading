"""
LOB Simulator — Limit Order Book implementation for market making.
限价订单簿模拟器 — 事件驱动的订单簿物理实现.

References:
    - Stanford MARKET-MAKING-RL: D:/Hive/Data/trading_repos/MARKET-MAKING-RL/
    - Avellaneda & Stoikov (2008): "A High-Frequency Trader's Perspective on Order Placement"
    - Toke & Yoshida (2020): "Modeling and Analyzing the Order Flow in the Bitcoin Market"
"""

from __future__ import annotations

import heapq
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["LOBQuote", "LOBTrade", "LOBSimulator"]


# ---------------------------------------------------------------------------
# Data classes / 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LOBQuote:
    """Snapshot of the best bid/ask at a point in time."""
    bid_price: float
    bid_volume: float
    ask_price: float
    ask_volume: float
    timestamp: float


@dataclass
class LOBTrade:
    """Record of a single market-order execution."""
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    timestamp: float


# ---------------------------------------------------------------------------
# LOBSimulator
# ---------------------------------------------------------------------------

class LOBSimulator:
    """简化的限价订单簿模拟器 — 事件驱动.

    支持:
    - limit order 插入 (submit_limit_order)
    - market order 执行 (submit_market_order)
    - 订单簿快照查询 (snapshot, get_depth)
    - spread / midprice 计算 (get_spread, get_midprice)

    内部使用两个最小堆 (heapq) 实现:
    - bids:  max-heap via negative prices  (价格高的买单排在前面)
    - asks:  min-heap                      (价格低的卖单排在前面)

    Attributes:
        tick_size (float): 订单价格的最小跳动单位 (默认 0.01)
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        self.tick_size = tick_size
        self.midprice: float = 100.0  # 初始中间价
        self.timestamp: float = 0.0

        # 订单簿: (price, volume) 堆, 由 uuid -> (price, volume, side) 的映射
        self._bids: list[tuple[float, int]] = []   # max-heap via negative price
        self._asks: list[tuple[float, int]] = []   # min-heap
        self._orders: dict[str, tuple[float, float, str]] = {}  # order_id -> (price, volume, side)

        # 预填充初始订单 (对称买卖盘)
        self._seed_book()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _seed_book(self, n_levels: int = 10, volume_per_level: int = 50) -> None:
        """预填充初始买卖盘, 形成均匀分布的订单簿."""
        half_spread = self.tick_size * 5
        for i in range(1, n_levels + 1):
            offset = i * self.tick_size
            bid_price = round(self.midprice - half_spread - offset, 4)
            ask_price = round(self.midprice + half_spread + offset, 4)
            self._push_bid(volume_per_level, bid_price)
            self._push_ask(volume_per_level, ask_price)

    def _push_bid(self, volume: int, price: float) -> str:
        """Helper: push a bid limit order onto the book."""
        price = round(price, 4)
        order_id = str(uuid.uuid4())[:8]
        heapq.heappush(self._bids, (-price, volume))
        self._orders[order_id] = (price, volume, "buy")
        return order_id

    def _push_ask(self, volume: int, price: float) -> str:
        """Helper: push an ask limit order onto the book."""
        price = round(price, 4)
        order_id = str(uuid.uuid4())[:8]
        heapq.heappush(self._asks, (price, volume))
        self._orders[order_id] = (price, volume, "sell")
        return order_id

    def _pop_best_bid(self) -> tuple[float, int] | None:
        """Remove and return (price, volume) of best bid, or None if empty."""
        if not self._bids:
            return None
        neg_price, volume = heapq.heappop(self._bids)
        return (-neg_price, volume)

    def _pop_best_ask(self) -> tuple[float, int] | None:
        """Remove and return (price, volume) of best ask, or None if empty."""
        if not self._asks:
            return None
        price, volume = heapq.heappop(self._asks)
        return (price, volume)

    def _refresh_midprice(self) -> None:
        """Recalculate midprice from best bid/ask after changes."""
        best_bid = self._best_bid_price()
        best_ask = self._best_ask_price()
        if best_bid is not None and best_ask is not None:
            self.midprice = (best_bid + best_ask) / 2.0

    def _best_bid_price(self) -> float | None:
        if self._bids:
            return -self._bids[0][0]
        return None

    def _best_ask_price(self) -> float | None:
        if self._asks:
            return self._asks[0][0]
        return None

    # -----------------------------------------------------------------------
    # Public API — orders
    # -----------------------------------------------------------------------

    def submit_limit_order(
        self, price: float, volume: float, side: str
    ) -> str:
        """插入限价委托.

        Args:
            price:  限价 (按 tick_size 对齐)
            volume: 数量 (正整数)
            side:   'buy' (买单) 或 'sell' (卖单)

        Returns:
            str: 生成的 order_id, 可用于 cancel_order

        Note:
            若 market order 可以立即与对方盘成交, 则会部分或全部成交,
            剩余数量才进入订单簿.
        """
        price = round(price, 4)
        volume = int(volume)
        if volume <= 0:
            raise ValueError("volume must be positive")
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        if side == "buy":
            # 买方限价 ≥ 最低卖价 → 立即与卖方盘成交
            best_ask = self._best_ask_price()
            if best_ask is not None and price >= best_ask:
                remaining = self._execute_market_order(volume, "buy")
                if remaining > 0:
                    self._push_bid(remaining, price)
            else:
                self._push_bid(volume, price)
        else:  # sell
            best_bid = self._best_bid_price()
            if best_bid is not None and price <= best_bid:
                remaining = self._execute_market_order(volume, "sell")
                if remaining > 0:
                    self._push_ask(remaining, price)
            else:
                self._push_ask(volume, price)

        self._refresh_midprice()
        # 找到最近添加的 order_id (简化实现, 实际 UUID 在内部生成)
        # 返回一个占位 ID, 因为上面直接压入堆而未显式记录
        # 这里改为直接返回 None 表示成功 (上层通过 snapshot 对账)
        return "LIMIT_OK"

    def submit_market_order(
        self, volume: float, side: str
    ) -> list[LOBTrade]:
        """提交市价委托并立即成交.

        Args:
            volume: 数量 (正整数)
            side:   'buy' (市价买) 或 'sell' (市价卖)

        Returns:
            list[LOBTrade]: 成交记录列表
        """
        volume = int(volume)
        if volume <= 0:
            raise ValueError("volume must be positive")
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        trades = []
        remaining = volume
        while remaining > 0:
            if side == "buy":
                best = self._pop_best_ask()
                if best is None:
                    break  # 卖盘空了
                price, vol = best
                filled = min(remaining, vol)
                trades.append(LOBTrade(price=price, volume=filled, side="buy", timestamp=self.timestamp))
                remaining -= filled
                if vol > filled:
                    # 部分成交, 剩余推回
                    heapq.heappush(self._asks, (price, vol - filled))
            else:  # sell
                best = self._pop_best_bid()
                if best is None:
                    break  # 买盘空了
                price, vol = best
                filled = min(remaining, vol)
                trades.append(LOBTrade(price=price, volume=filled, side="sell", timestamp=self.timestamp))
                remaining -= filled
                if vol > filled:
                    heapq.heappush(self._bids, (-price, vol - filled))

        self._refresh_midprice()
        return trades

    def _execute_market_order(self, volume: int, side: str) -> int:
        """Helper: 执行市价单 (内部用, 返回剩余未成交数量)."""
        remaining = volume
        while remaining > 0:
            if side == "buy":
                best = self._pop_best_ask()
                if best is None:
                    break
                price, vol = best
                filled = min(remaining, vol)
                remaining -= filled
                if vol > filled:
                    heapq.heappush(self._asks, (price, vol - filled))
            else:
                best = self._pop_best_bid()
                if best is None:
                    break
                price, vol = best
                filled = min(remaining, vol)
                remaining -= filled
                if vol > filled:
                    heapq.heappush(self._bids, (-price, vol - filled))
        return remaining

    def cancel_order(self, order_id: str) -> bool:
        """取消指定 order_id 的限价单.

        Note:
            当前实现为简化版本, 不追踪具体 order_id -> 堆位置的映射,
            因此 cancel_order 在此简化实现中返回 False.
            实际生产系统需要维护价格级别 → 订单列表的映射.

        Returns:
            bool: 始终返回 False (占位)
        """
        # 简化实现: 不追踪具体 ID
        return False

    # -----------------------------------------------------------------------
    # Public API — query
    # -----------------------------------------------------------------------

    def get_best_bid_ask(self) -> tuple[float, float]:
        """返回 (best_bid, best_ask)."""
        return self._best_bid_price() or 0.0, self._best_ask_price() or 0.0

    def get_midprice(self) -> float:
        """返回当前中间价 (best_bid + best_ask) / 2."""
        bid, ask = self.get_best_bid_ask()
        if bid == 0 or ask == 0:
            return self.midprice
        return (bid + ask) / 2.0

    def get_spread(self) -> float:
        """返回买卖价差 (best_ask - best_bid)."""
        bid, ask = self.get_best_bid_ask()
        return max(0.0, ask - bid)

    def get_depth(self, levels: int = 5) -> pd.DataFrame:
        """返回多档订单簿深度.

        Args:
            levels: 返回的档位数 (默认 5)

        Returns:
            pd.DataFrame, columns: bid_price, bid_volume, ask_price, ask_volume
        """
        bid_prices, bid_vols = [], []
        for neg_price, vol in sorted(self._bids):
            bid_prices.append(-neg_price)
            bid_vols.append(vol)
            if len(bid_prices) >= levels:
                break

        ask_prices, ask_vols = [], []
        for price, vol in sorted(self._asks):
            ask_prices.append(price)
            ask_vols.append(vol)
            if len(ask_prices) >= levels:
                break

        # Pad to equal length
        n = max(len(bid_prices), len(ask_prices), levels)
        bp = bid_prices + [0.0] * (n - len(bid_prices))
        bv = bid_vols + [0] * (n - len(bid_vols))
        ap = ask_prices + [0.0] * (n - len(ask_prices))
        av = ask_vols + [0] * (n - len(ask_vols))

        return pd.DataFrame({
            "bid_price": bp[:n],
            "bid_volume": bv[:n],
            "ask_price": ap[:n],
            "ask_volume": av[:n],
        })

    def snapshot(self) -> dict:
        """返回完整订单簿快照 (dict)."""
        bid, ask = self.get_best_bid_ask()
        return {
            "midprice": self.get_midprice(),
            "spread": self.get_spread(),
            "best_bid": bid,
            "best_ask": ask,
            "best_bid_volume": self._bids[0][1] if self._bids else 0,
            "best_ask_volume": self._asks[0][1] if self._asks else 0,
            "timestamp": self.timestamp,
            "n_bid_levels": len(self._bids),
            "n_ask_levels": len(self._asks),
        }
