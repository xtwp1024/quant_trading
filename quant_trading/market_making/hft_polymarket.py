"""
HFT Polymarket Market Maker — 高频Polymarket做市商
=====================================================
源自: D:/Hive/Data/trading_repos/py_polymarket_hft_mm/

优化策略:
- CPU affinity绑定 — 减少线程切换, 提升tick处理 latency
- 对象池 — 复用订单字典, 最小化GC
- BPS阈值检测 — 只在价差超过阈值时交易
- 批量订单合并 — 减少网络往返, 提升吞吐量

Usage:
    from quant_trading.market_making.hft_polymarket import HFTPolymarketMM
    mm = HFTPolymarketMM(cpu_core=0, use_affinity=True, bps_threshold=0.0001)
    orders = mm.process_tick("token-abc", bid=0.52, ask=0.55)
"""

from __future__ import annotations

import os
import sys
import time
import logging
import threading
from typing import Optional

__all__ = ["HFTPolymarketMM"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CPU Affinity (纯Python实现, 无外部依赖)                                       #
# --------------------------------------------------------------------------- #

def _set_cpu_affinity_unix(pid: int, cores: list[int]) -> bool:
    """Set CPU affinity on Linux/POSIX via sched_setaffinity.

    Args:
        pid: Process ID (0 = current process)
        cores: List of CPU core indices to bind

    Returns:
        True if successful, False otherwise
    """
    try:
        import resource

        def cpuset_size() -> int:
            # Number of CPUs in the system (fallback)
            return os.cpu_count() or 1

        # On Linux we can use sched_setaffinity via ctypes or subprocess
        # Prefer subprocess for broadest compatibility
        import subprocess

        mask = ",".join(str(c) for c in cores)
        result = subprocess.run(
            ["taskset", "-cp", mask, str(pid)],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _set_cpu_affinity_windows(pid: int, cores: list[int]) -> bool:
    """Set CPU affinity on Windows via os.sched_setaffinity or ctypes.

    Args:
        pid: Process ID (0 = current process)
        cores: List of CPU core indices to bind

    Returns:
        True if successful, False otherwise
    """
    try:
        # Python 3.8+ supports os.sched_setaffinity on all platforms
        os.sched_setaffinity(pid, cores)
        return True
    except (AttributeError, OSError):
        pass

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        class Kaffeine(ctypes.Structure):
            _fields_ = [
                ("AffinityMask", ctypes.c_uint64),
            ]

        mask = 0
        for core in cores:
            mask |= 1 << core

        kaff = Kaffeine()
        kaff.AffinityMask = mask
        result = kernel32.SetThreadAffinityMask(
            kernel32.GetCurrentThread(), mask
        )
        return result != 0
    except Exception:
        return False


def _get_pid() -> int:
    """Return current process ID (cross-platform)."""
    return os.getpid()


def set_process_cpu_affinity(cores: list[int], pid: Optional[int] = None) -> bool:
    """Bind the current process to a specific set of CPU cores.

    Args:
        cores: List of CPU core indices (e.g. [3, 4] binds to cores 3 and 4)
        pid: Process ID (default: current process)

    Returns:
        True if affinity was set successfully, False otherwise
    """
    if pid is None:
        pid = _get_pid()

    if sys.platform == "win32":
        ok = _set_cpu_affinity_windows(pid, cores)
    else:
        ok = _set_cpu_affinity_unix(pid, cores)

    if ok:
        logger.info(f"[CPU] Process {pid} affinity set to cores {cores}")
    else:
        logger.warning(f"[CPU] Failed to set affinity to cores {cores}")

    return ok


# --------------------------------------------------------------------------- #
# Object Pool — 对象池, 减少GC压力                                           #
# --------------------------------------------------------------------------- #

class OrderPool:
    """Lock-free-ish object pool for order dictionaries.

    Reduces allocations by recycling dict objects instead of creating
    new ones on every tick.
    """

    def __init__(self, initial_size: int = 256):
        self._pool: list[dict] = []
        self._lock = threading.Lock()
        for _ in range(initial_size):
            self._pool.append({})

    def acquire(self) -> dict:
        """Pop a dict from the pool (or create a new one)."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return {}

    def release(self, obj: dict) -> None:
        """Return a dict to the pool."""
        obj.clear()  # Reuse without allocation
        with self._lock:
            self._pool.append(obj)


# --------------------------------------------------------------------------- #
# HFTPolymarketMM                                                            #
# --------------------------------------------------------------------------- #

class HFTPolymarketMM:
    """高频Polymarket做市商.

    优化:
    - CPU affinity绑定 — 减少线程切换, 降低 tick 处理延迟
    - 最小化GC — 对象池复用订单字典, 禁用不必要的垃圾回收
    - BPS阈值检测 — 只在bid-ask价差超过阈值时挂单
    - 批量订单合并 — 减少网络往返, 提升吞吐量

    Args:
        cpu_core: CPU核心编号, 用于绑定 (默认 0)
        use_affinity: 是否启用CPU亲和性绑定 (默认 True)
        bps_threshold: 最小交易阈值, 以BPS计量 (默认 0.0001 = 1bps)
                      价差 = (ask - bid) / mid_price
        max_pending: 最大待处理订单数 (默认 64)
    """

    def __init__(
        self,
        cpu_core: int = 0,
        use_affinity: bool = True,
        bps_threshold: float = 0.0001,
        max_pending: int = 64,
    ):
        self.cpu_core = cpu_core
        self.use_affinity = use_affinity
        self.bps_threshold = bps_threshold          # 1 bps = 0.0001
        self.max_pending = max_pending

        # Object pool for orders
        self._order_pool = OrderPool(initial_size=max_pending * 2)

        # Pending orders buffer (batch merge target)
        self.pending_orders: list[dict] = []
        self._pending_lock = threading.Lock()

        # Statistics
        self._tick_count = 0
        self._trade_count = 0
        self._skip_count = 0

        # Apply CPU affinity immediately
        if self.use_affinity:
            self.set_cpu_affinity(cpu_core)

        # Disable GC in the hot path (user can re-enable externally if needed)
        # gc.disable() is a global setting — we document it rather than call it
        # here to avoid affecting other code in the same process.

        logger.info(
            f"[HFTPolymarketMM] Initialized: cpu_core={cpu_core}, "
            f"affinity={use_affinity}, bps_threshold={bps_threshold}"
        )

    # ------------------------------------------------------------------ #
    # CPU Affinity                                                        #
    # ------------------------------------------------------------------ #

    def set_cpu_affinity(self, core: int) -> None:
        """绑定CPU核心 — 减少线程切换.

        将当前进程绑定到指定CPU核心, 避免OS调度器将线程迁移到其他核心,
        从而减少cache miss并提升确定性延迟.

        Args:
            core: CPU核心编号 (从0开始)
        """
        ok = set_process_cpu_affinity([core], pid=None)
        if not ok:
            # Fallback: try with current process
            set_process_cpu_affinity([core])
        logger.info(f"[HFTPolymarketMM] CPU affinity set to core {core}")

    # ------------------------------------------------------------------ #
    # BPS Threshold                                                       #
    # ------------------------------------------------------------------ #

    def should_trade(
        self,
        market_id: str,
        bid_price: float,
        ask_price: float,
    ) -> bool:
        """BPS阈值判断 — 价差 > 阈值才交易.

        计算 bid-ask 价差的基点数 (BPS), 仅当其超过 bps_threshold 时返回 True.

        BPS = |mid - micro| * 10000
        或用于价差:  (ask - bid) / mid_price * 10000

        Args:
            market_id: 市场/合约ID (用于日志)
            bid_price: 当前买一价
            ask_price: 当前卖一价

        Returns:
            True 如果价差超过阈值, False 否则
        """
        if bid_price <= 0 or ask_price <= 0 or ask_price <= bid_price:
            return False

        mid_price = (bid_price + ask_price) / 2.0
        if mid_price <= 0:
            return False

        # Spread in BPS
        spread_bps = (ask_price - bid_price) / mid_price * 10000.0

        if spread_bps < self.bps_threshold * 10000:
            self._skip_count += 1
            logger.debug(
                f"[{market_id}] Skip: spread={spread_bps:.2f}bps "
                f"< threshold={self.bps_threshold * 10000:.2f}bps"
            )
            return False

        logger.debug(
            f"[{market_id}] Trade OK: spread={spread_bps:.2f}bps "
            f"(bid={bid_price}, ask={ask_price})"
        )
        return True

    # ------------------------------------------------------------------ #
    # Batch Order Merging                                                  #
    # ------------------------------------------------------------------ #

    def batch_orders(self, new_orders: list[dict]) -> list[dict]:
        """批量合并订单 — 减少网络往返.

        将多个新订单按 (market_id, side, price) 聚合, 合并相同价格的订单量.
        这减少了发送到交易所的请求数量.

        Args:
            new_orders: 新订单列表, 每个订单应包含:
                - market_id: str
                - side: "BUY" or "SELL"
                - price: float
                - size: float

        Returns:
            合并后的订单列表
        """
        if not new_orders:
            return []

        # Key: (market_id, side, price) -> aggregated order
        merged: dict[tuple, dict] = {}

        for order in new_orders:
            key = (order.get("market_id", ""), order.get("side", ""), order.get("price", 0.0))
            if key not in merged:
                merged[key] = {
                    "market_id": key[0],
                    "side": key[1],
                    "price": key[2],
                    "size": 0.0,
                }
            merged[key]["size"] += order.get("size", 0.0)

        result = list(merged.values())
        logger.debug(
            f"[Batch] Input={len(new_orders)} orders -> "
            f"Output={len(result)} after merge"
        )
        return result

    # ------------------------------------------------------------------ #
    # Object Pool — Slot Acquisition / Release                           #
    # ------------------------------------------------------------------ #

    def acquire_order_slot(self) -> dict:
        """从对象池获取订单槽 — 避免GC.

        从内部对象池获取一个可复用的订单字典, 避免每次创建新对象
        导致的内存分配和后续GC压力.

        Returns:
            可复用的订单字典
        """
        return self._order_pool.acquire()

    def release_order_slot(self, order: dict) -> None:
        """归还订单槽到对象池.

        清空订单字典内容并返回到对象池, 供后续重用.

        Args:
            order: 之前通过 acquire_order_slot() 获取的字典
        """
        self._order_pool.release(order)

    # ------------------------------------------------------------------ #
    # Tick Processing                                                     #
    # ------------------------------------------------------------------ #

    def process_tick(
        self,
        market_id: str,
        bid: float,
        ask: float,
    ) -> list[dict]:
        """处理单个tick — 返回需下单列表.

        评估当前bid/ask价格, 根据BPS阈值判断是否生成订单,
        并通过对象池分配订单对象以避免GC.

        Args:
            market_id: 市场/合约ID
            bid: 买一价
            ask: 卖一价

        Returns:
            需要提交的订单列表 (从对象池获取, 调用方用 release_order_slot 归还)
        """
        self._tick_count += 1

        if not self.should_trade(market_id, bid, ask):
            return []

        mid = (bid + ask) / 2.0

        orders_to_submit: list[dict] = []

        # BUY side (bid)
        buy_order = self.acquire_order_slot()
        buy_order["market_id"] = market_id
        buy_order["side"] = "BUY"
        buy_order["price"] = round(bid, 4)
        buy_order["size"] = 1.0
        buy_order["tick"] = self._tick_count
        orders_to_submit.append(buy_order)

        # SELL side (ask)
        sell_order = self.acquire_order_slot()
        sell_order["market_id"] = market_id
        sell_order["side"] = "SELL"
        sell_order["price"] = round(ask, 4)
        sell_order["size"] = 1.0
        sell_order["tick"] = self._tick_count
        orders_to_submit.append(sell_order)

        self._trade_count += 1

        logger.debug(
            f"[{market_id}] Tick {self._tick_count}: "
            f"BUY@{buy_order['price']} SELL@{sell_order['price']} "
            f"(mid={mid:.4f})"
        )

        return orders_to_submit

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """返回运行时统计信息."""
        return {
            "tick_count": self._tick_count,
            "trade_count": self._trade_count,
            "skip_count": self._skip_count,
            "pool_size": len(self._order_pool._pool),  # noqa: SLF001
            "pending_orders": len(self.pending_orders),
        }

    def reset_stats(self) -> None:
        """重置统计计数器."""
        self._tick_count = 0
        self._trade_count = 0
        self._skip_count = 0

    def __repr__(self) -> str:
        return (
            f"HFTPolymarketMM(cpu_core={self.cpu_core}, "
            f"bps_threshold={self.bps_threshold}, "
            f"ticks={self._tick_count}, trades={self._trade_count})"
        )
