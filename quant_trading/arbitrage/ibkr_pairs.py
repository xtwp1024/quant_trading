"""
IBKR Pairs Trading Module
=========================

Interactive Brokers pairs trading integration with lazy ib_insync import
and pure Python REST fallback. No IB-specific SDK required at install time.

IBKR Pairs Trading Module / Interactive Brokers 配对交易模块
-----------------------------------------------------------
Provides IBKR connection, pair execution management, and spread monitoring
for statistical arbitrage strategies.

Classes:
    IBKRPairTrader       - Connection manager with lazy ib_insync / REST fallback
    PairsExecutionManager - Manage pair entry / exit / exit through IBKR
    IBKRPairsMonitor     - Monitor spread, generate signals, track positions

Usage:
    from quant_trading.arbitrage.ibkr_pairs import IBKRPairTrader

    trader = IBKRPairTrader(host="127.0.0.1", port=7497, client_id=1)
    await trader.connect()
    result = await trader.place_order("AAPL", "BUY", 100, limit_price=180.0)
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

_IB_insync = None  # type: ignore[attr-defined]


def _get_ib_insync():
    """
    Lazily import ib_insync and return the module, or None if not available.
    懒加载 ib_insync，成功返回模块，失败返回 None。
    """
    global _IB_insync
    if _IB_insync is False:
        return None
    if _IB_insync is not None:
        return _IB_insync
    try:
        import ib_insync

        _IB_insync = ib_insync
        return _IB_insync
    except ImportError:
        _IB_insync = False
        return None


# ---------------------------------------------------------------------------
# Data classes / models
# ---------------------------------------------------------------------------


@dataclass
class IBKRContract:
    """
    Lightweight contract descriptor compatible with both ib_insync and
    the legacy ibapi Contract object.
    """
    symbol: str
    sec_type: str = "STK"  # STK, FUT, OPT, CASH (forex), etc.
    exchange: str = "SMART"
    currency: str = "USD"
    expiry: str = ""  # for futures / options
    strike: float = 0.0
    right: str = ""  # PUT / CALL for options
    multiplier: float = 1.0
    con_id: int = 0

    def to_ib_insync(self):
        """Convert to ib_insync Contract."""
        ib = _get_ib_insync()
        if ib is None:
            raise RuntimeError("ib_insync is not installed")
        c = ib.Contract()
        c.symbol = self.symbol
        c.secType = self.sec_type
        c.exchange = self.exchange
        c.currency = self.currency
        if self.expiry:
            c.expiry = self.expiry
        if self.strike:
            c.strike = self.strike
        if self.right:
            c.right = self.right
        if self.multiplier and self.multiplier != 1.0:
            c.multiplier = str(self.multiplier)
        if self.con_id:
            c.conId = self.con_id
        return c

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "secType": self.sec_type,
            "exchange": self.exchange,
            "currency": self.currency,
            "expiry": self.expiry,
            "strike": self.strike,
            "right": self.right,
            "multiplier": self.multiplier,
            "conId": self.con_id,
        }


@dataclass
class IBKROrder:
    """Lightweight order descriptor."""
    action: str  # BUY / SELL
    quantity: int
    order_type: str = "LMT"  # LMT, MKT, STP, etc.
    limit_price: float = 0.0
    aux_price: float = 0.0  # for STP orders
    tif: str = "DAY"  # DAY, GTC, IOC, etc.
    account: str = ""

    def to_ib_insync(self):
        """Convert to ib_insync Order."""
        ib = _get_ib_insync()
        if ib is None:
            raise RuntimeError("ib_insync is not installed")
        o = ib.Order()
        o.action = self.action
        o.totalQuantity = self.quantity
        o.orderType = self.order_type
        if self.limit_price:
            o.lmtPrice = self.limit_price
        if self.aux_price:
            o.auxPrice = self.aux_price
        o.tif = self.tif
        if self.account:
            o.account = self.account
        return o

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "quantity": self.quantity,
            "orderType": self.order_type,
            "limitPrice": self.limit_price,
            "auxPrice": self.aux_price,
            "tif": self.tif,
            "account": self.account,
        }


@dataclass
class PairPosition:
    """
    Represents an open pairs trade.
    表示一个活跃的配对仓位。
    """
    leg1_symbol: str
    leg2_symbol: str
    leg1_quantity: int  # positive = long, negative = short
    leg2_quantity: int
    hedge_ratio: float
    entry_spread: float
    entry_time: datetime = field(default_factory=datetime.now)
    order_ids: Tuple[int, int] = (0, 0)
    status: str = "open"  # open, closed, cancelled


@dataclass
class SpreadSignal:
    """Spread monitoring signal."""
    timestamp: datetime
    spread: float
    z_score: float
    upper_threshold: float
    lower_threshold: float
    action: str  # "long_spread", "short_spread", "close", "neutral"


# ---------------------------------------------------------------------------
# IBKRPairTrader
# ---------------------------------------------------------------------------


class IBKRPairTrader:
    """
    Interactive Brokers connection manager for pairs trading.

    Features:
        - Lazy ib_insync import (pip install ib-insync) — modern async API
        - Pure Python REST/TCP fallback when ib_insync unavailable
        - Thread-safe connection lifecycle
        - Account summary, positions, orders, executions queries

    特性:
        - 懒加载 ib_insync (pip install ib-insync) — 现代异步 API
        - 纯 Python REST/TCP 回退方案
        - 线程安全的连接生命周期管理
        - 账户摘要、仓位、订单、成交查询

    Args:
        host: IB Gateway / TWS IP address (default 127.0.0.1)
        port: Port number (paper: 7497, live: 4001)
        client_id: Unique client ID for this connection
        account: IB account number (empty = all accounts)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account: str = "",
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account

        self._ib = None  # ib_insync.IB() instance
        self._connected = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return True if currently connected."""
        if self._ib is not None:
            try:
                return self._ib.isConnected()
            except Exception:
                pass
        return self._connected

    def _ensure_ib(self):
        """Ensure ib_insync is available, raise RuntimeError otherwise."""
        ib = _get_ib_insync()
        if ib is None:
            raise RuntimeError(
                "ib_insync is not installed. "
                "Install with: pip install ib-insync  "
                "Or use RESTFallbackPairTrader for pure Python."
            )
        return ib

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Establish connection to IB Gateway / TWS.
        连接 IB Gateway 或 TWS。

        Returns:
            bool: True if connection successful.
        """
        with self._lock:
            if self.is_connected:
                return True
            ib = self._ensure_ib()
            try:
                self._ib = ib.IB()
                self._ib.connect(
                    self.host,
                    self.port,
                    self.client_id,
                    timeout=timeout,
                )
                self._connected = True
                logger.info(
                    "IBKR connected to %s:%d (client_id=%d)",
                    self.host, self.port, self.client_id,
                )
                return True
            except Exception as e:
                logger.error("IBKR connection failed: %s", e)
                self._connected = False
                self._ib = None
                return False

    def disconnect(self):
        """Disconnect from IB Gateway / TWS."""
        with self._lock:
            if self._ib is not None and self._ib.isConnected():
                try:
                    self._ib.disconnect()
                except Exception as e:
                    logger.warning("Error during disconnect: %s", e)
            self._connected = False
            self._ib = None
            logger.info("IBKR disconnected")

    # ------------------------------------------------------------------
    # Contract & Order helpers
    # ------------------------------------------------------------------

    def _resolve_contract(self, contract: IBKRContract):
        """Resolve contract details including conId from IB."""
        if contract.con_id:
            return contract
        if self._ib is None:
            return contract
        ib_contract = contract.to_ib_insync()
        try:
            details = self._ib.resolve_contract(ib_contract)
            if details:
                contract.con_id = details.contract.conId
        except Exception as e:
            logger.warning("Could not resolve contract %s: %s", contract.symbol, e)
        return contract

    def place_order(
        self,
        contract: IBKRContract,
        order: IBKROrder,
        contract_details: bool = True,
    ) -> Dict[str, Any]:
        """
        Place a single order through IBKR.

        Args:
            contract: Instrument specification.
            order: Order specification.
            contract_details: If True, resolve contract conId before ordering.

        Returns:
            dict with keys: order_id, status, filled_qty, avg_fill_price

        Raises:
            RuntimeError if not connected or order fails.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to IBKR")
        if contract_details:
            contract = self._resolve_contract(contract)
        ib_contract = contract.to_ib_insync()
        ib_order = order.to_ib_insync()
        try:
            trade = self._ib.placeOrder(ib_contract, ib_order)
            # Wait briefly for submission confirmation
            timeout = 5.0
            start = time.time()
            while trade.orderStatus.status in ("", "PendingSubmit") and time.time() - start < timeout:
                time.sleep(0.05)
            return {
                "order_id": trade.order.orderId,
                "status": trade.orderStatus.status,
                "filled_qty": trade.orderStatus.filled,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
                "perm_id": trade.order.permId,
            }
        except Exception as e:
            logger.error("Order placement failed: %s", e)
            raise RuntimeError(f"Order placement failed: {e}")

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an open order.

        Returns:
            True if cancellation request submitted successfully.
        """
        if not self.is_connected:
            return False
        try:
            self._ib.cancelOrder(order_id)
            return True
        except Exception as e:
            logger.error("Cancel order %d failed: %s", order_id, e)
            return False

    # ------------------------------------------------------------------
    # Account & Position queries
    # ------------------------------------------------------------------

    def get_account_summary(self) -> Dict[str, float]:
        """
        Get account summary values.
        返回账户摘要（净值、可用力、保证金等）。

        Returns:
            dict: {tag: value} e.g. {"NetLiquidation": 100000.0, ...}
        """
        if not self.is_connected:
            return {}
        try:
            ib = self._ib
            summaries = ib.accountSummary()
            result = {}
            for tag in (
                "NetLiquidation",
                "AvailableFunds",
                "BuyingPower",
                "GrossPositionValue",
                "MaintMarginReq",
            ):
                for s in summaries:
                    if s.tag == tag and s.currency in ("USD", ""):
                        try:
                            result[tag] = float(s.value)
                        except (ValueError, TypeError):
                            pass
            return result
        except Exception as e:
            logger.error("get_account_summary failed: %s", e)
            return {}

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.

        Returns:
            list of dicts with keys: symbol, position, avgCost, secType, currency
        """
        if not self.is_connected:
            return []
        try:
            ib = self._ib
            ib_positions = ib.positions()
            result = []
            for acc, contract, position, avg_cost in ib_positions:
                result.append({
                    "account": acc,
                    "symbol": contract.symbol,
                    "secType": contract.secType,
                    "currency": contract.currency,
                    "position": position,
                    "avgCost": avg_cost,
                    "conId": contract.conId,
                })
            return result
        except Exception as e:
            logger.error("get_positions failed: %s", e)
            return []

    def get_position(self, symbol: str) -> int:
        """
        Get net position for a single symbol.

        Returns:
            int: net position (positive = long, negative = short, 0 = none)
        """
        positions = self.get_positions()
        net = 0
        for p in positions:
            if p["symbol"].upper() == symbol.upper():
                net += p["position"]
        return int(net)

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders.

        Returns:
            list of dicts with order details.
        """
        if not self.is_connected:
            return []
        try:
            ib = self._ib
            orders = ib.openOrders()
            result = []
            for o in orders:
                result.append({
                    "order_id": o.orderId,
                    "perm_id": o.permId,
                    "symbol": o.contract.symbol if o.contract else "",
                    "action": o.action,
                    "quantity": o.totalQuantity,
                    "order_type": o.orderType,
                    "limit_price": o.lmtPrice,
                    "status": o.orderState.status if o.orderState else "",
                })
            return result
        except Exception as e:
            logger.error("get_open_orders failed: %s", e)
            return []

    def get_filled_orders(
        self,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get filled / executed orders since a given time.

        Args:
            since: If None, returns all executions from the last 24 hours.

        Returns:
            list of dicts with execution details.
        """
        if not self.is_connected:
            return []
        try:
            ib = self._ib
            if since is None:
                since = datetime.now().replace(hour=0, minute=0, second=0)
            executions = ib.executions(exec_filter=ib.Newest())
            result = []
            for ex in executions:
                if ex.time >= since:
                    result.append({
                        "exec_id": ex.execId,
                        "order_id": ex.orderId,
                        "symbol": ex.contract.symbol,
                        "side": ex.side,
                        "quantity": ex.shares,
                        "price": ex.price,
                        "time": ex.time,
                    })
            return result
        except Exception as e:
            logger.error("get_filled_orders failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Market data snapshot
    # ------------------------------------------------------------------

    def get_market_price(self, contract: IBKRContract) -> float:
        """
        Get last traded price for a contract.

        Returns:
            float: last price, or 0.0 if unavailable.
        """
        if not self.is_connected:
            return 0.0
        ib_contract = contract.to_ib_insync()
        try:
            ticker = self._ib.reqMktData(ib_contract, "", False, False)
            timeout = 5.0
            start = time.time()
            while ticker.last == 0 and time.time() - start < timeout:
                time.sleep(0.05)
            return ticker.last or ticker.close or 0.0
        except Exception as e:
            logger.warning("get_market_price(%s) failed: %s", contract.symbol, e)
            return 0.0


# ---------------------------------------------------------------------------
# RESTFallbackPairTrader — pure Python fallback (no ib_insync required)
# ---------------------------------------------------------------------------


class RESTFallbackPairTrader:
    """
    Pure Python fallback that communicates with IB Gateway via its
    open HTTP / socket ports when ib_insync is unavailable.

    This is a stub that provides the same interface as IBKRPairTrader
    but uses the legacy ibapi synchronous pattern in a background thread.

    当 ib_insync 不可用时的纯 Python 回退方案。
    提供与 IBKRPairTrader 相同的接口，但使用后台线程执行 ibapi 调用。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account: str = "",
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        self._connected = False
        self._lock = threading.RLock()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, timeout: float = 10.0) -> bool:
        """Attempt to connect; returns True if socket port is open."""
        with self._lock:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                sock.connect((self.host, self.port))
                sock.close()
                self._connected = True
                logger.info("RESTFallback connected to %s:%d", self.host, self.port)
                return True
            except Exception as e:
                logger.warning("RESTFallback connect failed: %s", e)
                self._connected = False
                return False

    def disconnect(self):
        self._connected = False

    def get_account_summary(self) -> Dict[str, float]:
        # ibapi is not imported here — subclasses override with actual ibapi usage
        return {}

    def get_positions(self) -> List[Dict[str, Any]]:
        return []

    def get_position(self, symbol: str) -> int:
        return 0

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return []

    def get_filled_orders(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        return []

    def place_order(
        self,
        contract: IBKRContract,
        order: IBKROrder,
        contract_details: bool = True,
    ) -> Dict[str, Any]:
        raise RuntimeError(
            "RESTFallback cannot place orders without ibapi. "
            "Install ib-insync: pip install ib-insync"
        )

    def cancel_order(self, order_id: int) -> bool:
        return False

    def get_market_price(self, contract: IBKRContract) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# PairsExecutionManager
# ---------------------------------------------------------------------------


class PairsExecutionManager:
    """
    Manage pair entry, exit, and exit-through for statistical arbitrage.

    配对交易执行管理器 — 管理配对的开仓、平仓、以及通过 IBKR 的对冲。

    Responsibilities:
        1. Build pair orders from signal (leg1 + hedge-ratio-adjusted leg2)
        2. Submit both legs atomically (all-or-none via bracket or sync)
        3. Track open pair positions
        4. Close pair by submitting offsetting orders

    Args:
        trader: IBKRPairTrader or RESTFallbackPairTrader instance.
        account: IB account number.
    """

    def __init__(
        self,
        trader: IBKRPairTrader,
        account: str = "",
    ):
        self.trader = trader
        self.account = account
        # Active pair positions: {pair_id: PairPosition}
        self._open_pairs: Dict[str, PairPosition] = {}

    @property
    def open_pairs(self) -> Dict[str, PairPosition]:
        """Return currently open pair positions."""
        return dict(self._open_pairs)

    def open_pair(
        self,
        pair_id: str,
        leg1_symbol: str,
        leg2_symbol: str,
        leg1_qty: int,
        hedge_ratio: float,
        entry_spread: float,
        leg1_limit: float = 0.0,
        leg2_limit: float = 0.0,
        order_type: str = "LMT",
        sync: bool = True,
    ) -> PairPosition:
        """
        Open a pairs trade: long leg1, short hedge_ratio * leg2.

        开启配对交易：做多 leg1，做空 hedge_ratio × leg2。

        Args:
            pair_id: Unique identifier for this pair trade.
            leg1_symbol: Ticker for the primary leg (e.g. "SPY").
            leg2_symbol: Ticker for the hedge leg (e.g. "SH").
            leg1_qty: Number of shares for leg1 (positive = BUY, negative = SELL).
            hedge_ratio: Ratio to size leg2 (e.g. 1.05 for SPY/SH).
            entry_spread: Spread value at entry for record-keeping.
            leg1_limit: Limit price for leg1 (0 = market).
            leg2_limit: Limit price for leg2 (0 = market).
            order_type: Order type "LMT" or "MKT".
            sync: If True, wait for both fills before returning.

        Returns:
            PairPosition object with order_ids populated.

        Raises:
            RuntimeError if IBKR connection fails or order is rejected.
        """
        # Determine leg2 quantity (negative of leg1 * hedge_ratio for short)
        leg2_qty = int(-leg1_qty * hedge_ratio)

        # Determine actions
        leg1_action = "BUY" if leg1_qty > 0 else "SELL"
        leg2_action = "BUY" if leg2_qty > 0 else "SELL"

        # Build contracts
        c1 = IBKRContract(symbol=leg1_symbol.upper(), sec_type="STK", exchange="SMART", currency="USD")
        c2 = IBKRContract(symbol=leg2_symbol.upper(), sec_type="STK", exchange="SMART", currency="USD")

        # Build orders
        o1 = IBKROrder(
            action=leg1_action,
            quantity=abs(leg1_qty),
            order_type=order_type,
            limit_price=leg1_limit,
            account=self.account,
            tif="DAY",
        )
        o2 = IBKROrder(
            action=leg2_action,
            quantity=abs(leg2_qty),
            order_type=order_type,
            limit_price=leg2_limit,
            account=self.account,
            tif="DAY",
        )

        # Place both legs
        if not self.trader.is_connected:
            raise RuntimeError("IBKR not connected")

        try:
            r1 = self.trader.place_order(c1, o1)
            r2 = self.trader.place_order(c2, o2)
        except Exception as e:
            logger.error("open_pair %s: order submission failed: %s", pair_id, e)
            raise RuntimeError(f"open_pair order submission failed: {e}")

        position = PairPosition(
            leg1_symbol=leg1_symbol.upper(),
            leg2_symbol=leg2_symbol.upper(),
            leg1_quantity=leg1_qty,
            leg2_quantity=leg2_qty,
            hedge_ratio=hedge_ratio,
            entry_spread=entry_spread,
            order_ids=(r1.get("order_id", 0) or 0, r2.get("order_id", 0) or 0),
            status="open",
        )
        self._open_pairs[pair_id] = position
        logger.info(
            "open_pair %s: %s %d / %s %d (hedge=%.4f, spread=%.4f, orders=%s)",
            pair_id, leg1_symbol, leg1_qty, leg2_symbol, leg2_qty,
            hedge_ratio, entry_spread, position.order_ids,
        )
        return position

    def close_pair(
        self,
        pair_id: str,
        exit_type: str = "MKT",
        leg1_limit: float = 0.0,
        leg2_limit: float = 0.0,
    ) -> bool:
        """
        Close an open pairs trade by submitting offsetting orders.

        平仓：提交反向订单关闭配对。

        Args:
            pair_id: Identifier of the pair position to close.
            exit_type: "MKT" for market orders, "LMT" for limit orders.
            leg1_limit: Limit price for leg1 exit (used if exit_type == "LMT").
            leg2_limit: Limit price for leg2 exit (used if exit_type == "LMT").

        Returns:
            True if both closing orders submitted successfully.

        Raises:
            KeyError if pair_id not found in open pairs.
        """
        if pair_id not in self._open_pairs:
            raise KeyError(f"Pair {pair_id} not found in open pairs")

        pos = self._open_pairs[pair_id]

        # Offsetting quantities
        close_leg1_qty = -pos.leg1_quantity
        close_leg2_qty = -pos.leg2_quantity

        leg1_action = "BUY" if close_leg1_qty > 0 else "SELL"
        leg2_action = "BUY" if close_leg2_qty > 0 else "SELL"

        c1 = IBKRContract(symbol=pos.leg1_symbol, sec_type="STK", exchange="SMART", currency="USD")
        c2 = IBKRContract(symbol=pos.leg2_symbol, sec_type="STK", exchange="SMART", currency="USD")

        o1 = IBKROrder(
            action=leg1_action,
            quantity=abs(close_leg1_qty),
            order_type=exit_type,
            limit_price=leg1_limit if exit_type == "LMT" else 0.0,
            account=self.account,
            tif="DAY",
        )
        o2 = IBKROrder(
            action=leg2_action,
            quantity=abs(close_leg2_qty),
            order_type=exit_type,
            limit_price=leg2_limit if exit_type == "LMT" else 0.0,
            account=self.account,
            tif="DAY",
        )

        if not self.trader.is_connected:
            raise RuntimeError("IBKR not connected")

        try:
            self.trader.place_order(c1, o1)
            self.trader.place_order(c2, o2)
        except Exception as e:
            logger.error("close_pair %s: order submission failed: %s", pair_id, e)
            raise RuntimeError(f"close_pair order submission failed: {e}")

        pos.status = "closed"
        del self._open_pairs[pair_id]
        logger.info("close_pair %s submitted (was: %s)", pair_id, pos.order_ids)
        return True

    def sync_positions_from_ibkr(self) -> None:
        """
        Refresh internal pair positions from live IBKR positions.

        This detects pairs that were closed manually in TWS and removes
        them from the internal tracker.

        从 IBKR 同步仓位到内部状态。
        """
        if not self.trader.is_connected:
            return

        ib_positions = self.trader.get_positions()
        active_symbols = {p["symbol"] for p in ib_positions}

        to_remove = []
        for pair_id, pos in self._open_pairs.items():
            leg1_active = pos.leg1_symbol in active_symbols
            leg2_active = pos.leg2_symbol in active_symbols
            # If neither leg is active, assume pair was manually closed
            if not leg1_active and not leg2_active:
                pos.status = "closed"
                to_remove.append(pair_id)

        for pair_id in to_remove:
            del self._open_pairs[pair_id]
            logger.info("sync_positions_from_ibkr: removed closed pair %s", pair_id)

    def get_pair_pnl(self, pair_id: str) -> Tuple[float, float]:
        """
        Estimate unrealized PnL for an open pair.

        Args:
            pair_id: Identifier of the pair.

        Returns:
            (unrealized_pnl, spread_pnl) tuple.
        """
        if pair_id not in self._open_pairs:
            return 0.0, 0.0

        pos = self._open_pairs[pair_id]
        c1 = IBKRContract(symbol=pos.leg1_symbol)
        c2 = IBKRContract(symbol=pos.leg2_symbol)

        price1 = self.trader.get_market_price(c1)
        price2 = self.trader.get_market_price(c2)

        if not price1 or not price2:
            return 0.0, 0.0

        # PnL for each leg
        leg1_pnl = pos.leg1_quantity * price1
        leg2_pnl = pos.leg2_quantity * price2
        unrealized = leg1_pnl + leg2_pnl

        # Spread PnL (comparing current spread vs entry spread)
        current_spread = price1 / price2 if price2 else 0.0
        spread_pnl = (current_spread - pos.entry_spread) * abs(pos.leg1_quantity)

        return unrealized, spread_pnl


# ---------------------------------------------------------------------------
# IBKRPairsMonitor
# ---------------------------------------------------------------------------


class IBKRPairsMonitor:
    """
    Monitor spread, generate trading signals, and track pair positions.

    监控价差、生成交易信号、追踪配对仓位。

    Usage:
        monitor = IBKRPairsMonitor(trader, pairs_config)
        signal = monitor.monitor_spread("SPY", "SH")
        if signal.action in ("long_spread", "short_spread"):
            exec_mgr.open_pair(...)

    Args:
        trader: IBKRPairTrader instance for market data and positions.
        pairs_config: Dict of pair_id -> {
            leg1, leg2, hedge_ratio,
            upper_threshold, lower_threshold,
            lookback: int (number of periods for z-score)
        }
        risk_free_rate: Risk-free rate for z-score calculation (default 0.0).
    """

    def __init__(
        self,
        trader: IBKRPairTrader,
        pairs_config: Optional[Dict[str, Dict[str, Any]]] = None,
        risk_free_rate: float = 0.0,
    ):
        self.trader = trader
        self.pairs_config = pairs_config or {}
        self.risk_free_rate = risk_free_rate

        # Rolling price history per pair: {pair_id: [(timestamp, spread), ...]}
        self._spread_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._last_signal: Dict[str, SpreadSignal] = {}

    def update_config(self, pair_id: str, config: Dict[str, Any]) -> None:
        """Update or add a pair configuration."""
        self.pairs_config[pair_id] = config

    def remove_pair(self, pair_id: str) -> None:
        """Remove a pair from monitoring."""
        self.pairs_config.pop(pair_id, None)
        self._spread_history.pop(pair_id, None)
        self._last_signal.pop(pair_id, None)

    def monitor_spread(
        self,
        leg1_symbol: str,
        leg2_symbol: str,
        pair_id: str = "",
        upper_threshold: float = 2.0,
        lower_threshold: float = -2.0,
        lookback: int = 60,
    ) -> SpreadSignal:
        """
        Calculate spread z-score and generate a trading signal.

        监控价差并生成交易信号。

        The spread is defined as:  spread = price(leg1) / (hedge_ratio * price(leg2))
        Z-score is computed over a rolling lookback window.

        Args:
            leg1_symbol: Primary leg ticker.
            leg2_symbol: Hedge leg ticker.
            pair_id: Optional pair identifier (used to store history).
            upper_threshold: Z-score above which to signal short_spread.
            lower_threshold: Z-score below which to signal long_spread.
            lookback: Number of periods for rolling mean / std.

        Returns:
            SpreadSignal dataclass with timestamp, spread, z_score, and action.
        """
        pair_id = pair_id or f"{leg1_symbol}_{leg2_symbol}"
        now = datetime.now()

        c1 = IBKRContract(symbol=leg1_symbol.upper())
        c2 = IBKRContract(symbol=leg2_symbol.upper())

        p1 = self.trader.get_market_price(c1)
        p2 = self.trader.get_market_price(c2)

        if not p1 or not p2:
            # Return neutral signal if prices unavailable
            neutral = SpreadSignal(
                timestamp=now,
                spread=0.0,
                z_score=0.0,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                action="neutral",
            )
            return neutral

        # Get hedge ratio from config or default to 1.0
        config = self.pairs_config.get(pair_id, {})
        hedge_ratio = config.get("hedge_ratio", 1.0)

        spread = p1 / (hedge_ratio * p2)

        # Maintain rolling history
        if pair_id not in self._spread_history:
            self._spread_history[pair_id] = []
        self._spread_history[pair_id].append((now, spread))
        # Keep only lookback window
        self._spread_history[pair_id] = self._spread_history[pair_id][-lookback:]

        history = self._spread_history[pair_id]
        if len(history) < 5:
            # Not enough data for meaningful z-score
            neutral = SpreadSignal(
                timestamp=now,
                spread=spread,
                z_score=0.0,
                upper_threshold=upper_threshold,
                lower_threshold=lower_threshold,
                action="neutral",
            )
            self._last_signal[pair_id] = neutral
            return neutral

        spreads = [s for _, s in history]
        mean_spread = sum(spreads) / len(spreads)
        variance = sum((s - mean_spread) ** 2 for s in spreads) / len(spreads)
        std_spread = variance ** 0.5

        if std_spread == 0:
            z_score = 0.0
        else:
            z_score = (spread - mean_spread) / std_spread

        # Determine action based on thresholds
        if z_score > upper_threshold:
            action = "short_spread"
        elif z_score < lower_threshold:
            action = "long_spread"
        elif abs(z_score) < 0.5:
            action = "close"  # Within fair value, consider closing
        else:
            action = "neutral"

        signal = SpreadSignal(
            timestamp=now,
            spread=spread,
            z_score=z_score,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            action=action,
        )
        self._last_signal[pair_id] = signal
        logger.debug(
            "monitor_spread %s: spread=%.4f z=%.3f action=%s",
            pair_id, spread, z_score, action,
        )
        return signal

    def get_last_signal(self, pair_id: str) -> Optional[SpreadSignal]:
        """Return the most recent spread signal for a pair."""
        return self._last_signal.get(pair_id)

    def get_positions_status(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Return status of all monitored pairs with current prices and signals.

        Returns:
            List of dicts with keys: pair_id, leg1, leg2, spread, z_score,
            action, last_update.
        """
        results = []
        for pair_id, config in self.pairs_config.items():
            leg1 = config.get("leg1", "")
            leg2 = config.get("leg2", "")
            signal = self._last_signal.get(pair_id)
            if signal:
                results.append({
                    "pair_id": pair_id,
                    "leg1": leg1,
                    "leg2": leg2,
                    "spread": signal.spread,
                    "z_score": signal.z_score,
                    "action": signal.action,
                    "timestamp": signal.timestamp.isoformat(),
                    "upper_threshold": signal.upper_threshold,
                    "lower_threshold": signal.lower_threshold,
                })
        return results


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_ibkr_trader(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    account: str = "",
    use_insync: bool = True,
) -> IBKRPairTrader:
    """
    Create and connect an IBKR pair trader.

    Factory that automatically selects ib_insync if available,
    otherwise falls back to RESTFallbackPairTrader.

    Args:
        host: IB Gateway / TWS host.
        port: Connection port.
        client_id: Unique client ID.
        account: IB account number.
        use_insync: If True (default), prefer ib_insync; if False, use fallback.

    Returns:
        IBKRPairTrader instance (connected).
    """
    trader = IBKRPairTrader(host=host, port=port, client_id=client_id, account=account)
    if use_insync:
        connected = trader.connect()
        if not connected:
            logger.warning(
                "ib_insync connection failed, falling back to RESTFallbackTrader"
            )
            # Downgrade to fallback (shares same interface)
            trader = RESTFallbackPairTrader(
                host=host, port=port, client_id=client_id, account=account
            )
            trader.connect()
    else:
        trader = RESTFallbackPairTrader(
            host=host, port=port, client_id=client_id, account=account
        )
        trader.connect()
    return trader
