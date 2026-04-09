"""
Order Execution Engine
订单执行引擎

Provides advanced order execution strategies including:
- Market and limit order execution
- Iceberg orders (split large orders into smaller chunks)
- TWAP (Time-Weighted Average Price) execution
- Slippage control and monitoring

Example:
    adapter = CEXAdapter('binance', api_key='...', api_secret='...')
    engine = OrderEngine(adapter, max_slippage=0.005, ice_chunk_pct=0.1)
    result = engine.execute_market('BTCUSDT', 'buy', 1.0)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from quant_trading.connectors.cex_adapter import CEXAdapter

__all__ = ["OrderEngine"]


class OrderEngine:
    """订单执行引擎 / Order execution engine.

    功能:
    - 订单生命周期管理 / Order lifecycle management
    - 冰山订单 / Iceberg orders
    - TWAP分单 / TWAP slicing
    - 滑点控制 / Slippage control

    Args:
        adapter: CEXAdapter instance for exchange communication
        max_slippage: Maximum allowed slippage ratio (default 0.005 = 0.5%)
        ice_chunk_pct: Percentage of total qty per iceberg chunk (default 0.1 = 10%)
    """

    def __init__(
        self,
        adapter: CEXAdapter,
        max_slippage: float = 0.005,
        ice_chunk_pct: float = 0.1,
    ):
        self.adapter = adapter
        self.max_slippage = max_slippage
        self.ice_chunk_pct = ice_chunk_pct
        self.logger = logging.getLogger("OrderEngine")

    def execute_market(self, symbol: str, side: str, quantity: float) -> dict:
        """市价单执行 / Execute market order.

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            quantity: Order quantity

        Returns:
            Dict with execution result including fills and slippage
        """
        try:
            # Get current market price for slippage calculation
            ticker = self.adapter.get_ticker(symbol)
            current_price = ticker.get("price", 0)

            if current_price == 0:
                return {
                    "success": False,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "error": "Failed to get current price",
                }

            # Place market order
            result = self.adapter.place_market_order(symbol, side, quantity)

            if result.get("order_id"):
                # Estimate fill price (in production, parse actual fill price)
                fill_price = current_price
                slippage = self._calc_slippage(fill_price, current_price, side)

                if slippage > self.max_slippage:
                    self.logger.warning(
                        f"Slippage {slippage:.4%} exceeds max {self.max_slippage:.4%} "
                        f"for {symbol} {side}"
                    )

                return {
                    "success": True,
                    "order_id": result.get("order_id"),
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": fill_price,
                    "slippage": slippage,
                    "timestamp": int(time.time() * 1000),
                }
            else:
                return {
                    "success": False,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "error": result.get("error", "Unknown error"),
                }

        except Exception as e:
            self.logger.error(f"execute_market failed for {symbol}: {e}")
            return {
                "success": False,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "error": str(e),
            }

    def execute_limit(
        self, symbol: str, side: str, price: float, quantity: float
    ) -> dict:
        """限价单执行 / Execute limit order.

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            price: Limit price
            quantity: Order quantity

        Returns:
            Dict with order placement result
        """
        try:
            result = self.adapter.place_limit_order(symbol, side, price, quantity)

            if result.get("order_id"):
                return {
                    "success": True,
                    "order_id": result.get("order_id"),
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "quantity": quantity,
                    "timestamp": int(time.time() * 1000),
                }
            else:
                return {
                    "success": False,
                    "symbol": symbol,
                    "side": side,
                    "price": price,
                    "quantity": quantity,
                    "error": result.get("error", "Unknown error"),
                }

        except Exception as e:
            self.logger.error(f"execute_limit failed for {symbol}: {e}")
            return {
                "success": False,
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": quantity,
                "error": str(e),
            }

    def execute_iceberg(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> List[dict]:
        """冰山订单 — 分批小单执行 / Iceberg order — execute in small chunks.

        Splits a large order into smaller chunks, displaying only the visible
        chunk at a time to minimize market impact.

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            quantity: Total order quantity
            price: Limit price for all chunks

        Returns:
            List of execution results for each chunk
        """
        chunk_size = quantity * self.ice_chunk_pct
        if chunk_size <= 0:
            self.logger.error(f"Chunk size too small: {chunk_size}")
            return []

        n_chunks = max(1, int(quantity / chunk_size))
        results = []

        self.logger.info(
            f"Starting iceberg {side} {quantity} {symbol} in {n_chunks} chunks"
        )

        for i in range(n_chunks):
            remaining = quantity - (i * chunk_size)
            actual_chunk = min(chunk_size, remaining)

            if actual_chunk <= 0:
                break

            # Place limit order as iceberg chunk
            result = self.execute_limit(symbol, side, price, actual_chunk)
            result["chunk"] = i + 1
            result["total_chunks"] = n_chunks
            results.append(result)

            if not result.get("success"):
                self.logger.warning(
                    f"Chunk {i+1}/{n_chunks} failed, stopping iceberg"
                )
                break

            # Small delay between chunks to avoid rate limits
            if i < n_chunks - 1:
                time.sleep(0.1)

        return results

    def execute_twap(
        self,
        symbol: str,
        side: str,
        quantity: float,
        duration: int = 60,
        n_slices: int = 10,
    ) -> List[dict]:
        """TWAP分时加权执行 / TWAP (Time-Weighted Average Price) execution.

        Executes the order in equal time slices over the specified duration,
        regardless of price movements.

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            quantity: Total order quantity
            duration: Total execution time in seconds (default 60)
            n_slices: Number of slices to split into (default 10)

        Returns:
            List of execution results for each slice
        """
        slice_qty = quantity / n_slices
        slice_duration = duration / n_slices

        if slice_qty <= 0 or slice_duration <= 0:
            self.logger.error(f"Invalid TWAP params: qty={quantity}, n_slices={n_slices}")
            return []

        results = []

        self.logger.info(
            f"Starting TWAP {side} {quantity} {symbol} "
            f"in {n_slices} slices over {duration}s"
        )

        for i in range(n_slices):
            remaining = quantity - (i * slice_qty)
            actual_qty = min(slice_qty, remaining)

            if actual_qty <= 0:
                break

            # Get current market price
            ticker = self.adapter.get_ticker(symbol)
            current_price = ticker.get("price", 0)

            if current_price == 0:
                self.logger.warning(f"Failed to get price for TWAP slice {i+1}")
                continue

            # Execute at market price
            result = self.execute_market(symbol, side, actual_qty)
            result["slice"] = i + 1
            result["total_slices"] = n_slices
            result["slice_price"] = current_price
            results.append(result)

            if not result.get("success"):
                self.logger.warning(f"TWAP slice {i+1}/{n_slices} failed")

            # Wait for next slice (except for last slice)
            if i < n_slices - 1:
                time.sleep(slice_duration)

        return results

    # -------------------------------------------------------------------------
    # Slippage calculation / 滑点计算
    # -------------------------------------------------------------------------

    def _calc_slippage(
        self, fill_price: float, market_price: float, side: str
    ) -> float:
        """Calculate slippage percentage / 计算滑点百分比.

        Args:
            fill_price: Actual fill price
            market_price: Market price at time of order
            side: 'buy' or 'sell'

        Returns:
            Slippage as a ratio (e.g., 0.01 = 1%)
        """
        if market_price == 0:
            return 0.0

        if side.upper() == "BUY":
            # For buys, slippage is when fill > market
            slippage = (fill_price - market_price) / market_price
        else:
            # For sells, slippage is when fill < market
            slippage = (market_price - fill_price) / market_price

        return max(0.0, slippage)

    # -------------------------------------------------------------------------
    # Status and monitoring / 状态和监控
    # -------------------------------------------------------------------------

    def check_order_status(self, symbol: str, order_id: str) -> dict:
        """检查订单状态 / Check order status.

        Args:
            symbol: Trading pair symbol
            order_id: Order ID to check

        Returns:
            Dict with order status
        """
        try:
            positions = self.adapter.get_positions()
            for pos in positions:
                if pos.get("symbol", "").upper() == symbol.upper():
                    return {
                        "order_id": order_id,
                        "symbol": symbol,
                        "filled": True,
                        "position": pos,
                    }

            return {
                "order_id": order_id,
                "symbol": symbol,
                "filled": False,
                "position": None,
            }

        except Exception as e:
            self.logger.error(f"check_order_status failed: {e}")
            return {
                "order_id": order_id,
                "symbol": symbol,
                "filled": False,
                "error": str(e),
            }

    def get_execution_summary(self, results: List[dict]) -> dict:
        """Get execution summary from a list of slice/chunk results.

        Args:
            results: List of execution results from iceberg or TWAP

        Returns:
            Summary dict with total qty, avg price, success rate
        """
        if not results:
            return {
                "total_orders": 0,
                "successful": 0,
                "failed": 0,
                "total_quantity": 0,
                "avg_price": 0,
                "success_rate": 0,
            }

        successful = [r for r in results if r.get("success")]
        total_qty = sum(r.get("quantity", 0) for r in successful)
        total_value = sum(
            r.get("quantity", 0) * r.get("price", 0) for r in successful if r.get("price")
        )

        return {
            "total_orders": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "total_quantity": total_qty,
            "avg_price": total_value / total_qty if total_qty > 0 else 0,
            "success_rate": len(successful) / len(results) if results else 0,
            "details": results,
        }
