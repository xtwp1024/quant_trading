"""
Order Flow Factors (A1-A16, A20-A27).

Tick-by-tick order arrival, buy/sell classification, and cancellation metrics
from Chinese A-Share level-2 order data (逐笔委托数据).

Factor Groups
-------------
A1-A4:   Order arrival counts and quantities (60s window + cumulative)
A5-A8:   Buy/sell order breakdown (60s window)
A9:      Fill-and-kill orders (NaN — not applicable in China)
A10-A16: Cancellation metrics (counts, quantities, cumulative, VWAP)
A20-A23: Cancellation ratios (vs. arrived orders)
A24-A27: Order size statistics (5-min mean/std for buy/sell)

Data Schema (逐笔委托数据)
--------------------------
Exchflg      : Exchange flag (int)
Code         : Stock code (string)
Code_Mkt     : Market code, e.g. "000001.SZ" (string)
Qdate        : Trading date, "YYYY-MM-DD" (string)
Qtime        : Trading time, "HH:MM:SS" (string)
SetNo        : Session number (int)
OrderRecNo   : Order record number (int)
OrderPr      : Order price (float)
OrderVol     : Order volume (float)
OrderKind    : Order kind (string)
FunctionCode : "1"=buy, "2"=sell (string)

Usage
-----
from quant_trading.factors.order_flow import OrderFlowFactors
fc = OrderFlowFactors(order_path="./data/order.csv", factors_index_path="./factors/index.csv")
A1 = fc.calculate_A1()
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Union

__all__ = ["OrderFlowFactors"]


class OrderFlowFactors:
    """
    Order flow factor calculator for Chinese A-Share tick data.

    Wraps the high-frequency FactorCalculator for order-side factors only.
    For complete 39-factor coverage, use HighFrequencyFactors directly.
    """

    def __init__(
        self,
        order_path: str,
        trade_path: Optional[str] = None,
        factors_index_second_path: Optional[str] = None,
    ) -> None:
        """
        Initialize with order (and optionally trade) data paths.

        Parameters
        ----------
        order_path : str
            Path to order CSV.
        trade_path : str, optional
            Path to trade CSV (needed for cancellation factors A10-A16).
        factors_index_second_path : str, optional
            Path to second-level index. If not provided, constructed from order data.
        """
        from quant_trading.factors.high_freq import FactorCalculator, ORDER_DTYPE, TRADE_DTYPE

        self.order = pd.read_csv(order_path, dtype=ORDER_DTYPE)

        if trade_path:
            self.trade = pd.read_csv(trade_path, dtype=TRADE_DTYPE)
        else:
            self.trade = None

        # Build index from order data if not provided
        if factors_index_second_path:
            import os
            if os.path.exists(factors_index_second_path):
                self.factors_index_second = pd.read_csv(factors_index_second_path)
                self.factors_index_second = self.factors_index_second.set_index(
                    ["Code_Mkt", "Qdate", "Qtime"]
                )
            else:
                raise FileNotFoundError(factors_index_second_path)
        else:
            self.factors_index_second = (
                self.order[["Code_Mkt", "Qdate", "Qtime"]]
                .drop_duplicates()
                .set_index(["Code_Mkt", "Qdate", "Qtime"])
            )

        # Delegate to FactorCalculator for actual computations
        self._calc = FactorCalculator(
            order_path=order_path,
            trade_path=trade_path or "",
            factors_index_second_path="",  # bypass file read; we set index directly
        )
        self._calc.order = self.order
        self._calc.trade = self.trade
        self._calc.factors_index_second = self.factors_index_second

    # ---- Order arrival factors ----

    def calculate_A1(self) -> pd.DataFrame:
        """Number of orders arriving in the last 60 s."""
        return self._calc.calculate_A1()

    def calculate_A2(self) -> pd.DataFrame:
        """Total number of arrived orders up to that time (cumulative)."""
        return self._calc.calculate_A2()

    def calculate_A3(self) -> pd.DataFrame:
        """Quantity of arrived orders in the last 60 s."""
        return self._calc.calculate_A3()

    def calculate_A4(self) -> pd.DataFrame:
        """Total quantity of arrived orders up to that time (cumulative)."""
        return self._calc.calculate_A4()

    def calculate_A5(self) -> pd.DataFrame:
        """Number of buy orders arriving in the last 60 s."""
        return self._calc.calculate_A5()

    def calculate_A6(self) -> pd.DataFrame:
        """Number of sell orders arriving in the last 60 s."""
        return self._calc.calculate_A6()

    def calculate_A7(self) -> pd.DataFrame:
        """Quantity of buy orders arriving in the last 60 s."""
        return self._calc.calculate_A7()

    def calculate_A8(self) -> pd.DataFrame:
        """Quantity of sell orders arriving in the last 60 s."""
        return self._calc.calculate_A8()

    def calculate_A9(self) -> pd.DataFrame:
        """
        Number of fill-and-kill orders in the last 60 s.
        China A-Share does not support FOK — returns NaN.
        """
        return self._calc.calculate_A9()

    # ---- Cancellation factors (require trade data) ----

    def calculate_A10(self) -> pd.DataFrame:
        """Number of cancelled orders in the last 60 s."""
        if self.trade is None:
            raise ValueError("trade_path required for A10-A16 factors")
        return self._calc.calculate_A10()

    def calculate_A11(self) -> pd.DataFrame:
        """Quantity of cancelled orders in the last 60 s."""
        if self.trade is None:
            raise ValueError("trade_path required for A10-A16 factors")
        return self._calc.calculate_A11()

    def calculate_A12(self) -> pd.DataFrame:
        """Number of cancelled buy orders in the last 60 s."""
        if self.trade is None:
            raise ValueError("trade_path required for A10-A16 factors")
        return self._calc.calculate_A12()

    def calculate_A13(self) -> pd.DataFrame:
        """Number of cancelled sell orders in the last 60 s."""
        if self.trade is None:
            raise ValueError("trade_path required for A10-A16 factors")
        return self._calc.calculate_A13()

    def calculate_A14(self) -> pd.DataFrame:
        """Quantity of cancelled buy orders in the last 60 s."""
        if self.trade is None:
            raise ValueError("trade_path required for A10-A16 factors")
        return self._calc.calculate_A14()

    def calculate_A15(self) -> pd.DataFrame:
        """Quantity of cancelled sell orders in the last 60 s."""
        if self.trade is None:
            raise ValueError("trade_path required for A10-A16 factors")
        return self._calc.calculate_A15()

    def calculate_A16(self) -> pd.DataFrame:
        """Total number of cancelled orders up to that time (cumulative)."""
        if self.trade is None:
            raise ValueError("trade_path required for A10-A16 factors")
        return self._calc.calculate_A16()

    # ---- VWAP cancellation factors ----

    def calculate_A17(self) -> pd.DataFrame:
        """VWAP of cancelled orders up to that time."""
        if self.trade is None:
            raise ValueError("trade_path required for A17-A19 factors")
        return self._calc.calculate_A17()

    def calculate_A18(self) -> pd.DataFrame:
        """VWAP of cancelled buy orders up to that time."""
        if self.trade is None:
            raise ValueError("trade_path required for A17-A19 factors")
        return self._calc.calculate_A18()

    def calculate_A19(self) -> pd.DataFrame:
        """VWAP of cancelled sell orders up to that time."""
        if self.trade is None:
            raise ValueError("trade_path required for A17-A19 factors")
        return self._calc.calculate_A19()

    # ---- Order ratio factors ----

    def calculate_A20(self) -> pd.DataFrame:
        """Ratio: # cancelled orders / # arrived orders in last 60 s."""
        if self.trade is None:
            raise ValueError("trade_path required for A20-A23 factors")
        return self._calc.calculate_A20()

    def calculate_A21(self) -> pd.DataFrame:
        """Ratio: qty cancelled / qty arrived in last 60 s."""
        if self.trade is None:
            raise ValueError("trade_path required for A20-A23 factors")
        return self._calc.calculate_A21()

    def calculate_A22(self) -> pd.DataFrame:
        """Ratio: total # cancelled / total # arrived up to that time."""
        if self.trade is None:
            raise ValueError("trade_path required for A20-A23 factors")
        return self._calc.calculate_A22()

    def calculate_A23(self) -> pd.DataFrame:
        """Ratio: total qty cancelled / total qty arrived up to that time."""
        if self.trade is None:
            raise ValueError("trade_path required for A20-A23 factors")
        return self._calc.calculate_A23()

    # ---- Order size statistics (5-min resampled) ----

    def calculate_A24(self) -> pd.DataFrame:
        """Average quantity of buy orders in the last 5 min."""
        return self._calc.calculate_A24()

    def calculate_A25(self) -> pd.DataFrame:
        """Average quantity of sell orders in the last 5 min."""
        return self._calc.calculate_A25()

    def calculate_A26(self) -> pd.DataFrame:
        """Volatility (std) of buy order quantity in the last 5 min."""
        return self._calc.calculate_A26()

    def calculate_A27(self) -> pd.DataFrame:
        """Volatility (std) of sell order quantity in the last 5 min."""
        return self._calc.calculate_A27()
