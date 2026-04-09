"""
Trade Flow Factors (A28-A39).

Tick-by-tick trade arrival and directional (buyer/seller-initiated) metrics
from Chinese A-Share level-2 trade data (逐笔成交数据).

Factor Groups
-------------
A28-A29: VWAP of all trades (5-min window + cumulative)
A30-A31: VWAP of buyer/seller-initiated trades (5-min window)
A32-A33: Count of buyer/seller-initiated trades (60s window)
A34-A35: Volume of buyer/seller-initiated trades (60s window)
A36-A37: Trade direction ratios (count ratio + volume ratio, 60s)
A38-A39: Cumulative trade direction ratios (total count + volume)

Data Schema (逐笔成交数据)
--------------------------
Exchflg         : Exchange flag (int)
Code            : Stock code (string)
Code_Mkt        : Market code, e.g. "000001.SZ" (string)
Qdate           : Trading date, "YYYY-MM-DD" (string)
Qtime           : Trading time, "HH:MM:SS" (string)
SetNo           : Session number (int)
RecNo           : Trade record number (int)
BuyOrderRecNo   : Buy order record number (int)
SellOrderRecNo  : Sell order record number (int)
Tprice          : Trade price (float)
Tvolume         : Trade volume (float)
Tsum            : Trade amount (float)
Tvolume_accu    : Cumulative volume (float)
OrderKind       : Order kind (string)
FunctionCode    : "F"=fill (regular trade), "C"=cancel (string)
Trdirec         : Trade direction: "5"=buyer-initiated, "1"=seller-initiated (string)

Note on Trade Direction
-----------------------
In Chinese A-Share level-2 data, Trdirec indicates which side initiated the trade:
  - "5" = buyer-initiated ( uptick,主动买)
  - "1" = seller-initiated (downtick,主动卖)

Key Ratios
----------
Order-to-Trade Ratio (OTR)     : A1 / A32  (order flow vs trade flow intensity)
Trade Imbalance                : (A34 - A35) / (A34 + A35)  (signed volume imbalance)
Cumulative Trade Imbalance     : A39  (cumulative volume-based direction ratio)

Usage
-----
from quant_trading.factors.trade_flow import TradeFlowFactors
fc = TradeFlowFactors(trade_path="./data/trade.csv", factors_index_path="./factors/index.csv")
A32 = fc.calculate_A32()
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

__all__ = ["TradeFlowFactors"]


class TradeFlowFactors:
    """
    Trade flow factor calculator for Chinese A-Share tick data.

    Wraps the high-frequency FactorCalculator for trade-side factors only.
    """

    def __init__(
        self,
        trade_path: str,
        order_path: Optional[str] = None,
        factors_index_second_path: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        trade_path : str
            Path to trade CSV.
        order_path : str, optional
            Path to order CSV (needed for A20-A23 ratio factors).
        factors_index_second_path : str, optional
            Path to second-level index. If not provided, constructed from trade data.
        """
        from quant_trading.factors.high_freq import FactorCalculator, ORDER_DTYPE, TRADE_DTYPE

        self.trade = pd.read_csv(trade_path, dtype=TRADE_DTYPE)

        if order_path:
            self.order = pd.read_csv(order_path, dtype=ORDER_DTYPE)
        else:
            self.order = None

        # Build index from trade data if not provided
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
                self.trade[["Code_Mkt", "Qdate", "Qtime"]]
                .drop_duplicates()
                .set_index(["Code_Mkt", "Qdate", "Qtime"])
            )

        # Delegate to FactorCalculator
        self._calc = FactorCalculator(
            order_path=order_path or "",
            trade_path=trade_path,
            factors_index_second_path="",  # bypass file read
        )
        self._calc.order = self.order
        self._calc.trade = self.trade
        self._calc.factors_index_second = self.factors_index_second

    # ---- Trade VWAP factors ----

    def calculate_A28(self) -> pd.DataFrame:
        """VWAP of all trades in the last 5 min."""
        return self._calc.calculate_A28()

    def calculate_A29(self) -> pd.DataFrame:
        """VWAP of all trades up to that time (cumulative)."""
        return self._calc.calculate_A29()

    def calculate_A30(self) -> pd.DataFrame:
        """VWAP of buyer-initiated trades in the last 5 min."""
        return self._calc.calculate_A30()

    def calculate_A31(self) -> pd.DataFrame:
        """VWAP of seller-initiated trades in the last 5 min."""
        return self._calc.calculate_A31()

    # ---- Trade direction count factors ----

    def calculate_A32(self) -> pd.DataFrame:
        """Number of buyer-initiated trades in the last 60 s."""
        return self._calc.calculate_A32()

    def calculate_A33(self) -> pd.DataFrame:
        """Number of seller-initiated trades in the last 60 s."""
        return self._calc.calculate_A33()

    # ---- Trade direction volume factors ----

    def calculate_A34(self) -> pd.DataFrame:
        """Quantity (volume) of buyer-initiated trades in the last 60 s."""
        return self._calc.calculate_A34()

    def calculate_A35(self) -> pd.DataFrame:
        """Quantity (volume) of seller-initiated trades in the last 60 s."""
        return self._calc.calculate_A35()

    # ---- Trade direction ratio factors ----

    def calculate_A36(self) -> pd.DataFrame:
        """Ratio: # buyer-initiated / # seller-initiated trades in last 60 s."""
        return self._calc.calculate_A36()

    def calculate_A37(self) -> pd.DataFrame:
        """Ratio: qty buyer-initiated / qty seller-initiated trades in last 60 s."""
        return self._calc.calculate_A37()

    def calculate_A38(self) -> pd.DataFrame:
        """Cumulative ratio: total # buyer-initiated / total # seller-initiated trades."""
        return self._calc.calculate_A38()

    def calculate_A39(self) -> pd.DataFrame:
        """Cumulative ratio: total qty buyer-initiated / total qty seller-initiated trades."""
        return self._calc.calculate_A39()

    # ------------------------------------------------------------------
    # Convenience derived metrics
    # ------------------------------------------------------------------

    def trade_imbalance(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Volume-weighted trade imbalance: (V_buy - V_sell) / (V_buy + V_sell).

        Range: [-1, +1]
        > 0 indicates buying pressure
        < 0 indicates selling pressure

        Parameters
        ----------
        data : pd.DataFrame, optional
            Trade data. Uses self.trade if not provided.

        Returns
        -------
        pd.DataFrame
            Trade imbalance series with standard factor columns.
        """
        if data is None:
            data = self.trade

        buyer_vol = self.calculate_A34(data)
        seller_vol = self.calculate_A35(data)

        buyer_vol.set_index(["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True)
        seller_vol.set_index(["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True)

        total = buyer_vol["A34"] + seller_vol["A35"]
        imbalance = (buyer_vol["A34"] - seller_vol["A35"]) / total.replace(0, np.nan)

        result = imbalance.reset_index()
        result.columns = ["ticker_str", "info_date_ymd", "info_time_hms", "trade_imbalance"]
        return result

    def order_trade_ratio(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Order-to-Trade Ratio (OTR): # orders / # trades in last 60 s.

        High OTR indicates many orders relative to actual executions
        (potential large latent liquidity).

        Parameters
        ----------
        data : pd.DataFrame, optional
            Order data. Uses self.order if not provided.

        Returns
        -------
        pd.DataFrame
            OTR series with standard factor columns.
        """
        if self.order is None:
            raise ValueError("order_path required for order_trade_ratio")

        if data is None:
            data = self.order

        order_count = self._calc.calculate_A1(data)
        trade_count = self.calculate_A32()

        order_count.set_index(["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True)
        trade_count.set_index(["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True)

        otr = order_count["A1"] / trade_count["A32"].replace(0, np.nan)

        result = otr.reset_index()
        result.columns = ["ticker_str", "info_date_ymd", "info_time_hms", "order_trade_ratio"]
        return result
