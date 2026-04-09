"""
High-Frequency Factors — 39 Chinese A-Share High-Frequency Factors.

Based on tick-by-tick order flow and trade flow data from Chinese A-Stocks.
Original implementation: https://github.com/jeremy-feng/high-frequency-factors

Factor Categories
------------------
Order Flow (A1-A16):     Order arrival, buy/sell, cancellation metrics
VWAP Cancellation (A17-A19): Volume-weighted avg price of cancelled orders
Order Ratios (A20-A27):  Cancellation ratios, order size statistics
Trade Flow (A28-A39):    Trade arrival, buyer/seller-initiated metrics

Data Sources
------------
- Order data: 逐笔委托数据 (order_stkhf*.csv)
- Trade data: 逐笔成交数据 (trade_stkhf*.csv)
- Index:      factors_index_second.csv (second-level time index)

Trading Hours (China A-Share)
------------------------------
Morning:  9:30-11:30 (first trade at 9:31, last at 11:30)
Afternoon: 13:00-15:00 (first trade at 13:01, last at 15:00)
Total: 240 minutes per day, 1200 seconds per day

Sampling: left-open, right-closed intervals
Resampling: second-level -> minute-level via .resample('1Min', label="right", closed="right")

Dependencies
-----------
pandas
numpy
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

__all__ = ["HighFrequencyFactors", "FactorCalculator"]


# ----------------------------------------------------------------------
# Data type schemas (matching CSRC standard for Chinese tick data)
# ----------------------------------------------------------------------

ORDER_DTYPE = {
    "Exchflg": "int",
    "Code": "string",
    "Code_Mkt": "string",
    "Qdate": "string",
    "Qtime": "string",
    "SetNo": "int",
    "OrderRecNo": "int",
    "OrderPr": "float",
    "OrderVol": "float",
    "OrderKind": "string",
    "FunctionCode": "string",
}

TRADE_DTYPE = {
    "Exchflg": "int",
    "Code": "string",
    "Code_Mkt": "string",
    "Qdate": "string",
    "Qtime": "string",
    "SetNo": "int",
    "RecNo": "int",
    "BuyOrderRecNo": "int",
    "SellOrderRecNo": "int",
    "Tprice": "float",
    "Tvolume": "float",
    "Tsum": "float",
    "Tvolume_accu": "float",
    "OrderKind": "string",
    "FunctionCode": "string",
    "Trdirec": "string",
}

# FunctionCode meanings (Chinese A-Share)
# Order side:  "1" = buy, "2" = sell
# Trade side:  "F" = fill (regular trade), "C" = cancel
# Trade direction (Trdirec): "5" = buyer-initiated, "1" = seller-initiated


# ----------------------------------------------------------------------
# Factor Calculator
# ----------------------------------------------------------------------

class FactorCalculator:
    """
    Core calculator for the 39 high-frequency factors.

    Parameters
    ----------
    order_path : str
        Path to order data CSV (逐笔委托数据)
    trade_path : str
        Path to trade data CSV (逐笔成交数据)
    factors_index_second_path : str
        Path to factors index CSV with second-level time index
    """

    def __init__(
        self,
        order_path: str,
        trade_path: str,
        factors_index_second_path: str,
    ) -> None:
        import os

        self.order_path = order_path
        self.trade_path = trade_path
        self.factors_index_second_path = factors_index_second_path

        self.order = pd.read_csv(order_path, dtype=ORDER_DTYPE)
        self.trade = pd.read_csv(trade_path, dtype=TRADE_DTYPE)

        if os.path.exists(factors_index_second_path):
            self.factors_index_second = pd.read_csv(factors_index_second_path)
            self.factors_index_second = self.factors_index_second.set_index(
                ["Code_Mkt", "Qdate", "Qtime"]
            )
        else:
            raise FileNotFoundError(
                f"factors_index_second not found at {factors_index_second_path}"
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _format_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize output columns: ticker_str, info_date_ymd, info_time_hms."""
        data = data.reset_index()
        data.rename(
            columns={
                "Code_Mkt": "ticker_str",
                "Qdate": "info_date_ymd",
                "Qtime": "info_time_hms",
            },
            inplace=True,
        )
        data["ticker_str"] = data["ticker_str"].apply(lambda x: x.split(".")[0])
        data["info_date_ymd"] = data["info_date_ymd"].apply(
            lambda x: int(x.replace("-", ""))
        )
        data["info_time_hms"] = data["info_time_hms"].apply(
            lambda x: int(x.replace(":", ""))
        )
        return data

    def _reindex_fill_zero(self, series: pd.Series) -> pd.Series:
        """Reindex to factor_index and fill missing with 0."""
        return series.reindex(self.factors_index_second.index, fill_value=0)

    def _rolling_sum_60s(
        self, series: pd.Series, group_cols: list[str] = ["Code_Mkt", "Qdate"]
    ) -> pd.Series:
        """Rolling 60-second sum grouped by ticker and date."""
        return (
            series.groupby(by=group_cols)
            .rolling(60, closed="left")
            .sum()
            .droplevel(level=[0, 1])
        )

    # ------------------------------------------------------------------
    # Order Flow Factors (A1-A9)
    # ------------------------------------------------------------------

    def calculate_A1(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Number of orders arriving in the last 60 s."""
        if data is None:
            data = self.order
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderRecNo": "A1"}, inplace=True)
        factor["A1"] = factor["A1"].astype(float)
        return factor

    def calculate_A2(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Total number of arrived orders up to that time (cumulative)."""
        if data is None:
            data = self.order
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        factor = self._reindex_fill_zero(factor)
        factor = factor.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderRecNo": "A2"}, inplace=True)
        factor["A2"] = factor["A2"].astype(float)
        return factor

    def calculate_A3(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Quantity (volume) of arrived orders in the last 60 s."""
        if data is None:
            data = self.order
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderVol": "A3"}, inplace=True)
        factor["A3"] = factor["A3"].astype(float)
        return factor

    def calculate_A4(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Total quantity of arrived orders up to that time (cumulative)."""
        if data is None:
            data = self.order
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        factor = self._reindex_fill_zero(factor)
        factor = factor.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderVol": "A4"}, inplace=True)
        factor["A4"] = factor["A4"].astype(float)
        return factor

    def calculate_A5(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Number of buy orders arriving in the last 60 s."""
        if data is None:
            data = self.order
        data = data[data["FunctionCode"] == "1"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderRecNo": "A5"}, inplace=True)
        factor["A5"] = factor["A5"].astype(float)
        return factor

    def calculate_A6(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Number of sell orders arriving in the last 60 s."""
        if data is None:
            data = self.order
        data = data[data["FunctionCode"] == "2"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderRecNo": "A6"}, inplace=True)
        factor["A6"] = factor["A6"].astype(float)
        return factor

    def calculate_A7(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Quantity of buy orders arriving in the last 60 s."""
        if data is None:
            data = self.order
        data = data[data["FunctionCode"] == "1"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderVol": "A7"}, inplace=True)
        factor["A7"] = factor["A7"].astype(float)
        return factor

    def calculate_A8(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Quantity of sell orders arriving in the last 60 s."""
        if data is None:
            data = self.order
        data = data[data["FunctionCode"] == "2"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderVol": "A8"}, inplace=True)
        factor["A8"] = factor["A8"].astype(float)
        return factor

    def calculate_A9(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Number of fill and kill orders arriving in the last 60 s.
        Note: China A-Share does not have FOK orders — returns NaN.
        """
        factor = pd.DataFrame(index=self.factors_index_second.index, columns=["A9"])
        factor["A9"] = np.NaN
        factor = self._format_factor(factor)
        factor["A9"] = factor["A9"].astype(float)
        return factor

    # ------------------------------------------------------------------
    # Cancellation Factors (A10-A16)
    # ------------------------------------------------------------------

    def calculate_A10(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Number of cancelled orders in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[data["FunctionCode"] == "C"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"RecNo": "A10"}, inplace=True)
        factor["A10"] = factor["A10"].astype(float)
        return factor

    def calculate_A11(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Quantity of cancelled orders in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[data["FunctionCode"] == "C"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"Tvolume": "A11"}, inplace=True)
        factor["A11"] = factor["A11"].astype(float)
        return factor

    def calculate_A12(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Number of cancelled buy orders in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[(data["FunctionCode"] == "C") & (data["BuyOrderRecNo"] != 0.0)]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"RecNo": "A12"}, inplace=True)
        factor["A12"] = factor["A12"].astype(float)
        return factor

    def calculate_A13(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Number of cancelled sell orders in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[(data["FunctionCode"] == "C") & (data["SellOrderRecNo"] != 0.0)]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"RecNo": "A13"}, inplace=True)
        factor["A13"] = factor["A13"].astype(float)
        return factor

    def calculate_A14(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Quantity of cancelled buy orders in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[(data["FunctionCode"] == "C") & (data["BuyOrderRecNo"] != 0.0)]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"Tvolume": "A14"}, inplace=True)
        factor["A14"] = factor["A14"].astype(float)
        return factor

    def calculate_A15(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Quantity of cancelled sell orders in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[(data["FunctionCode"] == "C") & (data["SellOrderRecNo"] != 0.0)]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        factor = self._reindex_fill_zero(factor)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"Tvolume": "A15"}, inplace=True)
        factor["A15"] = factor["A15"].astype(float)
        return factor

    def calculate_A16(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Total number of cancelled orders up to that time (cumulative)."""
        if data is None:
            data = self.trade
        data = data[data["FunctionCode"] == "C"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        factor = self._reindex_fill_zero(factor)
        factor = factor.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)
        factor = self._format_factor(factor)
        factor.rename(columns={"RecNo": "A16"}, inplace=True)
        factor["A16"] = factor["A16"].astype(float)
        return factor

    # ------------------------------------------------------------------
    # VWAP Cancellation Factors (A17-A19)
    # ------------------------------------------------------------------

    def calculate_A17(
        self,
        trade_data: Optional[pd.DataFrame] = None,
        order_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Volume weighted average price of cancelled orders up to that time."""
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order

        trade_data = trade_data[trade_data["FunctionCode"] == "C"].copy()
        trade_data = trade_data[
            ["Code_Mkt", "Qdate", "Qtime", "BuyOrderRecNo", "SellOrderRecNo", "Tvolume"]
        ]
        trade_data["OrderRecNo"] = (
            trade_data["BuyOrderRecNo"] + trade_data["SellOrderRecNo"]
        )

        order_data = order_data[
            ["Code_Mkt", "Qdate", "OrderRecNo", "OrderPr"]
        ]

        trade_order = pd.merge(
            trade_data,
            order_data,
            on=["Code_Mkt", "Qdate", "OrderRecNo"],
        )
        trade_order["Tamount"] = trade_order["Tvolume"] * trade_order["OrderPr"]
        trade_order["Tamount_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tamount"]
            .cumsum()
            .shift(1)
        )
        trade_order["Tvolume_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tvolume"]
            .cumsum()
            .shift(1)
        )
        trade_order["VWAP"] = (
            trade_order["Tamount_cumsum"] / trade_order["Tvolume_cumsum"]
        )

        factor = trade_order[["Code_Mkt", "Qdate", "Qtime", "VWAP"]]
        factor = factor.drop_duplicates(
            subset=["Code_Mkt", "Qdate", "Qtime"],
            keep="last",
        )
        factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
        factor = factor.reindex(self.factors_index_second.index, method="ffill")
        factor = self._format_factor(factor)
        factor.rename(columns={"VWAP": "A17"}, inplace=True)
        factor["A17"] = factor["A17"].astype(float)
        return factor

    def calculate_A18(
        self,
        trade_data: Optional[pd.DataFrame] = None,
        order_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Volume weighted average price of cancelled buy orders up to that time."""
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order

        trade_data = trade_data[
            (trade_data["FunctionCode"] == "C") & (trade_data["BuyOrderRecNo"] != 0.0)
        ].copy()
        trade_data = trade_data[
            ["Code_Mkt", "Qdate", "Qtime", "BuyOrderRecNo", "Tvolume"]
        ]

        order_data = order_data[
            ["Code_Mkt", "Qdate", "OrderRecNo", "OrderPr"]
        ]

        trade_order = pd.merge(
            trade_data,
            order_data,
            left_on=["Code_Mkt", "Qdate", "BuyOrderRecNo"],
            right_on=["Code_Mkt", "Qdate", "OrderRecNo"],
        )
        trade_order["Tamount"] = trade_order["Tvolume"] * trade_order["OrderPr"]
        trade_order["Tamount_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tamount"]
            .cumsum()
            .shift(1)
        )
        trade_order["Tvolume_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tvolume"]
            .cumsum()
            .shift(1)
        )
        trade_order["VWAP"] = (
            trade_order["Tamount_cumsum"] / trade_order["Tvolume_cumsum"]
        )

        factor = trade_order[["Code_Mkt", "Qdate", "Qtime", "VWAP"]]
        factor = factor.drop_duplicates(
            subset=["Code_Mkt", "Qdate", "Qtime"],
            keep="last",
        )
        factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
        factor = factor.reindex(self.factors_index_second.index, method="ffill")
        factor = self._format_factor(factor)
        factor.rename(columns={"VWAP": "A18"}, inplace=True)
        factor["A18"] = factor["A18"].astype(float)
        return factor

    def calculate_A19(
        self,
        trade_data: Optional[pd.DataFrame] = None,
        order_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Volume weighted average price of cancelled sell orders up to that time."""
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order

        trade_data = trade_data[
            (trade_data["FunctionCode"] == "C") & (trade_data["SellOrderRecNo"] != 0.0)
        ].copy()
        trade_data = trade_data[
            ["Code_Mkt", "Qdate", "Qtime", "SellOrderRecNo", "Tvolume"]
        ]

        order_data = order_data[
            ["Code_Mkt", "Qdate", "OrderRecNo", "OrderPr"]
        ]

        trade_order = pd.merge(
            trade_data,
            order_data,
            left_on=["Code_Mkt", "Qdate", "SellOrderRecNo"],
            right_on=["Code_Mkt", "Qdate", "OrderRecNo"],
        )
        trade_order["Tamount"] = trade_order["Tvolume"] * trade_order["OrderPr"]
        trade_order["Tamount_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tamount"]
            .cumsum()
            .shift(1)
        )
        trade_order["Tvolume_cumsum"] = (
            trade_order.groupby(["Code_Mkt", "Qdate"])["Tvolume"]
            .cumsum()
            .shift(1)
        )
        trade_order["VWAP"] = (
            trade_order["Tamount_cumsum"] / trade_order["Tvolume_cumsum"]
        )

        factor = trade_order[["Code_Mkt", "Qdate", "Qtime", "VWAP"]]
        factor = factor.drop_duplicates(
            subset=["Code_Mkt", "Qdate", "Qtime"],
            keep="last",
        )
        factor.set_index(["Code_Mkt", "Qdate", "Qtime"], inplace=True)
        factor = factor.reindex(self.factors_index_second.index, method="ffill")
        factor = self._format_factor(factor)
        factor.rename(columns={"VWAP": "A19"}, inplace=True)
        factor["A19"] = factor["A19"].astype(float)
        return factor

    # ------------------------------------------------------------------
    # Order Ratio Factors (A20-A27)
    # ------------------------------------------------------------------

    def calculate_A20(
        self,
        trade_data: Optional[pd.DataFrame] = None,
        order_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Ratio: # cancelled orders / # arrived orders in last 60 s."""
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order

        cancelled = trade_data[trade_data["FunctionCode"] == "C"]
        cancelled = cancelled.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        cancelled = cancelled.reindex(self.factors_index_second.index).fillna(0)
        cancelled = self._rolling_sum_60s(cancelled)

        arrived = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        arrived = arrived.reindex(self.factors_index_second.index).fillna(0)
        arrived = self._rolling_sum_60s(arrived)

        factor = cancelled / arrived
        factor.name = "A20"
        factor = self._format_factor(factor)
        factor["A20"] = factor["A20"].astype(float)
        return factor

    def calculate_A21(
        self,
        trade_data: Optional[pd.DataFrame] = None,
        order_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Ratio: qty cancelled orders / qty arrived orders in last 60 s."""
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order

        cancelled = trade_data[trade_data["FunctionCode"] == "C"]
        cancelled = cancelled.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        cancelled = cancelled.reindex(self.factors_index_second.index).fillna(0)
        cancelled = self._rolling_sum_60s(cancelled)

        arrived = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        arrived = arrived.reindex(self.factors_index_second.index).fillna(0)
        arrived = self._rolling_sum_60s(arrived)

        factor = cancelled / arrived
        factor.name = "A21"
        factor = self._format_factor(factor)
        factor["A21"] = factor["A21"].astype(float)
        return factor

    def calculate_A22(
        self,
        trade_data: Optional[pd.DataFrame] = None,
        order_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Ratio: total # cancelled / total # arrived up to that time (cumulative)."""
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order

        cancelled = trade_data[trade_data["FunctionCode"] == "C"]
        cancelled = cancelled.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        cancelled = cancelled.reindex(self.factors_index_second.index).fillna(0)
        cancelled = cancelled.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)

        arrived = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderRecNo"].count()
        arrived = arrived.reindex(self.factors_index_second.index).fillna(0)
        arrived = arrived.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)

        factor = cancelled / arrived
        factor.name = "A22"
        factor = self._format_factor(factor)
        factor["A22"] = factor["A22"].astype(float)
        return factor

    def calculate_A23(
        self,
        trade_data: Optional[pd.DataFrame] = None,
        order_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Ratio: total qty cancelled / total qty arrived up to that time (cumulative)."""
        if trade_data is None:
            trade_data = self.trade
        if order_data is None:
            order_data = self.order

        cancelled = trade_data[trade_data["FunctionCode"] == "C"]
        cancelled = cancelled.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        cancelled = cancelled.reindex(self.factors_index_second.index).fillna(0)
        cancelled = cancelled.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)

        arrived = order_data.groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"].sum()
        arrived = arrived.reindex(self.factors_index_second.index).fillna(0)
        arrived = arrived.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)

        factor = cancelled / arrived
        factor.name = "A23"
        factor = self._format_factor(factor)
        factor["A23"] = factor["A23"].astype(float)
        return factor

    def _resample_5min(self, series: pd.Series, agg: str = "mean") -> pd.Series:
        """Resample second-level series to 5-minute bars."""
        series = series.copy()
        series.index = series.index.set_levels(
            [
                series.index.levels[0],
                series.index.levels[1],
                pd.to_datetime(series.index.levels[2]),
            ]
        )

        if agg == "mean":
            resampled = series.groupby(
                [
                    pd.Grouper(level=0),
                    pd.Grouper(level=1),
                    pd.Grouper(level=2, freq="5min", label="right"),
                ]
            ).mean()
        elif agg == "std":
            resampled = series.groupby(
                [
                    pd.Grouper(level=0),
                    pd.Grouper(level=1),
                    pd.Grouper(level=2, freq="5min", label="right"),
                ]
            ).std()
        else:
            raise ValueError(f"Unsupported aggregation: {agg}")

        resampled.index = resampled.index.set_levels(
            [
                resampled.index.levels[0],
                resampled.index.levels[1],
                resampled.index.levels[2].strftime("%H:%M:%S"),
            ]
        )
        return resampled.reindex(self.factors_index_second.index)

    def calculate_A24(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Average quantity of buy orders in the last 5 min."""
        if data is None:
            data = self.order
        factor = (
            data[data["FunctionCode"] == "1"]
            .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
            .sum()
        )
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        factor = self._resample_5min(factor, agg="mean")
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderVol": "A24"}, inplace=True)
        factor["A24"] = factor["A24"].astype(float)
        return factor

    def calculate_A25(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Average quantity of sell orders in the last 5 min."""
        if data is None:
            data = self.order
        factor = (
            data[data["FunctionCode"] == "2"]
            .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
            .sum()
        )
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        factor = self._resample_5min(factor, agg="mean")
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderVol": "A25"}, inplace=True)
        factor["A25"] = factor["A25"].astype(float)
        return factor

    def calculate_A26(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Volatility (std) of buy order quantity in the last 5 min."""
        if data is None:
            data = self.order
        factor = (
            data[data["FunctionCode"] == "1"]
            .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
            .sum()
        )
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        factor = self._resample_5min(factor, agg="std")
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderVol": "A26"}, inplace=True)
        factor["A26"] = factor["A26"].astype(float)
        return factor

    def calculate_A27(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Volatility (std) of sell order quantity in the last 5 min."""
        if data is None:
            data = self.order
        factor = (
            data[data["FunctionCode"] == "2"]
            .groupby(["Code_Mkt", "Qdate", "Qtime"])["OrderVol"]
            .sum()
        )
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        factor = self._resample_5min(factor, agg="std")
        factor = self._format_factor(factor)
        factor.rename(columns={"OrderVol": "A27"}, inplace=True)
        factor["A27"] = factor["A27"].astype(float)
        return factor

    # ------------------------------------------------------------------
    # Trade Flow Factors (A28-A39)
    # ------------------------------------------------------------------

    def calculate_A28(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """VWAP of trades in the last 5 min."""
        if data is None:
            data = self.trade
        trade_data = data[data["FunctionCode"] == "F"].copy()
        trade_data = trade_data[
            ["Code_Mkt", "Qdate", "Qtime", "Tprice", "Tvolume"]
        ]
        trade_data["Tamount"] = trade_data["Tprice"] * trade_data["Tvolume"]
        factor = trade_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            ["Tvolume", "Tamount"]
        ].sum()

        factor.index = factor.index.set_levels(
            [
                factor.index.levels[0],
                factor.index.levels[1],
                pd.to_datetime(factor.index.levels[2]),
            ]
        )

        factor = factor.groupby(
            [
                pd.Grouper(level=0),
                pd.Grouper(level=1),
                pd.Grouper(level=2, freq="5min", label="right"),
            ]
        ).apply(lambda x: np.sum(x["Tamount"]) / np.sum(x["Tvolume"]))

        factor.index = factor.index.set_levels(
            [
                factor.index.levels[0],
                factor.index.levels[1],
                factor.index.levels[2].strftime("%H:%M:%S"),
            ]
        )

        factor = factor.reindex(self.factors_index_second.index)
        factor.name = "A28"
        factor = self._format_factor(factor)
        factor["A28"] = factor["A28"].astype(float)
        return factor

    def calculate_A29(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """VWAP of trades up to that time (cumulative)."""
        if data is None:
            data = self.trade
        trade_data = data[data["FunctionCode"] == "F"].copy()
        trade_data = trade_data[
            ["Code_Mkt", "Qdate", "Qtime", "Tprice", "Tvolume"]
        ]
        trade_data["Tamount"] = trade_data["Tprice"] * trade_data["Tvolume"]
        factor = trade_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            ["Tvolume", "Tamount"]
        ].sum()
        factor = factor.reindex(self.factors_index_second.index, fill_value=0)
        factor["Tamount_cumsum"] = (
            factor.groupby(["Code_Mkt", "Qdate"])["Tamount"].cumsum().shift(1)
        )
        factor["Tvolume_cumsum"] = (
            factor.groupby(["Code_Mkt", "Qdate"])["Tvolume"].cumsum().shift(1)
        )
        factor["VWAP"] = factor["Tamount_cumsum"] / factor["Tvolume_cumsum"]
        factor = factor[["VWAP"]]
        factor = self._format_factor(factor)
        factor.rename(columns={"VWAP": "A29"}, inplace=True)
        factor["A29"] = factor["A29"].astype(float)
        return factor

    def calculate_A30(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """VWAP of buyer-initiated trades in the last 5 min."""
        if data is None:
            data = self.trade
        trade_data = data[(data["FunctionCode"] == "F") & (data["Trdirec"] == "5")].copy()
        trade_data = trade_data[
            ["Code_Mkt", "Qdate", "Qtime", "Tprice", "Tvolume"]
        ]
        trade_data["Tamount"] = trade_data["Tprice"] * trade_data["Tvolume"]
        factor = trade_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            ["Tvolume", "Tamount"]
        ].sum()

        factor.index = factor.index.set_levels(
            [
                factor.index.levels[0],
                factor.index.levels[1],
                pd.to_datetime(factor.index.levels[2]),
            ]
        )

        factor = factor.groupby(
            [
                pd.Grouper(level=0),
                pd.Grouper(level=1),
                pd.Grouper(level=2, freq="5min", label="right"),
            ]
        ).apply(lambda x: np.sum(x["Tamount"]) / np.sum(x["Tvolume"]))

        factor.index = factor.index.set_levels(
            [
                factor.index.levels[0],
                factor.index.levels[1],
                factor.index.levels[2].strftime("%H:%M:%S"),
            ]
        )

        factor = factor.reindex(self.factors_index_second.index)
        factor.name = "A30"
        factor = self._format_factor(factor)
        factor["A30"] = factor["A30"].astype(float)
        return factor

    def calculate_A31(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """VWAP of seller-initiated trades in the last 5 min."""
        if data is None:
            data = self.trade
        trade_data = data[(data["FunctionCode"] == "F") & (data["Trdirec"] == "1")].copy()
        trade_data = trade_data[
            ["Code_Mkt", "Qdate", "Qtime", "Tprice", "Tvolume"]
        ]
        trade_data["Tamount"] = trade_data["Tprice"] * trade_data["Tvolume"]
        factor = trade_data.groupby(["Code_Mkt", "Qdate", "Qtime"])[
            ["Tvolume", "Tamount"]
        ].sum()

        factor.index = factor.index.set_levels(
            [
                factor.index.levels[0],
                factor.index.levels[1],
                pd.to_datetime(factor.index.levels[2]),
            ]
        )

        factor = factor.groupby(
            [
                pd.Grouper(level=0),
                pd.Grouper(level=1),
                pd.Grouper(level=2, freq="5min", label="right"),
            ]
        ).apply(lambda x: np.sum(x["Tamount"]) / np.sum(x["Tvolume"]))

        factor.index = factor.index.set_levels(
            [
                factor.index.levels[0],
                factor.index.levels[1],
                factor.index.levels[2].strftime("%H:%M:%S"),
            ]
        )

        factor = factor.reindex(self.factors_index_second.index)
        factor.name = "A31"
        factor = self._format_factor(factor)
        factor["A31"] = factor["A31"].astype(float)
        return factor

    def calculate_A32(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Number of buyer-initiated trades in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[data["Trdirec"] == "5"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        factor = factor.reindex(self.factors_index_second.index).fillna(0)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"RecNo": "A32"}, inplace=True)
        factor["A32"] = factor["A32"].astype(float)
        return factor

    def calculate_A33(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Number of seller-initiated trades in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[data["Trdirec"] == "1"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        factor = factor.reindex(self.factors_index_second.index).fillna(0)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"RecNo": "A33"}, inplace=True)
        factor["A33"] = factor["A33"].astype(float)
        return factor

    def calculate_A34(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Quantity of buyer-initiated trades in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[data["Trdirec"] == "5"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        factor = factor.reindex(self.factors_index_second.index).fillna(0)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"Tvolume": "A34"}, inplace=True)
        factor["A34"] = factor["A34"].astype(float)
        return factor

    def calculate_A35(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Quantity of seller-initiated trades in the last 60 s."""
        if data is None:
            data = self.trade
        data = data[data["Trdirec"] == "1"]
        factor = data.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        factor = factor.reindex(self.factors_index_second.index).fillna(0)
        factor = self._rolling_sum_60s(factor)
        factor = self._format_factor(factor)
        factor.rename(columns={"Tvolume": "A35"}, inplace=True)
        factor["A35"] = factor["A35"].astype(float)
        return factor

    def calculate_A36(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Ratio: # buyer-initiated / # seller-initiated trades in last 60 s."""
        if data is None:
            data = self.trade
        buyer = self.calculate_A32(data)
        seller = self.calculate_A33(data)
        buyer.set_index(["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True)
        seller.set_index(["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True)
        factor = buyer["A32"] / seller["A33"]
        factor.name = "A36"
        return factor.reset_index()

    def calculate_A37(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Ratio: qty buyer-initiated / qty seller-initiated trades in last 60 s."""
        if data is None:
            data = self.trade
        buyer = self.calculate_A34(data)
        seller = self.calculate_A35(data)
        buyer.set_index(["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True)
        seller.set_index(["ticker_str", "info_date_ymd", "info_time_hms"], inplace=True)
        factor = buyer["A34"] / seller["A35"]
        factor.name = "A37"
        return factor.reset_index()

    def calculate_A38(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Cumulative ratio: total # buyer-initiated / total # seller-initiated trades."""
        if data is None:
            data = self.trade

        buyer = data[data["Trdirec"] == "5"]
        buyer = buyer.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        buyer = buyer.reindex(self.factors_index_second.index, fill_value=0)
        buyer = buyer.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)

        seller = data[data["Trdirec"] == "1"]
        seller = seller.groupby(["Code_Mkt", "Qdate", "Qtime"])["RecNo"].count()
        seller = seller.reindex(self.factors_index_second.index, fill_value=0)
        seller = seller.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)

        factor = buyer / seller
        factor = self._format_factor(factor)
        factor.rename(columns={"RecNo": "A38"}, inplace=True)
        factor["A38"] = factor["A38"].astype(float)
        return factor

    def calculate_A39(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Cumulative ratio: total qty buyer-initiated / total qty seller-initiated trades."""
        if data is None:
            data = self.trade

        buyer = data[data["Trdirec"] == "5"]
        buyer = buyer.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        buyer = buyer.reindex(self.factors_index_second.index, fill_value=0)
        buyer = buyer.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)

        seller = data[data["Trdirec"] == "1"]
        seller = seller.groupby(["Code_Mkt", "Qdate", "Qtime"])["Tvolume"].sum()
        seller = seller.reindex(self.factors_index_second.index, fill_value=0)
        seller = seller.groupby(by=["Code_Mkt", "Qdate"]).cumsum().shift(1)

        factor = buyer / seller
        factor = self._format_factor(factor)
        factor.rename(columns={"Tvolume": "A39"}, inplace=True)
        factor["A39"] = factor["A39"].astype(float)
        return factor

    # ------------------------------------------------------------------
    # Batch calculation
    # ------------------------------------------------------------------

    def calculate_all(self) -> pd.DataFrame:
        """Compute all 39 factors and return as a single DataFrame."""
        factors = None
        for i in range(1, 40):
            calc_method = getattr(self, f"calculate_A{i}")
            df = calc_method()
            if factors is None:
                factors = df
            else:
                factors[f"A{i}"] = df[f"A{i}"]
        return factors

    def resample_to_minute(self, factors_second: pd.DataFrame) -> pd.DataFrame:
        """
        Resample second-level factors to minute-level.

        Parameters
        ----------
        factors_second : pd.DataFrame
            DataFrame with second-level factor data, indexed by datetime.

        Returns
        -------
        pd.DataFrame
            Minute-level factor data.
        """
        factors_second.index = pd.to_datetime(
            factors_second["info_date_ymd"].astype(str) + " " +
            factors_second["info_time_hms"].astype(str),
            format="%Y%m%d %H%M%S",
        )
        factors_minute = factors_second.resample(
            "1Min", label="right", closed="right"
        ).last()
        # Keep 9:31-11:30 and 13:01-15:00
        idx1 = factors_minute.index.indexer_between_time("9:31", "11:30")
        idx2 = factors_minute.index.indexer_between_time("13:01", "15:00")
        factors_minute = factors_minute.iloc[np.union1d(idx1, idx2)]
        return factors_minute.reset_index(drop=True)


# Alias for backwards compatibility
HighFrequencyFactors = FactorCalculator
