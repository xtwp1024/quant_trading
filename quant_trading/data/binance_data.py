"""Binance历史数据客户端 — Binance Vision.

数据来源: https://data.binance.vision/
支持: K线, 成交, 订单簿快照, 聚合交易

Binance Vision Historical Data Client.
Supports: Klines (OHLCV), Trades, Orderbook Snapshots, AggTrades.
"""

from __future__ import annotations

__all__ = ["BinanceDataClient"]

import os
from datetime import date, datetime
from io import BytesIO
from os import path
from time import time
from typing import Optional
from zipfile import ZipFile, BadZipFile

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BV_BASE_URL = "https://data.binance.vision/data"

ITV_ALIASES = {
    "1m": "1T", "3m": "3T", "5m": "5T",
    "15m": "15T", "30m": "30T",
}

LAST_DATA_POINT_DELAY = 86_400  # 1 day in seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _download_and_unzip(url: str, output_path: str) -> bool:
    """Download a zip file and extract it to output_path."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with ZipFile(BytesIO(resp.content)) as zf:
            zf.extractall(output_path)
        return True
    except (BadZipFile, requests.RequestException):
        return False


def _read_csv_from_dir(dir_path: str) -> pd.DataFrame:
    """Read the single CSV file inside dir_path."""
    file_path = path.join(dir_path, os.listdir(dir_path)[0])
    df = pd.read_csv(file_path, sep=",", usecols=[0, 1, 2, 3, 4, 5])
    df.columns = ["Opened", "Open", "High", "Low", "Close", "Volume"]
    df["Opened"] = pd.to_datetime(df["Opened"], unit="ms")
    return df


def _collect_monthly(url: str, output_folder: str, end_date: date) -> list[pd.DataFrame]:
    """Download monthly zip archives from Binance Vision back to start_date."""
    start_date = date(year=2017, month=1, day=1)
    delta = relativedelta(months=1)
    frames = []
    cur = end_date
    while cur >= start_date:
        month_str = str(cur)[:-3]  # YYYY-MM
        zip_url = url + f"{month_str}.zip"
        out_path = path.join(output_folder, month_str)
        if path.exists(out_path) and os.listdir(out_path)[0].endswith(".csv"):
            frames.append(_read_csv_from_dir(out_path))
        else:
            if _download_and_unzip(zip_url, out_path):
                frames.append(_read_csv_from_dir(out_path))
        cur -= delta
    frames.reverse()
    return frames


def _collect_daily(url: str, output_folder: str, start_date: date, end_date: date) -> list[pd.DataFrame]:
    """Download daily zip archives between start_date and end_date."""
    delta = relativedelta(days=1)
    frames = []
    cur = start_date
    while cur <= end_date:
        day_str = str(cur)  # YYYY-MM-DD
        zip_url = url + f"{day_str}.zip"
        out_path = path.join(output_folder, day_str)
        if path.exists(out_path) and os.listdir(out_path)[0].endswith(".csv"):
            frames.append(_read_csv_from_dir(out_path))
        else:
            if _download_and_unzip(zip_url, out_path):
                frames.append(_read_csv_from_dir(out_path))
        cur += delta
    return frames


def _fix_and_fill_df(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Drop duplicates, forward-fill missing rows, replace zero volume."""
    df.drop_duplicates(inplace=True)
    drop_vals = ["Opened", "Open", "High", "Low", "Close", "Volume"]
    df = df[~df.isin(drop_vals).any(axis=1)]
    freq = interval if "m" not in interval else ITV_ALIASES.get(interval, interval)
    fixed_dates = pd.DataFrame(
        pd.date_range(start=df.iloc[0, 0], end=df.iloc[-1, 0], freq=freq),
        columns=["Opened"],
    )
    df["Opened"] = pd.to_datetime(df["Opened"], format="%Y-%m-%d %H:%M:%S")
    if len(fixed_dates) > len(df):
        df = fixed_dates.merge(df, on="Opened", how="left")
        df.ffill(inplace=True)
    # Replace zero volume with tiny value to allow TA calculations
    df.iloc[:, -1] = df.iloc[:, -1].replace(0.0, 1e-8)
    return df


# ---------------------------------------------------------------------------
# BinanceDataClient
# ---------------------------------------------------------------------------
class BinanceDataClient:
    """Binance历史数据客户端 — Binance Vision.

    数据来源: https://data.binance.vision/
    支持: K线, 成交, 订单簿快照, 聚合交易

    Attributes:
        base_url: Base URL for Binance Vision API.
    """

    def __init__(self, base_url: str = "https://data.binance.vision/data/spot"):
        """Initialize the Binance Vision client.

        Args:
            base_url: Base URL. Defaults to spot market.
                      Options:
                      - https://data.binance.vision/data/spot
                      - https://data.binance.vision/data/futures/um
                      - https://data.binance.vision/data/futures/cm
        """
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Klines (OHLCV)
    # ------------------------------------------------------------------
    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取K线数据 / Fetch OHLCV klines.

        Args:
            symbol: Trading pair, e.g. 'BTCUSDT', 'BTCFDUSD'.
            interval: Candle interval.
                      Options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
            start_time: Start time in milliseconds (UTC).
            end_time: End time in milliseconds (UTC).
            limit: Max number of candles per request (max 1000).

        Returns:
            DataFrame with columns: Opened, Open, High, Low, Close, Volume.

        Example:
            >>> client = BinanceDataClient()
            >>> df = client.get_klines("BTCUSDT", "1h", limit=500)
        """
        symbol = symbol.upper()
        # Determine market type from base_url
        if "/futures/um" in self.base_url:
            market = "um"
            data_type = "klines"
            url = f"{self.base_url}/monthly/{data_type}/{symbol}/{interval}/{symbol}-{interval}-"
        elif "/futures/cm" in self.base_url:
            market = "cm"
            data_type = "klines"
            url = f"{self.base_url}/monthly/{data_type}/{symbol}/{interval}/{symbol}-{interval}-"
        else:  # spot
            market = "spot"
            data_type = "klines"
            url = f"{self.base_url}/monthly/{data_type}/{symbol}/{interval}/{symbol}-{interval}-"

        # Use date-based collection when no explicit start/end time
        if start_time is None and end_time is None:
            end_dt = date.today()
            start_dt = date(year=2017, month=1, day=1)

            monthly_url = url.replace("monthly", "daily")
            daily_frames = _collect_daily(
                monthly_url,
                f"./data/binance_vision/{market}/{data_type}/{symbol}{interval}",
                start_dt,
                end_dt,
            )

            if not daily_frames:
                return pd.DataFrame(columns=["Opened", "Open", "High", "Low", "Close", "Volume"])

            df = pd.concat(daily_frames, ignore_index=True)
            fixed = _fix_and_fill_df(df, interval)
            return fixed

        # Time-based request (REST)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        # Build the REST endpoint URL
        rest_url = self.base_url.replace("/spot", "/api/v3/klines") if "/spot" in self.base_url else f"{self.base_url}/api/v3/klines"
        resp = requests.get(rest_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame(columns=["Opened", "Open", "High", "Low", "Close", "Volume"])

        cols = ["Opened", "Open", "High", "Low", "Close", "Volume"]
        rows = [[
            datetime.fromtimestamp(int(r[0]) / 1000),
            float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
        ] for r in data]
        df = pd.DataFrame(rows, columns=cols)
        return df

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def get_trades(
        self,
        symbol: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取成交数据 / Fetch recent trades.

        Args:
            symbol: Trading pair, e.g. 'BTCUSDT'.
            limit: Number of recent trades (max 1000).

        Returns:
            DataFrame with columns: Id, Price, Qty, Time, IsBuyerMaker.
        """
        symbol = symbol.upper()
        url = f"{self.base_url}/api/v3/trades"
        params = {"symbol": symbol, "limit": limit}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame(columns=["Id", "Price", "Qty", "Time", "IsBuyerMaker"])

        rows = [[r["id"], float(r["price"]), float(r["qty"]), r["time"], r["isBuyerMaker"]] for r in data]
        return pd.DataFrame(rows, columns=["Id", "Price", "Qty", "Time", "IsBuyerMaker"])

    # ------------------------------------------------------------------
    # AggTrades (Aggregated Trades)
    # ------------------------------------------------------------------
    def get_agg_trades(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """获取聚合交易数据 / Fetch aggregated trades.

        Args:
            symbol: Trading pair, e.g. 'BTCUSDT'.
            start_time: Start time in milliseconds.
            end_time: End time in milliseconds.

        Returns:
            DataFrame with columns: AggId, Price, Qty, Time, IsBuyerMaker.
        """
        symbol = symbol.upper()
        url = f"{self.base_url}/api/v3/aggTrades"
        params = {"symbol": symbol}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return pd.DataFrame(columns=["AggId", "Price", "Qty", "Time", "IsBuyerMaker"])

        rows = [[r["a"], float(r["p"]), float(r["q"]), r["T"], r["m"]] for r in data]
        return pd.DataFrame(rows, columns=["AggId", "Price", "Qty", "Time", "IsBuyerMaker"])

    # ------------------------------------------------------------------
    # Orderbook Snapshots
    # ------------------------------------------------------------------
    def get_orderbook_snapshots(
        self,
        symbol: str,
        date: str,
    ) -> pd.DataFrame:
        """获取订单簿快照 (每天一次) / Fetch order book snapshots.

        Args:
            symbol: Trading pair, e.g. 'BTCUSDT'.
            date: Snapshot date in 'YYYY-MM-DD' format, e.g. '2023-01-01'.

        Returns:
            DataFrame with columns: LastUpdateId, Bids, Asks (nested).
            Use pd.json_normalize to flatten if needed.
        """
        symbol = symbol.upper()
        if "/futures/um" in self.base_url:
            url = f"{self.base_url}/daily/orderbook/{symbol}/{symbol}-orderbook-{date}.zip"
        elif "/futures/cm" in self.base_url:
            url = f"{self.base_url}/daily/orderbook/{symbol}/{symbol}-orderbook-{date}.zip"
        else:  # spot
            url = f"{self.base_url}/daily/orderbook/{symbol}/{symbol}-orderbook-{date}.zip"

        out_dir = f"./data/binance_vision_orderbook/{symbol}/{date}"
        os.makedirs(out_dir, exist_ok=True)

        if _download_and_unzip(url, out_dir):
            file_path = path.join(out_dir, f"{symbol}-orderbook-{date}.json")
            with open(file_path, "r") as f:
                import json
                data = json.load(f)
            df = pd.DataFrame([{
                "LastUpdateId": data.get("lastUpdateId"),
                "Bids": data.get("bids", []),
                "Asks": data.get("asks", []),
            }])
            return df
        return pd.DataFrame(columns=["LastUpdateId", "Bids", "Asks"])

    # ------------------------------------------------------------------
    # Download to CSV
    # ------------------------------------------------------------------
    def download_to_csv(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        output_dir: str = "./data",
    ) -> list[str]:
        """下载数据并保存为CSV / Download klines and save as CSV.

        Args:
            symbol: Trading pair, e.g. 'BTCUSDT'.
            interval: Candle interval, e.g. '1m', '1h', '1d'.
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
            output_dir: Directory to save CSV files.

        Returns:
            List of saved file paths.
        """
        symbol = symbol.upper()
        if "/futures/um" in self.base_url:
            market = "um"
        elif "/futures/cm" in self.base_url:
            market = "cm"
        else:
            market = "spot"

        base_url = f"{self.base_url}/daily/klines/{symbol}/{interval}"
        out_dir = path.join(output_dir, f"binance_vision/futures_{market}/klines/{symbol}{interval}")
        os.makedirs(out_dir, exist_ok=True)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

        frames = _collect_daily(base_url, out_dir, start_dt, end_dt)
        if not frames:
            return []

        df = pd.concat(frames, ignore_index=True)
        fixed = _fix_and_fill_df(df, interval)

        out_path = path.join(out_dir, f"{symbol}-{interval}.csv")
        fixed.to_csv(out_path, index=False)
        return [out_path]


# ---------------------------------------------------------------------------
# Convenience function matching the original by_BinanceVision() signature
# ---------------------------------------------------------------------------
def by_BinanceVision(
    ticker: str = "BTCBUSD",
    interval: str = "1m",
    market_type: str = "um",
    data_type: str = "klines",
    start_date: str = "",
    end_date: str = "",
    split: bool = False,
    delay: int = LAST_DATA_POINT_DELAY,
) -> pd.DataFrame | tuple:
    """Download historical data from Binance Vision (convenience function).

    Args:
        ticker: Symbol, e.g. 'BTCUSDT'.
        interval: Candle interval, e.g. '1m', '5m', '1h'.
        market_type: 'spot', 'um' (USD-M futures), or 'cm' (COIN-M futures).
        data_type: 'klines' (OHLCV) or 'trades'.
        start_date: 'YYYY-MM-DD' start filter (optional).
        end_date: 'YYYY-MM-DD' end filter (optional).
        split: If True, return (timestamps, features) tuple.
        delay: Seconds after which to re-download (default 1 day).

    Returns:
        DataFrame or (timestamps, features) tuple if split=True.
    """
    if market_type == "um" or market_type == "cm":
        base_url = f"{BV_BASE_URL}/futures/{market_type}"
    elif market_type == "spot":
        base_url = f"{BV_BASE_URL}/spot"
    else:
        raise ValueError("market_type must be one of {spot, um, cm}")

    client = BinanceDataClient(base_url=base_url)

    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_dt = date.today()

    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        start_dt = date(year=2017, month=1, day=1)

    out_dir = f"./data/binance_vision/futures_{market_type}/{data_type}/{ticker}{interval}"

    monthly_url = f"{BV_BASE_URL}/futures/{market_type}/monthly/{data_type}/{ticker}/{interval}/{ticker}-{interval}-"
    daily_url = f"{BV_BASE_URL}/futures/{market_type}/daily/{data_type}/{ticker}/{interval}/{ticker}-{interval}-"

    monthly_frames = _collect_monthly(monthly_url, out_dir, end_dt)
    daily_frames = _collect_daily(daily_url, out_dir, start_dt, end_dt)

    frames = monthly_frames + daily_frames
    if not frames:
        return pd.DataFrame(columns=["Opened", "Open", "High", "Low", "Close", "Volume"])

    df = pd.concat(frames, ignore_index=True)
    fixed = _fix_and_fill_df(df, interval)

    csv_path = out_dir + ".csv"
    os.makedirs(path.dirname(csv_path), exist_ok=True)
    fixed.to_csv(csv_path, index=False)

    if start_date:
        fixed = fixed[(fixed["Opened"] >= start_date) & (fixed["Opened"] <= end_date)]

    if split:
        return fixed.iloc[:, 0], fixed.iloc[:, 1:]
    return fixed
