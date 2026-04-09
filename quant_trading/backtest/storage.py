"""
回测框架 - 历史数据存储模块
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class OHLCV:
    """K线数据"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestConfig:
    """回测配置"""
    symbol: str = "ETH/USDT:USDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_balance: float = 10000.0
    commission: float = 0.0004  # 0.04%
    slippage: float = 0.0002   # 0.02%
    leverage: int = 1
    max_position_pct: float = 0.10  # 默认10%仓位，避免全仓操作风险


class DataStorage:
    """历史数据存储"""

    def __init__(self, db_path: str = "data/backtest.db"):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self):
        """确保数据库schema存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_time
            ON ohlcv(symbol, timeframe, timestamp)
        """)
        conn.commit()
        conn.close()

    def save_ohlcv(self, symbol: str, timeframe: str, data: List[OHLCV]):
        """保存K线数据"""
        if not data:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        values = [
            (symbol, timeframe, d.timestamp, d.open, d.high, d.low, d.close, d.volume)
            for d in data
        ]

        cursor.executemany("""
            INSERT OR REPLACE INTO ohlcv
            (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, values)

        conn.commit()
        conn.close()

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: int = 0,
        end_time: int = int(time.time() * 1000),
    ) -> List[OHLCV]:
        """获取K线数据"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
              AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (symbol, timeframe, start_time, end_time))

        rows = cursor.fetchall()
        conn.close()

        return [
            OHLCV(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
            )
            for row in rows
        ]

    def get_ohlcv_dataframe(
        self,
        symbol: str,
        timeframe: str,
        start_time: int = 0,
        end_time: int = int(time.time() * 1000),
    ) -> pd.DataFrame:
        """获取K线DataFrame"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
              AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, conn, params=(symbol, timeframe, start_time, end_time))
        conn.close()

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def has_data(self, symbol: str, timeframe: str) -> bool:
        """检查是否有数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0

    def get_date_range(self, symbol: str, timeframe: str) -> Tuple[Optional[int], Optional[int]]:
        """获取数据时间范围"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe))
        row = cursor.fetchone()
        conn.close()
        return row[0], row[1]


class SignalGenerator:
    """信号生成器基类"""

    def __init__(self, params: Dict = None):
        self.params = params or {}

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """生成交易信号

        Returns:
            pd.Series: 1=LONG, -1=SHORT, 0=NO_SIGNAL
        """
        raise NotImplementedError
