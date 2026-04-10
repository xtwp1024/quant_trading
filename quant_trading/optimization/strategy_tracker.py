"""
Strategy Performance Tracker
============================

Track performance of each strategy over time with rolling window metrics (7d, 30d, 90d).
Store in SQLite database for persistent tracking and analysis.
"""
import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy."""
    strategy_name: str
    symbol: str
    timeframe: str
    timestamp: datetime

    # Raw metrics
    roi: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    avg_trade: float

    # Rolling window metrics (computed on fetch)
    roi_7d: float = 0.0
    roi_30d: float = 0.0
    roi_90d: float = 0.0
    sharpe_7d: float = 0.0
    sharpe_30d: float = 0.0
    sharpe_90d: float = 0.0


class StrategyTracker:
    """
    Strategy Performance Tracker with Rolling Window Metrics

    Stores strategy performance in SQLite and computes rolling window metrics
    over 7-day, 30-day, and 90-day periods.
    """

    def __init__(self, db_path: str = "optimization/strategy_tracker.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Strategy performance records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    roi REAL,
                    sharpe REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    profit_factor REAL,
                    avg_trade REAL,
                    equity_curve TEXT,
                    trade_log TEXT,
                    UNIQUE(strategy_name, symbol, timeframe, timestamp)
                )
            """)

            # Strategy metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_metadata (
                    strategy_name TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    first_seen DATETIME,
                    last_updated DATETIME,
                    total_trades_all_time INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    current_status TEXT DEFAULT 'active'
                )
            """)

            # Regime tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    regime TEXT NOT NULL,
                    volatility REAL,
                    trend_strength REAL,
                    market_type TEXT
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_perf_strategy_time
                ON strategy_performance(strategy_name, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_perf_timestamp
                ON strategy_performance(timestamp)
            """)

            conn.commit()

    def record_performance(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        metrics: Dict[str, float],
        equity_curve: Optional[List[float]] = None,
        trade_log: Optional[List[Dict]] = None
    ) -> int:
        """
        Record strategy performance metrics.

        Args:
            strategy_name: Name of the strategy
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '15m', '1h', '1d')
            metrics: Dictionary with keys: roi, sharpe, max_drawdown,
                     win_rate, total_trades, profit_factor, avg_trade
            equity_curve: Optional list of equity values for the period
            trade_log: Optional list of trade records

        Returns:
            Record ID
        """
        timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert performance record
            cursor.execute("""
                INSERT OR REPLACE INTO strategy_performance
                (strategy_name, symbol, timeframe, timestamp, roi, sharpe,
                 max_drawdown, win_rate, total_trades, profit_factor, avg_trade,
                 equity_curve, trade_log)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_name, symbol, timeframe, timestamp,
                metrics.get('roi', 0.0),
                metrics.get('sharpe', 0.0),
                metrics.get('max_drawdown', 0.0),
                metrics.get('win_rate', 0.0),
                int(metrics.get('total_trades', 0)),
                metrics.get('profit_factor', 0.0),
                metrics.get('avg_trade', 0.0),
                json.dumps(equity_curve) if equity_curve else None,
                json.dumps(trade_log) if trade_log else None
            ))

            # Update metadata
            cursor.execute("""
                INSERT OR REPLACE INTO strategy_metadata
                (strategy_name, symbol, timeframe, last_updated, total_trades_all_time)
                VALUES (?, ?, ?, ?,
                    COALESCE((SELECT total_trades_all_time FROM strategy_metadata
                              WHERE strategy_name = ?), 0) + ?)
            """, (
                strategy_name, symbol, timeframe, timestamp,
                strategy_name, metrics.get('total_trades', 0)
            ))

            # Update first_seen if new
            cursor.execute("""
                UPDATE strategy_metadata
                SET first_seen = COALESCE(first_seen, ?)
                WHERE strategy_name = ?
            """, (timestamp, strategy_name))

            conn.commit()

            return cursor.lastrowid

    def record_regime(
        self,
        regime: str,
        volatility: float,
        trend_strength: float,
        market_type: str = "unknown"
    ) -> None:
        """Record market regime classification."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO market_regimes (timestamp, regime, volatility, trend_strength, market_type)
                VALUES (?, ?, ?, ?, ?)
            """, (datetime.now(), regime, volatility, trend_strength, market_type))
            conn.commit()

    def get_rolling_metrics(
        self,
        strategy_name: str,
        symbol: str = "BTC-USDT",
        timeframe: str = "15m"
    ) -> Dict[str, Dict[str, float]]:
        """
        Get rolling window metrics for a strategy.

        Returns dict with keys: '7d', '30d', '90d', each containing:
        - roi, sharpe, win_rate, total_trades, max_drawdown
        """
        windows = {
            '7d': 7,
            '30d': 30,
            '90d': 90
        }

        results = {}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            for window_name, days in windows.items():
                cutoff = datetime.now() - timedelta(days=days)

                cursor.execute("""
                    SELECT
                        AVG(roi) as avg_roi,
                        AVG(sharpe) as avg_sharpe,
                        AVG(win_rate) as avg_win_rate,
                        SUM(total_trades) as total_trades,
                        AVG(max_drawdown) as avg_max_drawdown,
                        AVG(profit_factor) as avg_profit_factor
                    FROM strategy_performance
                    WHERE strategy_name = ?
                    AND symbol = ?
                    AND timeframe = ?
                    AND timestamp >= ?
                """, (strategy_name, symbol, timeframe, cutoff))

                row = cursor.fetchone()

                if row and row['avg_roi'] is not None:
                    results[window_name] = {
                        'roi': row['avg_roi'] or 0.0,
                        'sharpe': row['avg_sharpe'] or 0.0,
                        'win_rate': row['avg_win_rate'] or 0.0,
                        'total_trades': row['total_trades'] or 0,
                        'max_drawdown': row['avg_max_drawdown'] or 0.0,
                        'profit_factor': row['avg_profit_factor'] or 0.0
                    }
                else:
                    results[window_name] = {
                        'roi': 0.0, 'sharpe': 0.0, 'win_rate': 0.0,
                        'total_trades': 0, 'max_drawdown': 0.0, 'profit_factor': 0.0
                    }

        return results

    def get_all_strategies_performance(self) -> pd.DataFrame:
        """Get current performance for all strategies in the pool."""
        with sqlite3.connect(self.db_path) as conn:
            # Get latest metrics for each strategy
            query = """
                WITH LatestMetrics AS (
                    SELECT strategy_name, symbol, timeframe,
                           MAX(timestamp) as latest_ts
                    FROM strategy_performance
                    GROUP BY strategy_name, symbol, timeframe
                )
                SELECT
                    sp.strategy_name,
                    sp.symbol,
                    sp.timeframe,
                    sp.timestamp,
                    sp.roi,
                    sp.sharpe,
                    sp.max_drawdown,
                    sp.win_rate,
                    sp.total_trades,
                    sp.profit_factor,
                    sp.avg_trade,
                    sm.is_active,
                    sm.current_status
                FROM strategy_performance sp
                INNER JOIN LatestMetrics lm
                    ON sp.strategy_name = lm.strategy_name
                    AND sp.symbol = lm.symbol
                    AND sp.timeframe = lm.timeframe
                    AND sp.timestamp = lm.latest_ts
                LEFT JOIN strategy_metadata sm
                    ON sp.strategy_name = sm.strategy_name
                ORDER BY sp.sharpe DESC
            """
            df = pd.read_sql_query(query, conn)

        return df

    def get_strategy_history(
        self,
        strategy_name: str,
        days: int = 30
    ) -> pd.DataFrame:
        """Get historical performance for a strategy."""
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM strategy_performance
                WHERE strategy_name = ?
                AND timestamp >= ?
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn, params=(strategy_name, cutoff))

        return df

    def get_recent_performance(
        self,
        days: int = 7
    ) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated recent performance across all strategies.
        Returns dict mapping strategy_name to metrics dict.
        """
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    strategy_name,
                    AVG(roi) as avg_roi,
                    AVG(sharpe) as avg_sharpe,
                    AVG(win_rate) as avg_win_rate,
                    SUM(total_trades) as total_trades,
                    AVG(max_drawdown) as avg_max_drawdown
                FROM strategy_performance
                WHERE timestamp >= ?
                GROUP BY strategy_name
                ORDER BY avg_sharpe DESC
            """, (cutoff,))

            results = {}
            for row in cursor.fetchall():
                results[row['strategy_name']] = {
                    'roi': row['avg_roi'] or 0.0,
                    'sharpe': row['avg_sharpe'] or 0.0,
                    'win_rate': row['avg_win_rate'] or 0.0,
                    'total_trades': row['total_trades'] or 0,
                    'max_drawdown': row['avg_max_drawdown'] or 0.0
                }

        return results

    def check_strategy_criteria(
        self,
        strategy_name: str,
        min_sharpe: float = 0.5,
        min_win_rate: float = 45.0,
        min_trades: int = 100
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a strategy meets the minimum criteria for pool inclusion.

        Returns:
            (meets_criteria, metrics_dict)
        """
        metrics_30d = self.get_rolling_metrics(strategy_name).get('30d', {})

        meets_sharpe = metrics_30d.get('sharpe', 0) >= min_sharpe
        meets_win_rate = metrics_30d.get('win_rate', 0) >= min_win_rate
        meets_trades = metrics_30d.get('total_trades', 0) >= min_trades

        meets_criteria = meets_sharpe and meets_win_rate and meets_trades

        details = {
            'sharpe': metrics_30d.get('sharpe', 0),
            'sharpe_required': min_sharpe,
            'sharpe_met': meets_sharpe,
            'win_rate': metrics_30d.get('win_rate', 0),
            'win_rate_required': min_win_rate,
            'win_rate_met': meets_win_rate,
            'total_trades': metrics_30d.get('total_trades', 0),
            'min_trades_required': min_trades,
            'trades_met': meets_trades
        }

        return meets_criteria, details

    def get_pool_status(self) -> Dict[str, Any]:
        """Get overall status of the strategy pool."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Count strategies by status
            cursor.execute("""
                SELECT current_status, COUNT(*) as count
                FROM strategy_metadata
                GROUP BY current_status
            """)
            status_counts = {row['current_status']: row['count']
                           for row in cursor.fetchall()}

            # Get latest regime
            cursor.execute("""
                SELECT regime, timestamp
                FROM market_regimes
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            latest_regime = cursor.fetchone()

            # Get performance summary
            cursor.execute("""
                SELECT
                    COUNT(DISTINCT strategy_name) as total_strategies,
                    AVG(sharpe) as avg_sharpe,
                    MAX(sharpe) as best_sharpe,
                    SUM(total_trades) as total_trades
                FROM (
                    SELECT strategy_name, sharpe, total_trades
                    FROM strategy_performance
                    WHERE timestamp >= ?
                )
            """, (datetime.now() - timedelta(days=7),))

            perf_row = cursor.fetchone()

        return {
            'status_counts': status_counts,
            'total_strategies': sum(status_counts.values()),
            'active_strategies': status_counts.get('active', 0),
            'under_review_strategies': status_counts.get('under_review', 0),
            'removed_strategies': status_counts.get('removed', 0),
            'latest_regime': {
                'regime': latest_regime['regime'] if latest_regime else 'unknown',
                'timestamp': latest_regime['timestamp'] if latest_regime else None
            },
            'performance_summary': {
                'avg_sharpe_7d': perf_row['avg_sharpe'] or 0,
                'best_sharpe_7d': perf_row['best_sharpe'] or 0,
                'total_trades_7d': perf_row['total_trades'] or 0
            }
        }

    def update_strategy_status(
        self,
        strategy_name: str,
        status: str
    ) -> None:
        """Update strategy status (active, under_review, removed)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE strategy_metadata
                SET current_status = ?, last_updated = ?
                WHERE strategy_name = ?
            """, (status, datetime.now(), strategy_name))
            conn.commit()

    def get_underperforming_strategies(
        self,
        roi_threshold: float = -5.0,
        sharpe_threshold: float = 0.3,
        drawdown_threshold: float = 35.0
    ) -> List[Dict[str, Any]]:
        """Get strategies that are underperforming and should be reviewed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cutoff = datetime.now() - timedelta(days=7)

            cursor.execute("""
                SELECT
                    sp.strategy_name,
                    AVG(sp.roi) as avg_roi,
                    AVG(sp.sharpe) as avg_sharpe,
                    AVG(sp.max_drawdown) as avg_drawdown,
                    SUM(sp.total_trades) as total_trades
                FROM strategy_performance sp
                INNER JOIN strategy_metadata sm ON sp.strategy_name = sm.strategy_name
                WHERE sp.timestamp >= ?
                AND sm.current_status = 'active'
                GROUP BY sp.strategy_name
                HAVING avg_roi < ? OR avg_sharpe < ? OR avg_drawdown > ?
            """, (cutoff, roi_threshold, sharpe_threshold, drawdown_threshold))

            return [dict(row) for row in cursor.fetchall()]

    def add_sample_data(self) -> None:
        """Add sample data for testing the tracker."""
        sample_strategies = [
            ("TrendFollowing_EMAs", "BTC-USDT", "15m"),
            ("MeanReversion_RSI", "BTC-USDT", "15m"),
            ("GridTrading_BTC", "BTC-USDT", "1h"),
            ("Breakout_Squeeze", "ETH-USDT", "15m"),
            ("MarketMaker_PRO", "BTC-USDT", "1m"),
        ]

        import random
        random.seed(42)

        for strat_name, symbol, timeframe in sample_strategies:
            # Generate 90 days of sample data
            for days_ago in range(90, 0, -1):
                timestamp = datetime.now() - timedelta(days=days_ago)

                # Random but somewhat realistic metrics
                base_roi = random.uniform(-2, 5)
                roi = base_roi + random.uniform(-1, 1)
                sharpe = random.uniform(0.3, 2.5) if roi > 0 else random.uniform(-0.5, 0.5)
                win_rate = random.uniform(40, 65)
                total_trades = random.randint(5, 50)
                max_drawdown = random.uniform(5, 30)
                profit_factor = random.uniform(0.8, 2.5)
                avg_trade = random.uniform(-50, 200)

                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO strategy_performance
                        (strategy_name, symbol, timeframe, timestamp, roi, sharpe,
                         max_drawdown, win_rate, total_trades, profit_factor, avg_trade)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        strat_name, symbol, timeframe, timestamp,
                        roi, sharpe, max_drawdown, win_rate,
                        total_trades, profit_factor, avg_trade
                    ))

                    # Update metadata
                    cursor.execute("""
                        INSERT OR REPLACE INTO strategy_metadata
                        (strategy_name, symbol, timeframe, first_seen, last_updated,
                         total_trades_all_time, is_active, current_status)
                        VALUES (?, ?, ?, ?, ?, ?, 1, 'active')
                    """, (
                        strat_name, symbol, timeframe,
                        datetime.now() - timedelta(days=90),
                        datetime.now(),
                        total_trades * 90
                    ))

                    conn.commit()


def main():
    """Test the strategy tracker."""
    print("\n" + "="*80)
    print("Strategy Performance Tracker Test")
    print("="*80)

    tracker = StrategyTracker()

    # Add sample data
    print("\n[1] Adding sample data...")
    tracker.add_sample_data()
    print("    Sample data added.")

    # Get pool status
    print("\n[2] Pool Status:")
    status = tracker.get_pool_status()
    for key, value in status.items():
        print(f"    {key}: {value}")

    # Get recent performance
    print("\n[3] Recent Performance (7d):")
    recent = tracker.get_recent_performance(days=7)
    for strat, metrics in recent.items():
        print(f"    {strat}: Sharpe={metrics['sharpe']:.2f}, ROI={metrics['roi']:.2f}%")

    # Get rolling metrics for a specific strategy
    print("\n[4] Rolling Metrics for TrendFollowing_EMAs:")
    metrics = tracker.get_rolling_metrics("TrendFollowing_EMAs")
    for window, data in metrics.items():
        print(f"    {window}: Sharpe={data['sharpe']:.2f}, ROI={data['roi']:.2f}%")

    # Check criteria
    print("\n[5] Strategy Criteria Check:")
    for strat_name in recent.keys():
        meets, details = tracker.check_strategy_criteria(strat_name)
        status_str = "PASS" if meets else "FAIL"
        print(f"    {strat_name}: {status_str}")
        if not meets:
            print(f"      Sharpe: {details['sharpe']:.2f} (req: {details['sharpe_required']})")
            print(f"      WinRate: {details['win_rate']:.1f}% (req: {details['win_rate_required']}%)")
            print(f"      Trades: {details['total_trades']} (req: {details['min_trades_required']})")

    # Get underperforming strategies
    print("\n[6] Underperforming Strategies:")
    underperforming = tracker.get_underperforming_strategies()
    if underperforming:
        for strat in underperforming:
            print(f"    {strat['strategy_name']}: ROI={strat['avg_roi']:.2f}%, "
                  f"Sharpe={strat['avg_sharpe']:.2f}")
    else:
        print("    None found.")

    # Record a sample regime
    print("\n[7] Recording sample regime...")
    tracker.record_regime("trending_high_vol", 0.75, 0.82, "bullish")
    print("    Regime recorded.")

    print("\n" + "="*80)
    print("Tracker test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
