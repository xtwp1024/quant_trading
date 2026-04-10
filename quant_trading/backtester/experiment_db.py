"""
Experiment Database Schema for Quant Trading System

SQLite-based experiment tracking database for storing backtest runs,
trade logs, and optimization trials.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from pathlib import Path


class ExperimentDB:
    """SQLite database for experiment tracking."""

    def __init__(self, db_path: str = "experiments.db"):
        """
        Initialize the experiment database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _cursor(self):
        """Context manager for database cursor."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database tables."""
        with self._cursor() as cursor:
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    description TEXT,
                    status TEXT DEFAULT 'pending'
                )
            """)

            # Backtest runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    run_params TEXT,  -- JSON
                    metrics TEXT,     -- JSON
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

            # Trade logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    reason TEXT,
                    strategy_name TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id)
                )
            """)

            # Optimization trials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_trials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    trial_params TEXT NOT NULL,  -- JSON
                    score REAL,
                    win_rate REAL,
                    profit_loss_ratio REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    total_return REAL,
                    trade_count INTEGER,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_strategy
                ON experiments(strategy_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_created
                ON experiments(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_runs_experiment
                ON backtest_runs(experiment_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_logs_run
                ON trade_logs(backtest_run_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_logs_symbol
                ON trade_logs(symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimization_trials_experiment
                ON optimization_trials(experiment_id)
            """)

    # ==================== Experiment CRUD ====================

    def create_experiment(
        self,
        strategy_name: str,
        market: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        description: Optional[str] = None
    ) -> int:
        """
        Create a new experiment.

        Args:
            strategy_name: Name of the strategy
            market: Market symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            description: Optional description

        Returns:
            Experiment ID
        """
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO experiments (strategy_name, market, start_date, end_date, description)
                VALUES (?, ?, ?, ?, ?)
            """, (strategy_name, market, start_date, end_date, description))
            return cursor.lastrowid

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_experiments(
        self,
        strategy_name: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filters."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        if market:
            query += " AND market = ?"
            params.append(market)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._cursor() as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def update_experiment_status(self, experiment_id: int, status: str) -> None:
        """Update experiment status."""
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE experiments SET status = ? WHERE id = ?",
                (status, experiment_id)
            )

    # ==================== Backtest Run CRUD ====================

    def log_backtest_run(
        self,
        experiment_id: int,
        run_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log a backtest run.

        Args:
            experiment_id: ID of the parent experiment
            run_params: Strategy parameters (JSON)
            metrics: Performance metrics (JSON)

        Returns:
            Backtest run ID
        """
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO backtest_runs (experiment_id, run_params, metrics)
                VALUES (?, ?, ?)
            """, (
                experiment_id,
                json.dumps(run_params) if run_params else None,
                json.dumps(metrics) if metrics else None
            ))
            return cursor.lastrowid

    def get_backtest_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get backtest run by ID."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get('run_params'):
                    result['run_params'] = json.loads(result['run_params'])
                if result.get('metrics'):
                    result['metrics'] = json.loads(result['metrics'])
                return result
            return None

    def get_experiment_runs(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Get all backtest runs for an experiment."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM backtest_runs
                WHERE experiment_id = ?
                ORDER BY created_at DESC
            """, (experiment_id,))
            runs = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('run_params'):
                    result['run_params'] = json.loads(result['run_params'])
                if result.get('metrics'):
                    result['metrics'] = json.loads(result['metrics'])
                runs.append(result)
            return runs

    # ==================== Trade Log CRUD ====================

    def log_trade(
        self,
        backtest_run_id: int,
        symbol: str,
        entry_time: str,
        exit_time: Optional[str] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        quantity: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        reason: Optional[str] = None,
        strategy_name: Optional[str] = None
    ) -> int:
        """
        Log a single trade.

        Args:
            backtest_run_id: ID of the parent backtest run
            symbol: Trading symbol
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            pnl: Profit/Loss in absolute terms
            pnl_pct: Profit/Loss in percentage
            reason: Exit reason
            strategy_name: Name of the strategy

        Returns:
            Trade log ID
        """
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO trade_logs (
                    backtest_run_id, symbol, entry_time, exit_time,
                    entry_price, exit_price, quantity, pnl, pnl_pct, reason, strategy_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                backtest_run_id, symbol, entry_time, exit_time,
                entry_price, exit_price, quantity, pnl, pnl_pct, reason, strategy_name
            ))
            return cursor.lastrowid

    def log_trades(self, backtest_run_id: int, trades: List[Dict[str, Any]]) -> List[int]:
        """
        Log multiple trades at once.

        Args:
            backtest_run_id: ID of the parent backtest run
            trades: List of trade dictionaries

        Returns:
            List of trade log IDs
        """
        trade_ids = []
        with self._cursor() as cursor:
            for trade in trades:
                cursor.execute("""
                    INSERT INTO trade_logs (
                        backtest_run_id, symbol, entry_time, exit_time,
                        entry_price, exit_price, quantity, pnl, pnl_pct, reason, strategy_name
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_run_id,
                    trade.get('symbol'),
                    trade.get('entry_time'),
                    trade.get('exit_time'),
                    trade.get('entry_price'),
                    trade.get('exit_price'),
                    trade.get('quantity'),
                    trade.get('pnl'),
                    trade.get('pnl_pct'),
                    trade.get('reason'),
                    trade.get('strategy_name')
                ))
                trade_ids.append(cursor.lastrowid)
        return trade_ids

    def get_run_trades(self, backtest_run_id: int) -> List[Dict[str, Any]]:
        """Get all trades for a backtest run."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM trade_logs
                WHERE backtest_run_id = ?
                ORDER BY entry_time
            """, (backtest_run_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_trades_by_symbol(self, symbol: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get trades by symbol."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM trade_logs
                WHERE symbol = ?
                ORDER BY entry_time DESC
                LIMIT ?
            """, (symbol, limit))
            return [dict(row) for row in cursor.fetchall()]

    # ==================== Optimization Trial CRUD ====================

    def log_optimization_trial(
        self,
        experiment_id: int,
        trial_params: Dict[str, Any],
        score: Optional[float] = None,
        win_rate: Optional[float] = None,
        profit_loss_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        total_return: Optional[float] = None,
        trade_count: Optional[int] = None
    ) -> int:
        """
        Log an optimization trial.

        Args:
            experiment_id: ID of the parent experiment
            trial_params: Trial parameters (JSON)
            score: Optimization score
            win_rate: Win rate percentage
            profit_loss_ratio: Profit/Loss ratio
            max_drawdown: Maximum drawdown percentage
            sharpe_ratio: Sharpe ratio
            total_return: Total return percentage
            trade_count: Number of trades

        Returns:
            Trial ID
        """
        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO optimization_trials (
                    experiment_id, trial_params, score, win_rate,
                    profit_loss_ratio, max_drawdown, sharpe_ratio, total_return, trade_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                json.dumps(trial_params),
                score, win_rate, profit_loss_ratio, max_drawdown,
                sharpe_ratio, total_return, trade_count
            ))
            return cursor.lastrowid

    def get_best_trial(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get the best trial for an experiment based on score."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM optimization_trials
                WHERE experiment_id = ?
                ORDER BY score DESC
                LIMIT 1
            """, (experiment_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('trial_params'):
                    result['trial_params'] = json.loads(result['trial_params'])
                return result
            return None

    def get_experiment_trials(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Get all trials for an experiment."""
        with self._cursor() as cursor:
            cursor.execute("""
                SELECT * FROM optimization_trials
                WHERE experiment_id = ?
                ORDER BY score DESC
            """, (experiment_id,))
            trials = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('trial_params'):
                    result['trial_params'] = json.loads(result['trial_params'])
                trials.append(result)
            return trials

    # ==================== Query Methods ====================

    def compare_strategies(
        self,
        strategy_names: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Compare metrics across multiple strategies.

        Args:
            strategy_names: List of strategy names to compare
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of strategy comparison results
        """
        results = []
        for name in strategy_names:
            with self._cursor() as cursor:
                # Get latest run for this strategy
                query = """
                    SELECT br.*, e.strategy_name, e.market, e.start_date, e.end_date
                    FROM backtest_runs br
                    JOIN experiments e ON br.experiment_id = e.id
                    WHERE e.strategy_name = ?
                """
                params = [name]

                if start_date:
                    query += " AND e.start_date >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND e.end_date <= ?"
                    params.append(end_date)

                query += " ORDER BY br.created_at DESC LIMIT 1"

                cursor.execute(query, params)
                row = cursor.fetchone()

                if row:
                    result = dict(row)
                    if result.get('metrics'):
                        result['metrics'] = json.loads(result['metrics'])
                    if result.get('run_params'):
                        result['run_params'] = json.loads(result['run_params'])
                    results.append(result)

        return results

    def get_experiment_results(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get comprehensive results for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary with experiment, runs, trades, and trials
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {}

        runs = self.get_experiment_runs(experiment_id)

        # Aggregate trade data
        all_trades = []
        for run in runs:
            trades = self.get_run_trades(run['id'])
            all_trades.extend(trades)

        trials = self.get_experiment_trials(experiment_id)
        best_trial = self.get_best_trial(experiment_id)

        return {
            'experiment': experiment,
            'runs': runs,
            'trades': all_trades,
            'trials': trials,
            'best_trial': best_trial,
            'summary': self._calculate_summary(runs, all_trades)
        }

    def _calculate_summary(
        self,
        runs: List[Dict[str, Any]],
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not trades:
            return {}

        pnls = [t['pnl'] for t in trades if t['pnl'] is not None]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(pnls) if pnls else 0,
            'total_pnl': sum(pnls) if pnls else 0,
            'avg_win': sum(winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(losing_trades) / len(losing_trades) if losing_trades else 0,
            'profit_loss_ratio': (
                abs(sum(winning_trades) / len(winning_trades)) /
                abs(sum(losing_trades) / len(losing_trades))
                if winning_trades and losing_trades else 0
            ),
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
        }

    def query_by_metrics(
        self,
        min_sharpe: Optional[float] = None,
        min_win_rate: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_trades: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query experiments by performance metrics.

        Args:
            min_sharpe: Minimum Sharpe ratio
            min_win_rate: Minimum win rate
            max_drawdown: Maximum drawdown (filter out worse)
            min_trades: Minimum number of trades

        Returns:
            List of matching experiments with metrics
        """
        results = []
        with self._cursor() as cursor:
            # Get all completed experiments with their latest runs
            cursor.execute("""
                SELECT DISTINCT e.*, br.metrics, br.run_params
                FROM experiments e
                LEFT JOIN backtest_runs br ON e.id = br.experiment_id
                WHERE e.status = 'completed'
                ORDER BY br.created_at DESC
            """)

            for row in cursor.fetchall():
                exp = dict(row)
                if exp.get('metrics'):
                    metrics = json.loads(exp['metrics'])

                    # Apply filters
                    if min_sharpe is not None:
                        if metrics.get('sharpe_ratio', 0) < min_sharpe:
                            continue
                    if min_win_rate is not None:
                        if metrics.get('win_rate', 0) < min_win_rate:
                            continue
                    if max_drawdown is not None:
                        if metrics.get('max_drawdown', 999) > max_drawdown:
                            continue

                    exp['metrics'] = metrics
                    if exp.get('run_params'):
                        exp['run_params'] = json.loads(exp['run_params'])
                    results.append(exp)

        return results

    def delete_experiment(self, experiment_id: int) -> None:
        """Delete an experiment and all related data."""
        with self._cursor() as cursor:
            # Delete in correct order due to foreign keys
            cursor.execute(
                "DELETE FROM trade_logs WHERE backtest_run_id IN "
                "(SELECT id FROM backtest_runs WHERE experiment_id = ?)",
                (experiment_id,)
            )
            cursor.execute(
                "DELETE FROM optimization_trials WHERE experiment_id = ?",
                (experiment_id,)
            )
            cursor.execute(
                "DELETE FROM backtest_runs WHERE experiment_id = ?",
                (experiment_id,)
            )
            cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))

    def close(self) -> None:
        """Close database connection (no-op for sqlite3 but good practice)."""
        pass


# Default metrics schema
METRICS_SCHEMA = {
    'total_return': float,
    'sharpe_ratio': float,
    'calmar_ratio': float,
    'win_rate': float,
    'profit_loss_ratio': float,
    'max_drawdown': float,
    'recovery_factor': float,
    'trade_count': int,
    'avg_holding_period': float,  # in hours
    'best_trade': float,
    'worst_trade': float,
}


def validate_metrics(metrics: Dict[str, Any]) -> bool:
    """Validate that metrics contain required fields."""
    required = ['total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown', 'trade_count']
    return all(k in metrics for k in required)
