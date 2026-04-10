"""
Experiment Manager for Quant Trading System

High-level interface for creating experiments, logging backtest runs,
tracking trades, and managing optimization trials.
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from .experiment_db import ExperimentDB, validate_metrics


class ExperimentManager:
    """
    High-level manager for quant trading experiments.

    Provides a clean interface for:
    - Creating and managing experiments
    - Logging backtest runs with metrics
    - Tracking individual trades
    - Running optimization trials
    - Comparing strategy performance
    """

    def __init__(self, db_path: str = "experiments.db"):
        """
        Initialize the experiment manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db = ExperimentDB(db_path)

    def create_experiment(
        self,
        strategy_name: str,
        market: str,
        params: Optional[Dict[str, Any]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        description: Optional[str] = None
    ) -> int:
        """
        Create a new experiment.

        Args:
            strategy_name: Name of the strategy (e.g., 'MA_Cross', 'RSI_Reversion')
            market: Market symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            params: Strategy parameters
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            description: Optional experiment description

        Returns:
            Experiment ID

        Example:
            >>> exp_id = manager.create_experiment(
            ...     strategy_name='RSI_Reversion',
            ...     market='BTCUSDT',
            ...     params={'rsi_period': 14, 'oversold': 30, 'overbought': 70},
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31'
            ... )
        """
        experiment_id = self.db.create_experiment(
            strategy_name=strategy_name,
            market=market,
            start_date=start_date,
            end_date=end_date,
            description=description
        )

        # Log initial backtest run with params if provided
        if params:
            self.log_backtest_run(experiment_id, run_params=params)

        return experiment_id

    def log_backtest_run(
        self,
        experiment_id: int,
        metrics: Optional[Dict[str, Any]] = None,
        run_params: Optional[Dict[str, Any]] = None,
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Log a backtest run for an experiment.

        Args:
            experiment_id: ID of the parent experiment
            metrics: Performance metrics dictionary
            run_params: Parameters used for this run
            trades: Optional list of trade dictionaries

        Returns:
            Backtest run ID

        Metrics should include:
            - total_return: Total return percentage
            - sharpe_ratio: Sharpe ratio
            - calmar_ratio: Calmar ratio
            - win_rate: Win rate percentage (0-100)
            - profit_loss_ratio: Average win / average loss
            - max_drawdown: Maximum drawdown percentage
            - recovery_factor: Recovery factor
            - trade_count: Number of trades
            - avg_holding_period: Average holding time in hours
            - best_trade: Best single trade return
            - worst_trade: Worst single trade return
        """
        # Validate metrics if provided
        if metrics and not validate_metrics(metrics):
            raise ValueError("Metrics missing required fields")

        run_id = self.db.log_backtest_run(
            experiment_id=experiment_id,
            run_params=run_params,
            metrics=metrics
        )

        # Log trades if provided
        if trades:
            self.log_trades(run_id, trades)

        return run_id

    def log_trades(self, run_id: int, trades: List[Dict[str, Any]]) -> List[int]:
        """
        Log trades for a backtest run.

        Args:
            run_id: Backtest run ID
            trades: List of trade dictionaries. Each trade should have:
                - symbol: Trading symbol
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp (optional if still open)
                - entry_price: Entry price
                - exit_price: Exit price (optional)
                - quantity: Position size
                - pnl: Profit/Loss
                - pnl_pct: Profit/Loss percentage
                - reason: Exit reason (optional)
                - strategy_name: Strategy name (optional)

        Returns:
            List of trade log IDs
        """
        # Validate trade data
        required_fields = ['symbol', 'entry_time', 'pnl']
        for i, trade in enumerate(trades):
            missing = [f for f in required_fields if f not in trade]
            if missing:
                raise ValueError(f"Trade {i} missing required fields: {missing}")

        return self.db.log_trades(run_id, trades)

    def log_single_trade(
        self,
        run_id: int,
        symbol: str,
        entry_time: str,
        pnl: float,
        exit_time: Optional[str] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        quantity: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        reason: Optional[str] = None,
        strategy_name: Optional[str] = None
    ) -> int:
        """
        Log a single trade.

        Args:
            run_id: Backtest run ID
            symbol: Trading symbol
            entry_time: Entry timestamp
            pnl: Profit/Loss
            exit_time: Exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            pnl_pct: Profit/Loss percentage
            reason: Exit reason
            strategy_name: Strategy name

        Returns:
            Trade log ID
        """
        return self.db.log_trade(
            backtest_run_id=run_id,
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
            strategy_name=strategy_name
        )

    def log_optimization_trial(
        self,
        experiment_id: int,
        trial_params: Dict[str, Any],
        score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log an optimization trial result.

        Args:
            experiment_id: ID of the parent experiment
            trial_params: Parameters for this trial
            score: Optimization objective score
            metrics: Performance metrics from the trial

        Returns:
            Trial ID

        Note:
            If metrics is provided, it will be used to populate:
            - win_rate, profit_loss_ratio, max_drawdown
            - sharpe_ratio, total_return, trade_count
        """
        if metrics is None:
            metrics = {}

        trial_id = self.db.log_optimization_trial(
            experiment_id=experiment_id,
            trial_params=trial_params,
            score=score or metrics.get('score'),
            win_rate=metrics.get('win_rate'),
            profit_loss_ratio=metrics.get('profit_loss_ratio'),
            max_drawdown=metrics.get('max_drawdown'),
            sharpe_ratio=metrics.get('sharpe_ratio'),
            total_return=metrics.get('total_return'),
            trade_count=metrics.get('trade_count')
        )

        return trial_id

    def get_experiment_results(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get comprehensive results for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary containing:
                - experiment: Experiment details
                - runs: List of backtest runs
                - trades: All trades from all runs
                - trials: Optimization trials
                - best_trial: Best performing trial
                - summary: Aggregated statistics
        """
        return self.db.get_experiment_results(experiment_id)

    def compare_strategies(
        self,
        strategy_names: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Compare performance across multiple strategies.

        Args:
            strategy_names: List of strategy names to compare
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of strategy results with metrics, sorted by latest first

        Example:
            >>> results = manager.compare_strategies(
            ...     ['RSI_Reversion', 'MA_Cross', 'Grid_Trading'],
            ...     start_date='2024-01-01',
            ...     end_date='2024-12-31'
            ... )
        """
        return self.db.compare_strategies(strategy_names, start_date, end_date)

    def list_experiments(
        self,
        strategy_name: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filters.

        Args:
            strategy_name: Filter by strategy name
            market: Filter by market
            limit: Maximum number of results

        Returns:
            List of experiment dictionaries
        """
        return self.db.list_experiments(strategy_name, market, limit)

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get a single experiment by ID."""
        return self.db.get_experiment(experiment_id)

    def update_experiment_status(self, experiment_id: int, status: str) -> None:
        """
        Update experiment status.

        Status values: 'pending', 'running', 'completed', 'failed'
        """
        valid_statuses = ['pending', 'running', 'completed', 'failed']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        self.db.update_experiment_status(experiment_id, status)

    def query_by_metrics(
        self,
        min_sharpe: Optional[float] = None,
        min_win_rate: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_trades: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query experiments by performance thresholds.

        Args:
            min_sharpe: Minimum Sharpe ratio
            min_win_rate: Minimum win rate (percentage)
            max_drawdown: Maximum drawdown (percentage)
            min_trades: Minimum number of trades

        Returns:
            List of experiments meeting all criteria
        """
        return self.db.query_by_metrics(
            min_sharpe=min_sharpe,
            min_win_rate=min_win_rate,
            max_drawdown=max_drawdown,
            min_trades=min_trades
        )

    def delete_experiment(self, experiment_id: int) -> None:
        """
        Delete an experiment and all related data.

        WARNING: This will permanently delete the experiment and all
        associated backtest runs, trades, and trials.
        """
        self.db.delete_experiment(experiment_id)

    def export_results(self, experiment_id: int, filepath: Optional[str] = None) -> str:
        """
        Export experiment results to JSON.

        Args:
            experiment_id: Experiment ID
            filepath: Optional output path

        Returns:
            Path to the exported file
        """
        results = self.get_experiment_results(experiment_id)

        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"experiment_{experiment_id}_{timestamp}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        return filepath

    def close(self) -> None:
        """Close the experiment manager."""
        self.db.close()


# Convenience functions for quick usage

def create_default_metrics(
    trades: List[Dict[str, Any]],
    initial_capital: float = 100000.0
) -> Dict[str, Any]:
    """
    Calculate standard metrics from a list of trades.

    Args:
        trades: List of trade dictionaries with 'pnl' field
        initial_capital: Initial capital for return calculations

    Returns:
        Dictionary of calculated metrics
    """
    if not trades:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'max_drawdown': 0.0,
            'recovery_factor': 0.0,
            'trade_count': 0,
            'avg_holding_period': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
        }

    pnls = [t['pnl'] for t in trades if t.get('pnl') is not None]
    winning_trades = [p for p in pnls if p > 0]
    losing_trades = [p for p in pnls if p < 0]

    total_return = sum(pnls) / initial_capital * 100 if initial_capital else 0
    win_rate = len(winning_trades) / len(pnls) * 100 if pnls else 0

    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = abs(sum(losing_trades) / len(losing_trades)) if losing_trades else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss else 0

    # Calculate max drawdown
    cumulative = 0
    peak = initial_capital
    max_dd = 0
    for pnl in pnls:
        cumulative += pnl
        current = initial_capital + cumulative
        if current > peak:
            peak = current
        drawdown = (peak - current) / peak * 100 if peak else 0
        max_dd = max(max_dd, drawdown)

    recovery_factor = total_return / max_dd if max_dd else 0

    # Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)
    if len(pnls) > 1:
        returns = [p / initial_capital for p in pnls]
        mean_return = sum(returns) / len(returns)
        std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe_ratio = mean_return / std_return * (252 ** 0.5) if std_return else 0
    else:
        sharpe_ratio = 0

    # Calmar ratio
    calmar_ratio = total_return / max_dd if max_dd else 0

    # Average holding period
    holding_periods = []
    for t in trades:
        if t.get('entry_time') and t.get('exit_time'):
            try:
                entry = datetime.fromisoformat(t['entry_time'].replace('Z', '+00:00'))
                exit_t = datetime.fromisoformat(t['exit_time'].replace('Z', '+00:00'))
                holding_periods.append((exit_t - entry).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass

    avg_holding = sum(holding_periods) / len(holding_periods) if holding_periods else 0

    return {
        'total_return': round(total_return, 4),
        'sharpe_ratio': round(sharpe_ratio, 4),
        'calmar_ratio': round(calmar_ratio, 4),
        'win_rate': round(win_rate, 4),
        'profit_loss_ratio': round(profit_loss_ratio, 4),
        'max_drawdown': round(max_dd, 4),
        'recovery_factor': round(recovery_factor, 4),
        'trade_count': len(trades),
        'avg_holding_period': round(avg_holding, 4),
        'best_trade': round(max(pnls), 4) if pnls else 0,
        'worst_trade': round(min(pnls), 4) if pnls else 0,
    }
