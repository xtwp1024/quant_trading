"""Parallel Backtesting Engine - 多策略/多符号并行回测 (Phase 3).

多进程并行回测系统，充分利用多核CPU加速回测过程。

Features:
- Multi-symbol parallel backtesting
- Chunk strategy list across workers
- Progress tracking and result aggregation
- Memory-efficient data sharing

Usage
-----
```python
from quant_trading.backtester.parallel_backtest import ParallelBacktester

# 多符号回测
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
results = ParallelBacktester.run_multi_symbol(
    symbol_data={'BTCUSDT': df_btc, 'ETHUSDT': df_eth},
    strategy=strategy_func,
    params={'fast': 10, 'slow': 20}
)

# 多策略回测
results = ParallelBacktester.run_multi_strategy(
    data=df,
    strategies=[sma_cross, grid_trading, rsi_strategy],
    params_list=[{'fast': 10, 'slow': 20}, {'grid_size': 50}, {'rsi_period': 14}]
)
```
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import pandas as pd
import logging

try:
    from numba import jit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

__all__ = ["ParallelBacktester", "ParallelBacktestResult"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _backtest_worker_single(args):
    """Worker function for single-symbol single-strategy backtest.

    Args:
        args: tuple of (symbol, df_dict, strategy_func, params, config)

    Returns:
        dict with symbol, metrics, params
    """
    symbol, df_dict, strategy_func, params, config = args
    df = df_dict

    # Generate signals
    signals = strategy_func(df, **params)

    # Import here to avoid pickling issues
    from quant_trading.backtester.numba_engine import NumbaEngine, BacktestMetrics

    engine = NumbaEngine(
        commission=config.get('commission', 0.001),
        slippage=config.get('slippage', 0.0005),
        bars_per_year=config.get('bars_per_year', 365)
    )

    try:
        metrics = engine.backtest(
            df, signals,
            initial_capital=config.get('initial_capital', 100000.0),
            price_col=config.get('price_col', 'close')
        )
        return {
            'symbol': symbol,
            'params': params,
            'metrics': metrics,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'params': params,
            'metrics': None,
            'success': False,
            'error': str(e)
        }


def _backtest_worker_multi_strategy(args):
    """Worker function for multi-strategy backtest on single symbol.

    Args:
        args: tuple of (symbol, df, strategy_list, params_list, config)

    Returns:
        list of dicts with strategy_name, metrics, params
    """
    symbol, df, strategy_list, params_list, config = args

    from quant_trading.backtester.numba_engine import NumbaEngine

    engine = NumbaEngine(
        commission=config.get('commission', 0.001),
        slippage=config.get('slippage', 0.0005),
        bars_per_year=config.get('bars_per_year', 365)
    )

    results = []
    for strategy, params in zip(strategy_list, params_list):
        strategy_name = getattr(strategy, '__name__', str(strategy))
        try:
            signals = strategy(df, **params)
            metrics = engine.backtest(
                df, signals,
                initial_capital=config.get('initial_capital', 100000.0),
                price_col=config.get('price_col', 'close')
            )
            results.append({
                'symbol': symbol,
                'strategy': strategy_name,
                'params': params,
                'metrics': metrics,
                'success': True,
                'error': None
            })
        except Exception as e:
            results.append({
                'symbol': symbol,
                'strategy': strategy_name,
                'params': params,
                'metrics': None,
                'success': False,
                'error': str(e)
            })

    return results


def _aggregate_results(results: list, group_by: str = 'symbol') -> pd.DataFrame:
    """Aggregate backtest results into a DataFrame.

    Args:
        results: List of result dicts from workers
        group_by: 'symbol' or 'strategy'

    Returns:
        DataFrame with aggregated metrics
    """
    rows = []
    for r in results:
        if not r['success']:
            continue
        metrics = r['metrics']
        if metrics is None:
            continue

        row = {
            'symbol': r['symbol'],
            'strategy': r.get('strategy', 'default'),
            'params': str(r['params']),
            'total_return': metrics.total_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'max_drawdown_pct': metrics.max_drawdown_pct,
            'trade_count': metrics.trade_count,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'annual_return': metrics.annual_return,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParallelBacktestResult:
    """并行回测结果容器.

    Attributes:
        summary: Summary DataFrame with all metrics
        best_by_return: Best result by total return
        best_by_sharpe: Best result by Sharpe ratio
        total_time: Total execution time in seconds
        n_workers: Number of parallel workers used
        n_success: Number of successful backtests
        n_failed: Number of failed backtests
    """
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_by_return: Optional[dict] = None
    best_by_sharpe: Optional[dict] = None
    total_time: float = 0.0
    n_workers: int = 1
    n_success: int = 0
    n_failed: int = 0

    def __post_init__(self):
        if not self.summary.empty:
            if 'total_return' in self.summary.columns:
                best_idx = self.summary['total_return'].idxmax()
                self.best_by_return = self.summary.loc[best_idx].to_dict()
            if 'sharpe_ratio' in self.summary.columns:
                best_idx = self.summary['sharpe_ratio'].idxmax()
                self.best_by_sharpe = self.summary.loc[best_idx].to_dict()


# ---------------------------------------------------------------------------
# Main parallel backtester class
# ---------------------------------------------------------------------------

class ParallelBacktester:
    """多进程并行回测引擎.

    Usage
    -----
    ```python
    # 多符号并行回测
    results = ParallelBacktester.run_multi_symbol(
        symbol_data={'BTC': df_btc, 'ETH': df_eth},
        strategy=strategy_func,
        params={'fast': 10, 'slow': 20},
        n_workers=4
    )

    # 多策略并行回测
    results = ParallelBacktester.run_multi_strategy(
        data=df,
        strategies=[sma_cross, grid_trading],
        params_list=[{'fast': 10}, {'grid_size': 50}],
        n_workers=4
    )
    ```

    Class methods:
        run_multi_symbol: 多符号并行回测
        run_multi_strategy: 多策略并行回测
        run_param_grid: 参数网格并行搜索
    """

    @staticmethod
    def run_multi_symbol(
        symbol_data: dict[str, pd.DataFrame],
        strategy: Callable,
        params: dict,
        config: Optional[dict] = None,
        n_workers: int = -1,
        show_progress: bool = True,
    ) -> ParallelBacktestResult:
        """多符号并行回测.

        Args:
            symbol_data: Dict of symbol -> DataFrame
            strategy: Strategy function (df, **params) -> signals
            params: Strategy parameters
            config: Backtest config (commission, slippage, etc.)
            n_workers: Number of workers (-1 = all cores)
            show_progress: Show progress bar

        Returns:
            ParallelBacktestResult with aggregated results
        """
        import time
        start_time = time.time()

        if n_workers == -1:
            n_workers = mp.cpu_count()

        config = config or {}
        # Convert DataFrames to dicts for sharing
        df_dicts = {k: v.to_dict('list') for k, v in symbol_data.items()}

        # Prepare work items
        work_items = [
            (symbol, df_dict, strategy, params, config)
            for symbol, df_dict in df_dicts.items()
        ]

        results = []
        failed = 0
        completed = 0
        total = len(work_items)

        if n_workers == 1:
            # Sequential
            for item in work_items:
                result = _backtest_worker_single(item)
                results.append(result)
                if not result['success']:
                    failed += 1
                completed += 1
                if show_progress:
                    logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_backtest_worker_single, item): item
                          for item in work_items}

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if not result['success']:
                        failed += 1
                    completed += 1
                    if show_progress:
                        logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

        elapsed = time.time() - start_time
        summary = _aggregate_results(results, group_by='symbol')

        return ParallelBacktestResult(
            summary=summary,
            total_time=elapsed,
            n_workers=n_workers,
            n_success=len(results) - failed,
            n_failed=failed
        )

    @staticmethod
    def run_multi_strategy(
        data: pd.DataFrame,
        strategies: list,
        params_list: list,
        config: Optional[dict] = None,
        n_workers: int = -1,
        show_progress: bool = True,
    ) -> ParallelBacktestResult:
        """多策略并行回测.

        Args:
            data: DataFrame with OHLCV data
            strategies: List of strategy functions
            params_list: List of params dicts (one per strategy)
            config: Backtest config
            n_workers: Number of workers (-1 = all cores)
            show_progress: Show progress bar

        Returns:
            ParallelBacktestResult with aggregated results
        """
        import time
        start_time = time.time()

        if n_workers == -1:
            n_workers = mp.cpu_count()

        config = config or {}
        df_dict = data.to_dict('list')

        # For single data with multiple strategies, we run in-process
        # to avoid overhead of multiprocessing for small tasks
        from quant_trading.backtester.numba_engine import NumbaEngine

        engine = NumbaEngine(
            commission=config.get('commission', 0.001),
            slippage=config.get('slippage', 0.0005),
            bars_per_year=config.get('bars_per_year', 365)
        )

        results = []
        completed = 0
        total = len(strategies)

        for strategy, params in zip(strategies, params_list):
            strategy_name = getattr(strategy, '__name__', str(strategy))
            try:
                signals = strategy(data, **params)
                metrics = engine.backtest(
                    data, signals,
                    initial_capital=config.get('initial_capital', 100000.0),
                    price_col=config.get('price_col', 'close')
                )
                results.append({
                    'symbol': 'multi',
                    'strategy': strategy_name,
                    'params': params,
                    'metrics': metrics,
                    'success': True,
                    'error': None
                })
            except Exception as e:
                results.append({
                    'symbol': 'multi',
                    'strategy': strategy_name,
                    'params': params,
                    'metrics': None,
                    'success': False,
                    'error': str(e)
                })
            completed += 1
            if show_progress:
                logger.info(f"Strategy {completed}/{total}: {strategy_name}")

        elapsed = time.time() - start_time
        summary = _aggregate_results(results, group_by='strategy')

        return ParallelBacktestResult(
            summary=summary,
            total_time=elapsed,
            n_workers=1,
            n_success=len([r for r in results if r['success']]),
            n_failed=len([r for r in results if not r['success']])
        )

    @staticmethod
    def run_param_grid(
        df: pd.DataFrame,
        strategy: Callable,
        param_grid: dict,
        config: Optional[dict] = None,
        n_workers: int = -1,
        metric: str = 'sharpe_ratio',
        show_progress: bool = True,
    ) -> tuple[ParallelBacktestResult, dict]:
        """参数网格并行搜索.

        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy function (df, **params) -> signals
            param_grid: Dict of param_name -> list of values
            config: Backtest config
            n_workers: Number of workers (-1 = all cores)
            metric: Metric to optimize ('sharpe_ratio' or 'total_return')
            show_progress: Show progress bar

        Returns:
            Tuple of (ParallelBacktestResult, best_params)
        """
        import itertools
        import time

        start_time = time.time()

        if n_workers == -1:
            n_workers = mp.cpu_count()

        config = config or {}

        # Generate all parameter combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))
        params_list = [dict(zip(keys, c)) for c in combos]

        logger.info(f"Testing {len(params_list)} parameter combinations with {n_workers} workers")

        # Prepare work items (use same df for all)
        df_dict = df.to_dict('list')
        work_items = [
            ('grid', df_dict, strategy, params, config)
            for params in params_list
        ]

        results = []
        failed = 0
        completed = 0
        total = len(work_items)

        if n_workers == 1 or len(work_items) <= 2:
            # Sequential for small grids
            for item in work_items:
                result = _backtest_worker_single(item)
                results.append(result)
                if not result['success']:
                    failed += 1
                completed += 1
                if show_progress:
                    logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_backtest_worker_single, item): item
                          for item in work_items}

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if not result['success']:
                        failed += 1
                    completed += 1
                    if show_progress:
                        logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

        elapsed = time.time() - start_time
        summary = _aggregate_results(results, group_by='symbol')

        # Find best params
        best_params = None
        if not summary.empty and metric in summary.columns:
            if metric == 'sharpe_ratio':
                best_idx = summary['sharpe_ratio'].idxmax()
            elif metric == 'total_return':
                best_idx = summary['total_return'].idxmax()
            else:
                best_idx = summary[metric].idxmax()
            best_params = summary.loc[best_idx, 'params']

        return ParallelBacktestResult(
            summary=summary,
            total_time=elapsed,
            n_workers=n_workers,
            n_success=len(results) - failed,
            n_failed=failed
        ), best_params

    @staticmethod
    def run_multi_symbol_multi_strategy(
        symbol_data: dict[str, pd.DataFrame],
        strategies: list,
        shared_params: Optional[dict] = None,
        strategy_params: Optional[list] = None,
        config: Optional[dict] = None,
        n_workers: int = -1,
        show_progress: bool = True,
    ) -> ParallelBacktestResult:
        """多符号 x 多策略网格并行回测.

        Args:
            symbol_data: Dict of symbol -> DataFrame
            strategies: List of strategy functions
            shared_params: Params shared across all strategy runs
            strategy_params: Per-strategy params list
            config: Backtest config
            n_workers: Number of workers (-1 = all cores)
            show_progress: Show progress bar

        Returns:
            ParallelBacktestResult with all combinations
        """
        import time
        import itertools
        start_time = time.time()

        if n_workers == -1:
            n_workers = mp.cpu_count()

        config = config or {}
        shared_params = shared_params or {}
        strategy_params = strategy_params or [{} for _ in strategies]

        # Generate all combinations
        symbols = list(symbol_data.keys())
        work_items = []

        for symbol in symbols:
            df = symbol_data[symbol]
            df_dict = df.to_dict('list')
            for strategy, params in zip(strategies, strategy_params):
                combined_params = {**shared_params, **params}
                work_items.append((
                    symbol, df_dict, strategy, combined_params, config
                ))

        logger.info(f"Running {len(work_items)} backtests "
                   f"({len(symbols)} symbols x {len(strategies)} strategies) "
                   f"with {n_workers} workers")

        results = []
        failed = 0
        completed = 0
        total = len(work_items)

        if n_workers == 1:
            for item in work_items:
                result = _backtest_worker_single(item)
                results.append(result)
                if not result['success']:
                    failed += 1
                completed += 1
                if show_progress:
                    logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_backtest_worker_single, item): item
                          for item in work_items}

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if not result['success']:
                        failed += 1
                    completed += 1
                    if show_progress:
                        logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

        elapsed = time.time() - start_time
        summary = _aggregate_results(results, group_by='symbol')

        return ParallelBacktestResult(
            summary=summary,
            total_time=elapsed,
            n_workers=n_workers,
            n_success=len(results) - failed,
            n_failed=failed
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def parallel_backtest_symbols(
    symbol_data: dict[str, pd.DataFrame],
    strategy: Callable,
    params: dict,
    n_workers: int = -1,
    **kwargs
) -> ParallelBacktestResult:
    """Convenience function for multi-symbol backtest."""
    return ParallelBacktester.run_multi_symbol(
        symbol_data=symbol_data,
        strategy=strategy,
        params=params,
        n_workers=n_workers,
        **kwargs
    )


def parallel_param_search(
    df: pd.DataFrame,
    strategy: Callable,
    param_grid: dict,
    n_workers: int = -1,
    metric: str = 'sharpe_ratio',
    **kwargs
) -> tuple[ParallelBacktestResult, dict]:
    """Convenience function for parameter grid search."""
    return ParallelBacktester.run_param_grid(
        df=df,
        strategy=strategy,
        param_grid=param_grid,
        n_workers=n_workers,
        metric=metric,
        **kwargs
    )
