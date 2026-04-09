"""Numba加速向量化回测引擎 — 核心计算内核.

Absorbs pybroker's high-speed Numba backtest concepts into 量化之神:

- ``_vectorized_backtest`` — Numba JIT加速核心计算内核 (63+ bt/sec)
- ``NumbaEngine`` — 回测引擎, 支持批量/流式/sandbox模式
- Pure NumPy fallback when Numba is not installed

Usage
-----
```python
from quant_trading.backtester import NumbaEngine

engine = NumbaEngine(commission=0.001, slippage=0.0005)
result = engine.backtest(df, signals, initial_capital=100000.0)
```

English:
    Numba-accelerated vectorized backtesting engine with support for
    batch, streaming, and shadow modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Union

__all__ = ["NumbaEngine", "BacktestMetrics", "TradeRecord"]


# ---------------------------------------------------------------------------
# Numba availability check + graceful fallback
# ---------------------------------------------------------------------------
try:
    from numba import jit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    # Stub decorators so code still type-checks and runs in pure NumPy mode
    def jit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    prange = range


# ---------------------------------------------------------------------------
# Numba-accelerated core backtest kernel
# ---------------------------------------------------------------------------
if _NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True, parallel=True)
    def _vectorized_backtest(
        prices: np.ndarray,
        signals: np.ndarray,
        position_size: float,
        commission: float,
    ) -> np.ndarray:
        """Numba加速向量化回测 — 核心计算内核.

        Args:
            prices: Price array (close prices).
            signals: Signal array (1=long, -1=short, 0=flat).
            position_size: Fixed capital allocated per trade.
            commission: Commission rate per trade (e.g. 0.001 = 0.1%%).

        Returns:
            Equity curve array (starts at 1.0).
        """
        n = len(prices)
        equity = np.ones(n)
        position = 0.0
        entry_price = 0.0

        for i in prange(n):
            sig = signals[i]
            if sig == 1 and position == 0:
                # Open long
                position = position_size / prices[i]
                entry_price = prices[i]
                equity[i] = equity[i - 1] if i > 0 else 1.0
            elif sig == -1 and position == 0:
                # Open short
                position = -position_size / prices[i]
                entry_price = prices[i]
                equity[i] = equity[i - 1] if i > 0 else 1.0
            elif sig == 0 and position != 0:
                # Close position
                pct_change = (prices[i] - prices[i - 1]) / prices[i - 1]
                equity[i] = equity[i - 1] * (1 + position * pct_change - commission)
                position = 0.0
            else:
                # Hold position
                if i > 0:
                    pct_change = (prices[i] - prices[i - 1]) / prices[i - 1]
                    equity[i] = equity[i - 1] * (1 + position * pct_change)

        return equity
else:
    def _vectorized_backtest(
        prices: np.ndarray,
        signals: np.ndarray,
        position_size: float,
        commission: float,
    ) -> np.ndarray:
        """Pure NumPy fallback when Numba is not installed."""
        n = len(prices)
        equity = np.ones(n)
        position = 0.0
        entry_price = 0.0

        for i in range(n):
            sig = signals[i]
            if sig == 1 and position == 0:
                position = position_size / prices[i]
                entry_price = prices[i]
                equity[i] = equity[i - 1] if i > 0 else 1.0
            elif sig == -1 and position == 0:
                position = -position_size / prices[i]
                entry_price = prices[i]
                equity[i] = equity[i - 1] if i > 0 else 1.0
            elif sig == 0 and position != 0:
                pct_change = (prices[i] - prices[i - 1]) / prices[i - 1]
                equity[i] = equity[i - 1] * (1 + position * pct_change - commission)
                position = 0.0
            else:
                if i > 0:
                    pct_change = (prices[i] - prices[i - 1]) / prices[i - 1]
                    equity[i] = equity[i - 1] * (1 + position * pct_change)

        return equity


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TradeRecord:
    """Record of a single trade / position.

    Attributes:
        entry_bar: Index of entry bar.
        exit_bar: Index of exit bar.
        direction: 1 for long, -1 for short.
        entry_price: Price at entry.
        exit_price: Price at exit.
        pnl: Realized PnL in capital units.
        return_pct: Return percentage.
    """

    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float


@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest run.

    Attributes:
        total_return: Total return (e.g. 0.15 = 15%).
        sharpe_ratio: Annualized Sharpe Ratio.
        max_drawdown: Maximum drawdown (e.g. -0.12 = -12%).
        max_drawdown_pct: Maximum drawdown percentage.
        trade_count: Number of trades.
        win_rate: Win rate (0-100).
        profit_factor: Profit factor (gross profit / gross loss).
        sortino_ratio: Annualized Sortino Ratio.
        calmar_ratio: Calmar Ratio.
        annual_return: Annualized return percentage.
        equity_curve: Equity curve array.
    """

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    trade_count: int
    win_rate: float
    profit_factor: float
    sortino_ratio: float
    calmar_ratio: float
    annual_return: float
    equity_curve: np.ndarray


# ---------------------------------------------------------------------------
# Numba-accelerated metric helpers
# ---------------------------------------------------------------------------
if _NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _max_drawdown(equity: np.ndarray) -> float:
        """Compute maximum drawdown from equity curve (cash)."""
        n = len(equity)
        if n == 0:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for i in range(n):
            if equity[i] > peak:
                peak = equity[i]
            dd = peak - equity[i]
            if dd > max_dd:
                max_dd = dd
        return -max_dd

    @jit(nopython=True, cache=True)
    def _max_drawdown_pct(equity: np.ndarray) -> float:
        """Compute maximum drawdown from equity curve (percentage)."""
        n = len(equity)
        if n == 0:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for i in range(n):
            if equity[i] > peak:
                peak = equity[i]
            if peak > 0:
                dd = (peak - equity[i]) / peak
                if dd > max_dd:
                    max_dd = dd
        return -max_dd * 100.0

    @jit(nopython=True, cache=True)
    def _sharpe_ratio(returns: np.ndarray, obs: int = 252) -> float:
        """Compute annualized Sharpe Ratio."""
        n = len(returns)
        if n == 0:
            return 0.0
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(returns) / std * np.sqrt(obs)

    @jit(nopython=True, cache=True)
    def _sortino_ratio(returns: np.ndarray, obs: int = 252) -> float:
        """Compute annualized Sortino Ratio (downside std only)."""
        n = len(returns)
        if n == 0:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0
        std = np.std(downside)
        if std == 0:
            return 0.0
        return np.mean(returns) / std * np.sqrt(obs)

    @jit(nopython=True, cache=True)
    def _profit_factor(pnls: np.ndarray) -> float:
        """Compute profit factor = gross profit / |gross loss|."""
        profits = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_profit = np.sum(profits) if len(profits) > 0 else 0.0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
else:
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = equity[0]
        max_dd = 0.0
        for i in range(len(equity)):
            if equity[i] > peak:
                peak = equity[i]
            dd = peak - equity[i]
            if dd > max_dd:
                max_dd = dd
        return -max_dd

    def _max_drawdown_pct(equity: np.ndarray) -> float:
        peak = equity[0]
        max_dd = 0.0
        for i in range(len(equity)):
            if equity[i] > peak:
                peak = equity[i]
            if peak > 0:
                dd = (peak - equity[i]) / peak
                if dd > max_dd:
                    max_dd = dd
        return -max_dd * 100.0

    def _sharpe_ratio(returns: np.ndarray, obs: int = 252) -> float:
        if len(returns) == 0:
            return 0.0
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(returns) / std * np.sqrt(obs)

    def _sortino_ratio(returns: np.ndarray, obs: int = 252) -> float:
        if len(returns) == 0:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0
        std = np.std(downside)
        if std == 0:
            return 0.0
        return np.mean(returns) / std * np.sqrt(obs)

    def _profit_factor(pnls: np.ndarray) -> float:
        profits = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_profit = np.sum(profits) if len(profits) > 0 else 0.0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 0.0
        return gross_profit / gross_loss


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------
class NumbaEngine:
    """Numba加速回测引擎 / Numba-accelerated backtesting engine.

    Parameters
    ----------
    commission : float, default 0.001
        Commission rate per trade (0.001 = 0.1%).
    slippage : float, default 0.0005
        Slippage rate applied to execution price.
    bars_per_year : int, default 252
        Number of trading bars per year (for annualization).

    Example
    -------
    ```python
    engine = NumbaEngine(commission=0.001, slippage=0.0005)
    result = engine.backtest(df, signals, initial_capital=100000.0)
    print(result.sharpe_ratio)
    ```
    """

    def __init__(
        self,
        commission: float = 0.001,
        slippage: float = 0.0005,
        bars_per_year: int = 252,
    ):
        self.commission = commission
        self.slippage = slippage
        self.bars_per_year = bars_per_year
        self._numba_available = _NUMBA_AVAILABLE

    def backtest(
        self,
        df: pd.DataFrame,
        signals: Union[pd.Series, np.ndarray],
        initial_capital: float = 100000.0,
        price_col: str = "close",
    ) -> BacktestMetrics:
        """Run a single backtest.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at least ``price_col`` column.
        signals : pd.Series or np.ndarray
            Array of signals: 1 = long, -1 = short, 0 = flat.
        initial_capital : float, default 100000.0
            Starting capital.
        price_col : str, default "close"
            Name of the price column to use.

        Returns
        -------
        BacktestMetrics
            Dataclass containing equity curve and performance metrics.
        """
        # Prepare arrays
        prices = df[price_col].values.astype(np.float64)
        if isinstance(signals, pd.Series):
            signals = signals.values.astype(np.int8)
        else:
            signals = np.asarray(signals, dtype=np.int8)

        n = len(prices)
        if n == 0 or len(signals) != n:
            raise ValueError("signals length must match df length.")

        # Apply slippage to prices (assume adverse slippage on entry/exit)
        exec_prices = prices * (1 + self.slippage)

        # Run Numba-accelerated core
        equity_raw = _vectorized_backtest(
            exec_prices, signals, float(initial_capital), self.commission
        )

        # Build returns from equity
        equity_curve = equity_raw * initial_capital
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = np.concatenate([[0.0], returns])

        # Extract trades
        trades = self._extract_trades(prices, signals, initial_capital)

        # Compute metrics
        total_return = float(equity_curve[-1] / equity_curve[0] - 1)
        sharpe = float(_sharpe_ratio(returns, self.bars_per_year))
        sortino = float(_sortino_ratio(returns, self.bars_per_year))
        max_dd = float(_max_drawdown(equity_curve))
        max_dd_pct = float(_max_drawdown_pct(equity_curve))

        pnls = np.array([t.pnl for t in trades], dtype=np.float64)
        pf = float(_profit_factor(pnls)) if len(pnls) > 0 else 0.0
        win_rate = float(len(pnls[pnls > 0]) / len(pnls) * 100) if len(pnls) > 0 else 0.0

        # Calmar
        max_dd_abs = abs(max_dd) if max_dd != 0 else 1e-10
        calmar = float((equity_curve[-1] - equity_curve[0]) / equity_curve[0] *
                       self.bars_per_year / n / max_dd_abs) if n > 0 else 0.0

        # Annual return
        ann_return = float(
            ((equity_curve[-1] / equity_curve[0]) ** (self.bars_per_year / n) - 1)
            if n > 0 else 0.0
        )

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            trade_count=len(trades),
            win_rate=win_rate,
            profit_factor=pf,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            annual_return=ann_return,
            equity_curve=equity_curve,
        )

    def _extract_trades(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        initial_capital: float,
    ) -> list[TradeRecord]:
        """Extract list of trades from price and signal arrays."""
        trades: list[TradeRecord] = []
        position = 0.0
        entry_bar = entry_price = 0
        direction = 0

        for i in range(len(prices)):
            sig = signals[i]
            if sig != 0 and position == 0:
                # Open
                direction = sig
                entry_bar = i
                entry_price = prices[i]
                position = initial_capital / entry_price * direction
            elif sig == 0 and position != 0:
                # Close
                exit_price = prices[i]
                pnl = position * (exit_price - entry_price) / entry_price
                ret_pct = (exit_price - entry_price) / entry_price * 100 * direction
                trades.append(
                    TradeRecord(
                        entry_bar=entry_bar,
                        exit_bar=i,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        return_pct=ret_pct,
                    )
                )
                position = 0.0

        return trades

    def batch_backtest(
        self,
        strategy_func: Callable,
        param_grid: dict,
        df: pd.DataFrame,
        price_col: str = "close",
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Parallel batch backtest over a parameter grid.

        Parameters
        ----------
        strategy_func : callable
            Function that receives (df, **params) and returns signals array.
        param_grid : dict
            Dictionary of parameter names → list of values to iterate.
        df : pd.DataFrame
            OHLCV DataFrame.
        price_col : str, default "close"
            Price column name.
        n_jobs : int, default 1
            Number of parallel jobs. If -1, uses all CPU cores.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per parameter combination and columns
            for each metric from BacktestMetrics.
        """
        import itertools
        from concurrent.futures import ProcessPoolExecutor, as_completed

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        results: list[dict] = []

        if n_jobs == 1:
            for combo in combos:
                params = dict(zip(keys, combo))
                signals = strategy_func(df, **params)
                metrics = self.backtest(df, signals, price_col=price_col)
                row = {"params": params}
                row.update({k: getattr(metrics, k) for k in metrics.__dataclass_fields__})
                results.append(row)
        else:
            if n_jobs == -1:
                n_jobs = None  # Let ProcessPoolExecutor decide

            def _run_one(args):
                combo, keys = args
                params = dict(zip(keys, combo))
                signals = strategy_func(df, **params)
                metrics = self.backtest(df, signals, price_col=price_col)
                return {"params": params, **{
                    k: getattr(metrics, k) for k in metrics.__dataclass_fields__
                }}

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [(combo, keys) for combo in combos]
                for future in as_completed(executor.map(_run_one, futures)):
                    results.append(future)

        out = pd.DataFrame(results)
        return out

    def stream_backtest(
        self,
        data_iter: Iterator[dict],
        signal_func: Callable,
        initial_capital: float = 100000.0,
    ) -> Iterator[dict]:
        """Streaming / shadow mode backtest — does not cache full dataset.

        Parameters
        ----------
        data_iter : Iterator[dict]
            Iterator yielding dicts with at least ``close`` key.
        signal_func : callable
            Function that receives the latest bar dict and returns a signal
            (1, -1, or 0).
        initial_capital : float, default 100000.0

        Yields
        ------
        dict
            Snapshot dict after each bar with keys:
            ``equity``, ``position``, ``signal``, ``bar``.
        """
        equity = initial_capital
        position = 0.0
        entry_price = 0.0
        direction = 0
        bar_count = 0

        for bar in data_iter:
            price = float(bar["close"])
            signal = signal_func(bar)

            # Execute signal
            if signal != 0 and position == 0:
                direction = signal
                entry_price = price * (1 + self.slippage)
                position = (equity * signal) / entry_price
            elif signal == 0 and position != 0:
                exit_price = price * (1 - self.slippage)
                pnl = position * (exit_price - entry_price) / entry_price
                equity += pnl - equity * self.commission
                position = 0.0
            else:
                # Mark-to-market
                if position != 0:
                    mtm_price = price * (1 - self.slippage if position < 0 else 1 + self.slippage)
                    equity += position * (mtm_price - entry_price) / entry_price

            bar_count += 1
            yield {
                "bar": bar_count,
                "equity": equity,
                "position": position,
                "signal": signal,
            }
