"""Signal quality metrics — measures the reliability and edge of a signal stream.

Provides:
- SignalQuality: Computes quality metrics for a signal + price series
  (edge, Sharpe,win rate, avg bars held, signal density)
- SignalMetrics: Dataclass holding computed quality metrics
- SignalFilter: Filters signals by quality threshold

Usage
-----
```python
from quant_trading.signal_modules.quality import SignalQuality, SignalMetrics

sq = SignalQuality()
metrics = sq.compute(signals, df)

print(f"Edge: {metrics.edge:.4f}")
print(f"Sharpe: {metrics.sharpe:.2f}")
print(f"Win rate: {metrics.win_rate:.1%}")
```
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal_modules.types import Signal, SignalType


@dataclass
class SignalMetrics:
    """Quality metrics for a signal stream.

    Attributes
    ----------
    edge : float
        Average return per trade (pct). Positive = profitable signal.
    sharpe : float
        Annualized Sharpe ratio of signal-driven returns.
    sortino : float
        Annualized Sortino ratio (downside deviation only).
    max_drawdown : float
        Maximum drawdown of equity curve (fraction).
    win_rate : float
        Fraction of trades that are profitable.
    avg_bars_held : float
        Average number of bars a position is held.
    signal_density : float
        Fraction of bars that produced a signal.
    total_trades : int
        Total number of completed trades.
    profit_factor : float
        Gross profit / gross loss.
    expectancy : float
        Expected return per trade including win rate.
    recovery_factor : float
        Net profit / max drawdown.
    """
    edge: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_bars_held: float = 0.0
    signal_density: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    recovery_factor: float = 0.0

    def to_dict(self) -> dict:
        return {
            "edge": self.edge,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "avg_bars_held": self.avg_bars_held,
            "signal_density": self.signal_density,
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "recovery_factor": self.recovery_factor,
        }

    def score(self) -> float:
        """Composite quality score [0, 1]. Higher = better signal."""
        return float(
            np.clip(
                (self.edge + 0.1) / 0.2 * 0.3  # edge component 30%
                + np.tanh(self.sharpe / 3) * 0.25  # Sharpe 25%
                + self.win_rate * 0.20  # win rate 20%
                + (1 - self.max_drawdown) * 0.15  # low DD 15%
                + min(self.recovery_factor / 5, 1) * 0.10,  # recovery 10%
                0, 1,
            )
        )


class SignalQuality:
    """Computes quality metrics from a signal stream and OHLCV data."""

    def __init__(self, annualization_factor: int = 252 * 24):
        """Initialize.

        Parameters
        ----------
        annualization_factor : int
            Bars per year for Sharpe/Sortino annualization.
            Default 252 (trading days) * 24 (hourly bars) = 6048.
        """
        self.annualization_factor = annualization_factor

    def compute(
        self,
        signals: List[Signal],
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> SignalMetrics:
        """Compute quality metrics.

        Parameters
        ----------
        signals : List[Signal]
            Signal stream (should be filtered to a single symbol).
        df : pd.DataFrame
            OHLCV DataFrame with `timestamp` and `price_col`.
        price_col : str
            Price column to use for returns.

        Returns
        -------
        SignalMetrics
        """
        if df.empty or not signals:
            return SignalMetrics()

        # Build timestamp → bar index map
        ts_to_idx = {}
        for idx, row in df.iterrows():
            ts = int(row["timestamp"]) if "timestamp" in df.columns else idx
            ts_to_idx[ts] = idx

        # Build equity curve from signals
        returns = np.zeros(len(df))
        position = 0  # 1=long, -1=short, 0=flat
        entry_price = 0.0
        entry_bar = 0
        trade_returns: List[float] = []
        bars_held: List[int] = []

        signal_bar_indices = set()

        for sig in sorted(signals, key=lambda s: s.timestamp):
            ts = sig.timestamp
            if ts not in ts_to_idx:
                continue

            bar_i = ts_to_idx[ts]
            signal_bar_indices.add(bar_i)

            if sig.type in (SignalType.BUY, SignalType.EXIT_SHORT):
                if position == -1:
                    # Close short
                    close_px = df.iloc[bar_i][price_col]
                    ret = (entry_price - close_px) / entry_price
                    trade_returns.append(ret)
                    bars_held.append(bar_i - entry_bar)
                    position = 0

                if sig.type == SignalType.BUY and position == 0:
                    position = 1
                    entry_price = df.iloc[bar_i][price_col]
                    entry_bar = bar_i

            elif sig.type in (SignalType.SELL, SignalType.EXIT_LONG, SignalType.CLOSE_ALL):
                if position == 1:
                    # Close long
                    close_px = df.iloc[bar_i][price_col]
                    ret = (close_px - entry_price) / entry_price
                    trade_returns.append(ret)
                    bars_held.append(bar_i - entry_bar)
                    position = 0

                if sig.type == SignalType.SELL and position == 0:
                    position = -1
                    entry_price = df.iloc[bar_i][price_col]
                    entry_bar = bar_i

            elif sig.type == SignalType.CLOSE_ALL and position != 0:
                close_px = df.iloc[bar_i][price_col]
                if position == 1:
                    ret = (close_px - entry_price) / entry_price
                else:
                    ret = (entry_price - close_px) / entry_price
                trade_returns.append(ret)
                bars_held.append(bar_i - entry_bar)
                position = 0

        # Compute metrics
        if not trade_returns:
            return SignalMetrics()

        tr = np.array(trade_returns)
        n = len(tr)

        gross_profit = float(np.sum(tr[tr > 0]))
        gross_loss = float(np.abs(np.sum(tr[tr < 0])))
        profit_factor = gross_profit / (gross_loss + 1e-10)

        win_rate = float(np.mean(tr > 0))
        expectancy = float(np.mean(tr))

        avg_bars_held = float(np.mean(bars_held)) if bars_held else 0.0
        signal_density = len(signal_bar_indices) / len(df) if len(df) > 0 else 0.0

        # Equity curve for Sharpe/DD
        equity = np.cumprod(1 + tr)
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        max_dd = float(np.abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

        # Annualized Sharpe
        mean_ret = np.mean(tr) * self.annualization_factor
        std_ret = np.std(tr) * np.sqrt(self.annualization_factor)
        sharpe = float(mean_ret / (std_ret + 1e-10))

        # Annualized Sortino
        downside = tr[tr < 0]
        downside_std = np.std(downside) * np.sqrt(self.annualization_factor) if len(downside) > 0 else 1e-10
        sortino = float(mean_ret / (downside_std + 1e-10))

        # Edge (avg return per trade)
        edge = float(np.mean(tr))

        # Recovery factor
        total_return = float(equity[-1] - 1) if len(equity) > 0 else 0.0
        recovery_factor = total_return / (max_dd + 1e-10) if max_dd > 0 else 0.0

        return SignalMetrics(
            edge=edge,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_bars_held=avg_bars_held,
            signal_density=signal_density,
            total_trades=n,
            profit_factor=profit_factor,
            expectancy=expectancy,
            recovery_factor=recovery_factor,
        )


@dataclass
class SignalFilter:
    """Filters signals by quality thresholds.

    Usage:
        sf = SignalFilter(min_strength=0.5, min_edge=0.001)
        filtered = sf.filter(signals)
    """
    min_strength: float = 0.0
    min_edge: Optional[float] = None  # Requires price data
    max_signal_density: float = 1.0  # Reject if too many signals per bar
    symbols: Optional[List[str]] = None  # Only allow these symbols

    def filter(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals by configured thresholds."""
        result = []
        for s in signals:
            if s.strength < self.min_strength:
                continue
            if self.symbols and s.symbol not in self.symbols:
                continue
            result.append(s)
        return result

    def filter_by_metrics(
        self,
        signals: List[Signal],
        df: pd.DataFrame,
        min_sharpe: float = 0.0,
        min_win_rate: float = 0.0,
        max_drawdown: float = 1.0,
    ) -> List[Signal]:
        """Keep only signals whose quality metrics pass thresholds."""
        sq = SignalQuality()
        metrics = sq.compute(signals, df)

        if metrics.sharpe < min_sharpe:
            return []
        if metrics.win_rate < min_win_rate:
            return []
        if metrics.max_drawdown > max_drawdown:
            return []
        return signals
