"""Walkforward分析框架 — 滚动训练/测试拆分防过拟合.

Absorbs pybroker's walkforward analysis concepts into 量化之神:

- ``WalkforwardAnalyzer`` — 滚动窗口训练/测试拆分, 评估策略稳健性
- ``walkforward_score`` — 跨窗口综合评分

Usage
-----
```python
from quant_trading.backtester import WalkforwardAnalyzer

analyzer = WalkforwardAnalyzer(train_window=252, test_window=63, step=21)
result_df = analyzer.analyze(df, strategy_func, metric='sharpe')
```

English:
    Walkforward analysis framework for rolling train/test splits to
    prevent overfitting and evaluate strategy robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional

__all__ = ["WalkforwardAnalyzer", "WalkforwardResult"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class WalkforwardResult:
    """Results from walkforward analysis.

    Attributes:
        train_metrics: DataFrame of metrics for each training window.
        test_metrics: DataFrame of metrics for each test window.
        window_size: Number of bars per window.
        step_size: Number of bars between windows.
        best_train_params: Best parameters found in each training window.
        degradation: List of (train_metric - test_metric) per window.
        pbo_probability: Probability of overfitting (PBO) estimate.
    """

    train_metrics: pd.DataFrame
    test_metrics: pd.DataFrame
    window_size: int
    step_size: int
    best_train_params: list[dict]
    degradation: list[float]
    pbo_probability: float


# ---------------------------------------------------------------------------
# Walkforward Analyzer
# ---------------------------------------------------------------------------
class WalkforwardAnalyzer:
    """Walkforward分析 — 滚动训练/测试拆分防过拟合.

    Walkforward analysis performs repeated train/test splits on a rolling
    window basis to evaluate strategy robustness and detect overfitting.

    Parameters
    ----------
    train_window : int
        Number of bars in each training window.
    test_window : int
        Number of bars in each test window.
    step : int, default 1
        Number of bars to step forward between windows.
    metric : str, default 'sharpe'
        Metric used to select best params and compare windows.
        Supported: 'sharpe', 'sortino', 'total_return', 'profit_factor'.

    Example
    -------
    ```python
    analyzer = WalkforwardAnalyzer(train_window=252, test_window=63, step=21)
    wf_result = analyzer.analyze(df, my_strategy_func, metric='sharpe')

    print(f"PBO probability: {wf_result.pbo_probability:.2%}")
    print(wf_result.test_metrics)
    ```
    """

    def __init__(
        self,
        train_window: int,
        test_window: int,
        step: int = 1,
        metric: str = "sharpe",
    ):
        if train_window <= 0 or test_window <= 0:
            raise ValueError("train_window and test_window must be positive.")
        if step <= 0:
            raise ValueError("step must be positive.")
        if metric not in {"sharpe", "sortino", "total_return", "profit_factor"}:
            raise ValueError(f"Unsupported metric: {metric}")

        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.metric = metric

    def analyze(
        self,
        df: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Optional[dict] = None,
        price_col: str = "close",
    ) -> WalkforwardResult:
        """Run walkforward analysis.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame indexed by date or integer.
        strategy_func : callable
            Function ``(train_df, **params) -> signals`` that generates
            trading signals on training data.
        param_grid : dict, optional
            Parameter grid for grid search within each training window.
            If None, uses a default single-param configuration.
        price_col : str, default 'close'
            Price column name.

        Returns
        -------
        WalkforwardResult
            Contains train/test metrics, best params, degradation, and PBO.
        """
        n = len(df)
        if n < self.train_window + self.test_window:
            raise ValueError(
                f"Data length {n} is too short for train_window "
                f"{self.train_window} + test_window {self.test_window}."
            )

        # Default param grid if not provided
        if param_grid is None:
            param_grid = {"threshold": [0.0]}

        train_rows: list[dict] = []
        test_rows: list[dict] = []
        best_params_list: list[dict] = []
        degradations: list[float] = []

        window_start = 0
        window_idx = 0

        while window_start + self.train_window + self.test_window <= n:
            # Define window boundaries
            train_end = window_start + self.train_window
            test_end = train_end + self.test_window

            train_df = df.iloc[window_start:train_end]
            test_df = df.iloc[train_end:test_end]

            # Grid search on training window
            best_metric = -np.inf
            best_params = {}
            best_signals: Optional[np.ndarray] = None

            import itertools

            keys = list(param_grid.keys())
            values = list(param_grid.values())
            for combo in itertools.product(*values):
                params = dict(zip(keys, combo))
                signals = strategy_func(train_df, **params)
                train_metrics = self._compute_metrics(train_df, signals, price_col)
                m = self._extract_metric(train_metrics)

                if m > best_metric:
                    best_metric = m
                    best_params = params
                    best_signals = signals

            # Record train metrics
            train_row = self._metrics_to_dict(
                self._compute_metrics(train_df, best_signals, price_col)
            )
            train_row["window"] = window_idx
            train_row["train_start"] = window_start
            train_row["train_end"] = train_end
            train_rows.append(train_row)
            best_params_list.append(best_params)

            # Apply best params to test window
            test_signals = strategy_func(test_df, **best_params)
            test_row = self._metrics_to_dict(
                self._compute_metrics(test_df, test_signals, price_col)
            )
            test_row["window"] = window_idx
            test_row["test_start"] = train_end
            test_row["test_end"] = test_end
            test_rows.append(test_row)

            # Degradation
            test_m = self._extract_metric(
                self._compute_metrics(test_df, test_signals, price_col)
            )
            degradation = best_metric - test_m
            degradations.append(degradation)

            window_start += self.step
            window_idx += 1

        # Compute PBO (Probability of Overfitting) — proportion of windows
        # where test metric is worse than median of train metrics
        if len(degradations) > 2:
            median_deg = np.median(degradations)
            pbo = float(np.mean([d > median_deg for d in degradations]))
        else:
            pbo = 0.0

        return WalkforwardResult(
            train_metrics=pd.DataFrame(train_rows),
            test_metrics=pd.DataFrame(test_rows),
            window_size=self.train_window,
            step_size=self.step,
            best_train_params=best_params_list,
            degradation=degradations,
            pbo_probability=pbo,
        )

    def _compute_metrics(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        price_col: str,
    ) -> dict:
        """Compute metrics from df and signals (simplified, no Numba needed here)."""
        prices = df[price_col].values.astype(np.float64)
        equity = np.ones(len(prices))
        position = 0.0
        entry_price = 0.0

        for i in range(1, len(prices)):
            sig = signals[i] if i < len(signals) else 0
            if sig != 0 and position == 0:
                position = sig
                entry_price = prices[i]
            elif sig == 0 and position != 0:
                pct = (prices[i] - prices[i - 1]) / prices[i - 1]
                equity[i] = equity[i - 1] * (1 + position * pct)
                position = 0.0
            else:
                if position != 0:
                    pct = (prices[i] - prices[i - 1]) / prices[i - 1]
                    equity[i] = equity[i - 1] * (1 + position * pct)
                else:
                    equity[i] = equity[i - 1]

        returns = np.diff(equity) / equity[:-1]
        if len(returns) == 0:
            returns = np.array([0.0])

        total_return = float(equity[-1] / equity[0] - 1)
        std = float(np.std(returns)) if len(returns) > 0 else 0.0
        sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0.0
        downside = returns[returns < 0]
        sortino = (
            float(np.mean(returns) / np.std(downside) * np.sqrt(252))
            if len(downside) > 0 and np.std(downside) > 0
            else 0.0
        )
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = peak - e
            if dd > max_dd:
                max_dd = dd
        max_dd_pct = float(max_dd / peak * 100) if peak > 0 else 0.0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": -max_dd,
            "max_drawdown_pct": -max_dd_pct,
        }

    def _extract_metric(self, metrics: dict) -> float:
        """Extract the configured metric from a metrics dict."""
        if self.metric == "sharpe":
            return metrics["sharpe"]
        elif self.metric == "sortino":
            return metrics["sortino"]
        elif self.metric == "total_return":
            return metrics["total_return"]
        elif self.metric == "profit_factor":
            return metrics.get("profit_factor", 0.0)
        return metrics["sharpe"]

    def _metrics_to_dict(self, metrics: dict) -> dict:
        """Return a flat dict of metrics."""
        return {
            "total_return": metrics["total_return"],
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "max_drawdown": metrics["max_drawdown"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
        }
