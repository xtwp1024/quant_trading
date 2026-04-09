"""
Performance Metrics — adapted from algorithmic-trading-utilities/portfolio_ops.py.

Unique features NOT in finclaw/matilda:
- Rolling alpha/beta (sliding window CAPM)
- Return distribution statistics (skewness, kurtosis)
- VaR/CVaR at multiple confidence levels (5% and 1%)
- Win rate, profit factor, risk-reward ratio

Designed to work with pandas Series of returns.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


class PerformanceMetrics:
    """
    Portfolio performance metrics calculator.

    Accepts daily returns (pd.Series) directly, plus optional benchmark returns.
    All ratio calculations are annualized (252 trading days).

    Example:
        >>> pm = PerformanceMetrics(returns=portfolio_returns, benchmark_returns=benchmark_returns)
        >>> metrics = pm.calculate_all()
        >>> rolling_ab = pm.rolling_alpha_beta(window=252)
    """

    def __init__(
        self,
        returns: pd.Series = None,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = 0.02 / 252,
    ):
        """
        Initialize performance metrics calculator.

        Args:
            returns (pd.Series): Daily portfolio returns (datetime-indexed).
            benchmark_returns (pd.Series, optional): Daily benchmark returns.
                If provided, enables alpha/beta calculations.
            risk_free_rate (float): Daily risk-free rate. Defaults to 0.02/252.
        """
        if returns is None or len(returns) < 2:
            raise ValueError("returns must be a non-empty pd.Series with at least 2 observations")

        self.returns = returns.dropna()
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns.index = pd.to_datetime(self.returns.index)

        self.benchmark_returns = None
        if benchmark_returns is not None:
            self.benchmark_returns = pd.Series(benchmark_returns).dropna()
            # Align to same dates as returns
            common_idx = self.returns.index.intersection(self.benchmark_returns.index)
            if len(common_idx) < 2:
                self.benchmark_returns = None
            else:
                self.benchmark_returns = self.benchmark_returns.loc[common_idx]
                self.returns = self.returns.loc[common_idx]

        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Core statistics
    # ------------------------------------------------------------------

    def average_return(self) -> float:
        """Mean daily return."""
        return float(self.returns.mean())

    def total_return(self) -> float:
        """Cumulative return over the period: (final / initial) - 1."""
        return float(self.returns.add(1).prod() - 1)

    def std_dev(self) -> float:
        """Standard deviation of daily returns."""
        return float(self.returns.std())

    def downside_std(self) -> float:
        """Standard deviation of negative returns only."""
        downside = self.returns[self.returns < 0]
        return float(downside.std()) if not downside.empty else 0.0

    # ------------------------------------------------------------------
    # Risk-adjusted ratios
    # ------------------------------------------------------------------

    def sharpe_ratio(self) -> float:
        """Daily Sharpe ratio: (mean - rfr) / std_dev."""
        sd = self.std_dev()
        return (self.average_return() - self.risk_free_rate) / sd if sd > 0 else np.nan

    def annualised_sharpe(self) -> float:
        """Annualised Sharpe ratio (252 trading days)."""
        return self.sharpe_ratio() * np.sqrt(252)

    def sortino_ratio(self) -> float:
        """Sortino ratio using downside deviation."""
        dr = self.downside_std()
        return (self.average_return() - self.risk_free_rate) / dr if dr > 0 else np.nan

    def annualised_sortino(self) -> float:
        """Annualised Sortino ratio."""
        return self.sortino_ratio() * np.sqrt(252)

    def calmar_ratio(self) -> float:
        """Calmar ratio: annualized mean return / max drawdown."""
        dd = self.max_drawdown()
        return self.average_return() * 252 / dd if dd > 0 else np.nan

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------

    def _equity_from_returns(self) -> pd.Series:
        """Rebuild equity curve from returns series (assuming starting at 1.0)."""
        return self.returns.add(1).cumprod()

    def drawdown_series(self) -> pd.Series:
        """Drawdown at each point: (peak - current) / peak."""
        equity = self._equity_from_returns()
        cum_max = equity.cummax()
        return (cum_max - equity) / cum_max

    def max_drawdown(self) -> float:
        """Maximum drawdown (peak-to-trough) as a positive fraction."""
        return float(self.drawdown_series().max())

    def average_drawdown(self) -> float:
        """Average drawdown over the period."""
        return float(self.drawdown_series().mean())

    def drawdown_duration(self) -> int:
        """
        Maximum drawdown duration in days (consecutive days below peak).
        Returns 0 if equity is always at or above peak.
        """
        dd = self.drawdown_series()
        if dd.empty:
            return 0
        is_dd = dd > 0
        boundaries = np.diff(np.concatenate(([0], is_dd.astype(int), [0])))
        run_starts = np.where(boundaries == 1)[0]
        run_ends = np.where(boundaries == -1)[0]
        if len(run_starts) == 0:
            return 0
        return int((run_ends - run_starts).max())

    # ------------------------------------------------------------------
    # Distribution statistics (VaR / CVaR)
    # ------------------------------------------------------------------

    def var(self, alpha: float = 0.05) -> float:
        """
        Value at Risk: the quantile of returns at the given significance level.

        Args:
            alpha (float): Significance level (e.g. 0.05 for 5%). Defaults to 0.05.

        Returns:
            float: VaR as a negative fraction (loss magnitude).
        """
        return float(self.returns.quantile(alpha))

    def cvar(self, alpha: float = 0.05) -> float:
        """
        Conditional VaR (Expected Shortfall): mean return in the worst alpha tail.

        Args:
            alpha (float): Significance level. Defaults to 0.05.

        Returns:
            float: CVaR as a negative fraction.
        """
        threshold = self.returns.quantile(alpha)
        tail = self.returns[self.returns <= threshold]
        return float(tail.mean()) if not tail.empty else np.nan

    def return_distribution_stats(self, alpha: float = 0.05) -> dict:
        """
        Return distribution statistics including VaR, CVaR, skewness, kurtosis.

        Args:
            alpha (float): Significance level for VaR/CVaR. Defaults to 0.05.

        Returns:
            dict: Keys: skewness, kurtosis, VaR, CVaR.
        """
        r = self.returns
        var_val = r.quantile(alpha)
        cvar_val = r[r <= var_val].mean()
        return {
            "skewness": float(skew(r)),
            "kurtosis": float(kurtosis(r)),
            "VaR": float(var_val),
            "CVaR": float(cvar_val) if not np.isnan(cvar_val) else np.nan,
        }

    # ------------------------------------------------------------------
    # Alpha / Beta (CAPM)
    # ------------------------------------------------------------------

    def alpha_beta(self) -> dict:
        """
        CAPM alpha and beta vs benchmark.

        Returns:
            dict: Keys 'alpha' (intercept) and 'beta' (slope).
        """
        if self.benchmark_returns is None or len(self.benchmark_returns) < 3:
            return {"alpha": np.nan, "beta": np.nan}

        # Align lengths
        bm = self.benchmark_returns.values
        rf = self.risk_free_rate
        y = self.returns.values - rf
        X = bm - rf

        # OLS: intercept (alpha) and slope (beta)
        X_arr = np.column_stack([np.ones_like(X), X])
        try:
            coeffs = np.linalg.lstsq(X_arr, y, rcond=None)[0]
        except Exception:
            return {"alpha": np.nan, "beta": np.nan}

        alpha, beta = coeffs[0], coeffs[1]
        return {"alpha": float(alpha), "beta": float(beta)}

    def rolling_alpha_beta(self, window: int = 252) -> pd.DataFrame:
        """
        Rolling CAPM alpha and beta over a specified window.

        Args:
            window (int): Rolling window size in days. Defaults to 252.

        Returns:
            pd.DataFrame: Columns 'alpha' and 'beta', datetime-indexed.
        """
        if self.benchmark_returns is None or len(self.returns) < window:
            return pd.DataFrame(columns=["alpha", "beta"])

        alphas, betas = [], []
        ret_vals = self.returns.values
        bm_vals = self.benchmark_returns.values
        n = len(ret_vals)
        rf = self.risk_free_rate

        for i in range(n - window + 1):
            y = ret_vals[i : i + window] - rf
            X = bm_vals[i : i + window] - rf
            X_arr = np.column_stack([np.ones_like(X), X])
            try:
                coeffs = np.linalg.lstsq(X_arr, y, rcond=None)[0]
                alphas.append(coeffs[0])
                betas.append(coeffs[1])
            except Exception:
                alphas.append(np.nan)
                betas.append(np.nan)

        index = self.returns.index[window - 1 :]
        return pd.DataFrame({"alpha": alphas, "beta": betas}, index=index)

    # ------------------------------------------------------------------
    # Rolling Sharpe
    # ------------------------------------------------------------------

    def rolling_sharpe(self, window: int = 252) -> pd.Series:
        """
        Rolling Sharpe ratio with a specified window.

        Args:
            window (int): Rolling window size in days. Defaults to 252.

        Returns:
            pd.Series: Rolling Sharpe (annualised).
        """
        roll_mean = self.returns.rolling(window).mean()
        roll_std = self.returns.rolling(window).std()
        sharpe = (roll_mean - self.risk_free_rate) / roll_std
        return sharpe * np.sqrt(252)

    # ------------------------------------------------------------------
    # Trading statistics
    # ------------------------------------------------------------------

    def win_rate(self) -> float:
        """Fraction of positive return days."""
        return float((self.returns > 0).sum() / len(self.returns))

    def profit_factor(self) -> float:
        """Ratio of gross profits to gross losses (absolute values)."""
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return float(gains / losses) if losses > 0 else np.nan

    def risk_reward_ratio(self) -> float:
        """Average gain / average loss (positive days mean / negative days mean)."""
        avg_gain = self.returns[self.returns > 0].mean()
        avg_loss = abs(self.returns[self.returns < 0].mean())
        return float(avg_gain / avg_loss) if avg_loss > 0 else np.nan

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def calculate_all(self) -> dict:
        """
        Aggregate all performance metrics into a dictionary.

        Returns:
            dict: Full set of performance metrics including
                  VaR_5%, VaR_1%, CVaR_5%, CVaR_1%, win rate, profit factor,
                  risk-reward ratio, and rolling alpha/beta stats.
        """
        dist_5 = self.return_distribution_stats(alpha=0.05)
        dist_1 = self.return_distribution_stats(alpha=0.01)
        ab = self.alpha_beta()

        # Rolling alpha/beta summary
        rab = self.rolling_alpha_beta(window=min(252, len(self.returns) // 2))
        rolling_ab_summary = {}
        if not rab.empty:
            rolling_ab_summary = {
                "rolling_alpha_mean": float(rab["alpha"].mean()),
                "rolling_alpha_std": float(rab["alpha"].std()),
                "rolling_beta_mean": float(rab["beta"].mean()),
                "rolling_beta_std": float(rab["beta"].std()),
            }

        metrics = {
            # Core
            "average_return": self.average_return(),
            "total_return": self.total_return(),
            "std_dev": self.std_dev(),
            # Ratios
            "sharpe_ratio": self.sharpe_ratio(),
            "annualised_sharpe": self.annualised_sharpe(),
            "sortino_ratio": self.sortino_ratio(),
            "annualised_sortino": self.annualised_sortino(),
            "calmar_ratio": self.calmar_ratio(),
            # Drawdown
            "max_drawdown": self.max_drawdown(),
            "average_drawdown": self.average_drawdown(),
            "drawdown_duration": self.drawdown_duration(),
            # Distribution
            "skewness": dist_5["skewness"],
            "kurtosis": dist_5["kurtosis"],
            "VaR_5%": dist_5["VaR"],
            "CVaR_5%": dist_5["CVaR"],
            "VaR_1%": dist_1["VaR"],
            "CVaR_1%": dist_1["CVaR"],
            # Trading stats
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "risk_reward_ratio": self.risk_reward_ratio(),
            # Alpha/Beta
            "alpha": ab["alpha"],
            "beta": ab["beta"],
        }
        metrics.update(rolling_ab_summary)
        return metrics

    def calculate_benchmark_metrics(self) -> dict:
        """
        Compute the same metrics for the benchmark (if available).

        Returns:
            dict: Full metrics dict for the benchmark, with alpha=0, beta=1.
        """
        if self.benchmark_returns is None:
            all_keys = [
                "average_return", "total_return", "std_dev",
                "sharpe_ratio", "annualised_sharpe", "sortino_ratio",
                "annualised_sortino", "calmar_ratio",
                "max_drawdown", "average_drawdown", "drawdown_duration",
                "skewness", "kurtosis",
                "VaR_5%", "CVaR_5%", "VaR_1%", "CVaR_1%",
                "win_rate", "profit_factor", "risk_reward_ratio",
                "alpha", "beta",
            ]
            return {k: np.nan for k in all_keys}

        bm_pm = PerformanceMetrics(
            returns=self.benchmark_returns,
            benchmark_returns=None,
            risk_free_rate=self.risk_free_rate,
        )
        bm_metrics = bm_pm.calculate_all()
        # Benchmark is the market — alpha=0, beta=1 by definition
        bm_metrics["alpha"] = 0.0
        bm_metrics["beta"] = 1.0
        return bm_metrics

    def report(self):
        """
        Print a formatted performance comparison table between strategy and benchmark.
        """
        if self.benchmark_returns is None:
            print("Benchmark returns not provided. Cannot generate comparison report.")
            return

        strategy_metrics = self.calculate_all()

        # Temporarily swap for benchmark calculation
        orig_ret, orig_bm = self.returns, self.benchmark_returns
        self.returns, self.benchmark_returns = self.benchmark_returns, None
        benchmark_metrics = self.calculate_benchmark_metrics()
        self.returns, self.benchmark_returns = orig_ret, orig_bm

        title = "Strategy vs Benchmark Performance Comparison"
        span = 58
        print("=" * span)
        print(f"{title:^{span}}")
        print("=" * span)

        label_w, value_w, bench_w = 26, 14, 14
        print(f"{'':>{label_w}} {'Strategy':>{value_w}} {'Benchmark':>{bench_w}}")
        print("-" * span)

        metrics_list = [
            ("Sharpe Ratio:", "sharpe_ratio", "ratio"),
            ("Annualised Sharpe:", "annualised_sharpe", "ratio"),
            ("Sortino Ratio:", "sortino_ratio", "ratio"),
            ("Annualised Sortino:", "annualised_sortino", "ratio"),
            ("Calmar Ratio:", "calmar_ratio", "ratio"),
            ("Cumulative Return:", "total_return", "pct"),
            ("Avg Daily Return:", "average_return", "pct"),
            ("Std Dev:", "std_dev", "pct"),
            ("Max Drawdown:", "max_drawdown", "pct"),
            ("Avg Drawdown:", "average_drawdown", "pct"),
            ("Drawdown Duration (days):", "drawdown_duration", "int"),
            ("Skewness:", "skewness", "float"),
            ("Kurtosis:", "kurtosis", "float"),
            ("VaR 5%:", "VaR_5%", "pct"),
            ("CVaR 5%:", "CVaR_5%", "pct"),
            ("VaR 1%:", "VaR_1%", "pct"),
            ("CVaR 1%:", "CVaR_1%", "pct"),
            ("Win Rate:", "win_rate", "pct"),
            ("Profit Factor:", "profit_factor", "float"),
            ("Risk-Reward Ratio:", "risk_reward_ratio", "float"),
            ("Alpha:", "alpha", "pct"),
            ("Beta:", "beta", "float"),
        ]

        for label, key, fmt in metrics_list:
            sv = strategy_metrics.get(key, np.nan)
            bv = benchmark_metrics.get(key, np.nan)

            if fmt == "pct":
                sv_f = f"{sv:.2%}" if pd.notna(sv) else "N/A"
                bv_f = f"{bv:.2%}" if pd.notna(bv) else "N/A"
            elif fmt == "int":
                sv_f = f"{int(sv)}" if pd.notna(sv) else "N/A"
                bv_f = f"{int(bv)}" if pd.notna(bv) else "N/A"
            else:
                sv_f = f"{sv:.2f}" if pd.notna(sv) else "N/A"
                bv_f = f"{bv:.2f}" if pd.notna(bv) else "N/A"

            print(f"\n{label:>{label_w}} {sv_f:>{value_w}} {bv_f:>{bench_w}}")

        print("\n" + "=" * span)
