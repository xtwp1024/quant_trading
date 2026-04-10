"""Real-Time Risk Metrics Calculator.

Provides real-time calculation of:
- Sharpe, Sortino, Calmar ratios
- Current drawdown
- VaR (Value at Risk) - historical and parametric
- Expected Shortfall (CVaR)
- Rolling metrics
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RealTimeMetrics:
    """Real-time risk metrics snapshot."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    current_drawdown: float
    max_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    tail_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    period_return: float
    volatility: float
    downside_deviation: float


# ---------------------------------------------------------------------------
# Real-Time Metrics Calculator
# ---------------------------------------------------------------------------

class RealTimeMetricsCalculator:
    """Calculate risk metrics in real-time from trade history and equity curve.

    Designed for low-latency metric updates during live trading sessions.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        trading_hours_per_day: float = 6.5,
    ):
        """
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year
            trading_hours_per_day: Hours of trading per day (for 24h crypto, use 24)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.trading_hours_per_day = trading_hours_per_day

        # History
        self.equity_curve: List[float] = []
        self.returns: List[float] = []
        self.trade_pnls: List[float] = []
        self.daily_returns: List[float] = []
        self.peak_equity: float = 0.0

        # Cached metrics
        self._cached_metrics: Optional[RealTimeMetrics] = None
        self._cache_valid: bool = False

    def update(
        self,
        equity: float,
        trade_pnl: Optional[float] = None,
        daily_return: Optional[float] = None,
    ) -> None:
        """Update with new data point.

        Args:
            equity: Current account equity
            trade_pnl: Individual trade PnL (if a trade closed)
            daily_return: Daily return percentage (if a day passed)
        """
        # Update equity curve
        self.equity_curve.append(equity)
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Calculate return if we have previous equity
        if len(self.equity_curve) >= 2:
            prev_equity = self.equity_curve[-2]
            if prev_equity > 0:
                ret = (equity - prev_equity) / prev_equity
                self.returns.append(ret)

        # Record trade PnL
        if trade_pnl is not None:
            self.trade_pnls.append(trade_pnl)

        # Record daily return
        if daily_return is not None:
            self.daily_returns.append(daily_return)

        # Invalidate cache
        self._cache_valid = False

    def calculate(self) -> RealTimeMetrics:
        """Calculate all real-time metrics.

        Returns:
            RealTimeMetrics dataclass with current metrics
        """
        if self._cache_valid and self._cached_metrics:
            return self._cached_metrics

        # Basic statistics
        total_trades = len(self.trade_pnls)
        period_return = self._calculate_period_return()
        volatility = self._calculate_volatility()
        downside_dev = self._calculate_downside_deviation()

        # Ratios
        sharpe = self._calculate_sharpe(volatility)
        sortino = self._calculate_sortino(downside_dev)
        calmar = self._calculate_calmar()

        # Drawdown
        current_dd, max_dd = self._calculate_drawdowns()

        # VaR and CVaR
        var_95, var_99 = self._calculate_var()
        cvar_95, cvar_99 = self._calculate_cvar()

        # Other metrics
        tail_ratio = self._calculate_tail_ratio()
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()

        self._cached_metrics = RealTimeMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            tail_ratio=tail_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            period_return=period_return,
            volatility=volatility,
            downside_deviation=downside_dev,
        )
        self._cache_valid = True

        return self._cached_metrics

    def _calculate_period_return(self) -> float:
        """Calculate total period return."""
        if len(self.equity_curve) < 2:
            return 0.0
        start = self.equity_curve[0]
        end = self.equity_curve[-1]
        if start <= 0:
            return 0.0
        return (end - start) / start

    def _calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        if len(self.returns) < 2:
            return 0.0
        mean_return = sum(self.returns) / len(self.returns)
        variance = sum((r - mean_return) ** 2 for r in self.returns) / (len(self.returns) - 1)
        annual_vol = math.sqrt(variance * self.periods_per_year)
        return annual_vol

    def _calculate_downside_deviation(self, target: float = 0.0) -> float:
        """Calculate annualized downside deviation (Sortino denominator)."""
        if len(self.returns) < 2:
            return 0.0
        downside_returns = [max(target - r, 0) ** 2 for r in self.returns]
        if not downside_returns:
            return 0.0
        down_var = sum(downside_returns) / len(self.returns)
        return math.sqrt(down_var * self.periods_per_year)

    def _calculate_sharpe(self, volatility: float) -> float:
        """Calculate Sharpe ratio."""
        if volatility <= 0 or len(self.returns) < 2:
            return 0.0
        mean_return = sum(self.returns) / len(self.returns)
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess_return = mean_return - rf_per_period
        return (excess_return / volatility) * math.sqrt(self.periods_per_year)

    def _calculate_sortino(self, downside_dev: float) -> float:
        """Calculate Sortino ratio."""
        if downside_dev <= 0 or len(self.returns) < 2:
            return 0.0
        mean_return = sum(self.returns) / len(self.returns)
        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess_return = mean_return - rf_per_period
        return (excess_return / downside_dev) * math.sqrt(self.periods_per_year)

    def _calculate_calmar(self) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if len(self.equity_curve) < 2:
            return 0.0
        _, max_dd = self._calculate_drawdowns()
        if abs(max_dd) < 1e-8:
            return 0.0
        ann_return = self._calculate_period_return() * self.periods_per_year / len(self.returns) if self.returns else 0
        return ann_return / abs(max_dd)

    def _calculate_drawbacks(self) -> Tuple[float, float]:
        """Calculate current and max drawdown."""
        if len(self.equity_curve) < 2:
            return 0.0, 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        current_dd = (peak - self.equity_curve[-1]) / peak if peak > 0 else 0.0
        return current_dd, max_dd

    def _calculate_drawdowns(self) -> Tuple[float, float]:
        """Calculate current and max drawdown."""
        if not self.equity_curve:
            return 0.0, 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        current_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        if peak > 0:
            current_dd = (peak - self.equity_curve[-1]) / peak
        return current_dd, max_dd

    def _calculate_var(self) -> Tuple[float, float]:
        """Calculate VaR at 95% and 99% confidence."""
        if len(self.returns) < 10:
            return 0.0, 0.0
        sorted_returns = sorted(self.returns)
        n = len(sorted_returns)
        # 95% VaR
        var_95_idx = max(int(n * 0.05), 0)
        var_95 = sorted_returns[var_95_idx]
        # 99% VaR
        var_99_idx = max(int(n * 0.01), 0)
        var_99 = sorted_returns[var_99_idx]
        return var_95, var_99

    def _calculate_cvar(self) -> Tuple[float, float]:
        """Calculate CVaR (Expected Shortfall) at 95% and 99%."""
        if len(self.returns) < 10:
            return 0.0, 0.0
        sorted_returns = sorted(self.returns)
        n = len(sorted_returns)
        # 95% CVaR - average of worst 5%
        var_95_idx = max(int(n * 0.05), 0)
        tail_95 = sorted_returns[:var_95_idx + 1]
        cvar_95 = sum(tail_95) / len(tail_95) if tail_95 else sorted_returns[0]
        # 99% CVaR - average of worst 1%
        var_99_idx = max(int(n * 0.01), 0)
        tail_99 = sorted_returns[:var_99_idx + 1]
        cvar_99 = sum(tail_99) / len(tail_99) if tail_99 else sorted_returns[0]
        return cvar_95, cvar_99

    def _calculate_tail_ratio(self) -> float:
        """Calculate tail ratio (95th pct return / 5th pct return abs)."""
        if len(self.returns) < 20:
            return 1.0
        sorted_returns = sorted(self.returns)
        n = len(sorted_returns)
        p95 = sorted_returns[int(n * 0.95)]
        p5 = sorted_returns[int(n * 0.05)]
        if abs(p5) < 1e-10:
            return float('inf') if p95 > 0 else 0.0
        return p95 / abs(p5)

    def _calculate_win_rate(self) -> float:
        """Calculate win rate (percentage of profitable trades)."""
        if not self.trade_pnls:
            return 0.0
        wins = sum(1 for pnl in self.trade_pnls if pnl > 0)
        return wins / len(self.trade_pnls)

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not self.trade_pnls:
            return 0.0
        gross_profit = sum(pnl for pnl in self.trade_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in self.trade_pnls if pnl < 0))
        if gross_loss < 1e-10:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def rolling_sharpe(self, window: int = 30) -> List[float]:
        """Calculate rolling Sharpe ratio over a window."""
        if len(self.returns) < window:
            return []
        rf_per_period = self.risk_free_rate / self.periods_per_year
        result = []
        for i in range(window, len(self.returns) + 1):
            window_returns = self.returns[i - window:i]
            mean_ret = sum(window_returns) / len(window_returns)
            variance = sum((r - mean_ret) ** 2 for r in window_returns) / (window - 1)
            std = math.sqrt(variance)
            if std > 0:
                sharpe = ((mean_ret - rf_per_period) / std) * math.sqrt(self.periods_per_year)
                result.append(sharpe)
            else:
                result.append(0.0)
        return result

    def rolling_max_drawdown(self, window: int = 30) -> List[float]:
        """Calculate rolling maximum drawdown over a window."""
        if len(self.equity_curve) < window:
            return []
        result = []
        for i in range(window, len(self.equity_curve) + 1):
            window_equity = self.equity_curve[i - window:i]
            peak = max(window_equity)
            current = window_equity[-1]
            dd = (peak - current) / peak if peak > 0 else 0.0
            result.append(dd)
        return result

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary."""
        m = self.calculate()
        return {
            "sharpe_ratio": round(m.sharpe_ratio, 4),
            "sortino_ratio": round(m.sortino_ratio, 4),
            "calmar_ratio": round(m.calmar_ratio, 4),
            "current_drawdown": round(m.current_drawdown, 6),
            "max_drawdown": round(m.max_drawdown, 6),
            "var_95": round(m.var_95, 6),
            "var_99": round(m.var_99, 6),
            "cvar_95": round(m.cvar_95, 6),
            "cvar_99": round(m.cvar_99, 6),
            "tail_ratio": round(m.tail_ratio, 4),
            "win_rate": round(m.win_rate, 4),
            "profit_factor": round(m.profit_factor, 4) if m.profit_factor != float('inf') else "inf",
            "total_trades": m.total_trades,
            "period_return": round(m.period_return, 6),
            "volatility": round(m.volatility, 6),
            "downside_deviation": round(m.downside_deviation, 6),
        }


# Convenience functions

def calculate_sharpe(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns list."""
    if len(returns) < 2:
        return 0.0
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std < 1e-10:
        return 0.0
    return (mean_ret - risk_free_rate / 252) / std * math.sqrt(252)


def calculate_sortino(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio from returns list."""
    if len(returns) < 2:
        return 0.0
    mean_ret = sum(returns) / len(returns)
    downside = [max(-risk_free_rate / 252, 0) ** 2 for r in returns]
    down_dev = math.sqrt(sum(downside) / len(returns)) * math.sqrt(252)
    if down_dev < 1e-10:
        return 0.0
    return (mean_ret - risk_free_rate / 252) / down_dev


def calculate_var(returns: List[float], confidence: float = 0.95) -> float:
    """Calculate Value at Risk from returns list."""
    if len(returns) < 10:
        return 0.0
    sorted_returns = sorted(returns)
    idx = max(int(len(sorted_returns) * (1 - confidence)), 0)
    return sorted_returns[idx]


def calculate_cvar(returns: List[float], confidence: float = 0.95) -> float:
    """Calculate Conditional VaR (Expected Shortfall) from returns list."""
    if len(returns) < 10:
        return 0.0
    sorted_returns = sorted(returns)
    idx = max(int(len(sorted_returns) * (1 - confidence)), 0)
    tail = sorted_returns[:idx + 1]
    return sum(tail) / len(tail) if tail else sorted_returns[0]


__all__ = [
    "RealTimeMetrics",
    "RealTimeMetricsCalculator",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_var",
    "calculate_cvar",
]
