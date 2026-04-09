"""
Backtest Analyzer - 回测结果分析器
================================

从 maverick-mcp 吸收的专业回测分析组件。

功能:
- VectorBT 风格的回测指标计算
- A-F 性能评级系统
- 策略对比与排名
- 风险评估与建议

Absorbed from: D:/Hive/Data/trading_repos/maverick-mcp/maverick_mcp/backtesting/analysis.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def convert_to_native(value: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32, np.float16)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, "item"):  # For numpy scalars
        return value.item()
    elif isinstance(value, float) and np.isnan(value):
        return None
    return value


@dataclass
class BacktestMetrics:
    """回测指标数据结构"""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_duration_days: float = 0.0
    recovery_factor: float = 0.0
    kelly_criterion: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": convert_to_native(self.total_return),
            "annual_return": convert_to_native(self.annual_return),
            "sharpe_ratio": convert_to_native(self.sharpe_ratio),
            "sortino_ratio": convert_to_native(self.sortino_ratio),
            "calmar_ratio": convert_to_native(self.calmar_ratio),
            "max_drawdown": convert_to_native(self.max_drawdown),
            "max_drawdown_pct": convert_to_native(self.max_drawdown_pct),
            "win_rate": convert_to_native(self.win_rate),
            "total_trades": int(self.total_trades),
            "profit_factor": convert_to_native(self.profit_factor),
            "avg_win": convert_to_native(self.avg_win),
            "avg_loss": convert_to_native(self.avg_loss),
            "best_trade": convert_to_native(self.best_trade),
            "worst_trade": convert_to_native(self.worst_trade),
            "avg_duration_days": convert_to_native(self.avg_duration_days),
            "recovery_factor": convert_to_native(self.recovery_factor),
            "kelly_criterion": convert_to_native(self.kelly_criterion),
        }


@dataclass
class BacktestTrade:
    """单笔交易记录"""
    entry_time: Any
    exit_time: Any
    pnl: float
    return_pct: float
    duration_days: float = 0.0


class BacktestAnalyzer:
    """
    回测结果分析器

    提供专业的回测性能分析，包括:
    - 风险调整后的指标计算
    - A-F 性能评级
    - 策略对比与排名
    - 改进建议

    使用方法:
        analyzer = BacktestAnalyzer()
        metrics = analyzer.calculate_metrics(equity_curve, trades)
        grade = analyzer.grade_performance(metrics)
        analysis = analyzer.analyze(results)
    """

    def __init__(self):
        """初始化分析器."""
        pass

    def calculate_metrics(
        self,
        equity_curve: List[float],
        trades: List[BacktestTrade],
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.0,
    ) -> BacktestMetrics:
        """
        从权益曲线和交易记录计算回测指标.

        Args:
            equity_curve: 权益曲线 [initial, ..., final]
            trades: 交易记录列表
            initial_capital: 初始资金
            risk_free_rate: 无风险利率 (年化)

        Returns:
            BacktestMetrics 对象
        """
        if not equity_curve or len(equity_curve) < 2:
            return BacktestMetrics()

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        metrics = BacktestMetrics()

        # 基本收益指标
        metrics.total_return = (equity[-1] - equity[0]) / equity[0]
        metrics.annual_return = self._calculate_annual_return(equity, len(equity))
        metrics.total_trades = len(trades)

        # 风险调整指标
        if len(returns) > 0 and np.std(returns) > 0:
            # 夏普比率
            excess_return = returns - risk_free_rate / 252
            metrics.sharpe_ratio = np.mean(excess_return) / np.std(excess_return) * np.sqrt(252)

            # 索提诺比率 (只考虑下行波动)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                metrics.sortino_ratio = np.mean(excess_return) / np.std(negative_returns) * np.sqrt(252)

        # 最大回撤
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        metrics.max_drawdown = np.min(drawdown)
        metrics.max_drawdown_pct = abs(metrics.max_drawdown)

        # 卡玛比率
        if metrics.max_drawdown_pct > 0 and metrics.annual_return > 0:
            metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown_pct

        # 交易统计
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]

            metrics.win_rate = len(winning_trades) / len(trades) if trades else 0

            if winning_trades:
                metrics.avg_win = np.mean([t.pnl for t in winning_trades])
                metrics.best_trade = max(t.pnl for t in winning_trades)

            if losing_trades:
                metrics.avg_loss = abs(np.mean([t.pnl for t in losing_trades]))
                metrics.worst_trade = min(t.pnl for t in losing_trades)

            total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
            total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0

            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses

            # 平均持仓天数
            durations = [t.duration_days for t in trades if t.duration_days > 0]
            if durations:
                metrics.avg_duration_days = np.mean(durations)

            # 恢复因子
            if metrics.max_drawdown > 0:
                metrics.recovery_factor = (equity[-1] - equity[0]) / abs(metrics.max_drawdown)

            # Kelly Criterion
            if metrics.win_rate > 0 and metrics.profit_factor > 0:
                win_rate = metrics.win_rate
                loss_rate = 1 - win_rate
                win_loss_ratio = metrics.profit_factor
                # Kelly = (p * b - q) / b
                kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
                metrics.kelly_criterion = max(0, min(1, kelly))

        return metrics

    def _calculate_annual_return(self, equity: np.ndarray, n_periods: int) -> float:
        """计算年化收益率."""
        if len(equity) < 2 or equity[0] <= 0:
            return 0.0

        total_return = equity[-1] / equity[0]
        # 假设每日数据，一年252个交易日
        years = n_periods / 252
        if years <= 0:
            return 0.0

        return (total_return ** (1 / years)) - 1

    def grade_performance(self, metrics: BacktestMetrics) -> str:
        """
        基于多指标给策略打分 (A-F).

        评分标准:
        - A (90%+): 卓越风险调整收益
        - B (80-89%): 良好
        - C (70-79%): 一般
        - D (60-69%): 较差
        - F (<60%): 不合格

        Args:
            metrics: 回测指标

        Returns:
            评级字母 (A/B/C/D/F)
        """
        score = 0
        max_score = 100

        # 夏普比率 (30分)
        sharpe = metrics.sharpe_ratio
        if sharpe >= 2.0:
            score += 30
        elif sharpe >= 1.5:
            score += 25
        elif sharpe >= 1.0:
            score += 20
        elif sharpe >= 0.5:
            score += 10
        else:
            score += 5

        # 总收益率 (25分)
        total_return = metrics.total_return
        if total_return >= 0.50:  # 50%+
            score += 25
        elif total_return >= 0.30:
            score += 20
        elif total_return >= 0.15:
            score += 15
        elif total_return >= 0.05:
            score += 10
        elif total_return > 0:
            score += 5

        # 胜率 (20分)
        win_rate = metrics.win_rate
        if win_rate >= 0.60:
            score += 20
        elif win_rate >= 0.50:
            score += 15
        elif win_rate >= 0.40:
            score += 10
        else:
            score += 5

        # 最大回撤 (15分)
        max_dd = abs(metrics.max_drawdown_pct)
        if max_dd <= 0.10:  # 小于10%
            score += 15
        elif max_dd <= 0.20:
            score += 12
        elif max_dd <= 0.30:
            score += 8
        elif max_dd <= 0.40:
            score += 4

        # 盈利因子 (10分)
        profit_factor = metrics.profit_factor
        if profit_factor >= 2.0:
            score += 10
        elif profit_factor >= 1.5:
            score += 8
        elif profit_factor >= 1.2:
            score += 5
        elif profit_factor > 1.0:
            score += 3

        # 转换为等级
        percentage = (score / max_score) * 100
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"

    def assess_risk(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """
        评估风险等级.

        Returns:
            风险评估字典
        """
        max_dd = abs(metrics.max_drawdown_pct)
        sortino = metrics.sortino_ratio
        sharpe = metrics.sharpe_ratio
        calmar = metrics.calmar_ratio

        # 风险等级
        if max_dd > 0.40:
            risk_level = "Very High"
        elif max_dd > 0.30:
            risk_level = "High"
        elif max_dd > 0.20:
            risk_level = "Medium"
        elif max_dd > 0.10:
            risk_level = "Low-Medium"
        else:
            risk_level = "Low"

        # 下行保护
        if sortino > 1.5:
            downside_protection = "Good"
        elif sortino > 0.5:
            downside_protection = "Moderate"
        else:
            downside_protection = "Poor"

        return {
            "risk_level": risk_level,
            "max_drawdown_pct": f"{max_dd * 100:.1f}%",
            "sortino_ratio": round(sortino, 2) if sortino else 0,
            "calmar_ratio": round(calmar, 2) if calmar else 0,
            "sharpe_ratio": round(sharpe, 2) if sharpe else 0,
            "downside_protection": downside_protection,
        }

    def analyze_trades(self, trades: List[BacktestTrade]) -> Dict[str, Any]:
        """
        分析交易质量.

        Returns:
            交易质量分析
        """
        if not trades:
            return {
                "quality": "No trades",
                "total_trades": 0,
                "frequency": "None",
            }

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        total_trades = len(trades)
        win_rate = len(winning) / total_trades if trades else 0

        # 交易频率
        if total_trades < 10:
            frequency = "Very Low"
        elif total_trades < 50:
            frequency = "Low"
        elif total_trades < 100:
            frequency = "Moderate"
        elif total_trades < 200:
            frequency = "High"
        else:
            frequency = "Very High"

        # 交易质量
        profit_factor = 0
        if winning and losing:
            total_wins = sum(t.pnl for t in winning)
            total_losses = abs(sum(t.pnl for t in losing))
            if total_losses > 0:
                profit_factor = total_wins / total_losses

        if win_rate >= 0.60 and profit_factor >= 1.5:
            quality = "Excellent"
        elif win_rate >= 0.50 and profit_factor >= 1.2:
            quality = "Good"
        elif win_rate >= 0.40:
            quality = "Average"
        else:
            quality = "Poor"

        return {
            "quality": quality,
            "total_trades": total_trades,
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "frequency": frequency,
            "win_rate": f"{win_rate * 100:.1f}%",
        }

    def identify_strengths(self, metrics: BacktestMetrics) -> List[str]:
        """识别策略优势."""
        strengths = []

        if metrics.sharpe_ratio >= 1.5:
            strengths.append("Excellent risk-adjusted returns (Sharpe >= 1.5)")
        if metrics.win_rate >= 0.60:
            strengths.append("High win rate (>= 60%)")
        if metrics.max_drawdown_pct <= 0.15:
            strengths.append("Low maximum drawdown (<= 15%)")
        if metrics.profit_factor >= 1.5:
            strengths.append("Strong profit factor (>= 1.5)")
        if metrics.sortino_ratio >= 2.0:
            strengths.append("Excellent downside protection (Sortino >= 2.0)")
        if metrics.calmar_ratio >= 1.0:
            strengths.append("Good return vs drawdown ratio (Calmar >= 1.0)")
        if metrics.recovery_factor >= 3.0:
            strengths.append("Quick drawdown recovery")
        if metrics.total_return >= 0.30:
            strengths.append("High total returns (>= 30%)")

        return strengths if strengths else ["Consistent performance"]

    def identify_weaknesses(self, metrics: BacktestMetrics) -> List[str]:
        """识别策略劣势."""
        weaknesses = []

        if metrics.sharpe_ratio < 0.5:
            weaknesses.append("Poor risk-adjusted returns (Sharpe < 0.5)")
        if metrics.win_rate < 0.40:
            weaknesses.append("Low win rate (< 40%)")
        if metrics.max_drawdown_pct > 0.30:
            weaknesses.append("High maximum drawdown (> 30%)")
        if metrics.profit_factor < 1.0:
            weaknesses.append("Unprofitable trades overall")
        if metrics.total_trades < 10:
            weaknesses.append("Insufficient trade signals (< 10 trades)")
        if metrics.sortino_ratio < 0:
            weaknesses.append("Poor downside protection")
        if metrics.total_return < 0:
            weaknesses.append("Negative returns")

        return weaknesses if weaknesses else ["Room for optimization"]

    def generate_recommendations(self, metrics: BacktestMetrics) -> List[str]:
        """生成改进建议."""
        recommendations = []

        # 风险管理的建议
        if metrics.max_drawdown_pct > 0.25:
            recommendations.append(
                "Consider implementing tighter stop-loss rules to reduce drawdowns"
            )

        # 胜率改进
        if metrics.win_rate < 0.45:
            recommendations.append("Refine entry signals to improve win rate")

        # 交易频率
        if metrics.total_trades < 20:
            recommendations.append(
                "Consider more sensitive parameters for increased signals"
            )
        elif metrics.total_trades > 200:
            recommendations.append("Filter signals to reduce overtrading")

        # 风险收益比
        if metrics.profit_factor < 1.5:
            recommendations.append("Adjust exit strategy for better risk-reward ratio")

        # 盈利因子
        if metrics.profit_factor < 1.2:
            recommendations.append(
                "Focus on cutting losses quicker and letting winners run"
            )

        # 夏普比率
        if metrics.sharpe_ratio < 1.0:
            recommendations.append("Consider position sizing based on volatility")

        # Kelly 建议
        kelly = metrics.kelly_criterion
        if kelly > 0 and kelly < 0.25:
            recommendations.append(
                f"Consider position size of {kelly * 100:.1f}% based on Kelly Criterion"
            )

        return (
            recommendations
            if recommendations
            else ["Strategy performing well, consider live testing"]
        )

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合分析回测结果.

        Args:
            results: 包含 'equity_curve', 'trades', 'initial_capital' 的字典

        Returns:
            完整分析报告
        """
        equity_curve = results.get("equity_curve", [])
        trades_data = results.get("trades", [])
        initial_capital = results.get("initial_capital", 10000.0)

        # 转换为 BacktestTrade 对象
        trades = []
        for t in trades_data:
            if isinstance(t, dict):
                trade = BacktestTrade(
                    entry_time=t.get("entry_time"),
                    exit_time=t.get("exit_time"),
                    pnl=t.get("pnl", 0),
                    return_pct=t.get("return", 0),
                    duration_days=t.get("duration_days", 0),
                )
            else:
                trade = t
            trades.append(trade)

        # 计算指标
        metrics = self.calculate_metrics(equity_curve, trades, initial_capital)

        # 生成分析
        return {
            "metrics": metrics.to_dict(),
            "grade": self.grade_performance(metrics),
            "risk_assessment": self.assess_risk(metrics),
            "trade_analysis": self.analyze_trades(trades),
            "strengths": self.identify_strengths(metrics),
            "weaknesses": self.identify_weaknesses(metrics),
            "recommendations": self.generate_recommendations(metrics),
            "summary": self._generate_summary(metrics),
        }

    def _generate_summary(self, metrics: BacktestMetrics) -> str:
        """生成文本总结."""
        total_return = metrics.total_return * 100
        sharpe = metrics.sharpe_ratio
        max_dd = metrics.max_drawdown_pct * 100
        win_rate = metrics.win_rate * 100
        total_trades = metrics.total_trades

        summary = f"策略生成 {total_return:.1f}% 收益率，夏普比率 {sharpe:.2f}。"
        summary += f" 最大回撤 {max_dd:.1f}%，胜率 {win_rate:.1f}%，共 {total_trades} 笔交易。"

        if sharpe >= 1.5 and max_dd <= 20:
            summary += " 整体表现卓越，风险调整收益强劲。"
        elif sharpe >= 1.0 and max_dd <= 30:
            summary += " 表现良好，风险水平可接受。"
        elif sharpe >= 0.5:
            summary += " 表现一般，可考虑优化。"
        else:
            summary += " 表现需要显著改进后再进行实盘。"

        return summary

    def compare_strategies(
        self, results_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        对比多个策略.

        Args:
            results_list: 策略回测结果列表

        Returns:
            对比分析报告
        """
        if not results_list:
            return {"error": "No results to compare"}

        comparisons = []

        for i, result in enumerate(results_list):
            equity = result.get("equity_curve", [])
            trades_data = result.get("trades", [])
            initial = result.get("initial_capital", 10000.0)

            trades = []
            for t in trades_data:
                if isinstance(t, dict):
                    trade = BacktestTrade(
                        entry_time=t.get("entry_time"),
                        exit_time=t.get("exit_time"),
                        pnl=t.get("pnl", 0),
                        return_pct=t.get("return", 0),
                    )
                else:
                    trade = t
                trades.append(trade)

            metrics = self.calculate_metrics(equity, trades, initial)

            comparisons.append({
                "strategy": result.get("strategy", f"Strategy {i+1}"),
                "parameters": result.get("parameters", {}),
                "total_return": metrics.total_return,
                "annual_return": metrics.annual_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_trades": metrics.total_trades,
                "grade": self.grade_performance(metrics),
                "metrics": metrics,
            })

        # 按夏普比率排序
        comparisons.sort(key=lambda x: x["sharpe_ratio"], reverse=True)

        # 添加排名
        for i, comp in enumerate(comparisons, 1):
            comp["rank"] = i

        # 找出各维度最佳
        best_return = max(comparisons, key=lambda x: x["total_return"])
        best_sharpe = max(comparisons, key=lambda x: x["sharpe_ratio"])
        best_drawdown = min(comparisons, key=lambda x: x["max_drawdown_pct"])
        best_win_rate = max(comparisons, key=lambda x: x["win_rate"])

        return {
            "rankings": [
                {k: v for k, v in comp.items() if k != "metrics"}
                for comp in comparisons
            ],
            "best_overall": comparisons[0] if comparisons else None,
            "best_return": best_return,
            "best_sharpe": best_sharpe,
            "best_drawdown": best_drawdown,
            "best_win_rate": best_win_rate,
            "summary": self._generate_comparison_summary(comparisons),
        }

    def _generate_comparison_summary(self, comparisons: List[Dict]) -> str:
        """生成策略对比总结."""
        if not comparisons:
            return "No strategies to compare"

        best = comparisons[0]
        summary = f"表现最好的策略是 {best['strategy']}，"
        summary += f"夏普比率 {best['sharpe_ratio']:.2f}，"
        summary += f"总收益率 {best['total_return'] * 100:.1f}%。"

        if len(comparisons) > 1:
            summary += f" 超越了 {len(comparisons) - 1} 个其他测试策略。"

        return summary


# 导出
__all__ = [
    "BacktestAnalyzer",
    "BacktestMetrics",
    "BacktestTrade",
    "convert_to_native",
]
