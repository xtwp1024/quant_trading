#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generator for ETH Long Runner.
自动报告输出 - 每日/每周报告

输出形式:
- 日报 (reports/daily_YYYYMMDD.md)
- 周报 (reports/weekly_YYYYWW.md)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger("ReportGenerator")


@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: str
    signal: str  # BUY/SELL/HOLD
    strength: float
    price: float
    target_zone: tuple  # (lower, upper)
    stop_loss: float
    score: float
    risk_level: str  # LOW/MEDIUM/HIGH


class ReportGenerator:
    """
    报告生成器

    生成格式化报告:
    1. 日报 - 每日收盘后
    2. 周报 - 每周一
    3. 实时信号 - 每次决策时
    """

    def __init__(self, reports_dir: str = None):
        """
        Args:
            reports_dir: 报告输出目录
        """
        import os
        if reports_dir is None:
            _base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            reports_dir = os.path.join(_base, "reports")
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily_report(
        self,
        date: datetime,
        signals: List[TradingSignal],
        consensus_score: float,
        indicators: Dict[str, Any],
        view_evolution: List[Dict],
        dream_result: Optional[Dict] = None,
        performance: Optional[Dict] = None
    ) -> str:
        """
        生成日报

        Args:
            date: 日期
            signals: 信号列表
            consensus_score: 共识评分
            indicators: 技术指标
            view_evolution: 观点演化数据
            dream_result: 梦境推演结果
            performance: 表现数据

        Returns:
            报告路径
        """
        date_str = date.strftime("%Y%m%d")
        filename = self.reports_dir / f"daily_{date_str}.md"

        lines = [
            "# ETH 量化日报",
            f"**日期**: {date.strftime('%Y-%m-%d')}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 执行摘要",
            "",
        ]

        # 当前信号
        if signals:
            latest = signals[-1]
            lines.extend([
                f"| 项目 | 值 |",
                f"|------|-----|",
                f"| 最终信号 | **{latest.signal}** |",
                f"| 信号强度 | {latest.strength:.2f} |",
                f"| 当前价格 | ${latest.price:.2f} |",
                f"| 目标区间 | ${latest.target_zone[0]:.2f} - ${latest.target_zone[1]:.2f} |",
                f"| 止损位 | ${latest.stop_loss:.2f} |",
                f"| 五力评分 | {consensus_score:+.3f} |",
                f"| 风险等级 | {latest.risk_level} |",
            ])

        lines.extend(["", "## 技术指标", ""])
        lines.extend([
            "| 指标 | 值 |",
            "|------|-----|",
            f"| RSI(14) | {indicators.get('rsi', 'N/A'):.1f} |" if isinstance(indicators.get('rsi'), (int, float)) else f"| RSI(14) | {indicators.get('rsi', 'N/A')} |",
            f"| MACD | {indicators.get('macd_dif', 0):.4f} / {indicators.get('macd_dea', 0):.4f} |",
            f"| 布林带 | ${indicators.get('boll_lower', 0):.2f} - ${indicators.get('boll_upper', 0):.2f} |",
            f"| ATR(14) | ${indicators.get('atr', 0):.2f} |",
            f"| 趋势 | {indicators.get('trend', 'N/A')} |",
        ])

        # 观点演化
        lines.extend(["", "## 观点演化", ""])
        if view_evolution:
            lines.extend([
                "| 时间 | 观点 | 置信度 | 价格 |",
                "|------|------|--------|------|",
            ])
            for view in view_evolution[-10:]:  # 最近10条
                view_type = view.get("view_type", "N/A")
                conf = view.get("confidence", 0)
                price = view.get("price", 0)
                ts = view.get("timestamp", "")[11:19]
                lines.append(f"| {ts} | {view_type} | {conf:.2f} | ${price:.2f} |")
        else:
            lines.append("*暂无观点数据*")

        # 梦境推演
        if dream_result:
            lines.extend(["", "## 梦境推演摘要", ""])
            lines.append(f"- 初始价格: ${dream_result.get('initial_price', 0):.2f}")
            lines.append(f"- 最终价格: ${dream_result.get('final_price', 0):.2f}")
            lines.append(f"- 最大涨幅: {dream_result.get('max_gain_pct', 0):.1f}%")
            lines.append(f"- 最大跌幅: {dream_result.get('max_loss_pct', 0):.1f}%")
            lines.append(f"- 策略胜率: {dream_result.get('win_rate', 0):.1%}")

        # 性能统计
        if performance:
            lines.extend(["", "## 性能统计", ""])
            lines.extend([
                f"- 决策总数: {performance.get('total_decisions', 0)}",
                f"- 胜率: {performance.get('win_rate', 0):.1%}",
                f"- 平均评分: {performance.get('avg_consensus_score', 0):.3f}",
            ])

        lines.extend(["", "---", f"*报告生成时间: {datetime.now().isoformat()}*"])

        content = "\n".join(lines)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"[REPORT] 日报已生成: {filename}")
        return str(filename)

    def generate_weekly_report(
        self,
        week_start: datetime,
        week_end: datetime,
        signals: List[TradingSignal],
        performance: Dict[str, Any],
        view_evolution: List[Dict],
        weight_changes: List[Dict]
    ) -> str:
        """
        生成周报

        Args:
            week_start: 周开始日期
            week_end: 周结束日期
            signals: 周内所有信号
            performance: 性能数据
            view_evolution: 观点演化
            weight_changes: 权重变化

        Returns:
            报告路径
        """
        year, week_num, _ = week_end.isocalendar()
        filename = self.reports_dir / f"weekly_{year}W{week_num:02d}.md"

        lines = [
            "# ETH 量化周报",
            f"**周次**: {year}W{week_num:02d}",
            f"**周期**: {week_start.strftime('%Y-%m-%d')} - {week_end.strftime('%Y-%m-%d')}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 执行摘要",
            "",
        ]

        # 统计
        buy_count = sum(1 for s in signals if s.signal == "BUY")
        sell_count = sum(1 for s in signals if s.signal == "SELL")
        hold_count = sum(1 for s in signals if s.signal == "HOLD")

        lines.extend([
            f"| 项目 | 值 |",
            f"|------|-----|",
            f"| 信号总数 | {len(signals)} |",
            f"| 买入信号 | {buy_count} |",
            f"| 卖出信号 | {sell_count} |",
            f"| 观望信号 | {hold_count} |",
            f"| 周胜率 | {performance.get('win_rate', 0):.1%} |",
            f"| 周累计P&L | {performance.get('total_pnl', 0):+.2f}% |",
        ])

        # 信号回顾
        lines.extend(["", "## 信号回顾", ""])
        if signals:
            lines.extend([
                "| 日期 | 信号 | 强度 | 价格 |",
                "|------|------|------|------|",
            ])
            for sig in signals:
                ts = sig.timestamp[:10]
                lines.append(f"| {ts} | {sig.signal} | {sig.strength:.2f} | ${sig.price:.2f} |")
        else:
            lines.append("*本周无信号*")

        # 观点演化
        lines.extend(["", "## 观点演化曲线", ""])
        if view_evolution:
            bull_days = sum(1 for v in view_evolution if v.get("view_type") == "BULL")
            bear_days = sum(1 for v in view_evolution if v.get("view_type") == "BEAR")
            lines.append(f"- 多头观点天数: {bull_days}")
            lines.append(f"- 空头观点天数: {bear_days}")
            lines.append(f"- 多空比: {bull_days/max(bear_days,1):.2f}")

        # 权重变化
        lines.extend(["", "## 权重调整", ""])
        if weight_changes:
            for change in weight_changes:
                lines.append(f"- {change['unit']}: {change['old']:.2f} → {change['new']:.2f} ({change['reason']})")
        else:
            lines.append("*本周无权重调整*")

        # 下周展望
        lines.extend(["", "## 下周展望", ""])
        if view_evolution:
            recent_views = view_evolution[-5:]
            if recent_views:
                dominant = max(set(v.get("view_type", "NEUTRAL") for v in recent_views), default="NEUTRAL")
                lines.append(f"根据本周表现，预计下周整体观点偏 **{dominant}**。")
        else:
            lines.append("*数据不足，无法预测*")

        lines.extend(["", "---", f"*报告生成时间: {datetime.now().isoformat()}*"])

        content = "\n".join(lines)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"[REPORT] 周报已生成: {filename}")
        return str(filename)

    def generate_signal_alert(self, signal: TradingSignal, indicators: Dict[str, Any]) -> str:
        """
        生成信号提醒

        Returns:
            格式化的信号字符串
        """
        emoji = "[GREEN]" if signal.signal == "BUY" else "[RED]" if signal.signal == "SELL" else "[NEUTRAL]"

        return f"""
{'='*60}
{emoji} ETH 交易信号
{'='*60}
信号: {signal.signal}
强度: {signal.strength:.2f}
价格: ${signal.price:.2f}
目标: ${signal.target_zone[0]:.2f} - ${signal.target_zone[1]:.2f}
止损: ${signal.stop_loss:.2f}
评分: {signal.score:+.3f}
风险: {signal.risk_level}
{'='*60}
"""
