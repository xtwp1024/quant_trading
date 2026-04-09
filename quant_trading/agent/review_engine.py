#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review Engine for ETH Long Runner.
复盘引擎 - 对比signal vs 实际价格, 更新记忆

功能:
1. 对比决策信号与实际价格变动
2. 计算P&L
3. 更新MemoryBank中的决策记录
4. 生成复盘报告
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger("ReviewEngine")


@dataclass
class ReviewResult:
    """复盘结果"""
    decision_id: int
    timestamp: str
    signal: str
    signal_price: float
    review_price: float
    holding_period_hours: float
    price_change_pct: float
    pnl: float  # 基于信号的盈亏
    correct: bool
    notes: str


class ReviewEngine:
    """
    复盘引擎

    每4小时执行一次:
    1. 获取最近N小时内产生的决策
    2. 对比决策价格与当前价格
    3. 评估决策正确性
    4. 更新MemoryBank
    """

    def __init__(self, memory_bank):
        """
        Args:
            memory_bank: MemoryBank实例
        """
        self.memory_bank = memory_bank

    def review_decisions(self, current_price: float, lookback_hours: int = 4) -> List[ReviewResult]:
        """
        复盘最近决策

        Args:
            current_price: 当前价格
            lookback_hours: 回看小时数

        Returns:
            复盘结果列表
        """
        decisions = self.memory_bank.get_recent_decisions(limit=50)

        results = []
        for decision in decisions:
            # 检查是否在回看时间范围内
            decision_time = datetime.fromisoformat(decision.timestamp)
            hours_elapsed = (datetime.now() - decision_time).total_seconds() / 3600

            if hours_elapsed < 1:  # 至少1小时后才复盘
                continue

            signal_price = decision.price
            price_change_pct = (current_price - signal_price) / signal_price * 100

            # 计算PnL (基于信号方向)
            if decision.signal == "BUY":
                pnl = price_change_pct  # 做多
            elif decision.signal == "SELL":
                pnl = -price_change_pct  # 做空 (反向)
            else:
                pnl = 0.0

            # 判断正确性
            if decision.signal == "HOLD":
                correct = abs(price_change_pct) < 1.0  # 震荡市中hold是合理的
            elif decision.signal == "BUY":
                correct = price_change_pct > 0.5  # 1%涨幅算正确
            elif decision.signal == "SELL":
                correct = price_change_pct < -0.5  # 下跌1%算正确
            else:
                correct = None

            # 生成复盘备注
            if correct:
                notes = f"[OK] 决策正确: signal={decision.signal}, pnl={pnl:.2f}%"
            else:
                notes = f"[ERR] 决策错误: signal={decision.signal}, pnl={pnl:.2f}%"

            result = ReviewResult(
                decision_id=decision.id,
                timestamp=decision.timestamp,
                signal=decision.signal,
                signal_price=signal_price,
                review_price=current_price,
                holding_period_hours=hours_elapsed,
                price_change_pct=price_change_pct,
                pnl=pnl,
                correct=correct,
                notes=notes
            )

            # 更新MemoryBank
            if decision.id and correct is not None:
                self.memory_bank.update_decision_pnl(decision.id, pnl, correct)

            results.append(result)
            logger.info(f"[REVIEW] 复盘: decision_id={decision.id}, {notes}")

        return results

    def calculate_signal_performance(self, days: int = 7) -> Dict[str, Any]:
        """
        计算信号表现统计

        Args:
            days: 统计天数

        Returns:
            性能统计字典
        """
        stats = self.memory_bank.get_signal_stats(days)

        win_rate = stats.get("win_rate", 0)
        total = stats.get("total", 0)
        correct = stats.get("correct", 0)

        # 计算期望P&L
        avg_score = stats.get("avg_score", 0)

        return {
            "period_days": days,
            "total_decisions": total,
            "correct_decisions": correct,
            "win_rate": win_rate * 100,  # 转换为百分比
            "avg_consensus_score": avg_score,
            "signal_breakdown": {
                "buy": stats.get("buy", 0),
                "sell": stats.get("sell", 0),
                "hold": stats.get("hold", 0)
            }
        }

    def generate_review_report(self, results: List[ReviewResult]) -> str:
        """
        生成复盘报告

        Args:
            results: 复盘结果列表

        Returns:
            格式化的复盘报告字符串
        """
        if not results:
            return "[REVIEW] 本周期无复盘数据"

        report_lines = ["=" * 60, "[REVIEW] 复盘报告", "=" * 60]

        total = len(results)
        correct = sum(1 for r in results if r.correct)
        win_rate = correct / total * 100 if total > 0 else 0

        avg_pnl = np.mean([r.pnl for r in results])
        total_pnl = sum(r.pnl for r in results)

        report_lines.append(f"复盘数量: {total}")
        report_lines.append(f"正确数量: {correct}/{total}")
        report_lines.append(f"胜率: {win_rate:.1f}%")
        report_lines.append(f"平均P&L: {avg_pnl:+.2f}%")
        report_lines.append(f"累计P&L: {total_pnl:+.2f}%")
        report_lines.append("-" * 60)
        report_lines.append("详细复盘:")

        for r in results:
            report_lines.append(
                f"  [{r.timestamp[:16]}] {r.signal} @ ${r.signal_price:.2f} → "
                f"${r.review_price:.2f} ({r.price_change_pct:+.2f}%) P&L={r.pnl:+.2f}% {r.notes}"
            )

        report_lines.append("=" * 60)
        return "\n".join(report_lines)
