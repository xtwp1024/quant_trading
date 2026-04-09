#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning Loop for ETH Long Runner.
权重学习更新 - 从盈亏中调整共识引擎权重

功能:
1. 分析近期决策的胜率
2. 根据P&L调整各分析模块的权重
3. 动态更新ConsensusEngine权重
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("LearningLoop")


@dataclass
class WeightUpdate:
    """权重更新记录"""
    timestamp: str
    unit: str
    old_weight: float
    new_weight: float
    reason: str


class LearningLoop:
    """
    学习循环

    根据决策表现调整权重:
    1. 分析各分析模块的贡献
    2. 调整权重以优化表现
    3. 记录权重变化历史
    """

    # 默认权重
    DEFAULT_WEIGHTS = {
        "technical": 1.5,  # 技术面权重
        "research": 1.2,  # 研究团队权重
        "debate": 1.0,  # 辩论引擎权重
        "sentiment": 0.8,  # 情绪权重
        "onchain": 0.6,  # 链上权重
    }

    # 权重调整步长
    LEARNING_RATE = 0.1
    MAX_WEIGHT = 3.0
    MIN_WEIGHT = 0.2

    def __init__(self, memory_bank, consensus_engine=None):
        """
        Args:
            memory_bank: MemoryBank实例
            consensus_engine: ConsensusEngine实例 (可选)
        """
        self.memory_bank = memory_bank
        self.consensus_engine = consensus_engine
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.weight_history: list[WeightUpdate] = []

    def learn(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        执行学习，更新权重

        Args:
            lookback_days: 回看天数

        Returns:
            学习结果
        """
        logger.info(f"[BRAIN] 学习循环开始: 回看{lookback_days}天")

        # 获取决策统计
        stats = self.memory_bank.get_signal_stats(lookback_days)
        win_rate = stats.get("win_rate", 0)
        total = stats.get("total", 0)

        if total < 5:
            logger.info("决策数量不足，跳过学习")
            return {"skipped": True, "reason": "insufficient_data"}

        # 分析近期决策
        recent_decisions = self.memory_bank.get_recent_decisions(limit=total)

        # 按信号类型分组分析
        signal_performance = defaultdict(lambda: {"correct": 0, "total": 0})
        for decision in recent_decisions:
            if decision.correct is not None:
                signal_performance[decision.signal]["total"] += 1
                if decision.correct:
                    signal_performance[decision.signal]["correct"] += 1

        # 计算调整
        updates = []
        for signal, perf in signal_performance.items():
            if perf["total"] < 2:
                continue

            signal_win_rate = perf["correct"] / perf["total"]

            # 如果胜率高于整体，调整相关权重
            if signal_win_rate > win_rate + 0.1:
                # 表现好，增加权重
                if signal == "BUY":
                    adjustment = self.LEARNING_RATE * (signal_win_rate - win_rate)
                    updates.append(("technical", adjustment, "技术面 BUY信号胜率高"))
                elif signal == "SELL":
                    updates.append(("sentiment", adjustment, "情绪 SELL信号胜率高"))

            elif signal_win_rate < win_rate - 0.1:
                # 表现差，减少权重
                adjustment = -self.LEARNING_RATE * (win_rate - signal_win_rate)
                if signal == "BUY":
                    updates.append(("technical", adjustment, "技术面 BUY信号胜率低"))
                elif signal == "SELL":
                    updates.append(("sentiment", adjustment, "情绪 SELL信号胜率低"))

        # 应用更新
        for unit, delta, reason in updates:
            old_weight = self.weights.get(unit, 1.0)
            new_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, old_weight + delta))
            self.weights[unit] = new_weight

            update = WeightUpdate(
                timestamp=datetime.now().isoformat(),
                unit=unit,
                old_weight=old_weight,
                new_weight=new_weight,
                reason=reason
            )
            self.weight_history.append(update)

            # 如果有consensus_engine，同步更新
            if self.consensus_engine and hasattr(consensus_engine, 'adjust_weight'):
                self.consensus_engine.adjust_weight(unit, delta)

        result = {
            "skipped": False,
            "period_days": lookback_days,
            "total_decisions": total,
            "overall_win_rate": win_rate,
            "signal_performance": dict(signal_performance),
            "weight_updates": [
                {
                    "unit": u.unit,
                    "old": u.old_weight,
                    "new": u.new_weight,
                    "reason": u.reason
                }
                for u in self.weight_history[-5:]
            ],
            "current_weights": self.weights.copy()
        }

        logger.info(f"[BRAIN] 学习完成: win_rate={win_rate:.1%}, updates={len(updates)}")
        return result

    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.weights.copy()

    def reset_weights(self):
        """重置权重到默认值"""
        self.weights = self.DEFAULT_WEIGHTS.copy()
        logger.info("[BRAIN] 权重已重置为默认值")

    def generate_learning_report(self) -> str:
        """生成学习报告"""
        report_lines = [
            "=" * 60,
            "[BRAIN] 权重学习报告",
            "=" * 60,
            f"总调整次数: {len(self.weight_history)}",
            "",
            "当前权重:",
        ]

        for unit, weight in sorted(self.weights.items()):
            default = self.DEFAULT_WEIGHTS.get(unit, 1.0)
            change = weight - default
            if change > 0:
                indicator = f"↑{change:+.2f}"
            elif change < 0:
                indicator = f"↓{change:+.2f}"
            else:
                indicator = "="
            report_lines.append(f"  {unit:15}: {weight:.2f} ({indicator})")

        if self.weight_history:
            report_lines.append("")
            report_lines.append("最近5次调整:")
            for update in reversed(self.weight_history[-5:]):
                report_lines.append(
                    f"  [{update.timestamp[11:19]}] {update.unit}: "
                    f"{update.old_weight:.2f} → {update.new_weight:.2f} ({update.reason})"
                )

        report_lines.append("=" * 60)
        return "\n".join(report_lines)
