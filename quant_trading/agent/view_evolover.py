#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
View Evolver for ETH Long Runner.
多空观点演化追踪

功能:
1. 记录每个周期的观点
2. 追踪观点变化趋势
3. 可视化观点演化曲线
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("ViewEvolver")


@dataclass
class ViewSnapshot:
    """观点快照"""
    timestamp: str
    view_type: str  # bull/bear/neutral
    confidence: float  # 置信度
    price: float
    bull_score: float
    bear_score: float
    consensus_score: float  # 综合评分


class ViewEvolver:
    """
    观点演化追踪器

    记录并分析观点的变化趋势:
    - 观点强度变化
    - 多空转换点
    - 观点一致性
    """

    def __init__(self, memory_bank):
        """
        Args:
            memory_bank: MemoryBank实例
        """
        self.memory_bank = memory_bank
        self.current_view: Optional[ViewSnapshot] = None
        self.view_history: List[ViewSnapshot] = []

    def record_view(
        self,
        view_type: str,
        confidence: float,
        price: float,
        bull_score: float,
        bear_score: float,
        consensus_score: float
    ):
        """
        记录当前观点

        Args:
            view_type: BULL/BEAR/NEUTRAL
            confidence: 置信度
            price: 当前价格
            bull_score: 多头得分
            bear_score: 空头得分
            consensus_score: 综合评分
        """
        snapshot = ViewSnapshot(
            timestamp=datetime.now().isoformat(),
            view_type=view_type,
            confidence=confidence,
            price=price,
            bull_score=bull_score,
            bear_score=bear_score,
            consensus_score=consensus_score
        )

        self.current_view = snapshot
        self.view_history.append(snapshot)

        # 保存到MemoryBank
        self.memory_bank.save_view(
            view_type=view_type.lower(),
            confidence=confidence,
            price=price,
            indicators={
                "bull_score": bull_score,
                "bear_score": bear_score,
                "consensus_score": consensus_score
            }
        )

        logger.info(f"[STATS] 观点记录: {view_type} {confidence:.2f} @ ${price:.2f}")

    def detect_view_change(self) -> Optional[Dict[str, Any]]:
        """
        检测观点变化

        Returns:
            变化信息字典，无变化返回None
        """
        if len(self.view_history) < 2:
            return None

        prev = self.view_history[-2]
        curr = self.view_history[-1]

        if prev.view_type != curr.view_type:
            return {
                "changed": True,
                "from": prev.view_type,
                "to": curr.view_type,
                "timestamp": curr.timestamp,
                "price_change": (curr.price - prev.price) / prev.price * 100,
                "confidence_change": curr.confidence - prev.confidence
            }

        return None

    def get_view_trend(self, lookback_count: int = 10) -> str:
        """
        获取观点趋势

        Args:
            lookback_count: 回看数量

        Returns:
            趋势描述: " strengthening", " weakening", " stable"
        """
        if len(self.view_history) < 3:
            return "stable"

        recent = self.view_history[-lookback_count:]
        if len(recent) < 3:
            return "stable"

        # 计算平均置信度变化
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]

        avg_conf_first = sum(v.confidence for v in first_half) / len(first_half)
        avg_conf_second = sum(v.confidence for v in second_half) / len(second_half)

        conf_change = avg_conf_second - avg_conf_first

        if conf_change > 0.1:
            return "strengthening"
        elif conf_change < -0.1:
            return "weakening"
        else:
            return "stable"

    def get_bull_bear_ratio(self, lookback_hours: int = 24) -> float:
        """
        计算多空比

        Args:
            lookback_hours: 回看小时数

        Returns:
            多空比 (bull/bear)
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        recent_views = [
            v for v in self.view_history
            if datetime.fromisoformat(v.timestamp) > cutoff
        ]

        if not recent_views:
            return 1.0

        bull_count = sum(1 for v in recent_views if v.view_type == "BULL")
        bear_count = sum(1 for v in recent_views if v.view_type == "BEAR")

        if bear_count == 0:
            return float(bull_count) if bull_count > 0 else 1.0

        return bull_count / bear_count

    def generate_view_report(self) -> str:
        """
        生成观点演化报告

        Returns:
            格式化的报告字符串
        """
        if not self.view_history:
            return "[STATS] 观点演化报告: 暂无数据"

        recent = self.view_history[-20:]  # 最近20条

        report_lines = [
            "=" * 60,
            "[STATS] 观点演化报告",
            "=" * 60,
            f"总记录数: {len(self.view_history)}",
            f"当前观点: {self.current_view.view_type if self.current_view else 'N/A'}",
            f"当前置信度: {self.current_view.confidence:.2f}" if self.current_view else "",
            f"多空比(24h): {self.get_bull_bear_ratio(24):.2f}",
            f"趋势: {self.get_view_trend()}",
            "-" * 60,
            "最近观点记录:",
        ]

        for v in reversed(recent):
            indicator = "[GREEN]" if v.view_type == "BULL" else "[RED]" if v.view_type == "BEAR" else "[NEUTRAL]"
            report_lines.append(
                f"  {indicator} [{v.timestamp[11:19]}] {v.view_type:6} "
                f"conf={v.confidence:.2f} price=${v.price:.2f}"
            )

        # 观点变化检测
        change = self.detect_view_change()
        if change:
            report_lines.append("-" * 60)
            report_lines.append(f"[WARN] 观点变化: {change['from']} → {change['to']}")
            report_lines.append(f"   价格变化: {change['price_change']:+.2f}%")

        report_lines.append("=" * 60)
        return "\n".join(report_lines)

    def get_view_evolution(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        获取观点演化数据 (用于报告)

        Args:
            days: 回看天数

        Returns:
            观点演化数据列表
        """
        cutoff = datetime.now() - timedelta(days=days)
        return [
            {
                "timestamp": v.timestamp,
                "view_type": v.view_type,
                "confidence": v.confidence,
                "price": v.price,
                "bull_score": v.bull_score,
                "bear_score": v.bear_score
            }
            for v in self.view_history
            if datetime.fromisoformat(v.timestamp) > cutoff
        ]

    def export_for_charting(self, lookback_hours: int = 168) -> List[Dict[str, Any]]:
        """
        导出数据用于图表绘制

        Args:
            lookback_hours: 回看小时数 (默认168=7天)

        Returns:
            用于图表的数据列表
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        return [
            {
                "timestamp": v.timestamp,
                "price": v.price,
                "view_type": v.view_type,
                "confidence": v.confidence,
                "bull_score": v.bull_score,
                "bear_score": v.bear_score,
                "consensus_score": v.consensus_score
            }
            for v in self.view_history
            if datetime.fromisoformat(v.timestamp) > cutoff
        ]
