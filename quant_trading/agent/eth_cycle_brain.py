#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETH Cycle Brain - Core Cognitive Loop for ETH Long Runner.
核心认知循环编排器

流程: 感知 → 研究 → 辩论 → 决策 → 复盘
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .technical_perception import TechnicalPerception, TechnicalIndicators
from .research_adapter import ResearchTeamAdapter, ResearchResult
from .debate_adapter import DebateEngineAdapter, DebateResult

logger = logging.getLogger("EthCycleBrain")


@dataclass
class CycleResult:
    """认知循环结果"""
    timestamp: str
    price: float
    indicators: TechnicalIndicators
    research: Optional[ResearchResult]
    debate: Optional[DebateResult]
    consensus_score: float
    signal: str  # BUY/SELL/HOLD
    signal_strength: float
    risk_level: str
    target_zone: tuple
    stop_loss: float


class EthCycleBrain:
    """
    ETH 认知循环大脑

    编排完整的认知流程:
    1. 感知(Perception): 技术指标计算
    2. 研究(Research): 6路并行分析师
    3. 辩论(Debate): Bull vs Bear 多轮对抗
    4. 决策(Consensus): 五力共识评分
    5. 复盘(Review): 更新记忆 (由ReviewEngine处理)
    """

    def __init__(
        self,
        consensus_engine=None,
        llm_bridge=None,
        memory_bank=None
    ):
        """
        Args:
            consensus_engine: 共识引擎 (five_force.consensus.ConsensusEngine)
            llm_bridge: LLM桥接器
            memory_bank: 记忆银行
        """
        self.perception = TechnicalPerception()
        self.research = ResearchTeamAdapter(llm_bridge)
        self.debate = DebateEngineAdapter(llm_bridge)
        self.consensus_engine = consensus_engine
        self.memory_bank = memory_bank

        # 权重 (用于共识融合)
        self.weights = {
            "technical": 1.5,
            "research": 1.2,
            "debate": 1.0,
        }

    async def run_cycle(self, ohlcv: Any, market_data: Dict[str, Any]) -> CycleResult:
        """
        运行完整认知循环

        Args:
            ohlcv: OHLCV数据 (numpy array)
            market_data: 市场数据字典

        Returns:
            CycleResult: 循环结果
        """
        timestamp = datetime.now().isoformat()
        price = market_data.get("price", ohlcv[-1][3] if len(ohlcv) > 0 else 0)

        logger.info(f"[BRAIN] 认知循环开始: {timestamp}")

        # ===================== 1. 感知 =====================
        indicators = self.perception.compute(ohlcv)
        if indicators is None:
            logger.error("技术指标计算失败")
            return None

        logger.info(f"[STATS] 感知完成: RSI={indicators.rsi:.1f}, trend={indicators.trend}")

        # ===================== 2. 研究 =====================
        research = await self.research.analyze(
            indicators={
                "rsi": indicators.rsi,
                "macd_hist": indicators.macd_hist,
                "ema12": indicators.ema12,
                "ema26": indicators.ema26,
                "trend": indicators.trend,
                "atr": indicators.atr,
                "atr_ratio": indicators.atr_ratio,
                "vol_state": indicators.vol_state,
            },
            market_data={
                "price": price,
                "volume": ohlcv[-1][4] if len(ohlcv) > 0 else 0
            }
        )

        logger.info(f"[RESEARCH] 研究完成: consensus={research.consensus}")

        # ===================== 3. 辩论 =====================
        debate = await self.debate.debate(
            research_result=self.research.summarize_for_debate(research),
            indicators={
                "price": price,
                "rsi": indicators.rsi,
                "macd_hist": indicators.macd_hist,
                "trend": indicators.trend,
                "atr_ratio": indicators.atr_ratio,
            }
        )

        logger.info(f"[DEBATE] 辩论完成: winner={debate.winner}")

        # ===================== 4. 决策 =====================
        consensus_score = self._fuse_consensus(indicators, research, debate)

        # 信号生成
        signals = self.perception.generate_signals(indicators)
        signal, strength = self.perception.signal_to_direction(signals)

        # 如果辩论结果与信号不符，根据辩论调整
        if debate.winner == "BULL" and signal == "SELL":
            strength *= 0.7  # 削弱信号
        elif debate.winner == "BEAR" and signal == "BUY":
            strength *= 0.7

        # 风险等级
        risk_level = self._calculate_risk(indicators, consensus_score)

        # 目标区间和止损
        target_zone, stop_loss = self._calculate_target_and_stop(price, indicators, consensus_score)

        result = CycleResult(
            timestamp=timestamp,
            price=price,
            indicators=indicators,
            research=research,
            debate=debate,
            consensus_score=consensus_score,
            signal=signal,
            signal_strength=strength,
            risk_level=risk_level,
            target_zone=target_zone,
            stop_loss=stop_loss
        )

        logger.info(
            f"[BRAIN] 认知循环完成: signal={signal}, strength={strength:.2f}, "
            f"score={consensus_score:.3f}, risk={risk_level}"
        )

        return result

    def _fuse_consensus(
        self,
        indicators: TechnicalIndicators,
        research: ResearchResult,
        debate: DebateResult
    ) -> float:
        """
        融合多源信息计算共识评分

        Returns:
            -1.0 (强空头) to 1.0 (强多头)
        """
        # 技术面贡献
        tech_score = 0.0
        if indicators.rsi < 30:
            tech_score += 0.3
        elif indicators.rsi > 70:
            tech_score -= 0.3
        elif indicators.rsi < 45:
            tech_score += 0.1
        elif indicators.rsi > 55:
            tech_score -= 0.1

        if indicators.macd_hist > 0:
            tech_score += 0.2
        else:
            tech_score -= 0.2

        if indicators.ema12 > indicators.ema26:
            tech_score += 0.2
        else:
            tech_score -= 0.2

        # 研究团队贡献
        research_score = research.bull_score - research.bear_score

        # 辩论贡献
        debate_score = debate.bull_score - debate.bear_score
        debate_score *= debate.confidence

        # 加权融合
        total_score = (
            tech_score * self.weights["technical"] +
            research_score * self.weights["research"] +
            debate_score * self.weights["debate"]
        )

        total_weight = sum(self.weights.values())
        consensus = total_score / total_weight

        # 归一化到 [-1, 1]
        consensus = max(-1.0, min(1.0, consensus))

        return consensus

    def _calculate_risk(self, indicators: TechnicalIndicators, consensus_score: float) -> str:
        """计算风险等级"""
        # 高波动
        if indicators.atr_ratio > 1.5:
            return "HIGH"

        # RSI极端值
        if indicators.rsi > 80 or indicators.rsi < 20:
            return "HIGH"

        # 共识评分极端
        if abs(consensus_score) > 0.7:
            return "MEDIUM"

        return "LOW"

    def _calculate_target_and_stop(
        self,
        price: float,
        indicators: TechnicalIndicators,
        consensus_score: float
    ) -> tuple[tuple[float, float], float]:
        """
        计算目标区间和止损

        Returns:
            (target_lower, target_upper), stop_loss
        """
        atr = indicators.atr
        support = indicators.support_levels[-1] if indicators.support_levels else price * 0.95
        resistance = indicators.resistance_levels[0] if indicators.resistance_levels else price * 1.05

        if consensus_score > 0:  # 多头
            target_upper = resistance
            target_lower = price + atr * 0.5
            stop_loss = price - atr * 2
        else:  # 空头
            target_upper = price - atr * 0.5
            target_lower = support
            stop_loss = price + atr * 2

        # 确保止损在合理范围
        if consensus_score > 0:
            if stop_loss < support:
                stop_loss = support * 0.98
        else:
            if stop_loss > resistance:
                stop_loss = resistance * 1.02

        return (target_lower, target_upper), stop_loss

    def adjust_weight(self, unit: str, delta: float):
        """调整权重"""
        if unit in self.weights:
            self.weights[unit] = max(0.1, min(3.0, self.weights[unit] + delta))
            logger.info(f"[WEIGHT] 权重调整: {unit} {self.weights[unit]-delta:.2f} → {self.weights[unit]:.2f}")
