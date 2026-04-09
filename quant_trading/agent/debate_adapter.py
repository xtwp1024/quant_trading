#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debate Engine Adapter for ETH Long Runner.
Bull vs Bear 多轮辩论适配器

复用: agent/debate_engine.py 中的辩论逻辑
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger("DebateAdapter")


class Stance(str, Enum):
    """辩论立场"""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


@dataclass
class Claim:
    """辩论论点"""
    agent_id: str
    content: str
    evidence: List[str] = field(default_factory=list)
    stance: str = "neutral"
    confidence: float = 0.5


@dataclass
class DebateResult:
    """辩论结果"""
    timestamp: str
    bull_claims: List[Claim]
    bear_claims: List[Claim]
    bull_score: float
    bear_score: float
    winner: str  # BULL/BEAR/DRAW
    verdict: str
    confidence: float
    rounds: int


class DebateEngineAdapter:
    """
    辩论引擎适配器

    流程:
    1. 第1轮陈述 - Bull和Bear各提出Claims
    2. 第2-N轮反驳 - 交叉辩论
    3. 裁判打分 - 输出最终判决

    复用 debate_engine.py 的 Claim/DebateResult 数据结构
    """

    def __init__(self, llm_bridge=None, max_rounds: int = 3):
        """
        Args:
            llm_bridge: 可选的LLM桥接器
            max_rounds: 最大辩论轮数
        """
        self.llm_bridge = llm_bridge
        self.max_rounds = max_rounds

    async def debate(self, research_result: Dict[str, Any], indicators: Dict[str, Any]) -> DebateResult:
        """
        执行多轮辩论

        Args:
            research_result: 研究团队结果
            indicators: 技术指标

        Returns:
            DebateResult: 辩论结果
        """
        timestamp = datetime.now().isoformat()
        price = indicators.get("price", 0)

        logger.info(f"[DEBATE] 辩论开始: price={price}")

        # 第1轮: 初始陈述
        bull_claims, bear_claims = await self._round_initial_statements(research_result, indicators)

        # 第2-N轮: 反驳
        for round_num in range(2, self.max_rounds + 1):
            logger.info(f"[DEBATE] 辩论第 {round_num} 轮")
            new_bull_claims, new_bear_claims = await self._round_rebuttal(
                bull_claims, bear_claims, indicators, round_num
            )
            bull_claims.extend(new_bull_claims)
            bear_claims.extend(new_bear_claims)

        # 计算得分
        bull_score = sum(c.confidence for c in bull_claims) / max(len(bull_claims), 1)
        bear_score = sum(c.confidence for c in bear_claims) / max(len(bear_claims), 1)

        # 判定胜负
        if bull_score > bear_score + 0.15:
            winner = "BULL"
            verdict = f"多头占据优势，bull_score={bull_score:.2f} > bear_score={bear_score:.2f}"
        elif bear_score > bull_score + 0.15:
            winner = "BEAR"
            verdict = f"空头占据优势，bear_score={bear_score:.2f} > bull_score={bull_score:.2f}"
        else:
            winner = "DRAW"
            verdict = f"双方势均力敌，bull={bull_score:.2f} vs bear={bear_score:.2f}"

        confidence = min(abs(bull_score - bear_score) + 0.3, 1.0)

        result = DebateResult(
            timestamp=timestamp,
            bull_claims=bull_claims,
            bear_claims=bear_claims,
            bull_score=bull_score,
            bear_score=bear_score,
            winner=winner,
            verdict=verdict,
            confidence=confidence,
            rounds=self.max_rounds
        )

        logger.info(f"[DEBATE] 辩论结束: winner={winner}, verdict={verdict}")
        return result

    async def _round_initial_statements(
        self, research_result: Dict, indicators: Dict
    ) -> tuple[List[Claim], List[Claim]]:
        """第1轮初始陈述"""
        bull_claims = []
        bear_claims = []

        price = indicators.get("price", 0)
        rsi = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_hist", 0)
        trend = indicators.get("trend", "neutral")
        consensus = research_result.get("consensus", "NEUTRAL")

        # Bull初始论点
        if rsi < 40:
            bull_claims.append(Claim(
                agent_id="BullAgent",
                content="RSI处于超卖区域，存在反弹机会",
                evidence=[f"RSI={rsi:.1f}<40"],
                stance="bull",
                confidence=0.7
            ))

        if macd_hist > 0:
            bull_claims.append(Claim(
                agent_id="BullAgent",
                content="MACD柱状图为正，多头动能占优",
                evidence=[f"MACD Hist={macd_hist:.4f}>0"],
                stance="bull",
                confidence=0.6
            ))

        if trend in ["强势上涨", "短期上涨"]:
            bull_claims.append(Claim(
                agent_id="BullAgent",
                content=f"趋势{trend}，顺势做多",
                evidence=[f"趋势={trend}"],
                stance="bull",
                confidence=0.6
            ))

        if consensus == "BULL":
            bull_claims.append(Claim(
                agent_id="BullAgent",
                content="研究团队共识看涨",
                evidence=[f"共识={consensus}"],
                stance="bull",
                confidence=0.5
            ))

        # Bear初始论点
        if rsi > 60:
            bear_claims.append(Claim(
                agent_id="BearAgent",
                content="RSI处于超买区域，存在回调风险",
                evidence=[f"RSI={rsi:.1f}>60"],
                stance="bear",
                confidence=0.7
            ))

        if macd_hist < 0:
            bear_claims.append(Claim(
                agent_id="BearAgent",
                content="MACD柱状图为负，空头动能占优",
                evidence=[f"MACD Hist={macd_hist:.4f}<0"],
                stance="bear",
                confidence=0.6
            ))

        if trend in ["弱势下跌", "反弹迹象"]:
            bear_claims.append(Claim(
                agent_id="BearAgent",
                content=f"趋势{trend}，顺势做空",
                evidence=[f"趋势={trend}"],
                stance="bear",
                confidence=0.6
            ))

        if consensus == "BEAR":
            bear_claims.append(Claim(
                agent_id="BearAgent",
                content="研究团队共识看跌",
                evidence=[f"共识={consensus}"],
                stance="bear",
                confidence=0.5
            ))

        # 如果双方都没有论点，添加默认论点
        if not bull_claims:
            bull_claims.append(Claim(
                agent_id="BullAgent",
                content="技术面无明显做空信号，维持多头观点",
                evidence=["RSI中性"],
                stance="bull",
                confidence=0.3
            ))

        if not bear_claims:
            bear_claims.append(Claim(
                agent_id="BearAgent",
                content="技术面无明显做多信号，维持空头观点",
                evidence=["RSI中性"],
                stance="bear",
                confidence=0.3
            ))

        return bull_claims, bear_claims

    async def _round_rebuttal(
        self,
        bull_claims: List[Claim],
        bear_claims: List[Claim],
        indicators: Dict,
        round_num: int
    ) -> tuple[List[Claim], List[Claim]]:
        """反驳轮"""
        new_bull = []
        new_bear = []

        rsi = indicators.get("rsi", 50)
        price = indicators.get("price", 0)
        atr = indicators.get("atr", 0)

        # Bull反驳Bear的论点
        for bear in bear_claims[-3:]:  # 只反驳最近3个论点
            if "RSI超买" in bear.content and rsi < 50:
                new_bull.append(Claim(
                    agent_id="BullAgent",
                    content=f"反驳: RSI已从超买回落至{rsi:.1f}，风险已释放",
                    evidence=[f"RSI={rsi:.1f}"],
                    stance="bull",
                    confidence=0.5
                ))
            if "MACD空头" in bear.content and indicators.get("macd_hist", 0) > 0:
                new_bull.append(Claim(
                    agent_id="BullAgent",
                    content="反驳: MACD已转正，多头信号确认",
                    evidence=["MACD Hist > 0"],
                    stance="bull",
                    confidence=0.5
                ))

        # Bear反驳Bull的论点
        for bull in bull_claims[-3:]:
            if "RSI超卖" in bull.content and rsi > 50:
                new_bear.append(Claim(
                    agent_id="BearAgent",
                    content=f"反驳: RSI已从超卖反弹至{rsi:.1f}，反弹可能结束",
                    evidence=[f"RSI={rsi:.1f}"],
                    stance="bear",
                    confidence=0.5
                ))
            if "MACD多头" in bull.content and indicators.get("macd_hist", 0) < 0:
                new_bear.append(Claim(
                    agent_id="BearAgent",
                    content="反驳: MACD已转负，空头信号确认",
                    evidence=["MACD Hist < 0"],
                    stance="bear",
                    confidence=0.5
                ))

        # 添加基于波动率的反驳
        atr_ratio = indicators.get("atr_ratio", 1.0)
        if atr_ratio > 1.5:
            new_bear.append(Claim(
                agent_id="BearAgent",
                content=f"高波动环境(ATR比率={atr_ratio:.2f})，风险加大",
                evidence=[f"ATR比率={atr_ratio:.2f}>1.5"],
                stance="bear",
                confidence=0.4
            ))
        elif atr_ratio < 0.7:
            new_bull.append(Claim(
                agent_id="BullAgent",
                content=f"低波动环境(ATR比率={atr_ratio:.2f})，趋势可能延续",
                evidence=[f"ATR比率={atr_ratio:.2f}<0.7"],
                stance="bull",
                confidence=0.4
            ))

        return new_bull, new_bear

    def to_consensus_format(self, result: DebateResult) -> Dict[str, Any]:
        """将辩论结果转换为共识引擎格式"""
        # 计算辩论对共识的影响
        net_score = result.bull_score - result.bear_score
        direction = 0.0
        if result.winner == "BULL":
            direction = 1.0 * result.confidence
        elif result.winner == "BEAR":
            direction = -1.0 * result.confidence

        return {
            "bull_score": result.bull_score,
            "bear_score": result.bear_score,
            "winner": result.winner,
            "net_direction": direction,
            "debate_confidence": result.confidence,
            "verdict": result.verdict
        }
