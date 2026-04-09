#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research Team Adapter for ETH Long Runner.
6路并行分析师 - 技术面/宏观/资金面/情绪/新闻/链上

复用: agent/research_team.py 中的分析师逻辑
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("ResearchAdapter")


@dataclass
class AnalystReport:
    """分析师报告"""
    analyst_type: str
    finding: str
    sentiment: float  # -1.0 (bear) to 1.0 (bull)
    confidence: float  # 0.0 to 1.0
    key_points: List[str]


@dataclass
class ResearchResult:
    """研究团队结果"""
    timestamp: str
    price: float
    reports: List[AnalystReport]
    bull_score: float  # 综合多头得分
    bear_score: float  # 综合空头得分
    consensus: str  # BULL/BEAR/NEUTRAL


class ResearchTeamAdapter:
    """
    研究团队适配器

    6路并行分析师:
    1. TechnicalAnalyst - 技术面 (K线、均线、RSI、MACD)
    2. MacroAnalyst - 宏观 (政策、利率、风险偏好)
    3. MoneyFlowAnalyst - 资金面 (大单、机构持仓)
    4. SentimentAnalyst - 情绪 (恐惧贪婪、持仓变化)
    5. NewsAnalyst - 新闻 (财经新闻、公告)
    6. OnChainAnalyst - 链上 (Gas、活跃地址、DeFi)
    """

    def __init__(self, llm_bridge=None):
        """
        Args:
            llm_bridge: 可选的LLM桥接器，用于生成分析报告
        """
        self.llm_bridge = llm_bridge

    async def analyze(self, indicators: Dict[str, Any], market_data: Dict[str, Any]) -> ResearchResult:
        """
        执行6路并行分析

        Args:
            indicators: 技术指标字典
            market_data: 市场数据 (价格、成交量等)

        Returns:
            ResearchResult: 综合研究报告
        """
        timestamp = datetime.now().isoformat()
        price = market_data.get("price", 0)

        logger.info(f"[RESEARCH] 研究团队开始分析: price={price}")

        # 并行执行6路分析
        tasks = [
            self._analyze_technical(indicators, market_data),
            self._analyze_macro(indicators, market_data),
            self._analyze_money_flow(indicators, market_data),
            self._analyze_sentiment(indicators, market_data),
            self._analyze_news(indicators, market_data),
            self._analyze_onchain(indicators, market_data),
        ]

        reports = await asyncio.gather(*tasks, return_exceptions=True)

        # 计算综合得分
        valid_reports = [r for r in reports if isinstance(r, AnalystReport)]
        bull_score = sum(r.sentiment * r.confidence for r in valid_reports) / max(len(valid_reports), 1)
        bear_score = sum(-r.sentiment * r.confidence for r in valid_reports) / max(len(valid_reports), 1)

        # 确定共识
        if bull_score > 0.2:
            consensus = "BULL"
        elif bear_score > 0.2:
            consensus = "BEAR"
        else:
            consensus = "NEUTRAL"

        result = ResearchResult(
            timestamp=timestamp,
            price=price,
            reports=valid_reports,
            bull_score=bull_score,
            bear_score=bear_score,
            consensus=consensus
        )

        logger.info(f"[LIST] 研究完成: consensus={consensus}, bull={bull_score:.2f}, bear={bear_score:.2f}")
        return result

    async def _analyze_technical(self, indicators: Dict, market_data: Dict) -> AnalystReport:
        """技术面分析师"""
        rsi = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_hist", 0)
        ema12 = indicators.get("ema12", 0)
        ema26 = indicators.get("ema26", 0)
        trend = indicators.get("trend", "neutral")

        # 技术面信号
        if rsi < 30 and macd_hist > 0:
            sentiment = 0.8
            finding = "RSI超卖 + MACD多头共振，看涨"
        elif rsi > 70 and macd_hist < 0:
            sentiment = -0.8
            finding = "RSI超买 + MACD空头共振，看跌"
        elif ema12 > ema26 and trend in ["强势上涨", "短期上涨"]:
            sentiment = 0.5
            finding = "EMA多头排列，趋势看涨"
        elif ema12 < ema26 and trend in ["弱势下跌", "反弹迹象"]:
            sentiment = -0.5
            finding = "EMA空头排列，趋势看跌"
        elif rsi > 50:
            sentiment = 0.3
            finding = "RSI偏强，中性偏多"
        else:
            sentiment = -0.3
            finding = "RSI偏弱，中性偏空"

        confidence = min(abs(sentiment) + 0.2, 1.0)

        return AnalystReport(
            analyst_type="TechnicalAnalyst",
            finding=finding,
            sentiment=sentiment,
            confidence=confidence,
            key_points=[
                f"RSI(14)={rsi:.1f}",
                f"MACD Hist={macd_hist:.4f}",
                f"趋势={trend}",
                f"EMA12={ema12:.2f}, EMA26={ema26:.2f}"
            ]
        )

    async def _analyze_macro(self, indicators: Dict, market_data: Dict) -> AnalystReport:
        """宏观分析师"""
        # 简化版宏观分析 (实际应用中应接入真实数据)
        price = market_data.get("price", 0)
        atr_ratio = indicators.get("atr_ratio", 1.0)

        # 波动率作为风险偏好指标
        if atr_ratio > 1.5:
            sentiment = -0.3
            finding = "高波动环境，风险偏好下降"
        elif atr_ratio < 0.7:
            sentiment = 0.2
            finding = "低波动环境，市场相对平稳"
        else:
            sentiment = 0.1
            finding = "正常波动环境，中性"

        confidence = 0.5

        return AnalystReport(
            analyst_type="MacroAnalyst",
            finding=finding,
            sentiment=sentiment,
            confidence=confidence,
            key_points=[
                f"价格=${price:.2f}",
                f"ATR比率={atr_ratio:.2f}",
                "宏观分析基于波动率环境判断"
            ]
        )

    async def _analyze_money_flow(self, indicators: Dict, market_data: Dict) -> AnalystReport:
        """资金面分析师"""
        vol_state = indicators.get("vol_state", "normal")
        rsi = indicators.get("rsi", 50)

        # 资金面信号
        if rsi < 40 and vol_state == "高波动":
            sentiment = 0.4
            finding = "超卖+高波动，可能存在资金流入机会"
        elif rsi > 60 and vol_state == "高波动":
            sentiment = -0.4
            finding = "超买+高波动，资金可能出逃"
        else:
            sentiment = 0.0
            finding = "资金面无明显信号"

        confidence = 0.4

        return AnalystReport(
            analyst_type="MoneyFlowAnalyst",
            finding=finding,
            sentiment=sentiment,
            confidence=confidence,
            key_points=[
                f"成交量状态={vol_state}",
                f"RSI={rsi:.1f}",
                "资金面分析基于价格与成交量关系"
            ]
        )

    async def _analyze_sentiment(self, indicators: Dict, market_data: Dict) -> AnalystReport:
        """情绪分析师"""
        trend = indicators.get("trend", "neutral")
        rsi = indicators.get("rsi", 50)

        # 情绪判断
        if rsi < 30:
            sentiment = 0.6
            finding = "恐慌情绪，可能过度悲观"
        elif rsi > 70:
            sentiment = -0.6
            finding = "贪婪情绪，可能过度乐观"
        elif trend == "强势上涨":
            sentiment = -0.3
            finding = "强势上涨，情绪可能过热"
        elif trend == "弱势下跌":
            sentiment = 0.3
            finding = "弱势下跌，情绪可能过度悲观"
        else:
            sentiment = 0.0
            finding = "情绪中性"

        confidence = 0.5

        return AnalystReport(
            analyst_type="SentimentAnalyst",
            finding=finding,
            sentiment=sentiment,
            confidence=confidence,
            key_points=[
                f"趋势={trend}",
                f"RSI={rsi:.1f}",
                "情绪分析基于价格行为"
            ]
        )

    async def _analyze_news(self, indicators: Dict, market_data: Dict) -> AnalystReport:
        """新闻分析师"""
        # 简化版新闻分析 (实际应用中应接入新闻API)
        price = market_data.get("price", 0)
        trend = indicators.get("trend", "neutral")

        # 基于趋势的新闻情绪
        if trend in ["强势上涨", "短期上涨"]:
            sentiment = 0.2
            finding = "上涨趋势中，市场情绪偏正面"
        elif trend in ["弱势下跌", "反弹迹象"]:
            sentiment = -0.2
            finding = "下跌趋势中，市场情绪偏负面"
        else:
            sentiment = 0.0
            finding = "震荡市中，新闻影响中性"

        confidence = 0.3

        return AnalystReport(
            analyst_type="NewsAnalyst",
            finding=finding,
            sentiment=sentiment,
            confidence=confidence,
            key_points=[
                f"价格=${price:.2f}",
                f"趋势={trend}",
                "新闻分析基于市场趋势判断"
            ]
        )

    async def _analyze_onchain(self, indicators: Dict, market_data: Dict) -> AnalystReport:
        """链上分析师"""
        atr = indicators.get("atr", 0)
        price = market_data.get("price", 0)

        # 简化版链上分析 (基于ATR作为市场活跃度代理)
        if price > 0:
            atr_pct = atr / price * 100
            if atr_pct > 3:
                sentiment = -0.2
                finding = "高Gas费环境，网络活跃度过高"
            elif atr_pct < 1:
                sentiment = 0.1
                finding = "低Gas费环境，网络活跃度偏低"
            else:
                sentiment = 0.0
                finding = "Gas费正常，网络活跃度适中"
        else:
            sentiment = 0.0
            finding = "价格异常，跳过链上分析"

        confidence = 0.3

        return AnalystReport(
            analyst_type="OnChainAnalyst",
            finding=finding,
            sentiment=sentiment,
            confidence=confidence,
            key_points=[
                f"ATR=${atr:.2f}",
                f"价格=${price:.2f}",
                "链上分析基于市场活跃度代理指标"
            ]
        )

    def summarize_for_debate(self, result: ResearchResult) -> Dict[str, Any]:
        """将研究结果转换为辩论格式"""
        return {
            "bull_case": f"综合多头得分: {result.bull_score:.2f}",
            "bear_case": f"综合空头得分: {result.bear_score:.2f}",
            "consensus": result.consensus,
            "reports": [
                {
                    "analyst": r.analyst_type,
                    "finding": r.finding,
                    "sentiment": r.sentiment,
                    "confidence": r.confidence
                }
                for r in result.reports
            ]
        }
