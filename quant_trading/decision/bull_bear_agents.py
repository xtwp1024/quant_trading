"""
Bull/Bear Adversarial Agents
=============================

Generates bullish and bearish perspectives for adversarial debate.
These agents analyze market conditions from opposing viewpoints.

Based on LLM-TradeBot's StrategyEngine.get_bull_perspective() and get_bear_perspective()
but adapted to work without LLM API calls (rule-based fallback).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class Stance(Enum):
    """Agent stance types"""
    STRONGLY_BULLISH = "STRONGLY_BULLISH"
    SLIGHTLY_BULLISH = "SLIGHTLY_BULLISH"
    NEUTRAL = "NEUTRAL"
    SLIGHTLY_BEARISH = "SLIGHTLY_BEARISH"
    STRONGLY_BEARISH = "STRONGLY_BEARISH"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class PerspectiveResult:
    """Result from Bull/Bear analysis"""
    stance: str
    reasons: List[str]
    confidence: float  # 0-100
    raw_output: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stance': self.stance,
            'reasons': self.reasons,
            'confidence': self.confidence,
            'raw_output': self.raw_output,
        }


@dataclass
class BullResult:
    """Bull agent result"""
    stance: str  # STRONGLY_BULLISH, SLIGHTLY_BULLISH, NEUTRAL, UNCERTAIN
    bullish_reasons: str  # Semicolon-separated reasons
    bull_confidence: float  # 0-100
    raw_response: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stance': self.stance,
            'bullish_reasons': self.bullish_reasons,
            'bull_confidence': self.bull_confidence,
            'raw_response': self.raw_response,
        }


@dataclass
class BearResult:
    """Bear agent result"""
    stance: str  # STRONGLY_BEARISH, SLIGHTLY_BEARISH, NEUTRAL, UNCERTAIN
    bearish_reasons: str  # Semicolon-separated reasons
    bear_confidence: float  # 0-100
    raw_response: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stance': self.stance,
            'bearish_reasons': self.bearish_reasons,
            'bear_confidence': self.bear_confidence,
            'raw_response': self.raw_response,
        }


class BaseAdversarialAgent:
    """Base class for adversarial agents"""

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the agent.

        Args:
            llm_client: Optional LLM client for AI-powered analysis.
                       If None, falls back to rule-based analysis.
        """
        self.llm_client = llm_client

    def analyze(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market from this agent's perspective.

        Args:
            market_context: Dict containing market data, indicators, etc.

        Returns:
            Dict with stance, reasons, and confidence
        """
        if self.llm_client is not None:
            return self._analyze_with_llm(market_context)
        else:
            return self._analyze_rule_based(market_context)

    def _analyze_with_llm(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM for analysis (requires API key)"""
        raise NotImplementedError("Subclasses must implement _analyze_with_llm")

    def _analyze_rule_based(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based analysis fallback"""
        raise NotImplementedError("Subclasses must implement _analyze_rule_based")


class BullAgent(BaseAdversarialAgent):
    """
    Bull Agent - The Optimist

    Analyzes market conditions to find bullish/bullish signals and reasons
    why the market could go up.
    """

    def analyze(self, market_context: Dict[str, Any]) -> BullResult:
        """
        Generate bullish perspective.

        Args:
            market_context: Market data including:
                - quant_analysis: Technical analysis results
                - four_layer_result: Four-layer filter results
                - trend_scores: Multi-period trend scores
                - oscillator_scores: RSI, KDJ etc.
                - sentiment: Market sentiment data

        Returns:
            BullResult with bullish stance and reasons
        """
        if self.llm_client is not None:
            return self._analyze_with_llm(market_context)
        else:
            return self._analyze_rule_based(market_context)

    def _analyze_rule_based(self, market_context: Dict[str, Any]) -> BullResult:
        """Rule-based bullish analysis"""
        bullish_reasons = []
        confidence = 50.0  # Base confidence
        stance = Stance.NEUTRAL

        quant = market_context.get('quant_analysis', {})
        four_layer = market_context.get('four_layer_result', {})
        trend_scores = market_context.get('trend_scores', {})
        osc_scores = market_context.get('oscillator_scores', {})
        sentiment = market_context.get('sentiment', {})

        # Check trend alignment
        trend_1h = trend_scores.get('trend_1h', 0)
        trend_15m = trend_scores.get('trend_15m', 0)
        trend_5m = trend_scores.get('trend_5m', 0)

        if trend_1h > 25:
            bullish_reasons.append(f"1h trend bullish (score: {trend_1h:.0f})")
            confidence += 10

        if trend_15m > 18:
            bullish_reasons.append(f"15m trend bullish (score: {trend_15m:.0f})")
            confidence += 5

        if trend_1h > 0 and trend_15m > 0 and trend_5m > 0:
            bullish_reasons.append("Multi-timeframe bullish alignment")
            confidence += 10

        # Check oscillators
        rsi = osc_scores.get('rsi_1h', 50)
        if rsi < 40:
            bullish_reasons.append(f"RSI oversold on 1h ({rsi:.1f})")
            confidence += 10
        elif rsi < 50:
            bullish_reasons.append(f"RSI neutral-low on 1h ({rsi:.1f})")
            confidence += 5

        # Check sentiment
        sent_score = sentiment.get('total_sentiment_score', 0)
        if sent_score > 30:
            bullish_reasons.append(f"Positive sentiment (score: {sent_score:.0f})")
            confidence += 10

        # Check four-layer result
        if four_layer.get('layer1_pass') and four_layer.get('trend_1h') == 'long':
            bullish_reasons.append("Four-layer trend fuel confirmed")
            confidence += 10

        if four_layer.get('layer3_pass'):
            bullish_reasons.append("15m setup zonebullish")
            confidence += 5

        if four_layer.get('layer4_pass'):
            bullish_reasons.append("5m trigger confirmed")
            confidence += 5

        # Check position (low position is bullish for long)
        position = market_context.get('position', {})
        position_pct = position.get('position_pct', 50)
        if position_pct < 30:
            bullish_reasons.append(f"Price at low position ({position_pct:.1f}%)")
            confidence += 10

        # Determine final stance
        if confidence >= 70:
            stance = Stance.STRONGLY_BULLISH
        elif confidence >= 55:
            stance = Stance.SLIGHTLY_BULLISH
        elif confidence >= 45:
            stance = Stance.NEUTRAL
        else:
            stance = Stance.UNCERTAIN

        # Cap confidence at 95
        confidence = min(confidence, 95)

        return BullResult(
            stance=stance.value,
            bullish_reasons="; ".join(bullish_reasons) if bullish_reasons else "No strong bullish signals",
            bull_confidence=confidence,
            raw_response={'rule_based': True, 'reasons': bullish_reasons},
        )

    def _analyze_with_llm(self, market_context: Dict[str, Any]) -> BullResult:
        """Use LLM for bullish analysis"""
        if self.llm_client is None:
            return self._analyze_rule_based(market_context)

        prompt = """You are a BULLISH market analyst. Your job is to find reasons WHY the market could go UP.

Analyze the provided market data and identify bullish signals.
Output your analysis in this EXACT JSON format:
{
  "stance": "STRONGLY_BULLISH",
  "bullish_reasons": "Your 3-5 key bullish observations, separated by semicolons",
  "bull_confidence": 75
}

stance must be one of: STRONGLY_BULLISH, SLIGHTLY_BULLISH, NEUTRAL, UNCERTAIN
bull_confidence should be 0-100 based on how strong the bullish case is.
Focus ONLY on bullish factors. Ignore bearish signals."""

        try:
            response = self.llm_client.chat(
                system_prompt=prompt,
                user_prompt=str(market_context),
                temperature=0.3,
                max_tokens=500,
            )
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON
            import json
            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                data = json.loads(match.group())
                return BullResult(
                    stance=data.get('stance', 'UNCERTAIN'),
                    bullish_reasons=data.get('bullish_reasons', ''),
                    bull_confidence=float(data.get('bull_confidence', 50)),
                    raw_response=data,
                )
        except Exception:
            pass

        # Fallback to rule-based
        return self._analyze_rule_based(market_context)


class BearAgent(BaseAdversarialAgent):
    """
    Bear Agent - The Pessimist

    Analyzes market conditions to find bearish signals and reasons
    why the market could go down.
    """

    def analyze(self, market_context: Dict[str, Any]) -> BearResult:
        """
        Generate bearish perspective.

        Args:
            market_context: Market data including:
                - quant_analysis: Technical analysis results
                - four_layer_result: Four-layer filter results
                - trend_scores: Multi-period trend scores
                - oscillator_scores: RSI, KDJ etc.
                - sentiment: Market sentiment data

        Returns:
            BearResult with bearish stance and reasons
        """
        if self.llm_client is not None:
            return self._analyze_with_llm(market_context)
        else:
            return self._analyze_rule_based(market_context)

    def _analyze_rule_based(self, market_context: Dict[str, Any]) -> BearResult:
        """Rule-based bearish analysis"""
        bearish_reasons = []
        confidence = 50.0  # Base confidence
        stance = Stance.NEUTRAL

        quant = market_context.get('quant_analysis', {})
        four_layer = market_context.get('four_layer_result', {})
        trend_scores = market_context.get('trend_scores', {})
        osc_scores = market_context.get('oscillator_scores', {})
        sentiment = market_context.get('sentiment', {})

        # Check trend alignment
        trend_1h = trend_scores.get('trend_1h', 0)
        trend_15m = trend_scores.get('trend_15m', 0)
        trend_5m = trend_scores.get('trend_5m', 0)

        if trend_1h < -25:
            bearish_reasons.append(f"1h trend bearish (score: {trend_1h:.0f})")
            confidence += 10

        if trend_15m < -18:
            bearish_reasons.append(f"15m trend bearish (score: {trend_15m:.0f})")
            confidence += 5

        if trend_1h < 0 and trend_15m < 0 and trend_5m < 0:
            bearish_reasons.append("Multi-timeframe bearish alignment")
            confidence += 10

        # Check oscillators
        rsi = osc_scores.get('rsi_1h', 50)
        if rsi > 60:
            bearish_reasons.append(f"RSI overbought on 1h ({rsi:.1f})")
            confidence += 10
        elif rsi > 50:
            bearish_reasons.append(f"RSI neutral-high on 1h ({rsi:.1f})")
            confidence += 5

        # Check sentiment
        sent_score = sentiment.get('total_sentiment_score', 0)
        if sent_score < -30:
            bearish_reasons.append(f"Negative sentiment (score: {sent_score:.0f})")
            confidence += 10

        # Check four-layer result
        if four_layer.get('layer1_pass') and four_layer.get('trend_1h') == 'short':
            bearish_reasons.append("Four-layer trend fuel confirmed")
            confidence += 10

        if four_layer.get('layer3_pass'):
            bearish_reasons.append("15m setup zone bearish")
            confidence += 5

        if four_layer.get('layer4_pass'):
            bearish_reasons.append("5m trigger confirmed")
            confidence += 5

        # Check position (high position is bearish for short)
        position = market_context.get('position', {})
        position_pct = position.get('position_pct', 50)
        if position_pct > 70:
            bearish_reasons.append(f"Price at high position ({position_pct:.1f}%)")
            confidence += 10

        # Determine final stance
        if confidence >= 70:
            stance = Stance.STRONGLY_BEARISH
        elif confidence >= 55:
            stance = Stance.SLIGHTLY_BEARISH
        elif confidence >= 45:
            stance = Stance.NEUTRAL
        else:
            stance = Stance.UNCERTAIN

        # Cap confidence at 95
        confidence = min(confidence, 95)

        return BearResult(
            stance=stance.value,
            bearish_reasons="; ".join(bearish_reasons) if bearish_reasons else "No strong bearish signals",
            bear_confidence=confidence,
            raw_response={'rule_based': True, 'reasons': bearish_reasons},
        )

    def _analyze_with_llm(self, market_context: Dict[str, Any]) -> BearResult:
        """Use LLM for bearish analysis"""
        if self.llm_client is None:
            return self._analyze_rule_based(market_context)

        prompt = """You are a BEARISH market analyst. Your job is to find reasons WHY the market could go DOWN.

Analyze the provided market data and identify bearish signals.
Output your analysis in this EXACT JSON format:
{
  "stance": "STRONGLY_BEARISH",
  "bearish_reasons": "Your 3-5 key bearish observations, separated by semicolons",
  "bear_confidence": 75
}

stance must be one of: STRONGLY_BEARISH, SLIGHTLY_BEARISH, NEUTRAL, UNCERTAIN
bear_confidence should be 0-100 based on how strong the bearish case is.
Focus ONLY on bearish factors. Ignore bullish signals."""

        try:
            response = self.llm_client.chat(
                system_prompt=prompt,
                user_prompt=str(market_context),
                temperature=0.3,
                max_tokens=500,
            )
            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON
            import json
            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                data = json.loads(match.group())
                return BearResult(
                    stance=data.get('stance', 'UNCERTAIN'),
                    bearish_reasons=data.get('bearish_reasons', ''),
                    bear_confidence=float(data.get('bear_confidence', 50)),
                    raw_response=data,
                )
        except Exception:
            pass

        # Fallback to rule-based
        return self._analyze_rule_based(market_context)


def get_adversarial_perspectives(
    market_context: Dict[str, Any],
    bull_agent: Optional[BullAgent] = None,
    bear_agent: Optional[BearAgent] = None,
) -> tuple[BullResult, BearResult]:
    """
    Convenience function to get both bull and bear perspectives.

    Args:
        market_context: Market data for analysis
        bull_agent: Optional BullAgent instance (creates default if None)
        bear_agent: Optional BearAgent instance (creates default if None)

    Returns:
        Tuple of (BullResult, BearResult)
    """
    if bull_agent is None:
        bull_agent = BullAgent()
    if bear_agent is None:
        bear_agent = BearAgent()

    bull_result = bull_agent.analyze(market_context)
    bear_result = bear_agent.analyze(market_context)

    return bull_result, bear_result
