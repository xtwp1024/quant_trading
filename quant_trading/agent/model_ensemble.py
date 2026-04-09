"""
Model Ensemble — 多模型集成决策 + 四层策略过滤器
=================================================

Architecture:
  Layer 1 (Trend Filter):  宏观趋势判断 — 趋势方向过滤
  Layer 2 (AI Filter):     LLM综合分析 — 多模型投票
  Layer 3 (Setup):          技术面确认 — 支撑/阻力/形态
  Layer 4 (Trigger):        入场信号触发 — 精确入场点

Features:
  - Platt calibration for model confidence
  - Weighted ensemble voting
  - Four-layer filter pipeline
  - Decision result with filter breakdown

Usage:
  >>> bridge = MultiLLMBridge({'deepseek': 'sk-xxx', 'openai': 'sk-xxx'})
  >>> ensemble = ModelEnsemble(bridge)
  >>> decision = ensemble.decide({'symbol': 'BTC', 'price': 50000, 'trend': 'up'})
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .multi_llm_bridge import (
    MultiLLMBridge,
    LLMResponse,
    platt_calibrate,
)

__all__ = [
    "ModelEnsemble",
    "ModelVote",
    "StrategyFilter",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StrategyFilter(Enum):
    """四层策略过滤器 / Four-layer strategy filter."""
    TREND = "trend"
    AI_FILTER = "ai_filter"
    SETUP = "setup"
    TRIGGER = "trigger"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ModelVote:
    """Single model vote result."""
    model: str
    vote: str  # e.g. "BUY", "SELL", "HOLD"
    confidence: float  # 0.0–1.0
    reasoning: str = ""


@dataclass
class FilterResult:
    """Result of a single filter layer."""
    filter_name: StrategyFilter
    passed: bool
    confidence: float = 0.5
    reason: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class EnsembleDecision:
    """Final decision from the model ensemble."""
    action: str  # "BUY", "SELL", "HOLD", "SKIP"
    confidence: float  # 0.0–1.0
    filters_passed: list[StrategyFilter] = field(default_factory=list)
    votes: list[ModelVote] = field(default_factory=list)
    filter_results: list[FilterResult] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "filters_passed": [f.value for f in self.filters_passed],
            "vote_summary": {
                v.model: {"vote": v.vote, "confidence": round(v.confidence, 3)}
                for v in self.votes
            },
            "reasoning": self.reasoning[:200] if self.reasoning else "",
        }


# ---------------------------------------------------------------------------
# Trend filter helpers
# ---------------------------------------------------------------------------

def _trend_direction(market_data: dict) -> tuple[str, float]:
    """Determine trend direction from market_data.

    Args:
        market_data: Dict with keys like 'price', 'sma_20', 'sma_50',
                     'trend', 'momentum', etc.

    Returns:
        (direction, confidence) where direction in {'up', 'down', 'sideways'}
    """
    # Priority: explicit trend field > SMA crossover > momentum
    if "trend" in market_data:
        t = str(market_data["trend"]).lower()
        if t in ("up", "bull", "bullish"):
            return "up", 0.9
        if t in ("down", "bear", "bearish"):
            return "down", 0.9
        if t in ("sideways", "neutral", "flat"):
            return "sideways", 0.7

    # SMA-based detection
    price = market_data.get("price", 0)
    sma_20 = market_data.get("sma_20", price)
    sma_50 = market_data.get("sma_50", price)

    if sma_20 > sma_50 * 1.01:
        return "up", 0.75
    if sma_20 < sma_50 * 0.99:
        return "down", 0.75
    return "sideways", 0.6


# ---------------------------------------------------------------------------
# Setup filter helpers (technical confirmation)
# ---------------------------------------------------------------------------

def _technical_setup(market_data: dict) -> tuple[bool, float, str]:
    """Evaluate technical setup (support/resistance/patterns).

    Returns:
        (passed, confidence, reason)
    """
    # Check RSI
    rsi = market_data.get("rsi", 50)
    if rsi < 30:
        return True, 0.7, f"Oversold RSI={rsi:.1f}"
    if rsi > 70:
        return True, 0.7, f"Overbought RSI={rsi:.1f}"

    # Check volume
    volume = market_data.get("volume_ratio", 1.0)
    if volume < 0.5:
        return False, 0.8, f"Low volume ratio={volume:.2f}"

    # Check ATR (volatility)
    atr_pct = market_data.get("atr_pct", 0.02)
    if atr_pct > 0.10:
        return False, 0.6, f"Extreme volatility ATR%={atr_pct:.2%}"

    return True, 0.6, "Technical setup acceptable"


# ---------------------------------------------------------------------------
# Trigger filter helpers
# ---------------------------------------------------------------------------

def _trigger_signal(market_data: dict) -> tuple[bool, float, str]:
    """Evaluate entry trigger conditions.

    Returns:
        (passed, confidence, reason)
    """
    # Check for clear entry signals
    macd_signal = market_data.get("macd_signal", 0)
    stochastic_k = market_data.get("stoch_k", 50)

    # Bullish trigger: MACD crosses above signal AND stochastic < 80
    if macd_signal > 0 and stochastic_k < 80:
        return True, 0.8, "Bullish MACD cross, stoch acceptable"

    # Bearish trigger: MACD crosses below signal AND stochastic > 20
    if macd_signal < 0 and stochastic_k > 20:
        return True, 0.8, "Bearish MACD cross, stoch acceptable"

    # No clear signal
    return False, 0.5, "No clear trigger signal"


# ---------------------------------------------------------------------------
# Main ensemble class
# ---------------------------------------------------------------------------

class ModelEnsemble:
    """多模型集成决策 — 四层过滤 + Platt校准.

    架构:
      Layer 1 (Trend Filter): 宏观趋势判断
      Layer 2 (AI Filter):     LLM综合分析 (MultiLLMBridge.ensemble_generate)
      Layer 3 (Setup):         技术面确认
      Layer 4 (Trigger):       入场信号触发

    Usage:
      bridge = MultiLLMBridge({'deepseek': 'sk-xxx'})
      ensemble = ModelEnsemble(bridge)
      decision = ensemble.decide({'symbol': 'BTC', 'price': 50000})
    """

    DEFAULT_MODELS: list[str] = ["deepseek", "openai", "anthropic"]
    DEFAULT_WEIGHTS: dict[str, float] = {
        "deepseek": 0.40,
        "openai": 0.35,
        "anthropic": 0.25,
    }

    def __init__(self, bridge: MultiLLMBridge):
        self.bridge = bridge
        self._calibration_state: dict[str, dict[str, float]] = {}

    def calibrate_confidence(
        self, raw_confidence: float, model: str
    ) -> float:
        """Platt校准 — 将原始LLM置信度转换为校准后概率.

        Args:
            raw_confidence: Raw model confidence (0.0–1.0)
            model: Model/provider name

        Returns:
            Calibrated probability in [0.01, 0.99]
        """
        # Per-model Platt parameters (a, b) — can be learned from history
        params = self._calibration_state.get(model, {"a": 0.90, "b": 0.0})
        return platt_calibrate(raw_confidence, a=params["a"], b=params["b"])

    def apply_filters(self, market_data: dict) -> list[FilterResult]:
        """应用四层过滤器 / Apply all four filter layers.

        Args:
            market_data: Dict containing market information:
                - symbol, price, trend
                - sma_20, sma_50, rsi, volume_ratio, atr_pct
                - macd_signal, stoch_k
                - Any other technical indicators

        Returns:
            List of FilterResult, one per layer.
        """
        results: list[FilterResult] = []

        # Layer 1: Trend Filter
        trend_dir, trend_conf = _trend_direction(market_data)
        results.append(FilterResult(
            filter_name=StrategyFilter.TREND,
            passed=trend_dir in ("up", "down"),
            confidence=trend_conf,
            reason=f"Trend={trend_dir} (conf={trend_conf:.2f})",
            details={"direction": trend_dir},
        ))

        # Layer 2: AI Filter (placeholder — filled by vote())
        results.append(FilterResult(
            filter_name=StrategyFilter.AI_FILTER,
            passed=False,  # Filled by vote()
            confidence=0.5,
            reason="Pending AI analysis",
        ))

        # Layer 3: Technical Setup
        setup_pass, setup_conf, setup_reason = _technical_setup(market_data)
        results.append(FilterResult(
            filter_name=StrategyFilter.SETUP,
            passed=setup_pass,
            confidence=setup_conf,
            reason=setup_reason,
        ))

        # Layer 4: Trigger Signal
        trigger_pass, trigger_conf, trigger_reason = _trigger_signal(market_data)
        results.append(FilterResult(
            filter_name=StrategyFilter.TRIGGER,
            passed=trigger_pass,
            confidence=trigger_conf,
            reason=trigger_reason,
        ))

        return results

    def vote(
        self,
        prompt: str,
        models: list[str] | None = None,
    ) -> list[ModelVote]:
        """多模型投票 / Query multiple models and return votes.

        Args:
            prompt: Trading question/prompt
            models: List of provider keys. Defaults to DEFAULT_MODELS.

        Returns:
            List of ModelVote, one per model.
        """
        models = models or self.DEFAULT_MODELS
        # Build weights dict (only for configured models)
        weights = {
            m: self.DEFAULT_WEIGHTS.get(m, 1.0 / len(models))
            for m in models
            if m in self.bridge.api_keys
        }

        try:
            ensemble_resp = self.bridge.ensemble_generate(
                prompt,
                models=list(weights.keys()),
                weights=weights,
            )
        except Exception as e:
            return [
                ModelVote(model=m, vote="HOLD", confidence=0.0, reasoning=f"Error: {e}")
                for m in models
            ]

        # Parse votes from ensemble response
        votes: list[ModelVote] = []
        raw_responses = ensemble_resp.raw_response.get("ensemble_responses", []) if ensemble_resp.raw_response else []

        for r in raw_responses:
            content = r.get("content", "")
            vote, reason = self._parse_vote(content)
            raw_conf = r.get("confidence", 0.5)
            calibrated = self.calibrate_confidence(raw_conf, r.get("model", "unknown"))
            votes.append(ModelVote(
                model=r.get("model", "unknown"),
                vote=vote,
                confidence=calibrated,
                reasoning=reason or content[:100],
            ))

        # If no parsed votes, create a synthetic HOLD
        if not votes:
            votes.append(ModelVote(
                model="ensemble",
                vote="HOLD",
                confidence=ensemble_resp.confidence,
                reasoning=ensemble_resp.content[:100] if ensemble_resp.content else "No response",
            ))

        return votes

    def _parse_vote(self, content: str) -> tuple[str, str]:
        """Parse trading vote (BUY/SELL/HOLD) and reasoning from LLM content."""
        content_lower = content.lower()
        if any(kw in content_lower for kw in ("buy", "long", "做多", "买入", "看涨")):
            vote = "BUY"
        elif any(kw in content_lower for kw in ("sell", "short", "做空", "卖出", "看跌")):
            vote = "SELL"
        else:
            vote = "HOLD"

        # Extract reasoning (first sentence or line)
        lines = content.strip().split("\n")
        reasoning = lines[0][:200] if lines else ""
        return vote, reasoning

    def decide(self, market_data: dict) -> dict:
        """最终决策 / Final decision combining all filters and votes.

        Args:
            market_data: Market data dict (see apply_filters)

        Returns:
            Dict with action, confidence, filters_passed, votes, reasoning.
            Matches EnsembleDecision.to_dict() format.
        """
        # 1. Apply four-layer filters
        filter_results = self.apply_filters(market_data)

        # 2. Build AI filter prompt
        symbol = market_data.get("symbol", "UNKNOWN")
        prompt = (
            f"分析 {symbol} 当前市场状况，决定操作方向。\n"
            f"价格: {market_data.get('price', 'N/A')}\n"
            f"趋势: {market_data.get('trend', 'N/A')}\n"
            f"RSI: {market_data.get('rsi', 'N/A')}\n"
            f"交易量比: {market_data.get('volume_ratio', 'N/A')}\n"
            f"MACD信号: {market_data.get('macd_signal', 'N/A')}\n"
            f"随机K值: {market_data.get('stoch_k', 'N/A')}\n"
            f"\n给出操作建议: BUY (买入), SELL (卖出), 或 HOLD (观望)，并简述理由。"
        )

        # 3. Run multi-model vote
        votes = self.vote(prompt)
        vote_agreements = [v.vote for v in votes]
        buy_count = vote_agreements.count("BUY")
        sell_count = vote_agreements.count("SELL")

        # 4. Update AI filter result
        ai_filter_result = next(
            (r for r in filter_results if r.filter_name == StrategyFilter.AI_FILTER),
            None,
        )
        if ai_filter_result:
            if buy_count > sell_count:
                ai_filter_result.passed = True
                ai_filter_result.confidence = buy_count / max(len(votes), 1)
                ai_filter_result.reason = f"BUY majority: {buy_count}/{len(votes)}"
            elif sell_count > buy_count:
                ai_filter_result.passed = True
                ai_filter_result.confidence = sell_count / max(len(votes), 1)
                ai_filter_result.reason = f"SELL majority: {sell_count}/{len(votes)}"
            else:
                ai_filter_result.passed = False
                ai_filter_result.confidence = 0.5
                ai_filter_result.reason = "No majority / tie"

        # 5. Determine final action
        all_passed = all(r.passed for r in filter_results)
        avg_confidence = sum(r.confidence for r in filter_results) / len(filter_results)

        if all_passed and buy_count > sell_count:
            action = "BUY"
        elif all_passed and sell_count > buy_count:
            action = "SELL"
        elif not all_passed:
            action = "SKIP"
        else:
            action = "HOLD"

        filters_passed = [r.filter_name for r in filter_results if r.passed]

        # 6. Build reasoning string
        reasons = [r.reason for r in filter_results if r.reason]
        reasoning = " | ".join(reasons[:3])

        decision = EnsembleDecision(
            action=action,
            confidence=avg_confidence,
            filters_passed=filters_passed,
            votes=votes,
            filter_results=filter_results,
            reasoning=reasoning,
        )

        return decision.to_dict()
