"""
Multi-Period Agent
==================

Summarizes multi-timeframe signals (1h/15m/5m) and four-layer status.
Designed to feed a concise, structured context to the Decision Core.

Based on LLM-TradeBot's MultiPeriodParserAgent.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class MultiPeriodResult:
    """Result from multi-period analysis"""
    alignment: bool  # Whether timeframes are aligned
    alignment_reason: str  # Human-readable alignment status
    bias: str  # BULLISH, BEARISH, NEUTRAL
    trend_scores: Dict[str, float]  # trend_1h, trend_15m, trend_5m
    oscillator_scores: Dict[str, float]  # osc_1h, osc_15m, osc_5m
    sentiment_score: float
    four_layer: Dict[str, Any]  # Four-layer filter results
    semantic_analyses: Dict[str, Any]  # Optional semantic analysis
    summary: str  # Human-readable summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alignment': self.alignment,
            'alignment_reason': self.alignment_reason,
            'bias': self.bias,
            'trend_scores': self.trend_scores,
            'oscillator_scores': self.oscillator_scores,
            'sentiment_score': self.sentiment_score,
            'four_layer': self.four_layer,
            'semantic_analyses': self.semantic_analyses,
            'summary': self.summary,
        }


class MultiPeriodAgent:
    """
    Multi-Period Parser Agent

    Analyzes multi-timeframe signal alignment and generates a summary
    for the Decision Core. Implements the priority: 1h > 15m > 5m.
    """

    def __init__(self):
        """Initialize the MultiPeriodAgent"""
        self.name = "multi_period_agent"

    @staticmethod
    def _score_to_sign(score: float, pos: float = 0, neg: float = 0) -> int:
        """
        Convert a score to a sign: 1 (positive), -1 (negative), 0 (neutral)

        Args:
            score: The score value
            pos: Positive threshold
            neg: Negative threshold

        Returns:
            1 if score >= pos, -1 if score <= neg, 0 otherwise
        """
        if score >= pos:
            return 1
        if score <= neg:
            return -1
        return 0

    def analyze(
        self,
        quant_analysis: Dict[str, Any],
        four_layer_result: Optional[Dict[str, Any]] = None,
        semantic_analyses: Optional[Dict[str, Any]] = None,
    ) -> MultiPeriodResult:
        """
        Analyze multi-timeframe signal alignment.

        Args:
            quant_analysis: Quant analysis containing trend and oscillator scores
            four_layer_result: Optional four-layer filter results
            semantic_analyses: Optional semantic analysis results

        Returns:
            MultiPeriodResult with alignment analysis
        """
        trend = quant_analysis.get('trend', {}) or {}
        oscillator = quant_analysis.get('oscillator', {}) or {}
        sentiment = quant_analysis.get('sentiment', {}) or {}

        # Extract trend scores for each timeframe
        t_1h = float(trend.get('trend_1h_score', 0) or 0)
        t_15m = float(trend.get('trend_15m_score', 0) or 0)
        t_5m = float(trend.get('trend_5m_score', 0) or 0)

        # Convert to signs using timeframe-specific thresholds
        sign_1h = self._score_to_sign(t_1h, 25, -25)
        sign_15m = self._score_to_sign(t_15m, 18, -18)
        sign_5m = self._score_to_sign(t_5m, 12, -12)

        # Determine alignment
        aligned, alignment_reason = self._check_alignment(sign_1h, sign_15m, sign_5m)

        # Determine bias
        bias = "BULLISH" if sign_1h > 0 else ("BEARISH" if sign_1h < 0 else "NEUTRAL")

        # Process four-layer results
        four_layer_result = four_layer_result or {}
        layer_flags = {
            "L1": bool(four_layer_result.get('layer1_pass', False)),
            "L2": bool(four_layer_result.get('layer2_pass', False)),
            "L3": bool(four_layer_result.get('layer3_pass', False)),
            "L4": bool(four_layer_result.get('layer4_pass', False)),
        }

        final_action = (four_layer_result.get('final_action', 'wait') or 'wait').upper()

        # Build summary
        summary = self._build_summary(
            alignment_reason=alignment_reason,
            t_1h=t_1h,
            t_15m=t_15m,
            t_5m=t_5m,
            final_action=final_action,
            layer_flags=layer_flags,
        )

        return MultiPeriodResult(
            alignment=aligned,
            alignment_reason=alignment_reason,
            bias=bias,
            trend_scores={
                'trend_1h': t_1h,
                'trend_15m': t_15m,
                'trend_5m': t_5m,
            },
            oscillator_scores={
                'osc_1h': float(oscillator.get('osc_1h_score', 0) or 0),
                'osc_15m': float(oscillator.get('osc_15m_score', 0) or 0),
                'osc_5m': float(oscillator.get('osc_5m_score', 0) or 0),
            },
            sentiment_score=float(sentiment.get('total_sentiment_score', 0) or 0),
            four_layer={
                'final_action': final_action,
                'layer_pass': layer_flags,
            },
            semantic_analyses=semantic_analyses or {},
            summary=summary,
        )

    def _check_alignment(
        self,
        sign_1h: int,
        sign_15m: int,
        sign_5m: int,
    ) -> tuple[bool, str]:
        """
        Check if multi-timeframe signals are aligned.

        Strategy:
        - Three timeframes aligned (all positive or all negative) -> Strong alignment
        - 1h and 15m aligned -> Partial alignment (5m may be noisy)
        - Other cases -> Not aligned

        Args:
            sign_1h: 1h trend sign (-1, 0, 1)
            sign_15m: 15m trend sign (-1, 0, 1)
            sign_5m: 5m trend sign (-1, 0, 1)

        Returns:
            (aligned: bool, reason: str)
        """
        # All three timeframes aligned
        if sign_1h == sign_15m == sign_5m and sign_1h != 0:
            direction = 'bull' if sign_1h > 0 else 'bear'
            return True, f"All timeframes aligned ({direction})"

        # 1h and 15m aligned (primary strategy timeframe)
        if sign_1h == sign_15m and sign_1h != 0:
            direction = 'bull' if sign_1h > 0 else 'bear'
            return True, f"1h+15m aligned ({direction})"

        # Not aligned - wait for 1h confirmation
        return False, (
            f"Misaligned (1h:{sign_1h}, 15m:{sign_15m}, 5m:{sign_5m}) - wait for 1h confirmation"
        )

    def _build_summary(
        self,
        alignment_reason: str,
        t_1h: float,
        t_15m: float,
        t_5m: float,
        final_action: str,
        layer_flags: Dict[str, bool],
    ) -> str:
        """Build human-readable summary"""
        return (
            f"Align={alignment_reason} | Trend 1h/15m/5m "
            f"{t_1h:+.0f}/{t_15m:+.0f}/{t_5m:+.0f} | "
            f"4-Layer {final_action} ("
            f"L1:{'Y' if layer_flags['L1'] else 'N'},"
            f"L2:{'Y' if layer_flags['L2'] else 'N'},"
            f"L3:{'Y' if layer_flags['L3'] else 'N'},"
            f"L4:{'Y' if layer_flags['L4'] else 'N'})"
        )

    def get_alignment_strength(self, result: MultiPeriodResult) -> float:
        """
        Calculate alignment strength score (0-100).

        Args:
            result: MultiPeriodResult from analyze()

        Returns:
            Alignment strength as a percentage
        """
        if result.alignment:
            # Fully aligned
            return 100.0

        sign_1h = self._score_to_sign(result.trend_scores.get('trend_1h', 0), 25, -25)
        sign_15m = self._score_to_sign(result.trend_scores.get('trend_15m', 0), 18, -18)
        sign_5m = self._score_to_sign(result.trend_scores.get('trend_5m', 0), 12, -12)

        # Count aligned timeframes
        aligned_count = 0
        if sign_1h == sign_15m and sign_1h != 0:
            aligned_count = 2
        elif sign_1h == sign_5m and sign_1h != 0:
            aligned_count = 2
        elif sign_15m == sign_5m and sign_15m != 0:
            aligned_count = 2
        elif sign_1h != 0:
            aligned_count = 1

        return (aligned_count / 3) * 100
