"""Multi-LLM ensemble prediction framework.

Based on Fully-Autonomous-Polymarket-AI-Trading-Bot's ensemble architecture:
  - Multi-model ensemble: GPT-4o (40%) + Claude 3.5 (35%) + Gemini 1.5 (25%)
  - Platt scaling calibration + historical learning
  - Adaptive model weighting based on per-category Brier scores
  - Trimmed mean / median / weighted aggregation

Key concepts:
- Skill-first integration: works as a Skill in the existing agent framework
- Adaptive calibration: learns from historical forecast → outcome data
- Ensemble disagreement detection: penalizes spread when models diverge
- Confidence-level output: LOW / MEDIUM / HIGH for downstream risk gating
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AggregationMethod(str, Enum):
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    WEIGHTED = "weighted"


# Default ensemble weights (can be overridden by adaptive learning)
DEFAULT_WEIGHTS = {
    "gpt-4o": 0.40,
    "claude-3-5-sonnet-20241022": 0.35,
    "gemini-1.5-pro": 0.25,
}

DEFAULT_MODELS = list(DEFAULT_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class ModelPrediction:
    """Raw prediction from a single LLM."""
    model_name: str
    probability: float          # 0.01–0.99
    confidence: ConfidenceLevel
    reasoning: str = ""
    invalidation_triggers: list[str] = field(default_factory=list)
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0


@dataclass
class CalibrationResult:
    """Result of Platt scaling + heuristic adjustments."""
    raw_probability: float
    calibrated_probability: float
    method: str
    adjustments: list[str] = field(default_factory=list)


@dataclass
class EnsembleResult:
    """Aggregated result from the multi-LLM ensemble."""
    probability: float          # final calibrated probability
    confidence: ConfidenceLevel
    individual_predictions: list[ModelPrediction] = field(default_factory=list)
    models_succeeded: int = 0
    models_failed: int = 0
    aggregation_method: str = "trimmed_mean"
    spread: float = 0.0          # max - min probability across models
    agreement_score: float = 0.0  # 1.0 = perfect agreement
    reasoning: str = ""
    invalidation_triggers: list[str] = field(default_factory=list)
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    # Calibration metadata
    calibration: CalibrationResult | None = None
    # Adaptive weight metadata
    weights_used: dict[str, float] = field(default_factory=dict)
    weights_source: str = "default"  # "default" | "adaptive" | "blended"

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": round(self.probability, 4),
            "confidence": self.confidence.value,
            "models_succeeded": self.models_succeeded,
            "models_failed": self.models_failed,
            "aggregation_method": self.aggregation_method,
            "spread": round(self.spread, 4),
            "agreement_score": round(self.agreement_score, 3),
            "reasoning": self.reasoning[:200],
            "invalidation_triggers": self.invalidation_triggers[:5],
            "key_evidence_count": len(self.key_evidence),
            "weights_used": {k: round(v, 3) for k, v in self.weights_used.items()},
            "weights_source": self.weights_source,
            "calibration": {
                "raw": round(self.calibration.raw_probability, 4) if self.calibration else None,
                "calibrated": round(self.calibration.calibrated_probability, 4) if self.calibration else None,
                "method": self.calibration.method if self.calibration else None,
            } if self.calibration else None,
        }


# ---------------------------------------------------------------------------
# Platt Scaling / Historical Calibration
# ---------------------------------------------------------------------------

# Global historical calibrator state
_calibrator_state: dict[str, Any] = {"a": 1.0, "b": 0.0, "is_fitted": False}


def _logit(p: float) -> float:
    """Safe logit transform."""
    p = max(0.01, min(0.99, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def platt_scale(prob: float, a: float = 0.90, b: float = 0.0) -> float:
    """Apply Platt-like shrinkage toward 0.5.

    Args:
        prob: Raw probability
        a: Shrinkage factor (0.9 = 10% shrinkage toward 0.5)
        b: Intercept offset
    """
    p = max(0.01, min(0.99, prob))
    logit_p = _logit(p)
    shrunk = logit_p * a + b
    return max(0.01, min(0.99, _sigmoid(shrunk)))


def calibrate_probability(
    raw_prob: float,
    evidence_quality: float = 0.5,
    num_contradictions: int = 0,
    ensemble_spread: float = 0.0,
    low_evidence_penalty: float = 0.15,
    historical_a: float = 0.90,
    historical_b: float = 0.0,
    use_historical: bool = False,
) -> CalibrationResult:
    """Calibrate a raw probability with Platt scaling + heuristic adjustments.

    Steps:
      1. Platt / historical calibration (shrinks extremes)
      2. Low-evidence penalty (pull toward 0.5 when quality is poor)
      3. Contradiction penalty (pull toward 0.5 per detected contradiction)
      4. Ensemble disagreement penalty (pull toward 0.5 when models diverge)

    Args:
        raw_prob: Raw model probability (0.01–0.99)
        evidence_quality: 0.0–1.0 quality score of supporting evidence
        num_contradictions: Number of conflicting evidence claims
        ensemble_spread: Probability spread across ensemble models
        low_evidence_penalty: Max penalty weight for poor evidence
        historical_a: Historical calibrator slope
        historical_b: Historical calibrator intercept
        use_historical: If True, apply learned calibration instead of Platt

    Returns:
        CalibrationResult with adjustments breakdown
    """
    adjustments: list[str] = []
    p = max(0.01, min(0.99, raw_prob))

    # Step 1: Platt / historical calibration
    if use_historical and _calibrator_state["is_fitted"]:
        p_cal = platt_scale(p, _calibrator_state["a"], _calibrator_state["b"])
        method = "historical"
    else:
        p_cal = platt_scale(p, historical_a, historical_b)
        method = "platt"

    if abs(p_cal - p) > 0.005:
        adjustments.append(f"extremity_shrink: {p:.3f} → {p_cal:.3f}")
        p = p_cal

    # Step 2: Low evidence penalty
    if evidence_quality < 0.4:
        penalty = low_evidence_penalty * (1.0 - evidence_quality)
        p_pen = p * (1.0 - penalty) + 0.5 * penalty
        adjustments.append(f"low_evidence (q={evidence_quality:.2f}): {p:.3f} → {p_pen:.3f}")
        p = p_pen

    # Step 3: Contradiction penalty
    if num_contradictions > 0:
        penalty = min(0.3, 0.1 * num_contradictions)
        p_con = p * (1.0 - penalty) + 0.5 * penalty
        adjustments.append(f"contradictions ({num_contradictions}): {p:.3f} → {p_con:.3f}")
        p = p_con

    # Step 4: Ensemble disagreement penalty
    if ensemble_spread > 0.10:
        penalty = min(0.25, ensemble_spread)
        p_spread = p * (1.0 - penalty) + 0.5 * penalty
        adjustments.append(f"ensemble_spread ({ensemble_spread:.2f}): {p:.3f} → {p_spread:.3f}")
        p = p_spread

    p = max(0.01, min(0.99, p))

    return CalibrationResult(
        raw_probability=raw_prob,
        calibrated_probability=p,
        method=method,
        adjustments=adjustments,
    )


# ---------------------------------------------------------------------------
# Adaptive Model Weighting
# ---------------------------------------------------------------------------

MIN_SAMPLES_PER_MODEL = 5
BLEND_FULL_CONFIDENCE_SAMPLES = 50


def compute_adaptive_weights(
    conn: sqlite3.Connection | None,
    category: str = "ALL",
    default_weights: dict[str, float] | None = None,
    models: list[str] | None = None,
) -> tuple[dict[str, float], str]:
    """Compute adaptive model weights from historical Brier scores.

    Uses inverse-Brier weighting (lower Brier = higher weight) with Bayesian
    blending toward default weights based on sample size.

    Args:
        conn: SQLite connection to model_forecast_log table
        category: Market category filter (or "ALL")
        default_weights: Fallback weights when insufficient data
        models: List of model names to include

    Returns:
        (weights_dict, source) where source is one of {"default", "adaptive", "blended"}
    """
    models = models or DEFAULT_MODELS
    default_weights = default_weights or DEFAULT_WEIGHTS

    if conn is None:
        return dict(default_weights), "default"

    try:
        if category == "ALL":
            rows = conn.execute("""
                SELECT model_name,
                       AVG((forecast_prob - actual_outcome) *
                           (forecast_prob - actual_outcome)) AS brier,
                       COUNT(*) AS cnt
                FROM model_forecast_log
                GROUP BY model_name
                HAVING cnt >= ?
            """, (MIN_SAMPLES_PER_MODEL,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT model_name,
                       AVG((forecast_prob - actual_outcome) *
                           (forecast_prob - actual_outcome)) AS brier,
                       COUNT(*) AS cnt
                FROM model_forecast_log
                WHERE category = ?
                GROUP BY model_name
                HAVING cnt >= ?
            """, (category, MIN_SAMPLES_PER_MODEL)).fetchall()
    except sqlite3.OperationalError:
        return dict(default_weights), "default"

    if not rows:
        return dict(default_weights), "default"

    # Raw inverse-Brier weights
    raw: dict[str, float] = {}
    min_samples = float("inf")
    for r in rows:
        brier = float(r["brier"])
        cnt = int(r["cnt"])
        min_samples = min(min_samples, cnt)
        raw[r["model_name"]] = 1.0 / max(brier, 0.001)

    # Blend factor based on sample size
    blend = min(1.0, min_samples / BLEND_FULL_CONFIDENCE_SAMPLES)

    # Blend learned + default weights
    final: dict[str, float] = {}
    for model in models:
        default_w = default_weights.get(model, 1.0 / len(models))
        if model in raw:
            blended = blend * raw[model] + (1.0 - blend) * default_w
            final[model] = blended
        else:
            final[model] = default_w

    # Normalize to sum to 1.0
    total = sum(final.values())
    if total > 0:
        final = {k: v / total for k, v in final.items()}

    source = "blended" if 0 < blend < 0.95 else ("adaptive" if blend >= 0.95 else "default")
    return final, source


# ---------------------------------------------------------------------------
# Ensemble Aggregation
# ---------------------------------------------------------------------------


def aggregate_predictions(
    predictions: list[ModelPrediction],
    method: AggregationMethod | str = AggregationMethod.TRIMMED_MEAN,
    weights: dict[str, float] | None = None,
    trim_fraction: float = 0.1,
) -> tuple[float, float]:
    """Aggregate probabilities from multiple models.

    Args:
        predictions: List of ModelPrediction objects
        method: Aggregation method
        weights: Model weights (required for weighted method)
        trim_fraction: Fraction to trim from each end (trimmed_mean only)

    Returns:
        (aggregated_probability, spread)
    """
    if not predictions:
        return 0.5, 0.0

    probs = [p.probability for p in predictions]

    if len(probs) == 1:
        return probs[0], 0.0

    if isinstance(method, str):
        method = AggregationMethod(method)

    if method == AggregationMethod.MEDIAN:
        sorted_p = sorted(probs)
        mid = len(sorted_p) // 2
        if len(sorted_p) % 2 == 0:
            agg = (sorted_p[mid - 1] + sorted_p[mid]) / 2.0
        else:
            agg = sorted_p[mid]

    elif method == AggregationMethod.WEIGHTED:
        weights = weights or {}
        total_weight = 0.0
        weighted_sum = 0.0
        for pred in predictions:
            w = weights.get(pred.model_name, 1.0 / len(predictions))
            weighted_sum += pred.probability * w
            total_weight += w
        agg = weighted_sum / total_weight if total_weight > 0 else 0.5

    else:  # TRIMMED_MEAN (default)
        if len(probs) <= 2:
            agg = sum(probs) / len(probs)
        else:
            sorted_p = sorted(probs)
            trim = max(1, int(len(sorted_p) * trim_fraction))
            trimmed = sorted_p[trim:-trim] if trim < len(sorted_p) // 2 else sorted_p
            agg = sum(trimmed) / len(trimmed) if trimmed else sum(probs) / len(probs)

    spread = max(probs) - min(probs)
    return agg, spread


def aggregate_confidence(predictions: list[ModelPrediction]) -> ConfidenceLevel:
    """Aggregate confidence levels from individual predictions."""
    if not predictions:
        return ConfidenceLevel.LOW

    order = {ConfidenceLevel.LOW: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.HIGH: 2}
    values = [order.get(p.confidence, 0) for p in predictions]
    avg = sum(values) / len(values)

    # If spread is large, reduce confidence
    probs = [p.probability for p in predictions]
    spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
    if spread > 0.15:
        return ConfidenceLevel.LOW

    if avg >= 1.5:
        return ConfidenceLevel.HIGH
    elif avg >= 0.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def merge_reasoning(predictions: list[ModelPrediction], max_items: int = 3) -> str:
    """Merge reasoning from multiple models."""
    reasons = [p.reasoning for p in predictions if p.reasoning]
    return " | ".join(reasons[:max_items])


def merge_invalidation_triggers(predictions: list[ModelPrediction]) -> list[str]:
    """Merge invalidation triggers deduplicated across models."""
    seen: set[str] = set()
    result: list[str] = []
    for pred in predictions:
        for t in pred.invalidation_triggers:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                result.append(t)
    return result[:5]


# ---------------------------------------------------------------------------
# Ensemble Forecaster (main entry point)
# ---------------------------------------------------------------------------


class MultiLLMEnsemble:
    """Multi-LLM ensemble for probabilistic forecasting.

    Integrates with the existing Skill framework:
      1. Query multiple LLMs concurrently (GPT-4o, Claude, Gemini)
      2. Aggregate predictions with trimmed mean / median / weighted schemes
      3. Apply Platt scaling + heuristic calibration
      4. Learn adaptive weights from historical forecast → outcome data
      5. Return calibrated probability with confidence level

    Usage::

        ensemble = MultiLLMEnsemble(models=["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"])
        result = await ensemble.forecast(
            question="Will Fed cut rates in March 2025?",
            evidence_quality=0.72,
            num_contradictions=0,
            evidence_bullets=[...],
            category="MACRO",
            db_conn=db_conn,
        )
    """

    def __init__(
        self,
        models: list[str] | None = None,
        weights: dict[str, float] | None = None,
        aggregation: AggregationMethod | str = AggregationMethod.TRIMMED_MEAN,
        trim_fraction: float = 0.1,
        calibration_a: float = 0.90,
        calibration_b: float = 0.0,
    ):
        self.models = models or DEFAULT_MODELS.copy()
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.aggregation = AggregationMethod(aggregation) if isinstance(aggregation, str) else aggregation
        self.trim_fraction = trim_fraction
        self.calibration_a = calibration_a
        self.calibration_b = calibration_b
        self._external_weights: dict[str, float] | None = None

    def set_adaptive_weights(self, weights: dict[str, float]) -> None:
        """Inject learned per-category weights from AdaptiveModelWeighter."""
        self._external_weights = weights

    async def forecast(
        self,
        question: str,
        evidence_quality: float = 0.5,
        num_contradictions: int = 0,
        evidence_bullets: list[str] | None = None,
        category: str = "ALL",
        db_conn: sqlite3.Connection | None = None,
        # Market features for signal injection
        volume_usd: float = 0.0,
        liquidity_usd: float = 0.0,
        spread_pct: float = 0.0,
        days_to_expiry: float = 30.0,
        price_momentum: float = 0.0,
        num_sources: int = 0,
        market_type: str = "UNKNOWN",
        # Override LLM query (if None, uses mock for dry-run / testing)
        _query_fn=None,  # async(model, prompt) -> ModelPrediction
    ) -> EnsembleResult:
        """Run the full ensemble forecasting pipeline.

        Args:
            question: Market question text
            evidence_quality: Quality score of supporting evidence (0–1)
            num_contradictions: Number of conflicting evidence claims
            evidence_bullets: Key evidence bullet points
            category: Market category for adaptive weighting
            db_conn: SQLite connection for adaptive weight learning
            volume_usd: Market volume in USD
            liquidity_usd: Market liquidity in USD
            spread_pct: Bid-ask spread percentage
            days_to_expiry: Days until market resolution
            price_momentum: 24h price change rate
            num_sources: Number of corroborating sources
            market_type: Market type classification
            _query_fn: Override LLM query function for testing

        Returns:
            EnsembleResult with calibrated probability and metadata
        """
        # Build prompt
        prompt = self._build_prompt(
            question=question,
            market_type=market_type,
            evidence_bullets=evidence_bullets or [],
            volume_usd=volume_usd,
            liquidity_usd=liquidity_usd,
            spread_pct=spread_pct,
            days_to_expiry=days_to_expiry,
            price_momentum=price_momentum,
            evidence_quality=evidence_quality,
            num_sources=num_sources,
        )

        # Query all models concurrently
        query_fn = _query_fn or self._mock_query
        tasks = [query_fn(model, prompt) for model in self.models]
        predictions = await self._gather_predictions(tasks)

        # Separate successes and failures
        successes = [p for p in predictions if not p.error]
        failures = [p for p in predictions if p.error]

        if not successes:
            return EnsembleResult(
                probability=0.5,
                confidence=ConfidenceLevel.LOW,
                models_succeeded=0,
                models_failed=len(failures),
                aggregation_method=self.aggregation.value,
            )

        # Compute adaptive weights if DB connection provided
        if db_conn is not None:
            self.weights, weights_source = compute_adaptive_weights(
                db_conn, category=category, default_weights=DEFAULT_WEIGHTS, models=self.models
            )
        else:
            weights_source = "default"

        # Inject external weights if set
        if self._external_weights:
            self.weights = dict(self._external_weights)
            weights_source = "adaptive"

        # Aggregate probabilities
        agg_prob, spread = aggregate_predictions(
            successes,
            method=self.aggregation,
            weights=self.weights,
            trim_fraction=self.trim_fraction,
        )

        # Compute agreement score
        agreement = max(0.0, 1.0 - spread * 2.0)

        # Aggregate confidence
        agg_confidence = aggregate_confidence(successes)

        # Calibration
        calibration = calibrate_probability(
            raw_prob=agg_prob,
            evidence_quality=evidence_quality,
            num_contradictions=num_contradictions,
            ensemble_spread=spread,
            historical_a=self.calibration_a,
            historical_b=self.calibration_b,
        )

        # Build result
        result = EnsembleResult(
            probability=calibration.calibrated_probability,
            confidence=agg_confidence,
            individual_predictions=predictions,
            models_succeeded=len(successes),
            models_failed=len(failures),
            aggregation_method=self.aggregation.value,
            spread=spread,
            agreement_score=agreement,
            reasoning=merge_reasoning(successes),
            invalidation_triggers=merge_invalidation_triggers(successes),
            key_evidence=_collect_key_evidence(successes),
            calibration=calibration,
            weights_used=dict(self.weights),
            weights_source=weights_source,
        )

        return result

    async def _gather_predictions(
        self, tasks: list[tuple]
    ) -> list[ModelPrediction]:
        """Gather model predictions from coroutines."""
        import asyncio
        results: list[ModelPrediction] = []
        for coro in tasks:
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                results.append(ModelPrediction(
                    model_name="unknown",
                    probability=0.5,
                    confidence=ConfidenceLevel.LOW,
                    error=str(e),
                ))
        return results

    def _build_prompt(
        self,
        question: str,
        market_type: str,
        evidence_bullets: list[str],
        volume_usd: float,
        liquidity_usd: float,
        spread_pct: float,
        days_to_expiry: float,
        price_momentum: float,
        evidence_quality: float,
        num_sources: int,
    ) -> str:
        """Build the forecasting prompt."""
        bullets_text = "\n".join(f"- {b}" for b in evidence_bullets) if evidence_bullets else "No evidence available."
        return f"""\
You are an expert probabilistic forecaster analyzing a prediction market.

MARKET QUESTION: {question}
MARKET TYPE: {market_type}

EVIDENCE SUMMARY:
Based on available data, provide your probability estimate.

TOP EVIDENCE BULLETS:
{bullets_text}

MARKET FEATURES:
- Volume: ${volume_usd:,.0f}
- Liquidity: ${liquidity_usd:,.0f}
- Spread: {spread_pct:.1%}
- Days to expiry: {days_to_expiry:.0f}
- Price momentum (24h): {price_momentum:+.3f}
- Evidence quality score: {evidence_quality:.2f}
- Sources analyzed: {num_sources}

TASK:
Based ONLY on the evidence above, produce an independent probability
estimate for the question. Do NOT try to guess or anchor to any market
price — form your own view from the evidence.

Return valid JSON:
{{
  "probability": <0.01-0.99>,
  "confidence_level": "LOW" | "MEDIUM" | "HIGH",
  "reasoning": "2-4 sentence explanation",
  "invalidation_triggers": ["specific event that would change this forecast"],
  "key_evidence": [{{"text": "evidence bullet", "source": "publisher", "impact": "supports/opposes/neutral"}}]
}}

RULES:
- Probability must be between 0.01 and 0.99.
- If evidence is weak (quality < 0.3), bias toward 0.50.
- If evidence contradicts itself, widen uncertainty toward 0.50.
- Never claim certainty. Express epistemic humility.
- Do NOT hallucinate data not present in the evidence.

Return ONLY valid JSON, no markdown fences.
"""

    async def _mock_query(self, model: str, prompt: str) -> ModelPrediction:
        """Mock LLM query for dry-run / testing."""
        import random
        await asyncio.sleep(0.05)  # Simulate network latency

        # Simple heuristic based on prompt keywords
        prob = 0.50 + random.uniform(-0.15, 0.15)
        prob = max(0.01, min(0.99, prob))

        conf_map = {"gpt-4o": "HIGH", "claude-3-5-sonnet-20241022": "HIGH", "gemini-1.5-pro": "MEDIUM"}
        confidence = ConfidenceLevel(conf_map.get(model, "MEDIUM"))

        return ModelPrediction(
            model_name=model,
            probability=prob,
            confidence=confidence,
            reasoning=f"[{model}] Auto-generated forecast for testing",
            latency_ms=50.0,
        )


def _collect_key_evidence(predictions: list[ModelPrediction]) -> list[dict[str, Any]]:
    """Collect deduplicated key evidence from predictions."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for pred in predictions:
        for ev in pred.key_evidence:
            text = ev.get("text", "")
            if text and text not in seen:
                seen.add(text)
                result.append(ev)
    return result[:8]


# ---------------------------------------------------------------------------
# Calibration History Recording
# ---------------------------------------------------------------------------


def record_forecast_outcome(
    conn: sqlite3.Connection,
    market_id: str,
    question: str,
    category: str,
    forecast_prob: float,
    actual_outcome: float,
    model_forecasts: dict[str, float],
    stake_usd: float = 0.0,
    entry_price: float = 0.0,
    exit_price: float = 0.0,
    pnl: float = 0.0,
    holding_hours: float = 0.0,
    edge_at_entry: float = 0.0,
    confidence: str = "MEDIUM",
    evidence_quality: float = 0.5,
    resolved_at: str | None = None,
) -> None:
    """Record a market resolution for calibration learning.

    Stores:
      1. calibration_history: (forecast_prob, actual_outcome) pairs
      2. model_forecast_log: per-model forecast accuracy
      3. performance_log: trade performance metrics

    Args:
        conn: SQLite connection
        market_id: Unique market identifier
        question: Market question text
        category: Market category
        forecast_prob: Bot's final forecast probability
        actual_outcome: 1.0 = YES resolved, 0.0 = NO resolved
        model_forecasts: Dict of model_name -> individual forecast probability
        stake_usd: Position size in USD
        entry_price: Entry price for the trade
        exit_price: Exit price
        pnl: Realized PnL
        holding_hours: Hours the position was held
        edge_at_entry: Edge at time of entry
        confidence: Confidence level at entry
        evidence_quality: Evidence quality at entry
        resolved_at: ISO timestamp of resolution
    """
    now_iso = resolved_at or _utc_now_iso()

    # 1. Calibration history
    try:
        conn.execute("""
            INSERT INTO calibration_history
                (forecast_prob, actual_outcome, recorded_at, market_id)
            VALUES (?, ?, ?, ?)
        """, (forecast_prob, actual_outcome, now_iso, market_id))
    except sqlite3.OperationalError:
        _create_calibration_tables(conn)

    # 2. Model forecast log
    try:
        for model_name, prob in model_forecasts.items():
            conn.execute("""
                INSERT INTO model_forecast_log
                    (model_name, market_id, category, forecast_prob,
                     actual_outcome, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_name, market_id, category, prob, actual_outcome, now_iso))
    except sqlite3.OperationalError:
        pass

    # 3. Performance log
    try:
        conn.execute("""
            INSERT INTO performance_log
                (market_id, question, category, forecast_prob,
                 actual_outcome, edge_at_entry, confidence,
                 evidence_quality, stake_usd, entry_price,
                 exit_price, pnl, holding_hours, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id, question, category,
            forecast_prob, actual_outcome,
            edge_at_entry, confidence,
            evidence_quality, stake_usd,
            entry_price, exit_price,
            pnl, holding_hours, now_iso,
        ))
    except sqlite3.OperationalError:
        pass

    conn.commit()


def _create_calibration_tables(conn: sqlite3.Connection) -> None:
    """Create calibration feedback tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS calibration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_prob REAL NOT NULL,
            actual_outcome REAL NOT NULL,
            recorded_at TEXT NOT NULL,
            market_id TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_forecast_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            market_id TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT '',
            forecast_prob REAL NOT NULL,
            actual_outcome REAL NOT NULL,
            recorded_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS performance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            question TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL DEFAULT '',
            forecast_prob REAL NOT NULL,
            actual_outcome REAL NOT NULL,
            edge_at_entry REAL NOT NULL DEFAULT 0,
            confidence TEXT NOT NULL DEFAULT '',
            evidence_quality REAL NOT NULL DEFAULT 0,
            stake_usd REAL NOT NULL DEFAULT 0,
            entry_price REAL NOT NULL DEFAULT 0,
            exit_price REAL NOT NULL DEFAULT 0,
            pnl REAL NOT NULL DEFAULT 0,
            holding_hours REAL NOT NULL DEFAULT 0,
            resolved_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cal_history_market
            ON calibration_history(market_id);
        CREATE INDEX IF NOT EXISTS idx_model_log_market
            ON model_forecast_log(market_id);
        CREATE INDEX IF NOT EXISTS idx_model_log_category
            ON model_forecast_log(category);
        CREATE INDEX IF NOT EXISTS idx_perf_market
            ON performance_log(market_id);
    """)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Async helpers (for backward compatibility)
# ---------------------------------------------------------------------------

import asyncio
