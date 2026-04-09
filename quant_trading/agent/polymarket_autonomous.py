"""PolymarketAutonomous — Fully-autonomous Polymarket trading agent.

Fully-Autonomous-Polymarket-AI-Trading-Bot 架构的独立实现：

架构 (Architecture):
  1. MultiModelEnsembleTrader   — REST-only multi-model ensemble (GPT-4o + Claude + Gemini)
  2. PlattCalibratorV2          — Platt scaling with calibration feedback loop
  3. RiskGate15                 — 15+ pre-trade risk checks
  4. FractionalKellySizerV2    — Fractional Kelly with 7 adjustment factors
  5. DryRunSafetyGate           — 三阶段 dry-run 安全门
  6. SelfCalibratingTrader      — 自校准交易器：从结果中学习
  7. PolymarketAutonomous        — 主自主交易Agent

Usage:
    from quant_trading.agent.polymarket_autonomous import PolymarketAutonomous

    agent = PolymarketAutonomous(dry_run=True)
    decision = await agent.decide("Will ETH exceed $5000 by end of 2025?")
    if decision.allowed:
        await agent.execute(decision)
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS = ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"]
DEFAULT_WEIGHTS = {
    "gpt-4o": 0.40,
    "claude-3-5-sonnet-20241022": 0.35,
    "gemini-1.5-pro": 0.25,
}
DEFAULT_ENSEMBLE_TIMEOUT_SECS = 30


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Decision(str, Enum):
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


@dataclass
class ModelForecast:
    """单模型预测 / Forecast from a single model."""
    model_name: str
    probability: float           # 0.01–0.99
    confidence_level: str = "LOW"
    reasoning: str = ""
    invalidation_triggers: list[str] = field(default_factory=list)
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0


@dataclass
class EnsembleDecision:
    """多模型集成结果 / Aggregated ensemble result."""
    probability: float            # 集成后的校准概率
    confidence_level: str
    individual_forecasts: list[ModelForecast] = field(default_factory=list)
    models_succeeded: int = 0
    models_failed: int = 0
    aggregation_method: str = "trimmed_mean"
    spread: float = 0.0          # max - min 跨模型概率差异
    agreement_score: float = 0.0  # 1.0 = 完全一致
    reasoning: str = ""
    invalidation_triggers: list[str] = field(default_factory=list)
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    raw_calibration_adjustments: list[str] = field(default_factory=list)


@dataclass
class MarketSnapshot:
    """市场数据快照 / Market data snapshot for risk evaluation."""
    market_id: str = ""
    question: str = ""
    market_type: str = "UNKNOWN"
    category: str = ""
    event_slug: str = ""
    volume_usd: float = 0.0
    liquidity_usd: float = 0.0
    bid_depth_5: float = 0.0
    ask_depth_5: float = 0.0
    spread_pct: float = 0.0
    current_price: float = 0.5     # YES价格 (隐含概率)
    hours_to_resolution: float = 720.0
    is_near_resolution: bool = False
    evidence_quality: float = 0.5
    num_sources: int = 0
    has_clear_resolution: bool = True
    price_volatility: float = 0.0
    top_bullets: list[str] = field(default_factory=list)


@dataclass
class RiskCheckResult:
    """风险检查结果 / Result of a single risk check."""
    passed: bool
    check_name: str
    reason: str = ""
    severity: str = "block"   # 'block' | 'warn' | 'info'


@dataclass
class RiskGateResult:
    """15+风险检查汇总 / Full risk gate evaluation result."""
    allowed: bool
    decision: str
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    drawdown_heat: int = 0
    drawdown_pct: float = 0.0
    kelly_multiplier: float = 1.0
    portfolio_gate: str = "ok"
    safety_gate_stage: str = "passed"
    estimated_fees_pct: float = 0.0


@dataclass
class PositionSizeResult:
    """仓位计算结果 / Position sizing result."""
    stake_usd: float
    kelly_fraction_used: float
    full_kelly_stake: float
    capped_by: str
    direction: str              # "BUY_YES" | "BUY_NO"
    token_quantity: float
    base_kelly: float = 0.0
    confidence_mult: float = 1.0
    drawdown_mult: float = 1.0
    timeline_mult: float = 1.0
    volatility_mult: float = 1.0
    regime_mult: float = 1.0
    category_mult: float = 1.0
    liquidity_mult: float = 1.0
    estimated_cost: float = 0.0
    estimated_fees: float = 0.0


@dataclass
class CalibrationFeedback:
    """校准反馈记录 / Calibration feedback for self-improvement."""
    market_id: str
    forecast_prob: float
    actual_outcome: float       # 0.0 或 1.0
    model_forecasts: dict[str, float] = field(default_factory=dict)
    stake_usd: float = 0.0
    pnl: float = 0.0
    timestamp: str = ""


@dataclass
class TradeDecision:
    """最终交易决策 / Final autonomous trading decision."""
    market_id: str
    question: str
    direction: str              # "BUY_YES" | "BUY_NO"
    probability: float          # 集成概率
    calibrated_probability: float
    confidence_level: str
    edge: float                 # model_prob - implied_prob
    net_edge: float             # edge - fees
    stake_usd: float
    kelly_fraction: float
    kelly_multiplier: float
    estimated_cost: float
    estimated_pnl: float
    allowed: bool
    decision: str
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ensemble: EnsembleDecision | None = None
    risk_gate: RiskGateResult | None = None
    position_size: PositionSizeResult | None = None
    dry_run: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "direction": self.direction,
            "probability": round(self.probability, 4),
            "calibrated_probability": round(self.calibrated_probability, 4),
            "confidence_level": self.confidence_level,
            "edge": round(self.edge, 4),
            "net_edge": round(self.net_edge, 4),
            "stake_usd": round(self.stake_usd, 2),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "estimated_cost": round(self.estimated_cost, 2),
            "estimated_pnl": round(self.estimated_pnl, 2),
            "allowed": self.allowed,
            "decision": self.decision,
            "violations": self.violations,
            "warnings": self.warnings,
            "dry_run": self.dry_run,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# PlattCalibratorV2
# ---------------------------------------------------------------------------

class PlattCalibratorV2:
    """Platt Scaling 校准器 V2 — 从历史预测-结果对中学习。

    方法 (Method):
      calibrated = sigmoid(a * logit(raw) + b)
      其中 logit(x) = ln(x / (1-x))

    7个调整因子 (7 Adjustment Factors):
      1. 极值收缩 (Extremity shrinkage): 0.9倍的logit压缩
      2. 低证据惩罚 (Low evidence penalty): 证据质量<0.4时向0.5靠拢
      3. 矛盾惩罚 (Contradiction penalty): 每检测到一个矛盾信号
      4. 集成分歧惩罚 (Ensemble disagreement penalty): 模型间spread大时
      5. 历史校准 (Historical calibration): 有足够样本时用学习型参数
      6. 时间衰减调整 (Time decay adjustment): 远期市场降低信心
      7. 置信度水平映射 (Confidence level mapping): LOW→更保守

    无sklearn时使用网格搜索备选方案。
    """

    SHRINKAGE_A = 0.90   # 默认压缩因子

    def __init__(self, min_samples: int = 30):
        self._min_samples = min_samples
        self._a: float = self.SHRINKAGE_A
        self._b: float = 0.0
        self._is_fitted: bool = False
        self._n_samples: int = 0
        self._brier_score: float = 1.0
        self._history: list[tuple[float, float]] = []   # (prob, outcome)

    def record(self, probability: float, actual_outcome: float) -> None:
        """记录一个预测结果用于后续校准 / Record a forecast outcome."""
        self._history.append((
            max(0.01, min(0.99, probability)),
            float(actual_outcome),
        ))

    def fit(self) -> bool:
        """从记录的历史数据重新拟合 / Refit from recorded history."""
        if len(self._history) < self._min_samples:
            self._is_fitted = False
            self._n_samples = len(self._history)
            return False

        probs = [h[0] for h in self._history]
        outcomes = [h[1] for h in self._history]
        return self._fit_arrays(probs, outcomes)

    def _fit_arrays(self, probs: list[float], outcomes: list[float]) -> bool:
        """使用数组数据拟合 / Fit using array data."""
        if len(probs) < self._min_samples:
            return False

        try:
            import numpy as np

            logits = [math.log(p / (1.0 - p)) for p in probs]
            X = np.array(logits).reshape(-1, 1)
            y = np.array(outcomes)

            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(X, y)

            self._a = float(lr.coef_[0][0])
            self._b = float(lr.intercept_[0])
            self._is_fitted = True
            self._n_samples = len(probs)

            calibrated = [self._apply(p) for p in probs]
            self._brier_score = sum(
                (c - o) ** 2 for c, o in zip(calibrated, outcomes)
            ) / len(outcomes)
            return True

        except ImportError:
            return self._fit_manual(probs, outcomes)
        except Exception:
            return False

    def _fit_manual(self, probs: list[float], outcomes: list[float]) -> bool:
        """无sklearn时的网格搜索拟合 / Manual grid search fit without sklearn."""
        if len(probs) < 5:
            return False

        best_a, best_b, best_loss = 1.0, 0.0, float("inf")
        for a_c in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3]:
            for b_c in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                preds = [self._apply_v2(a_c, b_c, p) for p in probs]
                loss = sum((pred - y) ** 2 for pred, y in zip(preds, outcomes))
                if loss < best_loss:
                    best_loss = loss
                    best_a, best_b = a_c, b_c

        self._a, self._b = best_a, best_b
        self._is_fitted = True
        self._n_samples = len(probs)
        calibrated = [self._apply_v2(best_a, best_b, p) for p in probs]
        self._brier_score = sum(
            (c - o) ** 2 for c, o in zip(calibrated, outcomes)
        ) / len(outcomes)
        return True

    def _apply(self, prob: float) -> float:
        """应用已学习的Sigmoid变换 / Apply learned sigmoid."""
        prob = max(0.01, min(0.99, prob))
        logit = math.log(prob / (1.0 - prob))
        return 1.0 / (1.0 + math.exp(-(self._a * logit + self._b)))

    def _apply_v2(self, a: float, b: float, prob: float) -> float:
        """Apply sigmoid with given parameters."""
        prob = max(0.01, min(0.99, prob))
        logit = math.log(prob / (1.0 - prob))
        return 1.0 / (1.0 + math.exp(-(a * logit + b)))

    def calibrate(
        self,
        raw_prob: float,
        evidence_quality: float = 0.5,
        num_contradictions: int = 0,
        ensemble_spread: float = 0.0,
        confidence_level: str = "MEDIUM",
        low_evidence_penalty: float = 0.15,
    ) -> tuple[float, list[str]]:
        """校准概率，附详细调整记录 / Calibrate probability with adjustment log.

        Args:
            raw_prob: 原始概率 (0.01–0.99)
            evidence_quality: 证据质量 (0.0–1.0)
            num_contradictions: 检测到的矛盾信号数量
            ensemble_spread: 模型间概率差异
            confidence_level: LOW / MEDIUM / HIGH
            low_evidence_penalty: 低证据惩罚强度

        Returns:
            (calibrated_probability, adjustments_log)
        """
        adjustments: list[str] = []
        p = max(0.01, min(0.99, raw_prob))

        # 1. Platt / historical calibration
        if self._is_fitted:
            p_cal = self._apply(p)
        else:
            logit = math.log(p / (1.0 - p))
            p_cal = 1.0 / (1.0 + math.exp(-(self.SHRINKAGE_A * logit)))

        if abs(p_cal - p) > 0.005:
            adjustments.append(f"platt: {p:.3f}→{p_cal:.3f}")
            p = p_cal

        # 2. Low evidence penalty
        if evidence_quality < 0.4:
            penalty = low_evidence_penalty * (1.0 - evidence_quality)
            p_pen = p * (1.0 - penalty) + 0.5 * penalty
            adjustments.append(f"low_evidence(q={evidence_quality:.2f}): {p:.3f}→{p_pen:.3f}")
            p = p_pen

        # 3. Contradiction penalty
        if num_contradictions > 0:
            penalty = min(0.3, 0.1 * num_contradictions)
            p_con = p * (1.0 - penalty) + 0.5 * penalty
            adjustments.append(f"contradictions(n={num_contradictions}): {p:.3f}→{p_con:.3f}")
            p = p_con

        # 4. Ensemble disagreement penalty
        if ensemble_spread > 0.10:
            penalty = min(0.25, ensemble_spread)
            p_spread = p * (1.0 - penalty) + 0.5 * penalty
            adjustments.append(f"ensemble_spread(s={ensemble_spread:.2f}): {p:.3f}→{p_spread:.3f}")
            p = p_spread

        # 5. Confidence level mapping
        if confidence_level == "LOW":
            p = p * 0.9 + 0.5 * 0.1
            adjustments.append(f"low_confidence_mapping: →{p:.3f}")
        elif confidence_level == "HIGH":
            adjustments.append("high_confidence: no adjustment")

        p = max(0.01, min(0.99, p))
        return p, adjustments

    @property
    def stats(self) -> dict[str, Any]:
        """返回校准统计 / Return calibration statistics."""
        return {
            "is_fitted": self._is_fitted,
            "n_samples": self._n_samples,
            "min_samples_required": self._min_samples,
            "a": round(self._a, 4),
            "b": round(self._b, 4),
            "brier_score": round(self._brier_score, 4),
        }

    def is_reliable(self) -> bool:
        """是否可靠 / Whether calibration is reliable."""
        return self._is_fitted and self._n_samples >= self._min_samples


# ---------------------------------------------------------------------------
# SelfCalibratingTrader — 自校准交易器
# ---------------------------------------------------------------------------

class SelfCalibratingTrader:
    """自校准交易器：从市场结果中学习，持续改进预测质量。

    Self-calibrating trader that:
      1. Records every forecast → outcome pair
      2. Retrains PlattCalibrator when enough new data accumulates
      3. Tracks per-model Brier scores for adaptive weighting
      4. Provides self-improvement metrics

    Usage:
        trader = SelfCalibratingTrader()
        trader.record_outcome(market_id="...", forecast_prob=0.72, actual_outcome=1.0)
        if trader.should_retrain():
            trader.retrain()
    """

    RETRAIN_INTERVAL = 10   # 每10个新结果触发一次重训练

    def __init__(self, calibrator: PlattCalibratorV2 | None = None):
        self._calibrator = calibrator or PlattCalibratorV2(min_samples=30)
        self._pending_records: list[CalibrationFeedback] = []
        self._model_history: dict[str, list[tuple[float, float]]] = {}  # model → [(prob, outcome)]
        self._total_recorded: int = 0
        self._last_retrain_n: int = 0

    def record_outcome(
        self,
        market_id: str,
        forecast_prob: float,
        actual_outcome: float,
        model_forecasts: dict[str, float] | None = None,
        stake_usd: float = 0.0,
        pnl: float = 0.0,
    ) -> None:
        """记录市场结果用于自学习 / Record a market resolution for learning."""
        feedback = CalibrationFeedback(
            market_id=market_id,
            forecast_prob=forecast_prob,
            actual_outcome=actual_outcome,
            model_forecasts=model_forecasts or {},
            stake_usd=stake_usd,
            pnl=pnl,
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        self._pending_records.append(feedback)
        self._total_recorded += 1

        # 更新模型历史
        if model_forecasts:
            for model_name, prob in model_forecasts.items():
                if model_name not in self._model_history:
                    self._model_history[model_name] = []
                self._model_history[model_name].append((prob, actual_outcome))

        # 更新全局校准器
        self._calibrator.record(forecast_prob, actual_outcome)

    def should_retrain(self) -> bool:
        """是否应该重训练 / Whether calibrator should be retrained."""
        return (
            len(self._pending_records) >= self.RETRAIN_INTERVAL
            and len(self._pending_records) - self._last_retrain_n >= self.RETRAIN_INTERVAL
        )

    def retrain(self) -> dict[str, Any]:
        """重训练校准器 / Retrain the calibrator from all history."""
        self._last_retrain_n = len(self._pending_records)
        success = self._calibrator.fit()
        stats = self._calibrator.stats
        stats["success"] = success
        stats["total_recorded"] = self._total_recorded
        return stats

    def get_model_brier_scores(self) -> dict[str, float]:
        """获取各模型的Brier分数 / Get Brier scores per model."""
        scores: dict[str, float] = {}
        for model, history in self._model_history.items():
            if len(history) < 3:
                continue
            brier = sum((p - o) ** 2 for p, o in history) / len(history)
            scores[model] = round(brier, 4)
        return scores

    def get_adaptive_weights(
        self,
        default_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """基于历史表现计算自适应权重 / Compute adaptive weights from historical performance."""
        default_weights = default_weights or DEFAULT_WEIGHTS
        brier_scores = self.get_model_brier_scores()

        if not brier_scores:
            return dict(default_weights)

        # 逆Brier加权：低Brier = 高权重
        inv_brier: dict[str, float] = {}
        for model, brier in brier_scores.items():
            inv_brier[model] = 1.0 / max(brier, 0.001)

        # 与默认权重混合
        total = sum(inv_brier.values())
        blended: dict[str, float] = {}
        for model in default_weights:
            default_w = default_weights.get(model, 1.0 / len(default_weights))
            if model in inv_brier:
                # 混合因子：基于样本数量
                n = len(self._model_history.get(model, []))
                blend = min(0.8, n / 50.0)
                blended[model] = blend * (inv_brier[model] / total) + (1 - blend) * default_w
            else:
                blended[model] = default_w

        # 归一化
        total_blend = sum(blended.values())
        if total_blend > 0:
            blended = {k: v / total_blend for k, v in blended.items()}

        return blended

    def flush_pending(self) -> list[CalibrationFeedback]:
        """清空并返回待处理记录 / Flush and return pending records."""
        records = list(self._pending_records)
        self._pending_records.clear()
        return records

    @property
    def calibrator(self) -> PlattCalibratorV2:
        return self._calibrator

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_recorded": self._total_recorded,
            "pending_records": len(self._pending_records),
            "models_tracked": list(self._model_history.keys()),
            "calibrator": self._calibrator.stats,
        }


# ---------------------------------------------------------------------------
# MultiModelEnsembleTrader — REST-only multi-model ensemble
# ---------------------------------------------------------------------------

class MultiModelEnsembleTrader:
    """REST-only 多模型集成交易器 (GPT-4o + Claude 3.5 + Gemini 1.5)。

    通过 urllib 发送REST请求，无 SDK 依赖。

    集成方法:
      - trimmed_mean: 去除最高最低后平均
      - median: 中位数
      - weighted: 基于自适应权重的加权平均

    三模型默认权重:
      GPT-4o:           40%
      Claude 3.5 Sonnet: 35%
      Gemini 1.5 Pro:    25%
    """

    FORECAST_PROMPT_TEMPLATE = """\
You are an expert probabilistic forecaster analyzing a prediction market.

MARKET QUESTION: {question}
MARKET TYPE: {market_type}

EVIDENCE SUMMARY:
{evidence_summary}

TOP EVIDENCE BULLETS:
{evidence_bullets}

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
  "reasoning": "2-4 sentence explanation of your estimate",
  "invalidation_triggers": ["specific event that would change this forecast"],
  "key_evidence": [{{"text": "evidence bullet", "source": "publisher", "impact": "supports/opposes/neutral"}}]
}}

RULES:
- Probability must be between 0.01 and 0.99.
- If evidence is weak (quality < 0.3), bias toward 0.50.
- If evidence contradicts itself, widen uncertainty toward 0.50.
- Never claim certainty.
- Do NOT hallucinate data.

Return ONLY valid JSON, no markdown fences."""

    def __init__(
        self,
        models: list[str] | None = None,
        weights: dict[str, float] | None = None,
        aggregation: str = "trimmed_mean",
        trim_fraction: float = 0.1,
        timeout_secs: int = DEFAULT_ENSEMBLE_TIMEOUT_SECS,
    ):
        self.models = models or DEFAULT_MODELS.copy()
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.aggregation = aggregation
        self.trim_fraction = trim_fraction
        self.timeout_secs = timeout_secs

    async def forecast(
        self,
        question: str,
        market_type: str = "UNKNOWN",
        evidence_bullets: list[str] | None = None,
        volume_usd: float = 0.0,
        liquidity_usd: float = 0.0,
        spread_pct: float = 0.0,
        days_to_expiry: float = 30.0,
        price_momentum: float = 0.0,
        evidence_quality: float = 0.5,
        num_sources: int = 0,
        evidence_summary: str = "",
        adaptive_weights: dict[str, float] | None = None,
    ) -> EnsembleDecision:
        """执行多模型集成预测 / Run multi-model ensemble forecasting.

        Args:
            question: 市场问题 / Market question
            market_type: 市场类型 / Market type classification
            evidence_bullets: 关键证据列表 / Key evidence bullets
            volume_usd: 成交量(USD) / Volume in USD
            liquidity_usd: 流动性(USD) / Liquidity in USD
            spread_pct: 买卖价差百分比 / Bid-ask spread as decimal
            days_to_expiry: 到期天数 / Days until expiry
            price_momentum: 24h价格动量 / 24h price momentum
            evidence_quality: 证据质量 0-1 / Evidence quality 0-1
            num_sources: 来源数量 / Number of sources
            evidence_summary: 证据摘要 / Evidence summary text
            adaptive_weights: 自适应权重覆盖 / Override weights for adaptive learning

        Returns:
            EnsembleDecision with aggregated forecast
        """
        if adaptive_weights:
            self.weights = dict(adaptive_weights)

        prompt = self._build_prompt(
            question=question,
            market_type=market_type,
            evidence_bullets=evidence_bullets or [],
            evidence_summary=evidence_summary,
            volume_usd=volume_usd,
            liquidity_usd=liquidity_usd,
            spread_pct=spread_pct,
            days_to_expiry=days_to_expiry,
            price_momentum=price_momentum,
            evidence_quality=evidence_quality,
            num_sources=num_sources,
        )

        # 并行查询所有模型
        tasks = [self._query_model(model, prompt) for model in self.models]
        forecasts = await asyncio.gather(*tasks, return_exceptions=True)

        # 解析结果
        valid: list[ModelForecast] = []
        errors: list[ModelForecast] = []
        for f in forecasts:
            if isinstance(f, Exception):
                errors.append(ModelForecast(model_name="unknown", probability=0.5, error=str(f)))
            elif isinstance(f, ModelForecast):
                if f.error:
                    errors.append(f)
                else:
                    valid.append(f)

        if not valid:
            return EnsembleDecision(
                probability=0.5,
                confidence_level="LOW",
                models_succeeded=0,
                models_failed=len(errors),
            )

        # 聚合概率
        agg_prob, spread = self._aggregate([(f.model_name, f.probability) for f in valid])
        agreement = max(0.0, 1.0 - spread * 2.0)

        # 聚合置信度
        conf_level = self._aggregate_confidence(valid)
        if spread > 0.15:
            conf_level = "LOW"

        # 合并推理和触发器
        reasoning = " | ".join(f.reasoning for f in valid[:3] if f.reasoning)
        all_triggers = self._merge_triggers(valid)
        all_evidence = self._merge_evidence(valid)

        return EnsembleDecision(
            probability=agg_prob,
            confidence_level=conf_level,
            individual_forecasts=valid,
            models_succeeded=len(valid),
            models_failed=len(errors),
            aggregation_method=self.aggregation,
            spread=spread,
            agreement_score=agreement,
            reasoning=reasoning,
            invalidation_triggers=all_triggers[:5],
            key_evidence=all_evidence[:8],
        )

    def _build_prompt(
        self,
        question: str,
        market_type: str,
        evidence_bullets: list[str],
        evidence_summary: str,
        volume_usd: float,
        liquidity_usd: float,
        spread_pct: float,
        days_to_expiry: float,
        price_momentum: float,
        evidence_quality: float,
        num_sources: int,
    ) -> str:
        bullets_text = "\n".join(f"- {b}" for b in evidence_bullets) if evidence_bullets else "No evidence available."
        return self.FORECAST_PROMPT_TEMPLATE.format(
            question=question,
            market_type=market_type,
            evidence_summary=evidence_summary or "No summary available.",
            evidence_bullets=bullets_text,
            volume_usd=volume_usd,
            liquidity_usd=liquidity_usd,
            spread_pct=spread_pct,
            days_to_expiry=days_to_expiry,
            price_momentum=price_momentum,
            evidence_quality=evidence_quality,
            num_sources=num_sources,
        )

    async def _query_model(self, model: str, prompt: str) -> ModelForecast:
        """通过REST API查询单个模型 / Query a single model via REST API."""
        start = time.monotonic()
        provider = self._get_provider(model)

        try:
            if provider == "openai":
                return await self._query_openai(model, prompt)
            elif provider == "anthropic":
                return await self._query_anthropic(model, prompt)
            elif provider == "google":
                return await self._query_google(model, prompt)
            else:
                return ModelForecast(model_name=model, probability=0.5, error=f"Unknown provider: {provider}")
        except Exception as e:
            return ModelForecast(
                model_name=model,
                probability=0.5,
                error=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )

    def _get_provider(self, model: str) -> str:
        """识别模型提供商 / Identify model provider."""
        m = model.lower()
        if "claude" in m:
            return "anthropic"
        elif "gemini" in m:
            return "google"
        return "openai"

    async def _query_openai(self, model: str, prompt: str) -> ModelForecast:
        """Query OpenAI GPT-4o via REST."""
        import os
        start = time.monotonic()
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return ModelForecast(model_name=model, probability=0.5, error="OPENAI_API_KEY not set")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a calibrated probabilistic forecaster. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with asyncio.timeout(self.timeout_secs):
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(
                    None,
                    lambda: urllib.request.urlopen(req, timeout=self.timeout_secs).read()
                )
            resp = json.loads(raw.decode("utf-8"))
            content = resp["choices"][0]["message"]["content"]
            return self._parse_response(model, content, start)
        except asyncio.TimeoutError:
            return ModelForecast(model_name=model, probability=0.5, error="Timeout", latency_ms=(time.monotonic() - start) * 1000)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            return ModelForecast(model_name=model, probability=0.5, error=f"HTTP {e.code}: {body}", latency_ms=(time.monotonic() - start) * 1000)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return ModelForecast(model_name=model, probability=0.5, error=f"Parse error: {e}", latency_ms=(time.monotonic() - start) * 1000)

    async def _query_anthropic(self, model: str, prompt: str) -> ModelForecast:
        """Query Anthropic Claude via REST."""
        import os
        start = time.monotonic()
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return ModelForecast(model_name=model, probability=0.5, error="ANTHROPIC_API_KEY not set")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "system": "You are a calibrated probabilistic forecaster. Return only valid JSON.",
            "messages": [{"role": "user", "content": prompt}],
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with asyncio.timeout(self.timeout_secs):
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(
                    None,
                    lambda: urllib.request.urlopen(req, timeout=self.timeout_secs).read()
                )
            resp = json.loads(raw.decode("utf-8"))
            content = resp["content"][0]["text"]
            return self._parse_response(model, content, start)
        except asyncio.TimeoutError:
            return ModelForecast(model_name=model, probability=0.5, error="Timeout", latency_ms=(time.monotonic() - start) * 1000)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            return ModelForecast(model_name=model, probability=0.5, error=f"HTTP {e.code}: {body}", latency_ms=(time.monotonic() - start) * 1000)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return ModelForecast(model_name=model, probability=0.5, error=f"Parse error: {e}", latency_ms=(time.monotonic() - start) * 1000)

    async def _query_google(self, model: str, prompt: str) -> ModelForecast:
        """Query Google Gemini via REST."""
        import os
        start = time.monotonic()
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return ModelForecast(model_name=model, probability=0.5, error="GOOGLE_API_KEY not set")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": f"You are a calibrated probabilistic forecaster. Return only valid JSON.\n\n{prompt}"}]}],
            "generation_config": {"temperature": 0.2, "maxOutputTokens": 1024},
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with asyncio.timeout(self.timeout_secs):
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(
                    None,
                    lambda: urllib.request.urlopen(req, timeout=self.timeout_secs).read()
                )
            resp = json.loads(raw.decode("utf-8"))
            content = resp["candidates"][0]["content"]["parts"][0]["text"]
            return self._parse_response(model, content, start)
        except asyncio.TimeoutError:
            return ModelForecast(model_name=model, probability=0.5, error="Timeout", latency_ms=(time.monotonic() - start) * 1000)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            return ModelForecast(model_name=model, probability=0.5, error=f"HTTP {e.code}: {body}", latency_ms=(time.monotonic() - start) * 1000)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return ModelForecast(model_name=model, probability=0.5, error=f"Parse error: {e}", latency_ms=(time.monotonic() - start) * 1000)

    def _parse_response(self, model: str, raw_text: str, start: float) -> ModelForecast:
        """解析LLM JSON响应 / Parse LLM JSON response."""
        raw_text = raw_text.strip()
        # Strip markdown fences
        if raw_text.startswith("```"):
            lines = raw_text.split("\n", 1)
            if len(lines) > 1:
                raw_text = lines[1]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return ModelForecast(
                model_name=model,
                probability=0.5,
                error=f"JSON decode failed: {raw_text[:100]}",
                latency_ms=(time.monotonic() - start) * 1000,
            )

        prob = max(0.01, min(0.99, float(parsed.get("probability", 0.5))))
        conf = parsed.get("confidence_level", "LOW")
        if conf not in ("LOW", "MEDIUM", "HIGH"):
            conf = "LOW"

        return ModelForecast(
            model_name=model,
            probability=prob,
            confidence_level=conf,
            reasoning=parsed.get("reasoning", ""),
            invalidation_triggers=parsed.get("invalidation_triggers", []),
            key_evidence=parsed.get("key_evidence", []),
            raw_response=parsed,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    def _aggregate(
        self,
        model_probs: list[tuple[str, float]],
    ) -> tuple[float, float]:
        """聚合多个模型的概率 / Aggregate model probabilities."""
        if not model_probs:
            return 0.5, 0.0

        probs = [p for _, p in model_probs]
        if len(probs) == 1:
            return probs[0], 0.0

        if self.aggregation == "median":
            sorted_p = sorted(probs)
            mid = len(sorted_p) // 2
            if len(sorted_p) % 2 == 0:
                agg = (sorted_p[mid - 1] + sorted_p[mid]) / 2.0
            else:
                agg = sorted_p[mid]

        elif self.aggregation == "weighted":
            total_w = 0.0
            weighted_sum = 0.0
            for name, p in model_probs:
                w = self.weights.get(name, 1.0 / len(model_probs))
                weighted_sum += p * w
                total_w += w
            agg = weighted_sum / total_w if total_w > 0 else 0.5

        else:  # trimmed_mean (default)
            if len(probs) <= 2:
                agg = sum(probs) / len(probs)
            else:
                sorted_p = sorted(probs)
                trim = max(1, int(len(sorted_p) * self.trim_fraction))
                trimmed = sorted_p[trim:-trim] if trim < len(sorted_p) // 2 else sorted_p
                agg = sum(trimmed) / len(trimmed) if trimmed else sum(probs) / len(probs)

        spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
        return agg, spread

    def _aggregate_confidence(self, forecasts: list[ModelForecast]) -> str:
        """聚合置信度水平 / Aggregate confidence levels."""
        if not forecasts:
            return "LOW"
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        values = [order.get(f.confidence_level, 0) for f in forecasts]
        avg = sum(values) / len(values)
        if avg >= 1.5:
            return "HIGH"
        elif avg >= 0.5:
            return "MEDIUM"
        return "LOW"

    def _merge_triggers(self, forecasts: list[ModelForecast]) -> list[str]:
        """合并去重的失效触发器 / Merge deduplicated invalidation triggers."""
        seen: set[str] = set()
        result: list[str] = []
        for f in forecasts:
            for t in f.invalidation_triggers:
                key = t.lower()
                if key not in seen:
                    seen.add(key)
                    result.append(t)
        return result

    def _merge_evidence(self, forecasts: list[ModelForecast]) -> list[dict[str, Any]]:
        """合并去重的关键证据 / Merge deduplicated key evidence."""
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for f in forecasts:
            for ev in f.key_evidence:
                text = ev.get("text", "")
                if text and text not in seen:
                    seen.add(text)
                    result.append(ev)
        return result


# ---------------------------------------------------------------------------
# FractionalKellySizerV2 — 7因子Fractional Kelly仓位计算器
# ---------------------------------------------------------------------------

class FractionalKellySizerV2:
    """Fractional Kelly 仓位计算器 V2 — 7个调整因子。

    Kelly公式 (二进制结果):
      f* = (p * b - q) / b
    其中:
      p = 获胜概率
      b = 赔率 (payout/cost - 1)
      q = 1 - p

    7个调整因子 (7 Adjustment Factors):
      1. Volatility (波动率): 高波动市场降低仓位
      2. Regime (市场状态): NORMAL/TRENDING/MEAN_REVERTING/HIGH_VOL调整
      3. Confidence (置信度): LOW=0.5, MEDIUM=0.75, HIGH=1.0
      4. Timeline (时间线): 临近 resolution 增加, 远期减少
      5. Evidence (证据): 低证据质量时降低
      6. Liquidity (流动性): 从不超过可流动性的X%
      7. Correlation (相关性): 同一类别仓位过多时减少
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_stake_per_market: float = 50.0,
        max_bankroll_fraction: float = 0.05,
        min_stake_usd: float = 1.0,
        bankroll: float = 5000.0,
        volatility_high_threshold: float = 0.15,
        volatility_med_threshold: float = 0.10,
        volatility_high_min_mult: float = 0.4,
        volatility_med_min_mult: float = 0.6,
        max_liquidity_pct: float = 0.05,
        transaction_fee_pct: float = 0.02,
        exit_fee_pct: float = 0.02,
        gas_cost_usd: float = 0.01,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_stake_per_market = max_stake_per_market
        self.max_bankroll_fraction = max_bankroll_fraction
        self.min_stake_usd = min_stake_usd
        self.bankroll = bankroll
        self.volatility_high_threshold = volatility_high_threshold
        self.volatility_med_threshold = volatility_med_threshold
        self.volatility_high_min_mult = volatility_high_min_mult
        self.volatility_med_min_mult = volatility_med_min_mult
        self.max_liquidity_pct = max_liquidity_pct
        self.transaction_fee_pct = transaction_fee_pct
        self.exit_fee_pct = exit_fee_pct
        self.gas_cost_usd = gas_cost_usd
        self._CONF_MULT = {"LOW": 0.5, "MEDIUM": 0.75, "HIGH": 1.0}
        self._REGIME_MULT = {
            "NORMAL": 1.0,
            "TRENDING": 1.15,
            "MEAN_REVERTING": 0.9,
            "HIGH_VOLATILITY": 0.6,
            "LOW_ACTIVITY": 0.8,
        }

    def compute_size(
        self,
        model_probability: float,
        implied_probability: float,
        direction: str,
        confidence_level: str = "MEDIUM",
        regime: str = "NORMAL",
        price_volatility: float = 0.0,
        hours_to_resolution: float = 720.0,
        evidence_quality: float = 0.5,
        liquidity_usd: float = 0.0,
        category: str = "",
        category_mult: float = 1.0,
        drawdown_mult: float = 1.0,
        num_correlated_positions: int = 0,
    ) -> PositionSizeResult:
        """计算Fractional Kelly最优仓位 / Compute optimal position size.

        Args:
            model_probability: 模型估计的概率 / Model estimated probability
            implied_probability: 市场隐含概率 / Market implied probability
            direction: "BUY_YES" 或 "BUY_NO" / Trade direction
            confidence_level: LOW / MEDIUM / HIGH
            regime: NORMAL / TRENDING / MEAN_REVERTING / HIGH_VOLATILITY / LOW_ACTIVITY
            price_volatility: 价格波动率 0-1 / Price volatility 0-1
            hours_to_resolution: 距离到期的小时数 / Hours until resolution
            evidence_quality: 证据质量 0-1 / Evidence quality 0-1
            liquidity_usd: 可用流动性(USD) / Available liquidity in USD
            category: 市场类别 / Market category
            category_mult: 类别乘数 / Category multiplier
            drawdown_mult: 回撤乘数 0-1 / Drawdown multiplier 0-1
            num_correlated_positions: 同类别已有仓位数量 / Existing positions in same category

        Returns:
            PositionSizeResult with full breakdown
        """
        # 胜率计算
        if direction == "BUY_YES":
            p = model_probability
            cost = implied_probability
        else:
            p = 1.0 - model_probability
            cost = 1.0 - implied_probability

        cost = max(0.01, min(0.99, cost))
        b = (1.0 / cost) - 1.0   # 赔率 / odds
        q = 1.0 - p

        # 基础Kelly
        if b > 0:
            full_kelly = max(0.0, (p * b - q) / b)
        else:
            full_kelly = 0.0
        base_kelly = full_kelly

        # ── 7个调整因子 ────────────────────────────────────────────────

        # Factor 3: Confidence multiplier
        conf_mult = self._CONF_MULT.get(confidence_level, 0.75)

        # Factor 2: Regime multiplier
        regime_mult = self._REGIME_MULT.get(regime, 1.0)

        # Factor 1: Volatility adjustment
        vol_mult = 1.0
        if price_volatility > self.volatility_high_threshold:
            vol_mult = max(
                self.volatility_high_min_mult,
                1.0 - (price_volatility - self.volatility_high_threshold) * 2,
            )
        elif price_volatility > self.volatility_med_threshold:
            vol_mult = max(
                self.volatility_med_min_mult,
                1.0 - (price_volatility - self.volatility_med_threshold) * 3,
            )

        # Factor 4: Timeline adjustment
        timeline_mult = 1.0
        near_hours = 48
        if hours_to_resolution <= near_hours:
            timeline_mult = 1.15   # 临近resolution略微增加
        elif hours_to_resolution > near_hours * 10:
            timeline_mult = 1.1    # 远期略微增加
        elif hours_to_resolution > 60 * 24:
            timeline_mult = 0.9    # 超远期减少

        # Factor 5: Evidence quality adjustment
        evidence_mult = 1.0
        if evidence_quality < 0.4:
            evidence_mult = 0.7
        elif evidence_quality < 0.55:
            evidence_mult = 0.85

        # Factor 7: Correlation adjustment
        corr_mult = 1.0
        if num_correlated_positions >= 3:
            corr_mult = 0.7
        elif num_correlated_positions >= 2:
            corr_mult = 0.85

        # 合并 Kelly fraction
        adj_kelly = (
            self.kelly_fraction
            * conf_mult
            * regime_mult
            * vol_mult
            * timeline_mult
            * evidence_mult
            * category_mult
            * drawdown_mult
            * corr_mult
        )
        full_kelly_stake = adj_kelly * self.bankroll

        # Factor 6: Liquidity cap
        if liquidity_usd > 0:
            max_liquidity = liquidity_usd * self.max_liquidity_pct
        else:
            max_liquidity = float("inf")

        # 应用硬上限
        max_bankroll = self.max_bankroll_fraction * self.bankroll
        stake = min(full_kelly_stake, self.max_stake_per_market, max_bankroll, max_liquidity)
        stake = max(0.0, stake)

        # 最小仓位
        if 0 < stake < self.min_stake_usd:
            stake = 0.0

        # 确定上限来源
        if stake == 0.0 and full_kelly_stake > 0:
            capped_by = "min_stake"
        elif stake >= self.max_stake_per_market - 0.01:
            capped_by = "max_stake"
        elif stake >= max_bankroll - 0.01:
            capped_by = "max_bankroll"
        elif liquidity_usd > 0 and stake >= max_liquidity - 0.01:
            capped_by = "liquidity"
        else:
            capped_by = "kelly"

        # 代币数量
        token_qty = stake / cost if cost > 0 else 0.0

        # 费用计算
        estimated_cost = stake
        estimated_fees = stake * (self.transaction_fee_pct + self.exit_fee_pct)

        return PositionSizeResult(
            stake_usd=round(stake, 2),
            kelly_fraction_used=round(adj_kelly, 4),
            full_kelly_stake=round(full_kelly_stake, 2),
            capped_by=capped_by,
            direction=direction,
            token_quantity=round(token_qty, 2),
            base_kelly=round(base_kelly, 4),
            confidence_mult=round(conf_mult, 2),
            drawdown_mult=round(drawdown_mult, 2),
            timeline_mult=round(timeline_mult, 2),
            volatility_mult=round(vol_mult, 2),
            regime_mult=round(regime_mult, 2),
            category_mult=round(category_mult, 2),
            liquidity_mult=round(min(1.0, max_liquidity / full_kelly_stake) if full_kelly_stake > 0 else 1.0, 2),
            estimated_cost=round(estimated_cost, 2),
            estimated_fees=round(estimated_fees, 2),
        )


# ---------------------------------------------------------------------------
# RiskGate15 — 15+预交易风险检查
# ---------------------------------------------------------------------------

class RiskGate15:
    """15+预交易风险检查器 / 15+ pre-trade risk checks.

    Checks:
      1.  Kill switch (manual)
      2.  Drawdown kill switch (auto)
      3.  Maximum daily loss
      4.  Maximum open positions
      5.  Minimum edge threshold (net edge after fees)
      6.  Minimum liquidity
      7.  Maximum spread
      8.  Evidence quality
      9.  Confidence level (reject LOW)
      10. Minimum implied probability floor
      11. Edge direction (positive after costs)
      12. Market type check
      13. Portfolio category exposure
      14. Portfolio event exposure
      15. Timeline endgame check
      16. Minimum stake floor

    Any BLOCK-severity failure prevents the trade.
    """

    def __init__(
        self,
        max_stake_per_market: float = 50.0,
        max_daily_loss: float = 500.0,
        max_open_positions: int = 25,
        min_edge: float = 0.04,
        min_liquidity: float = 2000.0,
        min_volume: float = 1000.0,
        max_spread: float = 0.06,
        kelly_fraction: float = 0.25,
        max_bankroll_fraction: float = 0.05,
        kill_switch: bool = False,
        bankroll: float = 5000.0,
        transaction_fee_pct: float = 0.02,
        exit_fee_pct: float = 0.02,
        min_implied_probability: float = 0.05,
        max_drawdown_pct: float = 0.20,
        max_category_exposure_pct: float = 0.35,
        max_single_event_exposure_pct: float = 0.25,
        max_correlated_positions: int = 4,
    ):
        self.max_stake_per_market = max_stake_per_market
        self.max_daily_loss = max_daily_loss
        self.max_open_positions = max_open_positions
        self.min_edge = min_edge
        self.min_liquidity = min_liquidity
        self.min_volume = min_volume
        self.max_spread = max_spread
        self.kelly_fraction = kelly_fraction
        self.max_bankroll_fraction = max_bankroll_fraction
        self.kill_switch = kill_switch
        self.bankroll = bankroll
        self.transaction_fee_pct = transaction_fee_pct
        self.exit_fee_pct = exit_fee_pct
        self.min_implied_probability = min_implied_probability
        self.max_drawdown_pct = max_drawdown_pct
        self.max_category_exposure_pct = max_category_exposure_pct
        self.max_single_event_exposure_pct = max_single_event_exposure_pct
        self.max_correlated_positions = max_correlated_positions

    def evaluate(
        self,
        edge: float,
        net_edge: float,
        implied_probability: float,
        confidence_level: str,
        market: MarketSnapshot,
        portfolio: dict[str, Any],
    ) -> RiskGateResult:
        """评估15+风险检查 / Evaluate all 15+ risk checks.

        Args:
            edge: Raw edge (model_prob - implied_prob)
            net_edge: Edge after fees
            implied_probability: Market price as probability
            confidence_level: LOW / MEDIUM / HIGH
            market: Market data snapshot
            portfolio: Portfolio state dict with keys:
                - daily_pnl: float
                - open_position_count: int
                - category_exposures: dict[str, float]
                - event_exposures: dict[str, float]
                - positions: list[dict]

        Returns:
            RiskGateResult with decision and detailed breakdown
        """
        violations: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []
        heat_level = 0
        dd_pct = 0.0
        kelly_mult = 1.0

        # ── Drawdown assessment ───────────────────────────────────────
        daily_pnl = portfolio.get("daily_pnl", 0.0)
        dd_pct = abs(daily_pnl) / max(self.bankroll, 1.0)
        if dd_pct >= self.max_drawdown_pct:
            heat_level = 2
            kelly_mult = 0.0
        elif dd_pct >= self.max_drawdown_pct * 0.75:
            heat_level = 2
            kelly_mult = 0.25
        elif dd_pct >= self.max_drawdown_pct * 0.5:
            heat_level = 1
            kelly_mult = 0.5

        # 1. Kill switch
        if self.kill_switch:
            violations.append("KILL_SWITCH: Trading disabled")
        else:
            passed.append("kill_switch: OK")

        # 2. Drawdown kill
        if heat_level >= 2 and kelly_mult <= 0:
            violations.append(f"DRAWDOWN_KILL: {dd_pct:.1%} >= {self.max_drawdown_pct:.0%}")
        elif heat_level >= 1:
            warnings.append(f"DRAWDOWN_HEAT: level={heat_level}, kelly_mult={kelly_mult:.2f}")
        else:
            passed.append("drawdown: healthy")

        # 3. Maximum daily loss
        if daily_pnl < 0:
            loss = abs(daily_pnl)
            if loss >= self.max_daily_loss:
                violations.append(f"MAX_DAILY_LOSS: ${loss:.2f} >= ${self.max_daily_loss:.2f}")
            else:
                passed.append(f"daily_loss: ${loss:.2f} < limit")
        else:
            passed.append(f"daily_pnl: +${daily_pnl:.2f}")

        # 4. Maximum open positions
        open_count = portfolio.get("open_position_count", 0)
        if open_count >= self.max_open_positions:
            violations.append(f"MAX_POSITIONS: {open_count} >= {self.max_open_positions}")
        else:
            passed.append(f"open_positions: {open_count} < limit")

        # 5. Minimum edge
        if abs(net_edge) < self.min_edge:
            violations.append(f"MIN_EDGE: |{net_edge:.4f}| < {self.min_edge}")
        else:
            passed.append(f"min_edge: {abs(net_edge):.4f} >= {self.min_edge}")

        # 6. Minimum liquidity
        total_depth = market.bid_depth_5 + market.ask_depth_5
        if total_depth > 0 and total_depth < self.min_liquidity:
            violations.append(f"MIN_LIQUIDITY: ${total_depth:.2f} < ${self.min_liquidity:.2f}")
        elif total_depth > 0:
            passed.append(f"liquidity: ${total_depth:.2f} >= ${self.min_liquidity:.2f}")
        else:
            warnings.append("LIQUIDITY: No depth data")

        # 7. Maximum spread
        if market.spread_pct > 0 and market.spread_pct > self.max_spread:
            violations.append(f"MAX_SPREAD: {market.spread_pct:.2%} > {self.max_spread:.2%}")
        elif market.spread_pct > 0:
            passed.append(f"spread: {market.spread_pct:.2%} <= {self.max_spread:.2%}")

        # 8. Evidence quality
        if market.evidence_quality < 0.55:
            violations.append(f"EVIDENCE_QUALITY: {market.evidence_quality:.2f} < 0.55")
        else:
            passed.append(f"evidence_quality: {market.evidence_quality:.2f} >= 0.55")

        # 9. Confidence level
        _CONF_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if _CONF_RANK.get(confidence_level, 0) < _CONF_RANK.get("MEDIUM", 1):
            violations.append(f"LOW_CONFIDENCE: {confidence_level} < MEDIUM")
        else:
            passed.append(f"confidence: {confidence_level} >= MEDIUM")

        # 10. Minimum implied probability
        if self.min_implied_probability > 0 and implied_probability < self.min_implied_probability:
            violations.append(f"MIN_IMPLIED_PROB: {implied_probability:.2%} < {self.min_implied_probability:.2%}")
        else:
            passed.append(f"implied_prob: {implied_probability:.2%} >= floor")

        # 11. Edge direction
        if not (net_edge > 0):
            violations.append(f"NEGATIVE_EDGE: net_edge={net_edge:.4f} not positive")
        else:
            passed.append(f"edge_direction: positive ({net_edge:.4f})")

        # 12. Market type
        if market.market_type == "UNKNOWN":
            warnings.append("MARKET_TYPE: Could not classify")

        # 13. Portfolio category exposure
        cat_exposure = portfolio.get("category_exposures", {}).get(market.category, 0.0)
        new_cat_pct = (cat_exposure + 0) / self.bankroll   # stake not yet known
        if new_cat_pct > self.max_category_exposure_pct:
            violations.append(f"CATEGORY_EXPOSURE: {new_cat_pct:.1%} > {self.max_category_exposure_pct:.0%}")
        else:
            passed.append("category_exposure: OK")

        # 14. Portfolio event exposure
        evt_exposure = portfolio.get("event_exposures", {}).get(market.event_slug, 0.0)
        new_evt_pct = evt_exposure / self.bankroll
        if new_evt_pct > self.max_single_event_exposure_pct:
            violations.append(f"EVENT_EXPOSURE: {new_evt_pct:.1%} > {self.max_single_event_exposure_pct:.0%}")
        else:
            passed.append("event_exposure: OK")

        # 15. Timeline endgame
        if market.is_near_resolution and market.hours_to_resolution < 6:
            warnings.append(f"TIMELINE: Only {market.hours_to_resolution:.1f}h to resolution")

        # 16. Minimum stake (checked at sizing stage but noted here)
        passed.append("min_stake: deferred to sizing")

        allowed = len(violations) == 0
        decision = Decision.TRADE if allowed else Decision.NO_TRADE

        return RiskGateResult(
            allowed=allowed,
            decision=decision.value,
            violations=violations,
            warnings=warnings,
            checks_passed=passed,
            drawdown_heat=heat_level,
            drawdown_pct=dd_pct,
            kelly_multiplier=kelly_mult,
            portfolio_gate="ok" if allowed else violations[0] if violations else "unknown",
            estimated_fees_pct=self.transaction_fee_pct + self.exit_fee_pct,
        )


# ---------------------------------------------------------------------------
# DryRunSafetyGate — 三阶段Dry-Run安全门
# ---------------------------------------------------------------------------

class DryRunSafetyGate:
    """三阶段Dry-Run安全门 / Three-stage dry-run safety gate.

    Stage 1 — Pre-trade:   验证 dry_run 模式, bankroll, config完整性
    Stage 2 — Pre-order:   验证仓位不超出当前限制
    Stage 3 — Pre-execution: 最终sanity检查

    Usage:
        gate = DryRunSafetyGate(dry_run=True, bankroll=5000)
        result = gate.stage1(portfolio_state)
        if result.allowed:
            result = gate.stage2(portfolio_state, proposed_size=20)
    """

    def __init__(
        self,
        dry_run: bool = True,
        bankroll: float = 5000.0,
        max_daily_loss: float = 500.0,
        max_open_positions: int = 25,
        max_stake_per_market: float = 50.0,
    ):
        self._dry_run = dry_run
        self._bankroll = bankroll
        self._max_daily_loss = max_daily_loss
        self._max_open_positions = max_open_positions
        self._max_stake_per_market = max_stake_per_market
        self._stages_passed: dict[int, bool] = {}

    def stage1(self, portfolio_state: dict[str, Any] | None = None) -> RiskGateResult:
        """Stage 1: Pre-trade validation."""
        portfolio_state = portfolio_state or {}
        violations: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []

        if not self._dry_run:
            violations.append("SAFETY: Live trading not enabled — dry_run must be True")
        else:
            passed.append("dry_run: enabled")

        if self._bankroll <= 0:
            violations.append(f"SAFETY: Bankroll ${self._bankroll} is not positive")
        else:
            passed.append(f"bankroll: ${self._bankroll:.2f} (sane)")

        daily_pnl = portfolio_state.get("daily_pnl", 0.0)
        if abs(daily_pnl) >= self._max_daily_loss:
            violations.append(f"SAFETY: Daily loss ${abs(daily_pnl):.2f} >= ${self._max_daily_loss:.2f}")
        else:
            passed.append(f"daily_loss: OK")

        self._stages_passed[1] = len(violations) == 0
        return RiskGateResult(
            allowed=len(violations) == 0,
            decision=Decision.TRADE.value if len(violations) == 0 else Decision.NO_TRADE.value,
            violations=violations,
            warnings=warnings,
            checks_passed=passed,
            safety_gate_stage="stage1_pre_trade",
        )

    def stage2(
        self,
        proposed_size_usd: float,
        portfolio_state: dict[str, Any] | None = None,
        market_id: str = "",
        category: str = "",
        event_slug: str = "",
    ) -> RiskGateResult:
        """Stage 2: Pre-order validation."""
        portfolio_state = portfolio_state or {}
        violations: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []

        if proposed_size_usd > self._max_stake_per_market:
            violations.append(f"SAFETY: Proposed stake ${proposed_size_usd:.2f} > max ${self._max_stake_per_market:.2f}")
        else:
            passed.append(f"stake_size: ${proposed_size_usd:.2f} <= max")

        open_count = portfolio_state.get("open_position_count", 0)
        if open_count >= self._max_open_positions:
            violations.append(f"SAFETY: Open positions {open_count} >= limit {self._max_open_positions}")
        else:
            passed.append(f"open_positions: {open_count} < limit")

        self._stages_passed[2] = len(violations) == 0
        return RiskGateResult(
            allowed=len(violations) == 0,
            decision=Decision.TRADE.value if len(violations) == 0 else Decision.NO_TRADE.value,
            violations=violations,
            warnings=warnings,
            checks_passed=passed,
            safety_gate_stage="stage2_pre_order",
        )

    def stage3(
        self,
        edge: float,
        net_edge: float,
        market: MarketSnapshot,
        slippage_tolerance: float = 0.01,
    ) -> RiskGateResult:
        """Stage 3: Pre-execution final sanity check."""
        violations: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []

        if net_edge <= 0:
            violations.append(f"SAFETY: Edge no longer positive (net_edge={net_edge:.4f})")
        else:
            passed.append(f"edge: positive ({net_edge:.4f})")

        total_depth = market.bid_depth_5 + market.ask_depth_5
        if total_depth > 0 and total_depth < 2000.0:
            violations.append(f"SAFETY: Liquidity ${total_depth:.2f} < $2000")
        elif total_depth > 0:
            passed.append(f"liquidity: OK")

        if market.spread_pct > 0.06:
            violations.append(f"SAFETY: Spread {market.spread_pct:.2%} > 6%")
        elif market.spread_pct > 0:
            passed.append(f"spread: OK")

        self._stages_passed[3] = len(violations) == 0
        return RiskGateResult(
            allowed=len(violations) == 0,
            decision=Decision.TRADE.value if len(violations) == 0 else Decision.NO_TRADE.value,
            violations=violations,
            warnings=warnings,
            checks_passed=passed,
            safety_gate_stage="stage3_pre_execution",
        )

    def all_passed(self) -> bool:
        """所有阶段是否都通过 / Whether all stages have passed."""
        return all(self._stages_passed.get(i, False) for i in (1, 2, 3))


# ---------------------------------------------------------------------------
# PolymarketAutonomous — 主自主交易Agent
# ---------------------------------------------------------------------------

class PolymarketAutonomous:
    """Fully-Autonomous Polymarket Trading Agent.

    完整自主交易管线:
      1. MultiModelEnsembleTrader   — GPT-4o + Claude + Gemini REST集成
      2. PlattCalibratorV2         — Platt Scaling校准
      3. RiskGate15                 — 15+风险检查
      4. FractionalKellySizerV2    — 7因子Kelly仓位管理
      5. DryRunSafetyGate           — 三阶段dry-run安全门
      6. SelfCalibratingTrader      — 自校准：从结果中学习

    Usage:
        agent = PolymarketAutonomous(dry_run=True)
        decision = await agent.decide(
            question="Will BTC exceed $100k by end of 2025?",
            market_id="...",
            market=market_snapshot,
        )
        if decision.allowed:
            result = await agent.execute(decision)

        # Record outcome for self-improvement
        agent.record_outcome(market_id="...", actual_outcome=1.0)

    Args:
        dry_run: 如果为True，不提交真实订单 / If True, no real orders submitted
        kelly_fraction: Fractional Kelly基础比例 / Base Kelly fraction
        bankroll: 总资金 / Total bankroll
    """

    def __init__(
        self,
        dry_run: bool = True,
        kelly_fraction: float = 0.25,
        bankroll: float = 5000.0,
        max_stake_per_market: float = 50.0,
        models: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ):
        self._dry_run = dry_run
        self._bankroll = bankroll

        # Core components
        self._ensemble = MultiModelEnsembleTrader(
            models=models or DEFAULT_MODELS.copy(),
            weights=weights or dict(DEFAULT_WEIGHTS),
        )
        self._calibrator = PlattCalibratorV2(min_samples=30)
        self._self_calibrator = SelfCalibratingTrader(self._calibrator)
        self._risk_gate = RiskGate15(
            kelly_fraction=kelly_fraction,
            max_stake_per_market=max_stake_per_market,
            bankroll=bankroll,
            kill_switch=dry_run,   # dry_run模式自动启用kill switch
        )
        self._kelly_sizer = FractionalKellySizerV2(
            kelly_fraction=kelly_fraction,
            max_stake_per_market=max_stake_per_market,
            bankroll=bankroll,
        )
        self._safety_gate = DryRunSafetyGate(
            dry_run=dry_run,
            bankroll=bankroll,
        )

        # Portfolio state (simplified in-memory)
        self._portfolio_state: dict[str, Any] = {
            "daily_pnl": 0.0,
            "open_position_count": 0,
            "category_exposures": {},
            "event_exposures": {},
            "positions": [],
        }

    # ── Main decision pipeline ────────────────────────────────────────────

    async def decide(
        self,
        question: str,
        market_id: str,
        market: MarketSnapshot,
        regime: str = "NORMAL",
        num_contradictions: int = 0,
        evidence_summary: str = "",
        evidence_bullets: list[str] | None = None,
    ) -> TradeDecision:
        """执行完整自主决策管线 / Run the full autonomous decision pipeline.

        Pipeline:
          1. Multi-model ensemble forecast (GPT-4o + Claude + Gemini)
          2. Platt calibration with evidence/spread adjustments
          3. 15+ risk checks
          4. Fractional Kelly position sizing
          5. Dry-run safety gate (3 stages)

        Args:
            question: 市场问题 / Market question
            market_id: 市场ID / Market ID
            market: 市场快照 / Market snapshot
            regime: 市场状态 / Market regime
            num_contradictions: 矛盾信号数量 / Number of contradictory signals
            evidence_summary: 证据摘要 / Evidence summary
            evidence_bullets: 证据要点列表 / Key evidence bullets

        Returns:
            TradeDecision with full breakdown and final decision
        """
        # Step 1: Multi-model ensemble forecast
        ensemble_result = await self._ensemble.forecast(
            question=question,
            market_type=market.market_type,
            evidence_bullets=evidence_bullets,
            volume_usd=market.volume_usd,
            liquidity_usd=market.liquidity_usd,
            spread_pct=market.spread_pct,
            days_to_expiry=market.hours_to_resolution / 24.0,
            price_momentum=0.0,
            evidence_quality=market.evidence_quality,
            num_sources=market.num_sources,
            evidence_summary=evidence_summary,
            adaptive_weights=self._self_calibrator.get_adaptive_weights(),
        )

        # Step 2: Platt calibration
        calibrated_prob, cal_adjustments = self._calibrator.calibrate(
            raw_prob=ensemble_result.probability,
            evidence_quality=market.evidence_quality,
            num_contradictions=num_contradictions,
            ensemble_spread=ensemble_result.spread,
            confidence_level=ensemble_result.confidence_level,
        )

        # Step 3: Edge calculation
        edge = calibrated_prob - market.current_price
        total_fees = self._risk_gate.transaction_fee_pct + self._risk_gate.exit_fee_pct
        net_edge = edge - total_fees

        # Determine direction
        if calibrated_prob > market.current_price:
            direction = "BUY_YES"
            p_win = calibrated_prob
        else:
            direction = "BUY_NO"
            p_win = 1.0 - calibrated_prob

        # Step 4: Risk gate evaluation
        risk_result = self._risk_gate.evaluate(
            edge=edge,
            net_edge=net_edge,
            implied_probability=market.current_price,
            confidence_level=ensemble_result.confidence_level,
            market=market,
            portfolio=self._portfolio_state,
        )

        # Step 5: Position sizing (if risk gate passed)
        position_size: PositionSizeResult | None = None
        if risk_result.allowed:
            # Get correlated positions count
            num_correlated = sum(
                1 for p in self._portfolio_state.get("positions", [])
                if p.get("category") == market.category
            )
            position_size = self._kelly_sizer.compute_size(
                model_probability=calibrated_prob,
                implied_probability=market.current_price,
                direction=direction,
                confidence_level=ensemble_result.confidence_level,
                regime=regime,
                price_volatility=market.price_volatility,
                hours_to_resolution=market.hours_to_resolution,
                evidence_quality=market.evidence_quality,
                liquidity_usd=market.liquidity_usd,
                category=market.category,
                category_mult=1.0,
                drawdown_mult=risk_result.kelly_multiplier,
                num_correlated_positions=num_correlated,
            )

        # Step 6: Dry-run safety gate stages
        if risk_result.allowed and position_size and position_size.stake_usd > 0:
            s1 = self._safety_gate.stage1(self._portfolio_state)
            if not s1.allowed:
                risk_result.allowed = False
                risk_result.decision = Decision.NO_TRADE.value
                risk_result.violations.extend(s1.violations)

            s2 = self._safety_gate.stage2(
                proposed_size_usd=position_size.stake_usd,
                portfolio_state=self._portfolio_state,
                market_id=market_id,
                category=market.category,
                event_slug=market.event_slug,
            )
            if not s2.allowed:
                risk_result.allowed = False
                risk_result.decision = Decision.NO_TRADE.value
                risk_result.violations.extend(s2.violations)

            s3 = self._safety_gate.stage3(
                edge=edge,
                net_edge=net_edge,
                market=market,
            )
            if not s3.allowed:
                risk_result.allowed = False
                risk_result.decision = Decision.NO_TRADE.value
                risk_result.violations.extend(s3.violations)

        # Compute estimated PnL
        if risk_result.allowed and position_size:
            if direction == "BUY_YES":
                if market.current_price >= 1.0:
                    est_pnl = 0.0
                else:
                    est_pnl = position_size.stake_usd * (1.0 / market.current_price - 1.0) - position_size.estimated_fees
            else:
                if market.current_price <= 0.0:
                    est_pnl = 0.0
                else:
                    est_pnl = position_size.stake_usd * (1.0 / (1.0 - market.current_price) - 1.0) - position_size.estimated_fees
        else:
            est_pnl = 0.0

        return TradeDecision(
            market_id=market_id,
            question=question,
            direction=direction,
            probability=round(ensemble_result.probability, 4),
            calibrated_probability=round(calibrated_prob, 4),
            confidence_level=ensemble_result.confidence_level,
            edge=round(edge, 4),
            net_edge=round(net_edge, 4),
            stake_usd=position_size.stake_usd if position_size else 0.0,
            kelly_fraction=position_size.kelly_fraction_used if position_size else 0.0,
            kelly_multiplier=risk_result.kelly_multiplier,
            estimated_cost=position_size.estimated_cost if position_size else 0.0,
            estimated_pnl=round(est_pnl, 2),
            allowed=risk_result.allowed,
            decision=risk_result.decision,
            violations=risk_result.violations,
            warnings=risk_result.warnings,
            ensemble=ensemble_result,
            risk_gate=risk_result,
            position_size=position_size,
            dry_run=self._dry_run,
            reason="TRADE allowed" if risk_result.allowed else f"BLOCKED: {'; '.join(risk_result.violations)}",
        )

    async def execute(self, decision: TradeDecision) -> dict[str, Any]:
        """执行已批准的交易决策 / Execute an approved trade decision.

        在 dry_run 模式下不提交真实订单。
        In dry_run mode, no real orders are submitted.

        Args:
            decision: 来自 decide() 的决策 / Decision from decide()

        Returns:
            Execution result dict
        """
        if not decision.allowed:
            return {
                "executed": False,
                "reason": "Trade not allowed",
                "violations": decision.violations,
            }

        if self._dry_run:
            return {
                "executed": False,
                "dry_run": True,
                "reason": "Dry-run mode — no real order submitted",
                "decision": decision.to_dict(),
            }

        # TODO: Integrate with real Polymarket CLOB API for live execution
        return {
            "executed": False,
            "reason": "Live execution not yet implemented — integrate with polymarket_clob",
            "decision": decision.to_dict(),
        }

    def record_outcome(
        self,
        market_id: str,
        actual_outcome: float,
        decision: TradeDecision,
    ) -> None:
        """记录市场结果用于自校准 / Record market resolution for self-calibration.

        Args:
            market_id: 市场ID / Market ID
            actual_outcome: 0.0 (NO) 或 1.0 (YES) / 0.0 (NO) or 1.0 (YES)
            decision: 之前的交易决策 / Previous trade decision
        """
        # Extract per-model forecasts
        model_forecasts: dict[str, float] = {}
        if decision.ensemble:
            for f in decision.ensemble.individual_forecasts:
                model_forecasts[f.model_name] = f.probability

        self._self_calibrator.record_outcome(
            market_id=market_id,
            forecast_prob=decision.calibrated_probability,
            actual_outcome=actual_outcome,
            model_forecasts=model_forecasts,
            stake_usd=decision.stake_usd,
            pnl=decision.estimated_pnl,
        )

        # Check if retraining is needed
        if self._self_calibrator.should_retrain():
            stats = self._self_calibrator.retrain()

    def get_stats(self) -> dict[str, Any]:
        """获取Agent统计信息 / Get agent statistics."""
        return {
            "dry_run": self._dry_run,
            "bankroll": self._bankroll,
            "self_calibration": self._self_calibrator.stats,
            "portfolio": dict(self._portfolio_state),
            "safety_gate_passed": self._safety_gate.all_passed(),
        }
