"""PolymarketAgent — Fully-autonomous Polymarket trading agent.

多模型集成 + 全风险检查 + Fractional Kelly仓位管理 + Platt校准 + Whale追踪

Architecture:
  1. Research: 多模型并行分析 (GPT-4o 40% / Claude 3.5 35% / Gemini 1.5 25%)
  2. Calibration: Platt校准提升置信度准确性
  3. Risk Check: 15+风险检查 (任一block即阻止交易)
  4. Position Sizing: Fractional Kelly (7个调整因子)
  5. Whale Tracking: 7阶段Smart Money扫描
  6. Execution: 三阶段dry-run安全门

Usage:
    from quant_trading.agent.polymarket_agent import PolymarketAgent, RiskCheckResult
    from quant_trading.agent.calibration import PlattCalibrator

    agent = PolymarketAgent(llm_bridge=bridge, kelly_fraction=0.25, dry_run=True)
    research = await agent.research("Will ETH exceed $4000 by end of 2024?")
    result = await agent.place_bet(market_id="...", prediction="yes", amount=100.0)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PolymarketAgent",
    "RiskCheckResult",
    "FractionalKellySizer",
    "WhaleSignal",
    "ResearchResult",
    "BetResult",
    "DryRunSafetyGate",
]


# ---------------------------------------------------------------------------
# Default ensemble weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "gpt-4o": 0.40,
    "claude-3-5-sonnet-20241022": 0.35,
    "gemini-1.5-pro": 0.25,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RiskCheckResult:
    """Result of a single risk check."""
    passed: bool
    check_name: str
    reason: str | None = None
    severity: str = "block"  # 'block' | 'warn' | 'info'

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "check_name": self.check_name,
            "reason": self.reason,
            "severity": self.severity,
        }


@dataclass
class WhaleSignal:
    """Whale / Smart Money conviction signal for a market."""
    market_id: str
    title: str = ""
    whale_count: int = 0
    total_whale_usd: float = 0.0
    conviction_score: float = 0.0   # 0–100
    direction: str = ""            # "BULLISH" | "BEARISH"
    signal_strength: str = ""       # "STRONG" | "MODERATE" | "WEAK"
    avg_whale_price: float = 0.0
    current_price: float = 0.0
    detected_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "title": self.title,
            "whale_count": self.whale_count,
            "total_whale_usd": round(self.total_whale_usd, 2),
            "conviction_score": round(self.conviction_score, 1),
            "direction": self.direction,
            "signal_strength": self.signal_strength,
            "avg_whale_price": round(self.avg_whale_price, 4),
            "current_price": round(self.current_price, 4),
            "detected_at": self.detected_at,
        }


@dataclass
class ResearchResult:
    """Result of multi-model research on a market question."""
    question: str
    consensus: float = 0.5           # weighted ensemble probability
    confidence: float = 0.5          # calibrated confidence
    gpt4o_view: float = 0.5
    claude_view: float = 0.5
    gemini_view: float = 0.5
    ensemble_spread: float = 0.0     # max - min across models
    agreement_score: float = 0.0    # 1.0 = perfect agreement
    reasoning: str = ""
    confidence_level: str = "LOW"  # LOW | MEDIUM | HIGH
    invalidation_triggers: list[str] = field(default_factory=list)
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    whale_signal: WhaleSignal | None = None
    raw_responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    research_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "consensus": round(self.consensus, 4),
            "confidence": round(self.confidence, 4),
            "gpt4o_view": round(self.gpt4o_view, 4),
            "claude_view": round(self.claude_view, 4),
            "gemini_view": round(self.gemini_view, 4),
            "ensemble_spread": round(self.ensemble_spread, 4),
            "agreement_score": round(self.agreement_score, 2),
            "reasoning": self.reasoning,
            "confidence_level": self.confidence_level,
            "invalidation_triggers": self.invalidation_triggers,
            "key_evidence": self.key_evidence,
            "whale_signal": self.whale_signal.to_dict() if self.whale_signal else None,
            "research_time_ms": round(self.research_time_ms, 1),
        }


@dataclass
class BetResult:
    """Result of a bet placement attempt."""
    market_id: str
    prediction: str
    stake_usd: float
    kelly_fraction: float
    direction: str                   # "BUY_YES" | "BUY_NO"
    allowed: bool
    decision: str                    # "TRADE" | "NO TRADE"
    risk_results: list[RiskCheckResult] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    research: ResearchResult | None = None
    estimated_cost: float = 0.0
    dry_run: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "prediction": self.prediction,
            "stake_usd": round(self.stake_usd, 2),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "direction": self.direction,
            "allowed": self.allowed,
            "decision": self.decision,
            "risk_results": [r.to_dict() for r in self.risk_results],
            "violations": self.violations,
            "warnings": self.warnings,
            "research": self.research.to_dict() if self.research else None,
            "estimated_cost": round(self.estimated_cost, 2),
            "dry_run": self.dry_run,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# FractionalKellySizer
# ---------------------------------------------------------------------------


class FractionalKellySizer:
    """Fractional Kelly position size calculator.

    Kelly formula for binary outcome:
      f* = (p * b - q) / b
    where:
      p = winning probability
      b = odds (payout / cost - 1)
      q = 1 - p

    Fractional Kelly: f* × k (k = 0.25 ~ 0.5 typical)

    7 Adjustment Factors:
      1. Volatility adjustment     — reduces sizing in volatile markets
      2. Liquidity adjustment      — caps stake at a fraction of available depth
      3. Confidence adjustment     — LOW=0.5, MEDIUM=0.75, HIGH=1.0
      4. Concentration adjustment  — reduces when portfolio is already large
      5. Duration adjustment        — shorter duration → reduced sizing
      6. Sentiment extremity       — extreme sentiment → reduced sizing
      7. Trend strength            — strong trend → increased sizing
    """

    def __init__(self, base_fraction: float = 0.25):
        self._base_fraction = base_fraction
        self._CONF_MULT = {"LOW": 0.5, "MEDIUM": 0.75, "HIGH": 1.0}

    def compute_size(
        self,
        market_data: dict[str, Any],
        confidence: float,
        factors: dict[str, Any] | None = None,
    ) -> float:
        """Compute Kelly-optimal position size in USD.

        Args:
            market_data: Dict with keys:
                - model_prob: float  (0.01–0.99)
                - implied_prob: float  (market price as probability)
                - prediction: str  ('yes' or 'no')
                - price: float  (current market price)
                - stake_usd: float  (proposed stake — used for limit checks)
                - liquidity_usd: float  (available liquidity)
                - volatility: float  (0.0–1.0 price volatility)
                - hours_to_resolution: float | None
                - portfolio_total_usd: float  (total portfolio value)
            confidence: float  (0.0–1.0 calibrated confidence)
            factors: Optional dict of adjustment factors (see 7 factors above)

        Returns:
            Recommended stake in USD (may be 0 if Kelly is negative).
        """
        factors = factors or {}
        m = market_data

        model_prob = m.get("model_prob", 0.5)
        implied_prob = m.get("implied_prob", 0.5)
        prediction = m.get("prediction", "yes").lower()
        price = m.get("price", 0.5)
        liquidity_usd = m.get("liquidity_usd", 0.0)
        volatility = m.get("volatility", 0.0)
        hours = m.get("hours_to_resolution")
        portfolio_total = m.get("portfolio_total_usd", 10_000.0)
        bankroll = m.get("bankroll", 10_000.0)
        max_stake = m.get("max_stake_per_market", 200.0)

        # Compute edge direction
        if prediction == "yes":
            p = model_prob
            cost = implied_prob
        else:
            p = 1.0 - model_prob
            cost = 1.0 - implied_prob

        cost = max(cost, 0.01)
        b = (1.0 / cost) - 1.0   # odds
        q = 1.0 - p

        # Full Kelly fraction
        if b > 0:
            full_kelly = max(0.0, (p * b - q) / b)
        else:
            full_kelly = 0.0

        # 1. Volatility adjustment
        vol_thresh_high = 0.3
        vol_thresh_med = 0.15
        vol_mult = 1.0
        if volatility > vol_thresh_high:
            vol_mult = max(0.25, 1.0 - (volatility - vol_thresh_high) * 2)
        elif volatility > vol_thresh_med:
            vol_mult = max(0.5, 1.0 - (volatility - vol_thresh_med) * 3)
        if "volatility_mult" in factors:
            vol_mult = factors["volatility_mult"]

        # 2. Liquidity adjustment
        liq_mult = 1.0
        if liquidity_usd > 0:
            stake_estimate = full_kelly * bankroll
            liq_pct = stake_estimate / liquidity_usd if liquidity_usd > 0 else 0
            if liq_pct > 0.10:
                liq_mult = min(liq_mult, 0.10 / liq_pct)
        if "liquidity_mult" in factors:
            liq_mult = factors["liquidity_mult"]

        # 3. Confidence adjustment
        conf_level = self._prob_to_level(confidence)
        conf_mult = self._CONF_MULT.get(conf_level, 0.5)
        if "confidence_mult" in factors:
            conf_mult = factors["confidence_mult"]

        # 4. Concentration adjustment (portfolio size)
        conc_mult = 1.0
        if portfolio_total > 0:
            current_exposure = (portfolio_total - bankroll) / portfolio_total
            conc_mult = max(0.3, 1.0 - current_exposure * 0.5)
        if "concentration_mult" in factors:
            conc_mult = factors["concentration_mult"]

        # 5. Duration adjustment
        dur_mult = 1.0
        if hours is not None:
            if hours < 1:
                dur_mult = 0.0
            elif hours < 6:
                dur_mult = 0.25
            elif hours < 24:
                dur_mult = 0.5
            elif hours < 72:
                dur_mult = 0.75
        if "duration_mult" in factors:
            dur_mult = factors["duration_mult"]

        # 6. Sentiment extremity adjustment
        sent_mult = 1.0
        extremity = abs(model_prob - 0.5) * 2   # 0 (50/50) → 1 (near certain)
        if extremity > 0.8:                      # near-certainty
            sent_mult = 0.7
        elif extremity > 0.6:                    # strong lean
            sent_mult = 0.85
        if "sentiment_mult" in factors:
            sent_mult = factors["sentiment_mult"]

        # 7. Trend strength adjustment
        trend_mult = 1.0
        if "trend_strength" in factors:
            trend = factors["trend_strength"]
            if trend > 0.7:
                trend_mult = 1.15
            elif trend < 0.3:
                trend_mult = 0.85
        if "trend_mult" in factors:
            trend_mult = factors["trend_mult"]

        # Combined fractional Kelly
        combined = (
            self._base_fraction
            * conf_mult
            * vol_mult
            * liq_mult
            * conc_mult
            * dur_mult
            * sent_mult
            * trend_mult
        )
        adj_kelly = full_kelly * combined

        # Compute stake in USD
        stake = adj_kelly * bankroll

        # Apply hard caps
        stake = min(stake, max_stake)
        stake = min(stake, liquidity_usd * 0.10 if liquidity_usd > 0 else stake)

        # Minimum floor
        if 0 < stake < 0.50:
            stake = 0.0

        return max(0.0, stake)

    def _prob_to_level(self, prob: float) -> str:
        """Map calibrated probability to confidence level."""
        if prob >= 0.70:
            return "HIGH"
        elif prob >= 0.55:
            return "MEDIUM"
        return "LOW"


# ---------------------------------------------------------------------------
# DryRunSafetyGate — 三阶段安全门
# ---------------------------------------------------------------------------


class DryRunSafetyGate:
    """三阶段dry-run安全门.

    Stage 1 — Pre-trade:  Verify dry_run flag, bankroll sanity, config integrity
    Stage 2 — Pre-order:   Verify position doesn't breach limits at order time
    Stage 3 — Pre-execution: Final sanity check before order submission
    """

    STAGE_NAMES = {
        1: "pre_trade",
        2: "pre_order",
        3: "pre_execution",
    }

    def __init__(self, dry_run: bool = True):
        self._dry_run = dry_run
        self._stages_passed: dict[int, bool] = {}

    def check(self, stage: int, context: dict[str, Any]) -> tuple[bool, str]:
        """Run safety gate at the specified stage.

        Args:
            stage: 1, 2, or 3
            context: Dict with stage-specific data

        Returns:
            (passed, reason)
        """
        if stage == 1:
            return self._stage1_pre_trade(context)
        elif stage == 2:
            return self._stage2_pre_order(context)
        elif stage == 3:
            return self._stage3_pre_execution(context)
        return False, f"Unknown stage {stage}"

    def _stage1_pre_trade(self, ctx: dict[str, Any]) -> tuple[bool, str]:
        """Stage 1: Verify paper_mode, bankroll sanity, config integrity."""
        if self._dry_run:
            # In dry-run mode, always pass but log
            return True, "Dry-run mode — Stage 1 passed"

        # Live mode checks
        bankroll = ctx.get("bankroll", 0)
        if bankroll <= 0:
            return False, f"Invalid bankroll: {bankroll}"

        config = ctx.get("config", {})
        if not config.get("api_key") or not config.get("private_key"):
            return False, "Missing API credentials in live mode"

        return True, "Stage 1 passed"

    def _stage2_pre_order(self, ctx: dict[str, Any]) -> tuple[bool, str]:
        """Stage 2: Verify position doesn't breach limits right now."""
        if self._dry_run:
            return True, "Dry-run mode — Stage 2 passed"

        # Check current open positions
        current_open = ctx.get("current_open", 0)
        max_open = ctx.get("max_open_positions", 10)
        if current_open >= max_open:
            return False, f"Max open positions: {current_open} >= {max_open}"

        # Check daily loss limit
        daily_pnl = ctx.get("daily_pnl", 0.0)
        max_daily_loss = ctx.get("max_daily_loss", 500.0)
        if daily_pnl < 0 and abs(daily_pnl) >= max_daily_loss:
            return False, f"Max daily loss: ${abs(daily_pnl):.2f} >= ${max_daily_loss:.2f}"

        # Check drawdown kill
        if ctx.get("drawdown_killed", False):
            return False, "Drawdown auto-kill is active"

        return True, "Stage 2 passed"

    def _stage3_pre_execution(self, ctx: dict[str, Any]) -> tuple[bool, str]:
        """Stage 3: Final sanity check before order submission."""
        if self._dry_run:
            return True, "Dry-run mode — Stage 3 passed (no real order)"

        # Verify market is still open
        if ctx.get("is_resolved", False):
            return False, "Market has already resolved"

        # Verify price hasn't moved too much since research
        price_move_pct = ctx.get("price_move_since_research", 0.0)
        max_price_move = ctx.get("max_price_move_pct", 0.05)   # 5% max
        if abs(price_move_pct) > max_price_move:
            return False, f"Price moved {price_move_pct:.2%} since research (max {max_price_move:.2%})"

        return True, "Stage 3 passed"


# ---------------------------------------------------------------------------
# PolymarketAgent
# ---------------------------------------------------------------------------


class PolymarketAgent:
    """Polymarket autonomous trading agent.

    Multi-model weighted ensemble:
      - GPT-4o:        40%
      - Claude 3.5:   35%
      - Gemini 1.5:   25%

    Workflow:
      1. research()       — 多模型并行分析, 集成置信度
      2. check_risk()     — 15+风险检查, 任一block即阻止
      3. place_bet()       — research + risk_check + Kelly仓位

    Args:
        llm_bridge:   MultiLLMBridge instance (from agent.multi_llm_bridge)
        kelly_fraction: Base fractional Kelly (default 0.25)
        dry_run:       If True, no real orders are submitted (default True)
    """

    def __init__(
        self,
        llm_bridge: Any,   # MultiLLMBridge — imported lazily to avoid circular deps
        kelly_fraction: float = 0.25,
        dry_run: bool = True,
    ):
        self._bridge = llm_bridge
        self._kelly_fraction = kelly_fraction
        self._dry_run = dry_run
        self._sizer = FractionalKellySizer(base_fraction=kelly_fraction)
        self._gate = DryRunSafetyGate(dry_run=dry_run)
        self._weights = dict(DEFAULT_WEIGHTS)

        # Whale tracking state
        self._whale_history: dict[str, Any] = {}   # market_id → last WhaleSignal

    # ── Research ─────────────────────────────────────────────────────────

    async def research(self, market_question: str) -> ResearchResult:
        """Research a market question using multi-model parallel analysis.

        Calls GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro in parallel,
        then aggregates using weighted trimmed mean.

        Args:
            market_question: The market question to research.

        Returns:
            ResearchResult with consensus, per-model views, confidence level,
            whale signal, and reasoning.
        """
        import time
        t0 = time.monotonic()

        # Build the research prompt
        prompt = self._build_research_prompt(market_question)

        # Parallel model calls
        async def call_gpt4o():
            return await self._call_model(
                "openai",        # maps to GPT-4o in bridge
                prompt,
                "gpt-4o",
            )

        async def call_claude():
            return await self._call_model(
                "anthropic",     # maps to Claude 3.5 Sonnet
                prompt,
                "claude-3-5-sonnet-20241022",
            )

        async def call_gemini():
            return await self._call_model(
                "gemini",
                prompt,
                "gemini-1.5-pro",
            )

        gpt4o_r, claude_r, gemini_r = await asyncio.gather(
            call_gpt4o(), call_claude(), call_gemini(),
            return_exceptions=True,
        )

        # Parse responses
        gpt4o_prob = self._extract_prob(gpt4o_r)
        claude_prob = self._extract_prob(claude_r)
        gemini_prob = self._extract_prob(gemini_r)

        # Weighted ensemble
        probs = {
            "gpt-4o": gpt4o_prob,
            "claude": claude_prob,
            "gemini": gemini_prob,
        }
        consensus = (
            self._weights["gpt-4o"] * gpt4o_prob
            + self._weights["claude-3-5-sonnet-20241022"] * claude_prob
            + self._weights["gemini-1.5-pro"] * gemini_prob
        )

        # Spread & agreement
        valid_probs = [p for p in probs.values() if p > 0]
        spread = (max(valid_probs) - min(valid_probs)) if valid_probs else 0.0
        avg_prob = sum(valid_probs) / len(valid_probs) if valid_probs else 0.5
        agreement = 1.0 - (spread / 0.5) if spread > 0 else 1.0

        # Confidence level
        conf_level = self._compute_confidence_level(consensus, spread, agreement)

        # Reasoning (use the most detailed response)
        reasoning = self._extract_reasoning(gpt4o_r) or self._extract_reasoning(claude_r)

        elapsed_ms = (time.monotonic() - t0) * 1000

        return ResearchResult(
            question=market_question,
            consensus=consensus,
            confidence=consensus,
            gpt4o_view=gpt4o_prob,
            claude_view=claude_prob,
            gemini_view=gemini_prob,
            ensemble_spread=spread,
            agreement_score=agreement,
            reasoning=reasoning,
            confidence_level=conf_level,
            invalidation_triggers=self._extract_triggers(gpt4o_r) or self._extract_triggers(claude_r),
            key_evidence=self._extract_evidence(gpt4o_r) or [],
            raw_responses={
                "gpt4o": self._safe_raw(gpt4o_r),
                "claude": self._safe_raw(claude_r),
                "gemini": self._safe_raw(gemini_r),
            },
            research_time_ms=elapsed_ms,
        )

    def _build_research_prompt(self, question: str) -> str:
        return f"""\
You are an expert probabilistic forecaster on prediction markets (Polymarket).

MARKET QUESTION: {question}

TASK:
1. Analyze the question based on your knowledge.
2. Estimate the probability (0.01–0.99, NEVER 0 or 1).
3. Identify 2-4 specific events/data that would CHANGE this forecast.
4. Provide your key supporting evidence.

Return ONLY valid JSON:
{{
  "probability": <0.01-0.99>,
  "reasoning": "2-3 sentence explanation",
  "invalidation_triggers": ["event that would change the forecast"],
  "key_evidence": [{{"text": "evidence", "source": "source name"}}]
}}

Return ONLY valid JSON, no markdown fences.
"""

    async def _call_model(
        self, provider: str, prompt: str, model_name: str
    ) -> dict[str, Any]:
        """Call a model via the MultiLLMBridge."""
        try:
            response = await self._bridge.ensemble_generate(
                prompt=prompt,
                models=[model_name],
                temperature=0.3,
                max_tokens=600,
            )
            # Extract content from response
            if hasattr(response, "content"):
                content = response.content
            elif isinstance(response, dict):
                content = response.get("content", "")
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            return self._parse_json_response(content)
        except Exception as e:
            return {"error": str(e)}

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n", 1)
            if len(lines) > 1:
                text = lines[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            return {"probability": 0.5, "reasoning": text[:200]}

    def _extract_prob(self, response: Any) -> float:
        """Extract probability from model response."""
        if isinstance(response, dict):
            prob = response.get("probability", 0.5)
            return max(0.01, min(0.99, float(prob)))
        return 0.5

    def _extract_reasoning(self, response: Any) -> str:
        if isinstance(response, dict):
            return response.get("reasoning", "")
        return ""

    def _extract_triggers(self, response: Any) -> list[str]:
        if isinstance(response, dict):
            triggers = response.get("invalidation_triggers", [])
            return triggers if isinstance(triggers, list) else []
        return []

    def _extract_evidence(self, response: Any) -> list[dict[str, Any]]:
        if isinstance(response, dict):
            ev = response.get("key_evidence", [])
            return ev if isinstance(ev, list) else []
        return []

    def _safe_raw(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return {k: v for k, v in response.items() if k != "error"}
        return {}

    def _compute_confidence_level(
        self, consensus: float, spread: float, agreement: float
    ) -> str:
        """Compute confidence level from ensemble statistics."""
        extremity = abs(consensus - 0.5) * 2   # 0→1
        if agreement >= 0.9 and extremity >= 0.4:
            return "HIGH"
        elif agreement >= 0.7 and extremity >= 0.2:
            return "MEDIUM"
        return "LOW"

    # ── Risk checks ───────────────────────────────────────────────────────

    def check_risk(
        self,
        position: dict[str, Any],
        config: Any | None = None,
    ) -> list[RiskCheckResult]:
        """Run 15+ risk checks on a proposed position.

        Any BLOCK-severity failure prevents the trade.

        Args:
            position: Dict with market and trade details.
            config: Optional RiskConfig instance.

        Returns:
            List of RiskCheckResult, one per check.
        """
        from quant_trading.agent import risk_checks as rc

        cfg = config or rc.RiskConfig()
        cfg.kill_switch = cfg.kill_switch or self._dry_run  # ensure dry_run sets kill_switch

        results = rc.run_risk_checks(position, config=cfg)

        # Map to our RiskCheckResult
        return [
            RiskCheckResult(
                passed=r.passed,
                check_name=r.check_name,
                reason=r.reason,
                severity=r.severity,
            )
            for r in results
        ]

    # ── Whale tracking ─────────────────────────────────────────────────────

    def update_whale_signal(self, market_id: str, signal: WhaleSignal) -> None:
        """Update stored whale signal for a market."""
        self._whale_history[market_id] = signal

    def get_whale_signal(self, market_id: str) -> WhaleSignal | None:
        """Get the last whale signal for a market."""
        return self._whale_history.get(market_id)

    async def scan_whales(self, market_ids: list[str]) -> dict[str, WhaleSignal]:
        """Scan whale wallets for multiple markets (7-stage liquidity scan).

        Stage 1: Fetch top wallets positions
        Stage 2: Score each wallet
        Stage 3: Detect position deltas (new/exit/size change)
        Stage 4: Group by market+outcome
        Stage 5: Compute conviction score
        Stage 6: Filter by min conviction
        Stage 7: Determine direction & strength

        Args:
            market_ids: List of market IDs to scan.

        Returns:
            Dict mapping market_id → WhaleSignal.
        """
        # Placeholder: full whale scanning would use the wallet_scanner from source
        # For now, return empty signals — actual whale tracking would call
        # WalletScanner.scan() from the Fully-Autonomous-Polymarket-AI-Trading-Bot
        return {mid: WhaleSignal(market_id=mid) for mid in market_ids}

    # ── Place bet ──────────────────────────────────────────────────────────

    async def place_bet(
        self,
        market_id: str,
        prediction: str,  # 'yes' or 'no'
        amount: float,    # USD amount (used as reference, Kelly overrides)
    ) -> BetResult:
        """Place a bet on a Polymarket market.

        Full workflow:
          1. Research: Multi-model analysis (if not already done)
          2. Dry-run Stage 1: Pre-trade safety gate
          3. Risk Check: 15+ checks
          4. Kelly Sizing: Compute optimal stake
          5. Dry-run Stage 2: Pre-order safety gate
          6. Dry-run Stage 3: Pre-execution safety gate

        Args:
            market_id: Polymarket condition/market ID.
            prediction: 'yes' or 'no'.
            amount: Proposed USD amount (Kelly sizing may override).

        Returns:
            BetResult with decision, risk results, and Kelly stake.
        """
        from quant_trading.agent import risk_checks as rc

        # Step 1: Research (lightweight — assumes market_data already available)
        research = await self._research_market(market_id, prediction)

        # Step 2: Dry-run Stage 1
        stage1_passed, stage1_reason = self._gate.check(1, {
            "bankroll": 10_000.0,
            "config": {},
        })
        if not stage1_passed:
            return self._make_blocked_result(
                market_id, prediction, amount, research,
                [RiskCheckResult(False, "dry_run_stage1", stage1_reason, "block")],
            )

        # Step 3: Build position dict for risk checks
        market_data = self._get_market_data(market_id, prediction, research)
        position = {
            "prediction": prediction,
            "stake_usd": amount,
            "net_edge": research.consensus - market_data.get("implied_prob", 0.5),
            "model_prob": research.consensus,
            "implied_prob": market_data.get("implied_prob", 0.5),
            "spread_pct": market_data.get("spread_pct", 0.02),
            "liquidity_usd": market_data.get("liquidity_usd", 0.0),
            "evidence_quality": market_data.get("evidence_quality", 0.5),
            "confidence_level": research.confidence_level,
            "market_type": market_data.get("market_type", "UNKNOWN"),
            "hours_to_resolution": market_data.get("hours_to_resolution"),
            "is_resolved": market_data.get("is_resolved", False),
            "daily_pnl": market_data.get("daily_pnl", 0.0),
            "current_open": market_data.get("current_open", 0),
            "drawdown_pct": market_data.get("drawdown_pct", 0.0),
            "drawdown_heat": market_data.get("drawdown_heat", 0),
            "dry_run": self._dry_run,
        }

        # Step 4: Risk checks
        risk_results = self.check_risk(position)

        # Check for BLOCK violations
        blocked = [r for r in risk_results if r.severity == "block" and not r.passed]
        if blocked:
            violations = [f"[{r.check_name}] {r.reason}" for r in blocked]
            return self._make_blocked_result(
                market_id, prediction, amount, research, risk_results, violations=violations,
            )

        # Step 5: Kelly sizing
        kelly_stake = self._sizer.compute_size(
            market_data={
                "model_prob": research.consensus,
                "implied_prob": market_data.get("implied_prob", 0.5),
                "prediction": prediction,
                "price": market_data.get("price", 0.5),
                "liquidity_usd": market_data.get("liquidity_usd", 0.0),
                "volatility": market_data.get("volatility", 0.0),
                "hours_to_resolution": market_data.get("hours_to_resolution"),
                "portfolio_total_usd": market_data.get("portfolio_total_usd", 10_000.0),
                "bankroll": 10_000.0,
                "max_stake_per_market": 200.0,
            },
            confidence=research.confidence,
            factors={},
        )

        # Step 6: Dry-run Stage 2
        stage2_passed, stage2_reason = self._gate.check(2, {
            "current_open": position["current_open"],
            "max_open_positions": 10,
            "daily_pnl": position["daily_pnl"],
            "max_daily_loss": 500.0,
            "drawdown_killed": position["drawdown_pct"] >= 0.25,
        })
        if not stage2_passed:
            return self._make_blocked_result(
                market_id, prediction, kelly_stake, research, risk_results,
                violations=[f"dry_run_stage2: {stage2_reason}"],
            )

        # Step 7: Dry-run Stage 3
        stage3_passed, stage3_reason = self._gate.check(3, {
            "is_resolved": position["is_resolved"],
            "price_move_since_research": 0.0,
            "max_price_move_pct": 0.05,
        })
        if not stage3_passed:
            return self._make_blocked_result(
                market_id, prediction, kelly_stake, research, risk_results,
                violations=[f"dry_run_stage3: {stage3_reason}"],
            )

        # Step 8: Warnings only (non-blocking)
        warnings = [
            f"[{r.check_name}] {r.reason}"
            for r in risk_results
            if r.severity == "warn" and not r.passed
        ]

        direction = "BUY_YES" if prediction.lower() == "yes" else "BUY_NO"
        estimated_cost = kelly_stake * market_data.get("price", 0.5)

        return BetResult(
            market_id=market_id,
            prediction=prediction,
            stake_usd=kelly_stake,
            kelly_fraction=self._kelly_fraction,
            direction=direction,
            allowed=True,
            decision="TRADE",
            risk_results=risk_results,
            violations=[],
            warnings=warnings,
            research=research,
            estimated_cost=estimated_cost,
            dry_run=self._dry_run,
            reason="Trade allowed — all checks passed" if not self._dry_run else "Dry-run — no real order submitted",
        )

    # ── Calibration ────────────────────────────────────────────────────────

    async def calibrate_predictions(
        self,
        historical_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Platt calibration — fit sigmoid(a*x + b) on historical results.

        Args:
            historical_results: List of dicts with keys:
                - confidence: float (0.0–1.0)
                - is_correct: bool

        Returns:
            Dict with calibration stats (a, b, brier_score, n_samples).
        """
        from quant_trading.agent.calibration import PlattCalibrator

        calibrator = PlattCalibrator(min_samples=30)
        pairs = [(r["confidence"], r["is_correct"]) for r in historical_results]
        success = calibrator.fit(pairs)
        return {
            "success": success,
            **calibrator.stats,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _research_market(
        self,
        market_id: str,
        prediction: str,
    ) -> ResearchResult:
        """Lightweight research using cached whale signal if available."""
        whale_sig = self.get_whale_signal(market_id)
        return ResearchResult(
            question=f"Market {market_id}",
            consensus=0.5,
            confidence=0.5,
            whale_signal=whale_sig,
        )

    def _get_market_data(self, market_id: str, prediction: str, research: ResearchResult) -> dict[str, Any]:
        """Fetch market data for a Polymarket market.

        In production this would call the Polymarket CLOB API.
        Returns placeholder data for dry-run / testing.
        """
        # Placeholder — in production, call polymarket_clob.get_orderbook()
        return {
            "market_id": market_id,
            "implied_prob": 0.5,
            "spread_pct": 0.02,
            "liquidity_usd": 1000.0,
            "price": 0.5,
            "evidence_quality": 0.5,
            "market_type": "GENERAL",
            "hours_to_resolution": 48.0,
            "is_resolved": False,
            "daily_pnl": 0.0,
            "current_open": 0,
            "drawdown_pct": 0.0,
            "drawdown_heat": 0,
            "volatility": 0.1,
            "portfolio_total_usd": 10_000.0,
        }

    def _make_blocked_result(
        self,
        market_id: str,
        prediction: str,
        stake_usd: float,
        research: ResearchResult | None,
        risk_results: list[RiskCheckResult],
        violations: list[str] | None = None,
    ) -> BetResult:
        return BetResult(
            market_id=market_id,
            prediction=prediction,
            stake_usd=stake_usd,
            kelly_fraction=self._kelly_fraction,
            direction=f"BUY_{prediction.upper()}",
            allowed=False,
            decision="NO TRADE",
            risk_results=risk_results,
            violations=violations or [],
            warnings=[],
            research=research,
            estimated_cost=0.0,
            dry_run=self._dry_run,
            reason="BLOCKED: " + "; ".join(violations or ["risk check failed"]),
        )
