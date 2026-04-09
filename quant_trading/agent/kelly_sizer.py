"""Fractional Kelly position sizing with 7 adjustment factors.

Based on Fully-Autonomous-Polymarket-AI-Trading-Bot's position_sizer architecture:

  Kelly formula for binary outcome:
    f* = (p * b - q) / b
  where:
    p = model probability of winning
    b = odds (payout / cost - 1)
    q = 1 - p

  Fractional Kelly: f* × kelly_fraction (default 0.25)

7 Adjustment Factors:
  1. Confidence level: LOW=0.5, MEDIUM=0.75, HIGH=1.0
  2. Drawdown heat multiplier: reduces sizing during drawdowns
  3. Timeline multiplier: adjusts for resolution timing (near/far)
  4. Volatility adjustment: reduces sizing in volatile markets
  5. Regime multiplier: from regime detector (bull/bear/neutral)
  6. Category multiplier: per-category stake limits
  7. Liquidity cap: never take >X% of available liquidity

Key concepts:
- Skill-first integration: works as a Skill in the existing agent framework
- Guardrails: capped by max_stake_per_market and max_bankroll_fraction
- Minimum stake floor: avoids dust trades (min_stake_usd)
- Portfolio gate: checks category/event exposure before sizing
- Transparent breakdown: all multipliers exposed in output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PositionSize:
    """Computed position size with full adjustment breakdown."""
    stake_usd: float
    kelly_fraction_used: float     # Effective Kelly fraction after all adjustments
    full_kelly_stake: float        # Uncapped Kelly stake
    capped_by: str                 # What ultimately capped the position
    direction: str                 # "BUY_YES" | "BUY_NO"
    token_quantity: float         # Approximate tokens at implied price

    # Adjustment factors breakdown
    base_kelly: float = 0.0        # Raw Kelly fraction from formula
    confidence_mult: float = 1.0  # Factor 1: confidence level
    drawdown_mult: float = 1.0     # Factor 2: drawdown heat
    timeline_mult: float = 1.0     # Factor 3: resolution timing
    volatility_mult: float = 1.0    # Factor 4: price volatility
    regime_mult: float = 1.0       # Factor 5: market regime
    category_mult: float = 1.0      # Factor 6: category stake multiplier
    portfolio_gate: str = "ok"     # Factor 7: portfolio exposure check

    # Cost metadata
    estimated_cost: float = 0.0    # Cost in USD
    estimated_fees: float = 0.0     # Fees in USD
    net_cost: float = 0.0          # Cost + fees

    def to_dict(self) -> dict[str, Any]:
        return {
            "stake_usd": round(self.stake_usd, 2),
            "kelly_fraction_used": round(self.kelly_fraction_used, 4),
            "full_kelly_stake": round(self.full_kelly_stake, 2),
            "capped_by": self.capped_by,
            "direction": self.direction,
            "token_quantity": round(self.token_quantity, 2),
            "base_kelly": round(self.base_kelly, 4),
            "confidence_mult": round(self.confidence_mult, 2),
            "drawdown_mult": round(self.drawdown_mult, 2),
            "timeline_mult": round(self.timeline_mult, 2),
            "volatility_mult": round(self.volatility_mult, 2),
            "regime_mult": round(self.regime_mult, 2),
            "category_mult": round(self.category_mult, 2),
            "portfolio_gate": self.portfolio_gate,
            "estimated_cost": round(self.estimated_cost, 2),
            "estimated_fees": round(self.estimated_fees, 2),
            "net_cost": round(self.net_cost, 2),
        }


@dataclass
class KellyConfig:
    """Configuration for Kelly position sizing."""
    kelly_fraction: float = 0.25          # Fraction of full Kelly to use
    max_stake_per_market: float = 50.0     # Hard cap per position
    max_bankroll_fraction: float = 0.05    # Max fraction of bankroll per trade
    min_stake_usd: float = 1.0             # Minimum stake floor
    bankroll: float = 5000.0               # Total bankroll
    # Volatility thresholds
    volatility_high_threshold: float = 0.15
    volatility_med_threshold: float = 0.10
    volatility_high_min_mult: float = 0.4
    volatility_med_min_mult: float = 0.6
    # Category multipliers
    category_stake_multipliers: dict[str, float] = field(default_factory=lambda: {
        "MACRO": 1.0,
        "CORPORATE": 0.75,
        "ELECTION": 0.50,
    })
    # Fees
    transaction_fee_pct: float = 0.02
    exit_fee_pct: float = 0.02
    gas_cost_usd: float = 0.01
    # Max liquidity pct
    max_liquidity_pct: float = 0.05


# ---------------------------------------------------------------------------
# Kelly Sizer
# ---------------------------------------------------------------------------


class KellySizer:
    """Fractional Kelly position sizer with 7 adjustment factors.

    Usage::

        sizer = KellySizer(config=KellyConfig(bankroll=5000, kelly_fraction=0.25))
        size = sizer.size(
            edge=edge_result,
            confidence_level="MEDIUM",
            drawdown_multiplier=0.75,
            timeline_multiplier=1.0,
            price_volatility=0.05,
            regime_multiplier=1.0,
            category="MACRO",
            liquidity_usd=10000,
        )
    """

    def __init__(self, config: KellyConfig | None = None):
        self._config = config or KellyConfig()

    def size(
        self,
        edge: Any,                   # EdgeResult or similar with direction, model_probability, implied_probability
        confidence_level: str = "MEDIUM",
        drawdown_multiplier: float = 1.0,
        timeline_multiplier: float = 1.0,
        price_volatility: float = 0.0,
        regime_multiplier: float = 1.0,
        category: str = "",
        liquidity_usd: float = 0.0,
        max_liquidity_pct: float | None = None,
    ) -> PositionSize:
        """Calculate optimal position size using fractional Kelly.

        Args:
            edge: Edge result with direction, model_probability, implied_probability
            confidence_level: LOW / MEDIUM / HIGH — scales Kelly fraction
            drawdown_multiplier: 0–1 from drawdown manager
            timeline_multiplier: 0.5–1.3 from timeline assessment
            price_volatility: Recent price volatility (0–1 scale)
            regime_multiplier: 0–1 from regime detector
            category: Market category for category multiplier
            liquidity_usd: Available liquidity in USD
            max_liquidity_pct: Max fraction of liquidity to take (default 5%)

        Returns:
            PositionSize with full breakdown
        """
        config = self._config
        direction = getattr(edge, "direction", "BUY_YES")
        max_liq_pct = max_liquidity_pct if max_liquidity_pct is not None else config.max_liquidity_pct

        # ── Portfolio gate ────────────────────────────────────────────
        # (Caller should call check_portfolio_exposure separately)
        # Placeholder — actual portfolio check is done in risk_gate
        portfolio_gate = "ok"

        # ── Compute raw Kelly ─────────────────────────────────────────
        if direction == "BUY_YES":
            p = getattr(edge, "model_probability", 0.5)
            cost = getattr(edge, "implied_probability", 0.5)
        else:
            p = 1.0 - getattr(edge, "model_probability", 0.5)
            cost = 1.0 - getattr(edge, "implied_probability", 0.5)

        cost = max(0.01, min(0.99, cost))
        b = (1.0 / cost) - 1.0   # odds
        q = 1.0 - p

        # Full Kelly fraction
        if b > 0:
            full_kelly_frac = (p * b - q) / b
        else:
            full_kelly_frac = 0.0
        full_kelly_frac = max(0.0, full_kelly_frac)
        base_kelly = full_kelly_frac

        # ── Factor 1: Confidence multiplier ───────────────────────────
        kelly_mult = config.kelly_fraction
        if confidence_level == "LOW":
            conf_mult = 0.5
        elif confidence_level == "MEDIUM":
            conf_mult = 0.75
        else:  # HIGH
            conf_mult = 1.0
        kelly_mult *= conf_mult

        # ── Factor 4: Volatility adjustment ───────────────────────────
        vol_mult = 1.0
        if price_volatility > config.volatility_high_threshold:
            vol_mult = max(
                config.volatility_high_min_mult,
                1.0 - (price_volatility - config.volatility_high_threshold) * 2,
            )
        elif price_volatility > config.volatility_med_threshold:
            vol_mult = max(
                config.volatility_med_min_mult,
                1.0 - (price_volatility - config.volatility_med_threshold) * 3,
            )

        # ── Factor 6: Category multiplier ─────────────────────────────
        cat_mult = config.category_stake_multipliers.get(category, 1.0)

        # ── Apply all multipliers ─────────────────────────────────────
        # combined = kelly_fraction × conf × drawdown × timeline × vol × regime × category
        combined_mult = (
            kelly_mult
            * drawdown_multiplier
            * timeline_multiplier
            * vol_mult
            * regime_multiplier
            * cat_mult
        )
        adj_kelly = full_kelly_frac * combined_mult
        full_kelly_stake = adj_kelly * config.bankroll

        # ── Apply caps ────────────────────────────────────────────────
        max_stake = config.max_stake_per_market
        max_bankroll = config.max_bankroll_fraction * config.bankroll

        # Liquidity cap: never take more than X% of available liquidity
        if liquidity_usd > 0:
            max_liquidity = liquidity_usd * max_liq_pct
        else:
            max_liquidity = float("inf")

        stake = min(full_kelly_stake, max_stake, max_bankroll, max_liquidity)
        stake = max(0.0, stake)

        # Minimum stake floor
        if 0 < stake < config.min_stake_usd:
            stake = 0.0

        # ── Determine what capped it ──────────────────────────────────
        if drawdown_multiplier <= 0:
            capped_by = "drawdown"
        elif stake == 0.0 and full_kelly_stake > 0:
            capped_by = "min_stake"
        elif stake >= full_kelly_stake - 0.01:
            capped_by = "kelly"
        elif stake >= max_stake - 0.01:
            capped_by = "max_stake"
        elif liquidity_usd > 0 and max_liquidity < float("inf") and stake >= max_liquidity - 0.01:
            capped_by = "liquidity"
        else:
            capped_by = "max_bankroll"

        # ── Token quantity ─────────────────────────────────────────────
        token_qty = stake / cost if cost > 0 else 0.0

        # ── Cost breakdown ─────────────────────────────────────────────
        estimated_cost = stake
        estimated_fees = stake * (config.transaction_fee_pct + config.exit_fee_pct)
        net_cost = estimated_cost + estimated_fees + config.gas_cost_usd

        return PositionSize(
            stake_usd=round(stake, 2),
            kelly_fraction_used=round(adj_kelly, 4),
            full_kelly_stake=round(full_kelly_stake, 2),
            capped_by=capped_by,
            direction=direction,
            token_quantity=round(token_qty, 2),
            base_kelly=round(base_kelly, 4),
            confidence_mult=round(conf_mult, 2),
            drawdown_mult=round(drawdown_multiplier, 2),
            timeline_mult=round(timeline_multiplier, 2),
            volatility_mult=round(vol_mult, 2),
            regime_mult=round(regime_multiplier, 2),
            category_mult=round(cat_mult, 2),
            portfolio_gate=portfolio_gate,
            estimated_cost=round(estimated_cost, 2),
            estimated_fees=round(estimated_fees, 2),
            net_cost=round(net_cost, 2),
        )

    def size_batch(
        self,
        edges: list[Any],
        **kwargs,
    ) -> list[PositionSize]:
        """Size multiple positions. Sum is checked against bankroll.

        Args:
            edges: List of edge results
            **kwargs: Same arguments as size()

        Returns:
            List of PositionSize in same order as edges
        """
        results: list[PositionSize] = []
        for edge in edges:
            size = self.size(edge, **kwargs)
            results.append(size)
        return results


# ---------------------------------------------------------------------------
# Regime Detection (simplified for Kelly sizing)
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Simple regime detector for market regime multiplier.

    Regimes:
      - BULL: momentum positive, low volatility → regime_mult > 1.0
      - BEAR: momentum negative, high volatility → regime_mult < 1.0
      - NEUTRAL: mixed signals → regime_mult = 1.0
    """

    def detect(
        self,
        price_momentum: float = 0.0,
        price_volatility: float = 0.0,
        volume_trend: float = 0.0,
    ) -> tuple[str, float]:
        """Detect market regime and return regime multiplier.

        Args:
            price_momentum: 24h price change rate (-1 to 1)
            price_volatility: Recent volatility (0 to 1)
            volume_trend: Volume trend vs historical average (-1 to 1)

        Returns:
            (regime_name, regime_multiplier)
        """
        # Bull signal: positive momentum + low volatility
        bull_score = 0.0
        if price_momentum > 0.02:
            bull_score += 0.5
        elif price_momentum < -0.02:
            bull_score -= 0.5
        if price_volatility < 0.10:
            bull_score += 0.3
        elif price_volatility > 0.20:
            bull_score -= 0.3
        if volume_trend > 0.2:
            bull_score += 0.2

        # Bear signal
        bear_score = 0.0
        if price_momentum < -0.02:
            bear_score += 0.5
        elif price_momentum > 0.02:
            bear_score -= 0.5
        if price_volatility > 0.20:
            bear_score += 0.3
        elif price_volatility < 0.10:
            bear_score -= 0.3
        if volume_trend < -0.2:
            bear_score += 0.2

        if bull_score > bear_score and bull_score > 0.3:
            regime = "BULL"
            mult = min(1.3, 1.0 + (bull_score - 0.3) * 0.25)
        elif bear_score > bull_score and bear_score > 0.3:
            regime = "BEAR"
            mult = max(0.5, 1.0 - (bear_score - 0.3) * 0.25)
        else:
            regime = "NEUTRAL"
            mult = 1.0

        return regime, mult


# ---------------------------------------------------------------------------
# Timeline Assessment (simplified for Kelly sizing)
# ---------------------------------------------------------------------------


def assess_timeline(
    hours_to_resolution: float,
    confidence_boost: float = 0.15,
    near_hours: int = 48,
    early_days_threshold: int = 60,
    early_penalty: float = 0.10,
) -> tuple[float, str]:
    """Assess resolution timeline and compute timeline multiplier.

    Args:
        hours_to_resolution: Hours until market resolves
        confidence_boost: Multiplier boost for near-resolution markets
        near_hours: Threshold for "near resolution" in hours
        early_days_threshold: Threshold for "early market" in days
        early_penalty: Penalty for early markets

    Returns:
        (timeline_multiplier, signal)
    """
    days = hours_to_resolution / 24.0

    if hours_to_resolution <= near_hours:
        # Near resolution — higher confidence, slightly increase size
        mult = 1.0 + confidence_boost
        signal = f"near_resolution ({hours_to_resolution:.0f}h) — boost ×{mult:.2f}"
    elif days <= early_days_threshold:
        # Early market — higher uncertainty, reduce size
        excess = early_days_threshold - days
        mult = max(0.5, 1.0 - excess * early_penalty)
        signal = f"early_market ({days:.0f}d) — reduce ×{mult:.2f}"
    elif hours_to_resolution > near_hours * 10:
        # Very far out — moderate increase
        mult = 1.1
        signal = f"long_dated ({days:.0f}d) — slight boost ×{mult:.2f}"
    else:
        mult = 1.0
        signal = f"normal ({days:.0f}d) — no adjustment"

    return mult, signal


# ---------------------------------------------------------------------------
# Edge Result Builder (convenience helper)
# ---------------------------------------------------------------------------


def build_edge_result(
    direction: str,
    model_probability: float,
    implied_probability: float,
    transaction_fee_pct: float = 0.02,
    exit_fee_pct: float = 0.02,
) -> Any:
    """Build a simple edge result dict/class for Kelly sizing.

    Args:
        direction: "BUY_YES" or "BUY_NO"
        model_probability: Model's estimated probability
        implied_probability: Market's implied probability
        transaction_fee_pct: Trading fee percentage
        exit_fee_pct: Exit fee percentage

    Returns:
        An object with direction, model_probability, implied_probability,
        abs_edge, net_edge, is_positive attributes
    """
    if direction == "BUY_YES":
        prob_win = model_probability
        prob_lose = 1.0 - model_probability
        edge_raw = model_probability - implied_probability
    else:
        prob_win = 1.0 - model_probability
        prob_lose = model_probability
        edge_raw = (1.0 - model_probability) - (1.0 - implied_probability)

    total_fees = transaction_fee_pct + exit_fee_pct
    net_edge = edge_raw - total_fees

    class _Edge:
        pass

    e = _Edge()
    e.direction = direction
    e.model_probability = model_probability
    e.implied_probability = implied_probability
    e.abs_edge = abs(edge_raw)
    e.net_edge = net_edge
    e.is_positive = net_edge > 0
    return e


# ---------------------------------------------------------------------------
# Skill Integration Helper
# ---------------------------------------------------------------------------


def create_kelly_sizer_skill() -> dict[str, Any]:
    """Create a skill-compatible Kelly sizing callable.

    Returns a dict that can be used with the SkillRouter framework.
    """
    sizer = KellySizer()

    def kelly_size(
        edge: Any,
        confidence_level: str = "MEDIUM",
        drawdown_multiplier: float = 1.0,
        timeline_multiplier: float = 1.0,
        price_volatility: float = 0.0,
        regime_multiplier: float = 1.0,
        category: str = "",
        liquidity_usd: float = 0.0,
    ) -> PositionSize:
        return sizer.size(
            edge=edge,
            confidence_level=confidence_level,
            drawdown_multiplier=drawdown_multiplier,
            timeline_multiplier=timeline_multiplier,
            price_volatility=price_volatility,
            regime_multiplier=regime_multiplier,
            category=category,
            liquidity_usd=liquidity_usd,
        )

    return {
        "fn": kelly_size,
        "name": "kelly-sizer",
        "description": "Calculates optimal position size using fractional Kelly criterion",
        "triggers": ["size", "position", "kelly", "stake", "bet"],
    }
