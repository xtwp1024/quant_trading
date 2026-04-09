"""Risk Checks Suite — 15+ deterministic risk checks for Polymarket trading.

Based on Fully-Autonomous-Polymarket-AI-Trading-Bot's risk_limits.py and
portfolio_risk.py, plus custom checks for the Polymarket context.

Checks performed (any BLOCK violation prevents the trade):
  1.  Kill Switch          — manual or drawdown auto-kill
  2.  Drawdown Heat        — heat level 1-3, Kelly multiplier applied
  3.  Max Daily Loss        — prevent catastrophic daily drawdowns
  4.  Max Open Positions    — avoid overtrading / overconcentration
  5.  Min Edge Threshold    — net edge after fees must exceed minimum
  6.  Min Liquidity         — sufficient order-book depth
  7.  Max Spread            — excessive spread erodes edge
  8.  Evidence Quality      — low-quality evidence → block
  9.  Confidence Level      — LOW confidence → block
  10. Min Implied Probability — block micro-probability markets
 11. Edge Direction         — edge must be genuinely positive after costs
 12. Market Type            — reject UNKNOWN / restricted types
  13. Portfolio Exposure    — category/event concentration check
  14. Timeline Endgame       — warn if < 6h to resolution
  15. Dry-Run Gate           — always pass in dry_run mode
  16. Bankroll Sanity        — stake must not exceed bankroll fraction
  17. Min Stake Floor        — reject dust positions

Severity levels:
  - BLOCK: trade is rejected immediately
  - WARN:  logged but does not prevent trade
  - INFO:  informational signal only

Usage:
    results = run_risk_checks(position=position_dict, config=risk_config)
    blocked = [r for r in results if r.severity == 'block' and not r.passed]
    if blocked:
        print("Trade blocked:", blocked)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RiskCheckResult",
    "RiskCheckSeverity",
    "run_risk_checks",
    "check_kill_switch",
    "check_drawdown",
    "check_max_daily_loss",
    "check_max_positions",
    "check_min_edge",
    "check_min_liquidity",
    "check_max_spread",
    "check_evidence_quality",
    "check_confidence_level",
    "check_min_implied_prob",
    "check_edge_direction",
    "check_market_type",
    "check_portfolio_exposure",
    "check_timeline_endgame",
    "check_bankroll_fraction",
    "check_min_stake",
    "RiskConfig",
]


# ---------------------------------------------------------------------------
# Severity & configuration
# ---------------------------------------------------------------------------


class RiskCheckSeverity(str):
    BLOCK = "block"
    WARN = "warn"
    INFO = "info"


# Default thresholds (can be overridden via RiskConfig)
_DEFAULT_KILL_SWITCH = False
_DEFAULT_MAX_DAILY_LOSS = 500.0         # USD
_DEFAULT_MAX_OPEN_POSITIONS = 10
_DEFAULT_MIN_EDGE = 0.02                # 2% minimum net edge
_DEFAULT_MIN_LIQUIDITY = 50.0           # USD depth
_DEFAULT_MAX_SPREAD = 0.05              # 5% of mid-price
_DEFAULT_MIN_EVIDENCE_QUALITY = 0.35
_DEFAULT_MIN_IMPLIED_PROB = 0.01        # floor for micro-prob markets
_DEFAULT_MAX_BANKROLL_FRACTION = 0.05   # max 5% of bankroll per trade
_DEFAULT_MIN_STAKE = 0.50              # USD floor


@dataclass
class RiskConfig:
    """Risk configuration parameters."""
    kill_switch: bool = _DEFAULT_KILL_SWITCH
    max_daily_loss: float = _DEFAULT_MAX_DAILY_LOSS
    max_open_positions: int = _DEFAULT_MAX_OPEN_POSITIONS
    min_edge: float = _DEFAULT_MIN_EDGE
    min_liquidity: float = _DEFAULT_MIN_LIQUIDITY
    max_spread: float = _DEFAULT_MAX_SPREAD
    min_evidence_quality: float = _DEFAULT_MIN_EVIDENCE_QUALITY
    min_confidence_level: str = "MEDIUM"  # LOW | MEDIUM | HIGH
    min_implied_probability: float = _DEFAULT_MIN_IMPLIED_PROB
    max_bankroll_fraction: float = _DEFAULT_MAX_BANKROLL_FRACTION
    min_stake: float = _DEFAULT_MIN_STAKE
    bankroll: float = 10_000.0          # total bankroll in USD
    max_stake_per_market: float = 200.0  # USD per market


# ---------------------------------------------------------------------------
# RiskCheckResult — single check output
# ---------------------------------------------------------------------------


@dataclass
class RiskCheckResult:
    """Result of a single risk check."""
    passed: bool
    check_name: str
    reason: str | None = None
    severity: str = RiskCheckSeverity.BLOCK  # 'block' | 'warn' | 'info'
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "check_name": self.check_name,
            "reason": self.reason,
            "severity": self.severity,
            **self.details,
        }


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

# Mapping from confidence string to numeric rank
_CONF_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def check_kill_switch(config: RiskConfig, dry_run: bool = True) -> RiskCheckResult:
    """1. Kill switch check — blocks all trading when enabled."""
    if config.kill_switch:
        return RiskCheckResult(
            passed=False,
            check_name="kill_switch",
            reason="Kill switch is ENABLED — all trading blocked",
            severity=RiskCheckSeverity.BLOCK,
        )
    return RiskCheckResult(
        passed=True,
        check_name="kill_switch",
        reason="Kill switch is off",
        severity=RiskCheckSeverity.INFO,
    )


def check_drawdown(
    drawdown_pct: float = 0.0,
    heat_level: int = 0,
) -> tuple[RiskCheckResult, float]:
    """2. Drawdown heat check.

    Returns:
        RiskCheckResult and the Kelly multiplier (heat_level → mult mapping).
    """
    kelly_mult = 1.0
    if heat_level >= 3:
        kelly_mult = 0.25
    elif heat_level == 2:
        kelly_mult = 0.50
    elif heat_level == 1:
        kelly_mult = 0.75

    if drawdown_pct >= 0.25:
        return RiskCheckResult(
            passed=False,
            check_name="drawdown_kill",
            reason=f"Drawdown {drawdown_pct:.1%} >= 25% — auto-kill triggered",
            severity=RiskCheckSeverity.BLOCK,
            details={"drawdown_pct": drawdown_pct, "heat_level": heat_level},
        ), kelly_mult

    if drawdown_pct >= 0.20:
        return RiskCheckResult(
            passed=False,
            check_name="drawdown_limit",
            reason=f"Drawdown {drawdown_pct:.1%} >= 20% hard limit",
            severity=RiskCheckSeverity.BLOCK,
            details={"drawdown_pct": drawdown_pct},
        ), kelly_mult

    if heat_level > 0:
        return RiskCheckResult(
            passed=True,
            check_name="drawdown_heat",
            reason=f"Drawdown heat level {heat_level}, Kelly mult {kelly_mult:.2f}",
            severity=RiskCheckSeverity.WARN,
            details={"drawdown_pct": drawdown_pct, "heat_level": heat_level, "kelly_mult": kelly_mult},
        ), kelly_mult

    return RiskCheckResult(
        passed=True,
        check_name="drawdown",
        reason="Drawdown within acceptable range",
        severity=RiskCheckSeverity.INFO,
        details={"drawdown_pct": drawdown_pct, "heat_level": 0},
    ), kelly_mult


def check_max_daily_loss(
    daily_pnl: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """3. Maximum daily loss check."""
    if daily_pnl < 0 and abs(daily_pnl) >= config.max_daily_loss:
        return RiskCheckResult(
            passed=False,
            check_name="max_daily_loss",
            reason=f"Daily loss ${abs(daily_pnl):.2f} >= limit ${config.max_daily_loss:.2f}",
            severity=RiskCheckSeverity.BLOCK,
            details={"daily_pnl": daily_pnl, "limit": config.max_daily_loss},
        )
    return RiskCheckResult(
        passed=True,
        check_name="max_daily_loss",
        reason=f"Daily loss ${abs(daily_pnl):.2f} < limit ${config.max_daily_loss:.2f}",
        severity=RiskCheckSeverity.INFO,
        details={"daily_pnl": daily_pnl},
    )


def check_max_positions(
    current_open: int,
    config: RiskConfig,
) -> RiskCheckResult:
    """4. Maximum open positions check."""
    if current_open >= config.max_open_positions:
        return RiskCheckResult(
            passed=False,
            check_name="max_open_positions",
            reason=f"{current_open} open positions >= limit {config.max_open_positions}",
            severity=RiskCheckSeverity.BLOCK,
            details={"current_open": current_open, "limit": config.max_open_positions},
        )
    return RiskCheckResult(
        passed=True,
        check_name="max_open_positions",
        reason=f"{current_open} open < limit {config.max_open_positions}",
        severity=RiskCheckSeverity.INFO,
        details={"current_open": current_open},
    )


def check_min_edge(
    net_edge: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """5. Minimum edge threshold check (uses net edge after fees)."""
    abs_edge = abs(net_edge)
    if abs_edge < config.min_edge:
        return RiskCheckResult(
            passed=False,
            check_name="min_edge",
            reason=f"Net edge |{abs_edge:.4f}| < threshold {config.min_edge}",
            severity=RiskCheckSeverity.BLOCK,
            details={"net_edge": net_edge, "threshold": config.min_edge},
        )
    return RiskCheckResult(
        passed=True,
        check_name="min_edge",
        reason=f"Net edge {net_edge:.4f} >= {config.min_edge}",
        severity=RiskCheckSeverity.INFO,
        details={"net_edge": net_edge},
    )


def check_min_liquidity(
    liquidity_usd: float,
    stake_usd: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """6. Minimum liquidity check — ensures stake is a small fraction of depth."""
    if liquidity_usd <= 0:
        return RiskCheckResult(
            passed=False,
            check_name="min_liquidity",
            reason="No liquidity data available",
            severity=RiskCheckSeverity.WARN,
            details={"liquidity_usd": 0.0},
        )

    if liquidity_usd < config.min_liquidity:
        return RiskCheckResult(
            passed=False,
            check_name="min_liquidity",
            reason=f"Depth ${liquidity_usd:.2f} < minimum ${config.min_liquidity:.2f}",
            severity=RiskCheckSeverity.BLOCK,
            details={"liquidity_usd": liquidity_usd, "threshold": config.min_liquidity},
        )

    # Soft check: stake should not be > 10% of liquidity
    if stake_usd > 0 and liquidity_usd > 0 and stake_usd > 0.10 * liquidity_usd:
        return RiskCheckResult(
            passed=True,
            check_name="liquidity_concentration",
            reason=f"Stake ${stake_usd:.2f} is > 10% of depth ${liquidity_usd:.2f}",
            severity=RiskCheckSeverity.WARN,
            details={"stake_usd": stake_usd, "liquidity_usd": liquidity_usd},
        )

    return RiskCheckResult(
        passed=True,
        check_name="min_liquidity",
        reason=f"Depth ${liquidity_usd:.2f} >= ${config.min_liquidity:.2f}",
        severity=RiskCheckSeverity.INFO,
        details={"liquidity_usd": liquidity_usd},
    )


def check_max_spread(
    spread_pct: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """7. Maximum spread check — wide spreads erode edge."""
    if spread_pct <= 0:
        return RiskCheckResult(
            passed=True,
            check_name="spread",
            reason="No spread data available",
            severity=RiskCheckSeverity.INFO,
            details={"spread_pct": 0.0},
        )

    if spread_pct > config.max_spread:
        return RiskCheckResult(
            passed=False,
            check_name="max_spread",
            reason=f"Spread {spread_pct:.2%} > threshold {config.max_spread:.2%}",
            severity=RiskCheckSeverity.BLOCK,
            details={"spread_pct": spread_pct, "threshold": config.max_spread},
        )
    return RiskCheckResult(
        passed=True,
        check_name="spread",
        reason=f"Spread {spread_pct:.2%} <= {config.max_spread:.2%}",
        severity=RiskCheckSeverity.INFO,
        details={"spread_pct": spread_pct},
    )


def check_evidence_quality(
    evidence_quality: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """8. Evidence quality threshold check."""
    if evidence_quality < config.min_evidence_quality:
        return RiskCheckResult(
            passed=False,
            check_name="evidence_quality",
            reason=f"Evidence quality {evidence_quality:.2f} < threshold {config.min_evidence_quality:.2f}",
            severity=RiskCheckSeverity.BLOCK,
            details={"evidence_quality": evidence_quality, "threshold": config.min_evidence_quality},
        )
    return RiskCheckResult(
        passed=True,
        check_name="evidence_quality",
        reason=f"Evidence quality {evidence_quality:.2f} >= {config.min_evidence_quality:.2f}",
        severity=RiskCheckSeverity.INFO,
        details={"evidence_quality": evidence_quality},
    )


def check_confidence_level(
    confidence_level: str,
    config: RiskConfig,
) -> RiskCheckResult:
    """9. Confidence level filter — LOW confidence is blocked by default."""
    min_conf = config.min_confidence_level
    rank_current = _CONF_RANK.get(confidence_level, 0)
    rank_min = _CONF_RANK.get(min_conf, 1)

    if rank_current < rank_min:
        return RiskCheckResult(
            passed=False,
            check_name="confidence_level",
            reason=f"Confidence '{confidence_level}' < minimum '{min_conf}'",
            severity=RiskCheckSeverity.BLOCK,
            details={"confidence_level": confidence_level, "min_required": min_conf},
        )
    return RiskCheckResult(
        passed=True,
        check_name="confidence_level",
        reason=f"Confidence '{confidence_level}' >= '{min_conf}'",
        severity=RiskCheckSeverity.INFO,
        details={"confidence_level": confidence_level},
    )


def check_min_implied_prob(
    implied_prob: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """10. Minimum implied probability — blocks micro-probability markets."""
    if implied_prob < config.min_implied_probability:
        return RiskCheckResult(
            passed=False,
            check_name="min_implied_prob",
            reason=f"Implied prob {implied_prob:.2%} < floor {config.min_implied_probability:.2%}",
            severity=RiskCheckSeverity.BLOCK,
            details={"implied_prob": implied_prob, "floor": config.min_implied_probability},
        )
    return RiskCheckResult(
        passed=True,
        check_name="min_implied_prob",
        reason=f"Implied prob {implied_prob:.2%} >= floor {config.min_implied_probability:.2%}",
        severity=RiskCheckSeverity.INFO,
        details={"implied_prob": implied_prob},
    )


def check_edge_direction(
    net_edge: float,
    prediction: str,  # 'yes' or 'no'
    model_prob: float,
    implied_prob: float,
) -> RiskCheckResult:
    """11. Edge direction check — edge must be positive in the right direction."""
    if prediction.lower() == "yes":
        # Model thinks YES is more likely than market implies
        if net_edge <= 0:
            return RiskCheckResult(
                passed=False,
                check_name="edge_direction",
                reason=f"Positive edge required for YES bet, got {net_edge:.4f}",
                severity=RiskCheckSeverity.BLOCK,
                details={"net_edge": net_edge, "prediction": prediction},
            )
    else:  # 'no'
        # Edge for NO: (1 - model_prob) > (1 - implied_prob) → implied_prob > model_prob
        edge_for_no = implied_prob - model_prob
        if edge_for_no <= 0:
            return RiskCheckResult(
                passed=False,
                check_name="edge_direction",
                reason=f"Positive edge required for NO bet, got {edge_for_no:.4f}",
                severity=RiskCheckSeverity.BLOCK,
                details={"edge_for_no": edge_for_no, "prediction": prediction},
            )

    direction = "YES" if prediction.lower() == "yes" else "NO"
    return RiskCheckResult(
        passed=True,
        check_name="edge_direction",
        reason=f"Edge direction positive for {direction}",
        severity=RiskCheckSeverity.INFO,
        details={"net_edge": net_edge, "prediction": prediction},
    )


def check_market_type(
    market_type: str,
    allowed_types: list[str] | None = None,
    restricted_types: list[str] | None = None,
) -> RiskCheckResult:
    """12. Market type check — reject UNKNOWN / restricted types."""
    if restricted_types and market_type in restricted_types:
        return RiskCheckResult(
            passed=False,
            check_name="market_type",
            reason=f"Market type '{market_type}' is restricted",
            severity=RiskCheckSeverity.BLOCK,
            details={"market_type": market_type, "restricted": restricted_types},
        )

    if market_type == "UNKNOWN":
        return RiskCheckResult(
            passed=False,
            check_name="market_type",
            reason="Cannot classify market type — rejecting",
            severity=RiskCheckSeverity.BLOCK,
            details={"market_type": "UNKNOWN"},
        )

    if allowed_types and market_type not in allowed_types:
        return RiskCheckResult(
            passed=True,
            check_name="market_type",
            reason=f"Market type '{market_type}' not in preferred list",
            severity=RiskCheckSeverity.WARN,
            details={"market_type": market_type, "allowed": allowed_types},
        )

    return RiskCheckResult(
        passed=True,
        check_name="market_type",
        reason=f"Market type '{market_type}' is acceptable",
        severity=RiskCheckSeverity.INFO,
        details={"market_type": market_type},
    )


def check_portfolio_exposure(
    category_exposure: dict[str, float] | None = None,
    event_exposure: dict[str, float] | None = None,
    max_category_fraction: float = 0.30,
    max_event_fraction: float = 0.20,
    bankroll: float = 10_000.0,
) -> RiskCheckResult:
    """13. Portfolio exposure check — prevents category/event concentration."""
    category_exposure = category_exposure or {}
    event_exposure = event_exposure or {}

    violations: list[str] = []

    for cat, fraction in category_exposure.items():
        if fraction > max_category_fraction:
            violations.append(f"Category '{cat}': {fraction:.1%} > {max_category_fraction:.1%}")

    for evt, fraction in event_exposure.items():
        if fraction > max_event_fraction:
            violations.append(f"Event '{evt}': {fraction:.1%} > {max_event_fraction:.1%}")

    if violations:
        return RiskCheckResult(
            passed=False,
            check_name="portfolio_exposure",
            reason="; ".join(violations),
            severity=RiskCheckSeverity.BLOCK,
            details={
                "category_exposure": category_exposure,
                "event_exposure": event_exposure,
                "max_category_fraction": max_category_fraction,
                "max_event_fraction": max_event_fraction,
            },
        )

    return RiskCheckResult(
        passed=True,
        check_name="portfolio_exposure",
        reason="Portfolio exposure within limits",
        severity=RiskCheckSeverity.INFO,
        details={"category_exposure": category_exposure, "event_exposure": event_exposure},
    )


def check_timeline_endgame(
    hours_to_resolution: float | None = None,
    is_resolved: bool = False,
) -> RiskCheckResult:
    """14. Timeline endgame check — warn when resolution is imminent."""
    if is_resolved:
        return RiskCheckResult(
            passed=False,
            check_name="timeline",
            reason="Market is already resolved",
            severity=RiskCheckSeverity.BLOCK,
            details={"is_resolved": True},
        )

    if hours_to_resolution is None:
        return RiskCheckResult(
            passed=True,
            check_name="timeline",
            reason="No timeline data available",
            severity=RiskCheckSeverity.INFO,
            details={"hours_to_resolution": None},
        )

    if hours_to_resolution < 1:
        return RiskCheckResult(
            passed=False,
            check_name="timeline_endgame",
            reason=f"Less than 1 hour to resolution — too late to trade",
            severity=RiskCheckSeverity.BLOCK,
            details={"hours_to_resolution": hours_to_resolution},
        )

    if hours_to_resolution < 6:
        return RiskCheckResult(
            passed=True,
            check_name="timeline_endgame",
            reason=f"Only {hours_to_resolution:.1f}h to resolution — consider exit only",
            severity=RiskCheckSeverity.WARN,
            details={"hours_to_resolution": hours_to_resolution},
        )

    return RiskCheckResult(
        passed=True,
        check_name="timeline",
        reason=f"{hours_to_resolution:.1f}h to resolution — OK",
        severity=RiskCheckSeverity.INFO,
        details={"hours_to_resolution": hours_to_resolution},
    )


def check_bankroll_fraction(
    stake_usd: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """15. Bankroll fraction check — stake must not exceed max fraction."""
    fraction = stake_usd / config.bankroll if config.bankroll > 0 else float("inf")

    if fraction > config.max_bankroll_fraction:
        return RiskCheckResult(
            passed=False,
            check_name="bankroll_fraction",
            reason=f"Stake ${stake_usd:.2f} ({fraction:.2%}) > max {config.max_bankroll_fraction:.2%} of bankroll",
            severity=RiskCheckSeverity.BLOCK,
            details={
                "stake_usd": stake_usd,
                "bankroll": config.bankroll,
                "fraction": fraction,
                "max_fraction": config.max_bankroll_fraction,
            },
        )

    return RiskCheckResult(
        passed=True,
        check_name="bankroll_fraction",
        reason=f"Stake ${stake_usd:.2f} ({fraction:.2%}) <= {config.max_bankroll_fraction:.2%}",
        severity=RiskCheckSeverity.INFO,
        details={"stake_usd": stake_usd, "fraction": fraction},
    )


def check_min_stake(
    stake_usd: float,
    config: RiskConfig,
) -> RiskCheckResult:
    """16. Minimum stake floor — avoids dust trades that are fee-inefficient."""
    if 0 < stake_usd < config.min_stake:
        return RiskCheckResult(
            passed=False,
            check_name="min_stake",
            reason=f"Stake ${stake_usd:.2f} < minimum ${config.min_stake:.2f}",
            severity=RiskCheckSeverity.BLOCK,
            details={"stake_usd": stake_usd, "min_stake": config.min_stake},
        )
    return RiskCheckResult(
        passed=True,
        check_name="min_stake",
        reason=f"Stake ${stake_usd:.2f} >= minimum ${config.min_stake:.2f}",
        severity=RiskCheckSeverity.INFO,
        details={"stake_usd": stake_usd},
    )


def check_dry_run_gate(dry_run: bool) -> RiskCheckResult:
    """17. Dry-run safety gate — logs the mode without blocking."""
    return RiskCheckResult(
        passed=True,
        check_name="dry_run_gate",
        reason="Dry-run mode active — no real orders will be submitted" if dry_run else "Live trading mode",
        severity=RiskCheckSeverity.INFO,
        details={"dry_run": dry_run},
    )


# ---------------------------------------------------------------------------
# Bulk runner
# ---------------------------------------------------------------------------


def run_risk_checks(
    position: dict[str, Any],
    config: RiskConfig | None = None,
) -> list[RiskCheckResult]:
    """Run all 17 risk checks against a position.

    Args:
        position: Dict with keys:
            - prediction: 'yes' or 'no'
            - stake_usd: proposed stake in USD
            - net_edge: expected edge after fees
            - model_prob: model's probability
            - implied_prob: market-implied probability
            - spread_pct: current spread as fraction
            - liquidity_usd: available liquidity in USD
            - evidence_quality: 0.0–1.0 evidence quality score
            - confidence_level: 'LOW' | 'MEDIUM' | 'HIGH'
            - market_type: market classification string
            - hours_to_resolution: hours until market resolves
            - is_resolved: bool
            - category_exposure: dict[str, float]
            - event_exposure: dict[str, float]
            - daily_pnl: today's PnL in USD
            - current_open: number of currently open positions
            - drawdown_pct: current drawdown as fraction
            - drawdown_heat: heat level 0-3
        config: RiskConfig with thresholds.

    Returns:
        List of RiskCheckResult, one per check.
        BLOCK-severity failures indicate the trade should be rejected.
    """
    config = config or RiskConfig()
    p = position
    results: list[RiskCheckResult] = []

    # 1. Kill switch
    results.append(check_kill_switch(config, dry_run=p.get("dry_run", True)))

    # 2. Drawdown
    dd_result, kelly_mult = check_drawdown(
        drawdown_pct=p.get("drawdown_pct", 0.0),
        heat_level=p.get("drawdown_heat", 0),
    )
    results.append(dd_result)

    # 3. Max daily loss
    results.append(check_max_daily_loss(
        daily_pnl=p.get("daily_pnl", 0.0),
        config=config,
    ))

    # 4. Max open positions
    results.append(check_max_positions(
        current_open=p.get("current_open", 0),
        config=config,
    ))

    # 5. Min edge
    results.append(check_min_edge(
        net_edge=p.get("net_edge", 0.0),
        config=config,
    ))

    # 6. Min liquidity
    results.append(check_min_liquidity(
        liquidity_usd=p.get("liquidity_usd", 0.0),
        stake_usd=p.get("stake_usd", 0.0),
        config=config,
    ))

    # 7. Max spread
    results.append(check_max_spread(
        spread_pct=p.get("spread_pct", 0.0),
        config=config,
    ))

    # 8. Evidence quality
    results.append(check_evidence_quality(
        evidence_quality=p.get("evidence_quality", 0.0),
        config=config,
    ))

    # 9. Confidence level
    results.append(check_confidence_level(
        confidence_level=p.get("confidence_level", "LOW"),
        config=config,
    ))

    # 10. Min implied probability
    results.append(check_min_implied_prob(
        implied_prob=p.get("implied_prob", 0.0),
        config=config,
    ))

    # 11. Edge direction
    results.append(check_edge_direction(
        net_edge=p.get("net_edge", 0.0),
        prediction=p.get("prediction", "yes"),
        model_prob=p.get("model_prob", 0.5),
        implied_prob=p.get("implied_prob", 0.5),
    ))

    # 12. Market type
    results.append(check_market_type(
        market_type=p.get("market_type", "UNKNOWN"),
        allowed_types=p.get("allowed_types"),
        restricted_types=p.get("restricted_types"),
    ))

    # 13. Portfolio exposure
    results.append(check_portfolio_exposure(
        category_exposure=p.get("category_exposure"),
        event_exposure=p.get("event_exposure"),
        max_category_fraction=p.get("max_category_fraction", 0.30),
        max_event_fraction=p.get("max_event_fraction", 0.20),
        bankroll=config.bankroll,
    ))

    # 14. Timeline
    results.append(check_timeline_endgame(
        hours_to_resolution=p.get("hours_to_resolution"),
        is_resolved=p.get("is_resolved", False),
    ))

    # 15. Bankroll fraction
    results.append(check_bankroll_fraction(
        stake_usd=p.get("stake_usd", 0.0),
        config=config,
    ))

    # 16. Min stake
    results.append(check_min_stake(
        stake_usd=p.get("stake_usd", 0.0),
        config=config,
    ))

    # 17. Dry run gate
    results.append(check_dry_run_gate(dry_run=p.get("dry_run", True)))

    return results
