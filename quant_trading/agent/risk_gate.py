"""Risk gate system — deterministic risk policy enforcement.

Based on Fully-Autonomous-Polymarket-AI-Trading-Bot's risk_limits + portfolio_risk
architecture. Applies 15+ risk checks before allowing any trade.

Checks:
  1. Kill switch (manual + drawdown auto-kill)
  2. Drawdown heat level
  3. Maximum stake per market
  4. Maximum daily loss
  5. Maximum open positions
  6. Minimum edge threshold (net edge after fees)
  7. Minimum liquidity
  8. Maximum spread
  9. Evidence quality threshold
  10. Confidence level filter (reject LOW)
  11. Minimum implied probability floor
  12. Edge direction (positive after costs)
  13. Market type check
  14. Portfolio category/event exposure
  15. Timeline endgame check (near-resolution warning)
  16. Dry-run safety gate (three-stage)

Three-Stage Dry-Run Safety Gate:
  Stage 1 — Pre-trade: Verify paper_mode, bankroll sanity, config integrity
  Stage 2 — Pre-order: Verify position doesn't breach limits at moment of order
  Stage 3 — Pre-execution: Final sanity check before order submission

Key concepts:
- Any single violation → NO TRADE (failsafe principle)
- Warnings are logged but do not block trades
- Portfolio-level gates evaluated before per-trade gates
- Dry-run mode prevents any real order submission
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Decision(str):
    TRADE = "TRADE"
    NO_TRADE = "NO TRADE"


@dataclass
class RiskCheckResult:
    """Result of risk gate evaluation."""
    allowed: bool
    decision: str  # "TRADE" | "NO TRADE"
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    # Drawdown state
    drawdown_heat: int = 0
    drawdown_pct: float = 0.0
    kelly_multiplier: float = 1.0
    # Portfolio state
    portfolio_gate: str = "ok"
    # Safety gate stage
    safety_gate_stage: str = "passed"
    # Cost breakdown
    estimated_fees_pct: float = 0.0
    estimated_gas_usd: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "decision": self.decision,
            "violations": self.violations,
            "warnings": self.warnings,
            "checks_passed": self.checks_passed,
            "drawdown_heat": self.drawdown_heat,
            "drawdown_pct": round(self.drawdown_pct, 4),
            "kelly_multiplier": round(self.kelly_multiplier, 4),
            "portfolio_gate": self.portfolio_gate,
            "safety_gate_stage": self.safety_gate_stage,
            "estimated_fees_pct": round(self.estimated_fees_pct, 4),
        }


@dataclass
class MarketFeatures:
    """Market data snapshot for risk evaluation."""
    market_id: str = ""
    question: str = ""
    market_type: str = "UNKNOWN"
    category: str = ""
    event_slug: str = ""
    # Liquidity & volume
    volume_usd: float = 0.0
    liquidity_usd: float = 0.0
    bid_depth_5: float = 0.0   # Top-5 bid depth in USD
    ask_depth_5: float = 0.0   # Top-5 ask depth in USD
    # Spread
    spread_pct: float = 0.0
    # Price
    current_price: float = 0.0  # YES price (implied probability)
    # Timeline
    hours_to_resolution: float = 720.0
    is_near_resolution: bool = False
    # Evidence
    evidence_quality: float = 0.5
    num_sources: int = 0
    has_clear_resolution: bool = True
    # Volatility
    price_volatility: float = 0.0
    # Top evidence bullets
    top_bullets: list[str] = field(default_factory=list)


@dataclass
class EdgeResult:
    """Edge calculation result for a potential trade."""
    direction: str = "BUY_YES"  # "BUY_YES" | "BUY_NO"
    model_probability: float = 0.5
    implied_probability: float = 0.5
    abs_edge: float = 0.0
    net_edge: float = 0.0  # After fees
    is_positive: bool = False
    confidence_level: str = "MEDIUM"
    entry_price: float = 0.0
    exit_fee_pct: float = 0.02
    transaction_fee_pct: float = 0.02


@dataclass
class PortfolioState:
    """Snapshot of current portfolio for exposure checks."""
    positions: list[dict[str, Any]] = field(default_factory=list)
    total_exposure_usd: float = 0.0
    daily_pnl: float = 0.0
    open_position_count: int = 0
    category_exposures: dict[str, float] = field(default_factory=dict)
    event_exposures: dict[str, float] = field(default_factory=dict)
    bankroll: float = 5000.0


@dataclass
class RiskConfig:
    """Risk configuration parameters."""
    max_stake_per_market: float = 50.0
    max_daily_loss: float = 500.0
    max_open_positions: int = 25
    min_edge: float = 0.04
    min_liquidity: float = 2000.0
    min_volume: float = 1000.0
    max_spread: float = 0.06
    kelly_fraction: float = 0.25
    max_bankroll_fraction: float = 0.05
    kill_switch: bool = False
    bankroll: float = 5000.0
    transaction_fee_pct: float = 0.02
    exit_fee_pct: float = 0.02
    gas_cost_usd: float = 0.01
    min_implied_probability: float = 0.05
    stop_loss_pct: float = 0.20
    take_profit_pct: float = 0.30
    max_holding_hours: float = 240.0
    min_stake_usd: float = 1.0
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
    # Portfolio limits
    max_category_exposure_pct: float = 0.35
    max_single_event_exposure_pct: float = 0.25
    max_correlated_positions: int = 4
    # Drawdown
    max_drawdown_pct: float = 0.20
    auto_reduce_at_warning: float = 0.50
    auto_reduce_at_critical: float = 0.25


# ---------------------------------------------------------------------------
# Drawdown Manager (simplified)
# ---------------------------------------------------------------------------


@dataclass
class DrawdownState:
    """Drawdown tracking state."""
    current_drawdown_pct: float = 0.0
    heat_level: int = 0  # 0=healthy, 1=warning, 2=critical
    kelly_multiplier: float = 1.0
    is_killed: bool = False
    consecutive_losses: int = 0


def assess_drawdown(
    portfolio_state: PortfolioState,
    risk_config: RiskConfig,
) -> DrawdownState:
    """Assess current drawdown state and compute heat level + Kelly multiplier.

    Returns DrawdownState with:
      - current_drawdown_pct: Peak-to-current drawdown as fraction
      - heat_level: 0 (healthy), 1 (warning), 2 (critical)
      - kelly_multiplier: Kelly fraction multiplier (0–1)
      - is_killed: True if drawdown exceeds max_drawdown_pct
      - consecutive_losses: Recent loss streak
    """
    dd_pct = portfolio_state.daily_pnl / max(portfolio_state.bankroll, 1.0)
    dd_pct = abs(dd_pct) if dd_pct < 0 else 0.0

    heat = 0
    kelly_mult = 1.0
    is_killed = False

    if dd_pct >= risk_config.max_drawdown_pct:
        heat = 2
        kelly_mult = 0.0
        is_killed = True
    elif dd_pct >= risk_config.max_drawdown_pct * risk_config.auto_reduce_at_critical:
        heat = 2
        kelly_mult = risk_config.auto_reduce_at_critical
    elif dd_pct >= risk_config.max_drawdown_pct * risk_config.auto_reduce_at_warning:
        heat = 1
        kelly_mult = risk_config.auto_reduce_at_warning

    # Consecutive losses reduce Kelly further
    if portfolio_state.daily_pnl < 0:
        loss_ratio = abs(portfolio_state.daily_pnl) / max(risk_config.max_daily_loss, 1.0)
        if loss_ratio > 0.5:
            kelly_mult *= 0.5

    return DrawdownState(
        current_drawdown_pct=dd_pct,
        heat_level=heat,
        kelly_multiplier=max(0.0, min(1.0, kelly_mult)),
        is_killed=is_killed,
    )


# ---------------------------------------------------------------------------
# Portfolio Exposure Check
# ---------------------------------------------------------------------------


def check_portfolio_exposure(
    portfolio_state: PortfolioState,
    new_category: str,
    new_event: str,
    new_size_usd: float,
    risk_config: RiskConfig,
) -> tuple[bool, str]:
    """Check if adding a new position would breach portfolio exposure limits.

    Args:
        portfolio_state: Current portfolio snapshot
        new_category: Category of the new position
        new_event: Event slug of the new position
        new_size_usd: Size of the new position in USD

    Returns:
        (can_add, reason) — can_add=False if any limit would be breached
    """
    # Category exposure check
    cat_exposure = portfolio_state.category_exposures.get(new_category, 0.0)
    new_cat_pct = (cat_exposure + new_size_usd) / portfolio_state.bankroll
    if new_cat_pct > risk_config.max_category_exposure_pct:
        return False, (
            f"Category '{new_category}' would reach {new_cat_pct:.1%} "
            f"(limit {risk_config.max_category_exposure_pct:.0%})"
        )

    # Event exposure check
    evt_exposure = portfolio_state.event_exposures.get(new_event, 0.0)
    new_evt_pct = (evt_exposure + new_size_usd) / portfolio_state.bankroll
    if new_evt_pct > risk_config.max_single_event_exposure_pct:
        return False, (
            f"Event '{new_event}' would reach {new_evt_pct:.1%} "
            f"(limit {risk_config.max_single_event_exposure_pct:.0%})"
        )

    # Single position concentration
    if new_size_usd / portfolio_state.bankroll > 0.25:
        return False, f"Single position ${new_size_usd:.0f} > 25% of bankroll"

    # Correlated positions (same category count as correlated)
    same_cat_count = sum(
        1 for p in portfolio_state.positions
        if p.get("category") == new_category
    ) + 1
    if same_cat_count > risk_config.max_correlated_positions:
        return False, (
            f"Would create {same_cat_count} positions in '{new_category}' "
            f"(limit {risk_config.max_correlated_positions})"
        )

    return True, "ok"


# ---------------------------------------------------------------------------
# Three-Stage Dry-Run Safety Gate
# ---------------------------------------------------------------------------


class DryRunSafetyGate:
    """Three-stage safety gate for dry-run / paper trading validation.

    Stage 1 — Pre-trade: Config integrity, paper_mode flag, bankroll sanity
    Stage 2 — Pre-order: Position limit check at moment of order
    Stage 3 — Pre-execution: Final sanity before order submission

    Usage::

        gate = DryRunSafetyGate(config=risk_config, dry_run=True)
        result = gate.evaluate_stage1(portfolio_state=portfolio)
        if not result.allowed:
            return  # Block trade
    """

    def __init__(
        self,
        risk_config: RiskConfig | None = None,
        dry_run: bool = True,
    ):
        self._config = risk_config or RiskConfig()
        self._dry_run = dry_run

    def stage1_pre_trade(self, portfolio_state: PortfolioState) -> RiskCheckResult:
        """Stage 1: Pre-trade validation.

        Validates:
          - dry_run mode is set correctly
          - bankroll is positive and sane
          - kill_switch is not active
          - daily loss hasn't exceeded threshold
        """
        violations: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []

        # Dry-run check
        if self._dry_run:
            passed.append("dry_run: enabled (paper mode)")
        else:
            violations.append("SAFETY: Live trading not enabled — dry_run must be True")

        # Bankroll sanity
        if self._config.bankroll <= 0:
            violations.append(f"SAFETY: Bankroll ${self._config.bankroll} is not positive")
        else:
            passed.append(f"bankroll: ${self._config.bankroll:.2f} (sane)")

        # Kill switch
        if self._config.kill_switch:
            violations.append("SAFETY: Kill switch is active")
        else:
            passed.append("kill_switch: inactive")

        # Daily loss
        if abs(portfolio_state.daily_pnl) >= self._config.max_daily_loss:
            violations.append(
                f"SAFETY: Daily loss ${abs(portfolio_state.daily_pnl):.2f} >= "
                f"limit ${self._config.max_daily_loss:.2f}"
            )
        else:
            passed.append(f"daily_loss: ${abs(portfolio_state.daily_pnl):.2f} < limit")

        allowed = len(violations) == 0
        return RiskCheckResult(
            allowed=allowed,
            decision=Decision.TRADE if allowed else Decision.NO_TRADE,
            violations=violations,
            warnings=warnings,
            checks_passed=passed,
            safety_gate_stage="stage1_pre_trade",
        )

    def stage2_pre_order(
        self,
        portfolio_state: PortfolioState,
        proposed_size_usd: float,
        market_id: str,
        category: str,
        event_slug: str,
    ) -> RiskCheckResult:
        """Stage 2: Pre-order validation.

        Validates:
          - Proposed size doesn't breach max_stake_per_market
          - Total open positions doesn't exceed max_open_positions
          - Portfolio exposure limits are not breached
        """
        violations: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []

        # Max stake per market
        if proposed_size_usd > self._config.max_stake_per_market:
            violations.append(
                f"SAFETY: Proposed stake ${proposed_size_usd:.2f} > "
                f"max ${self._config.max_stake_per_market:.2f}"
            )
        else:
            passed.append(f"stake_size: ${proposed_size_usd:.2f} <= max")

        # Max open positions
        if portfolio_state.open_position_count >= self._config.max_open_positions:
            violations.append(
                f"SAFETY: Open positions {portfolio_state.open_position_count} >= "
                f"limit {self._config.max_open_positions}"
            )
        else:
            passed.append(f"open_positions: {portfolio_state.open_position_count} < limit")

        # Portfolio exposure
        can_add, reason = check_portfolio_exposure(
            portfolio_state=portfolio_state,
            new_category=category,
            new_event=event_slug,
            new_size_usd=proposed_size_usd,
            risk_config=self._config,
        )
        if not can_add:
            violations.append(f"SAFETY: {reason}")
        else:
            passed.append("portfolio_exposure: OK")

        allowed = len(violations) == 0
        return RiskCheckResult(
            allowed=allowed,
            decision=Decision.TRADE if allowed else Decision.NO_TRADE,
            violations=violations,
            warnings=warnings,
            checks_passed=passed,
            safety_gate_stage="stage2_pre_order",
        )

    def stage3_pre_execution(
        self,
        edge: EdgeResult,
        features: MarketFeatures,
        portfolio_state: PortfolioState,
    ) -> RiskCheckResult:
        """Stage 3: Pre-execution final sanity check.

        Validates:
          - Edge is still positive after current spread/fees
          - Market liquidity hasn't dried up since evaluation
          - Price hasn't moved significantly (slippage check)
        """
        violations: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []

        # Positive edge check
        if not edge.is_positive:
            violations.append(
                f"SAFETY: Edge no longer positive (net_edge={edge.net_edge:.4f})"
            )
        else:
            passed.append(f"edge: positive ({edge.net_edge:.4f})")

        # Liquidity check
        total_depth = features.bid_depth_5 + features.ask_depth_5
        if total_depth > 0 and total_depth < self._config.min_liquidity:
            violations.append(
                f"SAFETY: Liquidity ${total_depth:.2f} < min ${self._config.min_liquidity:.2f}"
            )
        elif total_depth > 0:
            passed.append(f"liquidity: ${total_depth:.2f} >= ${self._config.min_liquidity:.2f}")
        else:
            warnings.append("LIQUIDITY: No depth data available")

        # Spread check
        if features.spread_pct > self._config.max_spread:
            violations.append(
                f"SAFETY: Spread {features.spread_pct:.2%} > max {self._config.max_spread:.2%}"
            )
        elif features.spread_pct > 0:
            passed.append(f"spread: {features.spread_pct:.2%} <= max")

        # Slippage check (price moved more than 1% since evaluation)
        slippage_tolerance = 0.01
        if features.current_price > 0 and edge.entry_price > 0:
            slippage = abs(features.current_price - edge.entry_price) / edge.entry_price
            if slippage > slippage_tolerance:
                violations.append(
                    f"SAFETY: Slippage {slippage:.2%} > tolerance {slippage_tolerance:.2%} "
                    f"(entry={edge.entry_price:.4f}, current={features.current_price:.4f})"
                )
            else:
                passed.append(f"slippage: {slippage:.2%} within tolerance")

        allowed = len(violations) == 0
        return RiskCheckResult(
            allowed=allowed,
            decision=Decision.TRADE if allowed else Decision.NO_TRADE,
            violations=violations,
            warnings=warnings,
            checks_passed=passed,
            safety_gate_stage="stage3_pre_execution",
        )


# ---------------------------------------------------------------------------
# Main Risk Gate Evaluator (15+ checks)
# ---------------------------------------------------------------------------


def evaluate_risk_gate(
    edge: EdgeResult,
    features: MarketFeatures,
    portfolio_state: PortfolioState,
    risk_config: RiskConfig,
    confidence_level: str = "MEDIUM",
    min_edge_override: float | None = None,
) -> RiskCheckResult:
    """Run the full 15+ risk checks. Returns TRADE only if ALL pass.

    Checks:
      1. Kill switch (manual)
      2. Drawdown kill switch (automatic)
      3. Minimum edge threshold (net edge after fees)
      4. Maximum daily loss
      5. Maximum open positions
      6. Minimum liquidity
      7. Maximum spread
      8. Evidence quality
      9. Confidence level filter
      10. Minimum implied probability floor
      11. Edge direction (positive after costs)
      12. Market type check
      13. Clear resolution source
      14. Portfolio exposure gate
      15. Timeline endgame check

    Args:
        edge: Edge calculation result
        features: Market features snapshot
        portfolio_state: Current portfolio snapshot
        risk_config: Risk configuration parameters
        confidence_level: LOW / MEDIUM / HIGH
        min_edge_override: Override minimum edge threshold

    Returns:
        RiskCheckResult with decision and detailed breakdown
    """
    violations: list[str] = []
    warnings: list[str] = []
    passed: list[str] = []
    heat_level = 0
    drawdown_pct = 0.0
    kelly_mult = 1.0

    # Drawdown state
    dd_state = assess_drawdown(portfolio_state, risk_config)
    heat_level = dd_state.heat_level
    drawdown_pct = dd_state.current_drawdown_pct
    kelly_mult = dd_state.kelly_multiplier

    # 1. Kill switch (manual)
    if risk_config.kill_switch:
        violations.append("KILL_SWITCH: Trading disabled via kill switch")
    else:
        passed.append("kill_switch: OK")

    # 2. Drawdown kill switch (automatic)
    if dd_state.is_killed:
        violations.append(
            f"DRAWDOWN_KILL: Auto kill-switch at {dd_state.current_drawdown_pct:.1%} drawdown"
        )
    elif drawdown_pct >= risk_config.max_drawdown_pct:
        violations.append(
            f"DRAWDOWN_LIMIT: {drawdown_pct:.1%} >= {risk_config.max_drawdown_pct:.0%} max"
        )
    elif heat_level >= 2:
        warnings.append(
            f"DRAWDOWN_HEAT: Level {heat_level}, Kelly × {kelly_mult:.2f}"
        )
    if heat_level == 0:
        passed.append("drawdown: healthy")

    # 3. Minimum edge (net edge after fees)
    net_edge = edge.net_edge
    abs_net = abs(net_edge)
    effective_min_edge = min_edge_override if min_edge_override is not None else risk_config.min_edge
    if abs_net < effective_min_edge:
        violations.append(
            f"MIN_EDGE: net |edge| {abs_net:.4f} < threshold {effective_min_edge}"
        )
    else:
        passed.append(f"min_edge: {abs_net:.4f} >= {effective_min_edge}")

    # 4. Maximum daily loss
    if portfolio_state.daily_pnl < 0:
        loss = abs(portfolio_state.daily_pnl)
        if loss >= risk_config.max_daily_loss:
            violations.append(
                f"MAX_DAILY_LOSS: ${loss:.2f} >= limit ${risk_config.max_daily_loss:.2f}"
            )
        else:
            passed.append(f"daily_loss: ${loss:.2f} < ${risk_config.max_daily_loss:.2f}")
    else:
        passed.append(f"daily_pnl: +${portfolio_state.daily_pnl:.2f} (healthy)")

    # 5. Maximum open positions
    if portfolio_state.open_position_count >= risk_config.max_open_positions:
        violations.append(
            f"MAX_POSITIONS: {portfolio_state.open_position_count} >= "
            f"limit {risk_config.max_open_positions}"
        )
    else:
        passed.append(
            f"open_positions: {portfolio_state.open_position_count} < "
            f"{risk_config.max_open_positions}"
        )

    # 6. Minimum liquidity
    total_depth = features.bid_depth_5 + features.ask_depth_5
    if total_depth > 0 and total_depth < risk_config.min_liquidity:
        violations.append(
            f"MIN_LIQUIDITY: ${total_depth:.2f} < threshold ${risk_config.min_liquidity:.2f}"
        )
    elif total_depth > 0:
        passed.append(f"liquidity: ${total_depth:.2f} >= ${risk_config.min_liquidity:.2f}")
    else:
        warnings.append("LIQUIDITY: No orderbook depth data")

    # 7. Maximum spread
    if features.spread_pct > 0 and features.spread_pct > risk_config.max_spread:
        violations.append(
            f"MAX_SPREAD: {features.spread_pct:.2%} > threshold {risk_config.max_spread:.2%}"
        )
    elif features.spread_pct > 0:
        passed.append(f"spread: {features.spread_pct:.2%} <= {risk_config.max_spread:.2%}")

    # 8. Evidence quality
    min_evidence = 0.55
    if features.evidence_quality < min_evidence:
        violations.append(
            f"EVIDENCE_QUALITY: {features.evidence_quality:.2f} < threshold {min_evidence:.2f}"
        )
    else:
        passed.append(f"evidence_quality: {features.evidence_quality:.2f} >= {min_evidence:.2f}")

    # 9. Confidence level filter (reject LOW)
    _CONF_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    min_conf = "MEDIUM"
    if _CONF_RANK.get(confidence_level, 0) < _CONF_RANK.get(min_conf, 0):
        violations.append(f"LOW_CONFIDENCE: {confidence_level} < minimum {min_conf}")
    else:
        passed.append(f"confidence: {confidence_level} >= {min_conf}")

    # 10. Minimum implied probability floor
    min_imp = risk_config.min_implied_probability
    if min_imp > 0 and edge.implied_probability < min_imp:
        violations.append(
            f"MIN_IMPLIED_PROB: {edge.implied_probability:.2%} < floor {min_imp:.2%}"
        )
    else:
        passed.append(f"implied_prob: {edge.implied_probability:.2%} >= {min_imp:.2%}")

    # 11. Edge direction
    if not edge.is_positive:
        violations.append(f"NEGATIVE_EDGE: net_edge {edge.net_edge:.4f} not positive")
    else:
        passed.append(f"edge_direction: positive ({edge.net_edge:.4f})")

    # 12. Market type check
    if features.market_type == "UNKNOWN":
        warnings.append("MARKET_TYPE: Could not classify market type")

    # 13. Clear resolution source
    if not features.has_clear_resolution:
        warnings.append("RESOLUTION: No clear resolution source defined")

    # 14. Portfolio exposure gate
    can_add, gate_reason = check_portfolio_exposure(
        portfolio_state=portfolio_state,
        new_category=features.category,
        new_event=features.event_slug,
        new_size_usd=0.0,  # 0 because we just check if a new position is allowed
        risk_config=risk_config,
    )
    if not can_add:
        violations.append(f"PORTFOLIO: {gate_reason}")
    else:
        passed.append("portfolio: OK")

    # 15. Timeline endgame check
    if features.is_near_resolution and features.hours_to_resolution < 6:
        warnings.append(
            f"TIMELINE: Only {features.hours_to_resolution:.1f}h to resolution — "
            "consider exit only"
        )

    # Determine decision
    allowed = len(violations) == 0
    decision = Decision.TRADE if allowed else Decision.NO_TRADE

    # Estimate fees
    estimated_fees = risk_config.transaction_fee_pct + risk_config.exit_fee_pct

    result = RiskCheckResult(
        allowed=allowed,
        decision=decision,
        violations=violations,
        warnings=warnings,
        checks_passed=passed,
        drawdown_heat=heat_level,
        drawdown_pct=drawdown_pct,
        kelly_multiplier=kelly_mult,
        portfolio_gate=gate_reason if not can_add else "ok",
        safety_gate_stage="full_evaluation",
        estimated_fees_pct=estimated_fees,
        estimated_gas_usd=risk_config.gas_cost_usd,
    )

    return result


# ---------------------------------------------------------------------------
# Skill Integration Helper
# ---------------------------------------------------------------------------


def create_risk_check_skill() -> dict[str, Any]:
    """Create a skill-compatible risk check callable.

    Returns a dict that can be used with the SkillRouter framework.
    """
    def risk_check(
        edge: EdgeResult,
        features: MarketFeatures,
        portfolio_state: PortfolioState,
        risk_config: RiskConfig | None = None,
        confidence_level: str = "MEDIUM",
    ) -> RiskCheckResult:
        config = risk_config or RiskConfig()
        return evaluate_risk_gate(
            edge=edge,
            features=features,
            portfolio_state=portfolio_state,
            risk_config=config,
            confidence_level=confidence_level,
        )

    return {
        "fn": risk_check,
        "name": "risk-check",
        "description": "Evaluates 15+ risk checks before allowing a trade",
        "triggers": ["risk", "check", "gate", "limit"],
    }
