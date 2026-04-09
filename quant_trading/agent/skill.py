"""Skill-first agent architecture.

Key concepts:
- Skill-first (not prompt-first): capabilities are explicit skills with metadata
- Skill scoring with dynamic feedback and fallback routing
- Multi-channel delivery: Telegram, Slack, Discord, Feishu, DingTalk, Email, WhatsApp, QQ
- Skill registry with priority and fallback chains
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol


# ---------------------------------------------------------------------------
# Channel types for multi-channel delivery
# ---------------------------------------------------------------------------


class Channel(Enum):
    """Supported delivery channels."""

    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    FEISHU = "feishu"
    DINGTALK = "dingtalk"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    QQ = "qq"
    INTERNAL = "internal"  # Default internal routing


# ---------------------------------------------------------------------------
# Skill execution result
# ---------------------------------------------------------------------------


@dataclass
class SkillResult:
    """Result returned after a skill executes."""

    skill_name: str
    success: bool
    output: str = ""
    error: str = ""
    channels: list[Channel] = field(default_factory=lambda: [Channel.INTERNAL])
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """
    A skill is a named capability with an execute function and optional scoring.

    Unlike prompt-first agents where the LLM decides what to do, skill-first
    means the agent routes requests to the best-matching skill based on
    compatibility scoring.

    Attributes:
        name: Unique skill identifier (e.g. "market-brief", "risk-check")
        description: Human-readable description of what the skill does.
        execute: Async function(text, context) -> SkillResult
        score_fn: Optional function(text, context) -> float for custom scoring.
        priority: Static priority 0-100 (default 50). Higher = preferred.
        triggers: List of keywords/phrases that auto-trigger this skill.
        required_tools: Tools that must be available for this skill to run.
        alternative_tools: Alternative tools that satisfy requirements.
        markets: List of markets this skill applies to (e.g. "a-share", "us").
        asset_classes: Asset classes this skill handles (e.g. "equity", "crypto").
        fallback_skills: Skill names to fall back to if this skill fails.
        output_format: Expected output format hint.
        risk: Risk level description.
        determinism: "script-backed", "tool-backed", "reference-backed", "prompt-only".
        always: If True, always included regardless of trigger match.
    """

    name: str
    description: str
    execute: Callable[..., SkillResult]
    score_fn: Callable[[str, dict[str, Any]], float] | None = None
    priority: int = 50
    triggers: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    alternative_tools: list[str] = field(default_factory=list)
    markets: list[str] = field(default_factory=list)
    asset_classes: list[str] = field(default_factory=list)
    fallback_skills: list[str] = field(default_factory=list)
    output_format: str | None = None
    risk: str | None = None
    determinism: str = "prompt-only"
    always: bool = False

    # Internal scoring helpers
    _rule_weight: float = 1.0
    _dynamic_weight: float = 1.0

    def __post_init__(self) -> None:
        # Normalize determinism score for sorting
        self._determinism_score = {
            "script-backed": 3,
            "tool-backed": 2,
            "reference-backed": 1,
            "prompt-only": 0,
        }.get(self.determinism, 0)

    def match_trigger(self, text: str) -> bool:
        """Return True if any trigger keyword matches in the text."""
        lowered = text.lower()
        return any(trigger.lower() in lowered for trigger in self.triggers if trigger.strip())

    def compute_rule_score(self, text: str) -> float:
        """
        Compute the rule-based score for this skill against a request.

        Combines: priority + longest trigger match + determinism bonus.
        """
        score = float(self.priority)
        lowered = text.lower()
        matched_lengths = [len(trigger) for trigger in self.triggers if trigger.lower() in lowered]
        if matched_lengths:
            score += max(matched_lengths) / 100.0
        score += self._determinism_score / 10.0
        return round(score, 4)

    def compute_dynamic_score(self, record: dict[str, Any]) -> float:
        """
        Compute effective score from historical routing outcomes.

        Applies time decay and confidence weighting to past success/failure records.
        """
        score = float(record.get("score", 0.0) or 0.0)
        last_used = _parse_iso(record.get("lastUsedAt"))
        if last_used:
            age_days = max(0.0, (datetime.now(UTC) - last_used).total_seconds() / 86400.0)
            if age_days > 30:
                score *= 0.7
            elif age_days > 14:
                score *= 0.9

        total_events = sum(
            int(record.get(key, 0) or 0)
            for key in ("successCount", "partialCount", "failureCount", "misrouteCount")
        )
        confidence = min(1.0, total_events / 10.0)
        return round(score * confidence, 4)

    def compute_final_score(self, text: str, record: dict[str, Any]) -> float:
        """Combine rule score and dynamic score for routing decision."""
        rule = self.compute_rule_score(text)
        dynamic = self.compute_dynamic_score(record)
        return round(rule + dynamic, 4)


# ---------------------------------------------------------------------------
# Skill router
# ---------------------------------------------------------------------------


@dataclass
class RouteResult:
    """Result of routing a request to skills."""

    primary_skill: Skill | None
    fallback_skills: list[Skill]
    diagnostics: list[dict[str, Any]]


class SkillRouter:
    """
    Routes requests to the best-matching skill using scoring and fallback chains.

    Key features:
    - Skill-first routing: match by triggers, priority, and dynamic score
    - Market/asset filtering: compatible skills only
    - Fallback chain: if primary fails, try fallback_skills
    - Dynamic scoring: learn from past routing outcomes
    - Multi-channel delivery architecture
    """

    def __init__(
        self,
        workspace: Path | None = None,
        score_store_path: Path | None = None,
    ) -> None:
        self._workspace = workspace or Path(".")
        self._score_path = score_store_path or (self._workspace / "data" / "skill_scores.json")
        self._score_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, Skill] = {}
        self._buckets: dict[str, dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # Registry
    # -----------------------------------------------------------------------

    def register(self, skill: Skill) -> None:
        """Register a skill in the router."""
        self._registry[skill.name] = skill

    def unregister(self, name: str) -> None:
        """Remove a skill from the registry."""
        self._registry.pop(name, None)

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._registry.get(name)

    def list_skills(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._registry.values())

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------

    def route(self, text: str, context: dict[str, Any] | None = None) -> RouteResult:
        """
        Route a request to the best matching skill(s).

        Args:
            text: The user's request text.
            context: Optional context including:
                - available_tools: set of tool names
                - markets: list of market names
                - asset_classes: list of asset class names
                - runtime_profile: runtime capabilities

        Returns:
            RouteResult with primary skill and fallback chain.
        """
        context = context or {}
        available_tools = context.get("available_tools")
        request_markets = set(context.get("markets", []))
        request_assets = set(context.get("asset_classes", []))
        runtime_profile = context.get("runtime_profile")

        candidates = self._get_compatible_skills(
            text,
            available_tools=available_tools,
            request_markets=request_markets,
            request_assets=request_assets,
            runtime_profile=runtime_profile,
        )

        if not candidates:
            return RouteResult(primary_skill=None, fallback_skills=[], diagnostics=[])

        # Score and sort
        scored = []
        diagnostics: list[dict[str, Any]] = []
        for skill in candidates:
            bucket_key = self._make_bucket_key(skill, text, context)
            record = self._load_record(bucket_key)
            final_score = skill.compute_final_score(text, record)
            rule_score = skill.compute_rule_score(text)
            dynamic_score = skill.compute_dynamic_score(record)
            scored.append((final_score, rule_score, skill))
            diagnostics.append({
                "name": skill.name,
                "finalScore": final_score,
                "ruleScore": rule_score,
                "dynamicScore": dynamic_score,
                "bucketKey": bucket_key,
                "record": dict(record),
            })

        # Sort by final score desc, then rule score desc, then name
        scored.sort(key=lambda x: (-x[0], -x[1], x[2].name))

        primary = scored[0][2] if scored else None
        fallback_names = list(primary.fallback_skills) if primary else []
        fallback_skills = [self.get(n) for n in fallback_names if self.get(n)]
        fallback_skills = [s for s in fallback_skills if s is not None]

        return RouteResult(
            primary_skill=primary,
            fallback_skills=fallback_skills,
            diagnostics=diagnostics,
        )

    async def execute(self, text: str, context: dict[str, Any] | None = None) -> SkillResult:
        """
        Route and execute the best matching skill with fallback support.

        Args:
            text: The user's request text.
            context: Optional context for routing.

        Returns:
            SkillResult from the first successful skill.
        """
        result = self.route(text, context)
        primary = result.primary_skill
        fallback_skills = result.fallback_skills

        # Try primary
        if primary:
            try:
                result = await primary.execute(text, context.copy() if context else {})
                if result.success:
                    self._record_outcome(primary.name, text, "success", context)
                    return result
                self._record_outcome(primary.name, text, "failure", context)
            except Exception as exc:
                result = SkillResult(
                    skill_name=primary.name,
                    success=False,
                    error=str(exc),
                )
                self._record_outcome(primary.name, text, "failure", context)

        # Try fallbacks
        for skill in fallback_skills:
            try:
                result = await skill.execute(text, context.copy() if context else {})
                if result.success:
                    self._record_outcome(skill.name, text, "success", context)
                    return result
                self._record_outcome(skill.name, text, "failure", context)
            except Exception as exc:
                result = SkillResult(
                    skill_name=skill.name,
                    success=False,
                    error=str(exc),
                )
                self._record_outcome(skill.name, text, "failure", context)

        return SkillResult(
            skill_name=primary.name if primary else "none",
            success=False,
            error="No skill succeeded",
        )

    # -----------------------------------------------------------------------
    # Skill compatibility
    # -----------------------------------------------------------------------

    def _get_compatible_skills(
        self,
        text: str,
        available_tools: set[str] | None = None,
        request_markets: set[str] | None = None,
        request_assets: set[str] | None = None,
        runtime_profile: dict[str, Any] | None = None,
    ) -> list[Skill]:
        """Return skills that are compatible with the request."""
        lowered = text.lower()
        candidates: list[Skill] = []

        for skill in self._registry.values():
            # Check always flag
            if skill.always:
                if self._skill_tools_available(skill, available_tools):
                    candidates.append(skill)
                continue

            # Check trigger match
            if not skill.match_trigger(lowered):
                continue

            # Check tool requirements
            if not self._skill_tools_available(skill, available_tools):
                continue

            # Check market compatibility
            if not self._skill_markets_compatible(skill, request_markets):
                continue

            # Check asset class compatibility
            if not self._skill_assets_compatible(skill, request_assets):
                continue

            candidates.append(skill)

        return candidates

    def _skill_tools_available(self, skill: Skill, available: set[str] | None) -> bool:
        """Return True if the skill's required tools are available."""
        required = {t for t in skill.required_tools if t.strip()}
        alternatives = {t for t in skill.alternative_tools if t.strip()}
        if not required and not alternatives:
            return True
        if available is None:
            return True
        if required and required.issubset(available):
            return True
        if alternatives and alternatives.issubset(available):
            return True
        return False

    def _skill_markets_compatible(self, skill: Skill, request: set[str]) -> bool:
        """Return True if skill markets are compatible with request markets."""
        skill_markets = {m.lower() for m in skill.markets}
        if not skill_markets:
            return True
        if not request:
            return True
        if "global" in skill_markets:
            return True
        return bool(skill_markets & request)

    def _skill_assets_compatible(self, skill: Skill, request: set[str]) -> bool:
        """Return True if skill asset classes are compatible with request."""
        skill_assets = {a.lower() for a in skill.asset_classes}
        if not skill_assets:
            return True
        if not request:
            return True
        return bool(skill_assets & request)

    # -----------------------------------------------------------------------
    # Dynamic scoring
    # -----------------------------------------------------------------------

    def _make_bucket_key(self, skill: Skill, text: str, context: dict[str, Any] | None) -> str:
        """Create a stable bucket key for dynamic score storage."""
        markets = context.get("markets", []) if context else []
        primary_market = "general"
        for candidate in ("a-share", "hong-kong", "us", "global", "mixed"):
            if candidate in markets:
                primary_market = candidate
                break
        task_type = "general"
        return f"{skill.name}|{primary_market}|{task_type}|default"

    def _load_record(self, bucket_key: str) -> dict[str, Any]:
        """Load score record for a bucket key."""
        self._ensure_loaded()
        return dict(self._buckets.get(bucket_key, {}))

    def _ensure_loaded(self) -> None:
        """Load scores from disk if not yet loaded."""
        if self._buckets:
            return
        if not self._score_path.exists():
            self._buckets = {}
            return
        try:
            payload = json.loads(self._score_path.read_text(encoding="utf-8"))
            self._buckets = payload.get("buckets", {}) if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, OSError):
            self._buckets = {}

    def _save_buckets(self) -> None:
        """Persist buckets to disk."""
        payload = {"version": 1, "buckets": self._buckets}
        self._score_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _DELTA_BY_OUTCOME = {
        "success": 0.20,
        "partial": 0.05,
        "failure": -0.30,
        "misroute": -0.40,
    }

    def _record_outcome(self, skill_name: str, text: str, outcome: str, context: dict[str, Any] | None) -> None:
        """Record a routing outcome for dynamic scoring."""
        skill = self.get(skill_name)
        if not skill:
            return
        bucket_key = self._make_bucket_key(skill, text, context)
        self._ensure_loaded()
        record = dict(self._buckets.get(bucket_key, {}))
        delta = self._DELTA_BY_OUTCOME.get(outcome, -0.30)
        score = _clamp(float(record.get("score", 0.0) or 0.0) + delta, -3.0, 3.0)
        record["score"] = round(score, 4)
        record["lastUsedAt"] = _utc_now_iso()
        if outcome == "success":
            record["successCount"] = int(record.get("successCount", 0) or 0) + 1
        elif outcome == "partial":
            record["partialCount"] = int(record.get("partialCount", 0) or 0) + 1
        elif outcome == "misroute":
            record["misrouteCount"] = int(record.get("misrouteCount", 0) or 0) + 1
        else:
            record["failureCount"] = int(record.get("failureCount", 0) or 0) + 1
        self._buckets[bucket_key] = record
        self._save_buckets()

    # -----------------------------------------------------------------------
    # Market request classification
    # -----------------------------------------------------------------------

    @staticmethod
    def classify_market_request(text: str) -> dict[str, Any]:
        """Classify a request into coarse market types."""
        lowered = text.lower()
        symbols = _extract_symbols(text)

        equity = any(
            token in lowered
            for token in (
                "stock", "stocks", "equity", "equities", "share", "shares",
                "earnings", "guidance", "股票", "个股", "a股", "港股", "美股",
                "财报", "业绩", "指数", "板块",
            )
        )
        crypto = any(
            token in lowered
            for token in (
                "crypto", "bitcoin", "ethereum", "solana", "altcoin",
                "token", "onchain", "加密", "比特币", "以太坊",
            )
        ) or bool(re.search(r"\b(BTC|ETH|SOL|XRP|DOGE|ADA)(?:-[A-Z]{2,4})?\b", text))
        metals = any(
            token in lowered
            for token in ("gold", "silver", "xau", "xag", "precious metals", "bullion", "黄金", "白银", "贵金属")
        )
        macro = any(
            token in lowered
            for token in (
                "macro", "fomc", "fed", "cpi", "nfp", "pmi", "gdp", "yield",
                "treasury", "dxy", "宏观", "美联储", "非农", "通胀",
            )
        )

        if equity:
            primary = "equity"
        elif crypto:
            primary = "crypto"
        elif metals:
            primary = "metals"
        elif macro:
            primary = "macro"
        else:
            primary = "general"

        return {
            "primary": primary,
            "equity": equity,
            "crypto": crypto,
            "metals": metals,
            "macro": macro,
            "symbols": symbols,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_iso(text: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into a datetime."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp value into the given range."""
    return max(lower, min(upper, value))


def _extract_symbols(text: str) -> list[str]:
    """Extract ticker symbols from text."""
    pattern = r"\b([A-Z]{1,5}(?:\.[A-Z]{1,3})?|[A-Z]{2,6}(?:-[A-Z]{2,6})?|(?:SH|SZ)?\d{6}|\d{4,5}\.HK)\b"
    return re.findall(pattern, text.upper())
