"""Multi-Agent Bull vs Bear Debate Engine.

多智能体辩论引擎 — 红蓝对抗多轮辩论框架.

Architecture:
- BullAgent: 多头辩论智能体, arguer for bullish stance
- BearAgent: 空头辩论智能体, arguer for bearish stance
- DebateEngine: 辩论引擎, orchestrates multi-round debate

流程:
1. 初始化 — 创建Bull/Bear Agent各N个
2. 第1轮陈述 — 各方提出Claims
3. 第2-N轮反驳 — 交叉辩论
4. 裁判打分 — 输出最终判决
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

__all__ = [
    # New API
    "Claim",
    "DebateResult",
    "DebateAgent",
    "BullAgent",
    "BearAgent",
    "DebateEngine",
    "Stance",
    # Legacy API (backward compatibility)
    "ClaimStatus",
    "build_empty_invest_debate_state",
    "build_empty_risk_debate_state",
    "build_empty_risk_feedback_state",
    "extract_risk_judge_result",
    "extract_tagged_json",
    "format_claims_for_prompt",
    "format_claim_subset_for_prompt",
    "strip_tagged_json",
    "summarize_game_theory_signals",
    "summarize_risk_feedback",
    "update_debate_state_with_payload",
    "safe_int",
]


class Stance(str, Enum):
    """辩论立场."""

    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# REST client fallback (no heavy SDK dependency)
# ---------------------------------------------------------------------------


def _default_llm_call(
    prompt: str,
    *,
    model: str = "gpt-4",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 30,
) -> str:
    """Simple REST LLM call fallback with basic retry.

    轻量级REST调用，不依赖重型SDK.
    """
    import os
    import urllib.request

    api_base = api_base or os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    endpoint = f"{api_base.rstrip('/')}/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Claim & DebateResult dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """辩论论点 / Debate claim.

    Attributes:
        agent_id: 提出该论点的智能体ID
        content: 论点内容文本
        evidence: 支撑证据列表
        stance: 立场 — 'bull' | 'bear' | 'neutral'
        confidence: 置信度 [0, 1]
        timestamp: 时间戳
    """

    agent_id: str
    content: str
    evidence: list[str]
    stance: str  # 'bull' | 'bear' | 'neutral'
    confidence: float  # [0, 1]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "evidence": self.evidence,
            "stance": self.stance,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DebateResult:
    """辩论结果 / Debate result.

    Attributes:
        topic: 辩论主题
        bull_claims: 多头方所有论点列表
        bear_claims: 空头方所有论点列表
        winner: 胜方 — 'bull' | 'bear' | 'tie'
        consensus: 共识摘要
        final_verdict: 最终判决文本
        confidence: 整体置信度 [0, 1]
    """

    topic: str
    bull_claims: list[Claim]
    bear_claims: list[Claim]
    winner: str  # 'bull' | 'bear' | 'tie'
    consensus: str
    final_verdict: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "bull_claims": [c.to_dict() for c in self.bull_claims],
            "bear_claims": [c.to_dict() for c in self.bear_claims],
            "winner": self.winner,
            "consensus": self.consensus,
            "final_verdict": self.final_verdict,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Base DebateAgent
# ---------------------------------------------------------------------------


class DebateAgent:
    """辩论智能体基类 / Base debate agent.

    所有辩论智能体的基类，提供立场一致的论点分析和反驳能力.

    Args:
        agent_id: 智能体唯一标识
        role: 角色描述 (e.g. "fundamental_analyst")
        stance: 初始立场 (bull/bear/neutral)
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        stance: str,
    ):
        self.agent_id = agent_id
        self.role = role
        self.stance = stance
        self._llm_call: Optional[Callable[..., str]] = None

    def set_llm(self, llm: Callable[..., str]) -> None:
        """设置LLM调用函数 / Set LLM callable."""
        self._llm_call = llm

    def _call_llm(self, prompt: str) -> str:
        if self._llm_call:
            return self._llm_call(prompt)
        return _default_llm_call(prompt)

    def analyze(self, topic: str, data: dict) -> Claim:
        """分析主题并生成论点 / Analyze topic and generate claim.

        Args:
            topic: 辩论主题
            data: 上下文数据 (行情、新闻、指标等)

        Returns:
            Claim: 生成的论点
        """
        context = json.dumps(data, ensure_ascii=False, indent=2)
        prompt = self._build_analysis_prompt(topic, context)
        response = self._call_llm(prompt)
        return self._parse_claim_response(response, topic)

    def rebuttal(self, opposing_claims: list[Claim]) -> Claim:
        """针对对方论点生成反驳 / Rebut opposing claims.

        Args:
            opposing_claims: 对方论点列表

        Returns:
            Claim: 反驳论点
        """
        if not opposing_claims:
            return Claim(
                agent_id=self.agent_id,
                content="无明确反驳目标.",
                evidence=[],
                stance=self.stance,
                confidence=0.5,
            )
        opposing_text = "\n".join(
            f"- [{c.agent_id}] {c.content} (confidence={c.confidence})"
            for c in opposing_claims
        )
        prompt = self._build_rebuttal_prompt(opposing_text)
        response = self._call_llm(prompt)
        return self._parse_claim_response(response, "")

    def _build_analysis_prompt(self, topic: str, context: str) -> str:
        return f"""You are a {self.role} with a {self.stance} stance.

辩论主题: {topic}

背景数据:
{context}

请以结构化方式提出你的核心论点，格式如下:
---
content: <你的核心论点，简洁有力>
evidence: [<证据1>, <证据2>, ...]
confidence: <0到1之间的置信度>
---

只返回格式化的论点，不要额外解释。"""

    def _build_rebuttal_prompt(self, opposing_text: str) -> str:
        return f"""You are a {self.role} with a {self.stance} stance.

对方的论点:
{opposing_text}

请针对对方论点提出有力的反驳:
---
content: <你的反驳论点>
evidence: [<反驳证据1>, <反驳证据2>, ...]
confidence: <0到1之间的置信度>
---

只返回格式化的论点，不要额外解释。"""

    def _parse_claim_response(self, response: str, topic: str) -> Claim:
        content = ""
        evidence: list[str] = []
        confidence = 0.6

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("content:"):
                content = line[len("content:") :].strip().strip('"')
            elif line.startswith("evidence:"):
                ev_str = line[len("evidence:") :].strip().strip("[]")
                if ev_str:
                    evidence = [e.strip().strip('"') for e in ev_str.split(",")]
            elif line.startswith("confidence:"):
                try:
                    confidence = float(line[len("confidence:") :].strip())
                except ValueError:
                    confidence = 0.6

        if not content:
            content = response.strip()[:200]

        return Claim(
            agent_id=self.agent_id,
            content=content,
            evidence=evidence,
            stance=self.stance,
            confidence=max(0.0, min(1.0, confidence)),
        )


# ---------------------------------------------------------------------------
# BullAgent
# ---------------------------------------------------------------------------


class BullAgent(DebateAgent):
    """多头辩论Agent / Bullish debate agent.

    专门论证价格上涨合理性的辩论智能体.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id, role="bull_advocate", stance=Stance.BULL)


# ---------------------------------------------------------------------------
# BearAgent
# ---------------------------------------------------------------------------


class BearAgent(DebateAgent):
    """空头辩论Agent / Bearish debate agent.

    专门论证价格下跌合理性的辩论智能体.
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id, role="bear_advocate", stance=Stance.BEAR)


# ---------------------------------------------------------------------------
# DebateEngine
# ---------------------------------------------------------------------------


class DebateEngine:
    """辩论引擎 — 红蓝对抗多轮辩论 / Red-vs-Blue multi-round debate engine.

     orchestrates multi-agent bull vs bear debate.

    流程:
        1. 初始化 — 创建Bull/Bear Agent各N个
        2. 第1轮陈述 — 各方提出Claims
        3. 第2-N轮反驳 — 交叉辩论
        4. 裁判打分 — 输出最终判决

    Args:
        n_bull: 多头Agent数量 (default 3)
        n_bear: 空头Agent数量 (default 3)
        n_rounds: 辩论轮数 (default 3)
        judge_llm: 裁判LLM调用函数 (optional)
    """

    def __init__(
        self,
        n_bull: int = 3,
        n_bear: int = 3,
        n_rounds: int = 3,
        judge_llm: Optional[Callable[..., str]] = None,
    ):
        self.n_bull = n_bull
        self.n_bear = n_bear
        self.n_rounds = n_rounds
        self.judge_llm = judge_llm

        self.bull_agents: list[BullAgent] = [
            BullAgent(agent_id=f"bull-{i + 1}") for i in range(n_bull)
        ]
        self.bear_agents: list[BearAgent] = [
            BearAgent(agent_id=f"bear-{i + 1}") for i in range(n_bear)
        ]

        for agent in self.bull_agents + self.bear_agents:
            if judge_llm:
                agent.set_llm(judge_llm)

    def add_judge(self, judge_llm: Callable[..., str]) -> None:
        """设置裁判LLM / Set judge LLM.

        Args:
            judge_llm: LLM调用函数
        """
        self.judge_llm = judge_llm

    def run_debate(self, topic: str, data: dict) -> DebateResult:
        """运行完整辩论流程 / Run full debate process.

        Args:
            topic: 辩论主题
            data: 上下文数据

        Returns:
            DebateResult: 辩论结果
        """
        bull_claims: list[Claim] = []
        bear_claims: list[Claim] = []

        # 第1轮: 各方陈述
        for agent in self.bull_agents:
            claim = agent.analyze(topic, data)
            bull_claims.append(claim)

        for agent in self.bear_agents:
            claim = agent.analyze(topic, data)
            bear_claims.append(claim)

        # 第2-N轮: 交叉反驳
        for round_idx in range(1, self.n_rounds):
            # Bull agents rebut bear claims
            new_bull_claims: list[Claim] = []
            for agent in self.bull_agents:
                claim = agent.rebuttal(bear_claims)
                new_bull_claims.append(claim)
            bull_claims.extend(new_bull_claims)

            # Bear agents rebut bull claims
            new_bear_claims: list[Claim] = []
            for agent in self.bear_agents:
                claim = agent.rebuttal(bull_claims)
                new_bear_claims.append(claim)
            bear_claims.extend(new_bear_claims)

        # 裁判打分
        winner, consensus, final_verdict, confidence = self._judge(topic, bull_claims, bear_claims)

        return DebateResult(
            topic=topic,
            bull_claims=bull_claims,
            bear_claims=bear_claims,
            winner=winner,
            consensus=consensus,
            final_verdict=final_verdict,
            confidence=confidence,
        )

    def _judge(
        self, topic: str, bull_claims: list[Claim], bear_claims: list[Claim]
    ) -> tuple[str, str, str, float]:
        """裁判评分 / Judge scoring.

        Returns:
            tuple: (winner, consensus, final_verdict, confidence)
        """
        if self.judge_llm:
            return self._llm_judge(topic, bull_claims, bear_claims)

        # Fallback: simple scoring
        bull_avg = sum(c.confidence for c in bull_claims) / max(len(bull_claims), 1)
        bear_avg = sum(c.confidence for c in bear_claims) / max(len(bear_claims), 1)

        diff = abs(bull_avg - bear_avg)
        if diff < 0.05:
            winner = "tie"
            confidence = 0.5
        elif bull_avg > bear_avg:
            winner = "bull"
            confidence = min(1.0, bull_avg)
        else:
            winner = "bear"
            confidence = min(1.0, bear_avg)

        consensus = (
            f"Bull avg confidence: {bull_avg:.2f}, Bear avg confidence: {bear_avg:.2f}"
        )
        final_verdict = f"辩论结果: {winner.upper()} 方胜出 (confidence={confidence:.2f})"

        return winner, consensus, final_verdict, confidence

    def _llm_judge(
        self, topic: str, bull_claims: list[Claim], bear_claims: list[Claim]
    ) -> tuple[str, str, str, float]:
        """使用LLM进行裁判评分 / Use LLM for judge scoring."""
        bull_text = "\n".join(
            f"- [{c.agent_id}] {c.content} (conf={c.confidence})"
            for c in bull_claims
        )
        bear_text = "\n".join(
            f"- [{c.agent_id}] {c.content} (conf={c.confidence})"
            for c in bear_claims
        )

        prompt = f"""辩论主题: {topic}

多头论点:
{bull_text}

空头论点:
{bear_text}

请进行裁判评分，返回JSON格式:
{{
    "winner": "bull" | "bear" | "tie",
    "consensus": "共识摘要",
    "final_verdict": "最终判决文本",
    "confidence": 0到1之间的置信度
}}
"""
        response = self.judge_llm(prompt)
        return self._parse_judge_response(response)

    def _parse_judge_response(self, response: str) -> tuple[str, str, str, float]:
        winner = "tie"
        consensus = ""
        final_verdict = ""
        confidence = 0.5

        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                winner = data.get("winner", "tie")
                consensus = data.get("consensus", "")
                final_verdict = data.get("final_verdict", "")
                confidence = float(data.get("confidence", 0.5))
        except (json.JSONDecodeError, ValueError):
            pass

        return winner, consensus, final_verdict, confidence


# ---------------------------------------------------------------------------
# Backward compatibility — legacy functions for existing modules
# ---------------------------------------------------------------------------


class ClaimStatus:
    """Legacy claim status constants."""
    OPEN = "open"
    ADDRESSED = "addressed"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


def extract_tagged_json(text: str, tag: str) -> dict[str, Any]:
    """Extract structured JSON from a tagged comment block in text."""
    import re
    pattern = rf"<!--\s*{re.escape(tag)}:\s*(\{{.*?\}})\s*-->"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def strip_tagged_json(text: str, tag: str) -> str:
    """Remove a tagged JSON comment block from text."""
    import re
    pattern = rf"\n?<!--\s*{re.escape(tag)}:\s*\{{.*?\}}\s*-->\s*"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def format_claims_for_prompt(
    claims: Any,
    focus_claim_ids: Any = None,
    empty_message: str = "当前没有已登记 claim，本轮请先提出 1 到 2 条最关键 claim。",
) -> str:
    """Format claims for injection into a debate prompt."""
    import re
    claim_list = list(claims) if claims else []
    if not claim_list:
        return empty_message

    focus_set = {str(item) for item in (focus_claim_ids or []) if str(item).strip()}
    lines: list[str] = []
    for claim in claim_list:
        claim_id = str(claim.get("claim_id", "")).strip()
        status = str(claim.get("status", "open")).strip() or "open"
        speaker = str(claim.get("speaker", "")).strip() or "Unknown"
        summary = str(claim.get("claim", "")).strip() or "未提供 claim 文本"
        evidence = claim.get("evidence") or []
        evidence_text = "；".join(str(item).strip() for item in evidence if str(item).strip()) or "无明确证据"
        prefix = "* " if claim_id in focus_set else "- "
        lines.append(
            f"{prefix}{claim_id} [{status}] {speaker}: {summary} | 证据: {evidence_text}"
        )
    return "\n".join(lines)


def format_claim_subset_for_prompt(
    claims: Any,
    claim_ids: Any,
    empty_message: str = "当前没有未解决 claim。",
) -> str:
    """Format only a specific subset of claims for a prompt."""
    claim_id_set = {str(item) for item in (claim_ids or []) if str(item).strip()}
    if not claim_id_set:
        return empty_message
    subset = [
        claim for claim in (claims or [])
        if str(claim.get("claim_id", "")) in claim_id_set
    ]
    return format_claims_for_prompt(subset, focus_claim_ids=claim_id_set, empty_message=empty_message)


def extract_risk_judge_result(text: str) -> dict[str, Any]:
    """Extract and parse a RISK_JUDGE tagged block from risk manager output."""
    judge_payload = extract_tagged_json(text, "RISK_JUDGE")
    cleaned_response = strip_tagged_json(text, "RISK_JUDGE")
    parse_failed = not bool(judge_payload)

    verdict = str(judge_payload.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "revise", "reject"}:
        parse_failed = True
        verdict = "reject"

    hard_constraints = [
        str(item).strip() for item in (judge_payload.get("hard_constraints") or []) if str(item).strip()
    ]
    soft_constraints = [
        str(item).strip() for item in (judge_payload.get("soft_constraints") or []) if str(item).strip()
    ]
    execution_preconditions = [
        str(item).strip() for item in (judge_payload.get("execution_preconditions") or []) if str(item).strip()
    ]
    de_risk_triggers = [
        str(item).strip() for item in (judge_payload.get("de_risk_triggers") or []) if str(item).strip()
    ]
    revision_reason = str(judge_payload.get("revision_reason", "")).strip()

    if parse_failed:
        revision_reason = revision_reason or "风控裁决机读块解析失败，按拒绝处理"
        if cleaned_response:
            cleaned_response = f"{cleaned_response}\n\n[系统说明] 风控裁决机读块解析失败，已按拒绝处理。"
        else:
            cleaned_response = "风控裁决机读块解析失败，已按拒绝处理。"

    return {
        "judge_payload": judge_payload,
        "cleaned_response": cleaned_response,
        "verdict": verdict,
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
        "execution_preconditions": execution_preconditions,
        "de_risk_triggers": de_risk_triggers,
        "revision_reason": revision_reason,
        "parse_failed": parse_failed,
    }


def summarize_game_theory_signals(signals: Any) -> str:
    """Format structured game theory signals as a readable summary."""
    payload = signals or {}
    if not payload:
        return "暂无结构化博弈信号。"

    players = payload.get("players") or []
    likely_actions = payload.get("likely_actions") or {}
    if isinstance(likely_actions, dict):
        action_lines = []
        for key, value in likely_actions.items():
            if isinstance(value, list):
                value_text = " / ".join(str(item) for item in value if str(item).strip())
            else:
                value_text = str(value)
            action_lines.append(f"{key}: {value_text}")
        actions_text = "; ".join(action_lines) if action_lines else "未提供"
    else:
        actions_text = str(likely_actions)

    return "\n".join(
        [
            f"局面: {payload.get('board', '未提供')}",
            f"参与者: {', '.join(str(item) for item in players) if players else '未提供'}",
            f"主导策略: {payload.get('dominant_strategy', '未提供')}",
            f"脆弱均衡: {payload.get('fragile_equilibrium', '未提供')}",
            f"潜在动作: {actions_text}",
            f"反共识信号: {payload.get('counter_consensus_signal', '未提供')}",
            f"置信度: {payload.get('confidence', '未提供')}",
        ]
    )


def summarize_risk_feedback(feedback: Any) -> str:
    """Format risk feedback state as a readable summary."""
    payload = feedback or {}
    verdict = str(payload.get("latest_risk_verdict", "")).strip()
    if not verdict:
        return "当前没有待处理的风控回退要求。"

    hard_constraints = payload.get("hard_constraints") or []
    soft_constraints = payload.get("soft_constraints") or []
    preconditions = payload.get("execution_preconditions") or []
    de_risk_triggers = payload.get("de_risk_triggers") or []

    return "\n".join(
        [
            f"风控裁决: {verdict}",
            f"是否要求重做: {'是' if payload.get('revision_required') else '否'}",
            f"打回原因: {payload.get('revision_reason', '未提供')}",
            f"硬约束: {'; '.join(str(item) for item in hard_constraints) if hard_constraints else '无'}",
            f"软约束: {'; '.join(str(item) for item in soft_constraints) if soft_constraints else '无'}",
            f"执行前提: {'; '.join(str(item) for item in preconditions) if preconditions else '无'}",
            f"降风险触发器: {'; '.join(str(item) for item in de_risk_triggers) if de_risk_triggers else '无'}",
        ]
    )


def _default_round_goal(domain: str, next_count: int) -> str:
    """Get the default round goal text for a debate domain."""
    goals = {
        "investment": [
            "建立最核心的正反两方 claim，并明确为何是现在。",
            "优先攻击对手最脆弱的假设，不要扩散议题。",
            "围绕时间窗口与触发条件，判断交易时机是否成立。",
            "围绕失败路径与失效条件，判断谁低估了回撤风险。",
            "检查剩余分歧是否仍有信息增量，否则准备收口。",
        ],
        "risk": [
            "建立最关键的执行风险 claim，明确风险预算冲突点。",
            "围绕仓位、止损、流动性约束，攻击对手最薄弱一环。",
            "判断哪些风险是可接受波动，哪些风险是硬性红线。",
            "逼迫双方给出可执行替代方案，而不是抽象立场。",
            "检查是否还存在未解决的高影响执行风险，否则准备收口。",
        ],
    }
    domain_key = domain if domain in goals else "investment"
    goal_list = goals[domain_key]
    index = min(max(next_count - 1, 0), len(goal_list) - 1)
    return goal_list[index]


def build_empty_invest_debate_state() -> dict[str, Any]:
    """Build an empty investment debate state."""
    return {
        "history": "",
        "bull_history": "",
        "bear_history": "",
        "bull_initial": "",
        "bear_initial": "",
        "bull_rebuttal": "",
        "bear_rebuttal": "",
        "current_speaker": "",
        "current_response": "",
        "judge_decision": "",
        "count": 0,
        "claims": [],
        "focus_claim_ids": [],
        "open_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
        "round_goal": _default_round_goal("investment", 1),
        "claim_counter": 0,
    }


def build_empty_risk_debate_state() -> dict[str, Any]:
    """Build an empty risk debate state."""
    return {
        "history": "",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "latest_speaker": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "judge_decision": "",
        "count": 0,
        "claims": [],
        "focus_claim_ids": [],
        "open_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
        "round_goal": _default_round_goal("risk", 1),
        "claim_counter": 0,
    }


def build_empty_risk_feedback_state() -> dict[str, Any]:
    """Build an empty risk feedback state."""
    return {
        "retry_count": 0,
        "max_retries": 1,
        "revision_required": False,
        "latest_risk_verdict": "",
        "hard_constraints": [],
        "soft_constraints": [],
        "execution_preconditions": [],
        "de_risk_triggers": [],
        "revision_reason": "",
    }


def _filter_known_claim_ids(values: Any, claim_map: Any) -> list[str]:
    result = []
    for item in _string_list(values):
        if item in claim_map:
            result.append(item)
    return result


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return fallback


def safe_int(value: Any, fallback: int = 0) -> int:
    """Safely convert value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _append_history(history: Any, argument: str) -> str:
    existing = str(history or "").strip()
    if not existing:
        return argument
    return f"{existing}\n{argument}"


def _fallback_summary(text: str) -> str:
    import re
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return "本轮未提取到有效摘要。"
    return compact[:120]


def update_debate_state_with_payload(
    *,
    state: Any,
    raw_response: str,
    speaker_label: str,
    speaker_key: str,
    stance: str,
    history_key: str,
    marker: str,
    claim_prefix: str,
    domain: str,
    speaker_field: str,
    store_current_response: bool = True,
) -> dict[str, Any]:
    """Update an investment or risk debate state by extracting claim payloads."""
    payload = extract_tagged_json(raw_response, marker)
    cleaned_response = strip_tagged_json(raw_response, marker)

    claims = [dict(item) for item in state.get("claims", [])]
    claim_map = {
        str(item.get("claim_id", "")).strip(): item
        for item in claims
        if str(item.get("claim_id", "")).strip()
    }

    claim_counter = int(state.get("claim_counter", 0) or 0)
    responded_claim_ids = _filter_known_claim_ids(payload.get("responded_claim_ids"), claim_map)
    resolved_claim_ids = _filter_known_claim_ids(payload.get("resolved_claim_ids"), claim_map)
    unresolved_claim_ids = _filter_known_claim_ids(payload.get("unresolved_claim_ids"), claim_map)

    open_claim_ids = set(_string_list(state.get("open_claim_ids")))
    resolved_set = set(_string_list(state.get("resolved_claim_ids")))
    unresolved_set = set(_string_list(state.get("unresolved_claim_ids")))

    for claim_id in responded_claim_ids:
        if claim_id in claim_map and claim_map[claim_id].get("status") == "open":
            claim_map[claim_id]["status"] = "addressed"

    for claim_id in resolved_claim_ids:
        if claim_id in claim_map:
            claim_map[claim_id]["status"] = "resolved"
        open_claim_ids.discard(claim_id)
        unresolved_set.discard(claim_id)
        resolved_set.add(claim_id)

    for claim_id in unresolved_claim_ids:
        if claim_id in claim_map:
            claim_map[claim_id]["status"] = "unresolved"
        open_claim_ids.add(claim_id)
        unresolved_set.add(claim_id)
        resolved_set.discard(claim_id)

    for claim_payload in payload.get("new_claims", []) or []:
        claim_text = str(claim_payload.get("claim", "")).strip()
        if not claim_text:
            continue
        claim_counter += 1
        claim_id = f"{claim_prefix}-{claim_counter}"
        evidence = [
            str(item).strip()
            for item in (claim_payload.get("evidence") or [])[:3]
            if str(item).strip()
        ]
        confidence = _safe_float(claim_payload.get("confidence"), 0.6)
        target_claim_ids = _filter_known_claim_ids(claim_payload.get("target_claim_ids"), claim_map)
        claim_entry = {
            "claim_id": claim_id,
            "speaker": speaker_label,
            "speaker_key": speaker_key,
            "stance": stance,
            "claim": claim_text,
            "evidence": evidence,
            "confidence": confidence,
            "status": "open",
            "target_claim_ids": target_claim_ids,
            "round_index": int(state.get("count", 0) or 0) + 1,
        }
        claims.append(claim_entry)
        claim_map[claim_id] = claim_entry
        open_claim_ids.add(claim_id)

    next_focus_claim_ids = _filter_known_claim_ids(payload.get("next_focus_claim_ids"), claim_map)
    if not next_focus_claim_ids:
        preferred_ids = list(unresolved_set) + [cid for cid in open_claim_ids if cid not in unresolved_set]
        next_focus_claim_ids = preferred_ids[:2]

    summary = str(payload.get("round_summary", "")).strip() or _fallback_summary(cleaned_response)
    round_goal = str(payload.get("round_goal", "")).strip() or _default_round_goal(
        domain, int(state.get("count", 0) or 0) + 1
    )

    argument = f"{speaker_label}: {cleaned_response}"
    new_state = dict(state)
    updates = {
        "history": _append_history(state.get("history", ""), argument),
        history_key: _append_history(state.get(history_key, ""), argument),
        "current_speaker": speaker_key,
        speaker_field: speaker_key,
        "count": int(state.get("count", 0) or 0) + 1,
        "claims": claims,
        "claim_counter": claim_counter,
        "open_claim_ids": sorted(open_claim_ids),
        "resolved_claim_ids": sorted(resolved_set),
        "unresolved_claim_ids": sorted(unresolved_set),
        "focus_claim_ids": next_focus_claim_ids,
        "round_summary": summary,
        "round_goal": round_goal,
    }
    if store_current_response:
        updates["current_response"] = argument
    new_state.update(updates)
    return new_state
