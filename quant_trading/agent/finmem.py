"""FinMem: 分层记忆架构 — FinMem-LLM-StockTrading (ICLR/AAAI) adaptation.

三层记忆系统:
1. ProfilingMemory  — 用户/市场概况记忆, 持久化的核心特征
2. WorkingMemory    — 工作记忆, 当前决策上下文窗口 (可调认知跨度)
3. DecisionMemory  — 决策记忆, 交易决策及其结果的历史

源自 FinMem-LLM-StockTrading:
  - Yu et al., "FinMem: A LLM-Powered Agent for Stock Trading" (ICLR/AAAI)
  - 分层记忆 + 可调认知跨度 + LLM驱动的交易决策
  - 核心机制: 感知(perceive) → 反思(reflect) → 决策(decide)

Usage:
    from quant_trading.agent.finmem import FinMemCoordinator, ProfilingMemory

    coordinator = FinMemCoordinator(llm_client=some_llm, horizon=10)
    coordinator.perceive("BTC突破50000阻力位", category="market_data")
    decision = coordinator.decide({"price": 50000, "volume": 1000})
    coordinator.reflect()
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol

import numpy as np

__all__ = [
    "MemoryItem",
    "ProfilingMemory",
    "WorkingMemory",
    "DecisionMemory",
    "FinMemCoordinator",
]


# --------------------------------------------------------------------------- #
# Protocols / type hints
# --------------------------------------------------------------------------- #

class LLMClient(Protocol):
    """LLM client protocol — compatible with OpenAI-style chat completions."""

    def generate(self, prompt: str, **kwargs) -> str:
        """Return LLM-generated text for the given prompt."""
        ...


# --------------------------------------------------------------------------- #
# MemoryItem
# --------------------------------------------------------------------------- #

@dataclass
class MemoryItem:
    """记忆单元 / Memory unit.

    Attributes:
        timestamp:  时间戳
        content:    记忆内容文本
        importance: 重要性评分 [0, 1]
        category:   记忆类别
                     - 'market_data': K线/价格/成交量等市场数据
                     - 'news':        新闻/公告/宏观事件
                     - 'trade':       交易信号/持仓变化
                     - 'reflection':  反思结果/策略调整
        embedding:   文本向量 (numpy array or None).
                     如果为 None, 将使用 hash-based pseudo-embedding.
    """
    timestamp: datetime
    content: str
    importance: float
    category: str
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(f"importance must be in [0, 1], got {self.importance}")
        if self.category not in ("market_data", "news", "trade", "reflection"):
            raise ValueError(f"Unknown category: {self.category!r}")


# --------------------------------------------------------------------------- #
# Embedding utilities (hash-based pseudo-embeddings)
# --------------------------------------------------------------------------- #

def _hash_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Generate a deterministic pseudo-embedding via SHA-256 hash.

    Args:
        text: Input text string.
        dim:  Embedding dimension (default 64).

    Returns:
        Normalised numpy array of shape (dim,).
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat or trim to reach `dim` floats
    values: list[float] = []
    for i in range(dim):
        byte_idx = i % len(h)
        values.append(float(h[byte_idx]) / 255.0)
    vec = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# --------------------------------------------------------------------------- #
# Layer 1 — ProfilingMemory
# --------------------------------------------------------------------------- #

@dataclass
class ProfilingMemory:
    """第一层记忆: 用户/市场概况记忆 — 持久化的核心特征.

    存储并维护投资者画像 (risk_tolerance, investment_horizon, preferred_sectors)
    以及市场概况 (market_regime, dominant_trend, sector_rotation).

    来源: FinMem-LLM-StockTrading 中的 long-term memory layer,
         但侧重于投资者/市场概况而非具体价格记忆.
    """

    max_items: int = 100
    _profile: dict = field(default_factory=dict)
    _items: list[MemoryItem] = field(default_factory=list)

    # Profile fields maintained by this layer
    PROFILE_KEYS: tuple[str, ...] = (
        "risk_tolerance",       # float [0, 1], 0=保守, 1=激进
        "investment_horizon",    # str, e.g. "short", "medium", "long"
        "preferred_sectors",      # list[str], e.g. ["tech", "finance"]
        "market_regime",          # str, e.g. "bull", "bear", "sideways"
        "dominant_trend",         # str, e.g. "uptrend", "downtrend"
        "volatility_level",       # float [0, 1]
        "last_updated",           # datetime
    )

    def __post_init__(self) -> None:
        # Initialise default profile
        for key in self.PROFILE_KEYS:
            if key not in self._profile:
                self._profile[key] = None
        self._profile["last_updated"] = datetime.now()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add(self, item: MemoryItem) -> None:
        """Add a memory item to the profiling layer.

        High-importance, reflection-category items are given extra weight
        when updating the profile.
        """
        if item.embedding is None:
            item.embedding = _hash_embedding(item.content)

        self._items.append(item)

        # Evict oldest items if over capacity
        if len(self._items) > self.max_items:
            self._items.pop(0)

        # If item is reflection or high importance, update profile
        if item.category in ("reflection", "trade") and item.importance > 0.6:
            self.update_profile(item.content)

    def get_profile(self) -> dict:
        """Return the current investor / market profile dict."""
        return dict(self._profile)

    def update_profile(self, observation: str) -> None:
        """Parse an observation string and update profile fields.

        Simple keyword-based extraction — replace with LLM calls in production.
        """
        obs_lower = observation.lower()

        # Risk tolerance
        risk_keywords = {
            "conservative": 0.2,
            "moderate": 0.5,
            "aggressive": 0.8,
            "high risk": 0.9,
            "low risk": 0.1,
        }
        for kw, val in risk_keywords.items():
            if kw in obs_lower:
                self._profile["risk_tolerance"] = val

        # Investment horizon
        horizon_map = {"short": "short", "swing": "medium", "long": "long"}
        for kw, val in horizon_map.items():
            if kw in obs_lower:
                self._profile["investment_horizon"] = val

        # Market regime
        regime_map = {
            "bull": "bull",
            "bear": "bear",
            "sideways": "sideways",
            "high volatility": "volatile",
        }
        for kw, val in regime_map.items():
            if kw in obs_lower:
                self._profile["market_regime"] = val

        # Trend
        if "uptrend" in obs_lower or "breakout" in obs_lower:
            self._profile["dominant_trend"] = "uptrend"
        elif "downtrend" in obs_lower or "breakdown" in obs_lower:
            self._profile["dominant_trend"] = "downtrend"

        # Sectors
        sector_pattern = re.findall(
            r"(?:sector|industry)\s+(\w+)",
            obs_lower,
        )
        if sector_pattern:
            existing = self._profile.get("preferred_sectors", [])
            self._profile["preferred_sectors"] = list(
                OrderedDict.fromkeys(existing + sector_pattern)
            )[:10]  # cap at 10

        self._profile["last_updated"] = datetime.now()

    def get_recent_items(self, n: int = 20) -> list[MemoryItem]:
        """Return the n most recent memory items."""
        return self._items[-n:]

    def clear(self) -> None:
        """Clear all profiling memory."""
        self._items.clear()
        for key in self.PROFILE_KEYS:
            self._profile[key] = None
        self._profile["last_updated"] = datetime.now()


# --------------------------------------------------------------------------- #
# Layer 2 — WorkingMemory
# --------------------------------------------------------------------------- #

@dataclass
class WorkingMemory:
    """第二层记忆: 工作记忆 — 当前决策上下文窗口 (可调认知跨度).

    FinMem 的核心创新之一: cognitive horizon (认知跨度) 可根据任务需求动态调整.
    horizon 参数控制工作记忆的窗口大小 — 较大的 horizon 适合长期趋势分析,
    较小的 horizon 适合短期高频决策.

    Attributes:
        horizon: 认知跨度, 即上下文窗口中保留的最大条目数 (default 10).
                 可通过 set_horizon() 动态调整.
    """

    horizon: int = 10
    _buffer: list[MemoryItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add(self, item: MemoryItem) -> None:
        """Add an item to the working memory buffer."""
        if item.embedding is None:
            item.embedding = _hash_embedding(item.content)
        self._buffer.append(item)
        self._evict_if_needed()

    def get_context(
        self,
        category: str | None = None,
        top_k: int | None = None,
    ) -> list[MemoryItem]:
        """Return the current context window.

        Args:
            category: Optional filter by category.
            top_k:    Return only the top-k most important items.

        Returns:
            List of MemoryItems in the working memory window.
        """
        items = self._buffer
        if category is not None:
            items = [it for it in items if it.category == category]
        if top_k is not None:
            items = sorted(items, key=lambda x: x.importance, reverse=True)[:top_k]
        return list(items)

    def set_horizon(self, horizon: int) -> None:
        """Dynamically adjust the cognitive horizon.

        Args:
            horizon: New window size (must be positive).
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        self.horizon = horizon
        self._evict_if_needed()

    def clear_old(self, cutoff: datetime) -> None:
        """Remove all items older than the given cutoff timestamp."""
        self._buffer = [it for it in self._buffer if it.timestamp >= cutoff]

    def clear(self) -> None:
        """Clear all working memory."""
        self._buffer.clear()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _evict_if_needed(self) -> None:
        """Evict oldest items when buffer exceeds horizon."""
        while len(self._buffer) > self.horizon:
            self._buffer.pop(0)

    def _score_item(self, item: MemoryItem, now: datetime) -> float:
        """Compute recency-adjusted importance score.

        Score = importance * exp(-age_in_minutes / decay_constant).
        Decay constant = 60 minutes.
        """
        age_minutes = (now - item.timestamp).total_seconds() / 60.0
        decay = np.exp(-age_minutes / 60.0)
        return item.importance * decay


# --------------------------------------------------------------------------- #
# Layer 3 — DecisionMemory
# --------------------------------------------------------------------------- #

@dataclass
class DecisionMemory:
    """第三层记忆: 决策记忆 — 交易决策及其结果的历史.

    记录每笔交易的:
      - 决策内容 (action, confidence, reasoning)
      - 执行结果 (PnL / return)
      - 记忆索引 (用于反馈更新)

    来源: FinMem-LLM-StockTrading 中的 decision log + reflection memory.
    """

    _decisions: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_decision(
        self,
        decision: dict,
        result: float | None = None,
    ) -> int:
        """Record a trading decision and its outcome.

        Args:
            decision: Decision dict with keys:
                      - action:      str (e.g. "buy", "sell", "hold")
                      - confidence: float [0, 1]
                      - reasoning:   str
                      - timestamp:   datetime (optional, defaults to now)
            result:   Execution result (e.g. PnL % or absolute return).
                      None means the trade is still open / result pending.

        Returns:
            Index of the recorded decision in the history list.
        """
        entry = {
            "action": decision.get("action", "unknown"),
            "confidence": decision.get("confidence", 0.5),
            "reasoning": decision.get("reasoning", ""),
            "timestamp": decision.get("timestamp") or datetime.now(),
            "result": result,
        }
        self._decisions.append(entry)
        return len(self._decisions) - 1

    def update_result(self, index: int, result: float) -> None:
        """Backfill the result for a previously recorded decision."""
        if 0 <= index < len(self._decisions):
            self._decisions[index]["result"] = result

    def get_decision_history(self, n: int = 50) -> list[dict]:
        """Return the n most recent decisions."""
        return list(self._decisions[-n:])

    def get_win_rate(self) -> float:
        """Compute win rate over closed trades (result is not None).

        win = result > 0.
        """
        closed = [d for d in self._decisions if d["result"] is not None]
        if not closed:
            return 0.0
        wins = sum(1 for d in closed if d["result"] > 0)
        return wins / len(closed)

    def get_avg_return(self) -> float:
        """Average return over closed trades."""
        closed = [d for d in self._decisions if d["result"] is not None]
        if not closed:
            return 0.0
        return sum(d["result"] for d in closed) / len(closed)

    def get_profitable_actions(self, min_return: float = 0.01) -> list[dict]:
        """Return actions that exceeded min_return threshold."""
        return [
            d for d in self._decisions
            if d["result"] is not None and d["result"] >= min_return
        ]

    def clear(self) -> None:
        """Clear all decision history."""
        self._decisions.clear()


# --------------------------------------------------------------------------- #
# FinMemCoordinator
# --------------------------------------------------------------------------- #

class FinMemCoordinator:
    """FinMem 协调器 — 三层记忆的协调与检索.

    核心工作流 (源自 FinMem-LLM-StockTrading):
      1. perceive(observation, category)  — 感知市场信息, 存入对应记忆层
      2. decide(state)                     — 基于三层记忆生成交易决策
      3. reflect()                        — 反思近期决策, 更新概况记忆

    Attributes:
        profiling:  第一层记忆 — 投资者/市场概况
        working:    第二层记忆 — 工作记忆 (当前上下文窗口)
        decision:   第三层记忆 — 决策历史
        horizon:    认知跨度 (即 working.horizon)

    Example:
        coordinator = FinMemCoordinator(llm_client=my_llm, horizon=12)
        coordinator.perceive("BTC 突破 50000 美元, 成交量放大", category="market_data")
        coordinator.perceive("美联储宣布不加息", category="news")
        decision = coordinator.decide({"price": 50000, "volume": 1e9, "position": 0})
        reflection = coordinator.reflect()
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        horizon: int = 10,
        logger: logging.Logger | None = None,
        embed_dim: int = 64,
    ) -> None:
        """
        Args:
            llm_client:  Optional LLM client for LLM-driven reflection / decide.
                         If None, a rule-based fallback is used.
            horizon:     Initial cognitive horizon for working memory.
            logger:      Optional logger instance.
            embed_dim:   Dimension of hash-based pseudo-embeddings.
        """
        self.llm_client = llm_client
        self.horizon = horizon
        self.logger = logger or logging.getLogger(__name__)
        self.embed_dim = embed_dim

        # Initialise three memory layers
        self.profiling = ProfilingMemory()
        self.working = WorkingMemory(horizon=horizon)
        self.decision = DecisionMemory()

        # Internal state
        self._pending_decision: dict | None = None

    # ------------------------------------------------------------------ #
    # perceive — Layer assignment
    # ------------------------------------------------------------------ #

    def perceive(
        self,
        observation: str,
        category: str = "market_data",
        importance: float = 0.5,
        timestamp: datetime | None = None,
    ) -> MemoryItem:
        """感知市场信息, 根据类别存入对应记忆层.

        Routing rules (FinMem-inspired):
          - 'market_data' → working memory (short-term context)
          - 'news'        → working memory (filtered by horizon)
          - 'trade'       → both working + profiling (update profile)
          - 'reflection'  → profiling memory (update investor/market profile)

        Args:
            observation: The observed text (price, news, decision, etc.).
            category:     One of 'market_data', 'news', 'trade', 'reflection'.
            importance:   Importance score [0, 1] for this observation.
            timestamp:    Event timestamp (default: now).

        Returns:
            The created MemoryItem.
        """
        ts = timestamp or datetime.now()
        item = MemoryItem(
            timestamp=ts,
            content=observation,
            importance=importance,
            category=category,
            embedding=_hash_embedding(observation, dim=self.embed_dim),
        )

        self.logger.info(
            f"[FinMem perceive] category={category} importance={importance:.2f} "
            f"content={observation[:80]!r}"
        )

        # Route to appropriate layers
        if category in ("market_data", "news"):
            self.working.add(item)

        elif category == "trade":
            self.working.add(item)
            self.profiling.add(item)  # Trade experiences update the profile

        elif category == "reflection":
            self.profiling.add(item)

        else:
            self.working.add(item)  # Default: working memory

        return item

    # ------------------------------------------------------------------ #
    # decide — LLM-driven or rule-based decision
    # ------------------------------------------------------------------ #

    def decide(self, state: dict) -> dict:
        """基于三层记忆生成交易决策.

        构建包含三层记忆摘要的 prompt, 调用 LLM (或使用规则 fallback).

        Args:
            state: Current market state dict, expected keys:
                   - price:      float, current price
                   - volume:     float, trading volume
                   - position:   float, current position (0 = flat)
                   - cash:       float, available cash
                   - portfolio_value: float

        Returns:
            Decision dict:
            {
                "action":     str,  # "buy" | "sell" | "hold"
                "confidence": float,  # [0, 1]
                "reasoning":  str,   # Human-readable reasoning
                "memory_items_used": list[MemoryItem],  # Items retrieved
            }
        """
        profile = self.profiling.get_profile()
        context = self.working.get_context()
        recent_decisions = self.decision.get_decision_history(n=10)
        win_rate = self.decision.get_win_rate()
        avg_return = self.decision.get_avg_return()

        # Build context summary strings
        context_str = "\n".join(
            f"  [{it.timestamp:%H:%M:%S}] [{it.category}] {it.content[:100]}"
            for it in context[-self.horizon:]
        ) or "  (empty)"

        decision_str = "\n".join(
            f"  [{d['timestamp']:%Y-%m-%d %H:%M}] {d['action']} @ "
            f"conf={d['confidence']:.2f} result={d['result']}"
            for d in recent_decisions
        ) or "  (no history)"

        # LLM-based decision path
        if self.llm_client is not None:
            prompt = self._build_decision_prompt(
                state=state,
                profile=profile,
                context_str=context_str,
                decision_str=decision_str,
                win_rate=win_rate,
                avg_return=avg_return,
            )
            try:
                raw = self.llm_client.generate(prompt)
                decision = self._parse_llm_decision(raw)
            except Exception as exc:
                self.logger.warning(f"LLM decide failed: {exc}, falling back to rule-based")
                decision = self._rule_based_decide(state, profile)
        else:
            decision = self._rule_based_decide(state, profile)

        # Record the decision
        decision["timestamp"] = datetime.now()
        decision["state_used"] = state
        idx = self.decision.record_decision(decision)
        self._pending_decision = decision.copy()
        self._pending_decision["_idx"] = idx

        self.logger.info(
            f"[FinMem decide] action={decision['action']} "
            f"confidence={decision['confidence']:.2f} "
            f"reasoning={decision['reasoning'][:80]!r}"
        )
        return decision

    # ------------------------------------------------------------------ #
    # reflect — Profile update via LLM or keyword extraction
    # ------------------------------------------------------------------ #

    def reflect(self) -> str:
        """反思近期决策, 更新概况记忆 (ProfilingMemory).

        分析最近的工作记忆和决策记录, 生成反思摘要, 写入 profiling layer.

        Returns:
            Reflection summary string.
        """
        context = self.working.get_context()
        recent_decisions = self.decision.get_decision_history(n=10)
        win_rate = self.decision.get_win_rate()

        if self.llm_client is not None:
            prompt = self._build_reflection_prompt(
                context=context,
                recent_decisions=recent_decisions,
                win_rate=win_rate,
            )
            try:
                reflection_text = self.llm_client.generate(prompt)
            except Exception as exc:
                self.logger.warning(f"LLM reflect failed: {exc}")
                reflection_text = self._rule_based_reflect(context, recent_decisions)
        else:
            reflection_text = self._rule_based_reflect(context, recent_decisions)

        # Store reflection in profiling memory
        reflection_item = MemoryItem(
            timestamp=datetime.now(),
            content=reflection_text,
            importance=0.8,
            category="reflection",
            embedding=_hash_embedding(reflection_text, dim=self.embed_dim),
        )
        self.profiling.add(reflection_item)

        self.logger.info(f"[FinMem reflect] {reflection_text[:120]!r}")
        return reflection_text

    # ------------------------------------------------------------------ #
    # Feedback — Record execution result for a pending decision
    # ------------------------------------------------------------------ #

    def record_result(self, result: float) -> None:
        """Record the execution result for the most recent decision.

        This closes the loop: decision → execution → result → reflection.

        Args:
            result: PnL or return percentage (e.g. 0.05 = 5% profit).
        """
        if self._pending_decision is not None:
            idx = self._pending_decision.get("_idx")
            if idx is not None:
                self.decision.update_result(idx, result)
            self._pending_decision = None

    # ------------------------------------------------------------------ #
    # Memory summary
    # ------------------------------------------------------------------ #

    def get_memory_summary(self) -> dict:
        """返回三层记忆的摘要.

        Returns:
            dict with keys: profiling, working, decision.
        """
        return {
            "profiling": {
                "profile": self.profiling.get_profile(),
                "item_count": len(self.profiling._items),
            },
            "working": {
                "horizon": self.working.horizon,
                "item_count": len(self.working._buffer),
                "items": [
                    {
                        "timestamp": it.timestamp.isoformat(),
                        "category": it.category,
                        "importance": it.importance,
                        "content": it.content[:60],
                    }
                    for it in self.working._buffer
                ],
            },
            "decision": {
                "total_decisions": len(self.decision._decisions),
                "win_rate": self.decision.get_win_rate(),
                "avg_return": self.decision.get_avg_return(),
                "recent": self.decision.get_decision_history(n=5),
            },
        }

    # ------------------------------------------------------------------ #
    # Internal prompt builders
    # ------------------------------------------------------------------ #

    def _build_decision_prompt(
        self,
        state: dict,
        profile: dict,
        context_str: str,
        decision_str: str,
        win_rate: float,
        avg_return: float,
    ) -> str:
        return (
            "You are a quantitative trading agent with memory.\n\n"
            "## Investor / Market Profile\n"
            f"{profile}\n\n"
            "## Recent Working Memory (cognitive horizon={self.horizon})\n"
            f"{context_str}\n\n"
            "## Recent Decision History\n"
            f"Win rate: {win_rate:.2%} | Avg return: {avg_return:.4f}\n"
            f"{decision_str}\n\n"
            "## Current Market State\n"
            f"Price: {state.get('price')} | Volume: {state.get('volume')} | "
            f"Position: {state.get('position')} | Cash: {state.get('cash')}\n\n"
            "Based on the above, output a JSON object with keys: "
            '{"action": "buy"|"sell"|"hold", "confidence": 0.0-1.0, "reasoning": "..."}'
        )

    def _build_reflection_prompt(
        self,
        context: list[MemoryItem],
        recent_decisions: list[dict],
        win_rate: float,
    ) -> str:
        ctx_str = "\n".join(
            f"[{it.category}] {it.content[:80]}" for it in context[-5:]
        )
        dec_str = "\n".join(
            f"{d['action']} @ conf={d['confidence']:.2f} result={d.get('result')}"
            for d in recent_decisions[-5:]
        )
        return (
            "You are a trading reflection agent.\n\n"
            f"## Recent Market Context\n{ctx_str}\n\n"
            f"## Recent Decisions\n{dec_str}\n\n"
            f"## Win Rate\n{win_rate:.2%}\n\n"
            "Write a brief reflection (1-3 sentences) on what the agent "
            "should learn or adjust based on this history."
        )

    # ------------------------------------------------------------------ #
    # LLM response parser
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_llm_decision(raw: str) -> dict:
        """Parse JSON or markdown code block from LLM output."""
        # Try JSON directly
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if match:
            try:
                import json
                obj = json.loads(match.group())
                return {
                    "action": str(obj.get("action", "hold").lower()),
                    "confidence": float(obj.get("confidence", 0.5)),
                    "reasoning": str(obj.get("reasoning", "")),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        # Fallback to rule-based on parse failure
        raw_lower = raw.lower()
        if "buy" in raw_lower and "sell" not in raw_lower:
            action = "buy"
        elif "sell" in raw_lower and "buy" not in raw_lower:
            action = "sell"
        else:
            action = "hold"
        conf = 0.5
        conf_match = re.search(r"confidence[:\s]+0?\.(\d+)", raw_lower)
        if conf_match:
            conf = float(f"0.{conf_match.group(1)}")
        return {"action": action, "confidence": conf, "reasoning": raw.strip()}

    # ------------------------------------------------------------------ #
    # Rule-based fallback (no LLM required)
    # ------------------------------------------------------------------ #

    def _rule_based_decide(self, state: dict, profile: dict) -> dict:
        """Simple rule-based decision when no LLM client is available.

        Rules:
          - If price up > 2% and position == 0 → buy (small size)
          - If price down > 2% and position > 0 → sell
          - Otherwise → hold
        """
        price = float(state.get("price", 0))
        position = float(state.get("position", 0))
        prev_price = getattr(self, "_prev_price", price)
        risk_tol = profile.get("risk_tolerance", 0.5)

        if price == 0:
            return {"action": "hold", "confidence": 0.0, "reasoning": "No price data"}

        pct_change = (price - prev_price) / prev_price if prev_price != 0 else 0.0
        self._prev_price = price

        if pct_change > 0.02 and position == 0:
            action = "buy"
            confidence = min(0.3 + risk_tol * 0.4, 0.9)
            reasoning = f"Price breakout +{pct_change:.2%}, no position"
        elif pct_change < -0.02 and position > 0:
            action = "sell"
            confidence = min(0.3 + risk_tol * 0.4, 0.9)
            reasoning = f"Price drop -{pct_change:.2%}, exiting position"
        else:
            action = "hold"
            confidence = 0.5
            reasoning = "No clear signal"

        return {"action": action, "confidence": confidence, "reasoning": reasoning}

    @staticmethod
    def _rule_based_reflect(
        context: list[MemoryItem],
        recent_decisions: list[dict],
    ) -> str:
        """Simple keyword-based reflection text."""
        if not context:
            return "No significant market signals detected."
        last = context[-1]
        wins = sum(1 for d in recent_decisions if (d.get("result") or 0) > 0)
        total = len(recent_decisions)
        win_rate = wins / total if total > 0 else 0.0
        return (
            f"Recent signal: [{last.category}] {last.content[:80]}. "
            f"Win rate over last {total} trades: {win_rate:.0%}."
        )


# --------------------------------------------------------------------------- #
# Optional: gymnasium-compatible wrapper
# --------------------------------------------------------------------------- #

try:
    import gymnasium as gym  # noqa: F401

    class FinMemGymWrapper:
        """Gymnasium-compatible wrapper for FinMemCoordinator.

        Exposes the standard `reset()` / `step()` interface.

        Note:
            This is a lightweight shim for environments that expect
            a gym-like agent API. Full position tracking must be
            handled by the external environment.
        """

        def __init__(
            self,
            llm_client: LLMClient | None = None,
            horizon: int = 10,
        ) -> None:
            self.coordinator = FinMemCoordinator(
                llm_client=llm_client,
                horizon=horizon,
            )
            self._last_obs: str = ""

        def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
            """Reset the agent's memory and return initial info."""
            # gymnasium-style seed is a no-op here but accepted for API compliance
            info = self.coordinator.get_memory_summary()
            return {}, info

        def step(
            self,
            action: dict | None = None,
        ) -> tuple[dict, float, bool, bool, dict]:
            """Execute one decision cycle.

            Args:
                action: Optional dict with 'result' (PnL) from the environment.
                        If provided, records the result for the last decision.

            Returns:
                (obs, reward, terminated, truncated, info)
                obs is the memory summary dict.
            """
            if action is not None and "result" in action:
                self.coordinator.record_result(action["result"])
            info = self.coordinator.get_memory_summary()
            # dummy values — real loop should be driven externally
            return {}, 0.0, False, False, info

    __all__ = list(__all__) + ["FinMemGymWrapper"]
except ImportError:
    # gymnasium not installed — skip the wrapper
    pass
