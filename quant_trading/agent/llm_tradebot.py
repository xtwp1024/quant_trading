"""
LLM-TradeBot: Multi-Provider LLM Trading System
=================================================

Absorbed from D:/Hive/Data/trading_repos/LLM-TradeBot/

Core capabilities:
- Adversarial decision pipeline with self-play validation
- 8-LLM provider support via REST (OpenAI, Anthropic, Gemini, DeepSeek, Groq, Cohere, Mistral, local)
- Multi-model ensemble voting across heterogeneous providers
- Risk management layer with configurable rules
- Trade execution adapter (plugs into existing connectors)

Pure Python stdlib + requests (lazy import).  No vendor SDKs.

Author: Absorbed from LLM-TradeBot (AI Trader Team)
Date: 2026-03-30
"""

from __future__ import annotations

import os
import json
import time
import logging
import re
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum

# ---------------------------------------------------------------------------
# Lazy requests import (stdlib fallback)
# ---------------------------------------------------------------------------
_has_requests = False
try:
    import requests as _requests
    _has_requests = True
except ImportError:
    _requests = None

def _http_post(url: str, headers: Dict[str, str], body: Dict[str, Any],
               timeout: int = 30) -> Dict[str, Any]:
    """
    Make an HTTP POST request using requests if available,
    otherwise fall back to urllib.  Returns parsed JSON.
    """
    if _has_requests:
        resp = _requests.post(url, json=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    # stdlib fallback using urllib
    import urllib.request
    import urllib.error
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # bubble up HTTP errors as exceptions with body
        body_text = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {e.reason}\n{body_text}"
        ) from e


def _http_get(url: str, headers: Optional[Dict[str, str]] = None,
              timeout: int = 30) -> Dict[str, Any]:
    """GET request with the same dual-backend strategy."""
    if _has_requests:
        resp = _requests.get(url, headers=headers or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    import urllib.request
    import urllib.error
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {e.reason}\n{body_text}"
        ) from e


# ---------------------------------------------------------------------------
# Logging helper (no external dependencies)
# ---------------------------------------------------------------------------
_logger = logging.getLogger("LLMTradeBot")


def _log(level: int, msg: str, *args, **kwargs):
    if _logger.isEnabledFor(level):
        _logger.log(level, msg.format(*args, **kwargs))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Action(str, Enum):
    """Standard trade action labels."""
    OPEN_LONG  = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG  = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD        = "hold"
    WAIT        = "wait"


@dataclass
class ProviderConfig:
    """
    Per-provider configuration.

    参数:
        provider:       Provider identifier (e.g. "openai", "deepseek")
        api_key:        API key (or env-var name prefixed with "$")
        base_url:       Override base URL (optional)
        model:          Model name (optional, uses provider default)
        temperature:    Sampling temperature (default 0.7)
        max_tokens:     Max output tokens (default 2048)
        timeout:        Request timeout in seconds (default 60)
        enabled:        Whether this provider is active (default True)
    """
    provider:    str
    api_key:     str
    base_url:    Optional[str]       = None
    model:       Optional[str]       = None
    temperature: float               = 0.7
    max_tokens:  int                 = 2048
    timeout:     int                 = 60
    enabled:     bool                = True

    # resolved at runtime
    _resolved_key: str = field(default="", repr=False)

    def resolve_key(self) -> str:
        """Resolve api_key from env-var if prefixed with $."""
        if self._resolved_key:
            return self._resolved_key
        key = self.api_key
        if key.startswith("$"):
            key = os.environ.get(key[1:], "")
        self._resolved_key = key
        return key


@dataclass
class ChatMessage:
    """A single chat message."""
    role:    str   # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """
    Unified LLM response from any provider.

    属性:
        content:         Generated text
        model:           Model used
        provider:        Provider identifier
        usage:           Token usage dict
        latency_ms:      Round-trip latency in ms
        raw_response:    Raw provider response (if available)
    """
    content:      str
    model:        str
    provider:     str
    usage:        Dict[str, int] = field(default_factory=dict)
    latency_ms:   int            = 0
    raw_response: Optional[Dict]  = None


@dataclass
class VoteResult:
    """
    Result from a single model vote.

    属性:
        provider:        Which provider generated this vote
        action:          Suggested action
        confidence:      Confidence score 0-100
        reasoning:       Human-readable reason
        raw_content:     Original text from the model
        latency_ms:     Provider latency
    """
    provider:     str
    action:       str
    confidence:   float
    reasoning:    str
    raw_content:  str
    latency_ms:   int = 0


@dataclass
class EnsembleResult:
    """
    Aggregated result from MultiModelVoter.

    属性:
        final_action:    Consensus action
        confidence:      Weighted confidence 0-100
        vote_counts:     Count of votes per action
        votes:           Individual VoteResult objects
        consensus_ratio: Fraction of voters agreeing on final_action
        disagreement_score: Entropy-based measure of voter disagreement
    """
    final_action:       str
    confidence:          float
    vote_counts:         Dict[str, int]
    votes:               List[VoteResult]
    consensus_ratio:     float
    disagreement_score:  float


@dataclass
class AdversarialResult:
    """
    Result from the adversarial self-play pipeline.

    属性:
        original_action:    Action proposed before challenge
        challenged_action:  Action after adversarial validation
        challenge_passes:   Whether the original survived all challenges
        challenges:         List of challenge descriptions
        final_confidence:   Adjusted confidence after challenges
        verdict:            "confirm", "reverse", "veto"
    """
    original_action:  str
    challenged_action: str
    challenge_passes:  bool
    challenges:       List[str]
    final_confidence: float
    verdict:           str  # "confirm" | "reverse" | "veto"


@dataclass
class TradeDecision:
    """
    Final trade decision ready for execution.

    属性:
        action:           Normalized action (HOLD / OPEN_LONG / etc.)
        symbol:           Trading symbol (e.g. "BTCUSDT")
        confidence:       Final confidence 0-100
        reasoning:        Human-readable explanation
        params:           Execution parameters (size, leverage, SL, TP…)
        adversarial_result: Adversarial validation result
        ensemble_result:  Ensemble voting result
        timestamp:        ISO timestamp
    """
    action:             str
    symbol:             str
    confidence:         float
    reasoning:          str
    params:             Dict[str, Any] = field(default_factory=dict)
    adversarial_result: Optional[AdversarialResult] = None
    ensemble_result:    Optional[EnsembleResult]  = None
    timestamp:          str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# ProviderRouter
# ---------------------------------------------------------------------------

class ProviderRouter:
    """
    Routes LLM chat requests to any of 8 supported providers via REST.

    用法示例 / Usage Example::

        router = ProviderRouter({
            "openai":  ProviderConfig(provider="openai",  api_key="$OPENAI_API_KEY"),
            "deepseek": ProviderConfig(provider="deepseek", api_key="$DEEPSEEK_API_KEY"),
        })
        resp = router.chat("openai", system="You are helpful.", user="Hello!")
        print(resp.content)

    Supported providers: openai, anthropic, gemini, deepseek, groq, cohere,
                         mistral, local
    """

    # Default base URLs for each provider
    DEFAULT_BASE_URLS: Dict[str, str] = {
        "openai":   "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "gemini":   "https://generativelanguage.googleapis.com/v1beta",
        "groq":     "https://api.groq.com/openai/v1",
        "cohere":   "https://api.cohere.ai/v1",
        "mistral":  "https://api.mistral.ai/v1",
        "local":    "http://localhost:8000/v1",
    }

    # Default model per provider
    DEFAULT_MODELS: Dict[str, str] = {
        "openai":    "gpt-4o",
        "deepseek":  "deepseek-chat",
        "anthropic": "claude-3-5-sonnet-20241022",
        "gemini":    "gemini-1.5-flash",
        "groq":      "llama-3.3-70b-versatile",
        "cohere":    "command-r-plus",
        "mistral":   "mistral-large-latest",
        "local":     "local-model",
    }

    def __init__(self, configs: Dict[str, ProviderConfig],
                 default_timeout: int = 60):
        """
        Initialize the router with per-provider configs.

        Args:
            configs:          Dict mapping provider name -> ProviderConfig
            default_timeout:  Default request timeout in seconds
        """
        self.configs: Dict[str, ProviderConfig] = {}
        for name, cfg in configs.items():
            cfg.provider = name  # normalise key
            if cfg.base_url is None:
                cfg.base_url = self.DEFAULT_BASE_URLS.get(name, "")
            if cfg.model is None:
                cfg.model = self.DEFAULT_MODELS.get(name, "unknown")
            self.configs[name] = cfg
        self.default_timeout = default_timeout
        _log(logging.INFO, "ProviderRouter initialized with providers: %s",
             list(self.configs.keys()))

    # -- public API ---------------------------------------------------------

    def chat(self, provider: str,
             system_prompt: str = "",
             user_prompt: str = "",
             messages: Optional[List[ChatMessage]] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             timeout: Optional[int] = None) -> LLMResponse:
        """
        Send a chat request to the named provider.

        Args:
            provider:     Provider name (e.g. "openai")
            system_prompt: System prompt (used if messages not provided)
            user_prompt:  User prompt   (used if messages not provided)
            messages:     Full message list (overrides system/user if given)
            temperature:  Override sampling temperature
            max_tokens:   Override max output tokens
            timeout:      Per-request timeout in seconds

        Returns:
            LLMResponse object
        """
        cfg = self._get_config(provider)
        msgs = messages or [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user",   content=user_prompt),
        ]
        if not temperature:
            temperature = cfg.temperature
        if not max_tokens:
            max_tokens = cfg.max_tokens
        if not timeout:
            timeout = cfg.timeout or self.default_timeout

        # Build per-provider request
        if provider == "anthropic":
            return self._chat_anthropic(cfg, msgs, temperature, max_tokens, timeout)
        if provider == "gemini":
            return self._chat_gemini(cfg, msgs, temperature, max_tokens, timeout)
        return self._chat_openai_compat(cfg, msgs, temperature, max_tokens, timeout)

    def chat_with_retry(self, provider: str,
                        system_prompt: str = "",
                        user_prompt: str = "",
                        messages: Optional[List[ChatMessage]] = None,
                        max_retries: int = 3,
                        temperature: Optional[float] = None,
                        max_tokens: Optional[int] = None) -> LLMResponse:
        """
        chat() with automatic retry on transient errors (429, 500-503).

        Returns:
            LLMResponse object (last error raised if all retries fail)
        """
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return self.chat(provider, system_prompt, user_prompt, messages,
                                 temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:
                last_err = exc
                # Check for retryable HTTP status
                retryable = False
                if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
                    retryable = exc.response.status_code in (429, 500, 502, 503, 504)
                elif isinstance(exc, RuntimeError) and "HTTP" in str(exc):
                    code = int(re.search(r"HTTP\s+(\d+)", str(exc)).group(1)) if re.search(r"HTTP\s+(\d+)", str(exc)) else 0
                    retryable = code in (429, 500, 502, 503, 504)
                if not retryable or attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                _log(logging.WARNING,
                     "Provider %s attempt %d/%d failed (%s), retrying in %.1fs",
                     provider, attempt + 1, max_retries, exc, wait)
                time.sleep(wait)
        raise last_err

    @property
    def enabled_providers(self) -> List[str]:
        """List of currently enabled provider names."""
        return [n for n, c in self.configs.items() if c.enabled]

    # -- private helpers -----------------------------------------------------

    def _get_config(self, provider: str) -> ProviderConfig:
        cfg = self.configs.get(provider)
        if cfg is None:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Available: {list(self.configs.keys())}"
            )
        if not cfg.enabled:
            raise ValueError(f"Provider '{provider}' is currently disabled.")
        return cfg

    def _chat_openai_compat(self, cfg: ProviderConfig,
                            messages: List[ChatMessage],
                            temperature: float,
                            max_tokens: int,
                            timeout: int) -> LLMResponse:
        """OpenAI-compatible REST endpoint (OpenAI, DeepSeek, Groq, Cohere, Mistral, local)."""
        url = f"{cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.resolve_key()}",
            "Content-Type":  "application/json",
        }
        body = {
            "model":       cfg.model,
            "messages":    [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        start = time.time()
        raw = _http_post(url, headers, body, timeout=timeout)
        latency = int((time.time() - start) * 1000)

        choice = (raw.get("choices") or [{}])[0]
        content = choice.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=raw.get("model", cfg.model),
            provider=cfg.provider,
            usage=raw.get("usage", {}),
            latency_ms=latency,
            raw_response=raw,
        )

    def _chat_anthropic(self, cfg: ProviderConfig,
                        messages: List[ChatMessage],
                        temperature: float,
                        max_tokens: int,
                        timeout: int) -> LLMResponse:
        """Anthropic Claude REST endpoint."""
        url = f"{cfg.base_url}/messages"
        headers = {
            "x-api-key":         cfg.resolve_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        }
        system_parts = []
        user_parts   = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                user_parts.append({"role": msg.role, "content": msg.content})

        body: Dict[str, Any] = {
            "model":       cfg.model,
            "messages":    user_parts,
            "max_tokens":  max_tokens,
        }
        if system_parts:
            body["system"] = "\n".join(system_parts)
        if temperature > 0:
            body["temperature"] = max(0.1, temperature)

        start = time.time()
        raw = _http_post(url, headers, body, timeout=timeout)
        latency = int((time.time() - start) * 1000)

        content = ""
        for block in raw.get("content", []):
            if block.get("type") == "text":
                content = block.get("text", "")
                break

        usage = raw.get("usage", {})
        return LLMResponse(
            content=content,
            model=raw.get("model", cfg.model),
            provider=cfg.provider,
            usage={
                "prompt_tokens":     usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens":      usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
            latency_ms=latency,
            raw_response=raw,
        )

    def _chat_gemini(self, cfg: ProviderConfig,
                      messages: List[ChatMessage],
                      temperature: float,
                      max_tokens: int,
                      timeout: int) -> LLMResponse:
        """Google Gemini REST endpoint."""
        url = (
            f"{cfg.base_url}/models/{cfg.model}:generateContent"
            f"?key={cfg.resolve_key()}"
        )
        headers = {"Content-Type": "application/json"}

        system_instruction = None
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                role = "model" if msg.role == "assistant" else msg.role
                contents.append({
                    "role":  role,
                    "parts": [{"text": msg.content}],
                })

        body: Dict[str, Any] = {
            "contents":            contents,
            "generationConfig": {
                "temperature":  temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            body["systemInstruction"] = system_instruction

        start = time.time()
        raw = _http_post(url, headers, body, timeout=timeout)
        latency = int((time.time() - start) * 1000)

        candidates = raw.get("candidates", [])
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                content = parts[0].get("text", "")

        um = raw.get("usageMetadata", {})
        return LLMResponse(
            content=content,
            model=cfg.model,
            provider=cfg.provider,
            usage={
                "prompt_tokens":     um.get("promptTokenCount", 0),
                "completion_tokens": um.get("candidatesTokenCount", 0),
                "total_tokens":      um.get("totalTokenCount", 0),
            },
            latency_ms=latency,
            raw_response=raw,
        )


# ---------------------------------------------------------------------------
# MultiModelVoter
# ---------------------------------------------------------------------------

class MultiModelVoter:
    """
    Run an ensemble vote across multiple LLM providers.

    Each provider independently reasons about the market and returns an
    action + confidence.  Results are aggregated using weighted majority
    voting or confidence-averaging.

    用法示例 / Usage Example::

        voter = MultiModelVoter(router, {
            "openai":   1.0,
            "deepseek": 1.0,
            "anthropic": 0.8,
        }, action_parser=default_action_parser)
        result = voter.vote(
            system="You are a trading analyst.",
            user="Analyse BTC/USDT now...",
        )
        print(result.final_action, result.confidence)

    Attributes:
        aggregation: One of "majority" (weighted counts) or
                    "confidence_weighted" (weighted avg confidence).
    """

    def __init__(self,
                 router: ProviderRouter,
                 weights: Optional[Dict[str, float]] = None,
                 action_parser: Optional[callable] = None,
                 aggregation: str = "majority"):
        """
        Args:
            router:         ProviderRouter instance
            weights:        Provider name -> float weight (default all 1.0)
            action_parser:  Callable(LLMResponse) -> VoteResult
                           (uses default_parse_vote if None)
            aggregation:    "majority" | "confidence_weighted"
        """
        self.router        = router
        self.weights       = weights or {}
        self.action_parser = action_parser or self._default_parse_vote
        self.aggregation   = aggregation

    def vote(self,
             system_prompt: str = "",
             user_prompt: str = "",
             messages: Optional[List[ChatMessage]] = None,
             enabled_providers: Optional[List[str]] = None,
             temperature: float = 0.3,
             max_tokens: int = 512) -> EnsembleResult:
        """
        Collect votes from all enabled providers and aggregate.

        Args:
            system_prompt:      System prompt for all providers
            user_prompt:         User prompt
            messages:            Full message list (overrides system/user)
            enabled_providers:   Subset of providers to query (default all)
            temperature:        Sampling temperature (kept low for votes)
            max_tokens:         Max output tokens per vote

        Returns:
            EnsembleResult
        """
        providers = enabled_providers or self.router.enabled_providers
        votes: List[VoteResult] = []

        for p in providers:
            try:
                cfg = self.router.configs.get(p)
                if cfg is None or not cfg.enabled:
                    continue
                resp = self.router.chat_with_retry(
                    p,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                vote = self.action_parser(resp)
                vote.provider = p
                votes.append(vote)
                _log(logging.DEBUG, "[%s] vote: action=%s conf=%.1f",
                     p, vote.action, vote.confidence)
            except Exception as exc:
                _log(logging.WARNING, "Provider %s vote failed: %s", p, exc)

        if not votes:
            return EnsembleResult(
                final_action="hold",
                confidence=0.0,
                vote_counts={"hold": 1},
                votes=[],
                consensus_ratio=0.0,
                disagreement_score=1.0,
            )

        return self._aggregate(votes)

    @staticmethod
    def _default_parse_vote(resp: LLMResponse) -> VoteResult:
        """
        Parse an LLMResponse into a VoteResult using lightweight heuristics.

        Strategy: look for JSON within the text, otherwise use keyword scoring.
        """
        raw = resp.content.strip()

        # Try JSON extraction
        action, confidence, reasoning = MultiModelVoter._extract_json(raw)

        # Fallback: keyword scoring
        if action == "unknown":
            action, confidence, reasoning = MultiModelVoter._keyword_vote(raw)

        # Normalise
        action = action.lower().strip()
        if action not in (
            "open_long", "open_short", "close_long", "close_short", "hold", "wait"
        ):
            action = "hold"

        return VoteResult(
            provider=resp.provider,
            action=action,
            confidence=float(max(0.0, min(100.0, confidence))),
            reasoning=reasoning,
            raw_content=raw,
            latency_ms=resp.latency_ms,
        )

    @staticmethod
    def _extract_json(raw: str
                      ) -> Tuple[str, float, str]:
        """Try to extract action/confidence/reasoning from JSON markdown or bare JSON."""
        # Try markdown-fenced JSON first
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if not m:
            m = re.search(r"(\{.*?\})", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                act = str(obj.get("action", "")).lower()
                conf = float(obj.get("confidence", obj.get("confidence_score", 50.0)))
                reason = str(obj.get("reasoning", obj.get("reason", "")))
                return act, conf, reason
            except Exception:
                pass
        return "unknown", 50.0, raw[:200]

    @staticmethod
    def _keyword_vote(text: str) -> Tuple[str, float, str]:
        """Heuristic vote from plain text when JSON parsing fails."""
        text_lower = text.lower()

        # Simple keyword scoring
        long_kw   = ["open_long", "做多", "买入", "long", "bullish", "buy"]
        short_kw  = ["open_short", "做空", "卖出", "short", "bearish", "sell"]
        hold_kw   = ["hold", "wait", "观望", "不操作", "no action"]
        close_kw  = ["close", "平仓", "exit", "平多", "平空"]

        def score(kws):
            return sum(1 for kw in kws if kw in text_lower)

        s_long  = score(long_kw)
        s_short = score(short_kw)
        s_hold  = score(hold_kw)
        s_close = score(close_kw)

        best = max(s_long, s_short, s_hold, s_close)
        if best == 0 or (s_long == best and s_short == best):
            return "hold", 50.0, text[:200]

        if s_long == best:   return "open_long",  60.0 + min(s_long * 5, 20), text[:200]
        if s_short == best:  return "open_short",  60.0 + min(s_short * 5, 20), text[:200]
        if s_close == best: return "close_long",  55.0 + min(s_close * 5, 20), text[:200]
        return "hold", 50.0, text[:200]

    def _aggregate(self, votes: List[VoteResult]) -> EnsembleResult:
        """Aggregate individual votes into an EnsembleResult."""
        weights = {v.provider: self.weights.get(v.provider, 1.0) for v in votes}

        if self.aggregation == "majority":
            return self._majority_aggregate(votes, weights)
        else:
            return self._confidence_weighted_aggregate(votes, weights)

    def _majority_aggregate(
        self, votes: List[VoteResult], weights: Dict[str, float]
    ) -> EnsembleResult:
        """Weighted majority vote."""
        action_weights: Dict[str, float] = {}
        for v in votes:
            action_weights[v.action] = action_weights.get(v.action, 0.0) + weights[v.provider]

        total = sum(action_weights.values())
        if total <= 0:
            return self._empty_result()

        best_action = max(action_weights, key=action_weights.get)
        best_weight = action_weights[best_action]

        # Confidence = proportion of weight on winning action * avg confidence on winner
        consensus_ratio = best_weight / total
        winner_confidences = [v.confidence for v in votes if v.action == best_action]
        confidence = sum(winner_confidences) / len(winner_confidences) if winner_confidences else 0.0

        # Disagreement score (normalised entropy)
        disagreement = self._entropy(action_weights, total)

        return EnsembleResult(
            final_action=best_action,
            confidence=confidence * consensus_ratio,
            vote_counts={a: sum(1 for v in votes if v.action == a) for a in action_weights},
            votes=votes,
            consensus_ratio=consensus_ratio,
            disagreement_score=disagreement,
        )

    def _confidence_weighted_aggregate(
        self, votes: List[VoteResult], weights: Dict[str, float]
    ) -> EnsembleResult:
        """Weighted-average confidence per action, pick highest."""
        action_scores: Dict[str, Tuple[float, float]] = {}
        for v in votes:
            w = weights.get(v.provider, 1.0)
            cur = action_scores.get(v.action, (0.0, 0.0))
            action_scores[v.action] = (cur[0] + v.confidence * w, cur[1] + w)

        total_weight = sum(w for _, w in action_scores.values())
        if total_weight <= 0:
            return self._empty_result()

        action_avg = {a: s / w for a, (s, w) in action_scores.items()}
        best_action = max(action_avg, key=action_avg.get)

        disagreement = self._entropy(
            {a: w for a, (_, w) in action_scores.items()}, total_weight
        )

        return EnsembleResult(
            final_action=best_action,
            confidence=action_avg[best_action],
            vote_counts={a: sum(1 for v in votes if v.action == a) for a in action_scores},
            votes=votes,
            consensus_ratio=action_scores[best_action][1] / total_weight,
            disagreement_score=disagreement,
        )

    @staticmethod
    def _entropy(action_weights: Dict[str, float], total: float) -> float:
        """Normalised entropy (0 = full agreement, 1 = max disagreement)."""
        if total <= 0:
            return 1.0
        probs = [w / total for w in action_weights.values() if w > 0]
        if not probs:
            return 1.0
        h = -sum(p * math.log(p + 1e-9) for p in probs)
        max_h = math.log(len(probs) + 1e-9)
        return h / (max_h + 1e-9)

    def _empty_result(self) -> EnsembleResult:
        return EnsembleResult(
            final_action="hold",
            confidence=0.0,
            vote_counts={},
            votes=[],
            consensus_ratio=0.0,
            disagreement_score=1.0,
        )


# ---------------------------------------------------------------------------
# AdversarialPipeline
# ---------------------------------------------------------------------------

class AdversarialPipeline:
    """
    Adversarial self-play pipeline that challenges a proposed decision
    from multiple angles before it is confirmed.

    Challenge categories:
      1. Counter-trend challenge   – "What if the 1h trend is against this?"
      2. Volatility challenge      – "What if volatility spikes 2x?"
      3. Liquidity challenge       – "What if spread widens 3x?"
      4. Time-horizon challenge    – "What if the signal is short-lived?"
      5. Sentiment reversal        – "What if social sentiment flips?"

    Each challenge runs an LLM-in-the-loop query and scores whether the
    original action is robust.  If too many challenges succeed, the action
    is reversed or vetoed.

    用法示例 / Usage Example::

        adv = AdversarialPipeline(router, system_prompt="You are a risk auditor.")
        result = adv.challenge(
            proposed_action="open_long",
            confidence=75.0,
            market_context={"trend": "bullish", "rsi": 65, "volume_ratio": 1.4},
        )
        print(result.verdict, result.final_confidence)
    """

    def __init__(self,
                 router: ProviderRouter,
                 system_prompt: str = "You are a rigorous trading risk auditor.",
                 challenge_count: int = 4,
                 confidence_threshold: float = 30.0,
                 veto_threshold: float = 60.0):
        """
        Args:
            router:               ProviderRouter for LLM calls
            system_prompt:        System prompt for the auditor LLM
            challenge_count:      How many challenge categories to run
            confidence_threshold: Challenges reduce confidence by at most this %
        """
        self.router             = router
        self.system_prompt     = system_prompt
        self.challenge_count   = challenge_count
        self.confidence_threshold = confidence_threshold
        self.veto_threshold    = veto_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def challenge(self,
                  proposed_action: str,
                  confidence: float,
                  market_context: Dict[str, Any],
                  challenge_provider: str = "deepseek",
                  temperature: float = 0.2,
                  max_tokens: int = 256) -> AdversarialResult:
        """
        Run adversarial challenges against a proposed decision.

        Args:
            proposed_action:  Action to challenge (e.g. "open_long")
            confidence:        Original confidence 0-100
            market_context:    Dict with market data (trend, rsi, volume_ratio, etc.)
            challenge_provider: Which provider to use for challenge queries
            temperature:       LLM sampling temperature
            max_tokens:        LLM max output tokens

        Returns:
            AdversarialResult
        """
        original = proposed_action
        challenges: List[str] = []
        challenge_failures = 0
        adjusted_conf = confidence

        for category in self._challenge_order():
            prompt = self._build_challenge_prompt(category, original, market_context)
            try:
                resp = self.router.chat_with_retry(
                    challenge_provider,
                    system_prompt=self.system_prompt,
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                verdict = self._parse_challenge_verdict(resp.content)
                challenges.append(f"{category}: {verdict}")

                if verdict in ("reverse", "veto"):
                    challenge_failures += 1
                    # Reduce confidence based on severity
                    penalty = 15.0 if verdict == "veto" else 10.0
                    adjusted_conf = max(0.0, adjusted_conf - penalty)
            except Exception as exc:
                _log(logging.WARNING, "Challenge %s failed: %s", category, exc)
                challenges.append(f"{category}: error ({exc})")

            if len(challenges) >= self.challenge_count:
                break

        # Determine final verdict
        if challenge_failures == 0:
            verdict = "confirm"
            final_action = original
        elif challenge_failures == 1:
            verdict = "confirm" if adjusted_conf >= self.confidence_threshold else "reverse"
            final_action = original if verdict == "confirm" else self._reverse_action(original)
        else:
            verdict = "veto"
            final_action = "hold"

        return AdversarialResult(
            original_action=original,
            challenged_action=final_action,
            challenge_passes=(verdict in ("confirm",) and final_action != "hold"),
            challenges=challenges,
            final_confidence=adjusted_conf,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    CHALLENGE_TEMPLATES: Dict[str, str] = {
        "counter_trend": (
            "A trading system proposes {action} on {symbol}. "
            "The 1-hour trend is opposite to this direction. "
            "Should this action be reversed or confirmed? "
            "Respond with JSON: {{\"verdict\": \"confirm|reverse|veto\", \"reason\": \"...\"}}"
        ),
        "volatility": (
            "A {action} signal appears on {symbol}. "
            "Market volatility is 2x the normal level. "
            "Is this action too risky to confirm? "
            "Respond with JSON: {{\"verdict\": \"confirm|reverse|veto\", \"reason\": \"...\"}}"
        ),
        "liquidity": (
            "A {action} signal is generated for {symbol}. "
            "Spread has widened 3x. Should this action be vetoed? "
            "Respond with JSON: {{\"verdict\": \"confirm|reverse|veto\", \"reason\": \"...\"}}"
        ),
        "time_horizon": (
            "A {action} signal on {symbol} is expected to last less than 15 minutes. "
            "Is this action worth taking? "
            "Respond with JSON: {{\"verdict\": \"confirm|reverse|veto\", \"reason\": \"...\"}}"
        ),
        "sentiment_reversal": (
            "A {action} signal appears on {symbol}. "
            "Social sentiment has reversed in the last hour. "
            "Should this action be reversed? "
            "Respond with JSON: {{\"verdict\": \"confirm|reverse|veto\", \"reason\": \"...\"}}"
        ),
    }

    def _challenge_order(self) -> List[str]:
        """Order in which challenges are applied (most important first)."""
        base = ["counter_trend", "volatility", "liquidity", "time_horizon", "sentiment_reversal"]
        random.shuffle(base)
        return base

    def _build_challenge_prompt(self, category: str, action: str,
                                 market_context: Dict[str, Any]) -> str:
        """Fill in the challenge template with market context."""
        symbol = market_context.get("symbol", "UNKNOWN")
        template = self.CHALLENGE_TEMPLATES.get(category, self.CHALLENGE_TEMPLATES["counter_trend"])
        return template.format(action=action, symbol=symbol)

    @staticmethod
    def _parse_challenge_verdict(text: str) -> str:
        """Extract verdict string from LLM response."""
        text = text.strip()
        m = re.search(r'"verdict"\s*:\s*"(\w+)"', text, re.IGNORECASE)
        if m:
            v = m.group(1).lower()
            if v in ("confirm", "reverse", "veto"):
                return v
        # Fallback keyword check
        if "veto" in text.lower():
            return "veto"
        if "reverse" in text.lower():
            return "reverse"
        return "confirm"

    @staticmethod
    def _reverse_action(action: str) -> str:
        """Return the opposite action."""
        mapping = {
            "open_long":  "open_short",
            "open_short": "open_long",
            "close_long": "close_short",
            "close_short": "close_long",
            "hold":       "hold",
            "wait":       "wait",
        }
        return mapping.get(action, "hold")


# ---------------------------------------------------------------------------
# TradeExecutor
# ---------------------------------------------------------------------------

class TradeExecutor:
    """
    Execution adapter that bridges LLMTradeBot decisions to the
    underlying trading connectors (Binance, etc.).

    This class is a thin, opinion-agnostic adapter.  It does NOT implement
    order routing or risk checks itself – those are delegated to the
    connectors and the RiskManager passed at construction.

    用法示例 / Usage Example::

        executor = TradeExecutor(connector=my_binance_connector)
        decision = TradeDecision(action="open_long", symbol="BTCUSDT", confidence=80.0,
                                 params={"size": 0.01, "leverage": 2, "stop_loss": 59000})
        result = executor.execute(decision)
        print(result)
    """

    # Default risk limits (used when no RiskManager is provided)
    DEFAULT_MAX_LEVERAGE        = 5
    DEFAULT_MAX_POSITION_PCT    = 30.0
    DEFAULT_MAX_RISK_PER_TRADE  = 1.5

    def __init__(self,
                 connector: Optional[Any] = None,
                 risk_manager: Optional[Any] = None,
                 dry_run: bool = True):
        """
        Args:
            connector:    Trading connector (e.g. BinanceClient).  If None,
                           only dry-run execution is possible.
            risk_manager:  Risk management layer.  If None, internal
                           limits are applied.
            dry_run:       If True, only simulate execution without sending orders.
        """
        self.connector   = connector
        self.risk_manager = risk_manager
        self.dry_run     = dry_run
        _log(logging.INFO, "TradeExecutor initialized (dry_run=%s, connector=%s)",
             dry_run, connector is not None)

    def execute(self, decision: TradeDecision) -> Dict[str, Any]:
        """
        Execute (or simulate) a TradeDecision.

        Args:
            decision: TradeDecision from LLMTradeBot

        Returns:
            Execution result dict with keys:
                success (bool), action, symbol, message,
                orders (list), execution_type (str)
        """
        action  = decision.action
        symbol  = decision.symbol
        params  = decision.params

        result: Dict[str, Any] = {
            "success":         False,
            "action":          action,
            "symbol":          symbol,
            "message":         "",
            "orders":          [],
            "execution_type":  "dry_run" if self.dry_run else "live",
            "timestamp":       datetime.now().isoformat(),
        }

        # Normalise action
        normalized = self._normalize_action(action)
        if normalized in ("hold", "wait"):
            result["success"] = True
            result["message"] = "No action required (hold/wait)."
            return result

        if self.dry_run:
            return self._execute_dry_run(normalized, symbol, params, result)

        return self._execute_live(normalized, symbol, params, result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_dry_run(self, action: str, symbol: str,
                         params: Dict, result: Dict) -> Dict[str, Any]:
        """Simulate execution without sending real orders."""
        size      = params.get("position_size", 0.0)
        leverage  = min(params.get("leverage", 1), self.DEFAULT_MAX_LEVERAGE)
        sl_pct    = params.get("stop_loss_pct", 1.5)
        tp_pct    = params.get("take_profit_pct", 3.0)

        result["success"] = True
        result["message"] = (
            f"[DRY RUN] Would execute {action} {symbol} "
            f"size={size} leverage={leverage}x SL={sl_pct}% TP={tp_pct}%"
        )
        result["orders"] = [{
            "orderId":     f"dry_{int(time.time())}",
            "symbol":      symbol,
            "side":        self._action_to_side(action),
            "type":        "market",
            "quantity":    size,
            "status":      "dry_run",
        }]
        _log(logging.INFO, "DRY RUN: %s", result["message"])
        return result

    def _execute_live(self, action: str, symbol: str,
                      params: Dict, result: Dict) -> Dict[str, Any]:
        """Execute via the real trading connector."""
        if self.connector is None:
            result["message"] = "No trading connector configured."
            return result

        # Apply risk checks if a risk_manager is available
        if self.risk_manager:
            risk_ok, risk_msg = self._apply_risk_checks(action, symbol, params)
            if not risk_ok:
                result["message"] = f"Risk check failed: {risk_msg}"
                return result

        size     = params.get("position_size", 100.0)
        leverage = min(params.get("leverage", 1), self.DEFAULT_MAX_LEVERAGE)
        sl       = params.get("stop_loss")
        tp       = params.get("take_profit")

        try:
            if action in ("open_long", "open_short"):
                order = self.connector.place_market_order(
                    symbol=symbol,
                    side=self._action_to_side(action),
                    quantity=size,
                    position_side=self._action_to_position_side(action),
                )
                result["orders"].append(order)

                if sl is not None or tp is not None:
                    sl_orders = []
                    if sl is not None and self.connector:
                        sl_orders = self.connector.set_stop_loss_take_profit(
                            symbol=symbol,
                            stop_loss_price=sl,
                            take_profit_price=tp,
                            position_side=self._action_to_position_side(action),
                        )
                        result["orders"].extend(sl_orders)

                result["success"] = True
                result["message"] = f"Live {action} {symbol} executed."

            elif action in ("close_long", "close_short"):
                # Reduce-only close
                order = self.connector.place_market_order(
                    symbol=symbol,
                    side=self._action_to_side(action),
                    quantity=abs(size),
                    reduce_only=True,
                )
                result["orders"].append(order)
                result["success"] = True
                result["message"] = f"Live {action} {symbol} executed."

            else:
                result["message"] = f"Unknown action: {action}"

        except Exception as exc:
            result["message"] = f"Execution failed: {exc}"
            _log(logging.ERROR, "Live execution error: %s", exc)

        return result

    def _apply_risk_checks(self, action: str, symbol: str,
                            params: Dict) -> Tuple[bool, str]:
        """Run risk manager checks. Returns (ok, message)."""
        if self.risk_manager is None:
            return True, "ok"

        # Build a minimal decision dict for the RiskManager interface
        decision = {
            "symbol":          symbol,
            "action":          action,
            "leverage":        params.get("leverage", 1),
            "position_size_pct": params.get("position_size_pct", 10.0),
            "stop_loss_pct":   params.get("stop_loss_pct", 1.5),
            "take_profit_pct": params.get("take_profit_pct", 3.0),
            "current_price":   params.get("current_price", 0),
        }

        try:
            is_valid, _, msg = self.risk_manager.validate_decision(
                decision,
                account_info=params.get("account_info", {}),
                position_info=params.get("position_info"),
                market_snapshot=params.get("market_snapshot"),
            )
            return is_valid, msg
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _normalize_action(action: str) -> str:
        """Canonicalise action string."""
        mapping = {
            "open_long":    "open_long",
            "long":         "open_long",
            "buy":          "open_long",
            "open_short":   "open_short",
            "short":        "open_short",
            "sell":         "open_short",
            "close_long":   "close_long",
            "close_short":  "close_short",
            "close":        "close_long",
            "hold":         "hold",
            "wait":         "wait",
            "none":         "hold",
        }
        return mapping.get(str(action).lower().strip(), "hold")

    @staticmethod
    def _action_to_side(action: str) -> str:
        return "BUY" if action in ("open_long", "close_short") else "SELL"

    @staticmethod
    def _action_to_position_side(action: str) -> str:
        if action in ("open_long", "close_short"):
            return "LONG"
        if action in ("open_short", "close_long"):
            return "SHORT"
        return "BOTH"


# ---------------------------------------------------------------------------
# LLMTradeBot (main orchestrator)
# ---------------------------------------------------------------------------

class LLMTradeBot:
    """
    Main orchestrator that combines ProviderRouter, MultiModelVoter,
    AdversarialPipeline, and TradeExecutor into a single decision loop.

    Workflow:
        1. Collect market context
        2. Run MultiModelVoter → ensemble action
        3. Run AdversarialPipeline → validate / reverse action
        4. Build TradeDecision with execution params
        5. Delegate to TradeExecutor

    用法示例 / Usage Example::

        bot = LLMTradeBot(config={
            "openai":   {"api_key": os.environ["OPENAI_API_KEY"]},
            "deepseek": {"api_key": os.environ["DEEPSEEK_API_KEY"]},
            "anthropic": {"api_key": os.environ["ANTHROPIC_API_KEY"]},
        })
        decision = bot.decide(
            system="You are an expert crypto trader.",
            user="Analyse BTC/USDT now.",
            market_context={"trend": "bullish", "rsi": 62, "volume_ratio": 1.5,
                            "symbol": "BTCUSDT"},
        )
        result = bot.execute(decision)
        print(result)

    Attributes:
        router:      ProviderRouter instance (accessible for direct use)
        voter:       MultiModelVoter instance
        adversarial: AdversarialPipeline instance
        executor:     TradeExecutor instance
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert cryptocurrency trading analyst. "
        "Analyse the provided market data and respond with a JSON object "
        "containing your trading decision.\n\n"
        "Output format:\n"
        "```json\n"
        "{\n"
        '  "action": "open_long|open_short|close_long|close_short|hold|wait",\n'
        '  "confidence": 0-100,\n'
        '  "reasoning": "brief explanation (max 50 words)",\n'
        '  "params": {\n'
        '    "leverage": 1-5,\n'
        '    "position_size_pct": 0-30,\n'
        '    "stop_loss_pct": 0.5-5.0,\n'
        '    "take_profit_pct": 1.0-10.0\n'
        "  }\n"
        "}\n"
        "```"
    )

    def __init__(self,
                 config: Dict[str, Dict[str, Any]],
                 system_prompt: Optional[str] = None,
                 enabled_providers: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 connectors: Optional[Dict[str, Any]] = None,
                 risk_manager: Optional[Any] = None,
                 adversarial_provider: str = "deepseek",
                 voter_provider: Optional[str] = None,
                 dry_run: bool = True):
        """
        Initialise LLMTradeBot.

        Args:
            config:                Provider configs. Keys are provider names,
                                   values are dicts with fields:
                                     api_key, base_url (opt), model (opt),
                                     temperature (opt), max_tokens (opt),
                                     enabled (opt)
            system_prompt:         System prompt for all LLM calls
            enabled_providers:     Subset of providers to use for voting
            weights:               Provider weights for voting
            connectors:            Dict of named connectors (e.g. {"binance": client})
            risk_manager:           Risk management object
            adversarial_provider:   Provider for adversarial challenges
            voter_provider:        Preferred single provider for fast voting
                                    (if None, uses all enabled_providers)
            dry_run:               Run in simulation mode
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Build ProviderConfig objects
        provider_configs: Dict[str, ProviderConfig] = {}
        for name, cfg_dict in config.items():
            provider_configs[name] = ProviderConfig(
                provider=name,
                api_key=cfg_dict.get("api_key", ""),
                base_url=cfg_dict.get("base_url"),
                model=cfg_dict.get("model"),
                temperature=float(cfg_dict.get("temperature", 0.7)),
                max_tokens=int(cfg_dict.get("max_tokens", 2048)),
                timeout=int(cfg_dict.get("timeout", 60)),
                enabled=bool(cfg_dict.get("enabled", True)),
            )

        self.router = ProviderRouter(
            configs=provider_configs,
            default_timeout=60,
        )

        # Filter to only enabled providers for voting
        voting_providers = enabled_providers or [
            n for n, c in self.router.configs.items() if c.enabled
        ]

        self.voter = MultiModelVoter(
            router=self.router,
            weights=weights,
            aggregation="majority",
        )
        self.voting_providers = [p for p in voting_providers if p in self.router.configs]

        self.adversarial = AdversarialPipeline(
            router=self.router,
            system_prompt=(
                "You are a rigorous trading risk auditor. "
                "For each challenge, respond ONLY with JSON: "
                "{\"verdict\": \"confirm|reverse|veto\", \"reason\": \"...\"}"
            ),
        )
        self.adversarial_provider = adversarial_provider

        connector = connectors.get("binance") if connectors else None
        self.executor = TradeExecutor(
            connector=connector,
            risk_manager=risk_manager,
            dry_run=dry_run,
        )

        _log(logging.INFO,
             "LLMTradeBot initialised: providers=%s, dry_run=%s",
             list(self.router.configs.keys()), dry_run)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self,
               system: Optional[str] = None,
               user: str = "",
               market_context: Optional[Dict[str, Any]] = None,
               providers: Optional[List[str]] = None,
               temperature: float = 0.3,
               run_adversarial: bool = True) -> TradeDecision:
        """
        Make a trading decision.

        Args:
            system:          Override system prompt
            user:            User prompt (market analysis request)
            market_context:  Market data dict (trend, rsi, volume_ratio, symbol, …)
            providers:       Override voting providers
            temperature:      LLM sampling temperature for voting
            run_adversarial: Whether to run adversarial validation

        Returns:
            TradeDecision
        """
        ctx = market_context or {}

        # --- Step 1: Ensemble vote ---
        effective_providers = providers or self.voting_providers
        ensemble_result = self.voter.vote(
            system_prompt=system or self.system_prompt,
            user_prompt=user,
            enabled_providers=effective_providers,
            temperature=temperature,
        )

        # --- Step 2: Adversarial pipeline ---
        adv_result: Optional[AdversarialResult] = None
        challenged_action = ensemble_result.final_action
        final_confidence = ensemble_result.confidence

        if run_adversarial and ensemble_result.final_action not in ("hold", "wait"):
            try:
                adv_result = self.adversarial.challenge(
                    proposed_action=ensemble_result.final_action,
                    confidence=final_confidence,
                    market_context=ctx,
                    challenge_provider=self.adversarial_provider,
                )
                challenged_action = adv_result.challenged_action
                final_confidence  = adv_result.final_confidence
            except Exception as exc:
                _log(logging.WARNING, "Adversarial pipeline failed: %s", exc)
                adv_result = None

        # --- Step 3: Build execution params ---
        params = self._build_execution_params(
            challenged_action, ctx, final_confidence, ensemble_result
        )

        return TradeDecision(
            action=challenged_action,
            symbol=ctx.get("symbol", "UNKNOWN"),
            confidence=final_confidence,
            reasoning=self._build_reasoning(ensemble_result, adv_result),
            params=params,
            adversarial_result=adv_result,
            ensemble_result=ensemble_result,
        )

    def execute(self, decision: TradeDecision) -> Dict[str, Any]:
        """
        Execute a TradeDecision via the TradeExecutor.

        Args:
            decision: TradeDecision from self.decide()

        Returns:
            Execution result dict
        """
        return self.executor.execute(decision)

    def decide_and_execute(self,
                            system: Optional[str] = None,
                            user: str = "",
                            market_context: Optional[Dict[str, Any]] = None,
                            providers: Optional[List[str]] = None,
                            temperature: float = 0.3,
                            run_adversarial: bool = True) -> Tuple[TradeDecision, Dict[str, Any]]:
        """
        Convenience method: call decide() then execute() in one shot.

        Returns:
            (TradeDecision, execution_result)
        """
        decision = self.decide(
            system=system,
            user=user,
            market_context=market_context,
            providers=providers,
            temperature=temperature,
            run_adversarial=run_adversarial,
        )
        result = self.execute(decision)
        return decision, result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_execution_params(
        self,
        action: str,
        ctx: Dict[str, Any],
        confidence: float,
        ensemble: EnsembleResult,
    ) -> Dict[str, Any]:
        """Build the execution params dict from market context and ensemble result."""

        # Base parameters
        leverage = int(ctx.get("leverage", 1))
        size_pct = float(ctx.get("position_size_pct", 10.0))
        sl_pct   = float(ctx.get("stop_loss_pct", 1.5))
        tp_pct   = float(ctx.get("take_profit_pct", 3.0))

        # Scale size by confidence
        conf_scale = min(max(confidence / 75.0, 0.3), 1.5)
        size_pct   = round(size_pct * conf_scale, 2)

        # Regime-aware adjustments
        regime = str(ctx.get("regime", "")).lower()
        if "trending" in regime:
            tp_pct = round(tp_pct * 1.5, 2)   # Let profits run
        elif "choppy" in regime or "volatile" in regime:
            leverage = min(leverage, 2)
            size_pct = round(size_pct * 0.7, 2)
            sl_pct   = round(sl_pct * 1.5, 2)

        # From ensemble votes
        if ensemble and ensemble.votes:
            raw_params = self._extract_params_from_votes(ensemble.votes)
            if raw_params:
                leverage  = raw_params.get("leverage", leverage)
                size_pct  = raw_params.get("position_size_pct", size_pct)
                sl_pct    = raw_params.get("stop_loss_pct", sl_pct)
                tp_pct    = raw_params.get("take_profit_pct", tp_pct)

        return {
            "leverage":           min(leverage, TradeExecutor.DEFAULT_MAX_LEVERAGE),
            "position_size_pct":  min(size_pct,  TradeExecutor.DEFAULT_MAX_POSITION_PCT),
            "stop_loss_pct":      round(sl_pct, 2),
            "take_profit_pct":    round(tp_pct, 2),
            "current_price":      ctx.get("current_price", 0),
            "symbol":             ctx.get("symbol", "UNKNOWN"),
            "account_info":       ctx.get("account_info", {}),
            "position_info":      ctx.get("position_info"),
            "market_snapshot":    ctx.get("market_snapshot"),
        }

    @staticmethod
    def _extract_params_from_votes(votes: List[VoteResult]) -> Dict[str, Any]:
        """Average numeric params across votes that include JSON."""
        params_list: List[Dict[str, Any]] = []
        for v in votes:
            m = re.search(r"```json\s*(\{.*?\})\s*```", v.raw_content, re.DOTALL)
            if not m:
                m = re.search(r"(\{.*?\})", v.raw_content, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    if "params" in obj:
                        params_list.append(obj["params"])
                except Exception:
                    pass

        if not params_list:
            return {}

        def avg_field(key: str, default: float) -> float:
            vals = [p.get(key, default) for p in params_list if isinstance(p.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else default

        return {
            "leverage":           int(avg_field("leverage", 1)),
            "position_size_pct":  round(avg_field("position_size_pct", 10.0), 2),
            "stop_loss_pct":      round(avg_field("stop_loss_pct", 1.5), 2),
            "take_profit_pct":    round(avg_field("take_profit_pct", 3.0), 2),
        }

    @staticmethod
    def _build_reasoning(ensemble: EnsembleResult,
                          adv: Optional[AdversarialResult]) -> str:
        parts = []
        if ensemble:
            parts.append(f"Ensemble: {ensemble.final_action} "
                          f"(conf={ensemble.confidence:.0f}%, "
                          f"consensus={ensemble.consensus_ratio:.0%})")
        if adv:
            parts.append(f"Adversarial: {adv.verdict} "
                          f"(conf={adv.final_confidence:.0f}%)")
        return " | ".join(parts)
