"""
Multi-LLM Bridge — 多LLM厂商桥接器
===================================

统一接口调用8+模型厂商 (DeepSeek / OpenAI / Anthropic / Qwen / Gemini / Kimi / MiniMax / Zhipu)。

Architecture:
  - REST-only HTTP client using urllib (no heavy SDK dependencies)
  - Provider-specific adapters for different API auth schemes
  - Concurrent batch generation support
  - Weighted ensemble generation

Usage:
  >>> bridge = MultiLLMBridge({'openai': 'sk-xxx', 'deepseek': 'sk-xxx'})
  >>> response = bridge.generate('BTC会涨吗?', model='deepseek')
  >>> ensemble_resp = bridge.ensemble_generate('分析市场趋势', models=['deepseek', 'openai', 'claude'])
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "MultiLLMBridge",
    "LLMResponse",
    "SUPPORTED_MODELS",
    # Re-export ModelEnsemble for convenience
    "ModelEnsemble",
]


# ---------------------------------------------------------------------------
# Supported providers and their base URLs
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "kimi": "https://api.moonshot.cn/v1",
    "minimax": "https://api.minimax.chat/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
}

# Default model identifiers per provider
DEFAULT_MODEL_NAMES: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
    "qwen": "qwen-plus",
    "gemini": "gemini-1.5-flash",
    "kimi": "kimi-plus",
    "minimax": "minimax-01-ai-agent",
    "zhipu": "glm-4-flash",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """LLM response with metadata."""
    model: str
    content: str
    confidence: float  # 0.0–1.0 (extracted from response or heuristic)
    latency_ms: float
    cost: float
    raw_response: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    def __post_init__(self):
        # Normalize confidence to [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))


# ---------------------------------------------------------------------------
# Platt calibration helpers (mirrors multi_llm_ensemble.py)
# ---------------------------------------------------------------------------

def _logit(p: float) -> float:
    """Safe logit transform."""
    p = max(0.01, min(0.99, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def platt_calibrate(raw_prob: float, a: float = 0.90, b: float = 0.0) -> float:
    """Apply Platt-like shrinkage toward 0.5.

    Args:
        raw_prob: Raw confidence probability (0.01–0.99)
        a: Shrinkage factor (0.9 = 10% shrinkage toward 0.5)
        b: Intercept offset
    """
    p = max(0.01, min(0.99, raw_prob))
    logit_p = _logit(p)
    shrunk = logit_p * a + b
    return max(0.01, min(0.99, _sigmoid(shrunk)))


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

class BaseAdapter:
    """Base adapter for LLM providers."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def build_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def build_url(self, base_url: str) -> str:
        return f"{base_url}/chat/completions"

    def build_body(self, messages: list[dict], **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def parse_response(self, data: dict) -> tuple[str, float, dict]:
        """Returns (content, confidence, usage_dict)."""
        raise NotImplementedError

    def estimate_cost(self, usage: dict) -> float:
        """Estimate cost in USD from usage dict. Override per-provider."""
        return 0.0


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI-compatible APIs (OpenAI, DeepSeek, Qwen, Kimi, MiniMax, Zhipu)."""

    def build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def build_body(self, messages: list[dict], **kwargs) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

    def parse_response(self, data: dict) -> tuple[str, float, dict]:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        usage = data.get("usage", {})
        # Heuristic confidence from finish_reason
        confidence = 0.8 if choice.get("finish_reason") == "stop" else 0.5
        return content, confidence, usage

    def estimate_cost(self, usage: dict) -> float:
        # Rough estimate for OpenAI GPT-4o
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        return pt * 0.00001 + ct * 0.00003


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude."""

    ANTHROPIC_VERSION = "2023-06-01"

    def build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def build_url(self, base_url: str) -> str:
        return f"{base_url}/messages"

    def build_body(self, messages: list[dict], **kwargs) -> dict[str, Any]:
        # Extract system message
        system_content = ""
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})

        body = {
            "model": self.model,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system_content:
            body["system"] = system_content
        temperature = kwargs.get("temperature", 0.7)
        if temperature > 0:
            body["temperature"] = max(0.1, temperature)
        return body

    def parse_response(self, data: dict) -> tuple[str, float, dict]:
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content = block.get("text", "")
                break
        usage = data.get("usage", {})
        confidence = 0.8 if content else 0.5
        return content, confidence, usage

    def estimate_cost(self, usage: dict) -> float:
        # Rough estimate for Claude 3.5 Sonnet
        pt = usage.get("input_tokens", 0)
        ct = usage.get("output_tokens", 0)
        return pt * 0.000003 + ct * 0.000015


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini."""

    def build_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def build_url(self, base_url: str) -> str:
        return f"{base_url}/models/{self.model}:generateContent?key={self.api_key}"

    def build_body(self, messages: list[dict], **kwargs) -> dict[str, Any]:
        contents = []
        system_instruction = None
        for msg in messages:
            if msg.get("role") == "system":
                system_instruction = {"parts": [{"text": msg.get("content", "")}]}
            else:
                role = "model" if msg.get("role") == "assistant" else msg.get("role")
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.get("content", "")}],
                })
        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 4096),
            },
        }
        if system_instruction:
            body["systemInstruction"] = system_instruction
        return body

    def parse_response(self, data: dict) -> tuple[str, float, dict]:
        content = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                content = parts[0].get("text", "")
        usage_metadata = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }
        confidence = 0.8 if content else 0.5
        return content, confidence, usage

    def estimate_cost(self, usage: dict) -> float:
        # Rough estimate for Gemini Flash
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        return pt * 0.000000125 + ct * 0.0000005


_ADAPTERS: dict[str, type[BaseAdapter]] = {
    "openai": OpenAIAdapter,
    "deepseek": OpenAIAdapter,
    "qwen": OpenAIAdapter,
    "kimi": OpenAIAdapter,
    "minimax": OpenAIAdapter,
    "zhipu": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
}


# ---------------------------------------------------------------------------
# HTTP request helper
# ---------------------------------------------------------------------------

def _http_post(url: str, headers: dict[str, str], body: dict[str, Any], timeout: int = 120) -> dict:
    """Execute HTTP POST with urllib, return parsed JSON."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code}: {body_text}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e.reason}") from e


# ---------------------------------------------------------------------------
# Main bridge class
# ---------------------------------------------------------------------------

class MultiLLMBridge:
    """多LLM厂商桥接器 — 统一接口调用8+模型厂商.

    使用方式:
        bridge = MultiLLMBridge({'openai': 'sk-xxx', 'deepseek': 'sk-xxx'})
        response = bridge.generate('BTC会涨吗?', model='deepseek')

    Attributes:
        clients: Dict of adapter per provider (built lazily on first use)
        api_keys: API keys per provider
        default_model: Default provider key
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        default_model: str = "deepseek",
        timeout: int = 120,
    ):
        self.api_keys: dict[str, str] = api_keys or {}
        self.default_model = default_model
        self.timeout = timeout
        self._adapters: dict[str, BaseAdapter] = {}

    def add_model(self, model: str, api_key: str) -> None:
        """添加模型配置 / Register a model with its API key."""
        self.api_keys[model] = api_key

    def _get_adapter(self, model: str) -> BaseAdapter:
        """Get or create adapter for the given model/provider."""
        if model in self._adapters:
            return self._adapters[model]

        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model: '{model}'. Supported: {list(SUPPORTED_MODELS.keys())}"
            )

        api_key = self.api_keys.get(model)
        if not api_key:
            raise ValueError(f"No API key configured for model: '{model}'")

        model_name = DEFAULT_MODEL_NAMES.get(model, "default")
        adapter_cls = _ADAPTERS.get(model, OpenAIAdapter)
        adapter = adapter_cls(api_key=api_key, model=model_name)
        self._adapters[model] = adapter
        return adapter

    def _messages_to_list(self, messages: list[tuple[str, str]]) -> list[dict[str, str]]:
        """Convert (role, content) tuples to API message format."""
        return [{"role": role, "content": content} for role, content in messages]

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> LLMResponse:
        """调用指定模型 (默认 self.default_model).

        Args:
            prompt: User prompt text
            model: Provider key (e.g. 'deepseek', 'openai'). Defaults to default_model.
            system_prompt: System prompt string.
            **kwargs: Passed to adapter (temperature, max_tokens).

        Returns:
            LLMResponse with content, confidence, latency_ms, cost.
        """
        model = model or self.default_model
        adapter = self._get_adapter(model)
        base_url = SUPPORTED_MODELS.get(model, "")

        messages = [("system", system_prompt), ("user", prompt)]
        body = adapter.build_body(self._messages_to_list(messages), **kwargs)
        url = adapter.build_url(base_url)
        headers = adapter.build_headers()

        start = time.perf_counter()
        try:
            data = _http_post(url, headers, body, timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"LLM request failed for {model}: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        content, confidence, usage = adapter.parse_response(data)
        cost = adapter.estimate_cost(usage)

        # Apply Platt calibration to confidence
        calibrated_confidence = platt_calibrate(confidence)

        return LLMResponse(
            model=model,
            content=content,
            confidence=calibrated_confidence,
            latency_ms=latency_ms,
            cost=cost,
            raw_response=data,
            usage=usage,
        )

    def batch_generate(
        self,
        prompts: list[str],
        model: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> list[LLMResponse]:
        """批量生成 / Batch generate for multiple prompts sequentially.

        Note: For true concurrency, use ensemble_generate() instead.

        Args:
            prompts: List of user prompts.
            model: Provider key. Defaults to default_model.
            system_prompt: System prompt.
            **kwargs: Passed to each generate() call.

        Returns:
            List of LLMResponse, one per prompt.
        """
        return [
            self.generate(p, model=model, system_prompt=system_prompt, **kwargs)
            for p in prompts
        ]

    def ensemble_generate(
        self,
        prompt: str,
        models: list[str] | None = None,
        weights: dict[str, float] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> LLMResponse:
        """多模型并行生成 + 加权集成 / Multi-model parallel generate + weighted ensemble.

        Runs all models concurrently using threads, then aggregates
        responses by weighted average of confidence scores.

        Args:
            prompt: User prompt sent to all models.
            models: List of provider keys. Defaults to all configured models.
            weights: Provider -> weight dict for aggregation.
                    If None, equal weights are used.
            system_prompt: System prompt.
            **kwargs: Passed to each generate() call.

        Returns:
            LLMResponse with aggregated content (best response) and
            weighted average confidence.
        """
        import concurrent.futures

        models = models or list(self.api_keys.keys())
        weights = weights or {m: 1.0 for m in models}

        # Normalize weights
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {m: w / total_w for m, w in weights.items()}

        def call_model(m: str) -> LLMResponse:
            return self.generate(prompt, model=m, system_prompt=system_prompt, **kwargs)

        responses: list[LLMResponse] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {executor.submit(call_model, m): m for m in models}
            for future in concurrent.futures.as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception as e:
                    model_key = futures[future]
                    responses.append(LLMResponse(
                        model=model_key,
                        content="",
                        confidence=0.0,
                        latency_ms=0.0,
                        cost=0.0,
                        raw_response={"error": str(e)},
                    ))

        # Aggregate: pick highest-confidence content, weighted avg confidence
        best_response = max(responses, key=lambda r: r.confidence)
        weighted_confidence = sum(
            r.confidence * weights.get(r.model, 0.0) for r in responses
        )
        avg_latency = sum(r.latency_ms for r in responses) / len(responses)
        total_cost = sum(r.cost for r in responses)

        return LLMResponse(
            model=",".join(models),
            content=best_response.content,
            confidence=weighted_confidence,
            latency_ms=avg_latency,
            cost=total_cost,
            raw_response={
                "ensemble": True,
                "responses": [
                    {"model": r.model, "confidence": r.confidence, "content": r.content}
                    for r in responses
                ],
            },
        )


# ---------------------------------------------------------------------------
# Re-export ModelEnsemble for single-module convenience import
# ---------------------------------------------------------------------------

def _lazy_import_model_ensemble():
    """Lazy import to avoid circular dependency at module load time."""
    from quant_trading.agent.model_ensemble import ModelEnsemble as ME
    return ME


# Make ModelEnsemble available directly from this module for the
# convenience import: from quant_trading.agent.multi_llm_bridge import ModelEnsemble
import sys
if sys.version_info >= (3, 10):
    pass  # lazy import pattern used below
else:
    # For Python < 3.10, use __getattr__ for lazy imports
    pass


def __getattr__(name: str):
    if name == "ModelEnsemble":
        return _lazy_import_model_ensemble()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
