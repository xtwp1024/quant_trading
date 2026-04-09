"""
llm_sentiment.py — LLM-Driven Sentiment & Risk Signal Generator.

LLMSentimentFactor provides a lightweight bridge between free-text market data
(news headlines, filings, social media) and the DAPO reinforcement-learning
trading system.

Key capabilities
----------------
1. **Sentiment scoring** — translate a news headline / market commentary into a
   continuous sentiment score in ``[-1, +1]``, where ``-1`` = strong sell signal
   and ``+1`` = strong buy signal.

2. **Risk signal generation** — translate market context into a risk level in
   ``[0, 1]``, where ``0`` = minimum risk (very safe) and ``1`` = maximum risk
   (very risky).

3. **Batch processing** — score lists of texts in a single API round-trip to
   minimise latency and token usage.

4. **Graceful fallback** — if the LLM API is unavailable (network error,
   invalid key, rate-limit), the generator returns neutral default values
   rather than crashing, making it safe for production use.

Design goals
------------
- **Zero heavy SDK dependencies** — uses only ``requests`` (or the stdlib
  ``urllib``) for HTTP.  ``torch``, ``numpy``, and ``pandas`` are the only
  required machine-learning dependencies.

- **Bilingual docstrings** — every public symbol is documented in both
  English (primary) and Chinese (中文).

- **DeepSeek / ChatGLM / OpenAI-compatible** — works with any chat-completion
  API that follows the OpenAI request/response schema.  Model endpoint and
  API key are configurable at construction time.

Integration with DAPOAgent
---------------------------
The LLM signals produced by this module are typically injected into the
:class:`~quant_trading.rl.crypto_env.CryptoTradingEnv` as extra state dimensions,
or used to adjust rewards via the DAPO reward formula::

    r' = r * (S_f^alpha) / (R_f^beta + eps)

See :mod:`quant_trading.rl.llm_sentiment_rl` for the full end-to-end pipeline.

Example — single headline scoring
---------------------------------
>>> factor = LLMSentimentFactor(api_key=os.environ["DEEPSEEK_API_KEY"])
>>> sent = factor.generate_sentiment("Fed signals rate cuts amid slowing inflation")
>>> risk = factor.generate_risk_signal({"vix": 30.5, "spread": -0.25})
>>> print(f"Sentiment={sent:.2f}  Risk={risk:.2f}")

Example — batch scoring
-----------------------
>>> texts = [
...     "Apple reports record quarterly earnings",
...     "Recession fears mount as GDP contracts",
...     "ECB holds rates steady",
... ]
>>> scores = factor.batch_generate(texts)
>>> print(scores)  # e.g. [0.72, -0.88, 0.05]

Example — integrating with a DataFrame
--------------------------------------
>>> df["llm_sentiment"] = df["headline"].apply(
...     lambda h: factor.generate_sentiment(h)
... )
>>> df["llm_risk"] = df.apply(
...     lambda row: factor.generate_risk_signal(row.to_dict()), axis=1
... )
"""

from __future__ import annotations

import logging
import os
import time
import re
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["LLMSentimentFactor"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Discrete 1-5 sentiment label -> continuous [-1, +1] mapping
# 1 = strong sell, 2 = moderate sell, 3 = neutral, 4 = moderate buy, 5 = strong buy
_SENTIMENT_DISCRETE_MAP = {
    1: -1.0,
    2: -0.5,
    3: 0.0,
    4: 0.5,
    5: 1.0,
}

# Discrete 1-5 risk label -> continuous [0, 1] mapping
# 1 = very safe (0.0), 2 = moderately safe (0.25), 3 = neutral (0.5),
# 4 = moderately risky (0.75), 5 = very risky (1.0)
_RISK_DISCRETE_MAP = {
    1: 0.0,
    2: 0.25,
    3: 0.5,
    4: 0.75,
    5: 1.0,
}

# Prompt templates
_SENTIMENT_PROMPT_TEMPLATE = (
    "You are a quantitative finance analyst. "
    "Classify the following market-related text into one of five sentiment categories:\\n"
    "1 = strong sell signal\\n"
    "2 = moderate sell signal\\n"
    "3 = neutral / hold\\n"
    "4 = moderate buy signal\\n"
    "5 = strong buy signal\\n\\n"
    "Respond with ONLY the integer category (1-5). No explanation.\\n"
    "Text: {text}"
)

_RISK_PROMPT_TEMPLATE = (
    "You are a quantitative risk analyst. "
    "Classify the following market-related text into one of five risk categories:\\n"
    "1 = very safe / low risk\\n"
    "2 = moderately safe\\n"
    "3 = neutral risk\\n"
    "4 = moderately risky\\n"
    "5 = very risky / high risk\\n\\n"
    "Respond with ONLY the integer category (1-5). No explanation.\\n"
    "Text: {text}"
)

_MARKET_DATA_RISK_PROMPT_TEMPLATE = (
    "You are a quantitative risk analyst. Based on the following market indicators, "
    "provide a risk score from 1 (very safe) to 5 (very risky).\\n\\n"
    "Indicators:\\n{indicators}\\n\\n"
    "Respond with ONLY the integer category (1-5). No explanation."
)

# ---------------------------------------------------------------------------
# HTTP client helpers (no heavy SDK — use urllib or requests)
# ---------------------------------------------------------------------------

def _get_default_http_client():
    """
    Return a simple HTTP client for LLM API calls.

    Tries to use ``requests`` if available, otherwise falls back to stdlib
    ``urllib.request``.
    """
    try:
        import requests
        return _RequestsClient()
    except ImportError:
        return _UrllibClient()


class _RequestsClient:
    """Lightweight requests-based chat-completion client."""

    def __init__(self, timeout: float = 15.0):
        import requests
        self._requests = requests
        self._timeout = timeout

    def chat_completions_create(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 8,
    ) -> dict:
        import requests

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self._requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()


class _UrllibClient:
    """Stdlib urllib-based fallback when ``requests`` is not available."""

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout

    def chat_completions_create(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 8,
    ) -> dict:
        import json
        import urllib.request

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Score parsing helpers
# ---------------------------------------------------------------------------

def _parse_discrete_score(raw: str) -> int:
    """
    Parse a [1-5] integer score from LLM raw output.

    Handles common formatting issues:
    - Trailing punctuation (., !, etc.)
    - Extra whitespace
    - Numbers embedded in text
    """
    raw = raw.strip().rstrip(".").rstrip("!").strip()
    try:
        return max(1, min(5, int(raw)))
    except ValueError:
        digits = re.findall(r"\d", raw)
        if digits:
            return max(1, min(5, int(digits[0])))
        raise ValueError(f"Cannot parse discrete score from: {raw!r}")


# ---------------------------------------------------------------------------
# LLMSentimentFactor
# ---------------------------------------------------------------------------

class LLMSentimentFactor:
    """
    LLM-driven sentiment / risk signal generator.

    使用 LLM 将文本（新闻标题、市场评论）转换为交易信号。
    支持 DeepSeek、ChatGLM、OpenAI 等兼容 ChatCompletion API 的大模型。

    Parameters
    ----------
    model_name : str, default "deepseek-chat"
        模型名称（API 路径中的模型标识符）。
        Examples: ``"deepseek-chat"``, ``"glm-4-flash"``, ``"gpt-4o-mini"``。
    api_key : str, optional
        API 密钥。如未传入，则尝试读取环境变量
        ``DEEPSEEK_API_KEY``（兼容） 或 ``OPENAI_API_KEY``。
    base_url : str, optional
        API base URL。如未传入，则默认使用 DeepSeek API
        ``https://api.deepseek.com``。
    timeout : float, default 15.0
        单次 API 调用超时时间（秒）。
    rate_limit_delay : float, default 0.1
        连续调用之间的最小延迟（秒），用于防止触发 429 限流。
    cache_size : int, default 10_000
        LRU 缓存条目上限。相同的文本不会重复调用 API。

    Attributes
    ----------
    default_sentiment : float
        API 不可用时返回的默认情感分数（0.0，即中性）。
    default_risk : float
        API 不可用时返回的默认风险等级（0.5，即中等风险）。

    Example
    -------
    >>> factor = LLMSentimentFactor(
    ...     model_name="deepseek-chat",
    ...     api_key=os.environ["DEEPSEEK_API_KEY"],
    ... )
    >>> sentiment = factor.generate_sentiment(
    ...     "Fed signals potential rate cuts amid cooling inflation"
    ... )
    >>> risk = factor.generate_risk_signal(
    ...     {"vix": 28.4, "credit_spread": -0.15, "yield_curve": -0.5}
    ... )
    >>> print(f"Sentiment={sentiment:.2f}, Risk={risk:.2f}")
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: str = "",
        base_url: Optional[str] = None,
        timeout: float = 15.0,
        rate_limit_delay: float = 0.1,
        cache_size: int = 10_000,
    ):
        self.model_name = model_name
        self._api_key = api_key or os.environ.get(
            "DEEPSEEK_API_KEY",
            os.environ.get("OPENAI_API_KEY", ""),
        )
        self._base_url = base_url or "https://api.deepseek.com"
        self._timeout = timeout
        self._delay = rate_limit_delay
        self._cache_size = cache_size

        self.default_sentiment = 0.0  # neutral
        self.default_risk = 0.5        # medium risk

        self._cache: dict[str, tuple[float, float]] = {}
        self._http_client: Optional[_RequestsClient | _UrllibClient] = None
        self._last_call_time = 0.0

    @property
    def _client(self):
        if self._http_client is None:
            self._http_client = _get_default_http_client()
        return self._http_client

    def _cache_key(self, text: str) -> str:
        """Deterministic cache key: first 128 chars + total length."""
        return f"{text[:128]}::{len(text)}"

    # ------------------------------------------------------------------
    # Core scoring methods
    # ------------------------------------------------------------------

    def generate_sentiment(self, news_text: str) -> float:
        """
        将文本转换为情感分数。

        Parameters
        ----------
        news_text : str
            市场相关的文本内容（新闻标题、评论、社交媒体帖子等）。
            建议控制在 2000 字符以内以节省 Token 成本。

        Returns
        -------
        float
            连续情感分数，范围 ``[-1.0, +1.0]``。
            - ``-1.0`` = 强烈卖出信号
            - `` 0.0`` = 中性 / 持有
            - ``+1.0`` = 强烈买入信号

        Notes
        -----
        API 不可用时返回中性分数 ``0.0``，不会抛出异常。

        Example
        -------
        >>> factor = LLMSentimentFactor(api_key="sk-...")
        >>> factor.generate_sentiment("Apple beats earnings expectations")
        0.5  # moderate buy
        """
        key = self._cache_key(news_text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0]

        score = self._call_llm_sentiment(news_text)
        self._cache[key] = (score, self.default_risk)

        if len(self._cache) > self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        return score

    def generate_risk_signal(self, market_data: dict) -> float:
        """
        根据市场数据字典生成风险等级。

        Parameters
        ----------
        market_data : dict
            市场指标字典。例如::

                {
                    "vix": 28.5,           # VIX 波动率指数
                    "credit_spread": -0.3,  # 信用利差
                    "yield_curve": -0.5,   # 收益率曲线斜率
                    "high_beta": 0.75,     # 高贝塔股票相对表现
                    ...                    # 其他自定义指标
                }

            只要能转为字符串提示词即可，支持任意键名。

        Returns
        -------
        float
            连续风险等级，范围 ``[0.0, 1.0]``。
            - ``0.0`` = 极低风险（非常安全）
            - ``0.5`` = 中等风险
            - ``1.0`` = 极高风险（非常危险）

        Notes
        -----
        API 不可用时返回中等风险分数 ``0.5``，不会抛出异常。

        Example
        -------
        >>> factor = LLMSentimentFactor(api_key="sk-...")
        >>> factor.generate_risk_signal({"vix": 35.0, "spread": -0.5})
        0.75  # moderately high risk
        """
        # Format market data into a readable string
        lines = [f"- {k}: {v}" for k, v in market_data.items()]
        indicators_str = "\n".join(lines)

        prompt = _MARKET_DATA_RISK_PROMPT_TEMPLATE.format(indicators=indicators_str)
        score = self._call_llm_risk(prompt)
        return score

    def batch_generate(self, texts: list[str]) -> list[float]:
        """
        批量将多个文本转换为情感分数。

        使用顺序 API 调用（每次调用之间遵守 rate_limit_delay）。
        如果需要真正的批量处理，建议使用提供 Batch API 的服务提供商。

        Parameters
        ----------
        texts : list[str]
            待处理的文本列表。

        Returns
        -------
        list[float]
            与输入等长的情感分数列表，范围 ``[-1.0, +1.0]``。
            任何 API 失败的元素将返回默认中性分数 ``0.0``。

        Example
        -------
        >>> factor = LLMSentimentFactor(api_key="sk-...")
        >>> scores = factor.batch_generate([
        ...     "Fed signals rate cuts",
        ...     "Recession fears grow",
        ...     "Markets hold steady",
        ... ])
        >>> print(scores)  # e.g. [0.5, -0.88, 0.0]
        """
        results = []
        for text in texts:
            try:
                results.append(self.generate_sentiment(text))
            except Exception as exc:
                logger.warning(f"batch_generate: scoring failed for text "
                               f"{text[:50]!r}: {exc}. Returning neutral.")
                results.append(self.default_sentiment)
            time.sleep(self._delay)
        return results

    # ------------------------------------------------------------------
    # Internal LLM API calls
    # ------------------------------------------------------------------

    def _call_llm_sentiment(self, text: str) -> float:
        """Call LLM for a discrete [1-5] sentiment score, return continuous."""
        prompt = _SENTIMENT_PROMPT_TEMPLATE.format(text=text[:2000])
        try:
            self._throttle()
            response = self._client.chat_completions_create(
                base_url=self._base_url,
                api_key=self._api_key,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            raw = response["choices"][0]["message"]["content"].strip()
            discrete = _parse_discrete_score(raw)
            return _SENTIMENT_DISCRETE_MAP[discrete]
        except Exception as exc:
            logger.warning(
                f"LLMSentimentFactor: sentiment API call failed ({exc}). "
                f"Returning neutral score (0.0)."
            )
            return self.default_sentiment

    def _call_llm_risk(self, prompt: str) -> float:
        """Call LLM for a discrete [1-5] risk score, return continuous [0, 1]."""
        try:
            self._throttle()
            response = self._client.chat_completions_create(
                base_url=self._base_url,
                api_key=self._api_key,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            raw = response["choices"][0]["message"]["content"].strip()
            discrete = _parse_discrete_score(raw)
            return _RISK_DISCRETE_MAP[discrete]
        except Exception as exc:
            logger.warning(
                f"LLMSentimentFactor: risk API call failed ({exc}). "
                f"Returning medium risk (0.5)."
            )
            return self.default_risk

    def _throttle(self) -> None:
        """Enforce minimum delay between API calls to avoid 429 errors."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call_time = time.time()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """清除内存中的 LLM 响应缓存。"""
        self._cache.clear()

    def __len__(self) -> int:
        """返回当前缓存条目数量。"""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_env(model_name: str = "deepseek-chat") -> "LLMSentimentFactor":
        """
        从环境变量创建 LLMSentimentFactor 实例（便捷工厂方法）。

        Parameters
        ----------
        model_name : str
            模型名称。

        Returns
        -------
        LLMSentimentFactor
            新实例，API 密钥从环境变量读取。

        Example
        -------
        >>> factor = LLMSentimentFactor.from_env("glm-4-flash")
        >>> factor.generate_sentiment("China PMI beats expectations")
        0.5
        """
        return LLMSentimentFactor(model_name=model_name)


# ---------------------------------------------------------------------------
# DataFrame enrichment helpers
# ---------------------------------------------------------------------------

def enrich_dataframe_with_sentiment(
    df: pd.DataFrame,
    text_column: str,
    api_key: str = "",
    model_name: str = "deepseek-chat",
    sentiment_col: str = "llm_sentiment",
    risk_col: Optional[str] = None,
    skip_existing: bool = True,
    delay: float = 0.1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enrich a DataFrame with LLM sentiment scores.

    在 DataFrame 中批量添加 LLM 情感分数列。
    该函数会原地修改 df 并返回（copy-on-write 保护）。

    Parameters
    ----------
    df : pd.DataFrame
        输入 DataFrame。
    text_column : str
        包含待分析文本的列名（如 ``"headline"``）。
    api_key : str, optional
        API 密钥。
    model_name : str
        LLM 模型名（默认 ``"deepseek-chat"``）。
    sentiment_col : str
        输出情感分数列名（默认 ``"llm_sentiment"``）。
    risk_col : str, optional
        如提供，则同时生成风险信号列。
    skip_existing : bool
        如为 True（默认），已非空的行跳过不处理，
        支持中断后恢复。
    delay : float
        API 调用间隔（秒），默认 0.1。
    verbose : bool
        是否每 50 行打印一次进度。

    Returns
    -------
    pd.DataFrame
        添加了 LLM 信号列的 DataFrame。

    Example
    -------
    >>> df = pd.read_csv("news_data.csv")
    >>> df = enrich_dataframe_with_sentiment(df, text_column="headline")
    >>> df[["date", "headline", "llm_sentiment"]].head()
    """
    factor = LLMSentimentFactor(model_name=model_name, api_key=api_key, rate_limit_delay=delay)
    df = df.copy()

    if sentiment_col not in df.columns:
        df[sentiment_col] = 0.0

    mask = df[text_column].notna()
    if skip_existing:
        mask = mask & (df[sentiment_col] == 0.0)

    total = mask.sum()
    if total == 0:
        logger.info("enrich_dataframe_with_sentiment: no rows to process.")
        return df

    if verbose:
        logger.info(f"Processing {total} rows for sentiment scoring...")

    for idx, (_, row) in enumerate(df[mask].iterrows()):
        if verbose and (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{total} rows...")
        try:
            score = factor.generate_sentiment(str(row[text_column])[:2000])
            df.at[idx, sentiment_col] = score
        except Exception as exc:
            logger.warning(f"Row {idx} scoring failed: {exc}")
        time.sleep(delay)

    if verbose:
        logger.info(f"Done. {total} rows scored.")
    return df
