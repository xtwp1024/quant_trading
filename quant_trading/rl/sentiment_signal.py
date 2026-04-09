"""
sentiment_signal.py — Standalone sentiment signal generation for RL-augmented trading.

Adapted from FinRL-DAPO-SR (D:/Hive/Data/trading_repos/FinRL-DAPO-SR/).

Purpose
-------
Provides DeepSeek API integration for generating sentiment and risk signals
that can be used to augment the RL state space.

This module differs from `dapo_llm_features.py` in that it focuses purely on
the LLM API call interface and is suitable for live / real-time inference,
whereas `dapo_llm_features` is oriented toward batch pre-processing and
on-policy reward adjustment during training.

Key functions
-------------
- :func:`get_sentiment_score`   — integer score [1-5] from text via DeepSeek
- :func:`get_risk_score`        — integer score [1-5] from text via DeepSeek
- :func:`batch_sentiment`       — process a series of texts (with caching)
- :class:`SentimentCache`       — simple dict-based cache to avoid redundant API calls

API key
-------
Set ``DEEPSEEK_API_KEY`` in the environment or pass ``api_key`` directly.
If no key is found, all functions return neutral stub values and log a warning.

References
----------
- FinRL-DAPO-SR: D:/Hive/Data/trading_repos/FinRL-DAPO-SR/
- train_dapo_llm_risk.py  — training pipeline that uses pre-computed CSV signals
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEUTRAL_SENTIMENT = 3
NEUTRAL_RISK = 3

_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_TIMEOUT = 15.0  # seconds

# Score semantics (consistent with FinRL-DAPO-SR)
SENTIMENT_LABELS = {
    1: "strong_sell",
    2: "moderate_sell",
    3: "neutral",
    4: "moderate_buy",
    5: "strong_buy",
}

RISK_LABELS = {
    1: "very_safe",
    2: "moderately_safe",
    3: "neutral_risk",
    4: "moderately_risky",
    5: "very_risky",
}


# ---------------------------------------------------------------------------
# DeepSeek client helper
# ---------------------------------------------------------------------------

def _get_client(api_key: Optional[str] = None) -> "OpenAI":  # lazy import
    """Return an authenticated DeepSeek / OpenAI-compatible client."""
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")

    if api_key is None:
        raise RuntimeError(
            "DeepSeek API key not found. Set DEEPSEEK_API_KEY in your environment "
            "or pass api_key=<your_key>."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package required for DeepSeek API calls. "
            "Install with: pip install openai"
        ) from exc

    return OpenAI(
        api_key=api_key,
        base_url=_DEEPSEEK_BASE_URL,
        timeout=_DEEPSEEK_TIMEOUT,
    )


def _make_prompt(text: str, task: str = "sentiment") -> str:
    """Build a zero-shot classification prompt for the DeepSeek model."""
    if task == "sentiment":
        return (
            "You are a quantitative finance analyst. "
            "Classify the following market-related text into one of five sentiment categories:\n"
            "1 = strong sell signal\n"
            "2 = moderate sell signal\n"
            "3 = neutral / hold\n"
            "4 = moderate buy signal\n"
            "5 = strong buy signal\n\n"
            "Respond with ONLY the integer category (1-5). No explanation. "
            f"Text: {text[:2000]}"
        )
    elif task == "risk":
        return (
            "You are a quantitative risk analyst. "
            "Classify the following market-related text into one of five risk categories:\n"
            "1 = very safe / low risk\n"
            "2 = moderately safe\n"
            "3 = neutral risk\n"
            "4 = moderately risky\n"
            "5 = very risky / high risk\n\n"
            "Respond with ONLY the integer category (1-5). No explanation. "
            f"Text: {text[:2000]}"
        )
    else:
        raise ValueError(f"Unknown task: {task!r}")


def _parse_score(raw: str) -> int:
    """Parse and validate a [1-5] integer score from LLM output."""
    raw = raw.strip()
    # Strip any trailing punctuation
    raw = raw.rstrip(".").rstrip("!")
    try:
        score = int(raw)
    except ValueError:
        # Try to extract first digit
        import re
        digits = re.findall(r"\d", raw)
        if digits:
            score = int(digits[0])
        else:
            raise ValueError(f"Cannot parse score from: {raw!r}")

    return max(1, min(5, score))


# ---------------------------------------------------------------------------
# Core API functions
# ---------------------------------------------------------------------------


def get_sentiment_score(
    text: str,
    api_key: Optional[str] = None,
    model: str = _DEEPSEEK_DEFAULT_MODEL,
    cache: Optional["SentimentCache"] = None,
) -> int:
    """
    Return an integer sentiment score [1-5] for the given text.

    Parameters
    ----------
    text : str
        Market-related text (news headline, filing, transcript, etc.).
    api_key : str, optional
        DeepSeek API key.
    model : str
        Model identifier for the DeepSeek endpoint.
    cache : SentimentCache, optional
        Cache instance to avoid redundant API calls for identical texts.

    Returns
    -------
    int
        Sentiment score in [1, 5].

    Example
    -------
    >>> score = get_sentiment_score(
    ...     "Fed holds rates steady; markets await inflation data",
    ... )
    >>> label = SENTIMENT_LABELS[score]
    >>> print(label)
    'neutral'
    """
    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid text input. Returning neutral (3).")
        return NEUTRAL_SENTIMENT

    # Check cache first
    if cache is not None:
        cached = cache.get(text)
        if cached is not None:
            return cached

    # Check API key availability
    try:
        client = _get_client(api_key)
    except RuntimeError as exc:
        logger.warning(f"{exc}. Returning neutral sentiment (3).")
        return NEUTRAL_SENTIMENT

    prompt = _make_prompt(text, task="sentiment")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        raw = response.choices[0].message.content.strip()
        score = _parse_score(raw)
    except Exception as exc:
        logger.warning(f"DeepSeek API call failed: {exc}. Returning neutral (3).")
        score = NEUTRAL_SENTIMENT

    if cache is not None:
        cache.set(text, score)

    logger.debug(f"Sentiment={score} | text={text[:60]!r}")
    return score


def get_risk_score(
    text: str,
    api_key: Optional[str] = None,
    model: str = _DEEPSEEK_DEFAULT_MODEL,
    cache: Optional["SentimentCache"] = None,
) -> int:
    """
    Return an integer risk score [1-5] for the given text.

    Parameters
    ----------
    text : str
        Market-related text to assess risk for.
    api_key : str, optional
        DeepSeek API key.
    model : str
        Model identifier for the DeepSeek endpoint.
    cache : SentimentCache, optional
        Cache to avoid redundant API calls.

    Returns
    -------
    int
        Risk score in [1, 5].

    Example
    -------
    >>> score = get_risk_score(
    ...     "Credit spreads widen sharply; corporate debt under pressure",
    ... )
    >>> print(RISK_LABELS[score])
    'very_risky'
    """
    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid text input. Returning neutral risk (3).")
        return NEUTRAL_RISK

    if cache is not None:
        cached = cache.get(text, key="risk")
        if cached is not None:
            return cached

    try:
        client = _get_client(api_key)
    except RuntimeError as exc:
        logger.warning(f"{exc}. Returning neutral risk (3).")
        return NEUTRAL_RISK

    prompt = _make_prompt(text, task="risk")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        raw = response.choices[0].message.content.strip()
        score = _parse_score(raw)
    except Exception as exc:
        logger.warning(f"DeepSeek API call failed: {exc}. Returning neutral (3).")
        score = NEUTRAL_RISK

    if cache is not None:
        cache.set(text, score, key="risk")

    logger.debug(f"Risk={score} | text={text[:60]!r}")
    return score


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------


def batch_sentiment(
    texts: list[str],
    api_key: Optional[str] = None,
    model: str = _DEEPSEEK_DEFAULT_MODEL,
    delay: float = 0.1,
    cache: Optional["SentimentCache"] = None,
) -> np.ndarray:
    """
    Process a list of texts and return an array of sentiment scores [1-5].

    Parameters
    ----------
    texts : list[str]
        List of market texts (one per asset/date row).
    api_key : str, optional
        DeepSeek API key.
    model : str
        DeepSeek model name.
    delay : float
        Seconds to sleep between API calls to avoid 429 errors.
    cache : SentimentCache, optional
        Shared cache across the batch.

    Returns
    -------
    np.ndarray (len(texts),)
        Sentiment scores in [1, 5].

    Example
    -------
    >>> headlines = [
    ...     "Fed cuts rates by 50bps",
    ...     "Unemployment rises unexpectedly",
    ...     "Markets trade mixed ahead of earnings",
    ... ]
    >>> scores = batch_sentiment(headlines)
    >>> print(scores)  # e.g. [5 1 3]
    """
    scores = np.full(len(texts), NEUTRAL_SENTIMENT, dtype=np.int32)

    for i, text in enumerate(texts):
        try:
            scores[i] = get_sentiment_score(text, api_key=api_key, model=model, cache=cache)
        except Exception as exc:
            logger.warning(f"batch_sentiment[{i}] failed: {exc}")
            scores[i] = NEUTRAL_SENTIMENT

        if i < len(texts) - 1 and delay > 0:
            time.sleep(delay)

    return scores


def batch_risk(
    texts: list[str],
    api_key: Optional[str] = None,
    model: str = _DEEPSEEK_DEFAULT_MODEL,
    delay: float = 0.1,
    cache: Optional["SentimentCache"] = None,
) -> np.ndarray:
    """
    Process a list of texts and return an array of risk scores [1-5].

    Parameters
    ----------
    texts : list[str]
        List of market texts.
    api_key : str, optional
        DeepSeek API key.
    model : str
        DeepSeek model name.
    delay : float
        Seconds to sleep between API calls.
    cache : SentimentCache, optional
        Shared cache across the batch.

    Returns
    -------
    np.ndarray (len(texts),)
        Risk scores in [1, 5].
    """
    scores = np.full(len(texts), NEUTRAL_RISK, dtype=np.int32)

    for i, text in enumerate(texts):
        try:
            scores[i] = get_risk_score(text, api_key=api_key, model=model, cache=cache)
        except Exception as exc:
            logger.warning(f"batch_risk[{i}] failed: {exc}")
            scores[i] = NEUTRAL_RISK

        if i < len(texts) - 1 and delay > 0:
            time.sleep(delay)

    return scores


# ---------------------------------------------------------------------------
# Simple cache
# ---------------------------------------------------------------------------


class SentimentCache:
    """
    In-memory cache for sentiment/risk API results.

    Uses MD5 of the input text as key.  Thread-unsafe — use a
    multiprocessing-safe cache (e.g. Redis) for multi-process training.

    Example
    -------
    >>> cache = SentimentCache(max_size=10000)
    >>> score = get_sentiment_score(news_text, cache=cache)
    >>> # same text again hits the cache
    """

    def __init__(self, max_size: int = 10_000):
        self._sentiment: dict[str, int] = {}
        self._risk: dict[str, int] = {}
        self._max_size = max_size

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str, key: str = "sentiment") -> Optional[int]:
        h = self._hash(text)
        if key == "risk":
            return self._risk.get(h)
        return self._sentiment.get(h)

    def set(self, text: str, score: int, key: str = "sentiment") -> None:
        if len(self._sentiment) >= self._max_size and key == "sentiment":
            # Simple FIFO eviction
            oldest = next(iter(self._sentiment))
            del self._sentiment[oldest]
        if len(self._risk) >= self._max_size and key == "risk":
            oldest = next(iter(self._risk))
            del self._risk[oldest]

        h = self._hash(text)
        if key == "risk":
            self._risk[h] = score
        else:
            self._sentiment[h] = score

    def clear(self) -> None:
        self._sentiment.clear()
        self._risk.clear()

    def __len__(self) -> int:
        return len(self._sentiment) + len(self._risk)
