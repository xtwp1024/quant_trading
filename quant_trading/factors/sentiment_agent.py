"""
SentimentAgent — news-based sentiment analysis.

OpenAI GPT path (when OPENAI_API_KEY is set):
    1. Fetch headlines from a news provider
    2. Call GPT to score each headline -1..+1
    3. Recency-weighted aggregation → ticker sentiment score

Rule-based fallback (no API key):
    - Keyword scoring on headline / description strings
    - Keywords: "beat", "miss", "upgrade", "downgrade", "lawsuit", "recall", etc.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

__all__ = ["SentimentAgent"]

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Keyword dictionaries for rule-based fallback
# ----------------------------------------------------------------------

POSITIVE_KEYWORDS = [
    "beat", "blowout", "exceed", "exceeds", "top", "outperform",
    "upgrade", "upgraded", "strong", "growth", "profit", "profitable",
    "surge", "soar", "rally", "gain", "up", "high", "record high",
    "buy", "bullish", "positive", "recovery", "expansion", "beat estimates",
    "revenue growth", "earnings growth", "dividend increase", "share buyback",
]

NEGATIVE_KEYWORDS = [
    "miss", "missed", "below", "worse", "weak", "loss", "decline",
    "downgrade", "downgraded", "cut", "sell", "bearish", "negative",
    "warning", "investigation", "lawsuit", "fraud", "recall", "scandal",
    "bankruptcy", "layoffs", "cut", "profit warning", "earnings miss",
    "revenue miss", "guidance cut", "downward", "drop", "fall", "plunge",
]

NEUTRAL_KEYWORDS = [
    "hold", "maintain", "in line", "mixed", "unchanged", "stable",
]


def _keyword_score(text: str) -> float:
    """Simple keyword-based sentiment score in [-1, 1]."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    text_lower = text.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    neu = sum(1 for kw in NEUTRAL_KEYWORDS if kw in text_lower)
    total = pos + neg + neu
    if total == 0:
        return 0.0
    return (pos - neg) / (total + 1e-9)


# ----------------------------------------------------------------------
# SentimentAgent
# ----------------------------------------------------------------------

class SentimentAgent:
    """
    Sentiment analysis for financial news.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. If None, falls back to rule-based scoring.
    news_fetcher : callable, optional
        Function (ticker: str, days: int) -> List[Dict] of news items with
        at least keys: ``title``, ``description``, ``published_utc``.
        If None, a built-in keyword scorer is used on synthetic headlines.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        news_fetcher: Optional[callable] = None,
        model: str = "gpt-4o-mini",
        cache_ttl_hours: float = 6.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.news_fetcher = news_fetcher
        self.model = model
        self.cache_ttl_hours = cache_ttl_hours

        self._sentiment: Dict[str, float] = {}
        self._history: Dict[str, Dict[str, float]] = {}
        self._last_update: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, ticker: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Analyze sentiment for a single ticker.

        Returns
        -------
        dict with keys:
            ticker, sentiment_score (float -1..+1),
            sentiment_label ("positive"/"neutral"/"negative"),
            count (number of news items), last_updated (ISO str)
        """
        now = datetime.now()
        if (
            not force_update
            and ticker in self._last_update
            and (now - self._last_update[ticker]).total_seconds()
            < self.cache_ttl_hours * 3600
        ):
            return {
                "ticker": ticker,
                "sentiment_score": self._sentiment.get(ticker, 0.0),
                "sentiment_label": _score_to_label(self._sentiment.get(ticker, 0.0)),
                "count": len(self._history.get(ticker, {})),
                "last_updated": self._last_update.get(ticker, now).isoformat(),
            }

        if self.news_fetcher is None:
            news_items = self._default_news(ticker)
        else:
            try:
                news_items = self.news_fetcher(ticker, days_back=7)
            except Exception as exc:
                logger.warning("News fetch failed for %s: %s", ticker, exc)
                news_items = []

        if not news_items:
            return self._neutral_result(ticker, now, error="no news available")

        analyzed = []
        for item in news_items:
            score = self._score_item(item)
            analyzed.append({**item, "sentiment_score": score})

        # Recency-weighted aggregate
        aggregate = self._aggregate(analyzed)

        self._sentiment[ticker] = aggregate
        self._last_update[ticker] = now
        date_key = now.strftime("%Y-%m-%d")
        self._history.setdefault(ticker, {})[date_key] = aggregate

        return {
            "ticker": ticker,
            "sentiment_score": round(aggregate, 4),
            "sentiment_label": _score_to_label(aggregate),
            "count": len(analyzed),
            "last_updated": now.isoformat(),
        }

    def analyze_batch(self, tickers: List[str]) -> Dict[str, Any]:
        """Analyze sentiment for multiple tickers."""
        return {ticker: self.analyze(ticker) for ticker in tickers}

    def get_history(
        self, ticker: str, days: int = 30
    ) -> Dict[str, float]:
        """Return historical sentiment scores for a ticker."""
        history = self._history.get(ticker, {})
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return {k: v for k, v in history.items() if k >= cutoff}

    def get_market_sentiment(
        self, tickers: Optional[List[str]] = None
    ) -> float:
        """
        Aggregate sentiment across a basket of tickers.
        Default basket: major US indices constituents.
        """
        if tickers is None:
            tickers = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL"]
        scores = [self._sentiment[t] for t in tickers if t in self._sentiment]
        return sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_item(self, item: Dict[str, Any]) -> float:
        """Score a single news item — GPT if available, else keywords."""
        title = item.get("title", "")
        description = item.get("description", "")

        if self.api_key:
            try:
                return self._gpt_score(title, description)
            except Exception as exc:
                logger.warning("GPT scoring failed, falling back: %s", exc)
        return _keyword_score(title + " " + description)

    def _gpt_score(self, title: str, description: str) -> float:
        """Call OpenAI to score a news item."""
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            prompt = (
                "You are a financial sentiment analyst. Score the headline on a scale from "
                "-1.0 (extremely bearish/negative for stock price) to +1.0 (extremely bullish/positive). "
                "Focus on implications for short-term stock movement.\n\n"
                f"Headline: {title}\n"
                f"Description: {description}\n\n"
                "Return ONLY a number between -1.0 and 1.0 with at most 3 decimal places."
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise financial sentiment scorer. Output only a single number.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=8,
            )
            text = response.choices[0].message.content.strip()
            # Extract first float
            match = re.search(r"-?[\d.]+", text)
            if match:
                score = float(match.group())
                return max(-1.0, min(1.0, score))
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
        return 0.0

    def _aggregate(self, items: List[Dict[str, Any]]) -> float:
        """
        Recency-weighted aggregation of scored news items.
        Most recent item gets weight 1.0, each older item gets weight 0.5^n.
        """
        if not items:
            return 0.0
        weighted_sum = 0.0
        weight_total = 0.0
        for i, item in enumerate(reversed(items)):
            recency_weight = 0.5**i
            score = item.get("sentiment_score", 0.0)
            weighted_sum += score * recency_weight
            weight_total += recency_weight
        return weighted_sum / weight_total if weight_total else 0.0

    def _default_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Stub: return empty when no news fetcher is configured."""
        logger.debug("No news fetcher configured for ticker %s", ticker)
        return []

    def _neutral_result(
        self, ticker: str, now: datetime, error: str = ""
    ) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "count": 0,
            "last_updated": now.isoformat(),
            "error": error,
        }


def _score_to_label(score: float) -> str:
    if score > 0.15:
        return "positive"
    elif score < -0.15:
        return "negative"
    return "neutral"
