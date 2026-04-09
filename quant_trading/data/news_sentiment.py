"""Simple news sentiment scoring for crypto exchange announcements.

This module provides a lightweight rule-based + lexicon sentiment scorer
for news headlines. It is designed to be used as a feature input for
quantitative trading strategies.

Features:
- Rule-based sentiment scoring (no external API required)
- Configurable bullish / bearish / neutral keyword lexicons
- Per-exchange and aggregate sentiment metrics
- Works with CryptoNewsItem from crypto_news.py

Note: For production use, consider upgrading to an LLM-based scorer
(e.g. quant_trading.core.sentiment) or a dedicated NLP service.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .crypto_news import CryptoNewsItem


# ---------------------------------------------------------------------------
# Default sentiment lexicons (case-insensitive)
# ---------------------------------------------------------------------------

DEFAULT_BULLISH_WORDS: List[str] = [
    # Listings & listings expansion
    "listing", "new listing", "will list", "going live", "launch", "launching",
    "上线", "上市", "上线交易",
    # Partnerships & adoption
    "partnership", "collaboration", "adopted", "integration", "integrated",
    "strategic", "join hands",
    # Rewards & incentives
    "reward", "airdrop", "bonus", "incentive", "giveaway", "promotion",
    # Trading & volume
    "trading", "volume", "liquidity", "deep order book",
    # Upgrades
    "upgrade", "upgraded", "update", "improved", "enhancement",
    # Positive market
    "surge", "rally", "soar", "jump", "gain", "high", "record",
    # Support & governance
    "support", "approved", "approval", "pass", "elected",
    # Financial
    "investment", "invest", "funding", "raised", "grant",
    # Mining & staking
    "staking", "mine", "yield",
]

DEFAULT_BEARISH_WORDS: List[str] = [
    # Delistings
    "delist", "delisting", "delisted", "remove", "removed", "suspend",
    "suspended", "termination", "terminate", "下线",
    # Warnings & restrictions
    "warning", "caution", "risk", "alert", "investor alert",
    "restricted", "restriction", "ban", "banned",
    # Negative market
    "crash", "plunge", "drop", "fall", "decline", "loss", "lose",
    "low", "bearish",
    # Regulatory
    "investigation", "probe", "investigate", "lawsuit", "sue",
    "regulatory", "compliance", "illegal", "fraud",
    # Security
    "hack", "exploit", "breach", "stolen", "attack", "vulnerability",
    # System issues
    "outage", "downtime", "disrupt", "failure", "malfunction", "maintenance",
    # Negative financial
    "bankruptcy", "insolvent", "default", "debt",
    # Negative governance
    "recall", "revoke", "cancel", "cancelled",
]

DEFAULT_NEUTRAL_WORDS: List[str] = [
    "update", "schedule", "announcement", "notice", "info",
    "release", "report", "statement", "press",
    "maintenance", "system", "service", "platform",
    "version", "patch", "fix",
]


# ---------------------------------------------------------------------------
# Score result
# ---------------------------------------------------------------------------


@dataclass
class SentimentScore:
    """Sentiment score for a single news item."""

    news_id: str
    exchange: str
    title: str
    raw_score: float  # continuous score in [-1, 1]
    label: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1, based on keyword match density
    flagged_keywords: List[str] = field(default_factory=list)

    @property
    def is_bullish(self) -> bool:
        return self.label == "bullish"

    @property
    def is_bearish(self) -> bool:
        return self.label == "bearish"

    @property
    def is_neutral(self) -> bool:
        return self.label == "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "news_id": self.news_id,
            "exchange": self.exchange,
            "title": self.title,
            "raw_score": self.raw_score,
            "label": self.label,
            "confidence": self.confidence,
            "flagged_keywords": self.flagged_keywords,
        }


@dataclass
class AggregateSentiment:
    """Aggregate sentiment metrics across a list of news items."""

    exchange: Optional[str]
    total: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_score: float
    bullish_ratio: float  # bullish / total
    bearish_ratio: float  # bearish / total
    confidence_avg: float
    timestamp: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exchange": self.exchange,
            "total": self.total,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "avg_score": self.avg_score,
            "bullish_ratio": self.bullish_ratio,
            "bearish_ratio": self.bearish_ratio,
            "confidence_avg": self.confidence_avg,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class NewsSentimentScorer:
    """Rule-based sentiment scorer for crypto news headlines.

    Uses a configurable keyword lexicon to classify news as bullish,
    bearish, or neutral. Provides both per-item scores and aggregate
    statistics per exchange.

    Args:
        bullish_words: List of bullish keywords/phrases.
        bearish_words: List of bearish keywords/phrases.
        neutral_words: List of neutral keywords/phrases.
        score_threshold: Score above this threshold -> bullish, below -> bearish.
        ignore_words: Words to strip before scoring (e.g. exchange names).
    """

    def __init__(
        self,
        bullish_words: Optional[List[str]] = None,
        bearish_words: Optional[List[str]] = None,
        neutral_words: Optional[List[str]] = None,
        score_threshold: float = 0.05,
        ignore_words: Optional[List[str]] = None,
    ):
        self.bullish_words = set(w.lower() for w in (bullish_words or DEFAULT_BULLISH_WORDS))
        self.bearish_words = set(w.lower() for w in (bearish_words or DEFAULT_BEARISH_WORDS))
        self.neutral_words = set(w.lower() for w in (neutral_words or DEFAULT_NEUTRAL_WORDS))
        self.score_threshold = score_threshold
        self.ignore_words = set(w.lower() for w in (ignore_words or []))

    def _clean_text(self, text: str) -> str:
        """Lowercase and strip noise characters."""
        text = text.lower()
        for w in self.ignore_words:
            text = re.sub(rf"\b{re.escape(w)}\b", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _find_keywords(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """Return (bullish_hits, bearish_hits, neutral_hits) from text."""
        bullish_hits = [w for w in self.bullish_words if w in text]
        bearish_hits = [w for w in self.bearish_words if w in text]
        neutral_hits = [w for w in self.neutral_words if w in text]
        return bullish_hits, bearish_hits, neutral_hits

    def _compute_raw_score(
        self,
        bullish_hits: List[str],
        bearish_hits: List[str],
        neutral_hits: List[str],
    ) -> Tuple[float, float]:
        """Compute raw score in [-1, 1] and confidence in [0, 1]."""
        total_keywords = len(bullish_hits) + len(bearish_hits) + len(neutral_hits)
        if total_keywords == 0:
            return 0.0, 0.0

        bullish_score = len(bullish_hits) / total_keywords
        bearish_score = len(bearish_hits) / total_keywords

        raw = bullish_score - bearish_score  # in [-1, 1]

        # Confidence is the fraction of content words that were keyword matches
        # (clamped to avoid division by zero; empty text -> 0 confidence)
        confidence = min(total_keywords / max(len(self.bullish_words) + len(self.bearish_words), 1), 1.0)

        return raw, confidence

    def score(self, item: CryptoNewsItem) -> SentimentScore:
        """Score a single news item."""
        text = self._clean_text(item.title + " " + item.desc)
        bullish_hits, bearish_hits, neutral_hits = self._find_keywords(text)
        raw_score, confidence = self._compute_raw_score(bullish_hits, bearish_hits, neutral_hits)

        all_keywords = bullish_hits + bearish_hits + neutral_hits

        if raw_score > self.score_threshold:
            label = "bullish"
        elif raw_score < -self.score_threshold:
            label = "bearish"
        else:
            label = "neutral"

        return SentimentScore(
            news_id=item.news_id,
            exchange=item.exchange,
            title=item.title,
            raw_score=round(raw_score, 4),
            label=label,
            confidence=round(confidence, 4),
            flagged_keywords=all_keywords,
        )

    def score_many(self, items: List[CryptoNewsItem]) -> List[SentimentScore]:
        """Score a list of news items."""
        return [self.score(item) for item in items]

    def aggregate(self, scores: List[SentimentScore], exchange: Optional[str] = None) -> AggregateSentiment:
        """Compute aggregate sentiment across a list of scores.

        Args:
            scores: List of SentimentScore objects.
            exchange: If provided, only aggregate for this exchange.
        """
        if exchange:
            scores = [s for s in scores if s.exchange == exchange.lower()]

        if not scores:
            return AggregateSentiment(
                exchange=exchange,
                total=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                avg_score=0.0,
                bullish_ratio=0.0,
                bearish_ratio=0.0,
                confidence_avg=0.0,
            )

        bullish = [s for s in scores if s.is_bullish]
        bearish = [s for s in scores if s.is_bearish]
        neutral = [s for s in scores if s.is_neutral]

        return AggregateSentiment(
            exchange=exchange,
            total=len(scores),
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            neutral_count=len(neutral),
            avg_score=round(sum(s.raw_score for s in scores) / len(scores), 4),
            bullish_ratio=round(len(bullish) / len(scores), 4),
            bearish_ratio=round(len(bearish) / len(scores), 4),
            confidence_avg=round(sum(s.confidence for s in scores) / len(scores), 4),
        )

    def aggregate_by_exchange(self, scores: List[SentimentScore]) -> Dict[str, AggregateSentiment]:
        """Compute aggregate sentiment grouped by exchange."""
        by_exchange: Dict[str, List[SentimentScore]] = defaultdict(list)
        for s in scores:
            by_exchange[s.exchange].append(s)
        return {ex: self.aggregate(scores, exchange=ex) for ex, scores in by_exchange.items()}

    def enrich_items(self, items: List[CryptoNewsItem]) -> List[CryptoNewsItem]:
        """Return a copy of items with sentiment_score field populated."""
        scored = self.score_many(items)
        result = []
        for item, score in zip(items, scored):
            d = item.to_dict()
            d["sentiment_score"] = score.raw_score
            result.append(CryptoNewsItem.from_dict(d))
        return result


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "NewsSentimentScorer",
    "SentimentScore",
    "AggregateSentiment",
    "DEFAULT_BULLISH_WORDS",
    "DEFAULT_BEARISH_WORDS",
    "DEFAULT_NEUTRAL_WORDS",
]
