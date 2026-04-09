# quant_trading/sentiment/finbert_sentiment.py
"""FinBERT sentiment analysis for financial news.

将金融文本映射到 [-1, +1] 情绪分数，支持 FinBERT 模型
或 VADER fallback。

Classes:
    FinBERTSentiment: FinBERT金融情绪分析
    VADERFallback: VADER情绪分析 fallback
    SentimentSignalGenerator: 情绪信号生成器 (FinBERT + RF)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# FinBERTSentiment
# --------------------------------------------------------------------------- #


class FinBERTSentiment:
    """FinBERT金融情绪分析 — 用于新闻文本.

    将金融文本映射到 [-1, +1] 情绪分数.
    使用FinBERT预训练模型 (ProsusAI/finbert) 或 fallback 到 VADER.

    Attributes:
        model_name: HuggingFace 模型名称或本地路径
        use_softmax: 是否对 logits 做 softmax (默认 True)
        fallback: fallback 策略，'vader' 或 'textblob'

    Example:
        >>> sentiment = FinBERTSentiment()
        >>> result = sentiment.analyze("Bitcoin surges to new all-time high")
        >>> print(result)
        {'sentiment': 'positive', 'score': 0.82, 'confidence': 0.95}
    """

    LABEL_MAP: dict[str, int] = {
        "positive": 1,
        "negative": -1,
        "neutral": 0,
    }

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        use_softmax: bool = True,
        fallback: str = "vader",
    ) -> None:
        self.model_name = model_name
        self.use_softmax = use_softmax
        self.fallback = fallback

        self._pipe = None
        self._tokenizer = None
        self._vader: Optional[VADERFallback] = None

        self._init_finbert()

    # ------------------------------------------------------------------ #
    # Private init
    # ------------------------------------------------------------------ #

    def _init_finbert(self) -> None:
        """Initialize FinBERT pipeline."""
        try:
            from transformers import AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=tokenizer,
                framework="pt",
                top_k=None,
            )
            self._tokenizer = tokenizer
            logger.info("FinBERT model '%s' loaded successfully.", self.model_name)
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load FinBERT model '{self.model_name}': {exc}. "
                f"Falling back to {self.fallback}.",
                RuntimeWarning,
            )
            self._pipe = None
            self._tokenizer = None
            self._init_fallback()

    def _init_fallback(self) -> None:
        """Initialize VADER fallback."""
        if self.fallback == "vader":
            self._vader = VADERFallback()
        else:
            self._vader = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, text: str) -> dict[str, Any]:
        """分析单条文本.

        Args:
            text: 输入金融文本

        Returns:
            dict with keys:
                - sentiment (str): 'positive' | 'negative' | 'neutral'
                - score (float): 映射到 [-1, +1] 的情绪分数
                - confidence (float): 置信度 [0, 1]
        """
        if self._pipe is not None:
            return self._analyze_finbert(text)
        elif self._vader is not None:
            return self._analyze_vader(text)
        else:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

    def batch_analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        """批量分析多条文本.

        Args:
            texts: 输入文本列表

        Returns:
            每条文本对应的分析结果列表
        """
        return [self.analyze(t) for t in texts]

    def news_to_signal(
        self,
        headlines: list[str],
        weights: Optional[list[float]] = None,
    ) -> float:
        """将多条新闻转换为单一交易信号 [-1, +1].

        Args:
            headlines: 新闻标题/摘要列表
            weights: 每条新闻的权重 (默认为等权重)

        Returns:
            合成信号值，范围 [-1.0, +1.0]

        Note:
            - score > 0 表示净买入信号
            - score < 0 表示净卖出信号
            - 0 表示中性
        """
        if not headlines:
            return 0.0

        results = self.batch_analyze(headlines)

        if weights is None:
            weights = [1.0] * len(results)
        else:
            weights = list(weights)

        if len(weights) != len(results):
            raise ValueError(
                f"Length mismatch: {len(weights)} weights vs {len(results)} headlines"
            )

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_score = sum(r["score"] * w for r, w in zip(results, weights))
        return weighted_score / total_weight

    # ------------------------------------------------------------------ #
    # Private internals
    # ------------------------------------------------------------------ #

    def _analyze_finbert(self, text: str) -> dict[str, Any]:
        """使用 FinBERT 分析文本."""
        try:
            # Truncate to 512 tokens
            encoded = self._tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            truncated_text = self._tokenizer.decode(
                encoded["input_ids"][0],
                skip_special_tokens=True,
            )

            raw = self._pipe(truncated_text)
            # pipe returns list of dicts like [{'label': 'positive', 'score': 0.x}, ...]
            if isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], list):
                    raw = raw[0]  # some models return [[{...}, ...]]

            # Sort by label
            label_scores: dict[str, float] = {}
            for item in raw:
                label_scores[item["label"].lower()] = item["score"]

            sentiment = self._best_label(label_scores)
            raw_score = label_scores.get(sentiment, 0.0)

            # Map to [-1, +1]
            if sentiment == "positive":
                score = raw_score
            elif sentiment == "negative":
                score = -raw_score
            else:
                # neutral mapped to 0
                score = 0.0

            confidence = raw_score

            return {
                "sentiment": sentiment,
                "score": float(score),
                "confidence": float(confidence),
            }

        except Exception as exc:
            logger.warning("FinBERT analysis failed: %s. Falling back to VADER.", exc)
            return self._analyze_vader(text)

    def _best_label(self, label_scores: dict[str, float]) -> str:
        """Return the label with the highest score."""
        return max(label_scores, key=label_scores.get)  # type: ignore[arg-type]

    def _analyze_vader(self, text: str) -> dict[str, Any]:
        """使用 VADER 分析文本 (fallback)."""
        if self._vader is None:
            self._vader = VADERFallback()

        vader_result = self._vader.polarity_scores(text)
        compound = vader_result["compound"]

        # Map compound [-1, +1] to sentiment label
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": float(compound),
            "confidence": abs(float(compound)),
        }


# --------------------------------------------------------------------------- #
# VADERFallback
# --------------------------------------------------------------------------- #


class VADERFallback:
    """VADER情绪分析 (无FinBERT时的fallback).

    使用 NLTK VADER lexicon 对文本进行情绪分析。
    无需外部模型，依赖 NLTK vader_lexicon。

    Example:
        >>> vader = VADERFallback()
        >>> scores = vader.polarity_scores("Bitcoin price is surging!")
        >>> print(scores)
        {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.5}
    """

    def __init__(self) -> None:
        self._sia = None
        self._init_vader()

    def _init_vader(self) -> None:
        """Initialize VADER SentimentIntensityAnalyzer."""
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            self._sia = SentimentIntensityAnalyzer()
        except ImportError:  # pragma: no cover
            warnings.warn(
                "NLTK not installed. Run: pip install nltk",
                ImportWarning,
            )
            self._sia = None

    def polarity_scores(self, text: str) -> dict[str, float]:
        """计算文本的情绪极性分数.

        Args:
            text: 输入文本

        Returns:
            dict with keys:
                - neg: 负面情绪比例 [0, 1]
                - neu: 中性情绪比例 [0, 1]
                - pos: 正面情绪比例 [0, 1]
                - compound: 综合情绪分数 [-1, +1]

        Note:
            compound 是归一化后的综合分数，直接反映情绪倾向。
        """
        if self._sia is None:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        return self._sia.polarity_scores(text)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# SentimentSignalGenerator
# --------------------------------------------------------------------------- #


class SentimentSignalGenerator:
    """情绪信号生成器 — FinBERT + RF预测.

    Combines FinBERT sentiment analysis with a Random Forest regressor
    to generate trading signals based on:
    1. News sentiment scores (FinBERT)
    2. RF-predicted 15-day moving average returns

    Attributes:
        sentiment_model: FinBERTSentiment 实例
        rf_predictor: sklearn RandomForestRegressor (optional)

    Example:
        >>> generator = SentimentSignalGenerator(FinBERTSentiment())
        >>> signal = generator.generate_signal(
        ...     news_texts=["BTC up 5% on ETF approval news"],
        ...     current_position_days=3,
        ... )
        >>> print(signal)
        {'signal': 0.65, 'confidence': 0.78, 'news_count': 1}
    """

    def __init__(
        self,
        sentiment_model: FinBERTSentiment,
        rf_predictor: Optional[Any] = None,
    ) -> None:
        self.sentiment_model = sentiment_model
        self.rf_predictor = rf_predictor

    def generate_signal(
        self,
        news_texts: list[str],
        current_position_days: int = 0,
    ) -> dict[str, Any]:
        """生成综合情绪信号.

        Args:
            news_texts: 当前周期的新闻文本列表
            current_position_days: 当前持仓已持续天数 (用于仓位衰减)

        Returns:
            dict with keys:
                - signal (float): 综合交易信号 [-1, +1]
                - confidence (float): 信号置信度 [0, 1]
                - news_count (int): 分析的新闻数量
                - sentiment_score (float): 原始情绪分数
                - rf_pred (float | None): RF预测收益 (若有)
        """
        news_count = len(news_texts)

        if news_count == 0:
            return {
                "signal": 0.0,
                "confidence": 0.0,
                "news_count": 0,
                "sentiment_score": 0.0,
                "rf_pred": None,
            }

        # FinBERT sentiment signal
        sentiment_score = self.sentiment_model.news_to_signal(news_texts)

        # Average confidence from individual analyses
        results = self.sentiment_model.batch_analyze(news_texts)
        avg_confidence = float(
            np.mean([r["confidence"] for r in results]) if results else 0.0
        )

        # RF prediction if available
        rf_pred: Optional[float] = None
        if self.rf_predictor is not None:
            try:
                # RF takes feature vector; use sentiment + position features
                features = self._build_rf_features(sentiment_score, current_position_days)
                rf_pred = float(self.rf_predictor.predict([features])[0])
            except Exception as exc:
                logger.warning("RF prediction failed: %s", exc)
                rf_pred = None

        # Combine sentiment score and RF prediction
        signal = self._combine_signals(sentiment_score, rf_pred)
        confidence = self._compute_confidence(avg_confidence, rf_pred)

        return {
            "signal": float(signal),
            "confidence": float(confidence),
            "news_count": news_count,
            "sentiment_score": float(sentiment_score),
            "rf_pred": rf_pred,
        }

    # ------------------------------------------------------------------ #
    # Private internals
    # ------------------------------------------------------------------ #

    def _build_rf_features(
        self,
        sentiment_score: float,
        position_days: int,
    ) -> list[float]:
        """构建RF模型输入特征向量.

        Feature order (must match training order):
            [sentiment_score, position_days, sentiment_momentum, ...]
        """
        # Simple feature set; extend as needed
        return [
            sentiment_score,
            float(position_days),
            abs(sentiment_score),  # sentiment magnitude
            1.0 if sentiment_score > 0 else -1.0 if sentiment_score < 0 else 0.0,
        ]

    def _combine_signals(
        self,
        sentiment_score: float,
        rf_pred: Optional[float],
    ) -> float:
        """Combine FinBERT sentiment and RF prediction into a single signal."""
        if rf_pred is None:
            return sentiment_score

        # Normalize rf_pred (typically ~[-0.1, +0.1]) to [-1, +1]
        rf_normalized = max(-1.0, min(1.0, rf_pred * 10))

        # Weighted combination (sentiment 40%, RF 60%)
        combined = 0.4 * sentiment_score + 0.6 * rf_normalized
        return max(-1.0, min(1.0, combined))

    def _compute_confidence(
        self,
        sentiment_confidence: float,
        rf_pred: Optional[float],
    ) -> float:
        """Compute overall confidence."""
        if rf_pred is None:
            return sentiment_confidence

        # Boost confidence if both sentiment and RF agree
        return min(1.0, sentiment_confidence + 0.1)
