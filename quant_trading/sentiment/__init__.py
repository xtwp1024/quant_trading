# quant_trading/sentiment/__init__.py
"""Sentiment analysis module for CryptoSentimentBertRfStrat.

This module provides:
- FinBERTSentiment: FinBERT-based financial news sentiment analysis
- VADERFallback: VADER sentiment fallback when FinBERT unavailable
- SentimentSignalGenerator: Combines FinBERT with RF predictions
- MemoryState: Working memory state for tracking sentiment history
- MemoryFeatureExtractor: Extracts time-series features from memory
- RedditLingoScorer: Reddit slang sentiment dictionary scorer
- RedditSentimentAnalyzer: Reddit scraping + slang + VADER + BERT/FAISS pipeline
- CryptoSentimentAnalyzer: CryptoSentimentBertRfStrat top-level orchestrator
- FinBERTWrapper: FinBERT / HF Inference API / keyword sentiment wrapper
- MemoryAugmentedFeatures: Rolling-window memory features (10/30-day)
- SentimentRandomForest: sklearn RandomForest strategy layer
- RedditBERTEncoder: Reddit text → BERT / TF-IDF embeddings
- FAISSVectorStore: FAISS vector similarity store (sklearn fallback)
- SentimentStockPredictor: Social sentiment → stock direction predictor
- RedditDataCollector: Reddit posts/comments collector (praw / requests fallback)
- generate_signals: End-to-end Reddit social sentiment → trading signal
- CongressDataCollector: Congressional disclosure scraper (pure urllib)
- CongressTradeAnalyzer: Congressional trading pattern analyzer
- CongressSentimentScorer: Sentiment scorer from congress buy/sell ratios
- PoliticianTrack: Individual politician trading history tracker
- get_congress_sentiment: Main API — returns sentiment + confidence

Exports:
    FinBERTSentiment, VADERFallback, SentimentSignalGenerator,
    MemoryState, MemoryFeatureExtractor,
    RedditLingoScorer, REDDIT_LINGO, RedditSentimentAnalyzer,
    CryptoSentimentAnalyzer, FinBERTWrapper,
    MemoryAugmentedFeatures, SentimentRandomForest,
    RedditBERTEncoder, FAISSVectorStore,
    SentimentStockPredictor, RedditDataCollector, generate_signals,
    CongressDataCollector, CongressTradeAnalyzer,
    CongressSentimentScorer, PoliticianTrack, get_congress_sentiment
"""

from quant_trading.sentiment.finbert_sentiment import (
    FinBERTSentiment,
    VADERFallback,
    SentimentSignalGenerator,
)

from quant_trading.sentiment.memory_features import (
    MemoryState,
    MemoryFeatureExtractor,
)

from quant_trading.sentiment.reddit_dict import (
    RedditLingoScorer,
    REDDIT_LINGO,
)

from quant_trading.sentiment.reddit_sentiment import (
    RedditSentimentAnalyzer,
)

from quant_trading.sentiment.crypto_sentiment_bert import (
    CryptoSentimentAnalyzer,
    FinBERTWrapper,
    MemoryAugmentedFeatures,
    SentimentRandomForest,
)

from quant_trading.sentiment.stock_reddit_bert import (
    RedditBERTEncoder,
    FAISSVectorStore,
    SentimentStockPredictor,
    RedditDataCollector,
    generate_signals,
)

from quant_trading.sentiment.congress_sentiment import (
    CongressDataCollector,
    CongressSentimentScorer,
    SenateTradeScraper,
    HouseTradeScraper,
    get_congress_sentiment,
)

__all__ = [
    # FinBERT
    "FinBERTSentiment",
    "VADERFallback",
    "SentimentSignalGenerator",
    # Memory
    "MemoryState",
    "MemoryFeatureExtractor",
    # Reddit
    "RedditLingoScorer",
    "REDDIT_LINGO",
    "RedditSentimentAnalyzer",
    # CryptoSentimentBertRfStrat
    "CryptoSentimentAnalyzer",
    "FinBERTWrapper",
    "MemoryAugmentedFeatures",
    "SentimentRandomForest",
    # stock_reddit_bert
    "RedditBERTEncoder",
    "FAISSVectorStore",
    "SentimentStockPredictor",
    "RedditDataCollector",
    "generate_signals",
    # Congress sentiment
    "CongressDataCollector",
    "CongressSentimentScorer",
    "SenateTradeScraper",
    "HouseTradeScraper",
    "get_congress_sentiment",
]
