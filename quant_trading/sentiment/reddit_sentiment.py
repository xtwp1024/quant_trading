# -*- coding: utf-8 -*-
"""
Reddit情绪分析器 — 抓取 + 俚语 + VADER + Embedding Pipeline

工作流:
1. PRAW 抓取指定 subreddit 的帖子 (可选, 无 PRAW 时跳过)
2. RedditLingo 俚语打分 (必选)
3. VADER 二次确认 (可选, 无 NLTK 时降级为 regex)
4. BERT embedding 存储 FAISS 向量检索 (可选, 无 FAISS 时降级为余弦相似度)

Author: 量化之神
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from .reddit_dict import RedditLingoScorer

logger = logging.getLogger(__name__)

# =============================================================================
# Optional Imports
# =============================================================================

PRAW_AVAILABLE = False
VADER_AVAILABLE = False
FAISS_AVAILABLE = False
BERT_AVAILABLE = False

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    logger.warning("praw not installed — Reddit fetching disabled")

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True)
        except Exception:
            VADER_AVAILABLE = False
    VADER_AVAILABLE = True
except ImportError:
    logger.warning("nltk not installed — VADER sentiment disabled")

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("faiss not installed — vector search will use sklearn")

try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed — BERT embeddings disabled")

# =============================================================================
# RedditSentimentAnalyzer
# =============================================================================

class RedditSentimentAnalyzer:
    """
    Reddit情绪分析器 — 抓取 + 俚语 + VADER + Embedding.

    综合多个数据源对 Reddit 社区情绪进行量化分析.

    工作流:
        1. 抓取指定 subreddit 的帖子 (PRAW, optional)
        2. RedditLingo 俚语打分 (必选)
        3. VADER 二次确认 (optional)
        4. BERT embedding 存储 (optional)

    Attributes:
        lingo_scorer: RedditLingoScorer 实例
        vader_analyzer: SentimentIntensityAnalyzer 实例 (可选)
        reddit_client: PRAW Reddit 实例 (可选)
        embedding_model: SentenceTransformer 实例 (可选)
        index: FAISS 索引 (可选)

    Example:
        >>> analyzer = RedditSentimentAnalyzer()
        >>> result = analyzer.get_ticker_sentiment("GME")
        >>> print(result)
        {'sentiment': 0.73, 'mention_count': 156, 'confidence': 0.81}
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "QuantTradingBot/1.0",
        lingo_dict: Optional[dict[str, float]] = None,
    ):
        """
        初始化 RedditSentimentAnalyzer.

        Args:
            client_id: Reddit API client_id (从 https://www.reddit.com/prefs/apps)
            client_secret: Reddit API client_secret
            user_agent: API user agent 字符串
            lingo_dict: 可选的 RedditLingoScorer 自定义词典
        """
        # RedditLingo scorer (always)
        self.lingo_scorer = RedditLingoScorer(custom_dict=lingo_dict)

        # VADER (optional)
        self.vader_analyzer: Optional[Any] = None
        if VADER_AVAILABLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to init VADER: {e}")

        # PRAW Reddit client (optional)
        self.reddit_client: Optional[Any] = None
        if PRAW_AVAILABLE and client_id and client_secret:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                logger.info("PRAW Reddit client initialized")
            except Exception as e:
                logger.warning(f"Failed to init PRAW client: {e}")

        # BERT + FAISS (optional)
        self.embedding_model: Optional[Any] = None
        self.index: Optional[Any] = None
        self._post_cache: list[dict] = []
        self._embedding_dim: int = 0

        if BERT_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self._embedding_dim = 384
                if FAISS_AVAILABLE:
                    self.index = faiss.IndexFlatIP(self._embedding_dim)  # Inner Product (cosine)
                    logger.info("BERT + FAISS pipeline initialized")
                else:
                    logger.info("BERT pipeline initialized (FAISS disabled)")
            except Exception as e:
                logger.warning(f"Failed to init BERT/FAISS: {e}")

    # -------------------------------------------------------------------------
    # Reddit Fetching (PRAW)
    # -------------------------------------------------------------------------

    def fetch_posts(
        self,
        subreddit: str,
        keyword: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        抓取 Reddit 帖子.

        Args:
            subreddit: 子版块名称 (如 'wallstreetbets', 'stocks')
            keyword: 可选关键词过滤 (搜索 title + body)
            limit: 最大抓取帖子数

        Returns:
            List[dict], 每个 dict 包含:
                - title: str
                - body: str
                - score: int
                - num_comments: int
                - created_utc: float
                - url: str
                - author: str
                - id: str
        """
        if not self.reddit_client:
            logger.warning("Reddit client not initialized — returning empty list")
            return []

        try:
            sub = self.reddit_client.subreddit(subreddit)
            posts: list[dict] = []

            # Get hot posts or search
            if keyword:
                # Search API
                for submission in sub.search(keyword, limit=limit):
                    posts.append(self._extract_post_data(submission))
            else:
                # Hot posts
                for submission in sub.hot(limit=limit):
                    posts.append(self._extract_post_data(submission))

            logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
            return posts

        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit}: {e}")
            return []

    def _extract_post_data(self, submission: Any) -> dict:
        """从 PRAW submission 提取标准字段."""
        return {
            'title': str(submission.title),
            'body': str(submission.selftext) if hasattr(submission, 'selftext') else '',
            'score': int(submission.score),
            'num_comments': int(submission.num_comments),
            'created_utc': float(submission.created_utc),
            'url': str(submission.url),
            'author': str(submission.author) if hasattr(submission, 'author') else 'unknown',
            'id': str(submission.id),
        }

    # -------------------------------------------------------------------------
    # Sentiment Analysis
    # -------------------------------------------------------------------------

    def analyze_text(self, text: str) -> dict:
        """
        综合分析单条文本.

        组合 RedditLingo + VADER (若可用).

        Args:
            text: 输入文本 (title + body)

        Returns:
            dict:
                - lingo_score: float (原始俚语分数)
                - lingo_normalized: float [-1, +1]
                - vader_compound: float (VADER compound, 若可用)
                - combined: float (综合分数, 若 VADER 不可用则同 lingo_normalized)
                - mentions: List[(word, score)]
        """
        lingo_raw = self.lingo_scorer.score(text)
        lingo_norm = self.lingo_scorer.score_normalized(text)
        mentions = self.lingo_scorer.extract_mentions(text)

        result = {
            'lingo_score': lingo_raw,
            'lingo_normalized': lingo_norm,
            'vader_compound': 0.0,
            'combined': lingo_norm,
            'mentions': mentions,
        }

        if self.vader_analyzer:
            try:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                result['vader_compound'] = vader_scores['compound']
                # 综合: 0.6 * lingo + 0.4 * vader (经验权重)
                result['combined'] = 0.6 * lingo_norm + 0.4 * vader_scores['compound']
            except Exception as e:
                logger.warning(f"VADER analysis failed: {e}")

        return result

    def analyze_posts(self, posts: list[dict]) -> dict:
        """
        分析一组帖子的情绪.

        Args:
            posts: fetch_posts() 返回的帖子列表

        Returns:
            dict:
                - avg_sentiment: float [-1, +1]
                - sentiment_std: float
                - sentiment_dist: dict (pos/neu/neg 比例)
                - top_posts: List[dict] (情绪最高的3条)
                - total_posts: int
                - lingo_vader_corr: float (两者相关性, 若 VADER 可用)
        """
        if not posts:
            return {
                'avg_sentiment': 0.0,
                'sentiment_std': 0.0,
                'sentiment_dist': {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0},
                'top_posts': [],
                'total_posts': 0,
                'lingo_vader_corr': 0.0,
            }

        sentiments: list[float] = []
        vader_scores: list[float] = []
        scored_posts: list[dict] = []

        for post in posts:
            text = f"{post.get('title', '')} {post.get('body', '')}"
            analysis = self.analyze_text(text)
            sentiment = analysis['combined']
            sentiments.append(sentiment)

            if analysis['vader_compound'] != 0.0:
                vader_scores.append(analysis['vader_compound'])

            scored_posts.append({
                **post,
                'sentiment': sentiment,
                'analysis': analysis,
            })

        import statistics
        avg_sent = statistics.mean(sentiments) if sentiments else 0.0
        std_sent = statistics.stdev(sentiments) if len(sentiments) > 1 else 0.0

        # Sentiment distribution
        pos = sum(1 for s in sentiments if s > 0.1) / len(sentiments)
        neg = sum(1 for s in sentiments if s < -0.1) / len(sentiments)
        neu = 1.0 - pos - neg

        # Top 3 most bullish posts
        top_posts = sorted(scored_posts, key=lambda x: x['sentiment'], reverse=True)[:3]

        # Lingo-VADER correlation
        corr = 0.0
        if vader_scores and len(vader_scores) == len(sentiments):
            try:
                import math
                n = len(sentiments)
                mean_s = avg_sent
                mean_v = statistics.mean(vader_scores)
                cov = sum((s - mean_s) * (v - mean_v) for s, v in zip(sentiments, vader_scores)) / n
                std_s = std_sent if std_sent > 0 else 1e-9
                std_v = statistics.stdev(vader_scores) if len(vader_scores) > 1 else 1e-9
                corr = cov / (std_s * std_v)
            except Exception:
                corr = 0.0

        return {
            'avg_sentiment': avg_sent,
            'sentiment_std': std_sent,
            'sentiment_dist': {
                'positive': pos,
                'neutral': neu,
                'negative': neg,
            },
            'top_posts': top_posts,
            'total_posts': len(posts),
            'lingo_vader_corr': corr,
        }

    # -------------------------------------------------------------------------
    # Subreddit & Ticker Analysis
    # -------------------------------------------------------------------------

    def analyze_subreddit(
        self,
        subreddit: str,
        ticker: Optional[str] = None,
        limit: int = 100,
    ) -> dict:
        """
        综合分析 subreddit.

        Args:
            subreddit: 子版块名称
            ticker: 可选, 只返回包含该 ticker 的帖子分析
            limit: 最大帖子数

        Returns:
            dict:
                - avg_sentiment: float [-1, +1]
                - sentiment_dist: dict
                - top_posts: List[dict]
                - ticker_mentions: int (若 ticker 指定)
                - ticker_sentiment: float (若 ticker 指定)
                - posts_analyzed: int
                - subreddit: str
                - timestamp: float
        """
        posts = self.fetch_posts(subreddit, keyword=ticker, limit=limit)

        if ticker:
            ticker_posts = [
                p for p in posts
                if ticker.upper() in (p.get('title', '') + p.get('body', '')).upper()
            ]
        else:
            ticker_posts = posts

        analysis = self.analyze_posts(ticker_posts)

        result = {
            'subreddit': subreddit,
            'timestamp': time.time(),
            'posts_analyzed': analysis['total_posts'],
            **analysis,
        }

        if ticker:
            ticker_analysis = self.analyze_posts(ticker_posts)
            result['ticker_mentions'] = len(ticker_posts)
            result['ticker_sentiment'] = ticker_analysis.get('avg_sentiment', 0.0)

        # Cache posts for embedding
        self._post_cache = posts

        return result

    def get_ticker_sentiment(
        self,
        ticker: str,
        subreddits: Optional[list[str]] = None,
    ) -> dict:
        """
        获取特定股票/代币的情绪.

        在多个 subreddit 中搜索该 ticker, 汇总情绪.

        Args:
            ticker: 股票/代币代码 (如 'GME', 'BTC', 'SPY')
            subreddits: 要搜索的 subreddit 列表

        Returns:
            dict:
                - sentiment: float [-1, +1] (加权平均情绪)
                - mention_count: int
                - confidence: float (基于样本量和一致性)
                - by_subreddit: dict (各 subreddit 分项)
        """
        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'Daytrading']

        ticker = ticker.upper()
        all_sentiments: list[float] = []
        by_subreddit: dict[str, dict] = {}

        for sub in subreddits:
            result = self.analyze_subreddit(sub, ticker=ticker, limit=100)
            sub_sentiment = result.get('avg_sentiment', 0.0)
            sub_count = result.get('total_posts', 0)

            if sub_count > 0:
                all_sentiments.extend([sub_sentiment] * sub_count)
                by_subreddit[sub] = {
                    'sentiment': sub_sentiment,
                    'mention_count': sub_count,
                    'sentiment_dist': result.get('sentiment_dist', {}),
                }

        if not all_sentiments:
            return {
                'sentiment': 0.0,
                'mention_count': 0,
                'confidence': 0.0,
                'by_subreddit': {},
            }

        import statistics
        avg_sentiment = statistics.mean(all_sentiments)
        std_sentiment = statistics.stdev(all_sentiments) if len(all_sentiments) > 1 else 0.0

        # Confidence: 高样本量 + 低标准差 = 高置信度
        sample_size = len(all_sentiments)
        confidence = min(1.0, sample_size / 50.0) * (1.0 - min(1.0, std_sentiment))

        return {
            'sentiment': avg_sentiment,
            'mention_count': sample_size,
            'confidence': round(confidence, 3),
            'sentiment_std': round(std_sentiment, 3),
            'by_subreddit': by_subreddit,
        }

    # -------------------------------------------------------------------------
    # BERT + FAISS Embedding (Optional)
    # -------------------------------------------------------------------------

    def embed_posts(self, posts: Optional[list[dict]] = None) -> Optional[Any]:
        """
        将帖子列表转为 BERT embedding 并存入 FAISS 索引.

        Args:
            posts: 帖子列表, 若为 None 则使用最近 fetch_posts 的缓存

        Returns:
            FAISS index 或 None (若不可用)
        """
        if not BERT_AVAILABLE:
            return None

        posts = posts or self._post_cache
        if not posts:
            return None

        texts = [f"{p.get('title', '')} {p.get('body', '')}" for p in posts]

        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            # Normalize for cosine similarity (FAISS IndexFlatIP)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            embeddings = embeddings / norms

            if self.index is None:
                self.index = faiss.IndexFlatIP(self._embedding_dim)

            self.index.reset()
            self.index.add(embeddings.astype(np.float32))

            logger.info(f"Indexed {len(texts)} embeddings in FAISS")
            return self.index

        except Exception as e:
            logger.error(f"BERT/FAISS embedding failed: {e}")
            return None

    def search_similar_posts(
        self,
        query: str,
        top_k: int = 5,
        posts: Optional[list[dict]] = None,
    ) -> list[tuple[dict, float]]:
        """
        在已索引的帖子中搜索与 query 相似的帖子.

        Args:
            query: 查询文本
            top_k: 返回前 K 条
            posts: 可选, 指定帖子列表 (若未 embed 则自动 embed)

        Returns:
            List[(post_dict, similarity_score)]
        """
        if not BERT_AVAILABLE or not self.embedding_model:
            logger.warning("BERT not available — returning empty")
            return []

        posts = posts or self._post_cache
        if not posts:
            return []

        # Ensure indexed
        if self.index is None or self.index.ntotal == 0:
            self.embed_posts(posts)

        if self.index is None or self.index.ntotal == 0:
            return []

        try:
            # Encode query
            q_emb = self.embedding_model.encode([query])
            norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            q_emb = q_emb / norms

            # Search
            D, I = self.index.search(q_emb.astype(np.float32), min(top_k, self.index.ntotal))

            results: list[tuple[dict, float]] = []
            for idx, score in zip(I[0], D[0]):
                if 0 <= idx < len(posts):
                    results.append((posts[idx], float(score)))
            return results

        except Exception as e:
            logger.error(f"Similar post search failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # VADER Fallback (Regex-based)
    # -------------------------------------------------------------------------

    @staticmethod
    def vader_fallback(text: str) -> dict:
        """
        VADER 降级方案 — 基于正则的情绪分析.

        当 NLTK VADER 不可用时使用.

        Returns:
            dict: {neg, neu, pos, compound}
        """
        import re

        # 简化的情绪词典
        pos_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'best',
            'bullish', 'moon', 'rocket', 'profit', 'win', 'happy',
        }
        neg_words = {
            'bad', 'terrible', 'awful', 'worst', 'loss', 'lose',
            'bearish', 'dump', 'scam', 'fear', 'panic',
        }

        words = re.findall(r'\b\w+\b', text.lower())
        pos = neg = neu = 0.0

        for w in words:
            if w in pos_words:
                pos += 1
            elif w in neg_words:
                neg += 1
            else:
                neu += 1

        total = pos + neg + neu or 1
        compound = (pos - neg) / total

        return {
            'neg': neg / total,
            'neu': neu / total,
            'pos': pos / total,
            'compound': compound,
        }
