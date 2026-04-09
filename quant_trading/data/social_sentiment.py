"""
Social Sentiment Analyzer for Quant Trading System
===================================================
Reddit social sentiment pipeline with:
- Reddit sentiment via PRAW + NLTK/VADER
- Apewisdom API integration for trending tickers
- BERT/GPT embeddings for Reddit posts
- FAISS vector database for embeddings
- Random Forest Regressor for sentiment-price correlation

Covers 15+ subreddits: wallstreetbets, bitcoin, CryptoCurrency, etc.
"""

import re
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ============================================================================
# Optional Imports - Core dependencies (fail gracefully if missing)
# ============================================================================

VADER_AVAILABLE = False
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    pass

NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer as NLTKVader
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        NLTK_AVAILABLE = True
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True)
            NLTK_AVAILABLE = True
        except Exception:
            pass
except ImportError:
    pass

PRAW_AVAILABLE = False
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    pass

SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import BertTokenizer, BertModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

DOTENV_AVAILABLE = False
try:
    import dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RedditPost:
    """Represents a Reddit post with metadata."""
    title: str
    body: str
    score: int
    created_utc: float
    url: str
    subreddit: str
    comments: List[Dict[str, Any]] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class TrendingTicker:
    """Represents a trending ticker from Apewisdom."""
    ticker: str
    mentions: int
    rank: int
    sentiment: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class SocialSentimentSignal:
    """Aggregate social sentiment signal for a ticker."""
    ticker: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    mention_count: int
    trending_score: float
    confidence: float  # 0 to 1
    source_breakdown: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Text Utilities
# ============================================================================

def clean_text(text: str) -> str:
    """Clean text for embedding/sentiment analysis."""
    if not text:
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = re.sub(r'\$([A-Z]{1,5})\b', r'\1', text)  # Keep ticker symbols but remove $
    return text.lower().strip()


def extract_tickers(text: str) -> List[str]:
    """Extract potential stock tickers from text using regex."""
    ticker_pattern = re.compile(r'\b([A-Z]{1,5})\b')
    tickers = ticker_pattern.findall(text)
    # Filter out common English words that are also ticker-like
    common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
                    'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS',
                    'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE',
                    'WAY', 'WHO', 'BOY', 'DID', 'GET', 'LET', 'PUT', 'SAY',
                    'SHE', 'TOO', 'USE', 'RMB', 'USD', 'EUR', 'GBP', 'BTC',
                    'ETH', 'DOGE', 'ADA', 'SOL', 'XRP'}
    return [t for t in tickers if t not in common_words]


# ============================================================================
# Subreddit Configuration
# ============================================================================

DEFAULT_SUBREDDITS = {
    # Stock-focused subreddits
    'wallstreetbets',
    'stocks',
    'pennystocks',
    'StockMarket',
    'EducatedInvesting',
    'wallstreetbetsnew',
    'investing',
    'Daytrading',
    'Stock_Picks',
    # Crypto subreddits
    'bitcoin',
    'btc',
    'CryptoCurrency',
    'BitcoinBeginners',
    'binance',
    'coinbase',
    'CryptoMoonShots',
    'CryptoMarkets',
    'CryptoTechnology',
    # Finance
    'finance',
    'personalfinance',
    'business',
}

CRYPTO_SUBREDDITS = {
    'bitcoin',
    'btc',
    'CryptoCurrency',
    'BitcoinBeginners',
    'binance',
    'coinbase',
    'CryptoMoonShots',
    'CryptoMarkets',
    'CryptoTechnology',
}

STOCK_SUBREDDITS = {
    'wallstreetbets',
    'stocks',
    'pennystocks',
    'StockMarket',
    'EducatedInvesting',
    'wallstreetbetsnew',
    'investing',
    'Daytrading',
    'Stock_Picks',
}


# ============================================================================
# Reddit Sentiment Collector
# ============================================================================

class RedditSentimentCollector:
    """
    Collects posts from Reddit subreddits for sentiment analysis.

    Features:
    - Multi-subreddit collection with ThreadPoolExecutor for parallel fetching
    - Time-based filtering (last 24 hours by default)
    - Comment extraction with nested replies
    - Ticker filtering based on financial keywords
    - Optional Reddit API (stubs gracefully if no API key)

    Usage:
        collector = RedditSentimentCollector()
        posts = collector.collect_posts(subreddits=['wallstreetbets', 'stocks'])
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "QuantTradingBot/1.0",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize Reddit client.

        Args:
            client_id: Reddit API client ID (or REDDIT_CLIENT_ID env var)
            client_secret: Reddit API client secret (or REDDIT_SECRET env var)
            user_agent: Reddit user agent string
            username: Reddit username (optional)
            password: Reddit password (optional)
        """
        self.reddit = None
        self._available = PRAW_AVAILABLE

        if not PRAW_AVAILABLE:
            logger.warning(
                "PRAW not available. Reddit collection will return stub data. "
                "Install with: pip install praw"
            )
            return

        # Load from environment if not provided
        if DOTENV_AVAILABLE:
            dotenv.load_dotenv()

        client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        client_secret = client_secret or os.getenv('REDDIT_SECRET')
        username = username or os.getenv('REDDIT_USERNAME')
        password = password or os.getenv('REDDIT_PASSWORD')

        if not client_id or not client_secret:
            logger.warning(
                "Reddit API credentials not provided. "
                "Set REDDIT_CLIENT_ID and REDDIT_SECRET environment variables. "
                "Reddit collection will return stub data."
            )
            return

        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                username=username,
                password=password,
            )
            # Test connection
            self.reddit.subreddit('wallstreetbets').title
            logger.info("Reddit API connection established")
        except Exception as e:
            logger.warning(f"Failed to connect to Reddit API: {e}. Using stub mode.")
            self.reddit = None

    @property
    def is_available(self) -> bool:
        """Check if Reddit API is available."""
        return self.reddit is not None

    def _fetch_comments(self, post: Any) -> Tuple[List[Dict[str, Any]], int]:
        """
        Fetch comments for a Reddit post.

        Args:
            post: PRAW submission object

        Returns:
            Tuple of (comments_data, comment_count)
        """
        post.comments.replace_more(limit=0)
        comments_data = []
        comment_count = 0

        for comment in post.comments.list():
            if isinstance(comment, praw.models.MoreComments):
                continue
            comment_count += 1
            comment_data = {
                'body': comment.body,
                'created_utc': datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                'score': comment.score,
                'replies': []
            }
            for reply in comment.replies:
                if isinstance(reply, praw.models.MoreComments):
                    continue
                comment_count += 1
                comment_data['replies'].append({
                    'body': reply.body,
                    'created_utc': datetime.utcfromtimestamp(reply.created_utc).isoformat(),
                    'score': reply.score
                })
            comments_data.append(comment_data)

        return comments_data, comment_count

    def _collect_from_subreddit(
        self,
        subreddit_name: str,
        limit: int = 100,
        time_filter_hours: int = 24,
    ) -> List[RedditPost]:
        """Collect posts from a single subreddit."""
        if not self.reddit:
            return []

        subreddit = self.reddit.subreddit(subreddit_name)
        cutoff_timestamp = (datetime.utcnow() - timedelta(hours=time_filter_hours)).timestamp()
        posts = []

        try:
            for post in subreddit.new(limit=limit):
                if post.created_utc < cutoff_timestamp:
                    continue

                # Filter for financial relevance
                combined_text = (post.title + ' ' + post.selftext).lower()
                financial_keywords = [
                    'stock', 'share', 'market', 'investment', 'price', 'trading',
                    'bullish', 'bearish', 'ipo', 'earnings', 'dividend', 'ticker',
                    'buy', 'sell', 'call', 'put', 'option', 'portfolio', 'fund'
                ]

                if not any(kw in combined_text for kw in financial_keywords):
                    continue

                comments_data, _ = self._fetch_comments(post)

                reddit_post = RedditPost(
                    title=post.title,
                    body=post.selftext,
                    score=post.score,
                    created_utc=post.created_utc,
                    url=post.url,
                    subreddit=subreddit_name,
                    comments=comments_data,
                )
                posts.append(reddit_post)

        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit_name}: {e}")

        return posts

    def collect_posts(
        self,
        subreddits: Optional[List[str]] = None,
        limit_per_subreddit: int = 100,
        time_filter_hours: int = 24,
        parallel: bool = True,
    ) -> List[RedditPost]:
        """
        Collect posts from multiple subreddits.

        Args:
            subreddits: List of subreddit names (default: DEFAULT_SUBREDDITS)
            limit_per_subreddit: Max posts per subreddit
            time_filter_hours: Only collect posts within this time window
            parallel: Use ThreadPoolExecutor for parallel fetching

        Returns:
            List of RedditPost objects
        """
        if not self.is_available:
            logger.info("Reddit not available, returning empty list")
            return []

        subreddits = subreddits or list(DEFAULT_SUBREDDITS)
        all_posts = []

        if parallel and len(subreddits) > 1:
            with ThreadPoolExecutor(max_workers=min(10, len(subreddits))) as executor:
                futures = {
                    executor.submit(
                        self._collect_from_subreddit,
                        sub,
                        limit_per_subreddit,
                        time_filter_hours
                    ): sub
                    for sub in subreddits
                }

                for future in as_completed(futures):
                    sub = futures[future]
                    try:
                        posts = future.result()
                        all_posts.extend(posts)
                        logger.info(f"Collected {len(posts)} posts from r/{sub}")
                    except Exception as e:
                        logger.error(f"Error collecting from r/{sub}: {e}")
        else:
            for sub in subreddits:
                posts = self._collect_from_subreddit(sub, limit_per_subreddit, time_filter_hours)
                all_posts.extend(posts)
                logger.info(f"Collected {len(posts)} posts from r/{sub}")

        return all_posts

    def get_ticker_posts(
        self,
        ticker: str,
        subreddits: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[RedditPost]:
        """
        Get posts mentioning a specific ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA', 'AAPL')
            subreddits: Subreddits to search (default: STOCK_SUBREDDITS)
            limit: Max posts to return

        Returns:
            List of RedditPost objects mentioning the ticker
        """
        if not self.is_available:
            return []

        subreddits = subreddits or list(STOCK_SUBREDDITS)
        ticker = ticker.upper()
        all_posts = []

        for sub_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(sub_name)
                query = f"{ticker} stock"
                for submission in subreddit.search(query, limit=limit):
                    comments_data, _ = self._fetch_comments(submission)
                    reddit_post = RedditPost(
                        title=submission.title,
                        body=submission.selftext,
                        score=submission.score,
                        created_utc=submission.created_utc,
                        url=submission.url,
                        subreddit=sub_name,
                        comments=comments_data,
                    )
                    all_posts.append(reddit_post)
            except Exception as e:
                logger.error(f"Error searching r/{sub_name} for {ticker}: {e}")

        return all_posts


# ============================================================================
# Sentiment Analyzer (VADER-based)
# ============================================================================

class SentimentAnalyzer:
    """
    Analyzes sentiment of text using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    VADER is specifically attuned to social media sentiment and handles:
    - Emojis and emoticons
    - Capitalization (intensity modification)
    - Punctuation (exclamation marks increase intensity)
    - Negations
    - But-of-a-thing constructions
    """

    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        self.analyzer = None
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        elif NLTK_AVAILABLE:
            try:
                self.analyzer = NLTKVader()
            except Exception:
                logger.warning("Failed to initialize NLTK VADER")

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text.

        Args:
            text: Input text

        Returns:
            Compound sentiment score between -1 (negative) and 1 (positive)
        """
        if not self.analyzer:
            # Return neutral sentiment if VADER not available
            return 0.0

        cleaned = clean_text(text)
        if not cleaned:
            return 0.0

        scores = self.analyzer.polarity_scores(cleaned)
        return scores['compound']

    def analyze_post(self, post: RedditPost) -> float:
        """
        Analyze sentiment of a Reddit post (title + body + comments).

        Args:
            post: RedditPost object

        Returns:
            Weighted sentiment score
        """
        # Title has higher weight
        title_sentiment = self.analyze(post.title) * 1.5
        body_sentiment = self.analyze(post.body) if post.body else 0.0

        # Comment sentiments weighted by score
        comment_sentiments = []
        for comment in post.comments:
            score = comment.get('score', 1)
            # Higher scored comments have more weight
            weight = min(max(score / 100, 1), 5)
            comment_sentiments.append(self.analyze(comment['body']) * weight)

        avg_comment_sentiment = np.mean(comment_sentiments) if comment_sentiments else 0.0

        # Weighted combination
        total_sentiment = (title_sentiment + body_sentiment * 0.5 + avg_comment_sentiment) / 3.0

        return float(np.clip(total_sentiment, -1, 1))


# ============================================================================
# BERT Embeddings for Posts
# ============================================================================

class SentimentEmbedder:
    """
    Generates BERT embeddings for Reddit posts.

    Uses BERT to create dense vector representations of post content
    for semantic similarity search and clustering.

    Supports:
    - BERT (transformers) embeddings
    - OpenAI GPT embeddings (if available)
    - FAISS vector storage for efficient similarity search
    - Stub mode when dependencies unavailable
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        embedding_dim: int = 768,
    ):
        """
        Initialize embedder.

        Args:
            model_name: HuggingFace model name for BERT
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.bert_tokenizer = None
        self.bert_model = None
        self.gpt_client = None

        # Try to initialize BERT
        if TRANSFORMERS_AVAILABLE:
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
                self.bert_model = BertModel.from_pretrained(model_name)
                self.bert_model.eval()
                logger.info(f"BERT model '{model_name}' loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load BERT model: {e}")

        # Try to initialize OpenAI GPT
        if OPENAI_AVAILABLE:
            try:
                if DOTENV_AVAILABLE:
                    dotenv.load_dotenv()
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.gpt_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI GPT embedder initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI GPT: {e}")

        if not self.bert_tokenizer and not self.gpt_client:
            logger.warning(
                "No embedding model available. Install transformers or openai. "
                "Using random embeddings as stub."
            )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector (768-dim for BERT)
        """
        if not text or not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)

        cleaned = clean_text(text)

        if self.gpt_client:
            return self._embed_with_gpt(cleaned)
        elif self.bert_tokenizer and self.bert_model:
            return self._embed_with_bert(cleaned)
        else:
            # Stub: return random embedding
            return np.random.rand(self.embedding_dim).astype(np.float32)

    def _embed_with_bert(self, text: str) -> np.ndarray:
        """Generate BERT embedding."""
        try:
            inputs = self.bert_tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            # Mean pooling of last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"BERT embedding error: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def _embed_with_gpt(self, text: str) -> np.ndarray:
        """Generate OpenAI GPT embedding."""
        try:
            response = self.gpt_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8192],
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"GPT embedding error: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def embed_posts(self, posts: List[RedditPost]) -> np.ndarray:
        """
        Generate embeddings for multiple posts.

        Args:
            posts: List of RedditPost objects

        Returns:
            numpy array of shape (n_posts, embedding_dim)
        """
        embeddings = []
        for post in posts:
            # Combine title and body for embedding
            text = f"{post.title} {post.body}"
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def embed_with_faiss(
        self,
        posts: List[RedditPost],
    ) -> Tuple[np.ndarray, Any]:
        """
        Create embeddings and store in FAISS index.

        Args:
            posts: List of RedditPost objects

        Returns:
            Tuple of (embeddings, faiss_index)
        """
        embeddings = self.embed_posts(posts)

        if FAISS_AVAILABLE and embeddings.shape[0] > 0:
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            return embeddings, index

        return embeddings, None

    def find_similar(
        self,
        index: Any,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar posts to query embedding.

        Args:
            index: FAISS index
            query_embedding: Query vector
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if FAISS_AVAILABLE and index is not None:
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            distances, indices = index.search(query_embedding, k)
            return distances[0], indices[0]

        # Stub: return random results
        n = index.ntotal if index is not None else 0
        if n == 0:
            return np.array([]), np.array([])
        indices = np.random.choice(n, min(k, n), replace=False)
        distances = np.random.rand(len(indices))
        return distances, indices


# ============================================================================
# Apewisdom Trending Ticker Finder
# ============================================================================

class TrendingTickerFinder:
    """
    Finds trending tickers using the Apewisdom API.

    Apewisdom aggregates social media sentiment for stocks and crypto,
    providing real-time trending data from Reddit and other platforms.

    API Endpoints:
    - https://apewisdom.io/api/v1.0/filter/all-stocks/page/1
    - https://apewisdom.io/api/v1.0/filter/all-crypto/page/1
    - https://apewisdom.io/api/v1.0/filter/4chan/page/1
    """

    BASE_URL = 'https://apewisdom.io/api/v1.0/filter'

    def __init__(self):
        """Initialize Apewisdom client."""
        self.session = None

    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Make HTTP request to Apewisdom API."""
        try:
            import requests
            url = f"{self.BASE_URL}/{endpoint}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Apewisdom API error: {response.status_code}")
                return None
        except ImportError:
            logger.warning("requests library not available")
            return None
        except Exception as e:
            logger.error(f"Apewisdom request failed: {e}")
            return None

    def get_trending_stocks(
        self,
        page: int = 1,
        pages: int = 1,
        filter_type: str = 'all-stocks',
    ) -> List[TrendingTicker]:
        """
        Get trending stock tickers.

        Args:
            page: Starting page number
            pages: Number of pages to fetch
            filter_type: Filter type ('all-stocks', 'reddit-stocks', 'stock-picks')

        Returns:
            List of TrendingTicker objects
        """
        tickers = []

        for p in range(page, page + pages):
            data = self._make_request(f'{filter_type}/page/{p}')
            if not data or 'results' not in data:
                continue

            for rank, item in enumerate(data['results'], start=1):
                ticker = TrendingTicker(
                    ticker=item.get('ticker', item.get('symbol', '')).upper(),
                    mentions=item.get('mentions', 0),
                    rank=rank + (p - 1) * 100,
                    sentiment=item.get('sentiment'),
                    timestamp=datetime.utcnow(),
                )
                tickers.append(ticker)

        return tickers

    def get_trending_crypto(
        self,
        page: int = 1,
        pages: int = 1,
    ) -> List[TrendingTicker]:
        """Get trending cryptocurrency tickers."""
        return self.get_trending_stocks(
            page=page,
            pages=pages,
            filter_type='all-crypto',
        )

    def get_top_tickers(
        self,
        n: int = 20,
        include_crypto: bool = True,
    ) -> List[TrendingTicker]:
        """
        Get top N trending tickers across all categories.

        Args:
            n: Number of top tickers to return
            include_crypto: Include cryptocurrency tickers

        Returns:
            List of top TrendingTicker objects
        """
        all_tickers = []

        # Get stocks (need more pages for volume)
        stock_pages = (n // 100) + 1
        stocks = self.get_trending_stocks(pages=stock_pages)
        all_tickers.extend(stocks)

        # Get crypto if requested
        if include_crypto:
            crypto_pages = (n // 100) + 1
            cryptos = self.get_trending_crypto(pages=crypto_pages)
            all_tickers.extend(cryptos)

        # Sort by mentions and take top N
        all_tickers.sort(key=lambda x: x.mentions, reverse=True)
        return all_tickers[:n]


# ============================================================================
# Social Sentiment Signal Generator
# ============================================================================

class SocialSentimentSignal:
    """
    Generates aggregate social sentiment signals for trading.

    Combines:
    - Reddit sentiment (VADER)
    - Apewisdom trending data
    - Post volume and engagement metrics
    - Historical sentiment trends

    Uses Random Forest to find sentiment-price correlations.
    """

    def __init__(
        self,
        collector: Optional[RedditSentimentCollector] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
        embedder: Optional[SentimentEmbedder] = None,
        ticker_finder: Optional[TrendingTickerFinder] = None,
    ):
        """
        Initialize signal generator.

        Args:
            collector: RedditSentimentCollector instance
            analyzer: SentimentAnalyzer instance
            embedder: SentimentEmbedder instance
            ticker_finder: TrendingTickerFinder instance
        """
        self.collector = collector or RedditSentimentCollector()
        self.analyzer = analyzer or SentimentAnalyzer()
        self.embedder = embedder or SentimentEmbedder()
        self.ticker_finder = ticker_finder or TrendingTickerFinder()
        self.rf_model = None
        self._init_rf_model()

    def _init_rf_model(self):
        """Initialize Random Forest model for sentiment-price correlation."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. RF model disabled.")
            return

        # Create a basic model with default parameters
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

    def analyze_reddit_sentiment(
        self,
        ticker: str,
        subreddits: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze Reddit sentiment for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            subreddits: Subreddits to search

        Returns:
            Dictionary with sentiment analysis results
        """
        posts = self.collector.get_ticker_posts(ticker, subreddits)

        if not posts:
            return {
                'ticker': ticker,
                'post_count': 0,
                'avg_sentiment': 0.0,
                'sentiment_std': 0.0,
                'total_score': 0,
                'source': 'reddit',
            }

        sentiments = []
        scores = []

        for post in posts:
            sentiment = self.analyzer.analyze_post(post)
            post.sentiment_score = sentiment
            sentiments.append(sentiment)
            scores.append(post.score)

        return {
            'ticker': ticker,
            'post_count': len(posts),
            'avg_sentiment': float(np.mean(sentiments)),
            'sentiment_std': float(np.std(sentiments)),
            'sentiment_min': float(np.min(sentiments)),
            'sentiment_max': float(np.max(sentiments)),
            'total_score': sum(scores),
            'avg_score': float(np.mean(scores)),
            'posts': posts,
            'source': 'reddit',
        }

    def get_trending_signals(
        self,
        n_tickers: int = 20,
    ) -> List[SocialSentimentSignal]:
        """
        Get sentiment signals for trending tickers.

        Args:
            n_tickers: Number of top trending tickers to analyze

        Returns:
            List of SocialSentimentSignal objects
        """
        trending = self.ticker_finder.get_top_tickers(n=n_tickers)
        signals = []

        for ticker_data in trending:
            signal = SocialSentimentSignal(
                ticker=ticker_data.ticker,
                timestamp=ticker_data.timestamp,
                sentiment_score=ticker_data.sentiment or 0.0,
                mention_count=ticker_data.mentions,
                trending_score=float(ticker_data.rank) / float(n_tickers),
                confidence=min(ticker_data.mentions / 1000, 1.0),
            )
            signals.append(signal)

        return signals

    def generate_signal(
        self,
        ticker: str,
        subreddits: Optional[List[str]] = None,
    ) -> SocialSentimentSignal:
        """
        Generate a complete sentiment signal for a ticker.

        Args:
            ticker: Stock ticker symbol
            subreddits: Subreddits to analyze

        Returns:
            SocialSentimentSignal object
        """
        # Get Apewisdom trending data
        trending = self.ticker_finder.get_top_tickers(n=100)
        ticker_data = next((t for t in trending if t.ticker == ticker.upper()), None)

        # Get Reddit sentiment
        reddit_analysis = self.analyze_reddit_sentiment(ticker, subreddits)

        # Combine signals
        sentiment = reddit_analysis.get('avg_sentiment', 0.0)
        mention_count = reddit_analysis.get('post_count', 0)

        if ticker_data:
            mention_count += ticker_data.mentions
            # Weighted combination of Reddit and Apewisdom sentiment
            sentiment = (sentiment * 0.6 + (ticker_data.sentiment or 0) * 0.4)

        confidence = min(mention_count / 100, 1.0) if mention_count > 0 else 0.0

        return SocialSentimentSignal(
            ticker=ticker.upper(),
            timestamp=datetime.utcnow(),
            sentiment_score=sentiment,
            mention_count=mention_count,
            trending_score=ticker_data.rank / 100 if ticker_data else 0.0,
            confidence=confidence,
            source_breakdown={
                'reddit': reddit_analysis.get('avg_sentiment', 0.0),
                'apewisdom': ticker_data.sentiment if ticker_data else 0.0,
            },
        )

    def train_sentiment_model(
        self,
        price_data: pd.DataFrame,
        sentiment_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Train Random Forest model to find sentiment-price correlations.

        Args:
            price_data: DataFrame with price columns (close, volume, etc.)
            sentiment_data: DataFrame with sentiment columns

        Returns:
            Dictionary with model metrics
        """
        if not SKLEARN_AVAILABLE or self.rf_model is None:
            return {'error': 'Model not available'}

        # Merge datasets on common index/date
        merged = pd.merge(price_data, sentiment_data, on='date', how='inner')
        if len(merged) < 10:
            return {'error': 'Insufficient data for training'}

        # Features: sentiment metrics
        feature_cols = ['sentiment', 'mention_count', 'post_count']
        for col in feature_cols:
            if col not in merged.columns:
                merged[col] = 0

        # Target: next day price change (percentage)
        merged['target'] = merged['close'].pct_change().shift(-1)

        # Drop NaN rows
        merged = merged.dropna(subset=feature_cols + ['target'])

        if len(merged) < 10:
            return {'error': 'Insufficient clean data for training'}

        X = merged[feature_cols]
        y = merged['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.rf_model.fit(X_train, y_train)
        y_pred = self.rf_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'mse': float(mse),
            'r2': float(r2),
            'train_size': len(X_train),
            'test_size': len(X_test),
        }

    def predict_from_sentiment(
        self,
        sentiment_signal: SocialSentimentSignal,
    ) -> Optional[float]:
        """
        Use trained model to predict price direction from sentiment.

        Args:
            sentiment_signal: SocialSentimentSignal object

        Returns:
            Predicted price change percentage, or None if model not available
        """
        if not SKLEARN_AVAILABLE or self.rf_model is None:
            return None

        features = np.array([[
            sentiment_signal.sentiment_score,
            sentiment_signal.mention_count,
            sentiment_signal.trending_score,
        ]])

        return float(self.rf_model.predict(features)[0])


# ============================================================================
# Factory Function
# ============================================================================

def create_social_sentiment_pipeline(
    reddit_credentials: Optional[Dict[str, str]] = None,
) -> Tuple[RedditSentimentCollector, SentimentAnalyzer, SentimentEmbedder, TrendingTickerFinder]:
    """
    Create a complete social sentiment pipeline.

    Args:
        reddit_credentials: Optional dict with Reddit API credentials

    Returns:
        Tuple of (collector, analyzer, embedder, ticker_finder)
    """
    if reddit_credentials:
        collector = RedditSentimentCollector(**reddit_credentials)
    else:
        collector = RedditSentimentCollector()

    analyzer = SentimentAnalyzer()
    embedder = SentimentEmbedder()
    ticker_finder = TrendingTickerFinder()

    return collector, analyzer, embedder, ticker_finder


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Social Sentiment Analyzer - Demo")
    print("=" * 50)

    # Initialize components
    collector = RedditSentimentCollector()
    analyzer = SentimentAnalyzer()
    embedder = SentimentEmbedder()
    ticker_finder = TrendingTickerFinder()

    print(f"Reddit API available: {collector.is_available}")
    print(f"VADER available: {VADER_AVAILABLE}")
    print(f"BERT available: {TRANSFORMERS_AVAILABLE}")
    print(f"FAISS available: {FAISS_AVAILABLE}")
    print(f"sklearn available: {SKLEARN_AVAILABLE}")

    # Test sentiment analysis (no API required)
    test_texts = [
        "GME to the moon! Moon soon! 🚀🚀🚀",
        "Market crashing, losing everything",
        "AAPL earnings look decent, holding steady",
    ]

    print("\nSentiment Analysis Test:")
    for text in test_texts:
        score = analyzer.analyze(text)
        print(f"  '{text[:40]}...' -> {score:.3f}")

    # Test ticker extraction
    print("\nTicker Extraction Test:")
    test_ticker_text = "Just bought 100 shares of NVDA and some BTC, also considering AAPL puts"
    tickers = extract_tickers(test_ticker_text)
    print(f"  Text: '{test_ticker_text}'")
    print(f"  Tickers: {tickers}")
