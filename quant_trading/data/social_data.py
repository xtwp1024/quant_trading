"""
Social Data Utilities for Quant Trading System
===============================================
Apewisdom API integration and subreddit filtering utilities.

Provides:
- Apewisdom API client for trending tickers
- Subreddit filtering and categorization
- Data validation and normalization
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class SentimentCategory(Enum):
    """Sentiment category classification."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class AssetType(Enum):
    """Asset type classification."""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    UNKNOWN = "unknown"


# Subreddit categories
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
    'ValueInvesting',
    'Dividends',
    'OptionsTrading',
    'Trading',
    'RobinHood',
    'StocksAndTrading',
    'WallStreetbetsELITE',
    'SPACs',
    'Momentum',
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
    'Cardano',
    'ethereum',
    'Solana',
    'Litecoin',
    'Ripple',
    'Dogecoin',
    'CryptoNews',
    'CryptoCurrencyTrading',
    'Altcoin',
    'BitcoinCA',
    'Etc',
}

FINANCE_SUBREDDITS = {
    'finance',
    'personalfinance',
    'business',
    'Economics',
    'FinancialPlanning',
    'Retirement',
    'Tax',
    'Accounting',
    'Investing',
    'Banking',
}

ALL_SUBREDDITS = STOCK_SUBREDDITS | CRYPTO_SUBREDDITS | FINANCE_SUBREDDITS

# Apewisdom filter types
APEWISDOM_FILTERS = {
    'all-stocks': 'All Stocks',
    'reddit-stocks': 'Reddit Stocks',
    'stock-picks': 'Stock Picks',
    'all-crypto': 'All Crypto',
    '4chan': '4chan',
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ApewisdomResult:
    """Result from Apewisdom API."""
    ticker: str
    mentions: int
    rank: int
    sentiment: Optional[float] = None
    sentiment_distribution: Optional[Dict[str, float]] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None


@dataclass
class SubredditInfo:
    """Information about a subreddit."""
    name: str
    category: AssetType
    subscribers: Optional[int] = None
    is_active: bool = True


@dataclass
class TickerMention:
    """Represents a ticker mention with metadata."""
    ticker: str
    source: str
    timestamp: datetime
    mentions: int
    sentiment: float
    context: Optional[str] = None


# ============================================================================
# Apewisdom API Client
# ============================================================================

class ApewisdomClient:
    """
    Client for Apewisdom API.

    Apewisdom aggregates social media sentiment and provides trending data
    for stocks, crypto, and other assets.

    API Documentation:
    - Base URL: https://apewisdom.io/api/v1.0/
    - Endpoints:
        - filter/{filter_type}/page/{page}
        - stocks/page/{page}
        - crypto/page/{page}

    Usage:
        client = ApewisdomClient()
        trending = client.get_trending(filter_type='all-stocks', pages=3)
    """

    BASE_URL = 'https://apewisdom.io/api/v1.0'

    def __init__(self, timeout: int = 10):
        """
        Initialize Apewisdom client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._session = None

    def _get_session(self):
        """Get or create requests session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': 'QuantTradingBot/1.0',
                'Accept': 'application/json',
            })
        return self._session

    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """
        Make HTTP request to Apewisdom API.

        Args:
            endpoint: API endpoint path

        Returns:
            Response JSON or None on error
        """
        try:
            import requests
            url = f"{self.BASE_URL}/{endpoint}"
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={'User-Agent': 'QuantTradingBot/1.0'},
            )
            response.raise_for_status()
            return response.json()
        except ImportError:
            logger.error("requests library not available")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Apewisdom API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Apewisdom API error: {e}")
            return None

    def get_trending(
        self,
        filter_type: str = 'all-stocks',
        page: int = 1,
        pages: int = 1,
    ) -> List[ApewisdomResult]:
        """
        Get trending tickers from Apewisdom.

        Args:
            filter_type: Type of filter ('all-stocks', 'reddit-stocks', 'all-crypto', etc.)
            page: Starting page number
            pages: Number of pages to fetch

        Returns:
            List of ApewisdomResult objects
        """
        results = []

        for p in range(page, page + pages):
            data = self._make_request(f'filter/{filter_type}/page/{p}')
            if not data or 'results' not in data:
                continue

            for item in data['results']:
                ticker = item.get('ticker', item.get('symbol', ''))
                if not ticker:
                    continue

                # Parse sentiment distribution if available
                sentiment_dist = None
                if 'sentiment_distribution' in item:
                    sentiment_dist = item['sentiment_distribution']

                result = ApewisdomResult(
                    ticker=ticker.upper(),
                    mentions=item.get('mentions', 0),
                    rank=item.get('rank', 0),
                    sentiment=item.get('sentiment'),
                    sentiment_distribution=sentiment_dist,
                    timestamp=datetime.utcnow(),
                    source=filter_type,
                )
                results.append(result)

        return results

    def get_stocks(self, page: int = 1, pages: int = 1) -> List[ApewisdomResult]:
        """Get trending stocks."""
        return self.get_trending('all-stocks', page, pages)

    def get_crypto(self, page: int = 1, pages: int = 1) -> List[ApewisdomResult]:
        """Get trending cryptocurrencies."""
        return self.get_trending('all-crypto', page, pages)

    def get_reddit_stocks(self, page: int = 1, pages: int = 1) -> List[ApewisdomResult]:
        """Get Reddit-trending stocks."""
        return self.get_trending('reddit-stocks', page, pages)

    def get_all_filtered(
        self,
        pages: int = 1,
        include_stocks: bool = True,
        include_crypto: bool = True,
        include_4chan: bool = False,
    ) -> List[ApewisdomResult]:
        """
        Get all trending tickers across multiple filters.

        Args:
            pages: Number of pages per filter
            include_stocks: Include stock tickers
            include_crypto: Include crypto tickers
            include_4chan: Include 4chan data

        Returns:
            Combined list of ApewisdomResult objects
        """
        results = []

        if include_stocks:
            results.extend(self.get_stocks(pages=pages))

        if include_crypto:
            results.extend(self.get_crypto(pages=pages))

        if include_4chan:
            results.extend(self.get_trending('4chan', pages=pages))

        return results

    def get_top_n(
        self,
        n: int = 20,
        asset_type: Optional[str] = None,
    ) -> List[ApewisdomResult]:
        """
        Get top N trending tickers.

        Args:
            n: Number of top tickers to return
            asset_type: 'stock', 'crypto', or None for both

        Returns:
            List of top N ApewisdomResult objects
        """
        if asset_type == 'stock':
            results = self.get_stocks(pages=max(1, n // 100 + 1))
        elif asset_type == 'crypto':
            results = self.get_crypto(pages=max(1, n // 100 + 1))
        else:
            results = self.get_all_filtered(pages=max(1, n // 100 + 1))

        # Sort by mentions descending
        results.sort(key=lambda x: x.mentions, reverse=True)
        return results[:n]


# ============================================================================
# Subreddit Utilities
# ============================================================================

class SubredditFilter:
    """
    Filter and categorize subreddits for targeted data collection.

    Usage:
        filter = SubredditFilter()
        stock_subs = filter.get_subreddits(category=AssetType.STOCK)
        active_subs = filter.get_active_subreddits(min_members=10000)
    """

    def __init__(self):
        """Initialize subreddit filter."""
        self._subreddit_db = self._build_subreddit_db()

    def _build_subreddit_db(self) -> Dict[str, SubredditInfo]:
        """Build subreddit database."""
        db = {}

        for name in STOCK_SUBREDDITS:
            db[name] = SubredditInfo(
                name=name,
                category=AssetType.STOCK,
                is_active=True,
            )

        for name in CRYPTO_SUBREDDITS:
            db[name] = SubredditInfo(
                name=name,
                category=AssetType.CRYPTO,
                is_active=True,
            )

        for name in FINANCE_SUBREDDITS:
            db[name] = SubredditInfo(
                name=name,
                category=AssetType.UNKNOWN,
                is_active=True,
            )

        return db

    def get_subreddits(
        self,
        category: Optional[AssetType] = None,
        active_only: bool = True,
    ) -> List[str]:
        """
        Get list of subreddits.

        Args:
            category: Filter by asset type category
            active_only: Only return active subreddits

        Returns:
            List of subreddit names
        """
        subreddits = []

        for name, info in self._subreddit_db.items():
            if active_only and not info.is_active:
                continue
            if category and info.category != category:
                continue
            subreddits.append(name)

        return sorted(subreddits)

    def get_stock_subreddits(self, active_only: bool = True) -> List[str]:
        """Get stock-related subreddits."""
        return self.get_subreddits(category=AssetType.STOCK, active_only=active_only)

    def get_crypto_subreddits(self, active_only: bool = True) -> List[str]:
        """Get crypto-related subreddits."""
        return self.get_subreddits(category=AssetType.CRYPTO, active_only=active_only)

    def get_finance_subreddits(self, active_only: bool = True) -> List[str]:
        """Get finance-related subreddits."""
        return self.get_subreddits(category=AssetType.UNKNOWN, active_only=active_only)

    def is_valid_subreddit(self, name: str) -> bool:
        """Check if subreddit name is valid."""
        return name in self._subreddit_db

    def get_category(self, subreddit: str) -> AssetType:
        """Get the category of a subreddit."""
        info = self._subreddit_db.get(subreddit.lower())
        return info.category if info else AssetType.UNKNOWN


# ============================================================================
# Sentiment Utilities
# ============================================================================

def classify_sentiment(score: float) -> SentimentCategory:
    """
    Classify sentiment score into category.

    Args:
        score: Sentiment score (-1 to 1)

    Returns:
        SentimentCategory enum value
    """
    if score <= -0.6:
        return SentimentCategory.VERY_BEARISH
    elif score <= -0.2:
        return SentimentCategory.BEARISH
    elif score <= 0.2:
        return SentimentCategory.NEUTRAL
    elif score <= 0.6:
        return SentimentCategory.BULLISH
    else:
        return SentimentCategory.VERY_BULLISH


def normalize_sentiment(score: float) -> float:
    """
    Normalize sentiment score to 0-1 range.

    Args:
        score: Sentiment score (-1 to 1)

    Returns:
        Normalized score (0 to 1)
    """
    return float((score + 1) / 2)


def sentiment_to_signal(score: float, threshold: float = 0.1) -> int:
    """
    Convert sentiment score to trading signal.

    Args:
        score: Sentiment score (-1 to 1)
        threshold: Minimum absolute score for signal

    Returns:
        1 (bullish), -1 (bearish), or 0 (neutral)
    """
    if abs(score) < threshold:
        return 0
    return 1 if score > 0 else -1


# ============================================================================
# Data Validation Utilities
# ============================================================================

def validate_ticker(ticker: str) -> bool:
    """
    Validate stock/crypto ticker symbol.

    Args:
        ticker: Ticker symbol to validate

    Returns:
        True if valid, False otherwise
    """
    if not ticker:
        return False

    ticker = ticker.upper()

    # Basic validation: 1-5 uppercase letters
    if not ticker.isalpha():
        return False
    if len(ticker) < 1 or len(ticker) > 5:
        return False

    # Filter out obvious non-tickers
    invalid = {'I', 'A', 'VS', 'THE', 'AND', 'FOR', 'NOT', 'ARE', 'BUT'}
    if ticker in invalid:
        return False

    return True


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol.

    Args:
        ticker: Raw ticker symbol

    Returns:
        Normalized ticker symbol
    """
    if not ticker:
        return ""

    # Remove common prefixes/suffixes
    ticker = ticker.upper().strip()
    ticker = ticker.replace('$', '')
    ticker = ticker.replace('.', '')

    # Remove exchange suffixes if present
    for suffix in [':NYSE', ':NASDAQ', ':AMEX', ':OTC', ':CRYPTO']:
        if ticker.endswith(suffix):
            ticker = ticker[:-len(suffix)]

    return ticker


def aggregate_mentions(
    mentions_list: List[TickerMention],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate ticker mentions from multiple sources.

    Args:
        mentions_list: List of TickerMention objects

    Returns:
        Dictionary mapping tickers to aggregated data
    """
    aggregated = {}

    for mention in mentions_list:
        ticker = mention.ticker.upper()

        if ticker not in aggregated:
            aggregated[ticker] = {
                'total_mentions': 0,
                'sources': set(),
                'sentiment_scores': [],
                'timestamps': [],
            }

        aggregated[ticker]['total_mentions'] += mention.mentions
        aggregated[ticker]['sources'].add(mention.source)
        aggregated[ticker]['sentiment_scores'].append(mention.sentiment)
        aggregated[ticker]['timestamps'].append(mention.timestamp)

    # Calculate aggregate statistics
    for ticker, data in aggregated.items():
        scores = data['sentiment_scores']
        data['avg_sentiment'] = float(np.mean(scores)) if scores else 0.0
        data['sentiment_std'] = float(np.std(scores)) if len(scores) > 1 else 0.0
        data['source_count'] = len(data['sources'])
        data['sources'] = list(data['sources'])

    return aggregated


# ============================================================================
# Factory Functions
# ============================================================================

def create_apewisdom_client(timeout: int = 10) -> ApewisdomClient:
    """
    Create an Apewisdom client.

    Args:
        timeout: Request timeout in seconds

    Returns:
        ApewisdomClient instance
    """
    return ApewisdomClient(timeout=timeout)


def create_subreddit_filter() -> SubredditFilter:
    """
    Create a SubredditFilter instance.

    Returns:
        SubredditFilter instance
    """
    return SubredditFilter()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("Social Data Utilities - Demo")
    print("=" * 50)

    # Test Apewisdom client
    client = ApewisdomClient()
    print(f"ApewisdomClient created successfully")

    # Test SubredditFilter
    sub_filter = SubredditFilter()
    stock_subs = sub_filter.get_stock_subreddits()
    crypto_subs = sub_filter.get_crypto_subreddits()
    print(f"Stock subreddits: {len(stock_subs)}")
    print(f"Crypto subreddits: {len(crypto_subs)}")

    # Test sentiment utilities
    print("\nSentiment Classification Test:")
    for score in [-0.8, -0.3, 0.0, 0.4, 0.9]:
        category = classify_sentiment(score)
        signal = sentiment_to_signal(score)
        normalized = normalize_sentiment(score)
        print(f"  Score {score:+.1f} -> {category.value}, signal={signal:+d}, normalized={normalized:.2f}")

    # Test ticker validation
    print("\nTicker Validation Test:")
    test_tickers = ["AAPL", "NVDA", "BTC", "XRP", "STOCK", "THE", "123", "AAPL NASDAQ", ""]
    for ticker in test_tickers:
        valid = validate_ticker(ticker)
        normalized = normalize_ticker(ticker)
        print(f"  '{ticker}' -> valid={valid}, normalized='{normalized}'")
