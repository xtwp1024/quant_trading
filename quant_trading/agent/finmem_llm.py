"""FinMem-LLM: Memory-Augmented LLM Trading Agent

Memory-augmented stock trading pipeline inspired by FinMem-LLM-StockTrading.
Provides layered memory (short/mid/long/reflect), REST-only LLM clients
(Gemini / OpenAI / TGI-compatible), and a complete trading agent loop.

Classes
-------
MemoryContext
    Structured context that tracks positions, P&L, market regime, and
    aggregated memory snippets across all temporal layers.
FinMemDataPipeline
    Data ingestion pipeline: fetches stock price, news, and sentiment.
MemoryAugmentedLLM
    REST-only LLM client (urllib) with memory context injection.
    Supports OpenAI-compatible, Gemini (Vertex AI), and TGI endpoints.
FinMemTradingAgent
    Main trading agent that combines MemoryContext + MemoryAugmentedLLM
    to produce buy/sell/hold decisions at each market step.

Pure Python / urllib only / No langchain / No external ML libs required.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
import logging
import threading
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Section 1: MemoryContext
# ---------------------------------------------------------------------------


class MemoryContext:
    """
    Structured memory context for tracking trading state across time.

    Tracks:
    - Current positions (shares held, entry price, unrealised P&L)
    - Cumulative realised P&L
    - Market regime (bull/bear/sideways)
    - Historical decisions with memory-layer tags
    - Short / Mid / Long / Reflection memory buffers (in-memory, no FAISS needed)

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. "AAPL".
    initial_cash : float
        Starting cash (default 100_000).
    lookback_window : int
        Number of days to keep in rolling short-term memory (default 7).
    """

    REGIME_BULL = "bull"
    REGIME_BEAR = "bear"
    REGIME_SIDEWAYS = "sideways"
    REGIME_UNKNOWN = "unknown"

    def __init__(
        self,
        symbol: str,
        initial_cash: float = 100_000.0,
        lookback_window: int = 7,
    ) -> None:
        self.symbol = symbol.upper()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.lookback_window = lookback_window

        # Position state
        self.shares_held = 0
        self.entry_price = 0.0
        self.realised_pnl = 0.0

        # Rolling P&L series (date -> daily pnl)
        self._pnl_series: Dict[str, float] = {}

        # Market regime (default unknown)
        self.market_regime: str = self.REGIME_UNKNOWN

        # Layered memory buffers: each entry is a dict with keys:
        #   text, date, importance, recency, access_count
        self.short_mem: List[Dict[str, Any]] = []   # news / intraday
        self.mid_mem: List[Dict[str, Any]] = []     # filings / weekly
        self.long_mem: List[Dict[str, Any]] = []    # SEC 10-K/10-Q / quarterly
        self.reflect_mem: List[Dict[str, Any]] = []  # agent self-reflection

        # History of decisions
        self.decision_history: List[Dict[str, Any]] = []

        # Auto-increment memory ID
        self._mem_id_counter = 0
        self._lock = threading.Lock()

        # Logging
        self._logger = logging.getLogger(f"MemoryContext[{symbol}]")
        self._logger.setLevel(logging.INFO)

    # ---- Memory management ----

    def _next_id(self) -> int:
        with self._lock:
            cid = self._mem_id_counter
            self._mem_id_counter += 1
            return cid

    def add_short(self, text: str, dt: Optional[date] = None, importance: float = 0.5) -> int:
        """Add a short-term memory entry (news headline, intraday event)."""
        mem_id = self._next_id()
        entry = {
            "id": mem_id,
            "layer": "short",
            "text": text,
            "date": dt or date.today(),
            "importance": importance,
            "recency": 1.0,
            "access_count": 0,
        }
        with self._lock:
            self.short_mem.append(entry)
            if len(self.short_mem) > self.lookback_window * 2:
                self.short_mem = self.short_mem[-self.lookback_window * 2:]
        return mem_id

    def add_mid(self, text: str, dt: Optional[date] = None, importance: float = 0.6) -> int:
        """Add a mid-term memory entry (quarterly filing, weekly summary)."""
        mem_id = self._next_id()
        entry = {
            "id": mem_id,
            "layer": "mid",
            "text": text,
            "date": dt or date.today(),
            "importance": importance,
            "recency": 1.0,
            "access_count": 0,
        }
        with self._lock:
            self.mid_mem.append(entry)
        return mem_id

    def add_long(self, text: str, dt: Optional[date] = None, importance: float = 0.8) -> int:
        """Add a long-term memory entry (annual report, SEC filing)."""
        mem_id = self._next_id()
        entry = {
            "id": mem_id,
            "layer": "long",
            "text": text,
            "date": dt or date.today(),
            "importance": importance,
            "recency": 1.0,
            "access_count": 0,
        }
        with self._lock:
            self.long_mem.append(entry)
        return mem_id

    def add_reflection(self, text: str, dt: Optional[date] = None, importance: float = 0.9) -> int:
        """Add a self-reflection memory entry (agent's own reasoning trace)."""
        mem_id = self._next_id()
        entry = {
            "id": mem_id,
            "layer": "reflection",
            "text": text,
            "date": dt or date.today(),
            "importance": importance,
            "recency": 1.0,
            "access_count": 0,
        }
        with self._lock:
            self.reflect_mem.append(entry)
        return mem_id

    def decay(self, factor: float = 0.95) -> None:
        """Apply recency decay to all memory layers (call each trading day)."""
        with self._lock:
            for mem_list in [self.short_mem, self.mid_mem, self.long_mem, self.reflect_mem]:
                for entry in mem_list:
                    entry["recency"] *= factor

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        layers: Optional[List[str]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Simple keyword-overlap relevance search across memory layers.
        Returns top_k (entry, score) tuples sorted by combined score
        (importance * recency * keyword_overlap).

        Parameters
        ----------
        query_text : str
            Query string to match against memory entries.
        top_k : int
            Maximum number of results to return.
        layers : list of str, optional
            Which layers to search. Defaults to all layers.

        Returns
        -------
        list of (entry, score) tuples.
        """
        if layers is None:
            layers = ["short", "mid", "long", "reflection"]

        query_words = set(query_text.lower().split())
        results: List[Tuple[Dict[str, Any], float]] = []

        with self._lock:
            for layer in layers:
                mem_list = getattr(self, f"{layer}_mem", [])
                for entry in mem_list:
                    entry_words = set(entry["text"].lower().split())
                    overlap = len(query_words & entry_words)
                    if overlap == 0:
                        continue
                    score = (overlap / max(len(query_words), 1)) * entry["importance"] * entry["recency"]
                    results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ---- Position & P&L management ----

    def update_position(
        self,
        price: float,
        direction: int,
        quantity: int = 1,
        dt: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Apply a trade action: direction +1 (buy), -1 (sell), 0 (hold).

        Returns a dict with realised_pnl and position snapshot.
        """
        dt = dt or date.today()
        prev_shares = self.shares_held
        prev_entry = self.entry_price

        if direction > 0:   # BUY
            cost = direction * quantity * price
            self.cash -= cost
            if prev_shares > 0:
                total_shares = prev_shares + quantity
                self.entry_price = (prev_entry * prev_shares + price * quantity) / total_shares
                self.shares_held = total_shares
            else:
                self.shares_held = quantity
                self.entry_price = price

        elif direction < 0:  # SELL (partial or full)
            sell_qty = min(abs(direction) * quantity, prev_shares)
            proceeds = sell_qty * price
            self.cash += proceeds
            self.realised_pnl += sell_qty * (price - prev_entry)
            self.shares_held -= sell_qty
            if self.shares_held == 0:
                self.entry_price = 0.0

        self._pnl_series[str(dt)] = self.unrealised_pnl(price)

        self._logger.info(
            f"[{dt}] action={direction} qty={quantity} price={price} "
            f"shares_held={self.shares_held} cash={self.cash:.2f} pnl={self.realised_pnl:.2f}"
        )
        return {
            "date": dt,
            "direction": direction,
            "price": price,
            "shares_held": self.shares_held,
            "cash": self.cash,
            "realised_pnl": self.realised_pnl,
        }

    def unrealised_pnl(self, current_price: float) -> float:
        """Unrealised P&L at current price."""
        if self.shares_held == 0:
            return 0.0
        return self.shares_held * (current_price - self.entry_price)

    def total_pnl(self, current_price: float) -> float:
        """Total P&L (realised + unrealised)."""
        return self.realised_pnl + self.unrealised_pnl(current_price)

    def update_regime(self, regime: str) -> None:
        """Set detected market regime."""
        if regime in (self.REGIME_BULL, self.REGIME_BEAR, self.REGIME_SIDEWAYS, self.REGIME_UNKNOWN):
            self.market_regime = regime

    def record_decision(
        self,
        decision: str,
        reason: str,
        memory_ids: Optional[List[int]] = None,
        dt: Optional[date] = None,
    ) -> None:
        """Append a trading decision to history."""
        self.decision_history.append({
            "date": dt or date.today(),
            "decision": decision,
            "reason": reason,
            "memory_ids": memory_ids or [],
            "position_snapshot": {
                "shares_held": self.shares_held,
                "entry_price": self.entry_price,
                "cash": self.cash,
                "realised_pnl": self.realised_pnl,
            },
        })

    # ---- Context injection for LLM prompts ----

    def build_context_str(
        self,
        current_price: float,
        current_date: Optional[date] = None,
        include_reflection: bool = True,
    ) -> str:
        """
        Build a human-readable context string for LLM prompt injection.
        """
        lines = []
        dt = current_date or date.today()

        lines.append(f"== MARKET CONTEXT [{dt}] ==")
        lines.append(f"Symbol: {self.symbol}")
        lines.append(f"Current Price: ${current_price:.2f}")
        lines.append(f"Cash: ${self.cash:.2f}")
        lines.append(f"Shares Held: {self.shares_held}")
        lines.append(f"Entry Price: ${self.entry_price:.2f}")
        lines.append(f"Realised P&L: ${self.realised_pnl:.2f}")
        lines.append(f"Unrealised P&L: ${self.unrealised_pnl(current_price):.2f}")
        lines.append(f"Total P&L: ${self.total_pnl(current_price):.2f}")
        lines.append(f"Market Regime: {self.market_regime}")

        if self.short_mem:
            lines.append("\n-- SHORT-TERM MEMORY (news / recent events) --")
            for e in self.short_mem[-5:]:
                lines.append(f"  [{e['date']}] (importance={e['importance']:.2f}) {e['text']}")

        if self.mid_mem:
            lines.append("\n-- MID-TERM MEMORY (filings / weekly) --")
            for e in self.mid_mem[-3:]:
                lines.append(f"  [{e['date']}] {e['text']}")

        if self.long_mem:
            lines.append("\n-- LONG-TERM MEMORY (SEC filings / annual) --")
            for e in self.long_mem[-2:]:
                lines.append(f"  [{e['date']}] {e['text']}")

        if include_reflection and self.reflect_mem:
            lines.append("\n-- REFLECTION MEMORY (agent reasoning) --")
            for e in self.reflect_mem[-3:]:
                lines.append(f"  [{e['date']}] {e['text']}")

        return "\n".join(lines)

    def summary(self, current_price: float) -> Dict[str, Any]:
        """Return a full state dict."""
        return {
            "symbol": self.symbol,
            "date": str(date.today()),
            "current_price": current_price,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "entry_price": self.entry_price,
            "realised_pnl": self.realised_pnl,
            "unrealised_pnl": self.unrealised_pnl(current_price),
            "total_pnl": self.total_pnl(current_price),
            "market_regime": self.market_regime,
            "short_mem_count": len(self.short_mem),
            "mid_mem_count": len(self.mid_mem),
            "long_mem_count": len(self.long_mem),
            "reflect_mem_count": len(self.reflect_mem),
            "decision_count": len(self.decision_history),
        }


# ---------------------------------------------------------------------------
# Section 2: FinMemDataPipeline
# ---------------------------------------------------------------------------

class FinMemDataPipeline:
    """
    Data ingestion pipeline for stock trading memory augmentation.

    Fetches / produces:
    - Price OHLCV (via yfinance or direct REST)
    - News headlines (mock; real implementation would call news API)
    - Basic sentiment estimation (keyword-based)

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. "AAPL".
    provider : str, default "yfinance"
        Data provider. Currently "yfinance" or "mock".
    """

    def __init__(
        self,
        symbol: str,
        provider: str = "yfinance",
    ) -> None:
        self.symbol = symbol.upper()
        self.provider = provider.lower()
        self._logger = logging.getLogger(f"FinMemDataPipeline[{self.symbol}]")
        self._logger.setLevel(logging.INFO)

    # ---- Price data ----

    def fetch_price(
        self,
        target_date: Optional[date],
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Fetch OHLCV price data for target_date.

        Returns a dict with keys: open, high, low, close, volume, date.
        """
        if target_date is None:
            target_date = date.today()
        start = target_date - timedelta(days=lookback_days * 2)
        end = target_date + timedelta(days=1)

        try:
            import yfinance
        except ImportError:
            self._logger.warning("yfinance not installed, using mock price data")
            return self._mock_price(target_date)

        ticker = yfinance.Ticker(self.symbol)
        hist = ticker.history(start=start, end=end)

        if hist.empty:
            self._logger.warning(f"No price data for {target_date}, using mock")
            return self._mock_price(target_date)

        idx = hist.index.astype(str)
        target_str = str(target_date)
        if target_str in idx:
            row = hist.loc[target_str]
        else:
            mask = idx < target_str
            if not mask.any():
                row = hist.iloc[-1]
            else:
                row = hist.loc[mask].iloc[-1]

        return {
            "date": target_date,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
        }

    def _mock_price(self, target_date: date) -> Dict[str, Any]:
        """Generate synthetic price data for testing."""
        import random
        base = 150.0 + random.uniform(-5, 5)
        return {
            "date": target_date,
            "open": round(base - 0.5, 2),
            "high": round(base + 1.0, 2),
            "low": round(base - 1.0, 2),
            "close": round(base, 2),
            "volume": random.randint(5_000_000, 20_000_000),
        }

    # ---- News data ----

    def fetch_news(
        self,
        target_date: Optional[date],
        max_headlines: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fetch news headlines for target_date.

        Returns a list of dicts: {headline, url, published_at, source}.
        Falls back to mock data if no API key is available.
        """
        if target_date is None:
            target_date = date.today()

        # Try Alpha Vantage News API if API key present
        av_key = os.environ.get("ALPHA_VANTAGE_KEY") or os.environ.get("NEWS_API_KEY")
        if av_key:
            return self._fetch_alpha_vantage_news(target_date, max_headlines, av_key)

        # Try Finnhub if key present
        fh_key = os.environ.get("FINNHUB_KEY")
        if fh_key:
            return self._fetch_finnhub_news(target_date, max_headlines, fh_key)

        self._logger.info("No news API key found, using mock news")
        return self._mock_news(target_date, max_headlines)

    def _fetch_alpha_vantage_news(
        self,
        target_date: date,
        max_headlines: int,
        api_key: str,
    ) -> List[Dict[str, Any]]:
        """Fetch via Alpha Vantage News API."""
        import urllib.request

        url = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&ticker={self.symbol}"
            f"&limit={max_headlines}&apikey={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            self._logger.warning(f"Alpha Vantage request failed: {exc}")
            return self._mock_news(target_date, max_headlines)

        articles = data.get("feed", [])[:max_headlines]
        result = []
        for art in articles:
            result.append({
                "headline": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("source", ""),
                "published_at": art.get("time_published", ""),
                "sentiment_score": float(art.get("overall_sentiment_score", 0.0)),
                "sentiment_label": art.get("sentiment_label", "neutral"),
            })
        return result

    def _fetch_finnhub_news(
        self,
        target_date: date,
        max_headlines: int,
        api_key: str,
    ) -> List[Dict[str, Any]]:
        """Fetch via Finnhub News API."""
        import urllib.request

        from_str = (target_date - timedelta(days=3)).isoformat()
        to_str = target_date.isoformat()
        url = (
            f"https://finnhub.io/api/v1/news"
            f"?category=general&from={from_str}&to={to_str}"
            f"&token={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            self._logger.warning(f"Finnhub request failed: {exc}")
            return self._mock_news(target_date, max_headlines)

        result = []
        for art in data[:max_headlines]:
            result.append({
                "headline": art.get("headline", ""),
                "url": art.get("url", ""),
                "source": art.get("source", ""),
                "published_at": art.get("datetime", ""),
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
            })
        return result

    def _mock_news(self, target_date: date, max_headlines: int) -> List[Dict[str, Any]]:
        """Generate synthetic news headlines for testing."""
        import random
        templates = [
            "{symbol} reports quarterly earnings per share of ${eps}, beating estimates.",
            "Analyst upgrades {symbol} to Buy with ${price} target.",
            "{symbol} announces ${amount}B stock buyback programme.",
            "Regulatory review of {symbol} latest product launch underway.",
            "Inflation data impacts {symbol} Q{q} guidance downward.",
            "{symbol} partners with {partner} on AI-driven supply chain.",
            "Short interest in {symbol} rises to {pct}% of float.",
            "{symbol} board declares quarterly dividend of ${div}.",
        ]
        q = (target_date.month - 1) // 3 + 1
        results = []
        for i in range(min(max_headlines, len(templates))):
            tpl = templates[i % len(templates)]
            headline = tpl.format(
                symbol=self.symbol,
                q=q,
                eps=round(random.uniform(1.0, 4.0), 2),
                price=round(random.uniform(160, 220), 0),
                amount=round(random.uniform(1, 10), 1),
                partner=random.choice(["Microsoft", "Google", "Amazon", "Nvidia"]),
                pct=round(random.uniform(3, 12), 1),
                div=round(random.uniform(0.2, 0.9), 2),
            )
            results.append({
                "headline": headline,
                "url": "https://example.com/mock",
                "source": "MockSource",
                "published_at": target_date.isoformat(),
                "sentiment_score": round(random.uniform(-0.5, 0.5), 3),
                "sentiment_label": "neutral",
            })
        return results

    # ---- Sentiment analysis (keyword-based) ----

    POSITIVE_WORDS = {
        "beat", "beats", "upgrade", "buy", "bullish", "growth", "soar",
        "surge", "rally", "record", "profit", "dividend", "buyback",
        "beat estimates", "exceeds", "outperform", "strong",
    }
    NEGATIVE_WORDS = {
        "miss", "misses", "downgrade", "sell", "bearish", "loss",
        "plunge", "crash", "risk", "lawsuit", "investigation",
        "recall", "fraud", "bankruptcy", "cut", "warn", "weak",
    }

    def estimate_sentiment(self, headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate sentiment from a list of headlines.

        Returns dict with keys: score (-1..1), label, positive_count,
        negative_count, neutral_count.
        """
        pos = neg = neu = 0
        total_score = 0.0
        for h in headlines:
            text = h.get("headline", "").lower()
            if "sentiment_score" in h:
                total_score += h["sentiment_score"]
            p_count = sum(1 for w in self.POSITIVE_WORDS if w in text)
            n_count = sum(1 for w in self.NEGATIVE_WORDS if w in text)
            if p_count > n_count:
                pos += 1
            elif n_count > p_count:
                neg += 1
            else:
                neu += 1

        count = len(headlines) or 1
        score = total_score / count if total_score != 0 else (pos - neg) / count
        score = max(-1.0, min(1.0, score))
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return {
            "score": round(score, 4),
            "label": label,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
        }

    # ---- Main pipeline: price + news + sentiment ----

    def ingest(
        self,
        target_date: Optional[date],
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Run the full data pipeline for target_date.

        Returns a combined dict:
            date, symbol, price_data, news (list), sentiment (dict).
        """
        if target_date is None:
            target_date = date.today()
        price_data = self.fetch_price(target_date, lookback_days)
        news = self.fetch_news(target_date)
        sentiment = self.estimate_sentiment(news)
        self._logger.info(
            f"[{target_date}] price=${price_data['close']:.2f} "
            f"sentiment={sentiment['label']}({sentiment['score']:.3f}) "
            f"news={len(news)} headlines"
        )
        return {
            "date": target_date,
            "symbol": self.symbol,
            "price_data": price_data,
            "news": news,
            "sentiment": sentiment,
        }

    # ---- Convenience: format news as memory strings ----

    @staticmethod
    def news_to_memory_strings(
        news: List[Dict[str, Any]],
        sentiment: Dict[str, Any],
    ) -> List[str]:
        """Convert raw news list into human-readable strings for memory storage."""
        lines = []
        label = sentiment.get("label", "neutral")
        score = sentiment.get("score", 0.0)
        lines.append(f"Market sentiment: {label} ({score:.3f}).")
        for art in news[:5]:
            headline = art.get("headline", "").strip()
            source = art.get("source", "unknown")
            if headline:
                lines.append(f"[{source}] {headline}")
        return lines


# ---------------------------------------------------------------------------
# Section 3: MemoryAugmentedLLM
# ---------------------------------------------------------------------------

class LLMCallError(Exception):
    """Raised when an LLM REST call fails or returns an unparseable response."""
    pass


class MemoryAugmentedLLM:
    """
    REST-only LLM client with memory-context injection.
    Supports OpenAI-compatible, Google Gemini (Vertex AI), and
    Text Generation Inference (TGI / HuggingFace) endpoints.

    Built with urllib only - no httpx, no langchain.

    Parameters
    ----------
    provider : str
        One of "openai", "gemini", "tgi".
    model : str
        Model name / identifier.
    endpoint : str, optional
        Full REST URL. If not provided, uses public defaults.
    api_key : str, optional
        API key. Falls back to env vars:
        OPENAI_API_KEY, GEMINI_API_KEY / GOOGLE_API_KEY, HF_TOKEN.
    system_message : str, optional
        System-prompt string (injected as the first system turn).
    timeout : int, default 60
        Request timeout in seconds.
    max_tokens : int, default 512
        Maximum tokens in generated response.
    temperature : float, default 0.3
        Sampling temperature.
    """

    PROVIDER_OPENAI = "openai"
    PROVIDER_GEMINI = "gemini"
    PROVIDER_TGI = "tgi"

    def __init__(
        self,
        provider: str,
        model: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: str = "You are a helpful financial analysis assistant.",
        timeout: int = 60,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.endpoint = endpoint or self._default_endpoint()
        self.api_key = api_key or self._resolve_api_key()
        self.system_message = system_message
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

        self._logger = logging.getLogger(f"MemoryAugmentedLLM[{provider}:{model}]")
        self._logger.setLevel(logging.INFO)

    # ---- API key resolution ----

    def _resolve_api_key(self) -> str:
        if self.provider == self.PROVIDER_OPENAI:
            return os.environ.get("OPENAI_API_KEY", "-")
        elif self.provider == self.PROVIDER_GEMINI:
            return os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "-"))
        elif self.provider == self.PROVIDER_TGI:
            return os.environ.get("HF_TOKEN", "-")
        return "-"

    def _default_endpoint(self) -> str:
        if self.provider == self.PROVIDER_OPENAI:
            return "https://api.openai.com/v1/chat/completions"
        elif self.provider == self.PROVIDER_GEMINI:
            return "https://api.googleusercontent.com/v1/models"
        elif self.provider == self.PROVIDER_TGI:
            return "http://localhost:8080/completion"
        return ""

    # ---- Prompt construction ----

    @staticmethod
    def build_llama2_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Convert a list of {role, content} dicts to a Llama-2 chat prompt string.
        """
        start = "<s>[INST] "
        end = " [/INST]"
        parts = []
        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg["content"]
            if role == "system" and idx == 0:
                parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                parts.append(content.strip())
            else:
                parts.append(f" [/INST] {content.strip()}</s><s>[INST] ")
        return start + "".join(parts) + end

    def build_prompt(
        self,
        user_content: str,
        memory_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build a messages list for the LLM.

        Parameters
        ----------
        user_content : str
            The main user prompt.
        memory_context : str, optional
            Pre-pended memory context string (from MemoryContext.build_context_str).

        Returns
        -------
        List of {role, content} dicts.
        """
        system = self.system_message
        if memory_context:
            system += f"\n\n[MEMORY CONTEXT]\n{memory_context}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    # ---- REST calls via urllib ----

    def _make_openai_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        return result

    def _make_gemini_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call Google Gemini via Vertex AI REST API."""
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        return result

    def _make_tgi_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        return result

    # ---- Response parsing ----

    def _parse_openai_response(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]

    def _parse_gemini_response(self, response: Dict[str, Any]) -> str:
        # Vertex AI structure
        preds = response.get("predictions", [])
        if preds:
            return preds[0].get("content", str(response))
        # Alternative: direct Gemini API
        candidates = response.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", str(response))
        return str(response)

    def _parse_tgi_response(self, response: Dict[str, Any]) -> str:
        return response.get("generated_text", "")

    # ---- Main call ----

    def call(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        memory_context: Optional[str] = None,
        json_mode: bool = False,
    ) -> str:
        """
        Make a synchronous LLM call.

        Parameters
        ----------
        prompt : str or list of {role, content} dicts
            User content. If a string, combined with memory_context into a messages list.
        memory_context : str, optional
            Memory context string to prepend.
        json_mode : bool, default False
            If True, request JSON-formatted responses.

        Returns
        -------
        str
            The LLM's response text.

        Raises
        ------
        LLMCallError
        """
        # Normalise prompt to messages list
        if isinstance(prompt, str):
            messages = self.build_prompt(prompt, memory_context)
        else:
            messages = prompt
            if memory_context:
                messages = [
                    {"role": "system", "content": self.system_message + "\n\n[MEMORY CONTEXT]\n" + memory_context}
                ] + messages

        try:
            if self.provider == self.PROVIDER_OPENAI:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                response = self._make_openai_request(payload)
                return self._parse_openai_response(response)

            elif self.provider == self.PROVIDER_GEMINI:
                contents = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    parts = [{"text": msg["content"]}]
                    contents.append({"role": role, "parts": parts})

                payload = {
                    "instances": [{"prompt": messages[-1]["content"]}],
                    "parameters": {
                        "temperature": self.temperature,
                        "maxOutputTokens": self.max_tokens,
                    },
                }
                response = self._make_gemini_request(payload)
                return self._parse_gemini_response(response)

            elif self.provider == self.PROVIDER_TGI:
                prompt_str = self.build_llama2_prompt(messages)
                payload = {
                    "inputs": prompt_str,
                    "parameters": {
                        "temperature": self.temperature,
                        "max_new_tokens": self.max_tokens,
                        "do_sample": True,
                        "top_p": 0.9,
                    },
                }
                response = self._make_tgi_request(payload)
                return self._parse_tgi_response(response)

            else:
                raise LLMCallError(f"Unknown provider: {self.provider}")

        except urllib.error.HTTPError as exc:
            self._logger.error(f"HTTP error {exc.code}: {exc.reason}")
            raise LLMCallError(f"HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            self._logger.error(f"URL error: {exc.reason}")
            raise LLMCallError(f"URL error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            self._logger.error(f"JSON decode error: {exc}")
            raise LLMCallError(f"JSON decode error: {exc}") from exc
        except Exception as exc:
            self._logger.error(f"Unexpected error: {exc}")
            raise LLMCallError(str(exc)) from exc

    # ---- Convenience: decision prompt ----

    def build_decision_prompt(
        self,
        symbol: str,
        current_price: float,
        decision_type: str = "trade",
    ) -> str:
        """
        Return a standard trade-decision user prompt.
        """
        if decision_type == "trade":
            return (
                f"You are a professional stock trading analyst for {symbol}.\n"
                f"Current price: ${current_price:.2f}.\n"
                f"Analyze the provided memory context and produce a trading decision.\n"
                f"Output ONLY a valid JSON object with keys:\n"
                f'{{"decision": "buy"|"sell"|"hold", "confidence": 0.0-1.0, "reason": "<brief reason>"}}\n'
                f"Provide exactly one of: buy, sell, or hold."
            )
        elif decision_type == "reflect":
            return (
                f"Given the trading history and memory context for {symbol}, "
                f"provide a brief self-reflection on recent decisions and any patterns observed.\n"
                f"Output ONLY a valid JSON:\n"
                f'{{"reflection": "<2-3 sentence self-reflection>", "insight": "<actionable insight>"}}'
            )
        return ""


# ---------------------------------------------------------------------------
# Section 4: FinMemTradingAgent
# ---------------------------------------------------------------------------

class FinMemTradingAgent:
    """
    Main memory-augmented trading agent.

    Combines:
    - MemoryContext  : structured position & P&L tracking
    - FinMemDataPipeline : price / news / sentiment ingestion
    - MemoryAugmentedLLM : REST LLM with memory-context injection

    Produces buy/sell/hold decisions with full reasoning traces stored in
    reflection memory.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. "AAPL".
    llm : MemoryAugmentedLLM
        Initialised LLM client.
    initial_cash : float, default 100_000
        Starting cash.
    lookback_window : int, default 7
        Days of short-term memory to retain.
    memory_top_k : int, default 5
        How many memory entries to inject into prompts.
    position_size : int, default 1
        Number of shares per trade unit.
    """

    DECISION_BUY = "buy"
    DECISION_SELL = "sell"
    DECISION_HOLD = "hold"

    def __init__(
        self,
        symbol: str,
        llm: MemoryAugmentedLLM,
        initial_cash: float = 100_000.0,
        lookback_window: int = 7,
        memory_top_k: int = 5,
        position_size: int = 1,
    ) -> None:
        self.symbol = symbol.upper()
        self.llm = llm
        self.position_size = position_size

        self.memory = MemoryContext(
            symbol=self.symbol,
            initial_cash=initial_cash,
            lookback_window=lookback_window,
        )
        self.pipeline = FinMemDataPipeline(symbol=self.symbol)
        self.memory_top_k = memory_top_k

        self._logger = logging.getLogger(f"FinMemTradingAgent[{self.symbol}]")
        self._logger.setLevel(logging.INFO)

        self._step_count = 0

    # ---- Core step ----

    def step(
        self,
        target_date: Optional[date] = None,
        auto_ingest: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute one trading step for target_date.

        1. Ingest data (price, news, sentiment) via FinMemDataPipeline.
        2. Update memory: news -> short_mem, sentiment -> short_mem.
        3. Query layered memory for context.
        4. Call LLM with memory-injected prompt.
        5. Parse decision (buy/sell/hold).
        6. Apply trade to MemoryContext.
        7. Record decision and (optionally) generate self-reflection.

        Parameters
        ----------
        target_date : date, optional
            Date to execute the step for. Defaults to today.
        auto_ingest : bool, default True
            If True, automatically ingest pipeline data.

        Returns
        -------
        dict with keys: date, decision, reason, confidence,
                        price_data, sentiment, memory_snapshot.
        """
        self._step_count += 1
        dt = target_date or date.today()
        self._logger.info(f"=== Step {self._step_count} [{dt}] ===")

        # -- Step 1: Ingest data --
        if auto_ingest:
            ingestion = self.pipeline.ingest(dt)
        else:
            ingestion = {
                "date": dt,
                "symbol": self.symbol,
                "price_data": self.pipeline.fetch_price(dt),
                "news": self.pipeline.fetch_news(dt),
                "sentiment": self.pipeline.estimate_sentiment(
                    self.pipeline.fetch_news(dt)
                ),
            }

        price_data = ingestion["price_data"]
        news = ingestion["news"]
        sentiment = ingestion["sentiment"]
        current_price = price_data["close"]

        # -- Step 2: Update memory --
        news_strings = FinMemDataPipeline.news_to_memory_strings(news, sentiment)
        for ns in news_strings:
            self.memory.add_short(ns, dt, importance=abs(sentiment["score"]) * 0.5 + 0.25)

        self._update_regime(price_data, sentiment)

        # -- Step 3: Query layered memory --
        query_text = " ".join(ns[:80] for ns in news_strings[:3])
        memory_results = self.memory.query(query_text, top_k=self.memory_top_k)
        memory_context_str = self.memory.build_context_str(current_price, dt)

        # -- Step 4: LLM decision --
        decision_prompt = self.llm.build_decision_prompt(
            self.symbol, current_price, decision_type="trade"
        )
        try:
            raw_response = self.llm.call(
                prompt=decision_prompt,
                memory_context=memory_context_str,
                json_mode=True,
            )
            parsed = self._parse_decision_response(raw_response)
        except LLMCallError as exc:
            self._logger.warning(f"LLM call failed, defaulting to hold: {exc}")
            parsed = {"decision": self.DECISION_HOLD, "confidence": 0.0, "reason": f"LLM error: {exc}"}

        decision = parsed.get("decision", self.DECISION_HOLD)
        confidence = parsed.get("confidence", 0.0)
        reason = parsed.get("reason", "")

        # -- Step 5: Apply trade --
        dir_map = {self.DECISION_BUY: 1, self.DECISION_SELL: -1, self.DECISION_HOLD: 0}
        direction = dir_map.get(decision, 0)

        trade_result = self.memory.update_position(
            price=current_price,
            direction=direction,
            quantity=self.position_size,
            dt=dt,
        )

        # -- Step 6: Record decision --
        memory_ids = [r[0]["id"] for r in memory_results]
        self.memory.record_decision(decision, reason, memory_ids, dt)

        # -- Step 7: Self-reflection (every 5 steps) --
        reflection_text = ""
        if self._step_count % 5 == 0:
            reflection_text = self._generate_reflection(current_price, dt)

        self._logger.info(
            f"[{dt}] Decision: {decision} (conf={confidence:.2f}) "
            f"price=${current_price:.2f} pnl=${self.memory.total_pnl(current_price):.2f}"
        )

        return {
            "date": dt,
            "step": self._step_count,
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "direction": direction,
            "price_data": price_data,
            "sentiment": sentiment,
            "trade_result": trade_result,
            "memory_snapshot": self.memory.summary(current_price),
            "reflection": reflection_text,
            "memory_top_k": [(r[0]["layer"], r[0]["id"], r[1]) for r in memory_results],
        }

    # ---- Regime detection ----

    def _update_regime(
        self,
        price_data: Dict[str, Any],
        sentiment: Dict[str, Any],
    ) -> None:
        """Detect market regime based on recent price change and sentiment."""
        close = price_data["close"]
        open_ = price_data["open"]
        day_return = (close - open_) / open_ if open_ else 0.0

        if day_return > 0.01 and sentiment["score"] > 0.1:
            regime = MemoryContext.REGIME_BULL
        elif day_return < -0.01 and sentiment["score"] < -0.1:
            regime = MemoryContext.REGIME_BEAR
        else:
            regime = MemoryContext.REGIME_SIDEWAYS
        self.memory.update_regime(regime)

    # ---- Response parsing ----

    def _parse_decision_response(self, raw: str) -> Dict[str, Any]:
        """
        Parse LLM raw response into a decision dict.

        Tries JSON first, then falls back to keyword extraction.
        """
        raw = raw.strip()
        # Strip markdown code fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = raw.rstrip("`").strip()

        try:
            parsed = json.loads(raw)
            if "decision" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # Keyword fallback
        lower = raw.lower()
        decision = self.DECISION_HOLD
        confidence = 0.5
        reason = raw[:200]

        if '"buy"' in lower or '"sell"' in lower or '"hold"' in lower:
            for kw in ['"buy"', '"sell"', '"hold"']:
                if kw in lower:
                    decision = kw.strip('"')
                    break
        elif "buy" in lower and "sell" not in lower:
            decision = self.DECISION_BUY
        elif "sell" in lower:
            decision = self.DECISION_SELL

        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
        }

    # ---- Self-reflection ----

    def _generate_reflection(self, current_price: float, dt: date) -> str:
        """Ask the LLM to generate a self-reflection and store it."""
        history = self.memory.decision_history[-5:]
        if not history:
            return ""

        hist_str = "\n".join(
            f"- {h['date']}: {h['decision']} (conf={h.get('confidence', 0):.2f}) - {h['reason'][:80]}"
            for h in history
        )
        prompt = (
            f"You are a quant agent for {self.symbol}. "
            f"Recent decisions:\n{hist_str}\n"
            f"Current price: ${current_price:.2f}. "
            f"Total P&L: ${self.memory.total_pnl(current_price):.2f}. "
            f"Market regime: {self.memory.market_regime}.\n"
            f"Provide a brief JSON reflection:\n"
            f'{{"reflection": "<2-3 sentence self-reflection>", "insight": "<actionable insight>"}}'
        )

        try:
            raw = self.llm.call(prompt, json_mode=True)
            parsed = json.loads(raw.strip())
            text = f"Reflection: {parsed.get('reflection', '')} Insight: {parsed.get('insight', '')}"
            self.memory.add_reflection(text, dt)
            return text
        except Exception as exc:
            self._logger.warning(f"Reflection generation failed: {exc}")
            return ""

    # ---- Batch / simulation run ----

    def run_backtest(
        self,
        start_date: date,
        end_date: date,
        step_days: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Run a backtest over a date range.

        Parameters
        ----------
        start_date : date
        end_date : date
        step_days : int, default 1
            Number of calendar days between steps.

        Returns
        -------
        List of step result dicts (same as step()).
        """
        results = []
        current = start_date
        while current <= end_date:
            result = self.step(target_date=current)
            results.append(result)
            current += timedelta(days=step_days)
            self.memory.decay()
        return results

    # ---- State summary ----

    def state_summary(self) -> Dict[str, Any]:
        """Return a human-readable agent state snapshot."""
        return self.memory.summary(0.0)
