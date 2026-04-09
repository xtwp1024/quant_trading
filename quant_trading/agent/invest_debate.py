"""Investment Debate Agent System — Bull vs Bear Multi-Agent Framework.

多智能体投资辩论系统 — 多头vs空头辩论架构.

Integrates with existing DebateEngine (Wave 5) from quant_trading.agent.debate_engine.
Provides Analyst Team (Market/Sentiment/News/Fundamentals) + Bull/Bear Researchers
+ Portfolio Manager for final investment decisions.

Architecture:
    AnalystTeam
        MarketAnalyst       — 技术面分析 (K线/均线/RSI/MACD/布林带)
        SentimentAnalyst    — 情绪分析 (内部交易+新闻舆情)
        NewsAnalyst         — 新闻分析 (财经新闻LLM情感分类)
        FundamentalsAnalyst — 基本面分析 (财务指标/估值/成长性)
    BullResearcher / BearResearcher  — 多头/空头辩论研究
    PortfolioManager        — 组合经理, 最终决策权威

Workflow:
    1. Four analysts run in parallel, each producing an AnalysisReport
    2. BullResearcher and BearResearcher debate using structured claims
    3. PortfolioManager synthesizes all output into a final TradingDecision
"""

from __future__ import annotations

import os
import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

__all__ = [
    # Core data classes
    "AnalysisReport",
    "TradingDecision",
    # Analyst agents
    "AnalystAgent",
    "MarketAnalyst",
    "SentimentAnalyst",
    "NewsAnalyst",
    "FundamentalsAnalyst",
    # Bull/Bear researchers (integrates with DebateEngine)
    "BullResearcher",
    "BearResearcher",
    # Portfolio Manager
    "PortfolioManager",
    # Utilities
    "build_llm_client",
    "SAFE_LLM_CALL",
]


# ---------------------------------------------------------------------------
# REST-only LLM client (no heavy SDK)
# ---------------------------------------------------------------------------


def build_llm_client(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    timeout: int = 30,
) -> Callable[[str], str]:
    """Build a simple REST-only LLM call function.

    Returns a callable that takes a prompt string and returns the LLM response.
    Uses urllib.request (stdlib only) — no heavy SDK dependency.

    Args:
        api_base: LLM API base URL. Defaults to LLM_API_BASE env var.
        api_key: API key. Defaults to OPENAI_API_KEY env var.
        model: Model name. Defaults to gpt-4o-mini.
        timeout: Request timeout in seconds.

    Returns:
        A callable LLM client function(prompt: str) -> str
    """
    _api_base = api_base or os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")
    _api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    _model = model
    _timeout = timeout

    def _call(prompt: str, temperature: float = 0.7) -> str:
        endpoint = f"{_api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": _model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if _api_key:
            headers["Authorization"] = f"Bearer {_api_key}"
        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception:
            return ""

    return _call


# Default safe LLM call — can be replaced with a real client
SAFE_LLM_CALL: Optional[Callable[[str], str]] = None


def _default_llm(prompt: str) -> str:
    """Default LLM call — returns empty string (stub).

    Replace with real client via PortfolioManager(llm=...) or set SAFE_LLM_CALL.
    """
    if SAFE_LLM_CALL is not None:
        return SAFE_LLM_CALL(prompt)
    return ""


# ---------------------------------------------------------------------------
# AnalysisReport — structured analyst output
# ---------------------------------------------------------------------------


@dataclass
class AnalysisReport:
    """分析报告 / Analyst report.

    Attributes:
        market_view: 市场观点 — 'bullish' | 'bearish' | 'neutral'
        sentiment: 情绪分数 — [-1, +1], negative=bearish, positive=bullish
        fundamentals: 基本面文字描述
        news_impact: 新闻影响描述
        confidence: 置信度 — [0, 1]
        supporting_evidence: 支撑证据列表
        opposing_evidence: 反向证据列表
        agent_id: 生成该报告的智能体ID
        timestamp: 生成时间戳
    """

    market_view: str  # 'bullish' | 'bearish' | 'neutral'
    sentiment: float  # [-1, +1]
    fundamentals: str
    news_impact: str
    confidence: float  # [0, 1]
    supporting_evidence: list[str] = field(default_factory=list)
    opposing_evidence: list[str] = field(default_factory=list)
    agent_id: str = "analyst"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_view": self.market_view,
            "sentiment": round(self.sentiment, 4),
            "fundamentals": self.fundamentals,
            "news_impact": self.news_impact,
            "confidence": round(self.confidence, 4),
            "supporting_evidence": self.supporting_evidence,
            "opposing_evidence": self.opposing_evidence,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalysisReport":
        ts = d.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            ts = datetime.now()
        return cls(
            market_view=d.get("market_view", "neutral"),
            sentiment=float(d.get("sentiment", 0.0)),
            fundamentals=d.get("fundamentals", ""),
            news_impact=d.get("news_impact", ""),
            confidence=float(d.get("confidence", 0.0)),
            supporting_evidence=d.get("supporting_evidence", []),
            opposing_evidence=d.get("opposing_evidence", []),
            agent_id=d.get("agent_id", "analyst"),
            timestamp=ts,
        )


# ---------------------------------------------------------------------------
# TradingDecision — portfolio manager output
# ---------------------------------------------------------------------------


@dataclass
class TradingDecision:
    """交易决策 / Trading decision.

    Attributes:
        action: 交易动作 — 'buy' | 'sell' | 'hold'
        quantity: 交易数量 (shares)
        entry_price: 入场价格 (None if hold)
        stop_loss: 止损价格 (None if hold)
        take_profit: 止盈价格 (None if hold)
        confidence: 决策置信度 — [0, 1]
        reasoning: 决策理由 (中英文)
        supporting_reports: 支撑该决策的分析报告列表
        risk_assessment: 风险评估文字描述
    """

    action: str  # 'buy' | 'sell' | 'hold'
    quantity: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0  # [0, 1]
    reasoning: str = ""
    supporting_reports: list[AnalysisReport] = field(default_factory=list)
    risk_assessment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "supporting_reports": [r.to_dict() for r in self.supporting_reports],
            "risk_assessment": self.risk_assessment,
        }


# ---------------------------------------------------------------------------
# AnalystAgent — base class for all analysts
# ---------------------------------------------------------------------------


class AnalystAgent:
    """分析师Agent基类 / Base class for analyst agents.

    All analyst agents inherit from this class and implement the
    `analyze` method to produce an AnalysisReport for a given symbol.

    Args:
        agent_id: Unique identifier for this agent instance.
        role: Human-readable role name (e.g., 'Market Technician').
        llm: Optional LLM call function. Defaults to _default_llm stub.
    """

    DEFAULT_LLM: Callable[[str], str] = _default_llm

    def __init__(
        self,
        agent_id: str,
        role: str,
        llm: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self._llm = llm or self.DEFAULT_LLM

    def analyze(self, symbol: str, data: dict[str, Any]) -> AnalysisReport:
        """分析给定标的并返回报告 / Analyze a symbol and return a report.

        Subclasses must override this method.

        Args:
            symbol: Ticker symbol, e.g. 'AAPL'.
            data: Market/financial data dictionary. Keys vary by subclass:
                - MarketAnalyst: {'prices': pd.DataFrame, ...}
                - SentimentAnalyst: {'insider_trades': [...], 'news': [...]}
                - NewsAnalyst: {'articles': [...]}
                - FundamentalsAnalyst: {'financials': [...]}

        Returns:
            AnalysisReport instance.
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    def llm_call(self, prompt: str, temperature: float = 0.7) -> str:
        """Call the LLM with a prompt / 使用LLM生成内容."""
        return self._llm(prompt)

    def _sentiment_signal(self, signal: str) -> float:
        """Map 'bullish'/'bearish'/'neutral' to [-1, +1]."""
        mapping = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
        return mapping.get(signal.lower(), 0.0)


# ---------------------------------------------------------------------------
# MarketAnalyst — technical analysis
# ---------------------------------------------------------------------------


class MarketAnalyst(AnalystAgent):
    """市场技术分析师 / Market technical analyst.

    Analyzes price data using multiple technical strategies:
    - Trend Following (EMA crossover, ADX)
    - Mean Reversion (Bollinger Bands, RSI, Z-score)
    - Momentum (price momentum, volume momentum)
    - Volatility Regime (ATR, volatility z-score)

    Data required in `data` dict:
        prices_df: pd.DataFrame with columns [open, high, low, close, volume]
        lookback_days: int, number of days for analysis (default 60)
    """

    def __init__(self, llm: Optional[Callable[[str], str]] = None) -> None:
        super().__init__(agent_id="market_analyst", role="Market Technician", llm=llm)

    def analyze(self, symbol: str, data: dict[str, Any]) -> AnalysisReport:
        import numpy as np

        prices_df = data.get("prices_df")
        if prices_df is None or len(prices_df) < 2:
            return AnalysisReport(
                market_view="neutral",
                sentiment=0.0,
                fundamentals="",
                news_impact="",
                confidence=0.0,
                supporting_evidence=[],
                opposing_evidence=["No price data available"],
                agent_id=self.agent_id,
            )

        close = prices_df["close"]
        high = prices_df.get("high", close)
        low = prices_df.get("low", close)
        volume = prices_df.get("volume", pd.Series([1] * len(close)))

        # --- Trend Following ---
        ema_8 = close.ewm(span=8, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        ema_55 = close.ewm(span=55, adjust=False).mean()
        short_trend = ema_8.iloc[-1] > ema_21.iloc[-1]
        med_trend = ema_21.iloc[-1] > ema_55.iloc[-1]

        # ADX
        adx_val = self._calc_adx(prices_df, 14)
        trend_strength = adx_val / 100.0
        if short_trend and med_trend:
            trend_signal = "bullish"
        elif not short_trend and not med_trend:
            trend_signal = "bearish"
        else:
            trend_signal = "neutral"

        # --- Mean Reversion ---
        ma_50 = close.rolling(50).mean()
        std_50 = close.rolling(50).std()
        z_score = (close.iloc[-1] - ma_50.iloc[-1]) / (std_50.iloc[-1] if std_50.iloc[-1] > 0 else 1e-8)
        rsi_14 = self._calc_rsi(close, 14)
        if z_score < -2 and rsi_14 < 35:
            mr_signal = "bullish"
        elif z_score > 2 and rsi_14 > 65:
            mr_signal = "bearish"
        else:
            mr_signal = "neutral"

        # --- Momentum ---
        mom_1m = close.pct_change().rolling(21).sum().iloc[-1]
        mom_3m = close.pct_change().rolling(63).sum().iloc[-1]
        vol_mom = volume.iloc[-1] / volume.rolling(21).mean().iloc[-1] if volume.iloc[-1] > 0 else 1.0
        if mom_1m > 0.05 and vol_mom > 1.0:
            mom_signal = "bullish"
        elif mom_1m < -0.05 and vol_mom > 1.0:
            mom_signal = "bearish"
        else:
            mom_signal = "neutral"

        # --- Combine signals ---
        weights = {"trend": 0.35, "mean_reversion": 0.30, "momentum": 0.35}
        signals = {"trend": trend_signal, "mean_reversion": mr_signal, "momentum": mom_signal}
        numeric = {"bullish": 1, "neutral": 0, "bearish": -1}
        weighted = sum(numeric[s] * weights[k] for k, s in signals.items())
        if weighted > 0.1:
            overall = "bullish"
        elif weighted < -0.1:
            overall = "bearish"
        else:
            overall = "neutral"

        conf = min(abs(weighted) * 2.0, 1.0) if overall != "neutral" else 0.5
        sentiment = self._sentiment_signal(overall) * conf

        supporting = []
        opposing = []
        if signals["trend"] == overall:
            supporting.append(f"Trend following bullish (ADX={adx_val:.1f})")
        if signals["mean_reversion"] == overall:
            supporting.append(f"Mean reversion signal (Z={z_score:.2f})")
        if signals["momentum"] == overall:
            supporting.append(f"Momentum positive ({mom_1m:.1%} 1M)")

        return AnalysisReport(
            market_view=overall,
            sentiment=sentiment,
            fundamentals=f"ADX={adx_val:.1f}, RSI={rsi_14:.1f}, Z={z_score:.2f}",
            news_impact="",
            confidence=conf,
            supporting_evidence=supporting,
            opposing_evidence=opposing,
            agent_id=self.agent_id,
        )

    @staticmethod
    def _calc_rsi(series: "pd.Series", period: int = 14) -> float:
        """Calculate RSI / 相对强弱指数."""
        import pandas as pd
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return float((100 - 100 / (1 + rs)).iloc[-1])

    @staticmethod
    def _calc_adx(df: "pd.DataFrame", period: int = 14) -> float:
        """Calculate ADX / 平均方向性指数."""
        import pandas as pd
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period).mean().iloc[-1]
        return float(adx)


# ---------------------------------------------------------------------------
# SentimentAnalyst —情绪分析
# ---------------------------------------------------------------------------


class SentimentAnalyst(AnalystAgent):
    """情绪分析师 / Sentiment analyst.

    Combines insider trading data with news sentiment to produce
    a weighted market view.

    Data required in `data` dict:
        insider_trades: list of dicts with 'transaction_shares' key
        news_sentiments: list of 'positive' | 'negative' | 'neutral' strings
        weights: dict with 'insider_weight' (default 0.3) and 'news_weight' (default 0.7)
    """

    def __init__(self, llm: Optional[Callable[[str], str]] = None) -> None:
        super().__init__(agent_id="sentiment_analyst", role="Sentiment Analyst", llm=llm)

    def analyze(self, symbol: str, data: dict[str, Any]) -> AnalysisReport:
        insider_trades: list[dict] = data.get("insider_trades", [])
        news_sentiments: list[str] = data.get("news_sentiments", [])
        weights = data.get("weights", {"insider": 0.3, "news": 0.7})

        insider_w = weights.get("insider", 0.3)
        news_w = weights.get("news", 0.7)

        # Insider signals
        bullish_insider = sum(1 for t in insider_trades if t.get("transaction_shares", 0) > 0)
        bearish_insider = sum(1 for t in insider_trades if t.get("transaction_shares", 0) < 0)
        if bullish_insider > bearish_insider:
            insider_signal = "bullish"
        elif bearish_insider > bullish_insider:
            insider_signal = "bearish"
        else:
            insider_signal = "neutral"

        # News signals
        bullish_news = news_sentiments.count("positive")
        bearish_news = news_sentiments.count("negative")
        neutral_news = news_sentiments.count("neutral")
        if bullish_news > bearish_news:
            news_signal = "bullish"
        elif bearish_news > bullish_news:
            news_signal = "bearish"
        else:
            news_signal = "neutral"

        # Weighted combination
        bull_w = bullish_insider * insider_w + bullish_news * news_w
        bear_w = bearish_insider * insider_w + bearish_news * news_w

        if bull_w > bear_w:
            overall = "bullish"
            conf = min(bull_w / (bull_w + bear_w + 1e-10), 1.0)
        elif bear_w > bull_w:
            overall = "bearish"
            conf = min(bear_w / (bull_w + bear_w + 1e-10), 1.0)
        else:
            overall = "neutral"
            conf = 0.5

        sentiment = self._sentiment_signal(overall) * conf

        supporting = []
        opposing = []
        if insider_signal == overall:
            supporting.append(f"{bullish_insider} bullish insider trades vs {bearish_insider} bearish")
        else:
            opposing.append(f"Insider signal ({insider_signal}) conflicts with overall ({overall})")
        if news_signal == overall:
            supporting.append(f"{bullish_news} positive / {bearish_news} negative news articles")
        else:
            opposing.append(f"News signal ({news_signal}) conflicts with overall ({overall})")

        return AnalysisReport(
            market_view=overall,
            sentiment=sentiment,
            fundamentals="",
            news_impact=f"Insider: {insider_signal} ({bullish_insider}up/{bearish_insider}dn), News: {news_signal} ({bullish_news}+/{bearish_news}-/{neutral_news}0)",
            confidence=conf,
            supporting_evidence=supporting,
            opposing_evidence=opposing,
            agent_id=self.agent_id,
        )


# ---------------------------------------------------------------------------
# NewsAnalyst — 新闻分析
# ---------------------------------------------------------------------------


class NewsAnalyst(AnalystAgent):
    """新闻分析师 / News analyst.

    Analyzes financial news headlines/articles via LLM for sentiment.
    Falls back to keyword-based scoring when LLM is unavailable.

    Data required in `data` dict:
        headlines: list of news headline strings
    """

    def __init__(self, llm: Optional[Callable[[str], str]] = None) -> None:
        super().__init__(agent_id="news_analyst", role="News Analyst", llm=llm)

    def analyze(self, symbol: str, data: dict[str, Any]) -> AnalysisReport:
        headlines: list[str] = data.get("headlines", [])

        if not headlines:
            return AnalysisReport(
                market_view="neutral",
                sentiment=0.0,
                fundamentals="",
                news_impact="No news available",
                confidence=0.0,
                supporting_evidence=[],
                opposing_evidence=[],
                agent_id=self.agent_id,
            )

        # Try LLM-based classification if available
        if self._llm is not None and self._llm is not _default_llm:
            results = self._llm_classify_headlines(symbol, headlines)
        else:
            results = self._keyword_classify(headlines)

        bullish = sum(1 for r in results if r == "bullish")
        bearish = sum(1 for r in results if r == "bearish")
        total = len(results)

        if bullish > bearish:
            overall = "bullish"
            conf = bullish / total if total > 0 else 0.0
        elif bearish > bullish:
            overall = "bearish"
            conf = bearish / total if total > 0 else 0.0
        else:
            overall = "neutral"
            conf = 0.5

        sentiment = self._sentiment_signal(overall) * conf

        supporting = [f"{bullish} bullish articles"] if bullish > 0 else []
        opposing = [f"{bearish} bearish articles"] if bearish > 0 else []

        return AnalysisReport(
            market_view=overall,
            sentiment=sentiment,
            fundamentals="",
            news_impact=f"{bullish} positive / {bearish} negative / {total - bullish - bearish} neutral",
            confidence=conf,
            supporting_evidence=supporting,
            opposing_evidence=opposing,
            agent_id=self.agent_id,
        )

    def _llm_classify_headlines(self, symbol: str, headlines: list[str]) -> list[str]:
        """Use LLM to classify headlines / 使用LLM分类新闻标题."""
        combined = "\n".join(f"- {h}" for h in headlines[:10])
        prompt = (
            f"Classify each news headline for {symbol} as 'bullish', 'bearish', or 'neutral'. "
            f"Respond with a JSON list of these strings only, e.g. ['bullish','bearish','neutral'].\n\n"
            f"Headlines:\n{combined}"
        )
        try:
            response = self.llm_call(prompt)
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return [str(x).lower().strip() for x in parsed if str(x).lower().strip() in ("bullish", "bearish", "neutral")]
        except Exception:
            pass
        return self._keyword_classify(headlines)

    @staticmethod
    def _keyword_classify(headlines: list[str]) -> list[str]:
        """Keyword-based headline classification / 基于关键词的新闻分类."""
        bullish_kw = ["beat", "surge", "gain", "upgrade", "buy", "growth", "profit", "record", "bullish", "soar", "jump", "optimism"]
        bearish_kw = ["miss", "fall", "loss", "downgrade", "sell", "decline", "cut", "warning", "bearish", "plunge", "risk", "concern"]
        results = []
        for h in headlines:
            h_lower = h.lower()
            bull = sum(1 for kw in bullish_kw if kw in h_lower)
            bear = sum(1 for kw in bearish_kw if kw in h_lower)
            if bull > bear:
                results.append("bullish")
            elif bear > bull:
                results.append("bearish")
            else:
                results.append("neutral")
        return results


# ---------------------------------------------------------------------------
# FundamentalsAnalyst — 基本面分析
# ---------------------------------------------------------------------------


class FundamentalsAnalyst(AnalystAgent):
    """基本面分析师 / Fundamentals analyst.

    Analyzes financial metrics across four dimensions:
    1. Profitability (ROE, Net Margin, Operating Margin)
    2. Growth (Revenue Growth, Earnings Growth, Book Value Growth)
    3. Financial Health (Current Ratio, Debt-to-Equity, FCF/EPS)
    4. Valuation (P/E, P/B, P/S ratios)

    Data required in `data` dict:
        financials: dict with keys:
            return_on_equity, net_margin, operating_margin,
            revenue_growth, earnings_growth, book_value_growth,
            current_ratio, debt_to_equity, free_cash_flow_per_share, earnings_per_share,
            price_to_earnings_ratio, price_to_book_ratio, price_to_sales_ratio
    """

    def __init__(self, llm: Optional[Callable[[str], str]] = None) -> None:
        super().__init__(agent_id="fundamentals_analyst", role="Fundamentals Analyst", llm=llm)

    def analyze(self, symbol: str, data: dict[str, Any]) -> AnalysisReport:
        f = data.get("financials", {})

        def safe(v: Any, default: float = 0.0) -> float:
            try:
                return float(v) if v is not None else default
            except (ValueError, TypeError):
                return default

        roe = safe(f.get("return_on_equity"))
        net_margin = safe(f.get("net_margin"))
        op_margin = safe(f.get("operating_margin"))

        rev_growth = safe(f.get("revenue_growth"))
        earn_growth = safe(f.get("earnings_growth"))
        bv_growth = safe(f.get("book_value_growth"))

        curr_ratio = safe(f.get("current_ratio"))
        de_ratio = safe(f.get("debt_to_equity"))
        fcf_ps = safe(f.get("free_cash_flow_per_share"))
        eps = safe(f.get("earnings_per_share"))

        pe = safe(f.get("price_to_earnings_ratio"))
        pb = safe(f.get("price_to_book_ratio"))
        ps = safe(f.get("price_to_sales_ratio"))

        # Profitability
        prof_score = sum([
            1 if roe > 0.15 else 0,
            1 if net_margin > 0.20 else 0,
            1 if op_margin > 0.15 else 0,
        ])
        if prof_score >= 2:
            prof_signal = "bullish"
        elif prof_score == 0:
            prof_signal = "bearish"
        else:
            prof_signal = "neutral"

        # Growth
        growth_score = sum([
            1 if rev_growth > 0.10 else 0,
            1 if earn_growth > 0.10 else 0,
            1 if bv_growth > 0.10 else 0,
        ])
        if growth_score >= 2:
            growth_signal = "bullish"
        elif growth_score == 0:
            growth_signal = "bearish"
        else:
            growth_signal = "neutral"

        # Financial health
        health_score = sum([
            1 if curr_ratio > 1.5 else 0,
            1 if de_ratio < 0.5 else 0,
            1 if (fcf_ps > 0 and eps > 0 and fcf_ps > eps * 0.8) else 0,
        ])
        if health_score >= 2:
            health_signal = "bullish"
        elif health_score == 0:
            health_signal = "bearish"
        else:
            health_signal = "neutral"

        # Valuation (lower ratios = bullish for value)
        val_score = sum([
            1 if pe < 25 else 0,
            1 if pb < 3 else 0,
            1 if ps < 5 else 0,
        ])
        if val_score >= 2:
            val_signal = "bullish"
        elif val_score == 0:
            val_signal = "bearish"
        else:
            val_signal = "neutral"

        # Combine
        all_signals = [prof_signal, growth_signal, health_signal, val_signal]
        numeric = {"bullish": 1, "neutral": 0, "bearish": -1}
        weighted = sum(numeric[s] for s in all_signals) / len(all_signals)

        if weighted > 0.1:
            overall = "bullish"
        elif weighted < -0.1:
            overall = "bearish"
        else:
            overall = "neutral"

        conf = min(abs(weighted) * 2.5, 1.0) if overall != "neutral" else 0.5
        sentiment = self._sentiment_signal(overall) * conf

        supporting = []
        opposing = []
        for label, sig in [("Profitability", prof_signal), ("Growth", growth_signal),
                             ("Financial Health", health_signal), ("Valuation", val_signal)]:
            if sig == overall:
                supporting.append(f"{label}: {sig}")
            else:
                opposing.append(f"{label}: {sig}")

        fundamentals_text = (
            f"ROE={roe:.1%}, NetMargin={net_margin:.1%}, RevGrowth={rev_growth:.1%}, "
            f"PE={pe:.1f}, PB={pb:.1f}, CurrRatio={curr_ratio:.2f}, D/E={de_ratio:.2f}"
        )

        return AnalysisReport(
            market_view=overall,
            sentiment=sentiment,
            fundamentals=fundamentals_text,
            news_impact="",
            confidence=conf,
            supporting_evidence=supporting,
            opposing_evidence=opposing,
            agent_id=self.agent_id,
        )


# ---------------------------------------------------------------------------
# BullResearcher / BearResearcher — integrate with existing DebateEngine
# ---------------------------------------------------------------------------


class BullResearcher:
    """多头研究智能体 / Bull researcher.

    Researches and synthesizes bull-side arguments for a given symbol.
    Integrates with the existing DebateEngine (BullAgent) from debate_engine.py.

    Args:
        llm: Optional LLM call function.
    """

    def __init__(self, llm: Optional[Callable[[str], str]] = None) -> None:
        self.agent_id = "bull_researcher"
        self._llm = llm or _default_llm

    def research(
        self,
        symbol: str,
        reports: list[AnalysisReport],
        debate_context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Research bull-side arguments / 研究多头论点.

        Args:
            symbol: Ticker symbol.
            reports: List of AnalysisReport from analyst team.
            debate_context: Optional prior debate context (claims from BearResearcher).

        Returns:
            Dict with 'claims' (list of str) and 'overall_confidence' (float).
        """
        prompt = self._build_prompt(symbol, reports, debate_context, stance="bullish")
        response = self._llm(prompt)

        try:
            parsed = json.loads(response)
            claims = parsed.get("claims", [])
            confidence = float(parsed.get("confidence", 0.5))
        except Exception:
            claims = [response] if response else []
            confidence = 0.5

        return {"claims": claims, "overall_confidence": confidence, "agent_id": self.agent_id}

    def _build_prompt(
        self,
        symbol: str,
        reports: list[AnalysisReport],
        debate_context: Optional[str],
        stance: str,
    ) -> str:
        report_summaries = "\n".join(
            f"[{r.agent_id}] {r.market_view} (conf={r.confidence:.2f}): {r.supporting_evidence[:2]}"
            for r in reports
        )
        ctx = f"\nPrior arguments to respond to:\n{debate_context}" if debate_context else ""
        return (
            f"You are a {stance} researcher for {symbol}.\n"
            f"Analyst reports:\n{report_summaries}\n{ctx}\n"
            f"Provide 2-4 bull-side claims as JSON: {{\"claims\": [...], \"confidence\": float}}"
        )


class BearResearcher:
    """空头研究智能体 / Bear researcher.

    Researches and synthesizes bear-side arguments for a given symbol.
    Integrates with the existing DebateEngine (BearAgent) from debate_engine.py.

    Args:
        llm: Optional LLM call function.
    """

    def __init__(self, llm: Optional[Callable[[str], str]] = None) -> None:
        self.agent_id = "bear_researcher"
        self._llm = llm or _default_llm

    def research(
        self,
        symbol: str,
        reports: list[AnalysisReport],
        debate_context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Research bear-side arguments / 研究空头论点.

        Args:
            symbol: Ticker symbol.
            reports: List of AnalysisReport from analyst team.
            debate_context: Optional prior debate context (claims from BullResearcher).

        Returns:
            Dict with 'claims' (list of str) and 'overall_confidence' (float).
        """
        prompt = self._build_prompt(symbol, reports, debate_context, stance="bearish")
        response = self._llm(prompt)

        try:
            parsed = json.loads(response)
            claims = parsed.get("claims", [])
            confidence = float(parsed.get("confidence", 0.5))
        except Exception:
            claims = [response] if response else []
            confidence = 0.5

        return {"claims": claims, "overall_confidence": confidence, "agent_id": self.agent_id}

    def _build_prompt(
        self,
        symbol: str,
        reports: list[AnalysisReport],
        debate_context: Optional[str],
        stance: str,
    ) -> str:
        report_summaries = "\n".join(
            f"[{r.agent_id}] {r.market_view} (conf={r.confidence:.2f}): {r.opposing_evidence[:2]}"
            for r in reports
        )
        ctx = f"\nPrior arguments to respond to:\n{debate_context}" if debate_context else ""
        return (
            f"You are a {stance} researcher for {symbol}.\n"
            f"Analyst reports:\n{report_summaries}\n{ctx}\n"
            f"Provide 2-4 bear-side claims as JSON: {{\"claims\": [...], \"confidence\": float}}"
        )


# ---------------------------------------------------------------------------
# PortfolioManager — 最终决策权威
# ---------------------------------------------------------------------------


class PortfolioManager:
    """组合经理 — 最终决策权威 / Portfolio Manager — Final Decision Authority.

    Integrates all Analyst and Researcher outputs to generate
    a final TradingDecision.

    The decision flow:
    1. Collect AnalysisReports from all analysts
    2. Run BullResearcher + BearResearcher debate (optional, rounds=1 by default)
    3. Synthesize into TradingDecision with risk assessment

    Args:
        analysts: List of AnalystAgent instances.
        bull_researcher: BullResearcher instance.
        bear_researcher: BearResearcher instance.
        llm: Optional LLM call function (used for synthesis).
        debate_rounds: Number of bull/bear debate rounds (default 1).
    """

    def __init__(
        self,
        analysts: list[AnalystAgent],
        bull_researcher: Optional[BullResearcher] = None,
        bear_researcher: Optional[BearResearcher] = None,
        llm: Optional[Callable[[str], str]] = None,
        debate_rounds: int = 1,
    ) -> None:
        self.analysts = analysts
        self.bull_researcher = bull_researcher or BullResearcher(llm=llm)
        self.bear_researcher = bear_researcher or BearResearcher(llm=llm)
        self._llm = llm or _default_llm
        self.debate_rounds = debate_rounds

    def make_decision(
        self,
        symbol: str,
        market_data: dict[str, Any],
        current_positions: dict[str, Any],
    ) -> TradingDecision:
        """生成最终交易决策 / Generate final trading decision.

        Args:
            symbol: Ticker symbol.
            market_data: Dict containing data for each analyst type:
                'prices_df': pd.DataFrame (for MarketAnalyst)
                'insider_trades': list (for SentimentAnalyst)
                'news_sentiments': list (for SentimentAnalyst)
                'headlines': list (for NewsAnalyst)
                'financials': dict (for FundamentalsAnalyst)
            current_positions: Dict describing current portfolio positions:
                {'cash': float, 'positions': {symbol: {'shares': int, 'avg_cost': float}}}

        Returns:
            TradingDecision instance.
        """
        # Step 1: Run all analysts in parallel (simulated)
        reports: list[AnalysisReport] = []
        for analyst in self.analysts:
            analyst_data = self._route_data(analyst.agent_id, market_data)
            try:
                report = analyst.analyze(symbol, analyst_data)
                reports.append(report)
            except Exception:
                pass

        # Step 2: Bull/Bear debate
        bull_ctx = None
        bear_ctx = None
        for round_i in range(self.debate_rounds):
            if round_i == 0:
                bull_result = self.bull_researcher.research(symbol, reports, bear_ctx)
                bear_result = self.bear_researcher.research(symbol, reports, bull_ctx)
            else:
                bull_result = self.bull_researcher.research(symbol, reports, bear_ctx)
                bear_result = self.bear_researcher.research(symbol, reports, bull_ctx)
            bull_ctx = "\n".join(bull_result.get("claims", []))
            bear_ctx = "\n".join(bear_result.get("claims", []))

        # Step 3: Synthesize final decision
        return self._synthesize(symbol, reports, bull_result, bear_result, current_positions)

    def assess_risk(
        self,
        decision: TradingDecision,
        portfolio: dict[str, Any],
    ) -> dict[str, Any]:
        """评估决策风险 / Assess decision risk.

        Args:
            decision: TradingDecision to assess.
            portfolio: Current portfolio dict.

        Returns:
            Dict with risk assessment details.
        """
        if decision.action == "hold":
            return {"risk_level": "none", "reason": "No position change"}

        cash = float(portfolio.get("cash", 0))
        position = portfolio.get("positions", {}).get(decision.action, {})
        shares = int(position.get("shares", 0)) if isinstance(position, dict) else 0
        entry = decision.entry_price or 0.0
        stop = decision.stop_loss or 0.0

        # Simple risk metrics
        if decision.action == "buy":
            cost = decision.quantity * entry
            if cost > cash:
                return {"risk_level": "high", "reason": "Insufficient cash"}
            max_loss_pct = (entry - stop) / entry if stop > 0 else 0.10
            return {
                "risk_level": "medium" if max_loss_pct < 0.10 else "high",
                "max_loss_pct": round(max_loss_pct, 4),
                "position_cost": round(cost, 2),
                "available_cash": round(cash, 2),
            }
        elif decision.action == "sell":
            if shares < decision.quantity:
                return {"risk_level": "high", "reason": "Insufficient shares to sell"}
            return {"risk_level": "low", "reason": "Selling existing position"}

        return {"risk_level": "unknown"}

    def _route_data(
        self,
        agent_id: str,
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Route relevant data to each analyst type."""
        routing = {
            "market_analyst": ["prices_df"],
            "sentiment_analyst": ["insider_trades", "news_sentiments", "weights"],
            "news_analyst": ["headlines"],
            "fundamentals_analyst": ["financials"],
        }
        keys = routing.get(agent_id, [])
        return {k: market_data.get(k) for k in keys}

    def _synthesize(
        self,
        symbol: str,
        reports: list[AnalysisReport],
        bull_result: dict[str, Any],
        bear_result: dict[str, Any],
        current_positions: dict[str, Any],
    ) -> TradingDecision:
        """Synthesize all reports + debate into a TradingDecision."""
        # Aggregate market view
        numeric_view = {"bullish": 1, "neutral": 0, "bearish": -1}
        if not reports:
            return TradingDecision(
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning="No analyst reports available",
            )

        # Weighted sentiment from analysts
        total_conf = sum(r.confidence for r in reports)
        if total_conf > 0:
            weighted_view = sum(numeric_view.get(r.market_view, 0) * r.confidence for r in reports) / total_conf
        else:
            weighted_view = 0.0

        # Factor in bull/bear debate
        bull_conf = bull_result.get("overall_confidence", 0.5)
        bear_conf = bear_result.get("overall_confidence", 0.5)
        debate_view = (bull_conf - bear_conf) * 0.3  # 30% weight to debate

        final_view_score = weighted_view + debate_view

        # Map to action
        cash = float(current_positions.get("cash", 0))
        positions = current_positions.get("positions", {})
        current_shares = 0
        if symbol in positions:
            pos = positions[symbol]
            current_shares = int(pos.get("shares", 0)) if isinstance(pos, dict) else 0

        # Current price (from any market report)
        current_price = 0.0
        for r in reports:
            if r.agent_id == "market_analyst" and r.fundamentals:
                # Try to extract price from fundamentals text
                pass
        # Fallback: use 0 if no price known
        if current_price <= 0:
            current_price = 100.0  # placeholder

        if final_view_score > 0.2 and cash >= current_price:
            action = "buy"
            quantity = min(max(int(cash * 0.5 / current_price), 1), 1000)
            entry = current_price
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.20
        elif final_view_score < -0.2 and current_shares > 0:
            action = "sell"
            quantity = current_shares
            entry = current_price
            stop_loss = None
            take_profit = None
        else:
            action = "hold"
            quantity = 0
            entry = None
            stop_loss = None
            take_profit = None

        # Confidence
        confidence = min(abs(final_view_score) * 1.5, 1.0) if final_view_score != 0 else 0.5

        reasoning_parts = [f"Analyst consensus: {'bullish' if weighted_view > 0 else 'bearish' if weighted_view < 0 else 'neutral'} (score={weighted_view:.3f})"]
        if bull_result.get("claims"):
            reasoning_parts.append(f"Bull claims: {bull_result['claims'][0][:80]}")
        if bear_result.get("claims"):
            reasoning_parts.append(f"Bear claims: {bear_result['claims'][0][:80]}")

        return TradingDecision(
            action=action,
            quantity=float(quantity),
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
            supporting_reports=reports,
            risk_assessment=self._assess_decision_risk(action, quantity, entry, stop_loss, cash, current_shares),
        )

    @staticmethod
    def _assess_decision_risk(
        action: str,
        quantity: float,
        entry: Optional[float],
        stop_loss: Optional[float],
        cash: float,
        current_shares: int,
    ) -> str:
        if action == "hold":
            return "No action taken — neutral market view."
        if action == "buy":
            cost = quantity * (entry or 0)
            risk = f"Buy {quantity} shares at ${entry:.2f}, total ${cost:.2f}, cash available ${cash:.2f}."
            if stop_loss:
                risk += f" Stop-loss: ${stop_loss:.2f}."
            return risk
        if action == "sell":
            return f"Sell {quantity} shares from {current_shares} held position."
        return "Unknown action."
