"""A-Share Bull vs Bear Multi-Agent Debate Framework.

A 股多空辩论智能体框架 — 15 Agent 辩论团队.

Architecture:
- BullBearDebate: 多空辩论引擎, orchestrates claim-driven bull/bear debate
- AShareResearcher: A 股研究智能体, China stock market research (AkShare data)
- ClaimValidator: 论点验证器, validates bull/bear claims against market data
- PortfolioConstructor: 组合构建器, builds portfolio from debate outcomes
- TradingAgentsAShareCoordinator: 协调器, orchestrates the full 15-agent team

Key Agents (7 core):
- MarketAnalyst: 技术面分析
- FundamentalAnalyst: 基本面分析
- TechnicalAnalyst: 技术指标分析 (alias for MarketAnalyst)
- SentimentAnalyst: 情绪分析
- MacroAnalyst: 宏观与板块分析
- BullResearcher: 多头研究员
- BearResearcher: 空头研究员

流程:
1. 多分析师并行研究 — Market / Fundamental / Sentiment / Macro
2. 多头空头分别建立 Claims
3. 交叉反驳 — ClaimValidator 评分
4. 裁判收口 — PortfolioConstructor 构建组合
5. 输出可执行交易方案

本模块使用纯 REST LLM 调用（urllib），不依赖重型 SDK。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "BullBearDebate",
    "AShareResearcher",
    "ClaimValidator",
    "PortfolioConstructor",
    "TradingAgentsAShareCoordinator",
    "AShareAnalystReport",
    "DebateOutcome",
    "PortfolioPosition",
    "TradingSignal",
    "DEFAULT_LLM_CALL",
    "build_llm_client",
]


# ---------------------------------------------------------------------------
# REST-only LLM client (urllib, no heavy SDK)
# ---------------------------------------------------------------------------

def _urllib_request(
    prompt: str,
    *,
    model: str = "gpt-4",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 30,
) -> str:
    """Lightweight REST LLM call via urllib — no SDK dependency.

    纯 urllib 实现，不依赖 openai/anthropic SDK.

    Returns:
        LLM response text, or error indicator string starting with "ERROR:".
    """
    api_base = api_base or os.environ.get("LLM_API_BASE")
    if not api_base:
        logger.error("LLM_API_BASE environment variable is required")
        return "ERROR:CONFIG:LLM_API_BASE not set"
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        logger.error("OPENAI_API_KEY is required but not set")
        return "ERROR:AUTH:API key not configured"

    endpoint = f"{api_base.rstrip('/')}/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error {e.code}: {e.reason}")
        return f"ERROR:HTTP:{e.code}"
    except urllib.error.URLError as e:
        logger.error(f"URL error: {e.reason}")
        return "ERROR:NETWORK"
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return "ERROR:PARSE"
    except Exception as e:
        logger.error(f"Unexpected error in LLM call: {e}")
        return f"ERROR:UNKNOWN:{e}"


DEFAULT_LLM_CALL = _urllib_request


def build_llm_client(
    model: str = "gpt-4",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 30,
) -> Callable[[str], str]:
    """Build a configured LLM client callable.

    构建可配置的 LLM 调用客户端.

    Args:
        model: 模型名称 (default "gpt-4")
        api_base: API endpoint (默认读取 LLM_API_BASE 环境变量)
        api_key: API key (默认读取 OPENAI_API_KEY 环境变量)
        temperature: 温度参数
        timeout: 超时秒数

    Returns:
        可调用对象: 输入 prompt 字符串，返回响应字符串
    """
    def client(prompt: str) -> str:
        return _urllib_request(
            prompt,
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
        )
    return client


# ---------------------------------------------------------------------------
# Data Enums
# ---------------------------------------------------------------------------

class Stance(str, Enum):
    """辩论立场 / Debate stance."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class VerdictDirection(str, Enum):
    """机读方向 / Verdict direction."""
    BULLISH = "看多"
    BEARISH = "看空"
    NEUTRAL = "中性"
    CAUTIOUS = "谨慎"


class DataHorizon(str, Enum):
    """数据时间窗口 / Analysis horizon."""
    SHORT = "short"    # 1-2周
    MEDIUM = "medium"  # 1-3月


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AShareAnalystReport:
    """A 股分析师报告 / Analyst report for A-share stocks.

    Attributes:
        agent: 分析智能体名称
        ticker: 股票代码
        verdict: 方向判断 (看多/看空/中性/谨慎)
        confidence: 置信度
        content: 报告正文
        timestamp: 时间戳
    """
    agent: str
    ticker: str
    verdict: str
    confidence: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "ticker": self.ticker,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Claim:
    """辩论论点 / Debate claim.

    Attributes:
        claim_id: 唯一标识
        agent_id: 提出该论点的智能体ID
        stance: 立场 — bull/bear/neutral
        content: 论点内容
        evidence: 支撑证据列表
        confidence: 置信度 [0, 1]
        status: 状态 — open/addressed/resolved/unresolved
        target_claim_ids: 针对的论点ID列表
        round_index: 辩论轮次
        timestamp: 时间戳
    """
    claim_id: str
    agent_id: str
    stance: str
    content: str
    evidence: list[str]
    confidence: float
    status: str = "open"
    target_claim_ids: list[str] = field(default_factory=list)
    round_index: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "agent_id": self.agent_id,
            "stance": self.stance,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "status": self.status,
            "target_claim_ids": self.target_claim_ids,
            "round_index": self.round_index,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DebateOutcome:
    """辩论结果 / Debate outcome.

    Attributes:
        topic: 辩论主题
        bull_claims: 多头论点列表
        bear_claims: 空头论点列表
        winner: 胜方 — bull/bear/tie
        bull_score: 多头总分
        bear_score: 空头总分
        consensus: 共识摘要
        final_verdict: 最终判决文本
        confidence: 整体置信度
        signal: 交易信号 (看多/看空/中性)
    """
    topic: str
    bull_claims: list[Claim]
    bear_claims: list[Claim]
    winner: str
    bull_score: float
    bear_score: float
    consensus: str
    final_verdict: str
    confidence: float
    signal: str = "中性"

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "bull_claims": [c.to_dict() for c in self.bull_claims],
            "bear_claims": [c.to_dict() for c in self.bear_claims],
            "winner": self.winner,
            "bull_score": self.bull_score,
            "bear_score": self.bear_score,
            "consensus": self.consensus,
            "final_verdict": self.final_verdict,
            "confidence": self.confidence,
            "signal": self.signal,
        }


@dataclass
class PortfolioPosition:
    """组合持仓 / Portfolio position.

    Attributes:
        ticker: 股票代码
        direction: 方向 — long/short/neutral
        weight: 权重 (0-1)
        entry_price: 入场价
        stop_loss: 止损价
        target_price: 目标价
        confidence: 置信度
        thesis: 持仓逻辑
    """
    ticker: str
    direction: str
    weight: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    confidence: float = 0.5
    thesis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "weight": self.weight,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_price": self.target_price,
            "confidence": self.confidence,
            "thesis": self.thesis,
        }


@dataclass
class TradingSignal:
    """交易信号 / Trading signal.

    Attributes:
        ticker: 股票代码
        action: 动作 — BUY/SELL/HOLD
        direction: 方向 — long/short/neutral
        confidence: 置信度
        horizon: 时间窗口
        thesis: 交易逻辑
        risks: 风险提示
        indicators: 参考指标
        timestamp: 时间戳
    """
    ticker: str
    action: str
    direction: str
    confidence: float
    horizon: str
    thesis: str
    risks: str
    indicators: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "direction": self.direction,
            "confidence": self.confidence,
            "horizon": self.horizon,
            "thesis": self.thesis,
            "risks": self.risks,
            "indicators": self.indicators,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# A-Share Data Fetcher (REST-only, pure Python)
# ---------------------------------------------------------------------------

class _AkshareFetcher:
    """AkShare data fetcher for A-share stocks.

    使用 AkShare 获取 A 股市场数据（需提前安装: pip install akshare）。
    所有网络请求通过 urllib 实现，不依赖 SDK 特有协议。
    """

    def __init__(self):
        self._ak = None
        self._lock = __import__("threading").RLock()

    def _get_ak(self):
        with self._lock:
            if self._ak is None:
                try:
                    import akshare as ak
                    self._ak = ak
                except ImportError:
                    raise RuntimeError(
                        "akshare is required for A-share data. "
                        "Install: pip install akshare"
                    )
            return self._ak

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> str:
        """获取 K 线数据 / Get OHLCV candlestick data."""
        ak = self._get_ak()
        code = self._normalize_symbol(symbol)
        try:
            df = ak.stock_zh_a_hist(symbol=code, start_date=start_date.replace("-", ""),
                                    end_date=end_date.replace("-", ""), adjust="qfq")
            if df is None or df.empty:
                return f"无数据: {symbol} [{start_date} - {end_date}]"
            cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
            existing = [c for c in cols if c in df.columns]
            return df[existing].tail(30).to_string(index=False)
        except Exception as exc:
            logger.error("Failed to fetch stock data for %s: %s", symbol, exc)
            return f"数据获取失败: {exc}"

    def get_indicators(self, symbol: str, indicator: str, look_back_days: int = 20) -> str:
        """获取技术指标 / Get technical indicator."""
        ak = self._get_ak()
        code = self._normalize_symbol(symbol)
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=(datetime.now() - timedelta(days=look_back_days * 2)).strftime("%Y%m%d"),
                                    end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
            if df is None or df.empty:
                return f"无数据: {symbol}"

            close = df["收盘"].values
            if indicator == "rsi":
                delta = (df["收盘"].diff()).values
                gain = delta.copy()
                loss = delta.copy()
                gain[gain < 0] = 0
                loss[loss > 0] = 0
                avg_gain = self._sma(gain, 14)
                avg_loss = self._sma(loss, 14)
                rs = avg_gain / (avg_loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                val = rsi[-1] if not math.isnan(rsi[-1]) else float('nan')
                return f"RSI(14): {val:.2f}" if not math.isnan(val) else "无数据"
            elif indicator in ("macd", "macds", "macdh"):
                ema12 = self._ema(close, 12)
                ema26 = self._ema(close, 26)
                macd = ema12 - ema26
                signal = self._ema(macd, 9)
                hist = macd - signal
                if indicator == "macd":
                    return f"MACD: {macd[-1]:.4f}"
                elif indicator == "macds":
                    return f"MACD Signal: {signal[-1]:.4f}"
                else:
                    return f"MACD Hist: {hist[-1]:.4f}"
            elif indicator == "boll":
                sma20 = self._sma(close, 20)
                std20 = self._std(close, 20)
                return f"BOLL: 中轨={sma20[-1]:.2f} 上轨={sma20[-1]+2*std20[-1]:.2f} 下轨={sma20[-1]-2*std20[-1]:.2f}"
            elif indicator == "atr":
                high, low = df["最高"].values, df["最低"].values
                prev_close = df["收盘"].shift(1).values
                tr = [max(h-l, max(abs(h-pc), abs(l-pc))) for h, l, pc in zip(high, low, prev_close)]
                atr = self._sma(tr, 14)
                return f"ATR(14): {atr[-1]:.4f}"
            elif indicator in ("close_10_ema", "close_50_sma", "close_200_sma"):
                period = int(re.search(r"\d+", indicator).group())
                if period == 200 and len(close) < 200:
                    return f"{period}日均线: 数据不足（仅{len(close)}天）"
                val = self._ema(close, period) if "ema" in indicator else self._sma(close, period)
                return f"{period}日均线: {val[-1]:.2f}"
            else:
                return f"指标 {indicator} 暂不支持"
        except Exception as exc:
            return f"指标获取失败: {exc}"

    def get_fundamentals(self, ticker: str, curr_date: Optional[str] = None) -> str:
        """获取基本面概要 / Get fundamental summary."""
        ak = self._get_ak()
        code = self._normalize_symbol(ticker)
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2020")
            if df is None or df.empty:
                return f"无数据: {ticker}"
            cols = ["日期", "净资产收益率(%)", "销售毛利率(%)", "资产负债率(%)", "存货周转率", "净利润增长率(%)"]
            existing = [c for c in cols if c in df.columns]
            return df[existing].tail(4).to_string(index=False)
        except Exception as exc:
            return f"基本面数据失败: {exc}"

    def get_news(self, ticker: str, look_back_days: int = 7) -> str:
        """获取近期新闻 / Get recent news."""
        ak = self._get_ak()
        code = self._normalize_symbol(ticker)
        try:
            df = ak.stock_news_em(symbol=code)
            if df is None or df.empty:
                return "无新闻数据"
            cutoff = datetime.now() - timedelta(days=look_back_days)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            recent = df[df["datetime"] >= cutoff]
            return recent[["datetime", "title"]].to_string(index=False) if not recent.empty else "近期无新闻"
        except Exception as exc:
            return f"新闻获取失败: {exc}"

    def _normalize_symbol(self, symbol: str) -> str:
        s = symbol.strip().lower()
        m = re.search(r"(\d{6})", s)
        if not m:
            return symbol
        return m.group(1)

    def _sma(self, arr, period: int):
        import numpy as np
        arr = np.array(arr, dtype=float)
        result = np.full(len(arr), np.nan)
        result[period - 1:] = np.convolve(arr, np.ones(period) / period, mode='valid')
        return result

    def _ema(self, arr, period: int):
        import numpy as np
        arr = np.array(arr, dtype=float)
        alpha = 2 / (period + 1)
        result = np.zeros(len(arr))
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    def _std(self, arr, period: int):
        import numpy as np
        arr = np.array(arr, dtype=float)
        result = np.full(len(arr), np.nan)
        for i in range(period - 1, len(arr)):
            result[i] = np.std(arr[i - period + 1:i + 1])
        return result


try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


# ---------------------------------------------------------------------------
# AShareResearcher — A-share market research using AkShare
# ---------------------------------------------------------------------------

class AShareResearcher:
    """A 股市场研究智能体 / A-share market research agent.

    基于 AkShare 数据源，对 A 股标的进行技术面、基本面、情绪面、宏观面分析。
    支持 MarketAnalyst / FundamentalAnalyst / SentimentAnalyst / MacroAnalyst 角色。

    Args:
        llm_call: LLM 调用函数，接收 prompt 字符串返回响应字符串
        ticker: 股票代码 (如 "600519")
        trade_date: 交易日期 (YYYY-MM-DD)
        horizon: 分析周期 — "short" 或 "medium"
        fetcher: 数据获取器 (默认使用 AkShareFetcher)
    """

    def __init__(
        self,
        llm_call: Callable[[str], str],
        ticker: str,
        trade_date: str,
        horizon: str = "medium",
        fetcher: Optional[Any] = None,
    ):
        # Validate ticker format (A-share: 6-digit codes like 000001, 600000)
        if not re.match(r"^\d{6}$", ticker):
            raise ValueError(f"Invalid A-share ticker: {ticker}. Expected 6-digit code")
        self.llm_call = llm_call
        self.ticker = ticker
        self.trade_date = trade_date
        self.horizon = horizon
        self.fetcher = fetcher or _AkshareFetcher()
        self._reports: dict[str, AShareAnalystReport] = {}

    # ------------------------------------------------------------------
    # Market Analyst — 技术面分析
    # ------------------------------------------------------------------
    def run_market_analyst(self) -> AShareAnalystReport:
        """运行技术面分析 / Run technical/market analysis.

        分析 K 线形态、移动均线、RSI、MACD、布林带等指标。

        Returns:
            AShareAnalystReport: 技术面分析报告
        """
        end_dt = datetime.strptime(self.trade_date, "%Y-%m-%d")
        days = 30 if self.horizon == "short" else 90
        start_dt = end_dt - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = self.trade_date

        stock_data = self.fetcher.get_stock_data(self.ticker, start_str, end_str)

        indicators = {}
        for ind in ["rsi", "macd", "boll", "atr", "close_10_ema", "close_50_sma"]:
            indicators[ind] = self.fetcher.get_indicators(self.ticker, ind, look_back_days=days)

        indicator_text = "\n".join(f"【{k}】{v}" for k, v in indicators.items())

        prompt = f"""你是 A 股技术面分析师。请对股票 {self.ticker} 在 {self.trade_date} 的技术面进行综合分析。

【K线数据（近{days}天）】
{stock_data}

【技术指标】
{indicator_text}

请全程使用中文输出，结构如下：
1. 价格行为与关键区间（支撑/阻力/突破位）
2. 趋势判断（短中长期是否一致）
3. 动量判断（拐点、背离）
4. 波动与仓位建议
5. 交易含义（偏多/偏空/震荡）
6. 机读摘要（格式固定，末尾必须包含）：
<!-- VERDICT: {{"direction": "看多", "reason": "不超过20字的核心结论"}} -->
direction 只可填：看多 / 看空 / 中性 / 谨慎"""

        content = self.llm_call(prompt)
        verdict = self._extract_verdict(content)

        report = AShareAnalystReport(
            agent="MarketAnalyst",
            ticker=self.ticker,
            verdict=verdict[0],
            confidence=verdict[1],
            content=content,
        )
        self._reports["market"] = report
        return report

    # ------------------------------------------------------------------
    # Fundamental Analyst — 基本面分析
    # ------------------------------------------------------------------
    def run_fundamental_analyst(self) -> AShareAnalystReport:
        """运行基本面分析 / Run fundamental analysis.

        分析财务报表、估值水平、盈利能力、资产负债等。

        Returns:
            AShareAnalystReport: 基本面分析报告
        """
        fundamentals = self.fetcher.get_fundamentals(self.ticker, self.trade_date)

        prompt = f"""你是 A 股基本面分析师。请对股票 {self.ticker} 在 {self.trade_date} 进行基本面分析。

【财务数据】
{fundamentals}

请全程使用中文输出，结构如下：
1. 商业模式与竞争力简述
2. 收入与盈利质量
3. 资产负债与现金流健康度
4. 核心风险（政策、需求、竞争）
5. 对中期持仓的结论
6. 机读摘要（格式固定，末尾必须包含）：
<!-- VERDICT: {{"direction": "看多", "reason": "不超过20字的核心结论"}} -->
direction 只可填：看多 / 看空 / 中性 / 谨慎"""

        content = self.llm_call(prompt)
        verdict = self._extract_verdict(content)

        report = AShareAnalystReport(
            agent="FundamentalAnalyst",
            ticker=self.ticker,
            verdict=verdict[0],
            confidence=verdict[1],
            content=content,
        )
        self._reports["fundamental"] = report
        return report

    # ------------------------------------------------------------------
    # Sentiment Analyst — 情绪分析
    # ------------------------------------------------------------------
    def run_sentiment_analyst(self) -> AShareAnalystReport:
        """运行情绪分析 / Run sentiment analysis.

        分析新闻舆情、资金流向、市场情绪温度。

        Returns:
            AShareAnalystReport: 情绪分析报告
        """
        news = self.fetcher.get_news(self.ticker, look_back_days=7)

        prompt = f"""你是 A 股情绪分析师。请对股票 {self.ticker} 在 {self.trade_date} 进行情绪分析。

【近期新闻】
{news}

请全程使用中文输出，结构如下：
1. 当前情绪温度（偏冷/中性/偏热）与证据
2. 关键情绪触发点与潜在反转信号
3. 情绪持续性判断
4. 交易影响（追涨/回撤买入/观望）
5. 风险提示
6. 机读摘要（格式固定，末尾必须包含）：
<!-- VERDICT: {{"direction": "看多", "reason": "不超过20字的核心结论"}} -->
direction 只可填：看多 / 看空 / 中性 / 谨慎"""

        content = self.llm_call(prompt)
        verdict = self._extract_verdict(content)

        report = AShareAnalystReport(
            agent="SentimentAnalyst",
            ticker=self.ticker,
            verdict=verdict[0],
            confidence=verdict[1],
            content=content,
        )
        self._reports["sentiment"] = report
        return report

    # ------------------------------------------------------------------
    # Macro Analyst — 宏观与板块分析
    # ------------------------------------------------------------------
    def run_macro_analyst(self) -> AShareAnalystReport:
        """运行宏观与板块分析 / Run macro and sector analysis.

        分析宏观环境、板块轮动、政策驱动信号。

        Returns:
            AShareAnalystReport: 宏观分析报告
        """
        # 简化：利用近期市场新闻作为宏观代理
        news = self.fetcher.get_news(self.ticker, look_back_days=14)
        stock_data = self.fetcher.get_stock_data(
            self.ticker,
            (datetime.strptime(self.trade_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d"),
            self.trade_date,
        )

        prompt = f"""你是 A 股宏观与板块分析师。请对股票 {self.ticker} 在 {self.trade_date} 进行宏观与板块分析。

【近期相关新闻】
{news}

【近期价格走势】
{stock_data}

请全程使用中文输出，结构如下：
1. 宏观与政策环境评估
2. 板块资金流向判断
3. 个股与板块关系
4. 宏观风险提示
5. 综合判断
6. 机读摘要（格式固定，末尾必须包含）：
<!-- VERDICT: {{"direction": "看多", "reason": "不超过20字的核心结论"}} -->
direction 只可填：看多 / 看空 / 中性 / 谨慎"""

        content = self.llm_call(prompt)
        verdict = self._extract_verdict(content)

        report = AShareAnalystReport(
            agent="MacroAnalyst",
            ticker=self.ticker,
            verdict=verdict[0],
            confidence=verdict[1],
            content=content,
        )
        self._reports["macro"] = report
        return report

    # ------------------------------------------------------------------
    # Run all analysts in parallel
    # ------------------------------------------------------------------
    def run_all_analysts(self) -> dict[str, AShareAnalystReport]:
        """运行所有分析师 / Run all analysts in parallel.

        并行执行 Market / Fundamental / Sentiment / Macro 分析。

        Returns:
            dict: 各分析师报告字典
        """
        import concurrent.futures

        def run_with_fallback(method_name: str):
            try:
                return getattr(self, method_name)()
            except Exception as exc:
                return AShareAnalystReport(
                    agent=method_name.replace("run_", "").replace("_analyst", "").title() + "Analyst",
                    ticker=self.ticker,
                    verdict="中性",
                    confidence="低",
                    content=f"分析失败: {exc}",
                )

        methods = [
            "run_market_analyst",
            "run_fundamental_analyst",
            "run_sentiment_analyst",
            "run_macro_analyst",
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_with_fallback, m): m for m in methods}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    method = futures[future]
                    logger.error(f"Analyst {method} failed with exception: {exc}")

        return self._reports

    def _extract_verdict(self, text: str) -> tuple[str, str]:
        """从报告中提取 VERDICT 机读块 / Extract VERDICT from report text."""
        m = re.search(r'<!--\s*VERDICT:\s*(\{.*?\})\s*-->', text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(1))
                return d.get("direction", "中性"), d.get("reason", "低")
            except (json.JSONDecodeError, ValueError):
                pass
        return "中性", "低"

    def get_summary(self) -> str:
        """获取所有报告摘要 / Get summary of all reports."""
        lines = [f"=== AShareResearcher Reports for {self.ticker} ({self.trade_date}) ===\n"]
        for key, report in self._reports.items():
            lines.append(f"\n[{report.agent}] verdict={report.verdict} confidence={report.confidence}")
            lines.append(report.content[:300] + "..." if len(report.content) > 300 else report.content)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClaimValidator — validates bull/bear claims against data
# ---------------------------------------------------------------------------

class ClaimValidator:
    """论点验证器 / Claim validator.

    对多头/空头 Claims 进行数据验证，评估其证据强度和置信度。

    Args:
        llm_call: LLM 调用函数
        researcher: AShareResearcher 实例（提供实时数据）
    """

    def __init__(
        self,
        llm_call: Callable[[str], str],
        researcher: Optional[AShareResearcher] = None,
    ):
        self.llm_call = llm_call
        self.researcher = researcher

    def validate_claim(self, claim: Claim, context: dict[str, str]) -> tuple[float, str]:
        """验证单条 Claim / Validate a single claim against data.

        Args:
            claim: 待验证的论点
            context: 上下文字典（包含 market_report / fundamental_report 等）

        Returns:
            tuple: (adjusted_confidence, validation_note)
        """
        prompt = f"""你是论点验证专家。请评估以下论点的证据强度。

论点: {claim.content}
立场: {claim.stance}
声称置信度: {claim.confidence}
证据: {'；'.join(claim.evidence) if claim.evidence else '无'}

背景数据:
市场报告: {context.get('market_report', '无')[:500]}
基本面报告: {context.get('fundamental_report', '无')[:500]}
情绪报告: {context.get('sentiment_report', '无')[:500]}

请评估：
1. 该论点的证据是否充分？
2. 置信度是否需要调整？
3. 给出一个简短的验证结论。

请以JSON格式回复：
{{"adjusted_confidence": 0.85, "validation_note": "该论点有较强数据支撑，证据充分。"}}
adjusted_confidence 为 0 到 1 之间的小数。"""

        try:
            response = self.llm_call(prompt)
            # Extract JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                conf = float(data.get("adjusted_confidence", claim.confidence))
                note = str(data.get("validation_note", ""))
                return max(0.0, min(1.0, conf)), note
        except (json.JSONDecodeError, ValueError, Exception):
            pass

        return claim.confidence, "验证失败，使用原始置信度"

    def score_claims(self, claims: list[Claim], context: dict[str, str]) -> list[Claim]:
        """对所有 Claims 进行验证评分 / Validate and rescore all claims.

        Args:
            claims: 论点列表
            context: 上下文字典

        Returns:
            list[Claim]: 调整置信度后的论点列表
        """
        scored = []
        for claim in claims:
            adj_conf, note = self.validate_claim(claim, context)
            claim.confidence = adj_conf
            claim.evidence.append(f"[验证备注] {note}")
            scored.append(claim)
        return scored

    def build_validation_report(self, bull_claims: list[Claim], bear_claims: list[Claim],
                                 context: dict[str, str]) -> str:
        """构建验证报告 / Build validation report for bull and bear claims.

        Args:
            bull_claims: 多头论点
            bear_claims: 空头论点
            context: 上下文

        Returns:
            str: 验证报告文本
        """
        bull_avg = sum(c.confidence for c in bull_claims) / max(len(bull_claims), 1)
        bear_avg = sum(c.confidence for c in bear_claims) / max(len(bear_claims), 1)

        lines = [
            "=== Claim Validation Report ===",
            f"多头论点数: {len(bull_claims)}, 平均置信度: {bull_avg:.3f}",
            f"空头论点数: {len(bear_claims)}, 平均置信度: {bear_avg:.3f}",
        ]
        for c in bull_claims:
            lines.append(f"  [BULL] {c.claim_id}: {c.content[:60]} (conf={c.confidence:.2f})")
        for c in bear_claims:
            lines.append(f"  [BEAR] {c.claim_id}: {c.content[:60]} (conf={c.confidence:.2f})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BullBearDebate — multi-agent debate with claim scoring
# ---------------------------------------------------------------------------

class BullBearDebate:
    """多空辩论引擎 / Bull vs Bear debate engine.

    基于 claim 驱动的多轮辩论框架，融合 AShareResearcher 的分析结果。

    流程:
    1. 初始化 — 接收分析师报告
    2. 多头建立 Claims — BullResearcher 提出论点
    3. 空头建立 Claims — BearResearcher 提出论点
    4. 交叉反驳 — ClaimValidator 评分
    5. 裁判收口 — 输出 DebateOutcome

    Args:
        llm_call: LLM 调用函数
        researcher: AShareResearcher 实例
        n_rounds: 辩论轮数 (default 3)
        claim_validator: ClaimValidator 实例
    """

    def __init__(
        self,
        llm_call: Callable[[str], str],
        researcher: AShareResearcher,
        n_rounds: int = 3,
        claim_validator: Optional[ClaimValidator] = None,
    ):
        self.llm_call = llm_call
        self.researcher = researcher
        self.n_rounds = n_rounds
        self.claim_validator = claim_validator or ClaimValidator(llm_call, researcher)
        self.claims: list[Claim] = []
        self._claim_counter = 0
        self._counter_lock = threading.Lock()

    def run_debate(self) -> DebateOutcome:
        """运行完整辩论流程 / Run full debate process.

        Returns:
            DebateOutcome: 辩论结果
        """
        reports = self.researcher._reports
        market_report = reports.get("market", AShareAnalystReport("", "", "", "", "")).content
        fundamental_report = reports.get("fundamental", AShareAnalystReport("", "", "", "", "")).content
        sentiment_report = reports.get("sentiment", AShareAnalystReport("", "", "", "", "")).content
        macro_report = reports.get("macro", AShareAnalystReport("", "", "", "", "")).content

        context = {
            "market_report": market_report,
            "fundamental_report": fundamental_report,
            "sentiment_report": sentiment_report,
            "macro_report": macro_report,
        }

        # 第1轮: 多头建立 Claims
        bull_claims_r1 = self._bull_researcher_round(
            context, round_idx=1, is_initial=True
        )
        # 第1轮: 空头建立 Claims
        bear_claims_r1 = self._bear_researcher_round(
            context, round_idx=1, is_initial=True
        )

        all_bull = list(bull_claims_r1)
        all_bear = list(bear_claims_r1)

        # 第2-N轮: 交叉反驳
        for round_idx in range(2, self.n_rounds + 1):
            # 多头反驳空头
            new_bull = self._bull_researcher_round(
                {**context, "opposing_claims": all_bear},
                round_idx=round_idx,
                is_initial=False,
            )
            all_bull.extend(new_bull)

            # 空头反驳多头
            new_bear = self._bear_researcher_round(
                {**context, "opposing_claims": all_bull},
                round_idx=round_idx,
                is_initial=False,
            )
            all_bear.extend(new_bear)

        # ClaimValidator 评分
        all_bull = self.claim_validator.score_claims(all_bull, context)
        all_bear = self.claim_validator.score_claims(all_bear, context)

        # 裁判收口
        outcome = self._judge_debate(all_bull, all_bear, context)

        self.claims = all_bull + all_bear
        return outcome

    def _bull_researcher_round(
        self,
        context: dict,
        round_idx: int,
        is_initial: bool = False,
    ) -> list[Claim]:
        """多头研究员轮次 / Bull researcher round."""
        opposing = context.get("opposing_claims", [])
        opposing_text = "\n".join(
            f"- [{c.claim_id}] {c.content} (conf={c.confidence:.2f})"
            for c in opposing
        ) if opposing else "暂无反驳目标"

        existing_claims_text = "\n".join(
            f"- [{c.claim_id}] {c.content}"
            for c in context.get("existing_claims", [])
        ) if context.get("existing_claims") else "暂无"

        focus_instruction = (
            "首轮请提出 1-2 条最核心的多头 claim，建立论证骨架。"
            if is_initial else
            f"请针对以下空头论点提出反驳：\n{opposing_text}\n\n已有多头 claim：\n{existing_claims_text}"
        )

        prompt = f"""你是 A 股多头研究员，股票 {self.researcher.ticker}，日期 {self.researcher.trade_date}。

分析背景:
【市场技术面】
{context.get('market_report', '无')[:800]}

【基本面】
{context.get('fundamental_report', '无')[:800]}

【情绪面】
{context.get('sentiment_report', '无')[:800]}

【宏观面】
{context.get('macro_report', '无')[:800]}

任务:
{focus_instruction}

请以证据链组织论点，给出置信度(0-1)。在正文末尾追加机读块：
<!-- DEBATE_STATE: {{"new_claims": [{{"claim": "论点内容不超过28字", "evidence": ["证据1", "证据2"], "confidence": 0.72}}], "responded_claim_ids": [], "resolved_claim_ids": [], "round_summary": "不超过50字"}} -->

请全程使用中文。"""

        response = self.llm_call(prompt)
        return self._parse_claims_from_response(response, stance="bull", round_idx=round_idx)

    def _bear_researcher_round(
        self,
        context: dict,
        round_idx: int,
        is_initial: bool = False,
    ) -> list[Claim]:
        """空头研究员轮次 / Bear researcher round."""
        opposing = context.get("opposing_claims", [])
        opposing_text = "\n".join(
            f"- [{c.claim_id}] {c.content} (conf={c.confidence:.2f})"
            for c in opposing
        ) if opposing else "暂无反驳目标"

        existing_claims_text = "\n".join(
            f"- [{c.claim_id}] {c.content}"
            for c in context.get("existing_claims", [])
        ) if context.get("existing_claims") else "暂无"

        focus_instruction = (
            "首轮请提出 1-2 条最核心的空头 claim，建立论证骨架。"
            if is_initial else
            f"请针对以下多头论点提出反驳：\n{opposing_text}\n\n已有空头 claim：\n{existing_claims_text}"
        )

        prompt = f"""你是 A 股空头研究员，股票 {self.researcher.ticker}，日期 {self.researcher.trade_date}。

分析背景:
【市场技术面】
{context.get('market_report', '无')[:800]}

【基本面】
{context.get('fundamental_report', '无')[:800]}

【情绪面】
{context.get('sentiment_report', '无')[:800]}

【宏观面】
{context.get('macro_report', '无')[:800]}

任务:
{focus_instruction}

请以证据链组织论点，给出置信度(0-1)。在正文末尾追加机读块：
<!-- DEBATE_STATE: {{"new_claims": [{{"claim": "论点内容不超过28字", "evidence": ["证据1", "证据2"], "confidence": 0.72}}], "responded_claim_ids": [], "resolved_claim_ids": [], "round_summary": "不超过50字"}} -->

请全程使用中文。"""

        response = self.llm_call(prompt)
        return self._parse_claims_from_response(response, stance="bear", round_idx=round_idx)

    def _parse_claims_from_response(
        self, response: str, stance: str, round_idx: int
    ) -> list[Claim]:
        """从 LLM 响应中解析 Claims / Parse claims from LLM response."""
        claims: list[Claim] = []

        # Extract DEBATE_STATE block
        m = re.search(r'<!--\s*DEBATE_STATE:\s*(\{.*?\})\s*-->', response, re.DOTALL)
        if not m:
            # Fallback: treat whole response as a single claim
            with self._counter_lock:
                self._claim_counter += 1
                claim_id = f"INV-{self._claim_counter}"
            claims.append(Claim(
                claim_id=claim_id,
                agent_id=f"{stance.upper()}Researcher",
                stance=stance,
                content=response[:200].strip(),
                evidence=[],
                confidence=0.6,
                round_index=round_idx,
            ))
            return claims

        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            return claims

        for item in payload.get("new_claims", []) or []:
            with self._counter_lock:
                self._claim_counter += 1
                claim_id = f"INV-{self._claim_counter}"
            claim_text = str(item.get("claim", "")).strip()
            if not claim_text:
                continue
            evidence = [str(e).strip() for e in (item.get("evidence") or []) if str(e).strip()]
            confidence = float(item.get("confidence", 0.6))
            confidence = max(0.0, min(1.0, confidence))

            claims.append(Claim(
                claim_id=claim_id,
                agent_id=f"{stance.upper()}Researcher",
                stance=stance,
                content=claim_text,
                evidence=evidence,
                confidence=confidence,
                round_index=round_idx,
            ))

        return claims

    def _judge_debate(
        self,
        bull_claims: list[Claim],
        bear_claims: list[Claim],
        context: dict[str, str],
    ) -> DebateOutcome:
        """裁判辩论并输出结果 / Judge debate and output outcome."""
        bull_avg = sum(c.confidence for c in bull_claims) / max(len(bull_claims), 1)
        bear_avg = sum(c.confidence for c in bear_claims) / max(len(bear_claims), 1)

        diff = abs(bull_avg - bear_avg)
        if diff < 0.05:
            winner = "tie"
            signal = "中性"
            confidence = 0.5
        elif bull_avg > bear_avg:
            winner = "bull"
            signal = "看多"
            confidence = min(1.0, bull_avg)
        else:
            winner = "bear"
            signal = "看空"
            confidence = min(1.0, bear_avg)

        # LLM 裁判（可选）
        try:
            verdict_text = self._llm_judge(bull_claims, bear_claims, context)
        except Exception:
            verdict_text = f"多空对决：多头置信度 {bull_avg:.2f} vs 空头置信度 {bear_avg:.2f}，{winner.upper()} 方胜出"

        return DebateOutcome(
            topic=f"A股辩论: {self.researcher.ticker}",
            bull_claims=bull_claims,
            bear_claims=bear_claims,
            winner=winner,
            bull_score=round(bull_avg, 4),
            bear_score=round(bear_avg, 4),
            consensus=f"多头平均置信度 {bull_avg:.2f}，空头平均置信度 {bear_avg:.2f}",
            final_verdict=verdict_text,
            confidence=round(confidence, 4),
            signal=signal,
        )

    def _llm_judge(self, bull_claims: list[Claim], bear_claims: list[Claim],
                   context: dict) -> str:
        """使用 LLM 进行裁判 / LLM-based judge."""
        bull_text = "\n".join(
            f"- [{c.claim_id}] {c.content} (conf={c.confidence:.2f})"
            for c in bull_claims
        )
        bear_text = "\n".join(
            f"- [{c.claim_id}] {c.content} (conf={c.confidence:.2f})"
            for c in bear_claims
        )

        prompt = f"""辩论主题: A 股 {self.researcher.ticker} 多空辩论

多头论点:
{bull_text}

空头论点:
{bear_text}

市场背景:
{context.get('market_report', '无')[:500]}

请进行裁判评分，返回 JSON 格式：
{{"winner": "bull" | "bear" | "tie", "consensus": "共识摘要", "final_verdict": "最终判决文本", "confidence": 0到1之间"}}

只返回 JSON，不要有其他文字。"""

        response = self.llm_call(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(response[start:end])
                winner = data.get("winner", "tie")
                consensus = data.get("consensus", "")
                final_verdict = data.get("final_verdict", "")
                confidence = float(data.get("confidence", 0.5))
                return final_verdict or f"{winner.upper()} 方胜出"
            except (json.JSONDecodeError, ValueError):
                pass

        return f"多头置信度 {sum(c.confidence for c in bull_claims)/max(len(bull_claims),1):.2f} vs 空头置信度 {sum(c.confidence for c in bear_claims)/max(len(bear_claims),1):.2f}"


# ---------------------------------------------------------------------------
# PortfolioConstructor — builds portfolio from debate outcomes
# ---------------------------------------------------------------------------

# Weight calculation constants for portfolio construction
# Bull weight: 80% from score ratio + 20% from confidence (max 95%)
_BULL_WEIGHT_RATIO = 0.8
_BULL_WEIGHT_CONFIDENCE = 0.2
_BULL_MAX_WEIGHT = 0.95
# Bear weight: 50% of ratio (max 50%, lower due to short selling risk)
_BEAR_WEIGHT_RATIO = 0.5
_BEAR_MAX_WEIGHT = 0.50


class PortfolioConstructor:
    """组合构建器 / Portfolio constructor from debate outcomes.

    基于辩论结果 (DebateOutcome) 构建投资组合持仓方案。

    Args:
        llm_call: LLM 调用函数
        max_positions: 最大持仓数量 (default 5)
        default_capital: 默认资金规模 (default 1000000)
    """

    def __init__(
        self,
        llm_call: Callable[[str], str],
        max_positions: int = 5,
        default_capital: float = 1_000_000.0,
    ):
        self.llm_call = llm_call
        self.max_positions = max_positions
        self.default_capital = default_capital

    def construct_portfolio(
        self,
        outcome: DebateOutcome,
        ticker: str,
        horizon: str = "medium",
    ) -> list[PortfolioPosition]:
        """从辩论结果构建组合 / Construct portfolio from debate outcome.

        Args:
            outcome: BullBearDebate 的辩论结果
            ticker: 股票代码
            horizon: 时间窗口

        Returns:
            list[PortfolioPosition]: 持仓列表
        """
        bull_avg = outcome.bull_score
        bear_avg = outcome.bear_score
        total = bull_avg + bear_avg

        if total < 0.01:
            # No meaningful signal
            return [PortfolioPosition(
                ticker=ticker,
                direction="neutral",
                weight=0.0,
                confidence=outcome.confidence,
                thesis="辩论无明确信号，组合空仓",
            )]

        # Long position if bull > bear
        if outcome.winner == "bull":
            direction = "long"
            weight = min(_BULL_MAX_WEIGHT, bull_avg / total * _BULL_WEIGHT_RATIO + outcome.confidence * _BULL_WEIGHT_CONFIDENCE)
        elif outcome.winner == "bear":
            direction = "short"
            weight = min(_BEAR_MAX_WEIGHT, bear_avg / total * _BEAR_WEIGHT_RATIO)
        else:
            direction = "neutral"
            weight = 0.1

        weight = max(0.0, min(1.0, weight))

        thesis_parts = []
        if outcome.bull_claims:
            best_bull = max(outcome.bull_claims, key=lambda c: c.confidence)
            thesis_parts.append(f"多头核心: {best_bull.content}")
        if outcome.bear_claims:
            best_bear = max(outcome.bear_claims, key=lambda c: c.confidence)
            thesis_parts.append(f"空头核心: {best_bear.content}")

        position = PortfolioPosition(
            ticker=ticker,
            direction=direction,
            weight=round(weight, 4),
            confidence=outcome.confidence,
            thesis=" | ".join(thesis_parts) if thesis_parts else outcome.final_verdict,
        )

        return [position]

    def build_trading_signal(
        self,
        outcome: DebateOutcome,
        ticker: str,
        horizon: str = "medium",
    ) -> TradingSignal:
        """从辩论结果生成交易信号 / Generate trading signal from debate outcome.

        Args:
            outcome: BullBearDebate 的辩论结果
            ticker: 股票代码
            horizon: 时间窗口

        Returns:
            TradingSignal: 交易信号
        """
        positions = self.construct_portfolio(outcome, ticker, horizon)
        position = positions[0]

        if position.direction == "long":
            action = "BUY"
        elif position.direction == "short":
            action = "SELL"
        else:
            action = "HOLD"

        thesis_parts = []
        if outcome.bull_claims:
            top_bull = sorted(outcome.bull_claims, key=lambda c: c.confidence, reverse=True)[:2]
            thesis_parts.append("多头: " + "; ".join(c.content for c in top_bull))
        if outcome.bear_claims:
            top_bear = sorted(outcome.bear_claims, key=lambda c: c.confidence, reverse=True)[:2]
            thesis_parts.append("空头: " + "; ".join(c.content for c in top_bear))

        risk_parts = []
        if outcome.bear_claims:
            top_bear = sorted(outcome.bear_claims, key=lambda c: c.confidence, reverse=True)[:1]
            risk_parts.append("空头风险: " + "; ".join(c.content for c in top_bear))
        risk_parts.append(f"置信度: {outcome.confidence:.2f}")

        return TradingSignal(
            ticker=ticker,
            action=action,
            direction=position.direction,
            confidence=outcome.confidence,
            horizon=horizon,
            thesis="\n".join(thesis_parts) if thesis_parts else outcome.final_verdict,
            risks="\n".join(risk_parts),
            indicators={
                "bull_score": str(outcome.bull_score),
                "bear_score": str(outcome.bear_score),
                "winner": outcome.winner,
            },
        )


# ---------------------------------------------------------------------------
# TradingAgentsAShareCoordinator — orchestrates the full 15-agent team
# ---------------------------------------------------------------------------

class TradingAgentsAShareCoordinator:
    """A 股 15 Agent 辩论协调器 / A-share 15-agent debate coordinator.

    协调完整的 A 股投资研究辩论流程：

    15 Agent 团队构成:
    1. MarketAnalyst (技术面)
    2. FundamentalAnalyst (基本面)
    3. TechnicalAnalyst (技术指标, alias for Market)
    4. SentimentAnalyst (情绪面)
    5. MacroAnalyst (宏观/板块)
    6. BullResearcher (多头研究员)
    7. BearResearcher (空头研究员)
    8-15. Extended researchers (行业/资金/政策等专家角色，由 LLM 模拟)

    流程:
    1. AShareResearcher 并行运行 4 个分析师
    2. BullBearDebate 进行 N 轮多空辩论
    3. ClaimValidator 验证 Claims
    4. PortfolioConstructor 构建组合
    5. 输出 TradingSignal

    Args:
        llm_call: LLM 调用函数
        ticker: 股票代码
        trade_date: 交易日期 (YYYY-MM-DD)
        horizon: 分析周期 (short/medium)
        n_debate_rounds: 辩论轮数 (default 3)
    """

    def __init__(
        self,
        llm_call: Callable[[str], str],
        ticker: str,
        trade_date: str,
        horizon: str = "medium",
        n_debate_rounds: int = 3,
    ):
        # Validate ticker format (A-share: 6-digit codes like 000001, 600000)
        if not re.match(r"^\d{6}$", ticker):
            raise ValueError(f"Invalid A-share ticker format: {ticker}. Expected 6-digit code (e.g., '000001', '600000')")

        # Validate trade_date format (YYYY-MM-DD)
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", trade_date):
            raise ValueError(f"Invalid trade_date format: {trade_date}. Expected YYYY-MM-DD")

        # Validate horizon
        if horizon not in ("short", "medium"):
            raise ValueError(f"Invalid horizon: {horizon}. Expected 'short' or 'medium'")

        self.llm_call = llm_call
        self.ticker = ticker
        self.trade_date = trade_date
        self.horizon = horizon
        self.n_debate_rounds = n_debate_rounds

        self.researcher = AShareResearcher(
            llm_call=llm_call,
            ticker=ticker,
            trade_date=trade_date,
            horizon=horizon,
        )
        self.claim_validator = ClaimValidator(llm_call, self.researcher)
        self.debate = BullBearDebate(
            llm_call=llm_call,
            researcher=self.researcher,
            n_rounds=n_debate_rounds,
            claim_validator=self.claim_validator,
        )
        self.portfolio_constructor = PortfolioConstructor(llm_call=llm_call)

        # 结果存储
        self.analyst_reports: dict[str, AShareAnalystReport] = {}
        self.debate_outcome: Optional[DebateOutcome] = None
        self.trading_signal: Optional[TradingSignal] = None

    def run(self) -> TradingSignal:
        """运行完整流程 / Run the full 15-agent pipeline.

        Returns:
            TradingSignal: 最终交易信号
        """
        # Step 1: 并行分析师研究
        logger.info(f"[Coordinator] Running analyst research for {self.ticker}...")
        self.analyst_reports = self.researcher.run_all_analysts()
        for key, report in self.analyst_reports.items():
            logger.info(f"[Coordinator] {report.agent}: {report.verdict} (conf={report.confidence})")

        # Step 2: 多空辩论
        logger.info(f"[Coordinator] Running {self.n_debate_rounds}-round debate...")
        self.debate_outcome = self.debate.run_debate()
        logger.info(f"[Coordinator] Debate winner: {self.debate_outcome.winner} "
              f"(bull={self.debate_outcome.bull_score:.3f} "
              f"bear={self.debate_outcome.bear_score:.3f})")

        # Step 3: 构建交易信号
        logger.info(f"[Coordinator] Constructing portfolio...")
        self.trading_signal = self.portfolio_constructor.build_trading_signal(
            self.debate_outcome, self.ticker, self.horizon
        )
        logger.info(f"[Coordinator] Signal: {self.trading_signal.action} "
              f"{self.ticker} conf={self.trading_signal.confidence:.2f}")

        return self.trading_signal

    def get_full_report(self) -> dict[str, Any]:
        """获取完整报告 / Get full analysis report.

        Returns:
            dict: 包含所有分析师报告、辩论结果、交易信号的完整报告
        """
        return {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "horizon": self.horizon,
            "analyst_reports": {
                k: v.to_dict() for k, v in self.analyst_reports.items()
            } if self.analyst_reports else {},
            "debate_outcome": self.debate_outcome.to_dict() if self.debate_outcome else {},
            "trading_signal": self.trading_signal.to_dict() if self.trading_signal else {},
        }

    def print_summary(self) -> None:
        """打印分析摘要 / Print analysis summary."""
        print("\n" + "=" * 60)
        print(f"A-Share Multi-Agent Analysis: {self.ticker} ({self.trade_date})")
        print("=" * 60)

        print("\n--- Analyst Reports ---")
        for key, report in self.analyst_reports.items():
            print(f"  [{report.agent}] {report.verdict} (conf={report.confidence})")

        if self.debate_outcome:
            print(f"\n--- Debate Outcome ---")
            print(f"  Winner: {self.debate_outcome.winner}")
            print(f"  Bull Score: {self.debate_outcome.bull_score:.4f}")
            print(f"  Bear Score: {self.debate_outcome.bear_score:.4f}")
            print(f"  Confidence: {self.debate_outcome.confidence:.4f}")
            print(f"  Final Verdict: {self.debate_outcome.final_verdict[:100]}")

        if self.trading_signal:
            print(f"\n--- Trading Signal ---")
            print(f"  Action: {self.trading_signal.action}")
            print(f"  Direction: {self.trading_signal.direction}")
            print(f"  Confidence: {self.trading_signal.confidence:.4f}")
            print(f"  Thesis: {self.trading_signal.thesis[:200]}")

        print("\n" + "=" * 60)
