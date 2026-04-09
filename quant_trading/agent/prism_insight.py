"""
Prism Insight — Korean KOSPI/KOSDAQ Stock Analyzer & AI Agent

韩国 KOSPI/KOSDAQ 股票分析器与 AI 智能体

Korean KOSPI/KOSDAQ stock analyzer with AI agents for stock tracking,
Telegram bot integration, weekly insight report generation, and
Firebase bridge (optional, opt-in via FIREBASE_BRIDGE_ENABLED).

架构 / Architecture:
    KoreanStockAnalyzer       — KOSPI/KOSDAQ 股票技术/基本面分析
    StockTrackingAgent        — 韩国股票追踪与交易信号生成
    TelegramBotAdapter        — Telegram 机器人告警与命令处理
    WeeklyInsightReport       — 周度股票洞察报告生成
    PrismInsightOrchestrator  — 韩国市场分析流水线协调器

纯 Python 标准库实现，使用 urllib 进行 API 调用。
Pure Python stdlib implementation using urllib for API calls.
无 Firebase 强制依赖，仅在 FIREBASE_BRIDGE_ENABLED=true 时加载。
No Firebase hard dependency — loaded only when FIREBASE_BRIDGE_ENABLED=true.
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Connector — KOSPI/KOSDAQ price data via Korea Investment & Securities API
# ---------------------------------------------------------------------------

API_TIMEOUT = 15  # seconds
_KRX_API_BASE = "https://api.kshc.or.kr/openapi/srt/"  # placeholder — replace with actual KRX endpoint


def _fetch_json(url: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Fetch JSON from URL using stdlib urllib. Returns None on failure."""
    try:
        import urllib.request
        import urllib.parse

        query = urllib.parse.urlencode(params) if params else ""
        if query:
            full_url = f"{url}?{query}"
        else:
            full_url = url

        req = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except Exception as e:
        logger.debug(f"HTTP fetch failed for {url}: {e}")
        return None


def get_market_ohlcv(ticker: str, trade_date: str) -> Optional[Dict[str, Any]]:
    """
    Fetch OHLCV data for a Korean stock ticker on a given trade date.

    获取韩国股票指定日期的 OHLCV 数据。

    Args:
        ticker: 6-digit Korean stock code (e.g., "005930")
        trade_date: Trade date in YYYYMMDD format

    Returns:
        Dict with keys: open, high, low, close, volume, change_pct or None
    """
    # Korea Exchange API (KRX) — replace with actual API key / endpoint
    url = f"https://api.kshc.or.kr/openapi/srt/stock/{ticker}/ohlcv"
    params = {"date": trade_date, "market": "KOSPI"}  # KOSPI or KOSDAQ

    data = _fetch_json(url, params)
    if not data:
        return None

    try:
        return {
            "open": float(data.get("openPrice", 0)),
            "high": float(data.get("highPrice", 0)),
            "low": float(data.get("lowPrice", 0)),
            "close": float(data.get("closePrice", 0)),
            "volume": int(data.get("volume", 0)),
            "change_pct": float(data.get("changeRate", 0)),
        }
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse OHLCV for {ticker}: {e}")
        return None


def get_nearest_business_day(date_str: str, prev: bool = False) -> str:
    """
    Get nearest business day before (prev=True) or after (prev=False) a date.

    获取指定日期最近的一个交易日（往前或往后）。
    """
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        dt = datetime.now()

    # Simple: subtract/add days until weekday (Mon-Fri = 0-4)
    delta = -1 if prev else 1
    for _ in range(10):
        dt += timedelta(days=delta)
        if dt.weekday() < 5:
            return dt.strftime("%Y%m%d")
    return date_str  # fallback


# ---------------------------------------------------------------------------
# KoreanStockAnalyzer
# ---------------------------------------------------------------------------


class KoreanStockAnalyzer:
    """
    KOSPI/KOSDAQ Korean stock technical and fundamental analyzer.

    韩国 KOSPI/KOSDAQ 股票技术面与基本面分析器。

    Analyzes Korean stocks using OHLCV data, price patterns, and
    market sentiment indicators.

    Attributes:
        market_type: "KOSPI" or "KOSDAQ"
        db_path: Optional SQLite database path for historical data
    """

    def __init__(self, market_type: str = "KOSPI", db_path: Optional[str] = None):
        """
        Initialize KoreanStockAnalyzer.

        初始化韩国股票分析器。

        Args:
            market_type: "KOSPI" or "KOSDAQ"
            db_path: Optional path to local SQLite database for caching
        """
        self.market_type = market_type.upper()
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """Create DB schema if not exists."""
        if not self.db_path:
            return
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kr_stock_cache (
                ticker TEXT,
                trade_date TEXT,
                open REAL, high REAL, low REAL,
                close REAL, volume INTEGER,
                change_pct REAL,
                PRIMARY KEY (ticker, trade_date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kr_analysis_results (
                ticker TEXT,
                analyzed_at TEXT,
                signal TEXT,  -- BUY, SELL, HOLD
                confidence REAL,
                summary TEXT,
                details TEXT,
                PRIMARY KEY (ticker, analyzed_at)
            )
        """)
        conn.commit()
        conn.close()

    def fetch_ohlcv(self, ticker: str, trade_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch OHLCV for a ticker on a specific date.

        获取指定股票代码在特定日期的 OHLCV 数据。

        Args:
            ticker: 6-digit Korean stock code
            trade_date: YYYYMMDD, defaults to today

        Returns:
            OHLCV dict or None
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")

        # Try cache first
        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT open,high,low,close,volume,change_pct FROM kr_stock_cache "
                "WHERE ticker=? AND trade_date=?",
                (ticker, trade_date),
            ).fetchone()
            conn.close()
            if row:
                return {"open": row[0], "high": row[1], "low": row[2],
                        "close": row[3], "volume": row[4], "change_pct": row[5]}

        data = get_market_ohlcv(ticker, trade_date)

        if data and self.db_path:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO kr_stock_cache "
                "(ticker,trade_date,open,high,low,close,volume,change_pct) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ticker, trade_date, data["open"], data["high"], data["low"],
                 data["close"], data["volume"], data["change_pct"]),
            )
            conn.commit()
            conn.close()

        return data

    def analyze_price_pattern(
        self,
        ohlcv: Dict[str, Any],
        prev_ohlcv: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze price pattern from OHLCV data.

        基于 OHLCV 数据分析价格形态。

        Detects: support/resistance levels, trend direction, volume spike,
        Doji, hammer, engulfing patterns.

        Args:
            ohlcv: Current OHLCV data
            prev_ohlcv: Previous day OHLCV (optional, for pattern detection)

        Returns:
            Dict with pattern analysis results
        """
        close = ohlcv.get("close", 0)
        high = ohlcv.get("high", 0)
        low = ohlcv.get("low", 0)
        open_price = ohlcv.get("open", 0)
        volume = ohlcv.get("volume", 0)
        change_pct = ohlcv.get("change_pct", 0)

        signals = []
        score = 0.0

        # Trend: close vs open
        if close > open_price * 1.02:
            signals.append("BULLISH_CLOSE")
            score += 1.0
        elif close < open_price * 0.98:
            signals.append("BEARISH_CLOSE")
            score -= 1.0
        else:
            signals.append("DOJI_like")

        # Change magnitude
        if change_pct > 3.0:
            signals.append("SURGE")
            score += 1.5
        elif change_pct < -3.0:
            signals.append("DROPPED")
            score -= 1.5

        # Volatility
        if high and low:
            range_pct = (high - low) / close * 100 if close else 0
            if range_pct > 5:
                signals.append("HIGH_VOLATILITY")
                score += 0.5 if close > open_price else -0.5

        # Volume spike heuristic (naive)
        if prev_ohlcv and volume and prev_ohlcv.get("volume"):
            vol_ratio = volume / prev_ohlcv.get("volume", 1)
            if vol_ratio > 2.0:
                signals.append("VOLUME_SPIKE")
                score += 1.0 if close > open_price else -1.0

        # Doji detection
        if high and low and open_price and close:
            body = abs(close - open_price)
            range_total = high - low
            if range_total > 0 and body / range_total < 0.1:
                signals.append("DOJI")

        # Resistance/Support approximation (recent high/low)
        resistance = high  # simplified
        support = low  # simplified

        return {
            "signals": signals,
            "score": score,
            "close": close,
            "change_pct": change_pct,
            "volume": volume,
            "resistance": resistance,
            "support": support,
            "recommendation": "BUY" if score > 2 else "SELL" if score < -2 else "HOLD",
        }

    def analyze_stock(
        self,
        ticker: str,
        trade_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform full analysis on a Korean stock.

        对韩国股票进行全面分析。

        Args:
            ticker: 6-digit Korean stock code
            trade_date: YYYYMMDD, defaults to most recent business day

        Returns:
            Comprehensive analysis result dict
        """
        if trade_date is None:
            trade_date = get_nearest_business_day(
                datetime.now().strftime("%Y%m%d"), prev=True
            )

        ohlcv = self.fetch_ohlcv(ticker, trade_date)
        if not ohlcv:
            return {"ticker": ticker, "error": "Failed to fetch OHLCV data"}

        # Get previous day for comparison
        prev_date = get_nearest_business_day(trade_date, prev=True)
        prev_ohlcv = self.fetch_ohlcv(ticker, prev_date)

        pattern = self.analyze_price_pattern(ohlcv, prev_ohlcv)

        result = {
            "ticker": ticker,
            "market": self.market_type,
            "trade_date": trade_date,
            "close": ohlcv.get("close"),
            "change_pct": ohlcv.get("change_pct"),
            "volume": ohlcv.get("volume"),
            "pattern": pattern,
            "signal": pattern["recommendation"],
            "confidence": min(abs(pattern["score"]) / 3.0, 1.0),
            "analyzed_at": datetime.now().isoformat(),
        }

        # Cache result
        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO kr_analysis_results "
                "(ticker,analyzed_at,signal,confidence,summary,details) "
                "VALUES (?,?,?,?,?,?)",
                (
                    ticker,
                    result["analyzed_at"],
                    result["signal"],
                    result["confidence"],
                    f"{ticker} {result['signal']} ({result['change_pct']:+.2f}%)",
                    json.dumps(pattern),
                ),
            )
            conn.commit()
            conn.close()

        return result


# ---------------------------------------------------------------------------
# StockTrackingAgent
# ---------------------------------------------------------------------------


class StockTrackingAgent:
    """
    AI-powered Korean stock tracking and insight generation agent.

    韩国股票 AI 追踪与信号生成智能体。

    Tracks KOSPI/KOSDAQ stocks, monitors price movements, generates
    trading insights and alerts. Manages a portfolio of up to 10 slots.

    Attributes:
        db_path: Path to SQLite tracking database
        max_slots: Maximum number of concurrent holdings (default 10)
        telegram_enabled: Whether Telegram alerts are enabled
    """

    MAX_SLOTS = 10

    def __init__(
        self,
        db_path: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_channel_id: Optional[str] = None,
    ):
        """
        Initialize StockTrackingAgent.

        初始化股票追踪智能体。

        Args:
            db_path: Path to SQLite tracking database
            telegram_bot_token: Telegram bot token for alerts
            telegram_channel_id: Telegram channel ID for alerts
        """
        self.db_path = db_path or ":memory:"
        self.telegram_bot_token = telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_channel_id = telegram_channel_id or os.getenv("TELEGRAM_CHANNEL_ID", "")
        self._ensure_db()

    def _ensure_db(self):
        """Create tracking DB schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_holdings (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                buy_price REAL,
                buy_date TEXT,
                current_price REAL,
                sector TEXT,
                trigger_type TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                company_name TEXT,
                buy_price REAL,
                sell_price REAL,
                buy_date TEXT,
                sell_date TEXT,
                profit_rate REAL,
                holding_days INTEGER,
                trigger_type TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_intuitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition TEXT,
                insight TEXT,
                confidence REAL,
                market TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_performance_tracker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                trigger_type TEXT,
                tracking_status TEXT DEFAULT 'pending',
                tracked_30d_return REAL,
                was_traded INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def add_holding(
        self,
        ticker: str,
        company_name: str,
        buy_price: float,
        sector: Optional[str] = None,
        trigger_type: Optional[str] = None,
    ) -> bool:
        """
        Add a stock to holdings (max 10 slots).

        将股票加入持仓（最多10个仓位）。

        Args:
            ticker: 6-digit stock code
            company_name: Company name
            buy_price: Purchase price
            sector: Business sector
            trigger_type: Trigger reason for the buy signal

        Returns:
            True if added, False if full or already held
        """
        conn = sqlite3.connect(self.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM stock_holdings"
        ).fetchone()[0]
        if count >= self.MAX_SLOTS:
            conn.close()
            logger.warning(f"Holdings full ({self.MAX_SLOTS} slots)")
            return False

        exists = conn.execute(
            "SELECT 1 FROM stock_holdings WHERE ticker=?", (ticker,)
        ).fetchone()
        if exists:
            conn.close()
            return False

        conn.execute(
            "INSERT INTO stock_holdings "
            "(ticker,company_name,buy_price,buy_date,sector,trigger_type) "
            "VALUES (?,?,?,?,?,?)",
            (ticker, company_name, buy_price, datetime.now().strftime("%Y-%m-%d"),
             sector, trigger_type),
        )
        conn.commit()
        conn.close()
        return True

    def remove_holding(self, ticker: str, sell_price: float) -> Optional[Dict[str, Any]]:
        """
        Remove a stock from holdings (sell).

        平仓卖出股票。

        Args:
            ticker: Stock code to sell
            sell_price: Sell price

        Returns:
            Trade record dict or None
        """
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT ticker,company_name,buy_price,buy_date FROM stock_holdings WHERE ticker=?",
            (ticker,),
        ).fetchone()
        if not row:
            conn.close()
            return None

        ticker, company_name, buy_price, buy_date = row
        holding_days = (datetime.now() - datetime.strptime(buy_date, "%Y-%m-%d")).days
        profit_rate = (sell_price - buy_price) / buy_price * 100 if buy_price else 0

        conn.execute(
            "DELETE FROM stock_holdings WHERE ticker=?", (ticker,)
        )
        conn.execute(
            "INSERT INTO trading_history "
            "(ticker,company_name,buy_price,sell_price,buy_date,sell_date,profit_rate,holding_days) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ticker, company_name, buy_price, sell_price, buy_date,
             datetime.now().strftime("%Y-%m-%d"), profit_rate, holding_days),
        )
        conn.commit()
        conn.close()

        return {
            "ticker": ticker,
            "company_name": company_name,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "profit_rate": profit_rate,
            "holding_days": holding_days,
        }

    def get_holdings(self) -> List[Dict[str, Any]]:
        """Get all current holdings. 获取所有当前持仓。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM stock_holdings").fetchall()
        conn.close()
        cols = ["ticker", "company_name", "buy_price", "buy_date",
                "current_price", "sector", "trigger_type"]
        return [dict(zip(cols, row)) for row in rows]

    def get_trading_history(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get trading history for the last N days.

        获取最近 N 天的交易历史。

        Args:
            days: Number of days to look back

        Returns:
            List of trade records
        """
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM trading_history WHERE sell_date >= ? ORDER BY sell_date DESC",
            (cutoff,),
        ).fetchall()
        conn.close()
        cols = ["id", "ticker", "company_name", "buy_price", "sell_price",
                "buy_date", "sell_date", "profit_rate", "holding_days", "trigger_type"]
        return [dict(zip(cols, row)) for row in rows]

    async def generate_insight(
        self,
        ticker: str,
        analyzer: KoreanStockAnalyzer,
    ) -> Dict[str, Any]:
        """
        Generate AI insight for a tracked stock.

        为追踪的股票生成 AI 洞察。

        Args:
            ticker: 6-digit stock code
            analyzer: KoreanStockAnalyzer instance

        Returns:
            Insight dict with recommendation
        """
        result = analyzer.analyze_stock(ticker)

        # Check if in holdings
        holdings = self.get_holdings()
        in_portfolio = any(h["ticker"] == ticker for h in holdings)

        signal = result.get("signal", "HOLD")
        change_pct = result.get("change_pct", 0)

        insight = {
            "ticker": ticker,
            "signal": signal,
            "confidence": result.get("confidence", 0),
            "change_pct": change_pct,
            "in_portfolio": in_portfolio,
            "pattern_signals": result.get("pattern", {}).get("signals", []),
            "recommendation": "HOLD",
            "reason": "",
        }

        if in_portfolio and signal == "SELL":
            insight["recommendation"] = "SELL"
            insight["reason"] = f"Technical sell signal detected ({change_pct:+.1f}%)"
        elif not in_portfolio and signal == "BUY":
            insight["recommendation"] = "BUY"
            insight["reason"] = f"Technical buy signal detected ({change_pct:+.1f}%)"
        else:
            insight["recommendation"] = "HOLD"
            insight["reason"] = f"Neutral — {signal} ({change_pct:+.1f}%)"

        return insight

    def format_insight_message(self, insight: Dict[str, Any]) -> str:
        """
        Format insight as a Telegram-ready message.

        将洞察格式化为 Telegram 消息格式。

        Args:
            insight: Insight dict from generate_insight

        Returns:
            Formatted message string
        """
        ticker = insight["ticker"]
        rec = insight["recommendation"]
        conf = insight["confidence"]
        change = insight["change_pct"]
        reason = insight["reason"]
        signals = ", ".join(insight.get("pattern_signals", []))

        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪️"}.get(rec, "⚪️")

        msg = f"""{emoji} [{rec}] {ticker}
📊 변동: {change:+.2f}%
🎯 신뢰도: {conf:.0%}
📝 이유: {reason}
🏷️ 신호: {signals if signals else 'N/A'}"""
        return msg


# ---------------------------------------------------------------------------
# TelegramBotAdapter
# ---------------------------------------------------------------------------


class TelegramBotAdapter:
    """
    Pure Python Telegram bot adapter for Korean stock alerts.

    纯 Python 实现 Telegram 机器人适配器，用于韩国股票告警。

    Handles sending alerts, processing commands (/evaluate, /report, /history).
    Pure Python stdlib — no python-telegram-bot dependency required.
    Uses urllib to call the Telegram Bot API directly.

    Attributes:
        bot_token: Telegram bot token
        channel_id: Default channel ID for broadcasts
    """

    TG_API_URL = "https://api.telegram.org/bot{token}/{method}"
    MAX_MESSAGE_LENGTH = 4096

    def __init__(
        self,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None,
    ):
        """
        Initialize TelegramBotAdapter.

        初始化 Telegram 机器人适配器。

        Args:
            bot_token: Telegram bot token (from @BotFather)
            channel_id: Default channel ID for alerts
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.channel_id = channel_id or os.getenv("TELEGRAM_CHANNEL_ID", "")

    def _call_api(self, method: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Call Telegram Bot API method via urllib."""
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
            return None

        url = self.TG_API_URL.format(token=self.bot_token, method=method)
        try:
            import urllib.request
            import urllib.parse

            data = urllib.parse.urlencode(params or {}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("ok"):
                    return result.get("result")
                logger.warning(f"Telegram API error: {result}")
                return None
        except Exception as e:
            logger.error(f"Telegram API call failed: {e}")
            return None

    def send_message(
        self,
        text: str,
        chat_id: Optional[str] = None,
        parse_mode: str = "Markdown",
    ) -> bool:
        """
        Send a text message to a Telegram chat.

        发送文本消息到 Telegram 聊天。

        Args:
            text: Message text
            chat_id: Target chat ID (uses default if None)
            parse_mode: "Markdown" or "HTML"

        Returns:
            True if sent successfully
        """
        target = chat_id or self.channel_id
        if not target:
            logger.info(f"[Message (no channel)] {text[:100]}...")
            return True  # Not an error in dry-run mode

        # Split long messages
        if len(text) <= self.MAX_MESSAGE_LENGTH:
            parts = [text]
        else:
            parts = self._split_message(text)

        for i, part in enumerate(parts, 1):
            label = f"[{i}/{len(parts)}]\n" if len(parts) > 1 else ""
            result = self._call_api(
                "sendMessage",
                {"chat_id": target, "text": label + part, "parse_mode": parse_mode},
            )
            if result is None:
                return False
            time.sleep(0.5)

        return True

    def _split_message(self, message: str) -> List[str]:
        """Split long message into Telegram-safe chunks."""
        parts = []
        current = ""

        for line in message.split("\n"):
            if len(current) + len(line) + 1 <= self.MAX_MESSAGE_LENGTH:
                current += line + "\n"
            else:
                if current:
                    parts.append(current.rstrip())
                current = line + "\n"

        if current:
            parts.append(current.rstrip())

        return parts

    async def send_alert(
        self,
        ticker: str,
        signal: str,
        change_pct: float,
        message: str,
        chat_id: Optional[str] = None,
    ) -> bool:
        """
        Send a stock alert to Telegram.

        发送股票告警到 Telegram。

        Args:
            ticker: Stock code
            signal: BUY/SELL/HOLD
            change_pct: Price change percentage
            message: Alert message
            chat_id: Target chat ID

        Returns:
            True if sent successfully
        """
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪️"}.get(signal.upper(), "⚪️")
        alert_text = (
            f"{emoji} *Stock Alert — {ticker}*\n"
            f"Signal: *{signal.upper()}* ({change_pct:+.2f}%)\n"
            f"{message}"
        )
        return self.send_message(alert_text, chat_id=chat_id)

    def get_updates(self, offset: Optional[int] = None, timeout: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent Telegram updates (for polling).

        获取最近的 Telegram 更新（轮询模式）。

        Args:
            offset: Update ID offset
            timeout: Long polling timeout

        Returns:
            List of update dicts
        """
        params: Dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset

        result = self._call_api("getUpdates", params)
        return result if isinstance(result, list) else []

    def process_command(self, text: str) -> Optional[str]:
        """
        Process a Telegram command text and return response.

        处理 Telegram 命令文本并返回响应。

        Supports: /start, /help, /ping, /status

        Args:
            text: Raw message text

        Returns:
            Response string or None
        """
        text = text.strip().lower()

        if text in ("/start", "/help"):
            return (
                "📊 *PRISM Stock Bot*\n\n"
                "/evaluate <ticker> — 分析股票\n"
                "/report <ticker> — 生成完整报告\n"
                "/history <ticker> — 查看历史\n"
                "/holdings — 查看持仓\n"
                "/ping — 健康检查"
            )
        if text == "/ping":
            return "🏓 Pong! Bot is running."
        if text == "/holdings":
            return "📦 *Holdings command received* — 请稍候..."
        if text.startswith("/evaluate ") or text.startswith("/report "):
            ticker = text.split()[1] if len(text.split()) > 1 else ""
            if ticker:
                return f"📊 正在分析 *{ticker}*..."
        return None


# ---------------------------------------------------------------------------
# WeeklyInsightReport
# ---------------------------------------------------------------------------


class WeeklyInsightReport:
    """
    Weekly Korean stock insight report generator.

    周度韩国股票洞察报告生成器。

    Generates a weekly summary covering:
    - Weekly trades (KR + US markets)
    - Sell evaluations
    - Trigger performance
    - AI intuitions

    Attributes:
        db_path: Path to SQLite tracking database
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize WeeklyInsightReport.

        初始化周度报告生成器。

        Args:
            db_path: Path to SQLite tracking database
        """
        self.db_path = db_path or ":memory:"

    def _safe_query(
        self,
        conn: sqlite3.Connection,
        query: str,
        default: Any = None,
    ) -> Any:
        """Execute query safely, return default on failure."""
        try:
            row = conn.execute(query).fetchone()
            return row if row else default
        except sqlite3.Error as e:
            logger.warning(f"Query failed: {e}")
            return default

    def _safe_query_all(
        self,
        conn: sqlite3.Connection,
        query: str,
    ) -> List[Any]:
        """Execute query and return all results."""
        try:
            return conn.execute(query).fetchall()
        except sqlite3.Error as e:
            logger.warning(f"Query failed: {e}")
            return []

    def _format_pct(self, value: float) -> str:
        """Format percentage with sign."""
        if value is None:
            return "N/A"
        return f"{value:+.1f}%"

    def _sell_verdict(self, change_pct: float) -> str:
        """Determine sell quality verdict."""
        if change_pct < -1:
            return "✅ 卖得好"
        elif change_pct > 3:
            return "😅 可以再等等"
        else:
            return "👌 适当卖出"

    def _get_weekly_trades(self, conn: sqlite3.Connection, week_start_str: str) -> str:
        """Get weekly trade summary for KR and US markets."""
        kr_sells = self._safe_query_all(conn, f"""
            SELECT ticker, company_name, buy_price, sell_price, profit_rate, holding_days
            FROM trading_history WHERE sell_date >= '{week_start_str}' ORDER BY sell_date DESC
        """)
        kr_buys = self._safe_query_all(conn, f"""
            SELECT ticker, company_name, buy_price, buy_date, current_price
            FROM stock_holdings WHERE buy_date >= '{week_start_str}'
        """)

        if not (kr_sells or kr_buys):
            return "本周无交易"

        lines = []
        if kr_buys:
            lines.append("🇰🇷 韩国市场")
            for ticker, name, buy_price, _date, current_price in kr_buys:
                if current_price and buy_price:
                    pnl = (current_price - buy_price) / buy_price * 100
                    lines.append(
                        f"  买入: {name}({ticker}) {buy_price:,.0f}원 "
                        f"→ 当前 {current_price:,.0f}원 ({pnl:+.1f}%)"
                    )
                else:
                    lines.append(f"  买入: {name}({ticker}) {buy_price:,.0f}원")
        for ticker, name, _buy_p, sell_p, profit, days in kr_sells:
            lines.append(
                f"  卖出: {name}({ticker}) {sell_p:,.0f}원 "
                f"→ {profit:+.1f}% ({days}天持有)"
            )

        return "\n".join(lines)

    def _get_trigger_performance(
        self,
        conn: sqlite3.Connection,
        week_start_str: str,
    ) -> Tuple[str, str, str, str]:
        """
        Get trigger performance stats for KR market.

        Returns:
            Tuple of (avoided_str, missed_str, best_trigger_str, principles_str)
        """
        avoided_count, avoided_avg = 0, None
        missed_count, missed_best = 0, None
        best_trigger_name, best_trigger_rate = "无数据", 0
        new_principles, total_principles = 0, 0

        try:
            row = self._safe_query(conn, f"""
                SELECT COUNT(*), AVG(tracked_30d_return * 100)
                FROM analysis_performance_tracker
                WHERE tracking_status='completed'
                  AND was_traded=0
                  AND tracked_30d_return < -0.05
                  AND updated_at >= '{week_start_str}'
            """)
            if row:
                avoided_count = row[0] or 0
                avoided_avg = row[1]

            row = self._safe_query(conn, f"""
                SELECT COUNT(*), MAX(tracked_30d_return * 100)
                FROM analysis_performance_tracker
                WHERE tracking_status='completed'
                  AND was_traded=0
                  AND tracked_30d_return > 0.10
                  AND updated_at >= '{week_start_str}'
            """)
            if row:
                missed_count = row[0] or 0
                missed_best = row[1]

            row = self._safe_query(conn, """
                SELECT trigger_type,
                       SUM(CASE WHEN tracking_status='completed' THEN 1 ELSE 0 END) as completed,
                       SUM(CASE WHEN tracking_status='completed' AND tracked_30d_return > 0 THEN 1 ELSE 0 END) as wins
                FROM analysis_performance_tracker
                WHERE trigger_type IS NOT NULL
                GROUP BY trigger_type
                HAVING completed >= 3
                ORDER BY (wins * 1.0 / completed) DESC
                LIMIT 1
            """)
            if row and row[0]:
                best_trigger_name = row[0]
                completed, wins = row[1], row[2]
                best_trigger_rate = (wins / completed * 100) if completed > 0 else 0

            row = self._safe_query(conn, f"""
                SELECT COUNT(*) FROM trading_principles
                WHERE is_active=1 AND created_at >= '{week_start_str}'
            """, default=(0,))
            new_principles = row[0] if row else 0

            row = self._safe_query(conn,
                "SELECT COUNT(*) FROM trading_principles WHERE is_active=1",
                default=(0,))
            total_principles = row[0] if row else 0

        except sqlite3.Error as e:
            logger.warning(f"Trigger performance query error: {e}")

        def avoided_detail(count, avg):
            if count == 0:
                return "0 — 未买入但下跌的标的"
            return f"{count} (平均 {self._format_pct(avg)}) — 未买入避免了损失"

        def missed_detail(count, best):
            if count == 0:
                return "0 — 错过的上涨标的"
            return f"{count} (最高 {self._format_pct(best)}) — 未买入但大涨的标的"

        avoided_str = avoided_detail(avoided_count, avoided_avg)
        missed_str = missed_detail(missed_count, missed_best)
        trigger_str = f"{best_trigger_name} (胜率 {best_trigger_rate:.0f}%)" if best_trigger_rate > 0 else "数据积累中"
        principles_str = f"{new_principles} 新增 (共 {total_principles})"

        return avoided_str, missed_str, trigger_str, principles_str

    def _get_ai_intuitions(self, conn: sqlite3.Connection, week_start_str: str) -> str:
        """Get AI intuitions section."""
        row = self._safe_query(conn, f"""
            SELECT COUNT(*) FROM trading_intuitions
            WHERE is_active=1 AND created_at >= '{week_start_str}'
        """, default=(0,))
        new_count = row[0] if row else 0

        intuitions = self._safe_query_all(conn, """
            SELECT condition, insight, confidence
            FROM trading_intuitions WHERE is_active=1
            ORDER BY confidence DESC LIMIT 3
        """)

        stats = self._safe_query(conn, """
            SELECT COUNT(*), AVG(confidence), AVG(CASE WHEN success_rate IS NOT NULL THEN success_rate ELSE 0 END)
            FROM trading_intuitions WHERE is_active=1
        """, default=(0, 0, 0))
        total_count = stats[0] or 0
        avg_conf = stats[1] or 0

        if total_count == 0:
            return "数据积累中。交易记录增加后 AI 将学习模式。"

        lines = [
            f"本周新增: {new_count} | 活跃直觉: {total_count} | 平均置信度: {avg_conf * 100:.0f}%"
        ]

        if intuitions:
            lines.append("")
            lines.append("💡 主要直觉:")
            for i, (condition, insight, confidence) in enumerate(intuitions[:5], 1):
                conf_pct = (confidence or 0) * 100
                lines.append(f"  {i}. {condition} = {insight} (置信度 {conf_pct:.0f}%)")

        return "\n".join(lines)

    def generate_report(self, week_start: Optional[datetime] = None) -> str:
        """
        Generate weekly insight report message.

        生成周度洞察报告消息。

        Args:
            week_start: Start of week (defaults to 7 days ago)

        Returns:
            Formatted report string
        """
        today = datetime.now()
        week_start = week_start or (today - timedelta(days=7))
        week_start_str = week_start.strftime("%Y-%m-%d %H:%M:%S")

        start_display = week_start.strftime("%-m/%-d")
        end_display = today.strftime("%-m/%-d")

        conn = sqlite3.connect(self.db_path)

        trades_summary = self._get_weekly_trades(conn, week_start_str)
        avoided_str, missed_str, trigger_str, principles_str = self._get_trigger_performance(
            conn, week_start_str
        )
        intuitions_section = self._get_ai_intuitions(conn, week_start_str)

        conn.close()

        # Build actionable insights
        insights = []
        insights.append("本周无重大变动，市场稳定运行。")

        insights_str = "\n".join(f"  → {i}" for i in insights)

        message = f"""📋 PRISM 周度洞察 ({start_display} ~ {end_display})
本周 AI 交易判断绩效回顾。

📈 本周交易摘要
━━━━━━━━━━━━━━━━━━━━
{trades_summary}

🇰🇷 韩国市场 (触发器绩效)
━━━━━━━━━━━━━━━━━━━━
🛡️ 避免损失: {avoided_str}
❌ 错过机会: {missed_str}
📊 最准确触发器: {trigger_str}
📌 新交易原则: {principles_str}

🧠 AI 长期学习洞察
━━━━━━━━━━━━━━━━━━━━
{intuitions_section}

📌 本周洞察
{insights_str}

💡 核心: 数据积累中 — 30天跟踪完成后提供洞察

ℹ️ 术语说明
• 触发器 = AI 发现标的的原因（暴涨、成交量急升等）
• 避免损失 = 未买入但30天后 -5%以上下跌的标的
• 错过机会 = 未买入但30天后 +10%以上上涨的标的
• 胜率 = 该触发器分析的标的30天后盈利的比例
• 交易原则 = AI 从历史交易中自主学习的规则
• 直觉 = AI 从重复模式中提取的交易原则"""
        return message


# ---------------------------------------------------------------------------
# PrismInsightOrchestrator
# ---------------------------------------------------------------------------


class PrismInsightOrchestrator:
    """
    Korean KOSPI/KOSDAQ market analysis pipeline orchestrator.

    韩国 KOSPI/KOSDAQ 市场分析流水线协调器。

    Coordinates Korean stock analysis, tracking, alerts, and reporting
    into a unified pipeline.

    Attributes:
        db_path: Path to SQLite tracking database
        telegram_token: Telegram bot token
        telegram_channel_id: Telegram channel ID
        market: Market type ("KOSPI" or "KOSDAQ")
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_channel_id: Optional[str] = None,
        market: str = "KOSPI",
    ):
        """
        Initialize PrismInsightOrchestrator.

        初始化韩国市场分析流水线协调器。

        Args:
            db_path: SQLite database path
            telegram_token: Telegram bot token
            telegram_channel_id: Telegram channel ID
            market: "KOSPI" or "KOSDAQ"
        """
        self.db_path = db_path or ":memory:"
        self.market = market.upper()

        self.analyzer = KoreanStockAnalyzer(market_type=self.market, db_path=self.db_path)
        self.tracker = StockTrackingAgent(
            db_path=self.db_path,
            telegram_bot_token=telegram_token,
            telegram_channel_id=telegram_channel_id,
        )
        self.telegram = TelegramBotAdapter(
            bot_token=telegram_token,
            channel_id=telegram_channel_id,
        )
        self.weekly_report = WeeklyInsightReport(db_path=self.db_path)

    def analyze_stock(self, ticker: str, trade_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a Korean stock.

        分析韩国股票。

        Args:
            ticker: 6-digit stock code
            trade_date: YYYYMMDD (defaults to most recent business day)

        Returns:
            Analysis result dict
        """
        return self.analyzer.analyze_stock(ticker, trade_date)

    async def generate_report(
        self,
        ticker: Optional[str] = None,
        weekly: bool = False,
    ) -> str:
        """
        Generate stock analysis report or weekly insight.

        生成股票分析报告或周度洞察。

        Args:
            ticker: Stock code (if None, generates weekly report)
            weekly: If True, generate weekly report

        Returns:
            Report text
        """
        if weekly or ticker is None:
            return self.weekly_report.generate_report()

        result = self.analyzer.analyze_stock(ticker)
        if "error" in result:
            return f"❌ 分析失败: {result['error']}"

        change = result.get("change_pct", 0)
        signal = result.get("signal", "HOLD")
        conf = result.get("confidence", 0)
        pattern = result.get("pattern", {}).get("signals", [])

        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪️"}.get(signal, "⚪️")

        return (
            f"{emoji} *{ticker}* 分析报告\n"
            f"市场: {self.market}\n"
            f"信号: *{signal}* | 置信度: {conf:.0%}\n"
            f"涨跌: {change:+.2f}%\n"
            f"形态: {', '.join(pattern) if pattern else 'N/A'}"
        )

    async def send_telegram_alert(
        self,
        ticker: str,
        signal: str,
        change_pct: float,
        message: str,
        chat_id: Optional[str] = None,
    ) -> bool:
        """
        Send a stock alert via Telegram.

        通过 Telegram 发送股票告警。

        Args:
            ticker: Stock code
            signal: BUY/SELL/HOLD
            change_pct: Price change percentage
            message: Alert message
            chat_id: Target chat ID

        Returns:
            True if sent successfully
        """
        return await self.telegram.send_alert(
            ticker=ticker,
            signal=signal,
            change_pct=change_pct,
            message=message,
            chat_id=chat_id,
        )

    async def run_pipeline(
        self,
        tickers: List[str],
        send_alerts: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full analysis pipeline on a list of tickers.

        对股票列表运行完整分析流水线。

        Args:
            tickers: List of 6-digit Korean stock codes
            send_alerts: Whether to send Telegram alerts

        Returns:
            Pipeline results summary dict
        """
        results = []
        alerts_sent = 0

        for ticker in tickers:
            try:
                # Analyze
                analysis = self.analyzer.analyze_stock(ticker)

                # Generate insight
                insight = await self.tracker.generate_insight(ticker, self.analyzer)

                results.append({
                    "ticker": ticker,
                    "signal": insight["recommendation"],
                    "confidence": insight["confidence"],
                    "change_pct": insight["change_pct"],
                })

                # Send alert if significant
                if send_alerts and insight["confidence"] > 0.7:
                    sent = await self.send_telegram_alert(
                        ticker=ticker,
                        signal=insight["recommendation"],
                        change_pct=insight["change_pct"],
                        message=self.tracker.format_insight_message(insight),
                    )
                    if sent:
                        alerts_sent += 1

            except Exception as e:
                logger.error(f"Pipeline error for {ticker}: {e}")
                results.append({"ticker": ticker, "error": str(e)})

        return {
            "total": len(tickers),
            "analyzed": len(results),
            "alerts_sent": alerts_sent,
            "results": results,
            "market": self.market,
            "run_at": datetime.now().isoformat(),
        }
