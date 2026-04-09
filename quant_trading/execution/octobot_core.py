"""
OctoBot Core Adapter

适配 OctoBot 加密交易机器人框架到本系统的 execution 模块。

OctoBot 是一个功能完整的开源加密货币交易机器人，支持：
- Telegram 机器人集成（实时告警和控制）
- 多交易所连接（Binance, Coinbase, Kraken, OKX, Bybit, Gate, KuCoin 等）
- 自动化交易策略（网格、马厩、DCA、信号交易等）
- 策略优化与回测
- 社区协作与信号订阅

此适配器将 OctoBot 的核心功能（配置管理、Telegram 通知、交易对齐、执行引擎）
集成到本系统的统一架构中，使用现有连接器（如 binance_rest.py）作为后端。

主要组件：
- OctoBotConfig：OctoBot 风格交易配置
- TelegramNotifier：纯 Python urllib 的 Telegram 机器人通知（无第三方 SDK 依赖）
- TradingAligner：跨多交易所交易对齐
- OctoBotExecutor：主执行引擎

Usage / 使用:
    >>> from quant_trading.execution.octobot_core import OctoBotExecutor, TelegramNotifier
    >>> # 初始化 Telegram 通知器
    >>> notifier = TelegramNotifier(bot_token="YOUR_BOT_TOKEN", chat_id="YOUR_CHAT_ID")
    >>> await notifier.send_alert("BTC/USDT 买入信号触发 @ 65000")
    >>>
    >>> # 初始化执行引擎
    >>> config = OctoBotConfig(exchanges={"binance": {"enabled": True}})
    >>> executor = OctoBotExecutor(config, notifier=notifier)
    >>> await executor.execute_trades([{"symbol": "BTC/USDT", "side": "buy", "qty": 0.01}])
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Try importing existing connectors as backend
try:
    from quant_trading.connectors.binance_rest import BinanceRESTClient
    _HAS_BINANCE = True
except ImportError:
    BinanceRESTClient = None
    _HAS_BINANCE = False

try:
    from quant_trading.execution.executor import (
        Order as QtOrder,
        OrderSide,
        OrderStatus,
        OrderType as QtOrderType,
    )
except ImportError:
    QtOrder = None
    OrderSide = None
    OrderStatus = None
    QtOrderType = None

logger = logging.getLogger(__name__)


# ====================
# Enums / 枚举类型
# ====================

class TradeSignalType(Enum):
    """
    交易信号类型 / Trade Signal Type

    参考 OctoBot 信号格式，定义支持的信号类型。
    """
    ENTRY = "entry"           # 入场信号
    EXIT = "exit"             # 出场信号
    DCA = "dca"               # 定投平均成本 (Dollar Cost Averaging)
    GRID = "grid"             # 网格交易
    SHORT = "short"           # 做空信号
    LONG = "long"             # 做多信号
    CLOSE_SHORT = "close_short"
    CLOSE_LONG = "close_long"
    STOP_LOSS = "stop_loss"   # 止损
    TAKE_PROFIT = "take_profit"  # 止盈


class AlertLevel(Enum):
    """
    告警级别 / Alert Level

    Telegram 消息的紧急程度分级。
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ExchangeSyncMode(Enum):
    """
    交易所同步模式 / Exchange Synchronization Mode

    多交易所间的交易同步策略。
    """
    SEQUENTIAL = "sequential"   # 顺序执行（一个接一个）
    PARALLEL = "parallel"       # 并行执行
    BEST_PRICE = "best_price"  # 最佳价格优先


# ====================
# Config Dataclass / 配置数据类
# ====================

@dataclass
class OctoBotConfig:
    """
    OctoBot 风格交易配置 / OctoBot-Style Trading Configuration

    用于配置 OctoBot 风格交易的所有参数，包括：
    - 多交易所连接配置
    - Telegram 通知配置
    - 策略参数
    - 风险管理设置
    - 同步模式

    Attributes:
        exchanges (Dict[str, Dict[str, Any]]): 交易所配置映射
            示例: {"binance": {"enabled": True, "api_key": "...", "api_secret": "..."}}
        telegram_token (str): Telegram Bot Token（用于通知）
        telegram_chat_id (str): Telegram 聊天 ID（用于通知）
        strategy_params (Dict[str, Any]): 策略参数字典
            - max_position_size: 最大仓位大小
            - risk_per_trade: 每笔交易风险比例
            - default_quantity: 默认交易数量
        sync_mode (ExchangeSyncMode): 多交易所同步模式
        paper_trade (bool): 是否使用模拟交易
        timeout (int): 请求超时时间（秒）
        max_retries (int): 最大重试次数

    Example:
        >>> config = OctoBotConfig(
        ...     exchanges={
        ...         "binance": {"enabled": True, "api_key": "xxx", "api_secret": "yyy"},
        ...         "bybit": {"enabled": True, "api_key": "aaa", "api_secret": "bbb"},
        ...     },
        ...     telegram_token="123456:ABC-DEF",
        ...     telegram_chat_id="-100123456",
        ...     strategy_params={"max_position_size": 0.1, "risk_per_trade": 0.02},
        ...     sync_mode=ExchangeSyncMode.BEST_PRICE,
        ...     paper_trade=True,
        ... )
    """

    exchanges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    telegram_token: str = ""
    telegram_chat_id: str = ""
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    sync_mode: ExchangeSyncMode = ExchangeSyncMode.SEQUENTIAL
    paper_trade: bool = True
    timeout: int = 30
    max_retries: int = 3

    def get_exchange_cred(self, exchange: str) -> Dict[str, str]:
        """Get credentials for a specific exchange."""
        return self.exchanges.get(exchange.lower(), {})

    def is_exchange_enabled(self, exchange: str) -> bool:
        """Check if an exchange is enabled in config."""
        ex_config = self.exchanges.get(exchange.lower(), {})
        return ex_config.get("enabled", False)


@dataclass
class TradeOrder:
    """
    交易订单 / Trade Order

    表示一个待执行的交易订单。

    Attributes:
        symbol (str): 交易对符号 (e.g., "BTC/USDT")
        side (str): 买卖方向 ("buy" or "sell")
        quantity (float): 交易数量
        price (Optional[float]): 限价价格（None 表示市价单）
        order_type (str): 订单类型 ("market", "limit", "stop_loss", etc.)
        exchange (str): 交易所名称（None 表示自动路由）
        signal_type (TradeSignalType): 信号类型
        metadata (Dict[str, Any]): 附加元数据
    """
    symbol: str
    side: str
    quantity: float
    price: Optional[float] = None
    order_type: str = "market"
    exchange: Optional[str] = None
    signal_type: TradeSignalType = TradeSignalType.ENTRY
    metadata: Dict[str, Any] = field(default_factory=dict)
    order_id: Optional[str] = None
    status: str = "pending"
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class ExecutionReport:
    """
    执行报告 / Execution Report

    包含交易执行结果的完整信息。

    Attributes:
        success (bool): 是否执行成功
        orders (List[TradeOrder]): 订单列表
        total_quantity (float): 总交易数量
        avg_price (float): 平均成交价格
        total_cost (float): 总交易成本
        execution_time_ms (float): 执行耗时（毫秒）
        errors (List[str]): 错误信息列表
        exchange_results (Dict[str, Any]): 各交易所执行结果
    """
    success: bool
    orders: List[TradeOrder]
    total_quantity: float = 0.0
    avg_price: float = 0.0
    total_cost: float = 0.0
    execution_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    exchange_results: Dict[str, Any] = field(default_factory=dict)


# ====================
# Telegram Notifier / Telegram 通知器
# ====================

class TelegramNotifier:
    """
    Telegram 机器人通知器 / Telegram Bot Notifier

    纯 Python urllib 实现的 Telegram Bot API 通知器，无需第三方 SDK 依赖。

    Features:
    - 文本消息发送
    - Markdown 格式化消息
    - 告警级别支持（INFO/WARNING/ERROR/CRITICAL）
    - 自动重试与错误处理
    - 连接测试

    OctoBot uses Telegram extensively for:
    - Real-time trade alerts
    - Portfolio status
    - Strategy notifications
    - Emergency stop commands

    Example:
        >>> notifier = TelegramNotifier(bot_token="123456:ABC-DEF", chat_id="-100123456")
        >>> await notifier.send_alert("BUY BTC @ 65000", level=AlertLevel.INFO)
        >>> await notifier.send_alert("Stop loss triggered!", level=AlertLevel.CRITICAL)
        >>> await notifier.send_trade_report(orders, execution_report)
    """

    BASE_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        timeout: int = 10,
        max_retries: int = 3,
    ):
        """
        初始化 Telegram 通知器 / Initialize Telegram Notifier

        Args:
            bot_token (str): Telegram Bot Token from @BotFather
            chat_id (str): Telegram Chat ID (channel, group, or user ID)
            timeout (int): 请求超时时间（秒）
            max_retries (int): 最大重试次数
        """
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout
        self._max_retries = max_retries
        self._logger = logging.getLogger("TelegramNotifier")

    def _build_url(self, method: str) -> str:
        """Build Telegram API URL for a given method."""
        return self.BASE_URL.format(token=self._token, method=method)

    def _send_request(
        self,
        method: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send HTTP request to Telegram API using urllib.

        Args:
            method: API method name (e.g., "sendMessage")
            data: POST data dictionary

        Returns:
            JSON response dict or None on failure
        """
        url = self._build_url(method)

        for attempt in range(self._max_retries):
            try:
                json_data = json.dumps(data or {}).encode("utf-8") if data else None

                req = urllib.request.Request(
                    url,
                    data=json_data,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "OctoBot-Notifier/1.0",
                    },
                )

                with urllib.request.urlopen(req, timeout=self._timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))

                    if result.get("ok"):
                        return result.get("result")
                    else:
                        self._logger.error(f"Telegram API error: {result.get('description')}")
                        return None

            except urllib.error.HTTPError as e:
                self._logger.warning(f"HTTP error {e.code} on attempt {attempt + 1}: {e.reason}")
                if attempt == self._max_retries - 1:
                    return None
            except urllib.error.URLError as e:
                self._logger.warning(f"URL error on attempt {attempt + 1}: {e.reason}")
                if attempt == self._max_retries - 1:
                    return None
            except Exception as e:
                self._logger.error(f"Unexpected error sending to Telegram: {e}")
                return None

            # Exponential backoff
            time.sleep(0.5 * (2 ** attempt))

        return None

    async def send_message(
        self,
        text: str,
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
        reply_markup: Optional[Dict] = None,
    ) -> bool:
        """
        发送文本消息 / Send Text Message

        Args:
            text (str): 消息文本（支持 Markdown 格式）
            parse_mode (str): 解析模式 ("Markdown" or "HTML")
            disable_notification (bool): 是否禁用通知
            reply_markup (Dict):  Reply keyboard markup (optional)

        Returns:
            bool: 是否发送成功

        Example:
            >>> await notifier.send_message("*BTC/USDT* 买入信号已触发")
            >>> await notifier.send_message("<b>紧急</b>止损单被触发!", parse_mode="HTML")
        """
        data = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }

        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)

        result = self._send_request("sendMessage", data)
        return result is not None

    async def send_alert(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        发送告警消息 / Send Alert Message

        根据告警级别格式化消息并发送。

        Args:
            message (str): 告警消息文本
            level (AlertLevel): 告警级别
            symbol (str): 相关交易对符号（可选）
            details (Dict): 附加详情（可选）

        Returns:
            bool: 是否发送成功

        Example:
            >>> await notifier.send_alert("BTC/USDT 突破关键阻力位", AlertLevel.WARNING)
            >>> await notifier.send_alert("多交易所连接失败", AlertLevel.ERROR)
        """
        # Build formatted message based on level
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }

        emoji = level_emoji.get(level, "ℹ️")
        level_text = level.name.upper()

        header = f"{emoji} *[{level_text}]*"
        if symbol:
            header += f" `{symbol}`"

        full_message = f"{header}\n\n{message}"

        # Add details if provided
        if details:
            details_text = "\n".join(f"  • `{k}`: {v}" for k, v in details.items())
            full_message += f"\n\n📊 Details:\n{details_text}"

        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message += f"\n\n⏰ {timestamp}"

        return await self.send_message(full_message)

    async def send_trade_report(
        self,
        orders: List[TradeOrder],
        report: ExecutionReport,
    ) -> bool:
        """
        发送交易执行报告 / Send Trade Execution Report

        格式化并发送交易执行结果的详细报告。

        Args:
            orders (List[TradeOrder]): 执行的订单列表
            report (ExecutionReport): 执行报告

        Returns:
            bool: 是否发送成功

        Example:
            >>> await notifier.send_trade_report(orders, report)
        """
        if not orders:
            return await self.send_alert("交易报告: 无订单执行", AlertLevel.INFO)

        symbols = ", ".join(set(o.symbol for o in orders))
        sides = ", ".join(set(o.side.upper() for o in orders))

        # Build report message
        header = f"📊 *Trade Execution Report*"
        body = f"""
✅ Orders: {len(orders)}
📈 Symbols: {symbols}
🔄 Sides: {sides}
💰 Total Qty: {report.total_quantity:.6f}
💵 Avg Price: ${report.avg_price:.4f}
💸 Total Cost: ${report.total_cost:.2f}
⏱ Exec Time: {report.execution_time_ms:.1f}ms
"""

        if report.errors:
            body += f"\n❌ Errors ({len(report.errors)}):\n"
            for err in report.errors[:3]:  # Show first 3 errors
                body += f"  • {err}\n"

        success_emoji = "✅" if report.success else "❌"
        status = f"{success_emoji} *{'SUCCESS' if report.success else 'FAILED'}*"

        full_message = f"{header}\n{status}\n{body}"

        return await self.send_message(full_message)

    async def send_signal(
        self,
        signal_type: TradeSignalType,
        symbol: str,
        price: float,
        quantity: float,
        strategy: str = "unknown",
        reason: Optional[str] = None,
    ) -> bool:
        """
        发送交易信号 / Send Trading Signal

        格式化并发送标准化的交易信号消息。

        Args:
            signal_type (TradeSignalType): 信号类型
            symbol (str): 交易对符号
            price (float): 参考价格
            quantity (float): 建议交易数量
            strategy (str): 策略名称
            reason (str): 信号原因（可选）

        Returns:
            bool: 是否发送成功

        Example:
            >>> await notifier.send_signal(
            ...     signal_type=TradeSignalType.ENTRY,
            ...     symbol="BTC/USDT",
            ...     price=65000.0,
            ...     quantity=0.01,
            ...     strategy="GridV1",
            ...     reason="RSI oversold bounce"
            ... )
        """
        signal_emoji = {
            TradeSignalType.ENTRY: "🟢",
            TradeSignalType.EXIT: "🔴",
            TradeSignalType.DCA: "💵",
            TradeSignalType.GRID: "📊",
            TradeSignalType.SHORT: "🔽",
            TradeSignalType.LONG: "🔼",
            TradeSignalType.CLOSE_SHORT: "🔼",
            TradeSignalType.CLOSE_LONG: "🔽",
            TradeSignalType.STOP_LOSS: "🛑",
            TradeSignalType.TAKE_PROFIT: "🎯",
        }

        emoji = signal_emoji.get(signal_type, "📌")
        signal_name = signal_type.name

        message = f"""
{emoji} *New Trading Signal*

📌 Type: `{signal_name}`
🪙 Symbol: `{symbol}`
💵 Price: `${price:.4f}`
📊 Quantity: `{quantity:.6f}`
🎯 Strategy: `{strategy}`
"""

        if reason:
            message += f"\n💡 Reason: {reason}"

        return await self.send_message(message)

    async def test_connection(self) -> bool:
        """
        测试 Telegram 连接 / Test Telegram Connection

        发送测试消息验证 Bot Token 和 Chat ID 是否有效。

        Returns:
            bool: 连接是否正常

        Example:
            >>> if await notifier.test_connection():
            ...     print("Telegram notification ready!")
        """
        result = self._send_request("getMe")

        if result:
            bot_name = result.get("username", "Unknown")
            self._logger.info(f"Telegram bot '{bot_name}' connected successfully")
            return True

        self._logger.error("Telegram connection test failed")
        return False

    async def get_chat_info(self) -> Optional[Dict[str, Any]]:
        """
        获取聊天信息 / Get Chat Information

        Returns:
            Chat info dict or None on failure
        """
        return self._send_request(f"getChat?chat_id={self._chat_id}")

    @property
    def is_configured(self) -> bool:
        """Check if Telegram notifier is properly configured."""
        return bool(self._token and self._chat_id)


# ====================
# Trading Aligner / 交易对齐器
# ====================

class TradingAligner:
    """
    跨交易所交易对齐器 / Cross-Exchange Trading Aligner

    将交易信号在多个交易所之间进行对齐和同步执行。

    Features:
    - 多交易所订单簿聚合
    - 最佳价格路由
    - 交易同步模式（顺序/并行/最佳价格）
    - 延迟和手续费考虑
    - 成交追踪与对齐报告

    OctoBot 的核心优势之一是能够同时在多个交易所执行交易，
    此适配器提供跨交易所的交易对齐功能。

    Example:
        >>> aligner = TradingAligner(config)
        >>> await aligner.connect_exchanges(["binance", "bybit", "okx"])
        >>> result = await aligner.align_and_execute(
        ...     symbol="BTC/USDT",
        ...     side="buy",
        ...     total_quantity=0.1,
        ...     sync_mode=ExchangeSyncMode.BEST_PRICE,
        ... )
    """

    def __init__(
        self,
        config: OctoBotConfig,
        notifier: Optional[TelegramNotifier] = None,
    ):
        """
        初始化交易对齐器 / Initialize Trading Aligner

        Args:
            config (OctoBotConfig): OctoBot 配置
            notifier (TelegramNotifier): 可选的通知器
        """
        self._config = config
        self._notifier = notifier
        self._clients: Dict[str, Any] = {}
        self._order_books: Dict[str, Dict[str, List]] = {}
        self._fees: Dict[str, float] = {
            "binance": 0.001,
            "bybit": 0.001,
            "okx": 0.0015,
            "gate_io": 0.002,
            "kucoin": 0.001,
            "default": 0.002,
        }
        self._logger = logging.getLogger("TradingAligner")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format (BTC/USDT -> BTCUSDT)."""
        return symbol.replace("/", "").upper()

    def _denormalize_symbol(self, symbol: str) -> str:
        """Denormalize symbol format (BTCUSDT -> BTC/USDT)."""
        if len(symbol) > 6:
            # Assume quote currency is last 4 or 3 chars (USDT, BTC, ETH)
            for i in range(3, len(symbol) - 3):
                potential_quote = symbol[i:]
                if potential_quote in ("USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"):
                    return f"{symbol[:i]}/{potential_quote}"
        return symbol

    async def connect_exchange(
        self,
        exchange: str,
        **kwargs,
    ) -> bool:
        """
        连接交易所 / Connect to Exchange

        Args:
            exchange (str): 交易所名称
            **kwargs: 额外参数

        Returns:
            bool: 是否连接成功
        """
        if exchange in self._clients:
            return True

        creds = self._config.get_exchange_cred(exchange)

        if exchange == "binance" and _HAS_BINANCE:
            client = BinanceRESTClient(
                api_key=creds.get("api_key", ""),
                api_secret=creds.get("api_secret", ""),
                timeout=self._config.timeout,
                **kwargs
            )
            self._clients[exchange] = client
            self._logger.info(f"Connected to {exchange}")
            return True

        # For other exchanges, create placeholder
        self._clients[exchange] = {"exchange": exchange, "connected": True}
        self._logger.info(f"Connected (placeholder) to {exchange}")
        return True

    async def connect_all_enabled(self) -> Dict[str, bool]:
        """
        连接配置中启用的所有交易所 / Connect All Enabled Exchanges

        Returns:
            Dict mapping exchange name to connection success status
        """
        results = {}

        for exchange in self._config.exchanges:
            if self._config.is_exchange_enabled(exchange):
                results[exchange] = await self.connect_exchange(exchange)

        return results

    async def get_best_price(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> Tuple[Optional[str], float, float]:
        """
        获取最佳价格和交易所 / Get Best Price and Exchange

        在所有连接的交易所中查找最佳价格。

        Args:
            symbol (str): 交易对符号
            side (str): 买卖方向 ("buy" or "sell")
            quantity (float): 交易数量

        Returns:
            Tuple of (exchange_name, best_price, effective_price)
            effective_price includes fees and slippage
        """
        normalized = self._normalize_symbol(symbol)
        results = {}

        async def fetch_price(exchange: str, client: Any) -> Tuple[str, Optional[float], float]:
            try:
                if exchange == "binance" and _HAS_BINANCE:
                    ob = client.get_order_book(normalized, limit=50)
                    if side.lower() == "buy":
                        book_side = ob.get("asks", [])
                    else:
                        book_side = ob.get("bids", [])

                    if not book_side:
                        return exchange, None, 0.0

                    # Find price for the requested quantity
                    remaining = quantity
                    total_cost = 0.0

                    for level in book_side[:10]:  # Top 10 levels
                        price, qty = float(level[0]), float(level[1])
                        fill_qty = min(remaining, qty)
                        total_cost += fill_qty * price
                        remaining -= fill_qty

                        if remaining <= 0:
                            break

                    if remaining > 0:
                        return exchange, None, 0.0  # Insufficient liquidity

                    avg_price = total_cost / quantity
                    fee = self._fees.get(exchange, self._fees["default"])
                    # Effective price including fees
                    if side.lower() == "buy":
                        effective = avg_price * (1 + fee)
                    else:
                        effective = avg_price * (1 - fee)

                    return exchange, avg_price, effective

            except Exception as e:
                self._logger.warning(f"Failed to get price from {exchange}: {e}")

            return exchange, None, 0.0

        # Fetch from all exchanges concurrently
        tasks = [fetch_price(ex, cl) for ex, cl in self._clients.items()]
        results_list = await asyncio.gather(*tasks)

        best_exchange = None
        best_price = None
        best_effective = None

        for ex, price, effective in results_list:
            if price is not None:
                results[ex] = {"price": price, "effective": effective}
                if side.lower() == "buy":
                    # For buys, lower effective price is better
                    if best_effective is None or effective < best_effective:
                        best_exchange = ex
                        best_price = price
                        best_effective = effective
                else:
                    # For sells, higher effective price is better
                    if best_effective is None or effective > best_effective:
                        best_exchange = ex
                        best_price = price
                        best_effective = effective

        return best_exchange, best_price or 0.0, best_effective or 0.0

    async def execute_sequential(
        self,
        orders: List[TradeOrder],
    ) -> ExecutionReport:
        """
        顺序执行订单 / Execute Orders Sequentially

        在多个交易所按顺序执行订单。

        Args:
            orders (List[TradeOrder]): 订单列表

        Returns:
            ExecutionReport: 执行报告
        """
        start_time = time.perf_counter()
        results: Dict[str, Any] = {}
        errors: List[str] = []
        all_orders = []

        for order in orders:
            exchange = order.exchange or list(self._clients.keys())[0]

            if exchange not in self._clients:
                errors.append(f"Exchange {exchange} not connected")
                continue

            client = self._clients[exchange]
            normalized = self._normalize_symbol(order.symbol)

            try:
                if self._config.paper_trade:
                    # Simulate execution
                    import random
                    fill_price = order.price or random.uniform(100, 70000)
                    slippage = fill_price * 0.0005
                    if order.side.lower() == "buy":
                        fill_price *= (1 + slippage)
                    else:
                        fill_price *= (1 - slippage)

                    order.status = "filled"
                    order.filled_qty = order.quantity
                    order.avg_fill_price = fill_price
                    order.order_id = f"PAPER_{exchange}_{int(time.time() * 1000)}"

                    results[exchange] = {
                        "success": True,
                        "order_id": order.order_id,
                        "filled_qty": order.filled_qty,
                        "avg_fill_price": fill_price,
                    }

                    self._logger.info(
                        f"[PAPER] {order.side.upper()} {order.quantity} {order.symbol} @ {fill_price} on {exchange}"
                    )
                else:
                    # Real execution (using backend connector)
                    if exchange == "binance" and _HAS_BINANCE:
                        resp = client.place_order(
                            symbol=normalized,
                            side=order.side,
                            order_type=order.order_type,
                            quantity=order.quantity,
                            price=order.price,
                        )
                        if resp and resp.get("orderId"):
                            order.order_id = str(resp["orderId"])
                            order.status = "filled"
                            order.filled_qty = float(resp.get("executedQty", order.quantity))
                            order.avg_fill_price = float(resp.get("avgPrice", order.price or 0))

                            results[exchange] = {
                                "success": True,
                                "order_id": order.order_id,
                                "response": resp,
                            }
                        else:
                            order.status = "rejected"
                            errors.append(f"Order rejected by {exchange}")
                            results[exchange] = {"success": False}
                    else:
                        order.status = "rejected"
                        errors.append(f"Exchange {exchange} not supported for live trading")
                        results[exchange] = {"success": False}

            except Exception as e:
                order.status = "error"
                errors.append(f"{exchange}: {str(e)}")
                results[exchange] = {"success": False, "error": str(e)}
                self._logger.error(f"Order execution error on {exchange}: {e}")

            all_orders.append(order)

        # Calculate totals
        total_qty = sum(o.filled_qty for o in all_orders if o.status == "filled")
        total_cost = sum(o.filled_qty * o.avg_fill_price for o in all_orders if o.status == "filled")
        avg_price = total_cost / total_qty if total_qty > 0 else 0

        exec_time = (time.perf_counter() - start_time) * 1000

        return ExecutionReport(
            success=len(errors) == 0,
            orders=all_orders,
            total_quantity=total_qty,
            avg_price=avg_price,
            total_cost=total_cost,
            execution_time_ms=exec_time,
            errors=errors,
            exchange_results=results,
        )

    async def execute_parallel(
        self,
        orders: List[TradeOrder],
    ) -> ExecutionReport:
        """
        并行执行订单 / Execute Orders in Parallel

        在多个交易所同时执行订单。

        Args:
            orders (List[TradeOrder]): 订单列表

        Returns:
            ExecutionReport: 执行报告
        """
        start_time = time.perf_counter()
        results: Dict[str, Any] = {}
        errors: List[str] = []
        all_orders = []

        async def execute_single(order: TradeOrder) -> TradeOrder:
            exchange = order.exchange or list(self._clients.keys())[0]

            if exchange not in self._clients:
                order.status = "error"
                order.metadata["error"] = f"Exchange {exchange} not connected"
                return order

            client = self._clients[exchange]
            normalized = self._normalize_symbol(order.symbol)

            try:
                if self._config.paper_trade:
                    import random
                    fill_price = order.price or random.uniform(100, 70000)
                    slippage = fill_price * 0.0005
                    if order.side.lower() == "buy":
                        fill_price *= (1 + slippage)
                    else:
                        fill_price *= (1 - slippage)

                    order.status = "filled"
                    order.filled_qty = order.quantity
                    order.avg_fill_price = fill_price
                    order.order_id = f"PAPER_{exchange}_{int(time.time() * 1000)}"

                    results[exchange] = {"success": True, "order_id": order.order_id}
                else:
                    if exchange == "binance" and _HAS_BINANCE:
                        resp = client.place_order(
                            symbol=normalized,
                            side=order.side,
                            order_type=order.order_type,
                            quantity=order.quantity,
                            price=order.price,
                        )
                        if resp and resp.get("orderId"):
                            order.order_id = str(resp["orderId"])
                            order.status = "filled"
                            order.filled_qty = float(resp.get("executedQty", order.quantity))
                            order.avg_fill_price = float(resp.get("avgPrice", order.price or 0))
                            results[exchange] = {"success": True, "response": resp}
                        else:
                            order.status = "rejected"
                            errors.append(f"Order rejected by {exchange}")
                            results[exchange] = {"success": False}
                    else:
                        order.status = "rejected"
                        errors.append(f"Exchange {exchange} not supported")
                        results[exchange] = {"success": False}

            except Exception as e:
                order.status = "error"
                errors.append(f"{exchange}: {str(e)}")
                results[exchange] = {"success": False, "error": str(e)}

            return order

        # Execute all orders concurrently
        all_orders = await asyncio.gather(*[execute_single(o) for o in orders])

        # Calculate totals
        total_qty = sum(o.filled_qty for o in all_orders if o.status == "filled")
        total_cost = sum(o.filled_qty * o.avg_fill_price for o in all_orders if o.status == "filled")
        avg_price = total_cost / total_qty if total_qty > 0 else 0

        exec_time = (time.perf_counter() - start_time) * 1000

        return ExecutionReport(
            success=len(errors) == 0,
            orders=list(all_orders),
            total_quantity=total_qty,
            avg_price=avg_price,
            total_cost=total_cost,
            execution_time_ms=exec_time,
            errors=errors,
            exchange_results=results,
        )

    async def align_and_execute(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        sync_mode: ExchangeSyncMode = ExchangeSyncMode.SEQUENTIAL,
        split_exchanges: Optional[List[str]] = None,
    ) -> Tuple[ExecutionReport, Optional[str]]:
        """
        对齐并执行交易 / Align and Execute Trade

        在多交易所间对齐并执行交易。

        Args:
            symbol (str): 交易对符号
            side (str): 买卖方向
            total_quantity (float): 总交易数量
            sync_mode (ExchangeSyncMode): 同步模式
            split_exchanges (List[str]): 指定要使用的交易所（None 表示全部）

        Returns:
            Tuple of (ExecutionReport, best_exchange)
        """
        exchanges_to_use = split_exchanges or list(self._clients.keys())

        if len(exchanges_to_use) <= 1:
            sync_mode = ExchangeSyncMode.SEQUENTIAL

        # Get best price if using best_price mode
        best_exchange = None
        if sync_mode == ExchangeSyncMode.BEST_PRICE:
            best_exchange, _, _ = await self.get_best_price(symbol, side, total_quantity)
            if best_exchange:
                exchanges_to_use = [best_exchange]

        # Create orders for each exchange
        orders = []
        qty_per_exchange = total_quantity / len(exchanges_to_use)

        for ex in exchanges_to_use:
            order = TradeOrder(
                symbol=symbol,
                side=side,
                quantity=qty_per_exchange,
                exchange=ex,
                signal_type=TradeSignalType.ENTRY if side.lower() == "buy" else TradeSignalType.EXIT,
            )
            orders.append(order)

        # Execute based on sync mode
        if sync_mode == ExchangeSyncMode.PARALLEL:
            report = await self.execute_parallel(orders)
        else:
            report = await self.execute_sequential(orders)

        return report, best_exchange

    async def close(self):
        """Close all exchange connections."""
        for client in self._clients.values():
            if hasattr(client, "close"):
                client.close()
        self._clients.clear()

    @property
    def connected_exchanges(self) -> List[str]:
        """Get list of connected exchanges."""
        return list(self._clients.keys())


# ====================
# OctoBot Executor / OctoBot 执行引擎
# ====================

class OctoBotExecutor:
    """
    OctoBot 执行引擎 / OctoBot Execution Engine

    整合配置管理、Telegram 通知和交易执行的主要执行引擎。

    Features:
    - 多交易所统一接口
    - 自动化交易执行
    - Telegram 实时通知
    - 交易信号处理
    - 执行报告生成
    - 策略参数管理

    这是 OctoBot 适配器的核心类，协调所有子组件完成交易执行流程。

    Usage:
        >>> executor = OctoBotExecutor(config, notifier=notifier)
        >>> await executor.start()
        >>> await executor.execute_trades([...])
        >>> await executor.stop()

    参考 OctoBot 主类 (octobot/octobot.py) 的核心架构：
    - 初始化 -> 连接交易所 -> 启动通知 -> 执行交易 -> 停止
    """

    def __init__(
        self,
        config: Optional[OctoBotConfig] = None,
        notifier: Optional[TelegramNotifier] = None,
        aligner: Optional[TradingAligner] = None,
    ):
        """
        初始化 OctoBot 执行引擎 / Initialize OctoBot Executor

        Args:
            config (OctoBotConfig): OctoBot 配置（默认新建空配置）
            notifier (TelegramNotifier): Telegram 通知器（默认从配置创建）
            aligner (TradingAligner): 交易对齐器（默认新建）
        """
        self._config = config or OctoBotConfig()
        self._notifier = notifier
        self._aligner = aligner

        # If notifier not provided, create from config
        if self._notifier is None and self._config.telegram_token:
            self._notifier = TelegramNotifier(
                bot_token=self._config.telegram_token,
                chat_id=self._config.telegram_chat_id,
            )

        # If aligner not provided, create new one
        if self._aligner is None:
            self._aligner = TradingAligner(self._config, self._notifier)

        self._is_running = False
        self._trade_history: List[ExecutionReport] = []
        self._logger = logging.getLogger("OctoBotExecutor")

    async def start(self) -> bool:
        """
        启动执行引擎 / Start the Executor

        连接所有配置的交易所，初始化 Telegram 通知。

        Returns:
            bool: 启动是否成功

        Example:
            >>> executor = OctoBotExecutor(config)
            >>> if await executor.start():
            ...     print("Executor started successfully")
        """
        if self._is_running:
            self._logger.warning("Executor already running")
            return True

        try:
            # Connect to exchanges
            connection_results = await self._aligner.connect_all_enabled()

            connected_count = sum(1 for v in connection_results.values() if v)
            self._logger.info(
                f"Connected to {connected_count}/{len(connection_results)} exchanges"
            )

            if connected_count == 0:
                self._logger.error("Failed to connect to any exchange")
                return False

            # Test Telegram if configured
            if self._notifier and self._notifier.is_configured:
                telegram_ok = await self._notifier.test_connection()
                if telegram_ok:
                    await self._notifier.send_alert(
                        "OctoBot Executor 已启动",
                        level=AlertLevel.INFO,
                    )
                else:
                    self._logger.warning("Telegram notification test failed")

            self._is_running = True
            self._logger.info("OctoBot Executor started successfully")
            return True

        except Exception as e:
            self._logger.error(f"Failed to start executor: {e}")
            return False

    async def stop(self):
        """
        停止执行引擎 / Stop the Executor

        关闭所有连接，保存状态。
        """
        if not self._is_running:
            return

        try:
            # Close exchange connections
            await self._aligner.close()

            # Send shutdown notification
            if self._notifier and self._notifier.is_configured:
                await self._notifier.send_alert(
                    "OctoBot Executor 已停止",
                    level=AlertLevel.INFO,
                )

            self._is_running = False
            self._logger.info("OctoBot Executor stopped")

        except Exception as e:
            self._logger.error(f"Error stopping executor: {e}")

    @property
    def is_running(self) -> bool:
        """Check if executor is running."""
        return self._is_running

    @property
    def config(self) -> OctoBotConfig:
        """Get the OctoBot configuration."""
        return self._config

    @property
    def notifier(self) -> Optional[TelegramNotifier]:
        """Get the Telegram notifier."""
        return self._notifier

    @property
    def aligner(self) -> TradingAligner:
        """Get the trading aligner."""
        return self._aligner

    def get_trade_history(self, limit: int = 100) -> List[ExecutionReport]:
        """Get recent trade history."""
        return self._trade_history[-limit:]

    async def execute_trades(
        self,
        trades: List[Dict[str, Any]],
        sync_mode: Optional[ExchangeSyncMode] = None,
    ) -> ExecutionReport:
        """
        执行交易列表 / Execute Trade List

        执行一组交易订单，支持多种同步模式。

        Args:
            trades (List[Dict[str, Any]]): 交易列表
                每项包含: symbol, side, quantity, price (optional), order_type (optional)
            sync_mode (ExchangeSyncMode): 同步模式（默认使用配置的模式）

        Returns:
            ExecutionReport: 执行报告

        Example:
            >>> trades = [
            ...     {"symbol": "BTC/USDT", "side": "buy", "quantity": 0.01, "price": 65000},
            ...     {"symbol": "ETH/USDT", "side": "buy", "quantity": 0.1, "price": 3500},
            ... ]
            >>> report = await executor.execute_trades(trades)
            >>> print(f"Executed {len(report.orders)} orders in {report.execution_time_ms:.1f}ms")
        """
        if not self._is_running:
            self._logger.error("Executor not started. Call start() first.")
            return ExecutionReport(
                success=False,
                orders=[],
                errors=["Executor not started"],
            )

        if not trades:
            return ExecutionReport(
                success=False,
                orders=[],
                errors=["No trades provided"],
            )

        sync_mode = sync_mode or self._config.sync_mode

        # Convert dicts to TradeOrder objects
        orders = []
        for t in trades:
            order = TradeOrder(
                symbol=t.get("symbol", "UNKNOWN"),
                side=t.get("side", "buy"),
                quantity=t.get("quantity", 0.0),
                price=t.get("price"),
                order_type=t.get("order_type", "market"),
                exchange=t.get("exchange"),
                signal_type=TradeSignalType(t.get("signal_type", "entry")),
                metadata=t.get("metadata", {}),
            )
            orders.append(order)

        # Send pre-execution alert
        if self._notifier and self._notifier.is_configured:
            symbols = ", ".join(set(o.symbol for o in orders))
            await self._notifier.send_alert(
                f"执行 {len(orders)} 个订单: {symbols}",
                level=AlertLevel.INFO,
            )

        # Execute based on sync mode
        if sync_mode == ExchangeSyncMode.PARALLEL:
            report = await self._aligner.execute_parallel(orders)
        elif sync_mode == ExchangeSyncMode.BEST_PRICE:
            # Group by symbol and find best prices
            report = await self._execute_best_price(orders)
        else:
            report = await self._aligner.execute_sequential(orders)

        # Store in history
        self._trade_history.append(report)

        # Send execution report
        if self._notifier and self._notifier.is_configured:
            await self._notifier.send_trade_report(orders, report)

        return report

    async def _execute_best_price(
        self,
        orders: List[TradeOrder],
    ) -> ExecutionReport:
        """
        最佳价格执行 / Best Price Execution

        按最佳价格模式执行订单，自动选择最优交易所。
        """
        # Group orders by symbol
        symbol_groups: Dict[str, List[TradeOrder]] = {}
        for order in orders:
            if order.symbol not in symbol_groups:
                symbol_groups[order.symbol] = []
            symbol_groups[order.symbol].append(order)

        all_reports: List[ExecutionReport] = []
        best_exchanges: Dict[str, str] = {}

        for symbol, symbol_orders in symbol_groups.items():
            total_qty = sum(o.quantity for o in symbol_orders)
            side = symbol_orders[0].side if symbol_orders else "buy"

            # Find best exchange for this symbol
            best_ex, _, _ = await self._aligner.get_best_price(symbol, side, total_qty)

            if best_ex:
                best_exchanges[symbol] = best_ex
                # Route all orders for this symbol to best exchange
                for order in symbol_orders:
                    order.exchange = best_ex

        # Execute all orders sequentially (they're now routed to best exchanges)
        report = await self._aligner.execute_sequential(orders)
        return report

    async def send_alert(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        symbol: Optional[str] = None,
    ) -> bool:
        """
        发送告警 / Send Alert

        通过 Telegram 发送告警消息（如果已配置）。

        Args:
            message (str): 告警消息
            level (AlertLevel): 告警级别
            symbol (str): 相关交易对

        Returns:
            bool: 是否发送成功

        Example:
            >>> await executor.send_alert("价格突破新高", AlertLevel.WARNING, "BTC/USDT")
        """
        if self._notifier and self._notifier.is_configured:
            return await self._notifier.send_alert(message, level, symbol)
        return False

    async def send_signal(
        self,
        signal_type: TradeSignalType,
        symbol: str,
        price: float,
        quantity: float,
        strategy: str = "unknown",
        reason: Optional[str] = None,
    ) -> bool:
        """
        发送交易信号 / Send Trading Signal

        通过 Telegram 发送标准化的交易信号（如果已配置）。

        Args:
            signal_type (TradeSignalType): 信号类型
            symbol (str): 交易对
            price (float): 参考价格
            quantity (float): 建议数量
            strategy (str): 策略名称
            reason (str): 信号原因

        Returns:
            bool: 是否发送成功
        """
        if self._notifier and self._notifier.is_configured:
            return await self._notifier.send_signal(
                signal_type, symbol, price, quantity, strategy, reason
            )
        return False

    async def execute_single_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        exchange: Optional[str] = None,
    ) -> ExecutionReport:
        """
        执行单个交易 / Execute Single Trade

        便捷方法：执行单个交易订单。

        Args:
            symbol (str): 交易对符号
            side (str): 买卖方向
            quantity (float): 交易数量
            price (Optional[float]): 限价价格
            exchange (Optional[str]): 交易所名称

        Returns:
            ExecutionReport: 执行报告

        Example:
            >>> report = await executor.execute_single_trade(
            ...     symbol="BTC/USDT",
            ...     side="buy",
            ...     quantity=0.01,
            ...     price=65000.0,
            ... )
        """
        return await self.execute_trades([{
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "exchange": exchange,
        }])

    # ---- Context Manager / 上下文管理器 ----

    async def __aenter__(self) -> "OctoBotExecutor":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# ====================
# Exports / 导出
# ====================

__all__ = [
    # Config
    "OctoBotConfig",
    # Data classes
    "TradeOrder",
    "ExecutionReport",
    # Enums
    "TradeSignalType",
    "AlertLevel",
    "ExchangeSyncMode",
    # Main classes
    "TelegramNotifier",
    "TradingAligner",
    "OctoBotExecutor",
]
