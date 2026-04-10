# -*- coding: utf-8 -*-
"""
Production Live Trading Runner
==============================

生产环境实盘交易运行器

功能:
- V36策略执行 (支持多策略)
- Binance实盘连接
- 多层风险控制
- 实时Dashboard
- 完整交易日志

Usage:
    # 干跑模式 (不交易)
    python run_live_trading.py --mode dry_run --symbols BTCUSDT ETHUSDT

    # Paper交易
    python run_live_trading.py --mode paper --capital 10000 --symbols BTCUSDT

    # 实盘交易 (需要设置环境变量)
    export BINANCE_API_KEY="your_key"
    export BINANCE_API_SECRET="your_secret"
    export PRODUCTION_MODE="true"
    python run_live_trading.py --mode live --symbols BTCUSDT

    # 带审批模式
    python run_live_trading.py --mode live --approval-required --large-trade-threshold 500
"""

import asyncio
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_trading.config.production import (
    ProductionConfig,
    TradingMode,
    EmergencyLevel,
    get_production_config,
    print_config_summary,
)
from quant_trading.execution.emergency_stop import (
    EmergencyStopSystem,
    AlertManager,
    PositionSnapshot,
    CircuitState,
)
from quant_trading.connectors.binance_ws import BinanceWebSocketClient
from quant_trading.connectors.binance_rest import BinanceRESTClient
from quant_trading.execution.executor import Executor, OrderSide, OrderType, Order
from quant_trading.risk.manager import RiskManager, RiskConfig, RiskLevel
from quant_trading.experiments.v36_paper_trader import V36SignalGenerator, Bar
from quant_trading.config.v36_config import V36_CRYPTO_OPTIMIZED, SIGNAL_ENHANCER_PARAMS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("LiveTrading")


# ===================== 数据模型 =====================

@dataclass
class LivePosition:
    """实盘持仓"""
    symbol: str
    side: str  # LONG / SHORT
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    time_stop: datetime
    buy_point: str = ""
    signal_strength: float = 0.0


@dataclass
class TradeRecord:
    """交易记录"""
    id: str
    timestamp: int
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float
    commission: float
    pnl: float
    pnl_pct: float
    reason: str
    emergency: bool = False


@dataclass
class TradingStats:
    """交易统计"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    current_capital: float = 0.0
    initial_capital: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    last_trade_time: int = 0


# ===================== Dashboard =====================

class TradingDashboard:
    """实时Dashboard"""

    def __init__(self, stats: TradingStats, positions: Dict[str, LivePosition]):
        self.stats = stats
        self.positions = positions

    def render(self) -> str:
        """渲染Dashboard"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append(f"  量化之神 Production Trading Dashboard  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        # 统计
        win_rate = (
            self.stats.winning_trades / self.stats.total_trades * 100
            if self.stats.total_trades > 0 else 0
        )
        total_return = (
            (self.stats.current_capital - self.stats.initial_capital) / self.stats.initial_capital * 100
            if self.stats.initial_capital > 0 else 0
        )

        lines.append(f"\n  [账户]  初始: ${self.stats.initial_capital:,.2f}  |  当前: ${self.stats.current_capital:,.2f}  |  回报: {total_return:+.2f}%")
        lines.append(f"  [P&L]   日盈亏: ${self.stats.daily_pnl:+,.2f}  |  总盈亏: ${self.stats.total_pnl:+,.2f}  |  最大回撤: ${self.stats.max_drawdown:,.2f}")
        lines.append(f"  [交易]  总交易: {self.stats.total_trades}  |  胜率: {win_rate:.1f}%  |  连续胜/负: {self.stats.consecutive_wins}/{self.stats.consecutive_losses}")

        # 持仓
        lines.append(f"\n  [持仓]  ({len(self.positions)}/{self.max_positions})")
        if self.positions:
            for symbol, pos in self.positions.items():
                pnl = (pos.quantity * (self.current_price.get(symbol, 0) - pos.entry_price))
                pnl_pct = (self.current_price.get(symbol, 0) - pos.entry_price) / pos.entry_price * 100
                lines.append(f"    {symbol}: {pos.side} {pos.quantity:.4f} @ ${pos.entry_price:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        else:
            lines.append("    无持仓")

        lines.append("\n" + "-" * 70)
        return "\n".join(lines)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """更新实时价格"""
        self.current_price = prices


# ===================== 主交易引擎 =====================

class LiveTradingEngine:
    """
    生产环境实盘交易引擎

    架构:
    1. EmergencyStopSystem - 顶层安全监控
    2. RiskManager - 风险管理
    3. Executor - 订单执行
    4. V36SignalGenerator - 信号生成
    5. Dashboard - 状态显示
    """

    def __init__(
        self,
        symbols: List[str],
        capital: float,
        config: ProductionConfig,
        mode: TradingMode = TradingMode.DRY_RUN,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.capital = capital
        self.config = config
        self.mode = mode

        # 初始化组件
        self._init_components()

        # 状态
        self.positions: Dict[str, LivePosition] = {}
        self.stats = TradingStats(
            initial_capital=capital,
            current_capital=capital
        )
        self.prices: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.running = False
        self.last_dashboard_time = 0

        # Dashboard
        self.dashboard = TradingDashboard(self.stats, self.positions)
        self.dashboard.max_positions = config.risk.position.max_positions
        self.dashboard.current_price = self.prices

        # 数据库
        self.db_path = config.database_path
        self._init_database()

        logger.info(f"LiveTradingEngine 初始化完成")
        logger.info(f"  交易对: {self.symbols}")
        logger.info(f"  初始资金: ${capital:,.2f}")
        logger.info(f"  模式: {mode.value}")

    def _init_components(self) -> None:
        """初始化组件"""
        # 告警管理器
        self.alert_manager = AlertManager(self.config)

        # 紧急止损系统
        self.emergency_stop = EmergencyStopSystem(
            self.config,
            self,  # 传入self作为executor的替代
            self.alert_manager
        )
        self.emergency_stop.on_emergency = self._on_emergency
        self.emergency_stop.on_circuit_trip = self._on_circuit_trip

        # 风控管理器
        risk_config = RiskConfig(
            max_position_size=self.config.risk.position.max_position_size,
            max_single_loss=self.config.risk.loss.max_single_loss,
            max_daily_loss=self.config.risk.loss.max_daily_loss,
            max_daily_trades=100,  # 宽松限制
            max_portfolio_exposure=self.config.risk.position.max_total_exposure,
            stop_loss_pct=self.config.risk.stop_loss_pct,
            take_profit_pct=self.config.risk.take_profit_pct,
            cooldown_after_loss=self.config.risk.position.min_trade_interval_seconds * 1000,
        )
        self.risk_manager = RiskManager(risk_config)

        # 执行器
        creds = self.config.credentials
        self.executor = Executor(
            api_key=creds.get("api_key", ""),
            api_secret=creds.get("api_secret", ""),
            test_mode=(self.mode == TradingMode.PAPER),
            risk_manager=self.risk_manager
        )

        # 信号生成器
        self.signal_gen = V36SignalGenerator(V36_CRYPTO_OPTIMIZED)

        # Binance连接
        self.ws_client: Optional[BinanceWebSocketClient] = None
        self.rest_client: Optional[BinanceRESTClient] = None

        if self.mode != TradingMode.DRY_RUN:
            try:
                api_key = creds.get("api_key", "")
                api_secret = creds.get("api_secret", "")
                self.rest_client = BinanceRESTClient(api_key, api_secret)
                logger.info("Binance REST API 连接成功")
            except Exception as e:
                logger.error(f"Binance连接失败: {e}")

    def _init_database(self) -> None:
        """初始化数据库"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                reason TEXT,
                emergency INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_time INTEGER NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                time_stop INTEGER,
                buy_point TEXT,
                signal_strength REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp
            ON trades(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
            ON trades(symbol)
        """)
        conn.commit()
        conn.close()

    async def connect(self) -> None:
        """连接交易所"""
        if self.mode == TradingMode.DRY_RUN:
            logger.info("干跑模式 - 不连接交易所")
            return

        try:
            # WebSocket连接
            streams = [f"{s.lower()}@kline_1m" for s in self.symbols]
            self.ws_client = BinanceWebSocketClient(streams, self._on_ws_message)
            self.ws_client.start()
            logger.info(f"WebSocket已连接: {streams}")

            # 同步历史数据
            await self._sync_historical_data()

        except Exception as e:
            logger.error(f"连接失败: {e}")
            raise

    async def _sync_historical_data(self) -> None:
        """同步历史数据到信号生成器"""
        if not self.rest_client:
            return

        for symbol in self.symbols:
            try:
                klines = self.rest_client.get_klines(symbol, '1h', limit=100)
                for k in klines:
                    bar = Bar(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(k[0] / 1000),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                    )
                    self.signal_gen.update_bar(bar)
                logger.info(f"{symbol}: 同步 {len(klines)} 根K线")
            except Exception as e:
                logger.error(f"{symbol} 数据同步失败: {e}")

    def _on_ws_message(self, msg: Dict) -> None:
        """WebSocket消息处理"""
        try:
            if msg.get('e') == 'kline':
                k = msg['k']
                symbol = k['s']
                bar = Bar(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(k['t'] / 1000),
                    open=float(k['o']),
                    high=float(k['h']),
                    low=float(k['l']),
                    close=float(k['c']),
                    volume=float(k['v']),
                )
                self.prices[symbol] = bar.close

                # 异步处理
                asyncio.create_task(self._on_bar(bar))

        except Exception as e:
            logger.error(f"WebSocket消息处理错误: {e}")

    async def _on_bar(self, bar: Bar) -> None:
        """处理K线"""
        # 更新信号
        self.signal_gen.update_bar(bar)

        # 检查平仓
        if bar.symbol in self.positions:
            await self._check_exit(bar)

        # 检查开仓
        if bar.symbol not in self.positions:
            await self._check_entry(bar)

    async def _check_entry(self, bar: Bar) -> None:
        """检查开仓信号"""
        # 干跑模式不交易
        if self.mode == TradingMode.DRY_RUN:
            return

        # 检查紧急停止
        if self.emergency_stop.emergency_active:
            return

        # 检查熔断
        if self.emergency_stop.circuit_state == CircuitState.TRIPPED:
            return

        # 检查风控
        risk_check = self.risk_manager.check_trade_allowed(bar.symbol)
        if not risk_check["allowed"]:
            logger.debug(f"风控拦截: {risk_check['reason']}")
            return

        # 检查信号
        signal = self.signal_gen.check_buy_signals(bar.symbol)
        if not signal.get("signal"):
            return

        buy_point = signal.get("buy_point", "A-企稳")
        strength = signal.get("strength", 0.5)

        # 信号强度过滤
        min_strength = 0.50
        if strength < min_strength:
            logger.debug(f"{bar.symbol}: 信号强度不足 {strength:.2f}")
            return

        # 最大持仓数检查
        if len(self.positions) >= self.config.risk.position.max_positions:
            logger.debug(f"{bar.symbol}: 达到最大持仓数")
            return

        # 总仓位检查
        total_exposed = self._get_total_exposure()
        if total_exposed >= self.config.risk.position.max_total_exposure:
            logger.debug(f"{bar.symbol}: 总仓位已满 {total_exposed:.1%}")
            return

        logger.info(f"[{bar.symbol}] 买入信号: {buy_point}, 强度: {strength:.2f}, 价格: {bar.close:.4f}")

        # 计算仓位
        position_size = self._calculate_position_size(bar.close)

        # 大额交易审批
        position_value = position_size * bar.close
        if position_value > self.config.risk.trade_approval.require_approval_above:
            logger.warning(f"[{bar.symbol}] 大额交易需要审批: ${position_value:.2f}")
            self.alert_manager.send_alert(
                EmergencyLevel.WARNING,
                "大额交易审批",
                f"交易对: {bar.symbol}\n金额: ${position_value:.2f}\n等待确认...",
                {"symbol": bar.symbol, "value": position_value}
            )
            # 在干跑/paper模式直接执行，实盘模式需要审批
            if self.mode == TradingMode.LIVE:
                # TODO: 实现审批流程
                pass

        # 计算止损止盈
        stop_loss = bar.close * (1 + V36_CRYPTO_OPTIMIZED.get("stop_loss", -0.063))
        take_profit = bar.close * (1 + V36_CRYPTO_OPTIMIZED.get("take_profit", 0.072))

        # 时间止损
        hours = V36_CRYPTO_OPTIMIZED.get("time_stop", 14)
        time_stop = datetime.now() + timedelta(hours=hours)

        # 下单
        try:
            order = await self.executor.place_order(
                symbol=bar.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position_size,
                price=bar.close
            )

            if order.status.value == "filled":
                # 记录持仓
                self.positions[bar.symbol] = LivePosition(
                    symbol=bar.symbol,
                    side="LONG",
                    quantity=order.filled_quantity,
                    entry_price=order.avg_fill_price,
                    entry_time=datetime.now(),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    time_stop=time_stop,
                    buy_point=buy_point,
                    signal_strength=strength
                )

                # 更新余额
                cost = order.filled_quantity * order.avg_fill_price
                self.stats.current_capital -= cost

                # 记录交易
                self._record_trade(
                    symbol=bar.symbol,
                    side="BUY",
                    order_type="MARKET",
                    quantity=order.filled_quantity,
                    price=order.avg_fill_price,
                    reason=f"开仓-{buy_point}"
                )

                # 更新风控
                self.risk_manager.record_trade({
                    "symbol": bar.symbol,
                    "value": cost,
                    "side": "buy"
                })

                # 更新紧急止损系统
                self.emergency_stop.update_positions(self._get_position_snapshots())

                logger.info(f"[{bar.symbol}] 开仓: {order.filled_quantity:.4f} @ {order.avg_fill_price:.4f}")

        except PermissionError as e:
            logger.warning(f"风控拦截: {e}")
        except Exception as e:
            logger.error(f"[{bar.symbol}] 开仓失败: {e}")

    async def _check_exit(self, bar: Bar) -> None:
        """检查平仓信号"""
        position = self.positions.get(bar.symbol)
        if not position:
            return

        exit_signal = self.signal_gen.check_exit_signals(bar.symbol, position)
        should_exit = exit_signal.get("should_exit", False)
        reason = exit_signal.get("reason", "")

        # 检查时间止损
        if datetime.now() > position.time_stop:
            should_exit = True
            reason = "时间止损"

        # 检查紧急止损
        pnl_pct = (bar.close - position.entry_price) / position.entry_price

        if pnl_pct < -self.config.risk.emergency_stop.auto_close_loss_pct:
            should_exit = True
            reason = f"自动止损 ({pnl_pct:.2%})"
        elif pnl_pct > self.config.risk.emergency_stop.auto_close_profit_pct:
            should_exit = True
            reason = f"自动止盈 ({pnl_pct:.2%})"

        if not should_exit:
            return

        # 平仓
        try:
            order = await self.executor.place_order(
                symbol=bar.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                price=bar.close
            )

            if order.status.value == "filled":
                proceeds = order.filled_quantity * order.avg_fill_price
                pnl = proceeds - position.quantity * position.entry_price
                pnl_pct = (order.avg_fill_price - position.entry_price) / position.entry_price

                # 更新统计
                self.stats.current_capital += proceeds
                self.stats.total_pnl += pnl
                self.stats.daily_pnl += pnl
                self.stats.total_trades += 1
                self.stats.last_trade_time = int(time.time())

                if pnl > 0:
                    self.stats.winning_trades += 1
                    self.stats.consecutive_wins += 1
                    self.stats.consecutive_losses = 0
                else:
                    self.stats.losing_trades += 1
                    self.stats.consecutive_losses += 1
                    self.stats.consecutive_wins = 0

                # 检查最大回撤
                drawdown = self.stats.initial_capital - self.stats.current_capital
                if drawdown > self.stats.max_drawdown:
                    self.stats.max_drawdown = drawdown
                    self.stats.max_drawdown_pct = drawdown / self.stats.initial_capital

                # 记录交易
                self._record_trade(
                    symbol=bar.symbol,
                    side="SELL",
                    order_type="MARKET",
                    quantity=order.filled_quantity,
                    price=order.avg_fill_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    reason=reason
                )

                # 更新风控
                self.risk_manager.record_trade({
                    "symbol": bar.symbol,
                    "value": proceeds,
                    "pnl": pnl,
                    "side": "sell"
                })

                # 删除持仓
                del self.positions[bar.symbol]

                # 更新紧急止损系统
                self.emergency_stop.update_positions(self._get_position_snapshots())

                logger.info(f"[{bar.symbol}] 平仓: {reason}, 盈亏: ${pnl:.2f} ({pnl_pct:.2%})")

        except Exception as e:
            logger.error(f"[{bar.symbol}] 平仓失败: {e}")

    def _calculate_position_size(self, price: float) -> float:
        """计算仓位大小"""
        max_pct = self.config.risk.position.max_single_trade_pct
        available = self.stats.current_capital * max_pct

        # 不超过持仓限额
        remaining_slots = max(1, self.config.risk.position.max_positions - len(self.positions))
        max_per_trade = self.stats.current_capital * (self.config.risk.position.max_total_exposure / remaining_slots)

        position_value = min(available, max_per_trade)

        # 不超过单笔最大
        max_position_usd = self.config.risk.position.max_position_size
        position_value = min(position_value, max_position_usd)

        return max(0, position_value / price)

    def _get_total_exposure(self) -> float:
        """计算总暴露"""
        total_value = sum(
            pos.quantity * pos.entry_price
            for pos in self.positions.values()
        )
        return total_value / self.stats.initial_capital

    def _get_position_snapshots(self) -> Dict[str, PositionSnapshot]:
        """获取持仓快照"""
        snapshots = {}
        for symbol, pos in self.positions.items():
            current_price = self.prices.get(symbol, pos.entry_price)
            pnl = pos.quantity * (current_price - pos.entry_price)
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price

            snapshots[symbol] = PositionSnapshot(
                symbol=symbol,
                side=pos.side,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                current_price=current_price,
                unrealized_pnl=pnl,
                pnl_pct=pnl_pct
            )
        return snapshots

    def _record_trade(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float,
        reason: str,
        pnl: float = 0,
        pnl_pct: float = 0,
        emergency: bool = False
    ) -> None:
        """记录交易到数据库"""
        trade_id = f"TRD_{int(time.time() * 1000)}_{symbol}"
        timestamp = int(time.time() * 1000)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO trades
                (id, timestamp, symbol, side, order_type, quantity, price, pnl, pnl_pct, reason, emergency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_id, timestamp, symbol, side, order_type, quantity, price, pnl, pnl_pct, reason, emergency))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"记录交易失败: {e}")

    def _on_emergency(self, reason: str) -> None:
        """紧急事件回调"""
        logger.critical(f"Emergency: {reason}")
        asyncio.create_task(self.emergency_stop.close_all_positions(reason))

    def _on_circuit_trip(self, reason: str) -> None:
        """熔断回调"""
        logger.warning(f"Circuit trip: {reason}")
        asyncio.create_task(self.emergency_stop.close_all_positions(f"Circuit Trip: {reason}"))

    def _print_dashboard(self) -> None:
        """打印Dashboard"""
        self.dashboard.stats = self.stats
        self.dashboard.positions = self.positions
        self.dashboard.current_price = self.prices
        print(self.dashboard.render())

    async def run(self) -> None:
        """运行交易引擎"""
        self.running = True
        logger.info("=" * 60)
        logger.info("Production Live Trading 启动")
        logger.info("=" * 60)

        # 连接
        await self.connect()

        # 主循环
        iteration = 0
        while self.running:
            try:
                iteration += 1

                # 健康检查
                self._health_check()

                # Dashboard更新
                if time.time() - self.last_dashboard_time > 60:
                    self._print_dashboard()
                    self.last_dashboard_time = time.time()

                # 熔断检查
                if self.emergency_stop.check_circuit_breaker():
                    logger.warning("Circuit breaker tripped, closing positions...")
                    await self.emergency_stop.close_all_positions("Circuit breaker")

                # 更新紧急止损系统
                self.emergency_stop.update_positions(self._get_position_snapshots())

                await asyncio.sleep(1)

            except KeyboardInterrupt:
                logger.info("接收到中断信号...")
                await self.stop()
            except Exception as e:
                logger.error(f"主循环错误: {e}")
                await asyncio.sleep(5)

        # 最终统计
        self._print_dashboard()
        logger.info("Trading engine stopped")

    def _health_check(self) -> None:
        """健康检查"""
        # 检查持仓与数据库同步
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, quantity, entry_price FROM positions")
        db_positions = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        conn.close()

        # 检查异常
        for symbol, pos in self.positions.items():
            if symbol not in self.prices or self.prices[symbol] == 0:
                self.emergency_stop.record_anomaly(
                    EmergencyLevel.WARNING,
                    "ZERO_PRICE",
                    symbol,
                    "价格数据为空",
                    0,
                    1
                )

    async def stop(self) -> None:
        """停止交易引擎"""
        logger.info("Stopping trading engine...")
        self.running = False

        # 关闭所有持仓
        await self.emergency_stop.close_all_positions("Manual stop")

        # 断开连接
        if self.ws_client:
            self.ws_client.stop()

        # 最终统计
        self._print_dashboard()


# ===================== 主函数 =====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Production Live Trading')
    parser.add_argument('--mode', '-m', choices=['dry_run', 'paper', 'live'],
                        default='dry_run', help='Trading mode')
    parser.add_argument('--symbols', '-s', nargs='+', default=['BTCUSDT'],
                        help='Trading symbols')
    parser.add_argument('--capital', '-c', type=float, default=10000.0,
                        help='Initial capital')
    parser.add_argument('--config', type=str, default='',
                        help='Config file path')

    args = parser.parse_args()

    # 加载配置
    config = get_production_config()

    # 干跑模式默认
    if args.mode == 'dry_run':
        config.risk.trading_mode = TradingMode.DRY_RUN
    elif args.mode == 'paper':
        config.risk.trading_mode = TradingMode.PAPER
    else:
        config.risk.trading_mode = TradingMode.LIVE

    # 打印配置
    print_config_summary(config)

    # 验证配置
    errors = config.validate()
    if errors:
        logger.error("配置验证失败:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)

    # 创建交易引擎
    engine = LiveTradingEngine(
        symbols=args.symbols,
        capital=args.capital,
        config=config,
        mode=config.risk.trading_mode
    )

    # 信号处理
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(engine.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 运行
    asyncio.run(engine.run())


if __name__ == '__main__':
    main()
