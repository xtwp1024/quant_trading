# -*- coding: utf-8 -*-
"""
V36 Paper Trading Runner
=======================

V36策略实盘-paper交易运行器
- 实时数据: Binance WebSocket K线
- 策略信号: V36三种买点
- 执行: Executor(test_mode=True) 模拟成交
- 风控: 内置止损/止盈/时间止损

Usage:
    python v36_paper_trader.py --symbols BTCUSDT ETHUSDT --capital 10000
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_trading.config.v36_config import V36_CRYPTO_OPTIMIZED, SIGNAL_ENHANCER_PARAMS
from quant_trading.connectors.binance_ws import BinanceWebSocketClient
from quant_trading.connectors.binance_rest import BinanceRESTClient
from quant_trading.execution.executor import Executor, OrderSide, OrderType
from quant_trading.experiments.v36_signal_enhancer import V36SignalEnhancer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("V36PaperTrader")


# ===================== 数据模型 =====================

@dataclass
class Position:
    """持仓"""
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    time_stop: datetime  # 超时自动平仓


@dataclass
class Bar:
    """K线数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ===================== V36 信号生成器 =====================

class V36SignalGenerator:
    """
    V36 买入信号生成器

    三种买点:
    1. 企稳: 价格站稳在5日线和10日线上方
    2. 回踩动态支撑: 价格回踩20日低点支撑位
    3. 强势回踩5日线 (已禁用，拖累策略)
    """

    def __init__(self, params: Dict = None):
        self.params = params or V36_CRYPTO_OPTIMIZED.copy()
        self.signal_enhancer = V36SignalEnhancer(SIGNAL_ENHANCER_PARAMS)

        # 缓存历史数据
        self.bars: Dict[str, List[Bar]] = {}

        # 信号阈值
        self.vol_ratio_threshold = self.params.get("vol_ratio_threshold", 1.4)

    def update_bar(self, bar: Bar) -> None:
        """更新K线数据"""
        if bar.symbol not in self.bars:
            self.bars[bar.symbol] = []
        self.bars[bar.symbol].append(bar)

        # 保持足够的历史数据
        max_bars = 100
        if len(self.bars[bar.symbol]) > max_bars:
            self.bars[bar.symbol] = self.bars[bar.symbol][-max_bars:]

    def get_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取pandas DataFrame用于计算"""
        bars = self.bars.get(symbol)
        if not bars or len(bars) < 25:
            return None

        data = {
            'date': [b.timestamp for b in bars],
            'open': [b.open for b in bars],
            'high': [b.high for b in bars],
            'low': [b.low for b in bars],
            'close': [b.close for b in bars],
            'volume': [b.volume for b in bars],
        }
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df

    def check_buy_signals(self, symbol: str) -> Dict:
        """
        检查买入信号

        Returns:
            dict with 'signal' (bool) and 'reason' (str)
        """
        df = self.get_df(symbol)
        if df is None or len(df) < 25:
            return {"signal": False, "reason": "数据不足"}

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 计算技术指标
        ma5 = df['close'].rolling(5, min_periods=1).mean().iloc[-1]
        ma10 = df['close'].rolling(10, min_periods=1).mean().iloc[-1]
        ma20 = df['close'].rolling(20, min_periods=1).mean().iloc[-1]

        # 布林带
        bb_period = 20
        bb_std = df['close'].rolling(bb_period, min_periods=1).std().iloc[-1]
        bb_upper = df['close'].rolling(bb_period, min_periods=1).mean().iloc[-1] + 2 * bb_std
        bb_lower = df['close'].rolling(bb_period, min_periods=1).mean().iloc[-1] - 2 * bb_std

        # Vol Ratio
        vol_ma = df['volume'].rolling(20, min_periods=1).mean().iloc[-1]
        vol_ratio = df['volume'].iloc[-1] / vol_ma if vol_ma > 0 else 0

        # 动态支撑 (20日低点)
        support_20d = df['low'].rolling(20, min_periods=1).min().shift(1).iloc[-1]

        signals = []

        # ========== 买点A: 企稳 ==========
        # 价格站稳在5日线和10日线上方
        buy_a = (latest['close'] > ma5 and latest['close'] > ma10 and
                 prev['close'] <= ma5)

        # ========== 买点B: 回踩动态支撑 ==========
        # 价格回踩20日低点支撑
        near_support = (latest['low'] <= support_20d * 1.02 and
                        latest['close'] > support_20d)
        buy_b = near_support and vol_ratio < self.vol_ratio_threshold

        # 汇总信号
        if buy_a:
            signals.append(("A-企稳", 0.4))
        if buy_b:
            signals.append(("B-回踩支撑", 0.6))

        # 综合评分最高的信号
        if signals:
            best_signal = max(signals, key=lambda x: x[1])
            return {
                "signal": True,
                "buy_point": best_signal[0],
                "strength": best_signal[1],
                "ma5": ma5,
                "ma10": ma10,
                "ma20": ma20,
                "vol_ratio": vol_ratio,
            }

        return {"signal": False, "reason": "无买点"}

    def check_exit_signals(self, symbol: str, position: Position) -> Dict:
        """
        检查平仓信号

        Args:
            symbol: 交易对
            position: 当前持仓

        Returns:
            dict with 'should_exit' (bool) and 'reason' (str)
        """
        df = self.get_df(symbol)
        if df is None or len(df) < 2:
            return {"should_exit": False, "reason": ""}

        latest = df.iloc[-1]
        current_price = latest['close']

        reasons = []

        # 1. 止损
        if current_price < position.stop_loss:
            return {
                "should_exit": True,
                "reason": f"止损: {position.stop_loss:.4f}",
                "pnl_pct": (current_price - position.entry_price) / position.entry_price
            }

        # 2. 止盈
        if current_price > position.take_profit:
            return {
                "should_exit": True,
                "reason": f"止盈: {position.take_profit:.4f}",
                "pnl_pct": (current_price - position.entry_price) / position.entry_price
            }

        # 3. 时间止损
        if datetime.now() > position.time_stop:
            return {
                "should_exit": True,
                "reason": "时间止损",
                "pnl_pct": (current_price - position.entry_price) / position.entry_price
            }

        # 4. 强制止损 (破10日线)
        ma10 = df['close'].rolling(10, min_periods=1).mean().iloc[-1]
        if current_price < ma10:
            reasons.append(f"破MA10: {ma10:.4f}")

        return {"should_exit": False, "reason": ""}


# ===================== Paper Trader =====================

class V36PaperTrader:
    """
    V36 Paper Trading Runner

    流程:
    1. REST API轮询获取K线数据 (或模拟数据模式)
    2. V36SignalGenerator生成信号
    3. Executor(test_mode=True)模拟下单
    4. 实时风控: 止损/止盈/时间止损
    """

    def __init__(
        self,
        symbols: List[str],
        capital: float = 10000.0,
        params: Dict = None,
        api_key: str = "",
        api_secret: str = "",
        poll_interval: int = 60,
        mock_mode: bool = False,  # 模拟数据模式
    ):
        self.symbols = [s.upper() for s in symbols]
        self.capital = capital
        self.params = params or V36_CRYPTO_OPTIMIZED.copy()
        self.poll_interval = poll_interval
        self.mock_mode = mock_mode

        # 初始化组件
        self.rest_client = None
        if not mock_mode:
            try:
                base_url = 'https://www.binance.bh'  # 使用可用端点
                self.rest_client = BinanceRESTClient(api_key, api_secret, base_url=base_url)
                # 测试连接
                self.rest_client.get_klines('BTCUSDT', '1m', limit=1)
                logger.info(f"Binance REST API 连接成功 ({base_url})")
            except Exception as e:
                logger.warning(f"Binance API 连接失败: {e}，切换到模拟模式")
                self.mock_mode = True

        self.executor = Executor(api_key=api_key, api_secret=api_secret, test_mode=True)
        self.signal_gen = V36SignalGenerator(self.params)

        # 持仓管理
        self.positions: Dict[str, Position] = {}
        self.balance = capital
        self.initial_capital = capital

        # 统计
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0

        # ========== 风控参数 ==========
        self.max_pos = 3                    # 最大同时持仓数
        self.max_pos_pct = 0.25             # 单仓最大占比 (25%)
        self.max_total_pct = 0.60            # 总仓位上限 (60%)
        self.min_signal_strength = 0.50      # 最小信号强度
        self.min_trade_interval = 300        # 最小交易间隔(秒)
        self.last_trade_time: Dict[str, float] = {}  # 上次交易时间

        # 模拟数据生成器
        self._mock_prices = {s: 50000.0 for s in self.symbols}

        logger.info(f"V36PaperTrader 初始化:")
        logger.info(f"  交易对: {self.symbols}")
        logger.info(f"  初始资金: {capital}")
        logger.info(f"  风控: max_pos={self.max_pos}, max_pos_pct={self.max_pos_pct}, min_strength={self.min_signal_strength}")
        logger.info(f"  模式: {'模拟数据' if self.mock_mode else '真实REST'}")

    async def on_kline(self, kline_data: Dict) -> None:
        """处理K线数据"""
        try:
            # 解析K线
            kline = kline_data.get('k', kline_data)
            symbol = kline['s']
            ts = datetime.fromtimestamp(kline['t'] / 1000)
            bar = Bar(
                symbol=symbol,
                timestamp=ts,
                open=float(kline['o']),
                high=float(kline['h']),
                low=float(kline['l']),
                close=float(kline['c']),
                volume=float(kline['v']),
            )

            # 更新数据
            self.signal_gen.update_bar(bar)

            # 检查平仓信号
            if symbol in self.positions:
                await self._check_exit(bar)

            # 检查买入信号
            if symbol not in self.positions:
                await self._check_entry(bar)

        except Exception as e:
            logger.error(f"K线处理错误: {e}")

    async def _check_entry(self, bar: Bar) -> None:
        """检查买入信号 (含风控检查)"""
        result = self.signal_gen.check_buy_signals(bar.symbol)
        if not result.get("signal"):
            return

        buy_point = result.get("buy_point", "A-企稳")
        strength = result.get("strength", 0.5)

        # ========== 风控检查 ==========

        # 1. 信号强度检查
        if strength < self.min_signal_strength:
            logger.info(f"[{bar.symbol}] 信号强度不足: {strength:.2f} < {self.min_signal_strength}")
            return

        # 2. 最大持仓数检查
        if len(self.positions) >= self.max_pos:
            logger.info(f"[{bar.symbol}] 达到最大持仓数 {self.max_pos}，跳过")
            return

        # 3. 交易间隔检查
        last_time = self.last_trade_time.get(bar.symbol, 0)
        if time.time() - last_time < self.min_trade_interval:
            logger.info(f"[{bar.symbol}] 交易间隔不足，跳过")
            return

        # 4. 总仓位检查
        total_exposed = self._get_total_exposed()
        if total_exposed >= self.max_total_pct:
            logger.info(f"[{bar.symbol}] 总仓位已满 {total_exposed:.1%}，跳过")
            return

        logger.info(f"[{bar.symbol}] 买入信号: {buy_point}, 强度: {strength:.2f}, 价格: {bar.close:.4f}")

        # 计算仓位
        position_size = self._calculate_position_size(bar.close)

        # 计算止损止盈
        stop_loss = bar.close * (1 + self.params.get("stop_loss", -0.063))
        take_profit = bar.close * (1 + self.params.get("take_profit", 0.072))

        # 时间止损
        hours = self.params.get("time_stop", 14)
        time_stop = datetime.now().replace(
            second=0, microsecond=0
        ) + pd.Timedelta(hours=hours).to_pytimedelta()

        # 下单
        order = await self.executor.place_order(
            symbol=bar.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=position_size,
            price=bar.close,
        )

        if order.status.value == "filled":
            # 记录交易时间
            self.last_trade_time[bar.symbol] = time.time()

            # 记录持仓
            self.positions[bar.symbol] = Position(
                symbol=bar.symbol,
                entry_price=order.avg_fill_price,
                quantity=order.filled_quantity,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_stop=time_stop,
            )

            # 更新余额
            cost = order.filled_quantity * order.avg_fill_price
            self.balance -= cost
            logger.info(f"[{bar.symbol}] 开仓: {order.filled_quantity:.4f} @ {order.avg_fill_price:.4f}, 成本: {cost:.2f}, 余额: {self.balance:.2f}")

    async def _check_exit(self, bar: Bar) -> None:
        """检查平仓信号"""
        position = self.positions.get(bar.symbol)
        if not position:
            return

        result = self.signal_gen.check_exit_signals(bar.symbol, position)
        if not result.get("should_exit"):
            return

        reason = result.get("reason", "未知")
        pnl_pct = result.get("pnl_pct", 0)

        # 平仓
        order = await self.executor.place_order(
            symbol=bar.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            price=bar.close,
        )

        if order.status.value == "filled":
            proceeds = order.filled_quantity * order.avg_fill_price
            self.balance += proceeds

            pnl = proceeds - position.quantity * position.entry_price
            self.total_pnl += pnl
            self.trade_count += 1

            if pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1

            logger.info(
                f"[{bar.symbol}] 平仓: {reason}, "
                f"盈亏: {pnl:.2f} ({pnl_pct:.2%}), 余额: {self.balance:.2f}"
            )

            del self.positions[bar.symbol]

    def _get_total_exposed(self) -> float:
        """计算当前已用仓位比例"""
        total_value = sum(
            pos.quantity * pos.entry_price
            for pos in self.positions.values()
        )
        return total_value / self.initial_capital

    def _calculate_position_size(self, price: float) -> float:
        """计算仓位大小 (基于风控参数)"""
        # 可用资金
        available = self.balance * self.max_pos_pct
        # 不超过剩余可仓位的1/3
        remaining_slots = max(1, self.max_pos - len(self.positions))
        max_per_trade = self.balance * (self.max_total_pct / remaining_slots)
        position_value = min(available, max_per_trade)
        return max(0, position_value / price)

    async def _ws_listener(self) -> None:
        """WebSocket监听任务"""
        async def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get('e') == 'kline':
                    await self.on_kline(data)
            except Exception as e:
                logger.error(f"WebSocket消息处理错误: {e}")

        # 订阅K线流
        streams = [f"{s.lower()}@kline_1m" for s in self.symbols]
        self.ws_client.streams = streams
        self.ws_client.callbacks = {s: on_message for s in streams}

        self.ws_client.start()
        logger.info(f"已订阅K线流: {streams}")

        # 保持运行
        while True:
            await asyncio.sleep(1)

    def print_stats(self) -> None:
        """打印统计"""
        total = self.win_count + self.loss_count
        win_rate = self.win_count / total if total > 0 else 0
        total_return = (self.balance - self.initial_capital) / self.initial_capital

        logger.info("=" * 50)
        logger.info("V36 Paper Trading 统计")
        logger.info("=" * 50)
        logger.info(f"交易次数: {self.trade_count}")
        logger.info(f"胜率: {win_rate:.2%} ({self.win_count}胜/{self.loss_count}负)")
        logger.info(f"总收益: {self.total_pnl:.2f}")
        logger.info(f"收益率: {total_return:.2%}")
        logger.info(f"当前余额: {self.balance:.2f}")
        logger.info(f"持仓数: {len(self.positions)}")

    def _generate_mock_kline(self, symbol: str) -> Bar:
        """生成模拟K线"""
        import random
        price = self._mock_prices[symbol]
        # 随机游走
        change = random.uniform(-0.005, 0.008)  # -0.5% ~ +0.8%
        price *= (1 + change)
        self._mock_prices[symbol] = price

        open_p = price * random.uniform(0.998, 1.002)
        high_p = price * random.uniform(1.001, 1.01)
        low_p = price * random.uniform(0.99, 0.999)
        vol = random.uniform(100, 500)

        return Bar(
            symbol=symbol,
            timestamp=datetime.now(),
            open=open_p,
            high=high_p,
            low=low_p,
            close=price,
            volume=vol,
        )

    async def _fetch_klines(self, symbol: str) -> List:
        """在线程池中获取K线(避免阻塞事件循环)"""
        loop = asyncio.get_event_loop()
        klines = await loop.run_in_executor(
            None, lambda: self.rest_client.get_klines(symbol, '1m', limit=3)
        )
        return klines

    async def run(self) -> None:
        """运行Paper Trader"""
        mode = "模拟数据" if self.mock_mode else "REST轮询"
        logger.info(f"V36 Paper Trader 启动 ({mode}模式)")
        logger.info("-" * 50)

        # 预热: 生成足够的历史数据
        if self.mock_mode:
            logger.info("预热: 生成历史K线数据...")
            for _ in range(60):  # 60根1分钟K线
                for symbol in self.symbols:
                    bar = self._generate_mock_kline(symbol)
                    self.signal_gen.update_bar(bar)

        try:
            iteration = 0
            while True:
                for symbol in self.symbols:
                    try:
                        # 获取K线
                        if self.mock_mode:
                            bar = self._generate_mock_kline(symbol)
                            self.signal_gen.update_bar(bar)
                        else:
                            klines = await self._fetch_klines(symbol)
                            for k in klines:
                                ts = datetime.fromtimestamp(k[0] / 1000)
                                bar = Bar(
                                    symbol=symbol,
                                    timestamp=ts,
                                    open=float(k[1]),
                                    high=float(k[2]),
                                    low=float(k[3]),
                                    close=float(k[4]),
                                    volume=float(k[5]),
                                )
                                self.signal_gen.update_bar(bar)

                        # 检查平仓
                        if symbol in self.positions:
                            await self._check_exit(self.signal_gen.bars[symbol][-1])

                        # 检查开仓
                        if symbol not in self.positions:
                            await self._check_entry(self.signal_gen.bars[symbol][-1])

                    except Exception as e:
                        logger.error(f"[{symbol}] 处理错误: {e}")

                iteration += 1
                if iteration % 5 == 0:  # 每5次迭代打印状态
                    self.print_stats()

                await asyncio.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("接收到中断信号，停止运行...")
            self.print_stats()


# ===================== 主函数 =====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='V36 Paper Trading')
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=['BTCUSDT'],
        help='交易对列表'
    )
    parser.add_argument(
        '--capital', '-c',
        type=float,
        default=10000.0,
        help='初始资金 (默认: 10000)'
    )
    parser.add_argument(
        '--params',
        type=str,
        default='',
        help='参数JSON文件路径'
    )

    args = parser.parse_args()

    # 加载参数
    params = V36_CRYPTO_OPTIMIZED.copy()
    if args.params and os.path.exists(args.params):
        with open(args.params, 'r') as f:
            params.update(json.load(f))

    # 创建trader并运行
    trader = V36PaperTrader(
        symbols=args.symbols,
        capital=args.capital,
        params=params,
    )

    asyncio.run(trader.run())


if __name__ == '__main__':
    main()
