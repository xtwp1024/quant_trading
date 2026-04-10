# -*- coding: utf-8 -*-
"""
Production Backtest
==================

生产环境回测 - 在实盘交易前验证策略盈利能力

功能:
1. 完整历史数据回测
2. 多市场环境压力测试
3. 策略稳定性验证
4. 风险指标评估

回测完成后必须满足以下条件才能进入实盘:
- 胜率 > 55%
- 盈亏比 > 1.3
- 最大回撤 < 15%
- 夏普比率 > 1.0

Usage:
    python production_backtest.py --symbols BTCUSDT ETHUSDT --start 2024-01-01 --end 2024-12-31
"""

import asyncio
import json
import logging
import os
import sys
import time
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import statistics

import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_trading.backtest.engine import BacktestEngine, BacktestConfig, OrderSide, Position, Trade, EquityPoint
from quant_trading.backtest.storage import DataStorage, OHLCV
from quant_trading.experiments.v36_paper_trader import V36SignalGenerator, Bar
from quant_trading.config.v36_config import V36_CRYPTO_OPTIMIZED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ProductionBacktest")


# ===================== 回测结果 =====================

@dataclass
class BacktestMetrics:
    """回测指标"""
    # 收益指标
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annual_return: float = 0.0
    annual_return_pct: float = 0.0

    # 交易指标
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_loss_ratio: float = 0.0

    # 风险指标
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0

    # 效率指标
    avg_trade_duration_hours: float = 0.0
    avg_profitable_trade_pct: float = 0.0
    avg_losing_trade_pct: float = 0.0

    # 连续性指标
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_consecutive_wins: int = 0
    current_consecutive_losses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": f"${self.total_return:,.2f}",
            "total_return_pct": f"{self.total_return_pct:.2%}",
            "annual_return": f"${self.annual_return:,.2f}",
            "annual_return_pct": f"{self.annual_return_pct:.2%}",
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate:.2%}",
            "profit_loss_ratio": f"{self.profit_loss_ratio:.2f}",
            "max_drawdown": f"${self.max_drawdown:,.2f}",
            "max_drawdown_pct": f"{self.max_drawdown_pct:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
        }


@dataclass
class MarketConditionResult:
    """市场环境结果"""
    condition: str  # BULL / BEAR / SIDEWAYS / VOLATILE
    period: str
    start_date: str
    end_date: str
    return_pct: float
    max_drawdown_pct: float
    trades: int
    win_rate: float


# ===================== 生产回测引擎 =====================

class ProductionBacktestEngine:
    """
    生产回测引擎

    验证策略在:
    1. 牛市环境
    2. 熊市环境
    3. 震荡市场
    4. 高波动市场
    的表现
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000.0,
        commission: float = 0.0004,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = commission

        # 数据存储
        self.storage = DataStorage()

        # 信号生成器
        self.signal_gens: Dict[str, V36SignalGenerator] = {
            symbol: V36SignalGenerator(V36_CRYPTO_OPTIMIZED)
            for symbol in self.symbols
        }

        # 结果
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.metrics = BacktestMetrics()
        self.condition_results: List[MarketConditionResult] = []

        logger.info(f"ProductionBacktestEngine 初始化")
        logger.info(f"  交易对: {self.symbols}")
        logger.info(f"  周期: {start_date} - {end_date}")
        logger.info(f"  初始资金: ${initial_capital:,.2f}")

    def run(self) -> BacktestMetrics:
        """运行完整回测"""
        logger.info("=" * 60)
        logger.info("开始生产回测")
        logger.info("=" * 60)

        # 加载数据
        data = self._load_data()
        if data.empty:
            logger.error("没有数据，回测终止")
            return self.metrics

        # 运行回测
        equity = self.initial_capital
        peak_equity = equity
        position = None
        trades = []
        equity_curve = []

        current_consecutive_wins = 0
        current_consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        trade_start_time = None

        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            symbol = row['symbol']

            # 更新信号生成器
            bar = Bar(
                symbol=symbol,
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            self.signal_gens[symbol].update_bar(bar)

            # 检查开仓信号
            if position is None:
                signal = self.signal_gens[symbol].check_buy_signals(symbol)

                if signal.get("signal"):
                    # 计算仓位 (10% of equity)
                    position_value = equity * 0.10
                    position_size = position_value / current_price

                    # 计算止损止盈
                    stop_loss = current_price * (1 + V36_CRYPTO_OPTIMIZED.get("stop_loss", -0.063))
                    take_profit = current_price * (1 + V36_CRYPTO_OPTIMIZED.get("take_profit", 0.072))

                    position = {
                        'symbol': symbol,
                        'side': 'LONG',
                        'entry_price': current_price,
                        'quantity': position_size,
                        'entry_time': timestamp,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'buy_point': signal.get('buy_point', ''),
                        'signal_strength': signal.get('strength', 0)
                    }
                    trade_start_time = timestamp

            # 检查平仓信号
            if position is not None:
                should_exit = False
                reason = ""
                pnl = 0.0
                pnl_pct = 0.0

                # 止损/止盈检查
                if current_price <= position['stop_loss']:
                    should_exit = True
                    reason = "止损"
                    pnl = (current_price - position['entry_price']) * position['quantity']
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                elif current_price >= position['take_profit']:
                    should_exit = True
                    reason = "止盈"
                    pnl = (current_price - position['entry_price']) * position['quantity']
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']

                # 时间止损 (14小时)
                if trade_start_time:
                    hours_held = (timestamp - trade_start_time).total_seconds() / 3600
                    if hours_held >= V36_CRYPTO_OPTIMIZED.get("time_stop", 14):
                        should_exit = True
                        reason = "时间止损"
                        pnl = (current_price - position['entry_price']) * position['quantity']
                        pnl_pct = (current_price - position['entry_price']) / position['entry_price']

                # 强制止损 (破10日线) - 简化检查
                ma10 = data.iloc[max(0, i-10):i]['close'].mean() if i > 10 else current_price
                if current_price < ma10 * 0.98:  # 2% below MA10
                    should_exit = True
                    reason = "MA10止损"
                    pnl = (current_price - position['entry_price']) * position['quantity']
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']

                if should_exit:
                    # 扣除手续费
                    commission_cost = equity * self.commission
                    net_pnl = pnl - commission_cost

                    # 记录交易
                    trade = {
                        'timestamp': int(timestamp.timestamp() * 1000),
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': net_pnl,
                        'pnl_pct': pnl_pct,
                        'reason': reason,
                        'buy_point': position['buy_point'],
                        'signal_strength': position['signal_strength'],
                        'holding_hours': (timestamp - position['entry_time']).total_seconds() / 3600 if position['entry_time'] else 0
                    }
                    trades.append(trade)

                    # 更新权益
                    equity += net_pnl
                    if equity > peak_equity:
                        peak_equity = equity

                    # 更新连续性统计
                    if net_pnl > 0:
                        current_consecutive_wins += 1
                        current_consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
                    else:
                        current_consecutive_losses += 1
                        current_consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)

                    position = None
                    trade_start_time = None

            # 记录权益曲线
            unrealized_pnl = 0
            if position:
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']

            drawdown = peak_equity - (equity + unrealized_pnl)
            drawdown_pct = drawdown / peak_equity if peak_equity > 0 else 0

            equity_curve.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'equity': equity + unrealized_pnl,
                'cash': equity,
                'position_value': unrealized_pnl,
                'drawdown': drawdown,
                'drawdown_pct': drawdown_pct,
                'in_position': position is not None
            })

        # 保存结果
        self.trades = trades
        self.equity_curve = equity_curve

        # 计算指标
        self._calculate_metrics()

        # 按市场环境分组
        self._analyze_market_conditions()

        return self.metrics

    def _load_data(self) -> pd.DataFrame:
        """加载K线数据"""
        all_data = []

        for symbol in self.symbols:
            # 转换为时间戳
            start_ts = int(pd.Timestamp(self.start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(self.end_date).timestamp() * 1000)

            # 尝试从数据库加载
            try:
                df = self.storage.get_ohlcv_dataframe(symbol, "1h", start_ts, end_ts)
                if not df.empty:
                    df['symbol'] = symbol
                    all_data.append(df)
                    logger.info(f"{symbol}: 加载 {len(df)} 根K线")
                    continue
            except Exception as e:
                logger.warning(f"{symbol}: 数据库加载失败: {e}")

            # 尝试从Binance API获取
            try:
                from quant_trading.connectors.binance_rest import BinanceRESTClient
                client = BinanceRESTClient("", "")
                klines = client.get_klines(symbol, "1h", start=start_ts, limit=500)

                if klines:
                    df_data = []
                    for k in klines:
                        df_data.append({
                            'timestamp': pd.Timestamp(k[0], unit='ms'),
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[5]),
                            'symbol': symbol
                        })

                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    all_data.append(df)

                    # 保存到数据库
                    ohlcv_list = [
                        OHLCV(
                            timestamp=int(k[0]),
                            open=float(k[1]),
                            high=float(k[2]),
                            low=float(k[3]),
                            close=float(k[4]),
                            volume=float(k[5])
                        )
                        for k in klines
                    ]
                    self.storage.save_ohlcv(symbol, "1h", ohlcv_list)

                    logger.info(f"{symbol}: 从API加载 {len(df)} 根K线")
            except Exception as e:
                logger.warning(f"{symbol}: API加载失败: {e}")

        if not all_data:
            return pd.DataFrame()

        # 合并所有数据
        combined = pd.concat(all_data)
        combined = combined.sort_index()

        return combined

    def _calculate_metrics(self) -> None:
        """计算回测指标"""
        if not self.trades:
            logger.warning("没有交易记录")
            return

        trades_df = pd.DataFrame(self.trades)

        # 收益指标
        final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
        self.metrics.total_return = final_equity - self.initial_capital
        self.metrics.total_return_pct = self.metrics.total_return / self.initial_capital

        # 计算年化收益
        if len(self.equity_curve) > 1:
            start_time = self.equity_curve[0]['timestamp']
            end_time = self.equity_curve[-1]['timestamp']
            years = (end_time - start_time) / (365.25 * 24 * 3600 * 1000)
            if years > 0:
                self.metrics.annual_return = self.metrics.total_return / years
                self.metrics.annual_return_pct = ((final_equity / self.initial_capital) ** (1/years) - 1) if years > 0 else 0

        # 交易指标
        self.metrics.total_trades = len(trades_df)
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        self.metrics.winning_trades = len(winning)
        self.metrics.losing_trades = len(losing)
        self.metrics.win_rate = len(winning) / len(trades_df) if len(trades_df) > 0 else 0

        if len(winning) > 0:
            self.metrics.avg_win = winning['pnl'].mean()
            self.metrics.avg_profitable_trade_pct = winning['pnl_pct'].mean() * 100

        if len(losing) > 0:
            self.metrics.avg_loss = losing['pnl'].mean()
            self.metrics.avg_losing_trade_pct = losing['pnl_pct'].mean() * 100

        if abs(self.metrics.avg_loss) > 0:
            self.metrics.profit_loss_ratio = self.metrics.avg_win / abs(self.metrics.avg_loss)

        # 风险指标
        equity_series = pd.Series([e['equity'] for e in self.equity_curve])
        self.metrics.max_drawdown = (equity_series.cummax() - equity_series).max()
        self.metrics.max_drawdown_pct = self.metrics.max_drawdown / self.initial_capital

        # 计算波动率
        if len(equity_curve) > 1:
            returns = equity_series.pct_change().dropna()
            self.metrics.volatility = returns.std() * np.sqrt(365 * 24)  # 年化

            # 夏普比率
            if returns.std() > 0:
                self.metrics.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * 24)

            # 索提诺比率 (只考虑下行波动)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                self.metrics.sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(365 * 24)

        # 卡玛比率
        if self.metrics.max_drawdown > 0:
            self.metrics.calmar_ratio = self.metrics.annual_return / self.metrics.max_drawdown

        # 连续性指标
        self.metrics.max_consecutive_wins = int(trades_df['pnl'] > 0).rolling(10).sum().max() if len(trades_df) > 0 else 0
        self.metrics.max_consecutive_losses = int(trades_df['pnl'] <= 0).rolling(10).sum().max() if len(trades_df) > 0 else 0

        # 平均交易时长
        if 'holding_hours' in trades_df.columns:
            self.metrics.avg_trade_duration_hours = trades_df['holding_hours'].mean()

    def _analyze_market_conditions(self) -> None:
        """分析不同市场环境下的表现"""
        if not self.equity_curve or not self.trades:
            return

        # 将数据分成4个季度
        df = pd.DataFrame(self.equity_curve)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 简化: 按月分组
        df['month'] = df['date'].dt.to_period('M')

        results = []
        for period, group in df.groupby('month'):
            start_equity = group.iloc[0]['equity']
            end_equity = group.iloc[-1]['equity']
            period_return = (end_equity - start_equity) / start_equity

            # 判断市场条件
            if period_return > 0.05:
                condition = "BULL"
            elif period_return < -0.05:
                condition = "BEAR"
            else:
                condition = "SIDEWAYS"

            # 该期间的交易
            period_trades = [t for t in self.trades if
                           period.start_time <= t['timestamp'] / 1000 < period.end_time]

            win_count = len([t for t in period_trades if t['pnl'] > 0])

            results.append(MarketConditionResult(
                condition=condition,
                period=str(period),
                start_date=str(period.start_time.date()),
                end_date=str(period.end_time.date()),
                return_pct=period_return * 100,
                max_drawdown_pct=group['drawdown_pct'].max() * 100,
                trades=len(period_trades),
                win_rate=win_count / len(period_trades) if period_trades else 0
            ))

        self.condition_results = results

    def print_report(self) -> bool:
        """
        打印回测报告

        Returns:
            True if passed all criteria, False otherwise
        """
        print("\n" + "=" * 70)
        print("  PRODUCTION BACKTEST REPORT")
        print("=" * 70)

        # 基本信息
        print(f"\n回测配置:")
        print(f"  交易对: {', '.join(self.symbols)}")
        print(f"  周期: {self.start_date} - {self.end_date}")
        print(f"  初始资金: ${self.initial_capital:,.2f}")
        print(f"  总交易次数: {self.metrics.total_trades}")

        # 收益
        print(f"\n收益指标:")
        print(f"  总收益: ${self.metrics.total_return:+,.2f} ({self.metrics.total_return_pct:+.2%})")
        print(f"  年化收益: ${self.metrics.annual_return:+,.2f} ({self.metrics.annual_return_pct:+.2%})")

        # 交易
        print(f"\n交易指标:")
        print(f"  胜率: {self.metrics.win_rate:.2%} ({self.metrics.winning_trades}胜/{self.metrics.losing_trades}负)")
        print(f"  盈亏比: {self.metrics.profit_loss_ratio:.2f}")
        print(f"  平均盈利: ${self.metrics.avg_win:+,.2f}")
        print(f"  平均亏损: ${self.metrics.avg_loss:+,.2f}")
        print(f"  平均交易时长: {self.metrics.avg_trade_duration_hours:.1f} 小时")

        # 风险
        print(f"\n风险指标:")
        print(f"  最大回撤: ${self.metrics.max_drawdown:,.2f} ({self.metrics.max_drawdown_pct:.2%})")
        print(f"  夏普比率: {self.metrics.sharpe_ratio:.2f}")
        print(f"  索提诺比率: {self.metrics.sortino_ratio:.2f}")
        print(f"  卡玛比率: {self.metrics.calmar_ratio:.2f}")
        print(f"  年化波动率: {self.metrics.volatility:.2%}")

        # 市场环境
        if self.condition_results:
            print(f"\n市场环境分析:")
            for result in self.condition_results:
                print(f"  [{result.condition}] {result.period}: "
                      f"回报 {result.return_pct:+.1f}%, "
                      f"回撤 {result.max_drawdown_pct:.1f}%, "
                      f"交易 {result.trades}, "
                      f"胜率 {result.win_rate:.0%}")

        # 验证标准
        print("\n" + "-" * 70)
        print("实盘准入标准检查:")
        print("-" * 70)

        criteria = [
            ("胜率 > 55%", self.metrics.win_rate > 0.55, f"{self.metrics.win_rate:.2%}"),
            ("盈亏比 > 1.3", self.metrics.profit_loss_ratio > 1.3, f"{self.metrics.profit_loss_ratio:.2f}"),
            ("最大回撤 < 15%", self.metrics.max_drawdown_pct < 0.15, f"{self.metrics.max_drawdown_pct:.2%}"),
            ("夏普比率 > 1.0", self.metrics.sharpe_ratio > 1.0, f"{self.metrics.sharpe_ratio:.2f}"),
            ("交易次数 > 10", self.metrics.total_trades > 10, f"{self.metrics.total_trades}"),
        ]

        all_passed = True
        for name, passed, value in criteria:
            status = "PASS" if passed else "FAIL"
            symbol = "OK" if passed else "NG"
            print(f"  [{symbol}] {name}: {value} ({status})")
            if not passed:
                all_passed = False

        print("\n" + "=" * 70)
        if all_passed:
            print("  STATUS: PASSED - 策略可以进入实盘")
        else:
            print("  STATUS: FAILED - 需要进一步优化策略")
        print("=" * 70)

        return all_passed

    def save_results(self, filepath: str) -> None:
        """保存回测结果"""
        results = {
            'config': {
                'symbols': self.symbols,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
                'commission': self.commission,
            },
            'metrics': self.metrics.to_dict(),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'market_conditions': [
                {
                    'condition': r.condition,
                    'period': r.period,
                    'return_pct': r.return_pct,
                    'max_drawdown_pct': r.max_drawdown_pct,
                    'trades': r.trades,
                    'win_rate': r.win_rate
                }
                for r in self.condition_results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"回测结果已保存到: {filepath}")


# ===================== 压力测试 =====================

class StressTestRunner:
    """压力测试运行器"""

    @staticmethod
    def run_slippage_test(base_results: BacktestMetrics, slippage_pct: float) -> BacktestMetrics:
        """滑点压力测试"""
        # 模拟滑点影响
        metrics = BacktestMetrics()
        # 简化: 滑点每增加0.1%, 收益降低约1%
        slippage_impact = slippage_pct * 10
        metrics.total_return_pct = base_results.total_return_pct * (1 - slippage_impact)
        metrics.max_drawdown_pct = base_results.max_drawdown_pct * (1 + slippage_impact)
        return metrics

    @staticmethod
    def run_commission_test(base_results: BacktestMetrics, commission_pct: float) -> BacktestMetrics:
        """手续费压力测试"""
        metrics = BacktestMetrics()
        metrics.total_return_pct = base_results.total_return_pct - (commission_pct * base_results.total_trades * 2)
        return metrics

    @staticmethod
    def run_volatility_test(base_results: BacktestMetrics, volatility_multiplier: float) -> BacktestMetrics:
        """波动率压力测试"""
        metrics = BacktestMetrics()
        metrics.max_drawdown_pct = base_results.max_drawdown_pct * volatility_multiplier
        metrics.sharpe_ratio = base_results.sharpe_ratio / volatility_multiplier if volatility_multiplier > 0 else 0
        return metrics


# ===================== 主函数 =====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Production Backtest')
    parser.add_argument('--symbols', '-s', nargs='+', default=['BTCUSDT'],
                        help='Trading symbols')
    parser.add_argument('--start', type=str, default='2024-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', '-c', type=float, default=10000.0,
                        help='Initial capital')
    parser.add_argument('--output', '-o', type=str, default='results/production_backtest.json',
                        help='Output file path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (for compatibility with run_all_experiments)')

    args = parser.parse_args()

    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 创建回测引擎
    engine = ProductionBacktestEngine(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )

    # 运行回测
    metrics = engine.run()

    # 打印报告
    passed = engine.print_report()

    # 保存结果
    engine.save_results(args.output)

    # 压力测试
    print("\n" + "=" * 70)
    print("  STRESS TESTS")
    print("=" * 70)

    stress = StressTestRunner()

    # 滑点测试
    print("\n滑点压力测试:")
    for slippage in [0.001, 0.002, 0.005, 0.01]:
        stress_metrics = stress.run_slippage_test(metrics, slippage)
        print(f"  {slippage:.1%} 滑点: 回报 {stress_metrics.total_return_pct:+.2%}, "
              f"回撤 {stress_metrics.max_drawdown_pct:.2%}")

    # 手续费测试
    print("\n手续费压力测试:")
    for commission in [0.0004, 0.001, 0.002]:
        stress_metrics = stress.run_commission_test(metrics, commission)
        print(f"  {commission:.2%} 手续费: 回报 {stress_metrics.total_return_pct:+.2%}")

    # 波动率测试
    print("\n波动率压力测试:")
    for vol_mult in [1.5, 2.0, 3.0]:
        stress_metrics = stress.run_volatility_test(metrics, vol_mult)
        print(f"  {vol_mult}x 波动率: 回撤 {stress_metrics.max_drawdown_pct:.2%}, "
              f"夏普 {stress_metrics.sharpe_ratio:.2f}")

    # 退出码
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
