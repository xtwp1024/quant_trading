"""
回测框架 - 回测引擎
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .storage import BacktestConfig, DataStorage, SignalGenerator


class OrderSide(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Position:
    """持仓"""
    side: OrderSide = OrderSide.FLAT
    entry_price: float = 0.0
    size: float = 0.0
    entry_time: int = 0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    """交易记录"""
    timestamp: int
    side: OrderSide
    price: float
    size: float
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class EquityPoint:
    """权益曲线点"""
    timestamp: int
    equity: float
    position: Position
    drawdown: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[EquityPoint] = field(default_factory=list)
    final_equity: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "final_equity": self.final_equity,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
        }


class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        config: BacktestConfig = None,
        storage: DataStorage = None,
    ):
        self.config = config or BacktestConfig()
        self.storage = storage or DataStorage()
        self.equity = self.config.initial_balance
        self.peak_equity = self.equity
        self.position = Position()
        self.trades: List[Trade] = []
        self.equity_curve: List[EquityPoint] = []

    def run(self, signal_generator: SignalGenerator) -> BacktestResult:
        """运行回测"""
        # 获取数据
        symbol = self.config.symbol
        timeframe = "1h"

        start_ts = int(pd.Timestamp(self.config.start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(self.config.end_date).timestamp() * 1000)

        df = self.storage.get_ohlcv_dataframe(symbol, timeframe, start_ts, end_ts)

        if df.empty:
            raise ValueError(f"No data found for {symbol} {timeframe}")

        # 生成信号
        signals = signal_generator.generate(df)

        # 合并信号到数据
        df['signal'] = signals.values

        # 遍历K线执行回测
        for idx, row in df.iterrows():
            timestamp = int(row['timestamp'])
            close = row['close']

            # 更新持仓盈亏
            if self.position.side != OrderSide.FLAT:
                self._update_position_pnl(close)

            # 检查信号
            signal = row['signal']

            if signal == 1 and self.position.side != OrderSide.LONG:
                # 做多信号
                if self.position.side == OrderSide.SHORT:
                    self._close_position(close, timestamp)
                self._open_position(OrderSide.LONG, close, timestamp)
            elif signal == -1 and self.position.side != OrderSide.SHORT:
                # 做空信号
                if self.position.side == OrderSide.LONG:
                    self._close_position(close, timestamp)
                self._open_position(OrderSide.SHORT, close, timestamp)
            elif signal == 0 and self.position.side != OrderSide.FLAT:
                # 平仓信号
                self._close_position(close, timestamp)

            # 记录权益曲线
            self._record_equity(timestamp, close)

        # 平掉最后持仓
        if self.position.side != OrderSide.FLAT:
            last_close = df.iloc[-1]['close']
            self._close_position(last_close, int(df.iloc[-1]['timestamp']))

        # 计算结果
        return self._calculate_results()

    def _open_position(self, side: OrderSide, price: float, timestamp: int):
        """开仓"""
        # 计算仓位大小 (使用风险管理的仓位控制)
        # 默认使用10%仓位，避免全仓操作导致的风险
        max_position_pct = getattr(self.config, 'max_position_pct', 0.10)
        position_value = self.equity * max_position_pct
        size = position_value / price

        self.position = Position(
            side=side,
            entry_price=price,
            size=size,
            entry_time=timestamp,
            unrealized_pnl=0.0,
        )

    def _close_position(self, price: float, timestamp: int):
        """平仓"""
        if self.position.side == OrderSide.FLAT:
            return

        # 计算盈亏
        if self.position.side == OrderSide.LONG:
            pnl = (price - self.position.entry_price) * self.position.size
        else:
            pnl = (self.position.entry_price - price) * self.position.size

        # 扣除手续费和滑点
        commission = self.equity * self.config.commission
        slippage_cost = self.equity * self.config.slippage
        net_pnl = pnl - commission - slippage_cost

        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            side=self.position.side,
            price=price,
            size=self.position.size,
            pnl=net_pnl,
            commission=commission,
            slippage=slippage_cost,
        )
        self.trades.append(trade)

        # 更新权益
        self.equity += net_pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # 重置持仓
        self.position = Position()

    def _update_position_pnl(self, current_price: float):
        """更新持仓盈亏"""
        if self.position.side == OrderSide.FLAT:
            return

        if self.position.side == OrderSide.LONG:
            self.position.unrealized_pnl = (
                current_price - self.position.entry_price
            ) * self.position.size
        else:
            self.position.unrealized_pnl = (
                self.position.entry_price - current_price
            ) * self.position.size

    def _record_equity(self, timestamp: int, close: float):
        """记录权益曲线"""
        self._update_position_pnl(close)
        current_equity = self.equity + self.position.unrealized_pnl
        drawdown = self.peak_equity - current_equity
        drawdown_pct = drawdown / self.peak_equity if self.peak_equity > 0 else 0

        self.equity_curve.append(EquityPoint(
            timestamp=timestamp,
            equity=current_equity,
            position=Position(
                side=self.position.side,
                entry_price=self.position.entry_price,
                size=self.position.size,
                entry_time=self.position.entry_time,
                unrealized_pnl=self.position.unrealized_pnl,
            ),
            drawdown=drawdown_pct,
        ))

    def _calculate_results(self) -> BacktestResult:
        """计算回测结果"""
        if not self.trades:
            return BacktestResult(
                config=self.config,
                final_equity=self.equity,
                equity_curve=self.equity_curve,
            )

        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]

        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing)) if losing else 0

        result = BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            final_equity=self.equity,
            max_drawdown=self.peak_equity - min(e.equity for e in self.equity_curve),
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(self.trades) if self.trades else 0,
            avg_win=total_wins / len(winning) if winning else 0,
            avg_loss=total_losses / len(losing) if losing else 0,
            profit_factor=total_wins / total_losses if total_losses > 0 else 0,
        )

        # 计算最大回撤百分比
        if self.equity_curve:
            max_dd = max(e.drawdown for e in self.equity_curve)
            result.max_drawdown_pct = max_dd

        # 计算夏普比率
        if len(self.equity_curve) > 1:
            returns = [
                (self.equity_curve[i+1].equity - self.equity_curve[i].equity)
                / self.equity_curve[i].equity
                for i in range(len(self.equity_curve) - 1)
                if self.equity_curve[i].equity > 0
            ]
            if returns:
                import numpy as np
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * (252**0.5) if np.std(returns) > 0 else 0

        return result


class EMAStrategy(SignalGenerator):
    """EMA交叉策略"""

    def __init__(self, fast: int = 7, slow: int = 20, params: Dict = None):
        super().__init__(params)
        self.fast = fast
        self.slow = slow

    def generate(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=self.fast).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow).mean()

        # 金叉做多，死叉做空
        df['signal'] = 0
        df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1
        df.loc[df['ema_fast'] < df['ema_slow'], 'signal'] = -1

        return df['signal']


class RSIStrategy(SignalGenerator):
    """RSI策略"""

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70, params: Dict = None):
        super().__init__(params)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()

        # 计算RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI超卖做多，RSI超买做空
        df['signal'] = 0
        df.loc[df['rsi'] < self.oversold, 'signal'] = 1
        df.loc[df['rsi'] > self.overbought, 'signal'] = -1

        return df['signal']


class CombinedStrategy(SignalGenerator):
    """组合策略: EMA + RSI + 趋势过滤"""

    def __init__(
        self,
        ema_fast: int = 7,
        ema_slow: int = 20,
        rsi_period: int = 14,
        rsi_oversold: int = 35,
        rsi_overbought: int = 65,
        params: Dict = None,
    ):
        super().__init__(params)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()

        # EMA
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 趋势判断
        df['trend_up'] = df['ema_fast'] > df['ema_slow']

        # 信号
        df['signal'] = 0

        # 做多: 上升趋势 + RSI超卖
        df.loc[df['trend_up'] & (df['rsi'] < self.rsi_oversold), 'signal'] = 1

        # 做空: 下降趋势 + RSI超买
        df.loc[(~df['trend_up']) & (df['rsi'] > self.rsi_overbought), 'signal'] = -1

        return df['signal']
