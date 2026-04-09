#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V36 Trend Strategy - 趋势交易策略
3种买点: 企稳 + 回踩动态支撑(20日低) + 强势回踩5日线

集成到量化之神股票池分类系统
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from quant_trading.signal.stock_pool import StockPoolManager, PoolType
from quant_trading.strategy.base import BaseStrategy, StrategyParams, Signal
from quant_trading.strategy.context import StrategyContext

logger = logging.getLogger("V36Strategy")


# ===================== V36 股票池 =====================
V36_STOCK_POOL = {
    "603803": "通信-CPO",
    "603499": "通信算力",
    "603222": "通信趋势",
    "000586": "通信妖股",
    "601869": "通信光缆",
    "300499": "光模块",
    "000062": "电子",
    "002902": "PCB",
    "002384": "PCB",
    "300602": "5G",
    "002364": "电力",
    "000601": "电力",
    "683339": "电网",
    "300933": "电网",
    "002156": "半导体",
    "300042": "存储",
    "300476": "PCB",
    "002645": "稀土",
    "002756": "特钢",
    "002424": "锗业",
}

V36_SECTOR_MAP = {
    "603803": "科技", "603499": "科技", "603222": "科技", "000586": "科技",
    "601869": "科技", "300499": "科技", "000062": "科技", "002902": "科技",
    "002384": "科技", "300602": "科技", "002156": "科技", "300042": "科技",
    "300476": "科技",
    "002364": "电力", "000601": "电力", "683339": "电网", "300933": "电网",
    "002645": "材料", "002756": "材料", "002424": "材料",
}


@dataclass
class V36Params(StrategyParams):
    """V36策略参数"""
    cash: float = 1000000
    max_pos: int = 8
    stop_loss: float = -0.07
    take_profit: float = 0.20
    time_stop: int = 6
    slippage: float = 0.01
    account_drawdown_limit: float = -0.08


class V36TrendStrategy(BaseStrategy):
    """
    V36趋势交易策略

    3种买点:
    1. 企稳: close > ma5 AND close > ma10
    2. 回踩动态支撑: 股价回踩20日低点
    3. 强势回踩5日线: 强势股回踩5日线
    """

    name: str = "v36_trend"

    def __init__(
        self,
        symbol: str,
        params: Optional[V36Params] = None,
        stock_pool: Optional[Dict[str, str]] = None
    ) -> None:
        super().__init__(symbol, params)
        self.stock_pool = stock_pool or V36_STOCK_POOL
        self.params = params or V36Params()
        self.pos: Dict[str, Dict[str, Any]] = {}
        self.buy_date: Dict[str, pd.Timestamp] = {}
        self.trade_count: int = 0

    def calculate_advanced_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算洗盘因子"""
        df = df.copy()

        # Bollinger Bands
        df['MA20'] = df['close'].rolling(20).mean()
        df['STD20'] = df['close'].rolling(20).std()
        df['Upper_Band'] = df['MA20'] + df['STD20'] * 2
        df['Lower_Band'] = df['MA20'] - df['STD20'] * 2
        df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']

        # Volume
        df['Vol_MA5'] = df['volume'].rolling(5).mean()
        df['Vol_MA20'] = df['volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Vol_MA5'] / df['Vol_MA20']

        # Spring Signal
        df['Low_10'] = df['low'].rolling(10).min().shift(1)
        df['Spring_Signal'] = np.where(
            (df['low'] < df['Low_10']) & (df['close'] > df['Low_10']) & (df['close'] > df['open']),
            1, 0
        )

        # Risk Flag
        df['Amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)
        df['Vol_Spike'] = df['volume'] > df['Vol_MA20'] * 2
        df['Risk_Flag'] = np.where(
            (df['Amplitude'] > 0.10) & df['Vol_Spike'] & (df['close'] > df['MA20'] * 1.15),
            1, 0
        )

        # Moving averages
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()

        # Trend
        df['up_trend'] = (df['ma20'] > df['ma20'].shift(1)) & (df['ma60'] > df['ma60'].shift(1))
        df['break_ma20'] = df['close'] < df['ma20'] * 0.96
        df['open_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Dynamic support/resistance (shift(1) to avoid look-ahead bias)
        df['support'] = df['low'].rolling(20).min().shift(1)
        df['resistance'] = df['high'].rolling(20).max().shift(1)

        return df

    @staticmethod
    def is_support_bounce(d: pd.Series, prev_ma20: float = 0) -> bool:
        """回踩动态支撑位（20日低点）"""
        good_trend = d.ma20 > prev_ma20 if prev_ma20 > 0 else d.ma20 > d.ma20
        near_support = (d.close <= d.support * 1.02) & (d.close >= d.support * 0.98)
        hold_support = d.low >= d.support * 0.97
        have_support = (
            (d.close > d.open) or
            ((d.close - d.low) > (d.high - d.close)) or
            (d.Vol_Ratio < 0.7)
        )
        return good_trend and near_support and hold_support and have_support

    @staticmethod
    def is_strong_ma5_buy(d: pd.Series, prev_ma5: float = 0, prev_ma20: float = 0) -> bool:
        """超级强势股回踩5日线"""
        good_trend = d.ma20 > prev_ma20 if prev_ma20 > 0 else d.ma20 > d.ma20
        ma5_strong = d.ma5 > prev_ma5 if prev_ma5 > 0 else d.ma5 > d.ma5
        touch_ma5 = abs(d.close - d.ma5) / d.ma5 < 0.015
        hold_ma5 = d.low > d.ma5 * 0.98
        have_support = (d.close > d.open) or (d.close > d.low * 1.01)
        return good_trend and ma5_strong and touch_ma5 and hold_ma5 and have_support

    def is_stabilization(self, d: pd.Series) -> bool:
        """企稳买点"""
        return (d.close > d.ma5) and (d.close > d.ma10)

    def check_buy_signals(
        self,
        d: pd.Series,
        prev_ma20: float,
        prev_ma5: float = 0
    ) -> tuple[bool, str]:
        """
        检查是否有买入信号

        Args:
            d: 当前行数据
            prev_ma20: 前一日MA20
            prev_ma5: 前一日MA5

        Returns:
            (has_signal, signal_type)
        """
        # 风险过滤
        if d.Risk_Flag == 1:
            return False, ""

        # 趋势过滤
        if d.ma20 < prev_ma20:
            return False, ""

        # 买点检测
        if self.is_stabilization(d):
            return True, "企稳"

        if self.is_support_bounce(d, prev_ma20):
            return True, "回踩支撑"

        if self.is_strong_ma5_buy(d, prev_ma5, prev_ma20):
            return True, "强势回踩5日线"

        return False, ""

    def check_sell_signals(self, d: pd.Series, cost: float) -> tuple[bool, str]:
        """
        检查是否应该卖出

        Returns:
            (should_sell, reason)
        """
        ret = (d.close - cost) / cost

        # 止损
        if ret <= self.params.stop_loss:
            return True, f"止损({ret:.1%})"

        # 止盈
        if ret >= self.params.take_profit:
            return True, f"止盈({ret:.1%})"

        # 时间止损
        return False, ""

    def is_forced_sell(self, d: pd.Series) -> bool:
        """强制止损: 破10日线"""
        break_ma10 = d.close < d.ma10 * 0.98
        gap_dump = d.open_gap > 0.03 and d.close < d.open * 0.985
        return break_ma10 or gap_dump

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        if len(data) < 60:
            return []

        signals = []
        df = self.calculate_advanced_factors(data)

        for i, (dt, d) in enumerate(df.iterrows()):
            if pd.isna(d.ma20) or pd.isna(d.ma5):
                continue

            # Get previous row values for shift comparisons
            prev_ma20 = df.iloc[i-1].ma20 if i > 0 else d.ma20
            prev_ma5 = df.iloc[i-1].ma5 if i > 0 else d.ma5

            # Convert datetime to milliseconds timestamp
            ts = int(pd.Timestamp(dt).timestamp() * 1000)

            has_signal, signal_type = self.check_buy_signals(d, prev_ma20, prev_ma5)
            if has_signal:
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=self.symbol,
                    timestamp=ts,
                    price=d.close,
                    strength=0.8,
                    reason=signal_type,
                ))

            # 检查持仓是否需要卖出
            if self.symbol in self.pos:
                should_sell, reason = self.check_sell_signals(d, self.pos[self.symbol]["cost"])
                if should_sell or self.is_forced_sell(d):
                    signals.append(Signal(
                        type=SignalType.EXIT_LONG,
                        symbol=self.symbol,
                        timestamp=ts,
                        price=d.close,
                        strength=1.0,
                        reason=reason,
                    ))

        return signals

    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        """计算仓位大小"""
        max_pos = self.params.max_pos
        return self.params.cash / max_pos / signal.price

    def update_position(self, symbol: str, action: str, price: float, vol: int = 0) -> None:
        """更新持仓"""
        if action == "buy":
            self.pos[symbol] = {"vol": vol, "cost": price}
            self.trade_count += 1
        elif action == "sell":
            if symbol in self.pos:
                del self.pos[symbol]


class V36Backtester:
    """
    V36策略回测器

    用于批量回测V36策略在各股票上的表现
    """

    def __init__(
        self,
        stocks_data: Dict[str, pd.DataFrame],
        params: Optional[V36Params] = None
    ):
        self.stocks_data = stocks_data
        self.params = params or V36Params()
        self.results: Dict[str, Dict[str, Any]] = {}

    def run(self) -> Dict[str, Dict[str, Any]]:
        """运行回测"""
        for symbol, df in self.stocks_data.items():
            if len(df) < 60:
                continue

            strategy = V36TrendStrategy(symbol, self.params)
            result = self._backtest_stock(strategy, df)
            if result:
                self.results[symbol] = result

        return self.results

    def _backtest_stock(self, strategy: V36TrendStrategy, df: pd.DataFrame) -> Optional[Dict]:
        """回测单只股票"""
        df = strategy.calculate_advanced_factors(df)
        trades = []
        pos = None
        buy_date = None
        trade_count = 0

        for i, (dt, d) in enumerate(df.iterrows()):
            if pd.isna(d.ma20):
                continue

            # Get previous row values
            prev_ma20 = df.iloc[i-1].ma20 if i > 0 else d.ma20
            prev_ma5 = df.iloc[i-1].ma5 if i > 0 else d.ma5

            # 买入逻辑
            if pos is None:
                has_signal, signal_type = strategy.check_buy_signals(d, prev_ma20, prev_ma5)
                if has_signal:
                    cost = d.close * (1 + self.params.slippage)
                    vol = int(self.params.cash / self.params.max_pos / cost // 100 * 100)
                    if vol >= 100:
                        pos = {"vol": vol, "cost": cost}
                        buy_date = pd.Timestamp(dt)
                        trade_count += 1
                        trades.append({
                            "buy_date": dt,
                            "buy_price": cost,
                            "signal_type": signal_type
                        })

            # 卖出逻辑
            elif pos is not None:
                sell = False
                reason = ""

                ret = (d.close - pos["cost"]) / pos["cost"]
                current_dt = pd.Timestamp(dt)
                days = (current_dt - buy_date).days

                # 止损/止盈/时间止损
                if ret <= self.params.stop_loss:
                    sell = True
                    reason = "止损"
                elif ret >= self.params.take_profit:
                    sell = True
                    reason = "止盈"
                elif days >= self.params.time_stop:
                    sell = True
                    reason = "时间止损"
                elif strategy.is_forced_sell(d):
                    sell = True
                    reason = "强制止损"

                if sell:
                    sell_price = d.close * (1 - self.params.slippage)
                    trades[-1].update({
                        "sell_date": dt,
                        "sell_price": sell_price,
                        "return": ret,
                        "reason": reason,
                        "days": days
                    })
                    pos = None

        if trades:
            returns = [t.get("return", 0) for t in trades if "return" in t]
            winning = sum(1 for r in returns if r > 0)
            return {
                "trades": trades,
                "trade_count": trade_count,
                "win_rate": winning / len(returns) if returns else 0,
                "avg_return": np.mean(returns) if returns else 0,
                "total_return": sum(returns) if returns else 0,
            }
        return None

    def print_summary(self) -> None:
        """打印回测摘要"""
        if not self.results:
            print("No results")
            return

        print("=" * 70)
        print("V36 Trend Strategy Backtest Summary")
        print("=" * 70)

        total_trades = sum(r["trade_count"] for r in self.results.values())
        all_returns = []
        for symbol, result in self.results.items():
            for trade in result["trades"]:
                if "return" in trade:
                    all_returns.append(trade["return"])

        if all_returns:
            winning = sum(1 for r in all_returns if r > 0)
            print(f"Total Trades: {total_trades}")
            print(f"Winning Rate: {winning / len(all_returns):.1%}")
            print(f"Avg Return: {np.mean(all_returns):.2%}")
            print(f"Total Return: {sum(all_returns):.2%}")

        print()
        print("Top Performers:")
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].get("total_return", 0),
            reverse=True
        )
        for symbol, result in sorted_results[:5]:
            print(f"  {symbol}: {result.get('total_return', 0):.2%} ({result.get('trade_count', 0)} trades)")

        print("=" * 70)


def run_v36_backtest(
    stock_data: Dict[str, pd.DataFrame],
    initial_cash: float = 1000000,
    max_pos: int = 8
) -> V36Backtester:
    """
    运行V36回测的便捷函数

    Args:
        stock_data: {symbol: DataFrame} 股票数据
        initial_cash: 初始资金
        max_pos: 最大持仓数

    Returns:
        V36Backtester回测结果
    """
    params = V36Params(
        cash=initial_cash,
        max_pos=max_pos,
        stop_loss=-0.07,
        take_profit=0.20,
        time_stop=6
    )

    backtester = V36Backtester(stock_data, params)
    results = backtester.run()
    return backtester


# ===================== CLI =====================
def main():
    """CLI入口"""
    import argparse
    from quant_trading.signal.stock_pool import AStockDataProvider
    import numpy as np

    parser = argparse.ArgumentParser(description="V36 Trend Strategy")
    parser.add_argument("--symbols", nargs="*", help="Stock symbols to backtest")
    parser.add_argument("--cash", type=float, default=1000000, help="Initial cash")
    parser.add_argument("--max-pos", type=int, default=8, help="Max positions")
    args = parser.parse_args()

    # 默认使用V36股票池
    symbols = args.symbols or list(V36_STOCK_POOL.keys())

    print(f"Loading data for {len(symbols)} stocks...")

    # 获取数据
    provider = AStockDataProvider()
    symbols_ohlcv, _ = provider.fetch_batch(symbols)

    if not symbols_ohlcv:
        print("No data loaded")
        return

    # 转换为DataFrame
    stock_data = {}
    for symbol, ohlcv in symbols_ohlcv.items():
        if len(ohlcv) >= 60:
            df = pd.DataFrame(ohlcv, columns=["open", "high", "low", "close", "volume"])
            stock_data[symbol] = df

    print(f"Loaded {len(stock_data)} stocks with sufficient data")

    # 运行回测
    backtester = run_v36_backtest(stock_data, args.cash, args.max_pos)
    backtester.print_summary()


if __name__ == "__main__":
    main()
