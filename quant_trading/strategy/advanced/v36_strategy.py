# -*- coding: utf-8 -*-
"""
V36 A股趋势策略
三种买点：企稳 + 回踩动态支撑(20日低) + 强势回踩5日线

策略逻辑:
1. 企稳: 价格站稳在5日线和10日线上方
2. 回踩动态支撑: 价格回踩20日低点支撑位
3. 强势回踩5日线: 强势股回调到5日线

风控:
- 强制止损: 破10日线走
- 止盈: 20%或6日内
- 止损: -7%
- 账户回撤: -8%强平

Author: V36
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalDirection, SignalType
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


__all__ = [
    "V36Params",
    "V36Strategy",
    "V36StockPool",
]


# ===================== 默认股票池 =====================
DEFAULT_STOCK_POOL = {
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

DEFAULT_SECTOR_MAP = {
    "603803": "科技", "603499": "科技", "603222": "科技", "000586": "科技",
    "601869": "科技", "300499": "科技", "000062": "科技", "002902": "科技",
    "002384": "科技", "300602": "科技", "002156": "科技", "300042": "科技",
    "300476": "科技",
    "002364": "电力", "000601": "电力", "683339": "电网", "300933": "电网",
    "002645": "材料", "002756": "材料", "002424": "材料",
}


# ===================== 参数 =====================
@dataclass
class V36Params(StrategyParams):
    """V36策略参数"""
    # 股票池
    stock_pool: Dict[str, str] = field(default_factory=lambda: DEFAULT_STOCK_POOL.copy())
    sector_map: Dict[str, str] = field(default_factory=lambda: DEFAULT_SECTOR_MAP.copy())

    # 风控参数
    max_sector: int = 4           # 最大持仓板块数
    slippage: float = 0.01        # 滑点
    market_stop_loss: float = -0.015  # 市场止损
    single_loss_limit: float = -0.07   # 单股止损
    hold_day_limit: int = 6       # 持有天数限制
    max_position_ratio: float = 0.8   # 最大仓位比例
    account_drawdown_limit: float = -0.08  # 账户回撤限制

    # 策略参数
    cash: float = 1000000.0      # 初始资金
    max_pos: int = 8              # 最大持仓数
    stop: float = -0.07           # 止损
    take: float = 0.20             # 止盈
    time_stop: int = 6            # 时间止损(天)

    # 因子参数
    bb_period: int = 20           # 布林带周期
    vol_ma_short: int = 5         # 成交量短周期
    vol_ma_long: int = 20         # 成交量长周期
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])


# ===================== 信号类型 =====================
class V36SignalType:
    """V36信号类型"""
    STABILIZATION = "企稳"           # 价格站稳5日10日线上
    SUPPORT_BOUNCE = "回踩支撑"       # 回踩20日低点
    MA5_BOUNCE = "强势回踩5日线"     # 强势股回调5日线


# ===================== 指标计算 =====================
def calculate_advanced_factors(df: pd.DataFrame, params: V36Params) -> pd.DataFrame:
    """计算高级因子"""
    p = params

    # 布林带
    df['MA20'] = df['close'].rolling(p.bb_period).mean()
    df['STD20'] = df['close'].rolling(p.bb_period).std()
    df['Upper_Band'] = df['MA20'] + df['STD20'] * 2
    df['Lower_Band'] = df['MA20'] - df['STD20'] * 2
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20'].replace(0, np.nan)

    # 成交量因子
    df['Vol_MA5'] = df['volume'].rolling(p.vol_ma_short).mean()
    df['Vol_MA20'] = df['volume'].rolling(p.vol_ma_long).mean()
    df['Vol_Ratio'] = df['Vol_MA5'] / df['Vol_MA20'].replace(0, np.nan)

    # 洗盘因子
    df['Low_10'] = df['low'].rolling(10).min().shift(1)
    df['Spring_Signal'] = np.where(
        (df['low'] < df['Low_10']) &
        (df['close'] > df['Low_10']) &
        (df['close'] > df['open']),
        1, 0
    )

    # 振幅和风险标志
    prev_close = df['close'].shift(1).replace(0, np.nan)
    df['Amplitude'] = (df['high'] - df['low']) / prev_close
    df['Vol_Spike'] = df['volume'] > df['Vol_MA20'] * 2
    df['Risk_Flag'] = np.where(
        (df['Amplitude'] > 0.10) & df['Vol_Spike'] & (df['close'] > df['MA20'] * 1.15),
        1, 0
    )

    # 均线
    for period in p.ma_periods:
        df[f'ma{period}'] = df['close'].rolling(period).mean()

    # 趋势判断
    df['up_trend'] = (df['ma20'] > df['ma20'].shift(1)) & (df['ma60'] > df['ma60'].shift(1))
    df['break_ma20'] = df['close'] < df['ma20'] * 0.96
    df['open_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    # 动态支撑/压力（20日高低点）- shift(1) 避免未来函数
    df['support'] = df['low'].rolling(20).min().shift(1)
    df['resistance'] = df['high'].rolling(20).max().shift(1)

    return df


# ===================== 买点判断 =====================
def is_support_bounce(d: pd.Series, prev_d: pd.Series = None) -> bool:
    """回踩动态支撑位（20日低点）"""
    if prev_d is None:
        return False
    good_trend = d['ma20'] > prev_d['ma20']
    near_support = (d['close'] <= d['support'] * 1.02) & (d['close'] >= d['support'] * 0.98)
    hold_support = d['low'] >= d['support'] * 0.97
    have_support = (
        (d['close'] > d['open']) or
        ((d['close'] - d['low']) > (d['high'] - d['close'])) or
        (d['Vol_Ratio'] < 0.7)
    )
    return good_trend and near_support and hold_support and have_support


def is_strong_ma5_buy(d: pd.Series, prev_d: pd.Series = None, prev2_d: pd.Series = None) -> bool:
    """超级强势股回踩5日线"""
    if prev_d is None or prev2_d is None:
        return False
    good_trend = d['ma20'] > prev_d['ma20']
    ma5_strong = d['ma5'] > prev2_d['ma5']
    touch_ma5 = abs(d['close'] - d['ma5']) / d['ma5'] < 0.015
    hold_ma5 = d['low'] > d['ma5'] * 0.98
    have_support = (d['close'] > d['open']) or (d['close'] > d['low'] * 1.01)
    return good_trend and ma5_strong and touch_ma5 and hold_ma5 and have_support


def is_stabilization(d: pd.Series) -> bool:
    """企稳买点: 价格站稳在5日线和10日线上方"""
    return d['close'] > d['ma5'] and d['close'] > d['ma10']


# ===================== 止损判断 =====================
def is_forced_sell(d: pd.Series) -> bool:
    """强制止损: 破10日线走"""
    break_ma10 = d['close'] < d['ma10'] * 0.98
    gap_dump = (d['open_gap'] > 0.03) and (d['close'] < d['open'] * 0.985)
    return break_ma10 or gap_dump


def should_take_profit(d: pd.Series, cost: float, buy_date: datetime,
                       current_date: datetime, params: V36Params) -> bool:
    """是否应该止盈"""
    ret = (d['close'] - cost) / cost
    days = (current_date - buy_date).days

    # 止盈条件
    if ret >= params.take:
        return True
    # 时间止损
    if days >= params.time_stop:
        return True
    # 破20日线
    if d['break_ma20']:
        return True
    # 止损
    if ret <= params.stop:
        return True

    return False


# ===================== 选股 =====================
class V36StockPool:
    """V36股票池"""

    def __init__(self, stock_pool: Dict[str, str], sector_map: Dict[str, str]):
        self.stock_pool = stock_pool
        self.sector_map = sector_map
        self.stocks: Dict[str, pd.DataFrame] = {}

    def load_data(self, df: pd.DataFrame, params: V36Params) -> None:
        """加载数据"""
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(str).str.zfill(6)

        for code, g in df.groupby("code"):
            if code not in self.stock_pool:
                continue
            g = g.set_index("date").sort_index()
            g = calculate_advanced_factors(g, params)
            self.stocks[code] = g.dropna()

    def pick(self, hold: Set[str], dt: datetime) -> List[str]:
        """选股"""
        pool = []
        for c in self.stocks:
            if c in hold or dt not in self.stocks[c]:
                continue

            df = self.stocks[c]
            idx = df.index.get_loc(dt)

            # 需要至少3行数据来做shift(2)
            if idx < 2:
                continue

            d = df.iloc[idx]
            prev_d = df.iloc[idx - 1]
            prev2_d = df.iloc[idx - 2]

            # 风险标志过滤
            if d['Risk_Flag'] == 1:
                continue
            # 下降趋势过滤
            if d['ma20'] < prev_d['ma20']:
                continue

            # 三种买点
            cond1 = is_stabilization(d)          # 企稳
            cond2 = is_support_bounce(d, prev_d)          # 回踩支撑
            cond3 = is_strong_ma5_buy(d, prev_d, prev2_d)  # 强势回踩5日线

            if not (cond1 or cond2 or cond3):
                continue

            # 高位阴跌过滤
            is_high = d['close'] > d['ma60'] * 1.15
            is_falling = d['close'] < d['open'] and d['close'] < prev_d['close']
            if is_high and is_falling:
                continue

            pool.append(c)
        return pool

    def get_buy_signal_type(self, c: str, dt: datetime) -> str:
        """获取买点类型"""
        if c not in self.stocks or dt not in self.stocks[c]:
            return "unknown"

        df = self.stocks[c]
        idx = df.index.get_loc(dt)

        if idx < 2:
            return "unknown"

        d = df.iloc[idx]
        prev_d = df.iloc[idx - 1]
        prev2_d = df.iloc[idx - 2]

        if is_support_bounce(d, prev_d):
            return V36SignalType.SUPPORT_BOUNCE
        elif is_strong_ma5_buy(d, prev_d, prev2_d):
            return V36SignalType.MA5_BOUNCE
        elif is_stabilization(d):
            return V36SignalType.STABILIZATION
        return "unknown"


# ===================== V36策略 =====================
class V36Strategy(BaseStrategy):
    """
    V36 A股趋势策略

    三种买点:
    1. 企稳: 价格站稳在5日线和10日线上方
    2. 回踩动态支撑: 价格回踩20日低点支撑位
    3. 强势回踩5日线: 强势股回调到5日线

    风控:
    - 强制止损: 破10日线走
    - 止盈: 20%或6日内
    - 止损: -7%
    - 账户回撤: -8%强平
    """

    name = "V36"

    def __init__(self, symbol: str, params: Optional[V36Params] = None) -> None:
        super().__init__(symbol, params)
        self.params = params or V36Params()
        self.pool = V36StockPool(
            self.params.stock_pool,
            self.params.sector_map
        )

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """加载DataFrame数据"""
        self.pool.load_data(df, self.params)

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号（兼容BaseStrategy接口）"""
        signals = []
        if self._data is None:
            self._data = data.copy()

        latest = self._data.iloc[-1]

        # 简单信号逻辑（用于单标的模式）
        if latest.get('up_trend', False):
            if is_stabilization(latest):
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=self.symbol,
                    timestamp=int(pd.Timestamp.now().timestamp() * 1000),
                    price=latest['close'],
                    strength=0.8,
                    reason="企稳信号",
                ))
            elif is_support_bounce(latest):
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=self.symbol,
                    timestamp=int(pd.Timestamp.now().timestamp() * 1000),
                    price=latest['close'],
                    strength=0.7,
                    reason="回踩支撑信号",
                ))
        elif latest.get('break_ma20', False):
            signals.append(Signal(
                type=SignalType.SELL,
                symbol=self.symbol,
                timestamp=int(pd.Timestamp.now().timestamp() * 1000),
                price=latest['close'],
                strength=0.6,
                reason="跌破20日线",
            ))

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """计算仓位大小"""
        return self.params.max_position_ratio / self.params.max_pos

    def get_required_history(self) -> int:
        """获取所需历史数据长度"""
        return 60


# ===================== 完整回测系统 =====================
class V36Backtester:
    """V36回测系统（独立运行）"""

    def __init__(self, params: Optional[V36Params] = None):
        self.params = params or V36Params()
        self.pool = V36StockPool(
            self.params.stock_pool,
            self.params.sector_map
        )
        self.cash = self.params.cash
        self.initial = self.params.cash
        self.pos: Dict[str, Dict[str, Any]] = {}
        self.buy_d: Dict[str, datetime] = {}
        self.nav: List[Dict[str, Any]] = []
        self.trade_count = 0

    def load_csv(self, path: str) -> None:
        """从CSV加载数据"""
        df = pd.read_csv(path, encoding="utf-8-sig")
        self.pool.load_data(df, self.params)

    def load_ashare(self, json_path: str) -> None:
        """从A股JSON文件加载数据"""
        from quant_trading.data.ashare_loader import load_ashare_json
        df = load_ashare_json(json_path)
        # 只加载股票池中的股票
        df = df[df['code'].isin(self.params.stock_pool.keys())]
        self.pool.load_data(df, self.params)

    def _buy(self, c: str, dt: datetime) -> None:
        """买入"""
        if len(self.pos) >= self.params.max_pos:
            return
        if c not in self.pool.stocks or dt not in self.pool.stocks[c]:
            return

        d = self.pool.stocks[c].loc[dt]
        p = d['close'] * (1 + self.params.slippage)
        vol = int(self.cash / self.params.max_pos / p // 100 * 100)
        if vol < 100:
            return

        self.cash -= vol * p
        self.pos[c] = {"vol": vol, "cost": p}
        self.buy_d[c] = dt
        self.trade_count += 1

    def _sell(self, c: str, dt: datetime) -> None:
        """卖出"""
        if c not in self.pos:
            return
        if c not in self.pool.stocks or dt not in self.pool.stocks[c]:
            return

        d = self.pool.stocks[c].loc[dt]
        p = d['close'] * (1 - self.params.slippage)
        self.cash += self.pos[c]["vol"] * p
        self.pos.pop(c)
        self.buy_d.pop(c)

    def run(self) -> pd.DataFrame:
        """运行回测"""
        # 获取所有日期
        all_dates: Set[datetime] = set()
        for c in self.pool.stocks:
            all_dates.update(self.pool.stocks[c].index)
        dates = sorted(all_dates)

        for dt in dates:
            # 总风控
            asset = self.cash
            for c in list(self.pos):
                if c in self.pool.stocks and dt in self.pool.stocks[c]:
                    asset += self.pos[c]["vol"] * self.pool.stocks[c].loc[dt]['close']
            if (asset / self.initial - 1) <= self.params.account_drawdown_limit:
                for c in list(self.pos):
                    self._sell(c, dt)
                continue

            # 强制止损
            for c in list(self.pos):
                if c in self.pool.stocks and dt in self.pool.stocks[c]:
                    d = self.pool.stocks[c].loc[dt]
                    if is_forced_sell(d):
                        self._sell(c, dt)

            # 止盈止损
            for c in list(self.pos):
                if c not in self.pool.stocks or dt not in self.pool.stocks[c]:
                    continue
                d = self.pool.stocks[c].loc[dt]
                ret = (d['close'] - self.pos[c]["cost"]) / self.pos[c]["cost"]
                days = (dt - self.buy_d[c]).days

                if ret <= self.params.stop or ret >= self.params.take or \
                   days >= self.params.time_stop or d['break_ma20']:
                    self._sell(c, dt)

            # 开仓
            for c in self.pool.pick(set(self.pos.keys()), dt):
                self._buy(c, dt)

            # 记录净值
            total = self.cash
            for c in self.pos:
                if c in self.pool.stocks and dt in self.pool.stocks[c]:
                    total += self.pos[c]["vol"] * self.pool.stocks[c].loc[dt]['close']
            self.nav.append({"date": dt, "total": total, "cash": self.cash, "pos_count": len(self.pos)})

        return pd.DataFrame(self.nav)

    def get_result(self) -> Dict[str, Any]:
        """获取回测结果"""
        df = pd.DataFrame(self.nav)
        if df.empty:
            return {"ret": 0, "trade_count": 0}

        ret = (df['total'].iloc[-1] / self.params.cash - 1) * 100

        # 计算最大回撤
        df['peak'] = df['total'].cummax()
        df['drawdown'] = (df['total'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min() * 100

        return {
            "ret": ret,
            "trade_count": self.trade_count,
            "max_drawdown": max_drawdown,
            "final_asset": df['total'].iloc[-1] if not df.empty else self.params.cash,
        }
