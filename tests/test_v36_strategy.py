# -*- coding: utf-8 -*-
"""V36 A股趋势策略单元测试"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from quant_trading.strategies.v36_strategy import (
    V36Params,
    V36StockPool,
    V36SignalType,
    is_stabilization,
    is_support_bounce,
    is_strong_ma5_buy,
    is_forced_sell,
    should_take_profit,
    calculate_advanced_factors,
    DEFAULT_STOCK_POOL,
    DEFAULT_SECTOR_MAP,
)


# ===================== 测试数据生成 =====================

def make_df(bars: list[dict], code: str = "000001") -> pd.DataFrame:
    """生成测试用DataFrame"""
    df = pd.DataFrame(bars)
    df["date"] = pd.date_range("2024-01-01", periods=len(bars), freq="D")
    df["code"] = code
    return df


def make_series(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1_000_000,
    ma5: float | None = None,
    ma10: float | None = None,
    ma20: float | None = None,
    ma60: float | None = None,
    support: float | None = None,
    resistance: float | None = None,
    Vol_Ratio: float | None = None,
    open_gap: float = 0.0,
    break_ma20: bool = False,
) -> pd.Series:
    """生成测试用pd.Series（模拟单根K线+指标）"""
    o = open_ if open_ is not None else close * 0.99
    h = high if high is not None else close * 1.02
    l = low if low is not None else close * 0.98
    vol = volume

    # 如果没有提供均线，计算简单值
    m5 = ma5 if ma5 is not None else close * 0.98
    m10 = ma10 if ma10 is not None else close * 0.96
    m20 = ma20 if ma20 is not None else close * 0.94
    m60 = ma60 if ma60 is not None else close * 0.90
    sup = support if support is not None else close * 0.90
    res = resistance if resistance is not None else close * 1.10
    vol_r = Vol_Ratio if Vol_Ratio is not None else 1.0

    return pd.Series({
        "open": o,
        "high": h,
        "low": l,
        "close": close,
        "volume": vol,
        "ma5": m5,
        "ma10": m10,
        "ma20": m20,
        "ma60": m60,
        "support": sup,
        "resistance": res,
        "Vol_Ratio": vol_r,
        "open_gap": open_gap,
        "break_ma20": break_ma20,
    })


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """为DataFrame添加技术指标"""
    params = V36Params()
    df = calculate_advanced_factors(df, params)
    return df


# ===================== 买点测试 =====================

class TestStabilization:
    """企稳买点测试"""

    def test_stabilization_above_ma5_ma10(self):
        """收盘价在5日和10日线上方时应触发企稳信号"""
        d = make_series(
            close=110.0,
            ma5=105.0,
            ma10=108.0,
            ma20=100.0,
        )
        assert is_stabilization(d) == True

    def test_stabilization_below_ma5(self):
        """收盘价在5日线下方时应不触发企稳信号"""
        d = make_series(
            close=100.0,
            ma5=110.0,
            ma10=108.0,
            ma20=100.0,
        )
        assert is_stabilization(d) == False

    def test_stabilization_below_ma10(self):
        """收盘价在10日线下方但5日线上方时应不触发"""
        d = make_series(
            close=100.0,
            ma5=98.0,
            ma10=105.0,
            ma20=100.0,
        )
        assert is_stabilization(d) == False


class TestSupportBounce:
    """回踩动态支撑测试"""

    def test_support_bounce_triggered(self):
        """ma20向上且价格回踩20日低点时触发"""
        current = make_series(
            close=102.0,
            open_=101.0,
            high=103.0,
            low=99.0,
            ma5=100.0,
            ma10=98.0,
            ma20=100.0,  # 向上
            support=100.0,
            Vol_Ratio=0.5,  # 缩量
        )
        prev = make_series(
            close=101.0,
            ma20=99.0,  # ma20 前一天更低
        )
        assert is_support_bounce(current, prev) == True

    def test_support_bounce_failing_ma20_falling(self):
        """ma20向下时不触发回踩支撑"""
        current = make_series(
            close=102.0,
            open_=101.0,
            high=103.0,
            low=99.0,
            ma5=100.0,
            ma10=98.0,
            ma20=99.0,  # 向下
            support=100.0,
        )
        prev = make_series(
            close=101.0,
            ma20=100.0,  # ma20 前一天更高
        )
        assert is_support_bounce(current, prev) == False

    def test_support_bounce_no_prev_data(self):
        """没有前一天数据时应返回False"""
        d = make_series(close=102.0)
        assert is_support_bounce(d, None) == False


class TestStrongMA5Buy:
    """强势回踩5日线测试"""

    def test_ma5_bounce_triggered(self):
        """强势股回踩5日线时触发"""
        current = make_series(
            close=103.5,  # 贴近5日线
            open_=103.0,
            high=105.0,
            low=102.5,
            ma5=104.0,  # 均线向上
            ma10=100.0,
            ma20=98.0,
        )
        prev = make_series(
            close=102.0,
            ma5=103.0,  # 前一天5日线更低（加速）
            ma20=97.5,
        )
        prev2 = make_series(
            close=101.0,
            ma5=101.5,  # 再前一天更低
            ma20=97.0,
        )
        assert is_strong_ma5_buy(current, prev, prev2) == True

    def test_ma5_bounce_no_ma5_strengthening(self):
        """5日线不加速时不触发"""
        # ma5_strong: d['ma5'] > prev2_d['ma5'] -> 104 > 104 = False
        current = make_series(
            close=103.5,
            open_=103.0,
            high=105.0,
            low=102.5,
            ma5=104.0,
            ma10=100.0,
            ma20=98.0,
        )
        prev = make_series(
            close=102.0,
            ma5=103.5,  # 5日线反而更高（减速）
            ma20=97.5,
        )
        prev2 = make_series(
            close=101.0,
            ma5=104.0,  # 再前一天更高
            ma20=97.0,
        )
        assert is_strong_ma5_buy(current, prev, prev2) == False

    def test_ma5_bounce_no_prev2_data(self):
        """没有前两天数据时应返回False"""
        d = make_series(close=103.5)
        prev = make_series(close=102.0)
        assert is_strong_ma5_buy(d, prev, None) == False


# ===================== 止损测试 =====================

class TestForcedSell:
    """强制止损测试"""

    def test_break_ma10_triggers_stop(self):
        """跌破10日线98%应触发强制止损"""
        d = make_series(
            close=96.0,  # 跌破10日线98%
            open_=100.0,
            high=101.0,
            low=95.0,
            ma10=100.0,
        )
        assert is_forced_sell(d) == True

    def test_gap_dump_triggers_stop(self):
        """高开低走（跌幅>3%）应触发止损"""
        d = make_series(
            close=101.0,  # 高开低走
            open_=105.0,  # 高开5%
            high=106.0,
            low=100.0,
            ma10=100.0,
            open_gap=0.05,  # 5%跳空
        )
        assert is_forced_sell(d) == True

    def test_no_stop_when_above_ma10(self):
        """价格在10日线上方时不触发止损"""
        d = make_series(
            close=105.0,
            open_=104.0,
            high=106.0,
            low=103.0,
            ma10=100.0,
        )
        assert is_forced_sell(d) == False


class TestShouldTakeProfit:
    """止盈止损判断测试"""

    def setup_method(self):
        """设置测试参数"""
        self.params = V36Params()

    def test_take_profit_at_20_percent(self):
        """盈利达到20%时应止盈"""
        d = make_series(close=120.0)
        buy_date = datetime(2024, 1, 1)
        current_date = datetime(2024, 1, 2)
        assert should_take_profit(d, 100.0, buy_date, current_date, self.params) == True

    def test_stop_loss_at_7_percent(self):
        """亏损达到7%时应止损"""
        d = make_series(close=93.0)
        buy_date = datetime(2024, 1, 1)
        current_date = datetime(2024, 1, 2)
        assert should_take_profit(d, 100.0, buy_date, current_date, self.params) == True

    def test_time_stop_at_6_days(self):
        """持有6天后应触发时间止损"""
        d = make_series(close=102.0)  # 盈利2%
        buy_date = datetime(2024, 1, 1)
        current_date = datetime(2024, 1, 7)  # 第7天
        assert should_take_profit(d, 100.0, buy_date, current_date, self.params) == True

    def test_no_exit_when_profitable(self):
        """盈利未达目标且未超时时应继续持有"""
        d = make_series(close=105.0)  # 盈利5%
        buy_date = datetime(2024, 1, 1)
        current_date = datetime(2024, 1, 3)  # 第3天
        assert should_take_profit(d, 100.0, buy_date, current_date, self.params) == False


# ===================== 边界条件测试 =====================

class TestEdgeCases:
    """边界条件测试"""

    def test_nan_in_close_returns_false(self):
        """收盘价为NaN时应返回False（不崩溃）"""
        d = make_series(close=np.nan)
        result = is_stabilization(d)
        assert result == False

    def test_nan_in_ma5_returns_false(self):
        """ma5为NaN时应返回False"""
        d = make_series(close=100.0, ma5=np.nan, ma10=110.0)
        result = is_stabilization(d)
        assert result == False

    def test_zero_support_handled(self):
        """支撑为0时不应崩溃"""
        d = make_series(close=100.0, support=0.0)
        prev = make_series(close=101.0, ma20=100.0)
        # 不应崩溃
        result = is_support_bounce(d, prev)
        assert result == False

    def test_zero_vol_ratio_handled(self):
        """Vol_Ratio为0时应能处理"""
        d = make_series(close=100.0, Vol_Ratio=0.0, ma5=98.0, ma10=96.0, ma20=94.0, support=90.0)
        prev = make_series(close=99.0, ma20=93.0)
        # 不应崩溃
        result = is_support_bounce(d, prev)
        # 条件判断应该能处理
        assert isinstance(result, (bool, np.bool_))


# ===================== 过滤器测试 =====================

class TestFilters:
    """过滤器测试"""

    def test_risk_flag_high_amplitude(self):
        """高位大振幅放量应被Risk_Flag标记"""
        # Risk_Flag 条件:
        # Amplitude > 0.10 AND Vol_Spike AND close > MA20 * 1.15
        # 这个测试验证 Risk_Flag 字段存在且可以被计算
        bars = []
        for i in range(30):
            close = 100.0 + i
            vol = 1_000_000
            bars.append({
                "open": close * 0.98,
                "high": close * 1.05,
                "low": close * 0.97,
                "close": close * 1.02,
                "volume": vol,
            })
        df = make_df(bars)
        df = add_indicators(df)
        d = df.iloc[-1]
        # Risk_Flag 字段应该存在且为0（正常情况）
        assert "Risk_Flag" in d
        assert d["Risk_Flag"] == 0


# ===================== V36StockPool 测试 =====================

class TestV36StockPool:
    """V36股票池测试"""

    def test_pool_loads_successfully(self):
        """股票池应能正确加载数据"""
        pool = V36StockPool(DEFAULT_STOCK_POOL, DEFAULT_SECTOR_MAP)
        bars = []
        for i in range(30):
            close = 100.0 + i
            bars.append({
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
            })
        df = make_df(bars, code="603803")
        pool.load_data(df, V36Params())
        assert "603803" in pool.stocks

    def test_pool_rejects_unknown_code(self):
        """不在股票池中的股票应被忽略"""
        pool = V36StockPool(DEFAULT_STOCK_POOL, DEFAULT_SECTOR_MAP)
        bars = []
        for i in range(30):
            close = 100.0 + i
            bars.append({
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
            })
        df = make_df(bars, code="999999")  # 不在池中
        pool.load_data(df, V36Params())
        assert "999999" not in pool.stocks


# ===================== 运行测试 =====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
