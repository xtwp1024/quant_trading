# -*- coding: utf-8 -*-
"""
V36 买点假设验证实验

实验目标:
- 买点A: 只有企稳（stabilization）
- 买点B: 只有回踩支撑（support bounce）
- 买点C: 只有强势回踩5日线（MA5 bounce）
- 买点D: 任意买点（当前逻辑）

验证指标: 胜率、盈亏比、总收益率、夏普比率、最大回撤
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os

from quant_trading.backtester.suite_engine import (
    BacktestDataset,
    StrategySignals,
    SuiteRiskConfig,
    SizingConfig,
    StopConfig,
    run_simple_backtest,
)
from quant_trading.strategy.advanced.v36_strategy import (
    V36Params,
    V36StockPool,
    DEFAULT_STOCK_POOL,
    calculate_advanced_factors,
    is_stabilization,
    is_support_bounce,
    is_strong_ma5_buy,
    is_forced_sell,
)
from quant_trading.data.ashare_loader import load_ashare_json


# ===================== 实验配置 =====================

@dataclass
class BuyPointVariant:
    """买点变体配置"""
    name: str
    short_name: str
    use_stabilization: bool = False
    use_support_bounce: bool = False
    use_ma5_bounce: bool = False
    description: str = ""


VARIANTS = [
    BuyPointVariant(
        name="买点A: 只有企稳",
        short_name="A_stabilization",
        use_stabilization=True,
        description="收盘价站稳在5日线和10日线上方"
    ),
    BuyPointVariant(
        name="买点B: 只有回踩支撑",
        short_name="B_support_bounce",
        use_support_bounce=True,
        description="回踩20日低点支撑"
    ),
    BuyPointVariant(
        name="买点C: 只有强势回踩5日线",
        short_name="C_ma5_bounce",
        use_ma5_bounce=True,
        description="强势股回踩5日线"
    ),
    BuyPointVariant(
        name="买点D: 任意买点（当前逻辑）",
        short_name="D_any",
        use_stabilization=True,
        use_support_bounce=True,
        use_ma5_bounce=True,
        description="三种买点任一触发"
    ),
]


# ===================== 数据加载 =====================

def load_v36_data(json_path: str = None) -> pd.DataFrame:
    """加载V36股票池数据"""
    if json_path is None:
        # Navigate from experiments/v36_buy_point_verification.py
        # up to project root: experiments -> quant_trading -> 量化之神
        exp_dir = os.path.dirname(__file__)  # experiments
        quant_dir = os.path.dirname(exp_dir)  # quant_trading
        project_dir = os.path.dirname(quant_dir)  # 量化之神 (project root)
        json_path = os.path.join(project_dir, 'data', 'ashare', 'all_stocks_data.json')
        json_path = os.path.normpath(json_path)

    print(f"加载数据: {json_path}")
    df = load_ashare_json(json_path)

    # 只保留V36股票池中的股票
    codes = list(DEFAULT_STOCK_POOL.keys())
    df = df[df['code'].isin(codes)]

    print(f"股票数量: {df['code'].nunique()}")
    print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

    return df


# ===================== 信号生成 =====================

def generate_signals_for_variant(
    df: pd.DataFrame,
    variant: BuyPointVariant,
    params: V36Params
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为指定买点变体生成entry和exit信号

    Returns:
        (entry_long, exit_long) - boolean arrays
    """
    # 计算指标
    df = calculate_advanced_factors(df.copy(), params)

    n = len(df)
    entry_long = np.zeros(n, dtype=bool)
    exit_long = np.zeros(n, dtype=bool)

    for i in range(n):
        if i < 2:
            continue

        d = df.iloc[i]
        prev_d = df.iloc[i - 1]
        prev2_d = df.iloc[i - 2]

        # 检查风险标志
        if d.get('Risk_Flag', 0) == 1:
            continue

        # 检查趋势（ma20向上）
        if d['ma20'] < prev_d['ma20']:
            continue

        # 检查买点条件
        has_signal = False
        signal_type = ""

        if variant.use_stabilization and is_stabilization(d):
            has_signal = True
            signal_type = "企稳"

        if variant.use_support_bounce and is_support_bounce(d, prev_d):
            has_signal = True
            signal_type = "回踩支撑"

        if variant.use_ma5_bounce and is_strong_ma5_buy(d, prev_d, prev2_d):
            has_signal = True
            signal_type = "强势回踩5日线"

        if has_signal:
            entry_long[i] = True

        # 检查持仓是否需要卖出
        # 使用止损/止盈逻辑
        # 这里简化处理：使用exit_long作为止损止盈信号
        if d.get('break_ma20', False):
            exit_long[i] = True
        elif d.get('close', 0) < d.get('ma10', 0) * 0.98:
            exit_long[i] = True

    return entry_long, exit_long


# ===================== 回测运行器 =====================

def run_variant_backtest(
    df: pd.DataFrame,
    variant: BuyPointVariant,
    params: V36Params = None
) -> Dict:
    """
    对单个买点变体运行回测

    使用V36Backtester的完整止盈止损逻辑
    """
    if params is None:
        params = V36Params()

    # 为当前变体创建自定义参数
    test_params = V36Params(
        cash=1000000.0,
        max_pos=1,
        stop=-0.07,
        take=0.20,
        time_stop=6,
        slippage=0.001,
    )

    # 创建简化的买点检测函数
    def check_buy(d, prev_d, prev2_d):
        """根据变体检查买点"""
        has_signal = False
        signal_type = ""

        if variant.use_stabilization and is_stabilization(d):
            has_signal = True
            signal_type = "stabilization"

        if variant.use_support_bounce and is_support_bounce(d, prev_d):
            has_signal = True
            signal_type = "support_bounce"

        if variant.use_ma5_bounce and is_strong_ma5_buy(d, prev_d, prev2_d):
            has_signal = True
            signal_type = "ma5_bounce"

        return has_signal, signal_type

    # 手动回测（简化版）
    df = df.copy().reset_index(drop=True)
    df = calculate_advanced_factors(df, test_params)

    trades = []
    pos = None
    buy_date = None
    buy_price = None
    entry_bar = 0

    for i in range(len(df)):
        if i < 2:
            continue

        d = df.iloc[i]
        prev_d = df.iloc[i-1]
        prev2_d = df.iloc[i-2]

        # 风险过滤
        if d.get('Risk_Flag', 0) == 1:
            continue

        # 趋势过滤
        if d['ma20'] < prev_d['ma20']:
            continue

        # 买入逻辑
        if pos is None:
            has_signal, signal_type = check_buy(d, prev_d, prev2_d)
            if has_signal:
                cost = d['close'] * (1 + test_params.slippage)
                pos = {'cost': cost, 'entry_bar': i}
                buy_date = d.name if hasattr(d, 'name') else i
                trades.append({
                    'entry_bar': i,
                    'entry_price': cost,
                    'signal_type': signal_type,
                    'variant': variant.short_name,
                })

        # 卖出逻辑
        elif pos is not None:
            ret = (d['close'] - pos['cost']) / pos['cost']
            bars_held = i - pos['entry_bar']
            sell = False
            reason = ""

            # 止损
            if ret <= test_params.stop:
                sell = True
                reason = "stop_loss"
            # 止盈
            elif ret >= test_params.take:
                sell = True
                reason = "take_profit"
            # 时间止损
            elif bars_held >= test_params.time_stop:
                sell = True
                reason = "time_stop"
            # 强制止损（破10日线）
            elif d['close'] < d['ma10'] * 0.98:
                sell = True
                reason = "force_sell"
            # 跌破20日线
            elif d.get('break_ma20', False):
                sell = True
                reason = "break_ma20"

            if sell:
                sell_price = d['close'] * (1 - test_params.slippage)
                trades[-1].update({
                    'exit_bar': i,
                    'exit_price': sell_price,
                    'pnl': (sell_price - pos['cost']) * 100,  # 简化：每手100股
                    'return': ret,
                    'reason': reason,
                    'bars_held': bars_held,
                })
                pos = None

    # 返回结果
    if trades:
        trades_df = pd.DataFrame(trades)
        return {
            'trades': trades_df,
            'total_return_pct': trades_df['return'].sum() * 100 if 'return' in trades_df.columns else 0.0,
            'num_trades': len(trades_df),
        }
    else:
        return {
            'trades': pd.DataFrame(),
            'total_return_pct': 0.0,
            'num_trades': 0,
        }


def calculate_metrics(result) -> Dict:
    """计算回测指标"""
    # result is a dict with 'trades' (DataFrame) and 'total_return_pct'
    trades_df = result.get('trades', pd.DataFrame())
    total_return = result.get('total_return_pct', 0.0)

    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_loss_ratio': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
        }

    # 计算盈亏
    pnls = trades_df['pnl'].values
    winning_trades = pnls[pnls > 0]
    losing_trades = pnls[pnls <= 0]

    avg_profit = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
    avg_loss = np.abs(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.0

    # 胜率
    win_rate = len(winning_trades) / len(pnls) if len(pnls) > 0 else 0.0

    # 盈亏比
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0.0

    # 总收益率（从backtest result获取）
    _total_return = 0.0
    if hasattr(result, 'total_return_pct'):
        _total_return = result.total_return_pct
    elif isinstance(result, dict):
        _total_return = result.get('total_return_pct', 0.0)
    total_return = _total_return

    # 计算夏普比率（简化版）
    if len(pnls) > 1:
        returns = pnls / 1000000.0  # 假设初始资金100万
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # 最大回撤（简化版）
    equity = np.cumsum(np.concatenate([[1000000], pnls]))
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = np.min(drawdown) * 100.0 if len(drawdown) > 0 else 0.0

    return {
        'total_trades': len(pnls),
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
    }


# ===================== 主实验 =====================

def run_verification_experiment():
    """运行买点验证实验"""
    print("=" * 70)
    print("V36 买点假设验证实验")
    print("=" * 70)

    # 加载数据
    df = load_v36_data()
    params = V36Params()

    # 准备数据
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # 按股票分别回测，然后汇总
    results_by_variant: Dict[str, List[Dict]] = {v.short_name: [] for v in VARIANTS}

    # 获取所有股票
    codes = df['code'].unique()
    print(f"\n开始回测 {len(codes)} 只股票...")

    for code in codes:
        stock_df = df[df['code'] == code].copy()
        if len(stock_df) < 60:
            continue

        stock_df = stock_df.reset_index(drop=True)

        for variant in VARIANTS:
            try:
                result = run_variant_backtest(stock_df, variant, params)
                metrics = calculate_metrics(result)
                metrics['symbol'] = code
                results_by_variant[variant.short_name].append(metrics)
            except Exception as e:
                print(f"  股票 {code} 变体 {variant.short_name} 回测失败: {e}")

        if len(results_by_variant[VARIANTS[0].short_name]) % 5 == 0:
            print(f"  已完成 {len(results_by_variant[VARIANTS[0].short_name])} 只股票")

    # 汇总结果
    print("\n" + "=" * 70)
    print("买点假设验证结果汇总")
    print("=" * 70)

    summary = []
    for variant in VARIANTS:
        variant_results = results_by_variant[variant.short_name]
        if not variant_results:
            continue

        # 合并所有股票的结果
        all_metrics = {}
        for key in ['total_trades', 'win_rate', 'total_return', 'max_drawdown']:
            values = [r[key] for r in variant_results if r[key] != 0]
            all_metrics[key] = np.mean(values) if values else 0.0

        # 盈亏比计算
        all_pnls = []
        for r in variant_results:
            if 'avg_profit' in r and 'avg_loss' in r:
                if r['avg_profit'] > 0:
                    all_pnls.append(r['avg_profit'] / r['avg_loss'] if r['avg_loss'] > 0 else 0.0)
        all_metrics['profit_loss_ratio'] = np.mean(all_pnls) if all_pnls else 0.0

        summary.append({
            'variant': variant.name,
            'description': variant.description,
            'total_trades': int(all_metrics['total_trades']),
            'win_rate': all_metrics['win_rate'],
            'profit_loss_ratio': all_metrics['profit_loss_ratio'],
            'total_return': all_metrics['total_return'],
            'max_drawdown': all_metrics['max_drawdown'],
        })

        print(f"\n{variant.name}")
        print(f"  描述: {variant.description}")
        print(f"  总交易次数: {int(all_metrics['total_trades'])}")
        print(f"  胜率: {all_metrics['win_rate']:.1%}")
        print(f"  盈亏比: {all_metrics['profit_loss_ratio']:.2f}")
        print(f"  总收益率: {all_metrics['total_return']:.2f}%")
        print(f"  最大回撤: {all_metrics['max_drawdown']:.2f}%")

    # 比较分析
    print("\n" + "=" * 70)
    print("买点比较分析")
    print("=" * 70)

    # 按胜率排序
    sorted_summary = sorted(summary, key=lambda x: x['win_rate'], reverse=True)

    print("\n按胜率排序:")
    for i, s in enumerate(sorted_summary, 1):
        print(f"  {i}. {s['variant']}: 胜率={s['win_rate']:.1%}, 盈亏比={s['profit_loss_ratio']:.2f}")

    # 找出最佳买点
    best = sorted_summary[0]
    print(f"\n最佳买点: {best['variant']}")
    print(f"  理由: 胜率 {best['win_rate']:.1%}")

    # 分析结论
    print("\n" + "=" * 70)
    print("买点假设验证结论")
    print("=" * 70)

    for s in summary:
        if s['win_rate'] < 0.4:
            conclusion = "拖累策略，建议降低权重或移除"
        elif s['win_rate'] > 0.6:
            conclusion = "显著有效，建议增强触发条件"
        else:
            conclusion = "效果一般，可保留作为辅助条件"
        print(f"  {s['variant']}: {conclusion}")

    return summary


# ===================== 独立买点测试 =====================

def test_single_buy_point():
    """测试单个买点类型是否工作正常"""
    print("\n" + "=" * 70)
    print("买点类型独立测试")
    print("=" * 70)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    # 模拟上涨趋势
    base_price = 100.0
    prices = [base_price]
    for _ in range(99):
        prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.02)))

    df = pd.DataFrame({
        'date': dates,
        'code': 'TEST',
        'open': [p * 0.99 for p in prices],
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000000] * 100,
    })

    params = V36Params()

    for variant in VARIANTS:
        entry, exit_signal = generate_signals_for_variant(df, variant, params)
        signal_count = np.sum(entry)
        print(f"  {variant.short_name}: {signal_count} 个买入信号")


# ===================== 入口 =====================

if __name__ == '__main__':
    # 先测试单个买点
    test_single_buy_point()

    # 运行完整实验
    summary = run_verification_experiment()

    print("\n实验完成!")
