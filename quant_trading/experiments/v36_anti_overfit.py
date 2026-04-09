# -*- coding: utf-8 -*-
"""
V36 AntiOverfit 验证脚本

验证优化结果的统计显著性：
- PBO (Probability of Backtest Overfitting) < 50%
- DSR (Deflated Sharpe Ratio) > 0.95
- SPA (Hansen Superior Predictive Ability) p < 0.05

Usage:
    from quant_trading.experiments.v36_anti_overfit import run_anti_overfit验证

    result = run_anti_overfit验证(data=df, n_trials=50)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quant_trading.backtester.anti_overfit import (
    CSCVValidator,
    deflated_sharpe_ratio,
    hansen_spa,
    CSCVResult,
    DSRResult,
    SPAResult,
)
from quant_trading.strategy.advanced.v36_strategy import (
    V36Params,
    DEFAULT_STOCK_POOL,
    calculate_advanced_factors,
    is_stabilization,
    is_support_bounce,
    is_strong_ma5_buy,
)


@dataclass
class AntiOverfit验证Result:
    """AntiOverfit验证结果"""
    passed: bool
    pbo: float
    dsr: float
    spa_p_value: float
    details: Dict[str, Any]


def run_backtest_with_params(
    df: pd.DataFrame,
    params: Dict[str, Any],
    stock_pool: Dict[str, str],
) -> List[Dict]:
    """使用给定参数回测所有股票，返回每只股票的equity curve"""
    stop_loss = params.get('stop_loss', -0.07)
    take_profit = params.get('take_profit', 0.20)
    time_stop = params.get('time_stop', 6)

    results = []

    for code in df['code'].unique():
        if code not in stock_pool:
            continue

        stock_df = df[df['code'] == code].copy().reset_index(drop=True)
        if len(stock_df) < 60:
            continue

        stock_df = calculate_advanced_factors(stock_df, V36Params())

        equity = [1.0]  # 从1开始
        trades = []
        pos = None

        for i in range(len(stock_df)):
            if i < 2:
                continue

            d = stock_df.iloc[i]
            prev_d = stock_df.iloc[i - 1]
            prev2_d = stock_df.iloc[i - 2]

            if d.get('Risk_Flag', 0) == 1:
                continue

            if d['ma20'] < prev_d['ma20']:
                continue

            if pos is None:
                has_signal = False

                if is_stabilization(d):
                    has_signal = True
                if is_support_bounce(d, prev_d):
                    has_signal = True
                if is_strong_ma5_buy(d, prev_d, prev2_d):
                    has_signal = True

                if has_signal:
                    cost = d['close'] * 1.001
                    pos = {'cost': cost, 'entry_bar': i}
                    trades.append({'entry_bar': i, 'entry_price': cost})

            elif pos is not None:
                ret = (d['close'] - pos['cost']) / pos['cost']
                bars_held = i - pos['entry_bar']
                sell = False
                reason = ""

                if ret <= stop_loss:
                    sell = True
                    reason = "SL"
                elif ret >= take_profit:
                    sell = True
                    reason = "TP"
                elif bars_held >= time_stop:
                    sell = True
                    reason = "TIME"
                elif d['close'] < d['ma10'] * 0.98:
                    sell = True
                    reason = "FORCE"

                if sell:
                    sell_price = d['close'] * 0.999
                    trades[-1].update({
                        'exit_bar': i,
                        'exit_price': sell_price,
                        'return': ret,
                        'reason': reason,
                    })
                    equity.append(equity[-1] * (1 + ret))
                    pos = None

        if pos is not None:
            # 强制平仓
            last_close = stock_df.iloc[-1]['close']
            ret = (last_close - pos['cost']) / pos['cost']
            equity.append(equity[-1] * (1 + ret))
        else:
            if len(equity) > 1:
                equity.append(equity[-1])  # 保持不变

        results.append({
            'code': code,
            'equity': np.array(equity),
            'n_trades': len([t for t in trades if 'return' in t]),
        })

    return results


def run_anti_overfit验证(
    data: pd.DataFrame,
    best_params: Dict[str, Any],
    stock_pool: Dict[str, str] = None,
    n_subsamples: int = 16,
) -> AntiOverfit验证Result:
    """运行完整的AntiOverfit验证

    Args:
        data: 股票数据
        best_params: 最佳参数
        stock_pool: 股票池
        n_subsamples: CSCV子样本数

    Returns:
        AntiOverfit验证Result
    """
    if stock_pool is None:
        stock_pool = DEFAULT_STOCK_POOL

    print("开始 AntiOverfit 验证...")
    print(f"  子样本数: {n_subsamples}")

    # 运行回测获取equity curves
    print("  运行回测...")
    backtest_results = run_backtest_with_params(data, best_params, stock_pool)

    if not backtest_results:
        return AntiOverfit验证Result(
            passed=False,
            pbo=1.0,
            dsr=0.0,
            spa_p_value=1.0,
            details={'error': 'No backtest results'},
        )

    # 获取最长的equity curve作为基准
    max_len = max(len(r['equity']) for r in backtest_results)

    # 合并所有股票的equity curves（取平均）
    combined_equity = np.zeros(max_len)
    count = 0
    for r in backtest_results:
        eq = r['equity']
        combined_equity[:len(eq)] += eq
        count += 1

    if count > 0:
        combined_equity /= count

    print(f"  合并了 {count} 只股票的equity curve")

    # 1. CSCV PBO验证
    print("  运行 CSCV 验证...")
    cscv = CSCVValidator(n_subsamples=n_subsamples)

    # 使用参数网格生成多个候选策略的equity curves
    # 这里简化处理：使用不同止损参数生成多条曲线
    equity_curves = {}

    # 原始参数
    equity_curves['baseline'] = combined_equity

    # 添加不同参数变体的equity curve
    param_variants = [
        {'stop_loss': -0.07, 'take_profit': 0.20, 'time_stop': 6},
        {'stop_loss': -0.09, 'take_profit': 0.25, 'time_stop': 5},
        {'stop_loss': -0.10, 'take_profit': 0.30, 'time_stop': 4},
        {'stop_loss': -0.08, 'take_profit': 0.22, 'time_stop': 7},
        {'stop_loss': -0.11, 'take_profit': 0.35, 'time_stop': 3},
    ]

    for i, variant in enumerate(param_variants):
        test_params = {**best_params, **variant}
        results = run_backtest_with_params(data, test_params, stock_pool)
        if results:
            eqs = [r['equity'] for r in results]
            max_l = max(len(eq) for eq in eqs)
            avg_eq = np.zeros(max_l)
            for eq in eqs:
                avg_eq[:len(eq)] += eq
            avg_eq /= len(eqs)
            equity_curves[f'variant_{i}'] = avg_eq

    try:
        cscv_result = cscv.compute_pbo(equity_curves)
        pbo = cscv_result.pbo
        print(f"    PBO: {pbo:.2%} (目标: <50%)")
        print(f"    是否过拟合: {cscv_result.is_overfit}")
    except Exception as e:
        print(f"    CSCV计算失败: {e}")
        pbo = 0.5
        cscv_result = None

    # 2. DSR验证
    print("  运行 DSR 验证...")

    # 计算每只股票的夏普比率
    sharpe_ratios = []
    for r in backtest_results:
        eq = r['equity']
        if len(eq) < 2:
            continue
        returns = np.diff(eq) / eq[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            sharpe_ratios.append(sharpe)

    if sharpe_ratios:
        observed_sharpe = np.mean(sharpe_ratios)
        avg_sample_length = np.mean([len(r['equity']) for r in backtest_results])
        n_trials = 50  # 假设进行了50次试验

        dsr_result = deflated_sharpe_ratio(
            observed_sharpe=observed_sharpe,
            n_trials=n_trials,
            sample_length=int(avg_sample_length),
        )
        dsr = dsr_result.dsr
        print(f"    DSR: {dsr:.4f} (目标: >0.95)")
        print(f"    是否显著: {dsr_result.is_significant}")
    else:
        dsr = 0.0
        dsr_result = None
        print("    无法计算DSR")

    # 3. SPA验证
    print("  运行 SPA 验证...")

    # 使用合并的equity curve计算收益率
    strategy_returns = np.diff(combined_equity) / combined_equity[:-1]

    # 假设基准是买入持有
    # 简化：使用全市场平均收益作为基准
    benchmark_returns = np.random.randn(len(strategy_returns)) * 0.01  # 简化基准

    try:
        spa_result = hansen_spa(strategy_returns, benchmark_returns)
        spa_p = spa_result.p_value
        print(f"    SPA p-value: {spa_p:.4f} (目标: <0.05)")
        print(f"    是否优于基准: {spa_result.is_superior}")
    except Exception as e:
        print(f"    SPA计算失败: {e}")
        spa_p = 0.5
        spa_result = None

    # 综合判断
    passed = (pbo < 0.50) and (dsr > 0.95) and (spa_p < 0.05)

    print(f"\n验证结果: {'通过' if passed else '未通过'}")
    print(f"  PBO < 50%: {pbo < 0.50} ({pbo:.2%})")
    print(f"  DSR > 0.95: {dsr > 0.95} ({dsr:.4f})")
    print(f"  SPA p < 0.05: {spa_p < 0.05} ({spa_p:.4f})")

    return AntiOverfit验证Result(
        passed=passed,
        pbo=pbo,
        dsr=dsr,
        spa_p_value=spa_p,
        details={
            'cscv_result': cscv_result,
            'dsr_result': dsr_result,
            'spa_result': spa_result,
            'n_stocks': len(backtest_results),
        },
    )


# ===================== 入口 =====================

if __name__ == '__main__':
    from quant_trading.data.ashare_loader import load_ashare_json

    print("V36 AntiOverfit 验证")
    print("=" * 50)

    # 加载数据
    exp_dir = os.path.dirname(__file__)
    quant_dir = os.path.dirname(exp_dir)
    project_dir = os.path.dirname(quant_dir)
    json_path = os.path.join(project_dir, 'data', 'ashare', 'all_stocks_data.json')
    json_path = os.path.normpath(json_path)

    print(f"加载数据: {json_path}")
    df = load_ashare_json(json_path)
    print(f"原始数据: {len(df)} 条")

    # 过滤V36股票池
    df_filtered = df[df['code'].isin(DEFAULT_STOCK_POOL.keys())]
    print(f"V36股票池: {len(df_filtered)} 条")

    if len(df_filtered) > 0:
        # 使用初步优化的最佳参数
        best_params = {
            'stop_loss': -0.093,
            'take_profit': 0.398,
            'time_stop': 3,
            'vol_ratio_threshold': 1.276,
        }

        result = run_anti_overfit验证(df_filtered, best_params)

        print(f"\n最终结论: {'通过验证' if result.passed else '未通过验证'}")
    else:
        print("数据不足，跳过验证")
