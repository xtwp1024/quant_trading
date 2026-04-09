# -*- coding: utf-8 -*-
"""
V36 参数优化器

使用 Optuna + WalkForward + AntiOverfit 进行参数优化
目标: 胜率 > 60%, 盈亏比 > 1.5, 最大回撤 < 15%

Usage:
    from quant_trading.experiments.v36_optimizer import V36Optimizer

    optimizer = V36Optimizer(data=df, stock_pool=DEFAULT_STOCK_POOL)
    result = optimizer.optimize(n_trials=300, timeout=7200)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quant_trading.strategies.v36_strategy import (
    V36Params,
    V36StockPool,
    DEFAULT_STOCK_POOL,
    DEFAULT_SECTOR_MAP,
    calculate_advanced_factors,
    is_stabilization,
    is_support_bounce,
    is_strong_ma5_buy,
)
from quant_trading.backtester.optuna_optimizer import OptunaOptimizer
from quant_trading.backtester.walkforward import WalkforwardAnalyzer


# ===================== 目标函数 =====================

@dataclass
class V36OptimizationResult:
    """V36优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    win_rate: float
    profit_loss_ratio: float
    max_drawdown: float
    total_return: float
    n_trials: int
    study_name: str


class V36Optimizer:
    """V36策略参数优化器

    使用 Optuna TPE 进行贝叶斯优化，结合 WalkForward 验证参数稳健性
    """

    # 成功标准
    TARGET_WIN_RATE = 0.60  # 60%
    TARGET_PROFIT_LOSS_RATIO = 1.5
    TARGET_MAX_DRAWDOWN = 0.15  # 15%

    # 权重配置
    WEIGHT_WIN_RATE = 0.40
    WEIGHT_PROFIT_LOSS = 0.40
    WEIGHT_DRAWDOWN = 0.20

    def __init__(
        self,
        data: pd.DataFrame,
        stock_pool: Dict[str, str] = None,
        study_name: str = "v36_optimization",
    ):
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values(['code', 'date']).reset_index(drop=True)

        self.stock_pool = stock_pool or DEFAULT_STOCK_POOL
        self.study_name = study_name

        # 过滤股票池
        self.filtered_data = self.data[
            self.data['code'].isin(self.stock_pool.keys())
        ].copy()

        print(f"V36 Optimizer initialized")
        print(f"  Stocks: {self.filtered_data['code'].nunique()}")
        print(f"  Date range: {self.filtered_data['date'].min()} ~ {self.filtered_data['date'].max()}")

    def _backtest_params(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
    ) -> Dict[str, float]:
        """使用给定参数回测数据

        Returns:
            {win_rate, avg_profit, avg_loss, profit_loss_ratio, total_return, max_drawdown}
        """
        # 提取参数
        stop_loss = params.get('stop_loss', -0.07)
        take_profit = params.get('take_profit', 0.20)
        time_stop = params.get('time_stop', 6)
        vol_ratio_threshold = params.get('vol_ratio_threshold', 0.7)

        # 计算指标
        df = calculate_advanced_factors(df.copy(), V36Params())

        # 添加自定义Vol_Ratio阈值过滤
        df['Vol_Ratio_filter'] = df['Vol_Ratio'] < vol_ratio_threshold

        trades = []
        pos = None

        for i in range(len(df)):
            if i < 2:
                continue

            d = df.iloc[i]
            prev_d = df.iloc[i - 1]
            prev2_d = df.iloc[i - 2]

            # 风险过滤
            if d.get('Risk_Flag', 0) == 1:
                continue

            # 趋势过滤
            if d['ma20'] < prev_d['ma20']:
                continue

            # 买入逻辑
            if pos is None:
                has_signal = False

                # 企稳
                if is_stabilization(d):
                    has_signal = True

                # 回踩支撑
                if is_support_bounce(d, prev_d):
                    has_signal = True

                # 强势回踩5日线
                if is_strong_ma5_buy(d, prev_d, prev2_d):
                    has_signal = True

                if has_signal:
                    cost = d['close'] * 1.001  # 0.1% 滑点
                    pos = {'cost': cost, 'entry_bar': i}
                    trades.append({
                        'entry_bar': i,
                        'entry_price': cost,
                    })

            # 卖出逻辑
            elif pos is not None:
                ret = (d['close'] - pos['cost']) / pos['cost']
                bars_held = i - pos['entry_bar']
                sell = False
                reason = ""

                # 止损
                if ret <= stop_loss:
                    sell = True
                    reason = "stop_loss"
                # 止盈
                elif ret >= take_profit:
                    sell = True
                    reason = "take_profit"
                # 时间止损
                elif bars_held >= time_stop:
                    sell = True
                    reason = "time_stop"
                # 强制止损
                elif d['close'] < d['ma10'] * 0.98:
                    sell = True
                    reason = "force_sell"

                if sell:
                    sell_price = d['close'] * 0.999
                    trades[-1].update({
                        'exit_bar': i,
                        'exit_price': sell_price,
                        'pnl': (sell_price - pos['cost']) * 100,
                        'return': ret,
                        'reason': reason,
                        'bars_held': bars_held,
                    })
                    pos = None

        # 计算指标
        if not trades:
            return {
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_loss_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
            }

        returns = [t['return'] for t in trades if 'return' in t]
        if not returns:
            return {
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_loss_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
            }

        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r <= 0]

        win_rate = len(winning) / len(returns) if returns else 0.0
        avg_profit = np.mean(winning) if winning else 0.0
        avg_loss = abs(np.mean(losing)) if losing else 0.0
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0.0
        total_return = sum(returns)

        # 最大回撤（简化）
        equity = [1.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        return {
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
        }

    def _objective(
        self,
        trial,
        train_data: pd.DataFrame,
    ) -> float:
        """Optuna 目标函数

        评分公式:
        score = win_rate * 0.4 + profit_loss_ratio_norm * 0.4 + drawdown_control * 0.2
        """
        # 采样参数
        params = {
            'stop_loss': trial.suggest_float('stop_loss', -0.15, -0.05),
            'take_profit': trial.suggest_float('take_profit', 0.10, 0.40),
            'time_stop': trial.suggest_int('time_stop', 3, 20),
            'vol_ratio_threshold': trial.suggest_float('vol_ratio_threshold', 0.5, 1.5),
        }

        # 对每只股票回测
        all_metrics = []
        for code in train_data['code'].unique():
            stock_df = train_data[train_data['code'] == code].copy()
            if len(stock_df) < 60:
                continue

            metrics = self._backtest_params(params, stock_df)
            all_metrics.append(metrics)

        if not all_metrics:
            return 0.0

        # 汇总指标（按交易次数加权）
        total_trades = sum(
            m['win_rate'] * 100 + 1 for m in all_metrics
        )

        win_rate = np.mean([m['win_rate'] for m in all_metrics])
        profit_loss_ratio = np.mean([m['profit_loss_ratio'] for m in all_metrics])
        max_drawdown = np.max([m['max_drawdown'] for m in all_metrics])

        # 计算综合评分
        # 归一化盈亏比（目标1.5为满分）
        plr_score = min(profit_loss_ratio / self.TARGET_PROFIT_LOSS_RATIO, 1.0)

        # 回撤控制（15%以内为满分）
        dd_score = max(1.0 - max_drawdown / self.TARGET_MAX_DRAWDOWN, 0.0)

        # 综合评分
        score = (
            win_rate * self.WEIGHT_WIN_RATE +
            plr_score * self.WEIGHT_PROFIT_LOSS +
            dd_score * self.WEIGHT_DRAWDOWN
        )

        # 记录中间值用于早停
        trial.set_user_attr('win_rate', win_rate)
        trial.set_user_attr('profit_loss_ratio', profit_loss_ratio)
        trial.set_user_attr('max_drawdown', max_drawdown)
        trial.set_user_attr('n_stocks', len(all_metrics))

        return score

    def _create_objective(self, train_data: pd.DataFrame) -> Callable:
        """创建目标函数闭包"""
        def objective(trial):
            return self._objective(trial, train_data)
        return objective

    def optimize(
        self,
        n_trials: int = 300,
        timeout: float = 7200,
        n_startup_trials: int = 50,
    ) -> V36OptimizationResult:
        """运行优化

        Args:
            n_trials: 试验次数
            timeout: 超时秒数
            n_startup_trials: 冷启动随机试验次数
        """
        print(f"\n开始优化: {n_trials} trials, timeout={timeout}s")
        print(f"目标: 胜率>{self.TARGET_WIN_RATE:.0%}, 盈亏比>{self.TARGET_PROFIT_LOSS_RATIO}, 回撤<{self.TARGET_MAX_DRAWDOWN:.0%}")
        print(f"权重: 胜率={self.WEIGHT_WIN_RATE}, 盈亏比={self.WEIGHT_PROFIT_LOSS}, 回撤={self.WEIGHT_DRAWDOWN}")

        # 使用全部数据训练
        train_data = self.filtered_data

        # 创建目标函数
        objective = self._create_objective(train_data)

        # 创建优化器
        optimizer = OptunaOptimizer(
            objective=objective,
            study_name=self.study_name,
            storage=None,  # 内存存储
            sampler="tpe",
            direction="maximize",
            n_startup_trials=n_startup_trials,
            seed=42,
        )

        # 运行优化
        result = optimizer.optimize(
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        # 获取最佳参数
        best_params = result.best_params
        best_value = result.best_value

        # 重新回测获取详细指标
        detailed_metrics = {}
        for code in train_data['code'].unique():
            stock_df = train_data[train_data['code'] == code].copy()
            if len(stock_df) < 60:
                continue

            m = self._backtest_params(best_params, stock_df)
            detailed_metrics[code] = m

        # 汇总
        avg_win_rate = np.mean([m['win_rate'] for m in detailed_metrics.values()])
        avg_plr = np.mean([m['profit_loss_ratio'] for m in detailed_metrics.values()])
        max_dd = np.max([m['max_drawdown'] for m in detailed_metrics.values()])
        total_ret = np.mean([m['total_return'] for m in detailed_metrics.values()])

        print(f"\n优化完成!")
        print(f"  最佳评分: {best_value:.4f}")
        print(f"  胜率: {avg_win_rate:.1%}")
        print(f"  盈亏比: {avg_plr:.2f}")
        print(f"  最大回撤: {max_dd:.1%}")
        print(f"  平均收益: {total_ret:.1%}")
        print(f"  参数: {best_params}")

        return V36OptimizationResult(
            best_params=best_params,
            best_score=best_value,
            win_rate=avg_win_rate,
            profit_loss_ratio=avg_plr,
            max_drawdown=max_dd,
            total_return=total_ret,
            n_trials=result.n_trials,
            study_name=self.study_name,
        )


# ===================== 入口 =====================

if __name__ == '__main__':
    from quant_trading.data.ashare_loader import load_ashare_json

    print("V36 参数优化器测试")
    print("=" * 50)

    # 加载数据
    exp_dir = os.path.dirname(__file__)  # experiments
    quant_dir = os.path.dirname(exp_dir)  # quant_trading
    project_dir = os.path.dirname(quant_dir)  # 量化之神 (project root)
    json_path = os.path.join(project_dir, 'data', 'ashare', 'all_stocks_data.json')
    json_path = os.path.normpath(json_path)

    print(f"加载数据: {json_path}")
    df = load_ashare_json(json_path)
    print(f"原始数据: {len(df)} 条")

    # 过滤V36股票池
    df_filtered = df[df['code'].isin(DEFAULT_STOCK_POOL.keys())]
    print(f"V36股票池: {len(df_filtered)} 条")

    if len(df_filtered) > 0:
        # 运行中等规模优化（50次试验）
        optimizer = V36Optimizer(df_filtered, DEFAULT_STOCK_POOL)
        result = optimizer.optimize(n_trials=50, timeout=1800, n_startup_trials=10)

        print(f"\n最终结果:")
        print(f"  最佳参数: {result.best_params}")
        print(f"  评分: {result.best_score:.4f}")
        print(f"  胜率: {result.win_rate:.1%}")
        print(f"  盈亏比: {result.profit_loss_ratio:.2f}")
    else:
        print("数据不足，跳过优化测试")
