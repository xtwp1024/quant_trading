# -*- coding: utf-8 -*-
"""
V36 Crypto 参数优化器

使用 Optuna 对Crypto市场进行V36策略参数优化

Usage:
    python v36_crypto_optimizer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quant_trading.strategy.advanced.v36_strategy import (
    V36Params,
    calculate_advanced_factors,
    is_stabilization,
    is_support_bounce,
    is_strong_ma5_buy,
)
from quant_trading.data.crypto_loader import CryptoDataLoader
from quant_trading.backtester.optuna_optimizer import OptunaOptimizer


@dataclass
class V36CryptoOptimizationResult:
    """V36 Crypto优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    win_rate: float
    profit_loss_ratio: float
    max_drawdown: float
    total_return: float
    n_trials: int
    study_name: str


class V36CryptoOptimizer:
    """V36策略Crypto参数优化器

    Crypto市场特性:
    - 更高波动性
    - 24/7交易
    - 更短持仓周期
    """

    # Crypto成功标准 (与A股不同)
    TARGET_WIN_RATE = 0.55  # 55%
    TARGET_PROFIT_LOSS_RATIO = 1.3
    TARGET_MAX_DRAWDOWN = 0.20  # 20%

    # 权重配置
    WEIGHT_WIN_RATE = 0.35
    WEIGHT_PROFIT_LOSS = 0.40
    WEIGHT_DRAWDOWN = 0.25

    # Crypto参数搜索范围
    STOP_LOSS_RANGE = (-0.10, -0.02)  # -2% ~ -10%
    TAKE_PROFIT_RANGE = (0.05, 0.25)   # 5% ~ 25%
    TIME_STOP_RANGE = (12, 72)          # 12~72小时
    VOL_RATIO_RANGE = (0.5, 1.5)

    def __init__(
        self,
        data: pd.DataFrame = None,
        symbols: List[str] = None,
        study_name: str = "v36_crypto_optimization",
    ):
        self.study_name = study_name

        if data is None:
            loader = CryptoDataLoader()
            self.data = loader.load()
        else:
            self.data = data.copy()

        if symbols:
            self.data = self.data[self.data['symbol'].isin(symbols)]

        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values(['symbol', 'date']).reset_index(drop=True)

        print(f"V36 Crypto Optimizer initialized")
        print(f"  Symbols: {self.data['symbol'].unique().tolist()}")
        print(f"  Date range: {self.data['date'].min()} ~ {self.data['date'].max()}")

    def _backtest_params(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
    ) -> Dict[str, float]:
        """使用给定参数回测数据"""
        stop_loss = params.get('stop_loss', -0.03)
        take_profit = params.get('take_profit', 0.10)
        time_stop = params.get('time_stop', 24)
        vol_ratio_threshold = params.get('vol_ratio_threshold', 0.7)

        # 计算指标
        df = calculate_advanced_factors(df.copy(), V36Params())

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

                if is_stabilization(d):
                    has_signal = True
                if is_support_bounce(d, prev_d):
                    has_signal = True
                if is_strong_ma5_buy(d, prev_d, prev2_d):
                    has_signal = True

                if has_signal:
                    cost = d['close'] * 1.002  # 0.2% 滑点 (Crypto更高)
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
                # 时间止损 (小时)
                elif bars_held >= time_stop:
                    sell = True
                    reason = "time_stop"
                # 强制止损
                elif d['close'] < d['ma10'] * 0.98:
                    sell = True
                    reason = "force_sell"

                if sell:
                    sell_price = d['close'] * 0.998
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
            return self._empty_metrics()

        returns = [t['return'] for t in trades if 'return' in t]
        if not returns:
            return self._empty_metrics()

        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r <= 0]

        win_rate = len(winning) / len(returns) if returns else 0.0
        avg_profit = np.mean(winning) if winning else 0.0
        avg_loss = abs(np.mean(losing)) if losing else 0.0
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0.0
        total_return = sum(returns)

        # 最大回撤
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

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_loss_ratio': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
        }

    def _objective(self, trial, train_data: pd.DataFrame) -> float:
        """Optuna 目标函数"""
        params = {
            'stop_loss': trial.suggest_float('stop_loss', *self.STOP_LOSS_RANGE),
            'take_profit': trial.suggest_float('take_profit', *self.TAKE_PROFIT_RANGE),
            'time_stop': trial.suggest_int('time_stop', *self.TIME_STOP_RANGE),
            'vol_ratio_threshold': trial.suggest_float('vol_ratio_threshold', *self.VOL_RATIO_RANGE),
        }

        all_metrics = []
        for symbol in train_data['symbol'].unique():
            symbol_df = train_data[train_data['symbol'] == symbol].copy()
            if len(symbol_df) < 30:
                continue

            metrics = self._backtest_params(params, symbol_df)
            all_metrics.append(metrics)

        if not all_metrics:
            return 0.0

        # 汇总指标
        win_rate = np.mean([m['win_rate'] for m in all_metrics])
        profit_loss_ratio = np.mean([m['profit_loss_ratio'] for m in all_metrics])
        max_drawdown = np.max([m['max_drawdown'] for m in all_metrics])

        # 评分
        plr_score = min(profit_loss_ratio / self.TARGET_PROFIT_LOSS_RATIO, 1.0)
        dd_score = max(1.0 - max_drawdown / self.TARGET_MAX_DRAWDOWN, 0.0)

        score = (
            win_rate * self.WEIGHT_WIN_RATE +
            plr_score * self.WEIGHT_PROFIT_LOSS +
            dd_score * self.WEIGHT_DRAWDOWN
        )

        trial.set_user_attr('win_rate', win_rate)
        trial.set_user_attr('profit_loss_ratio', profit_loss_ratio)
        trial.set_user_attr('max_drawdown', max_drawdown)

        return score

    def _create_objective(self, train_data: pd.DataFrame):
        def objective(trial):
            return self._objective(trial, train_data)
        return objective

    def optimize(
        self,
        n_trials: int = 50,
        timeout: float = 1800,
        n_startup_trials: int = 10,
    ) -> V36CryptoOptimizationResult:
        """运行优化"""
        print(f"\n开始 Crypto 优化: {n_trials} trials, timeout={timeout}s")
        print(f"目标: 胜率>{self.TARGET_WIN_RATE:.0%}, 盈亏比>{self.TARGET_PROFIT_LOSS_RATIO}, 回撤<{self.TARGET_MAX_DRAWDOWN:.0%}")

        train_data = self.data

        objective = self._create_objective(train_data)

        optimizer = OptunaOptimizer(
            objective=objective,
            study_name=self.study_name,
            storage=None,
            sampler="tpe",
            direction="maximize",
            n_startup_trials=n_startup_trials,
            seed=42,
        )

        result = optimizer.optimize(
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best_params = result.best_params
        best_value = result.best_value

        # 详细回测
        detailed_metrics = {}
        for symbol in train_data['symbol'].unique():
            symbol_df = train_data[train_data['symbol'] == symbol].copy()
            if len(symbol_df) < 30:
                continue
            m = self._backtest_params(best_params, symbol_df)
            detailed_metrics[symbol] = m

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

        return V36CryptoOptimizationResult(
            best_params=best_params,
            best_score=best_value,
            win_rate=avg_win_rate,
            profit_loss_ratio=avg_plr,
            max_drawdown=max_dd,
            total_return=total_ret,
            n_trials=result.n_trials,
            study_name=self.study_name,
        )


if __name__ == '__main__':
    print("V36 Crypto 参数优化器")
    print("=" * 50)

    # 加载数据
    loader = CryptoDataLoader()
    df = loader.load()
    print(f"加载数据: {len(df)} 条, {df['symbol'].nunique()} 个交易对")

    # 创建优化器
    optimizer = V36CryptoOptimizer(df)

    # 运行优化 (小规模测试)
    result = optimizer.optimize(n_trials=30, timeout=600, n_startup_trials=5)

    print(f"\n最终结果:")
    print(f"  最佳参数: {result.best_params}")
    print(f"  评分: {result.best_score:.4f}")
    print(f"  胜率: {result.win_rate:.1%}")
    print(f"  盈亏比: {result.profit_loss_ratio:.2f}")

    # 更新配置文件
    print("\n建议更新 quant_trading/config/v36_config.py 中的 V36_CRYPTO_OPTIMIZED")
    print(f"  stop_loss: {result.best_params['stop_loss']:.3f}")
    print(f"  take_profit: {result.best_params['take_profit']:.3f}")
    print(f"  time_stop: {result.best_params['time_stop']}")
    print(f"  vol_ratio_threshold: {result.best_params['vol_ratio_threshold']:.3f}")