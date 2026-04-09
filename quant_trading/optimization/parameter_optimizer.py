"""
参数优化器 - 网格搜索、贝叶斯优化
=================================

自动寻找最优策略参数
"""
import asyncio
import sys
import os
import yaml
import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Callable
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.database import DatabaseManager
from ..core.logger import setup_logger, logger
from optimization.strategy_selector import StrategySelector


class ParameterOptimizer:
    """参数优化器"""

    def __init__(self, config: Dict):
        self.config = config
        self.results = []

    async def grid_search(
        self,
        strategy_class: type,
        param_grid: Dict[str, List],
        symbol: str = "ETH-USDT-SWAP",
        days: int = 30,
        metric: str = 'ROI'
    ) -> pd.DataFrame:
        """
        网格搜索 - 遍历所有参数组合

        Args:
            strategy_class: 策略类
            param_grid: 参数网格 {param_name: [values]}
            symbol: 交易对
            days: 回测天数
            metric: 优化指标 ('ROI', 'Sharpe', 'WinRate')

        Returns:
            结果DataFrame
        """
        from optimization.backtest_runner import BacktestRunner

        logger.info(f"🔍 开始网格搜索...")
        logger.info(f"   策略: {strategy_class.__name__}")
        logger.info(f"   参数组合数: {self._count_combinations(param_grid)}")

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        logger.info(f"   总计: {len(all_combinations)} 种组合")

        results = []

        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))

            logger.info(f"\n[{i+1}/{len(all_combinations)}] 测试参数: {params}")

            try:
                # 回测
                result = await self._backtest_with_params(
                    strategy_class, params, symbol, days
                )

                result['params'] = params
                results.append(result)

                logger.info(f"   ROI: {result['ROI']:.2f}%, "
                          f"Sharpe: {result['Sharpe']:.2f}")

            except Exception as e:
                logger.error(f"   ❌ 错误: {e}")
                continue

        # 转换为DataFrame
        df = pd.DataFrame(results)

        # 添加参数列
        param_df = pd.DataFrame(df['params'].tolist())
        df = pd.concat([df.drop('params', axis=1), param_df], axis=1)

        # 按指标排序
        df = df.sort_values(metric, ascending=False)

        logger.info(f"\n✅ 网格搜索完成！")
        logger.info(f"   最佳参数: {df.iloc[0][param_names].to_dict()}")
        logger.info(f"   最佳{metric}: {df.iloc[0][metric]:.2f}")

        return df

    async def random_search(
        self,
        strategy_class: type,
        param_bounds: Dict[str, Tuple],
        n_iterations: int = 50,
        symbol: str = "ETH-USDT-SWAP",
        days: int = 30,
        metric: str = 'ROI'
    ) -> pd.DataFrame:
        """
        随机搜索 - 在参数空间内随机采样

        Args:
            strategy_class: 策略类
            param_bounds: 参数边界 {param: (min, max)}
            n_iterations: 迭代次数
            symbol: 交易对
            days: 回测天数
            metric: 优化指标

        Returns:
            结果DataFrame
        """
        logger.info(f"🎲 开始随机搜索...")
        logger.info(f"   策略: {strategy_class.__name__}")
        logger.info(f"   迭代次数: {n_iterations}")

        results = []
        best_score = -np.inf
        best_params = None

        for i in range(n_iterations):
            # 随机采样参数
            params = {}
            for param, (min_val, max_val) in param_bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param] = np.random.uniform(min_val, max_val)

            logger.info(f"\n[{i+1}/{n_iterations}] 测试参数: {params}")

            try:
                result = await self._backtest_with_params(
                    strategy_class, params, symbol, days
                )

                result['params'] = params
                results.append(result)

                score = result[metric]
                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"   🎯 新最佳! {metric}: {score:.2f}")

            except Exception as e:
                logger.error(f"   ❌ 错误: {e}")
                continue

        df = pd.DataFrame(results)
        param_df = pd.DataFrame(df['params'].tolist())
        df = pd.concat([df.drop('params', axis=1), param_df], axis=1)
        df = df.sort_values(metric, ascending=False)

        logger.info(f"\n✅ 随机搜索完成！")
        logger.info(f"   最佳参数: {best_params}")
        logger.info(f"   最佳{metric}: {best_score:.2f}")

        return df

    async def bayesian_optimization(
        self,
        strategy_class: type,
        param_bounds: Dict[str, Tuple],
        n_iterations: int = 30,
        symbol: str = "ETH-USDT-SWAP",
        days: int = 30,
        metric: str = 'ROI'
    ) -> Tuple[Dict, float, pd.DataFrame]:
        """
        贝叶斯优化 - 使用高斯过程进行高效优化

        Args:
            strategy_class: 策略类
            param_bounds: 参数边界 {param: (min, max)}
            n_iterations: 迭代次数
            symbol: 交易对
            days: 回测天数
            metric: 优化指标

        Returns:
            (best_params, best_score, history_df)
        """
        logger.info(f"🤖 开始贝叶斯优化...")
        logger.info(f"   策略: {strategy_class.__name__}")
        logger.info(f"   迭代次数: {n_iterations}")

        param_names = list(param_bounds.keys())
        n_params = len(param_names)

        # 初始随机采样
        X_init = []
        y_init = []

        n_init = min(10, n_iterations // 3)
        logger.info(f"\n📍 初始采样 ({n_init}次)...")

        for _ in range(n_init):
            params = self._random_sample(param_bounds)
            score = await self._evaluate_params(
                strategy_class, params, symbol, days, metric
            )

            X_init.append([params[p] for p in param_names])
            y_init.append(score)

        X = np.array(X_init)
        y = np.array(y_init)

        # 高斯过程
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

        best_score = max(y)
        best_params = dict(zip(param_names, X[np.argmax(y)]))

        logger.info(f"   初始最佳{metric}: {best_score:.2f}")

        # 贝叶斯优化迭代
        for i in range(n_init, n_iterations):
            # 拟合GP
            gp.fit(X, y)

            # 采集函数（期望改进）
            def acquisition(x):
                mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
                return (mu - best_score) / (sigma + 1e-8)

            # 优化采集函数
            X_next = self._optimize_acquisition(
                acquisition, param_bounds, n_trials=100
            )

            # 评估新点
            params = dict(zip(param_names, X_next))
            score = await self._evaluate_params(
                strategy_class, params, symbol, days, metric
            )

            X = np.vstack([X, X_next])
            y = np.append(y, score)

            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"[{i+1}/{n_iterations}] 🎯 改进! {metric}: {score:.2f}")
            else:
                logger.info(f"[{i+1}/{n_iterations}] {metric}: {score:.2f}")

        logger.info(f"\n✅ 贝叶斯优化完成！")
        logger.info(f"   最佳参数: {best_params}")
        logger.info(f"   最佳{metric}: {best_score:.2f}")

        # 历史记录
        history = pd.DataFrame(X, columns=param_names)
        history['score'] = y

        return best_params, best_score, history

    def _random_sample(self, param_bounds: Dict) -> Dict:
        """随机采样参数"""
        params = {}
        for param, (min_val, max_val) in param_bounds.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param] = np.random.randint(min_val, max_val + 1)
            else:
                params[param] = np.random.uniform(min_val, max_val)
        return params

    async def _evaluate_params(
        self,
        strategy_class: type,
        params: Dict,
        symbol: str,
        days: int,
        metric: str
    ) -> float:
        """评估参数"""
        try:
            result = await self._backtest_with_params(
                strategy_class, params, symbol, days
            )
            return result[metric]
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"参数评估失败: {e}, params={params}")
            return -999.0

    async def _backtest_with_params(
        self,
        strategy_class: type,
        params: Dict,
        symbol: str,
        days: int
    ) -> Dict:
        """使用指定参数回测"""
        # TODO: 实现真正的回测引擎集成
        # 当前返回模拟数据，调用者应意识到这是占位符实现
        raise NotImplementedError(
            "_backtest_with_params 需要实现真正的回测引擎集成。"
            "当前返回模拟数据仅用于测试目的。"
        )

    def _optimize_acquisition(
        self,
        acquisition: Callable,
        param_bounds: Dict,
        n_trials: int = 100
    ) -> np.ndarray:
        """优化采集函数"""
        param_names = list(param_bounds.keys())

        best_x = None
        best_val = -np.inf

        for _ in range(n_trials):
            x = []
            for param in param_names:
                min_val, max_val = param_bounds[param]
                if isinstance(min_val, int):
                    x.append(np.random.randint(min_val, max_val + 1))
                else:
                    x.append(np.random.uniform(min_val, max_val))

            val = acquisition(np.array(x))
            if val > best_val:
                best_val = val
                best_x = x

        return np.array(best_x)

    def _count_combinations(self, param_grid: Dict) -> int:
        """计算参数组合数"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count

    def save_results(
        self,
        results: pd.DataFrame,
        strategy_name: str,
        method: str,
        output_dir: str = "optimization/results"
    ) -> str:
        """保存优化结果"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{strategy_name}_{method}_{timestamp}.csv"
        filepath = output_path / filename

        results.to_csv(filepath, index=False)
        logger.info(f"✅ 结果已保存: {filepath}")

        return str(filepath)


# 预定义参数网格
GRID_STRATEGIES = {
    'martin': {
        'grid_spacing': [0.01, 0.015, 0.02, 0.025],
        'grid_levels': [3, 5, 7, 10],
        'take_profit': [0.01, 0.015, 0.02, 0.03],
        'stop_loss': [0.03, 0.05, 0.08],
    },
    'trend': {
        'ema_fast': [7, 12, 20],
        'ema_slow': [25, 50, 100],
        'rsi_period': [14, 21, 28],
        'rsi_overbought': [65, 70, 75],
        'rsi_oversold': [25, 30, 35],
    },
    'grid': {
        'upper_range': [0.02, 0.03, 0.05],
        'lower_range': [0.02, 0.03, 0.05],
        'grid_levels': [5, 7, 10, 15],
    }
}

# 预定义参数边界
PARAM_BOUNDS = {
    'martin': {
        'grid_spacing': (0.005, 0.03),
        'grid_levels': (3, 15),
        'take_profit': (0.005, 0.05),
        'stop_loss': (0.02, 0.10),
    },
    'trend': {
        'ema_fast': (5, 25),
        'ema_slow': (20, 100),
        'rsi_period': (10, 30),
    }
}


async def main():
    """主函数 - 参数优化示例"""

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    optimizer = ParameterOptimizer(config)

    # 示例：网格搜索马丁策略参数
    from modules.strategies.martin_strategy import MartinStrategy

    print("\n" + "="*80)
    print("网格搜索 - 马丁策略参数优化")
    print("="*80 + "\n")

    results = await optimizer.grid_search(
        strategy_class=MartinStrategy,
        param_grid=GRID_STRATEGIES['martin'],
        symbol="ETH-USDT-SWAP",
        days=30,
        metric='ROI'
    )

    # 保存结果
    optimizer.save_results(results, 'MartinStrategy', 'grid_search')

    print("\nTop 10 参数组合:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[停止] 用户中断")
