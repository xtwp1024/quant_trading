"""
Spot-Futures Arbitrage Core Framework
====================================

期现套利核心框架，包含:
- SpotPortfolioConstruction: 现货组合构建 (复制ETF/成分股)
- FuturesPricingArbitrageIntervals: 非套利区间计算
- SpotFuturesArbitrageur: 综合套利器

基于携带成本定价模型 (Cost-of-Carry Model):
F = S * e^(r-q)T

其中:
- F: 期货价格
- S: 现货价格
- r: 无风险利率
- q: 股息收益率
- T: 到期时间
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Literal


class SpotPortfolioConstruction:
    """
    现货组合构建器

    使用成分股复制方法或ETF构建方法来构造现货组合,
    追踪期货趋势并获得更高收益。

    支持的复制方法:
    - full: 全复制，每个成分股权重相等
    - sampling: 采样复制，随机选取一半成分股
    - hierarchical: 分层复制 (待实现)
    """

    def __init__(self, underlying_index_df: pd.DataFrame):
        """
        Args:
            underlying_index_df: 底层指数成分股权重数据
                                  Index: 成分股代码, Columns: 权重或其他属性
        """
        self.underlying_index_df = underlying_index_df

    def replication_method(self, replication_type: str = 'full') -> np.ndarray:
        """
        生成现货组合复制权重

        Args:
            replication_type: 'full', 'sampling', 或 'hierarchical'

        Returns:
            weights: 成分股权重数组
        """
        n = len(self.underlying_index_df)

        if replication_type == 'full':
            # 全复制: 等权重
            weights = np.ones(n) / n
            return weights

        elif replication_type == 'sampling':
            # 采样复制: 随机选取一半成分股
            sampled_indices = np.random.choice(n, n // 2, replace=False)
            weights = np.zeros(n)
            weights[sampled_indices] = 1 / len(sampled_indices)
            return weights

        elif replication_type == 'hierarchical':
            # 分层复制: 按行业或市值分层抽样
            raise NotImplementedError("Hierarchical replication not yet implemented")

        else:
            raise ValueError(f"Unknown replication_type: {replication_type}")

    def etf_construction(self, etf_name: str) -> np.ndarray:
        """
        基于ETF构建方法构造现货组合 (待实现)

        Args:
            etf_name: ETF名称或代码

        Returns:
            weights: 成分股权重数组
        """
        raise NotImplementedError("ETF construction not yet implemented")

    def calculate_tracking_error(
        self,
        portfolio_returns: pd.Series,
        index_returns: pd.Series
    ) -> float:
        """
        计算跟踪误差

        Args:
            portfolio_returns: 组合收益率序列
            index_returns: 指数收益率序列

        Returns:
            tracking_error: 年化跟踪误差
        """
        diff = portfolio_returns - index_returns
        tracking_error = np.std(diff) * np.sqrt(252)  # 年化
        return tracking_error


class FuturesPricingArbitrageIntervals:
    """
    期货定价非套利区间分析器

    基于携带成本定价模型计算期货非套利区间,
    当期货价格偏离区间时存在套利机会。

    区间边界计算: mean ± 2*std (基于历史carry cost)
    """

    def __init__(self, futures_price_df: pd.DataFrame, spot_price_df: pd.DataFrame):
        """
        Args:
            futures_price_df: 期货价格数据, 必须有 'price' 列
            spot_price_df: 现货价格数据, 必须有 'price' 列
        """
        self.futures_price_df = futures_price_df
        self.spot_price_df = spot_price_df

        # 预处理列名 (兼容 'Spot'/'Futures' 或 'price')
        if 'price' not in futures_price_df.columns and 'Futures' in futures_price_df.columns:
            self.futures_price_df = futures_price_df.rename(columns={'Futures': 'price'})
        if 'price' not in spot_price_df.columns and 'Spot' in spot_price_df.columns:
            self.spot_price_df = spot_price_df.rename(columns={'Spot': 'price'})

    def calculate_carry_cost(self) -> pd.Series:
        """
        计算持有成本 (carry cost)

        F - S 的差值代表持有成本

        Returns:
            carry_cost: 持有成本序列
        """
        return self.futures_price_df['price'] - self.spot_price_df['price']

    def calculate_arbitrage_interval(
        self,
        n_std: float = 2.0
    ) -> Tuple[float, float]:
        """
        计算非套利区间

        基于携带成本定价模型, 计算均值和标准差,
        设定上下界为均值 ± n_std * 标准差

        Args:
            n_std: 标准差倍数, 默认2倍标准差 (约95%置信区间)

        Returns:
            (lower_bound, upper_bound): 非套利区间下界和上界
        """
        carry_cost = self.calculate_carry_cost()
        std = np.std(carry_cost)
        mean = np.mean(carry_cost)
        upper_bound = mean + n_std * std
        lower_bound = mean - n_std * std
        return lower_bound, upper_bound

    def detect_arbitrage_opportunity(
        self,
        current_spread: Optional[float] = None,
        n_std: float = 2.0
    ) -> dict:
        """
        检测当前是否存在套利机会

        Args:
            current_spread: 当前价差 (期货-现货), 如果为None则使用最新数据
            n_std: 标准差倍数

        Returns:
            dict: {
                'has_opportunity': bool,
                'direction': 'long_spot_short_futures' / 'long_futures_short_spot' / None,
                'spread': float,
                'lower_bound': float,
                'upper_bound': float,
                'deviation': float
            }
        """
        lower_bound, upper_bound = self.calculate_arbitrage_interval(n_std)

        if current_spread is None:
            current_spread = float(self.calculate_carry_cost().iloc[-1])

        deviation = current_spread - (upper_bound + lower_bound) / 2

        result = {
            'has_opportunity': False,
            'direction': None,
            'spread': current_spread,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'deviation': deviation
        }

        if current_spread > upper_bound:
            # 期货价格偏高: 做空期货, 做多现货 (正向套利)
            result['has_opportunity'] = True
            result['direction'] = 'long_spot_short_futures'
        elif current_spread < lower_bound:
            # 期货价格偏低: 做多期货, 做空现货 (反向套利)
            result['has_opportunity'] = True
            result['direction'] = 'long_futures_short_spot'

        return result

    def general_equilibrium_model(self) -> Tuple[float, float]:
        """
        一般均衡模型计算非套利区间 (待实现)

        使用一般均衡模型来开发股票指数期货的封闭均衡定价模型

        Returns:
            (lower_bound, upper_bound): 基于均衡模型的非套利区间
        """
        raise NotImplementedError(
            "General equilibrium model not yet implemented. "
            "Use calculate_arbitrage_interval() instead."
        )


class SpotFuturesArbitrageur:
    """
    期现套利综合执行器

    整合现货组合构建、期货定价区间分析和交易信号生成,
    提供完整的期现套利框架。
    """

    def __init__(
        self,
        spot_price_df: pd.DataFrame,
        futures_price_df: pd.DataFrame,
        spot_portfolio_weights: Optional[np.ndarray] = None,
        replication_type: str = 'full'
    ):
        """
        Args:
            spot_price_df: 现货价格数据
            futures_price_df: 期货价格数据
            spot_portfolio_weights: 预设的现货组合权重, 如果为None则自动构建
            replication_type: 复制类型, 当spot_portfolio_weights为None时使用
        """
        self.spot_price_df = spot_price_df
        self.futures_price_df = futures_price_df

        # 初始化现货组合构建器
        self._underlying_index_df = self._infer_underlying_index()
        self.spot_portfolio = SpotPortfolioConstruction(self._underlying_index_df)

        if spot_portfolio_weights is None:
            self.spot_portfolio_weights = self.spot_portfolio.replication_method(replication_type)
        else:
            self.spot_portfolio_weights = spot_portfolio_weights

        # 初始化期货定价分析器
        self.pricing_intervals = FuturesPricingArbitrageIntervals(
            futures_price_df, spot_price_df
        )

    def _infer_underlying_index(self) -> pd.DataFrame:
        """
        从现货价格数据推断底层指数

        Returns:
            underlying_index_df: 底层指数数据
        """
        # 如果只有单一现货价格, 创建单列虚拟数据
        n = len(self.spot_price_df)
        return pd.DataFrame({'symbol': ['SPOT'] * n, 'weight': [1.0] * n})

    def calculate_spread(self) -> pd.Series:
        """
        计算当前期货与现货的价差

        Returns:
            spread: 价差序列 (期货 - 现货)
        """
        return self.pricing_intervals.calculate_carry_cost()

    def generate_signal(
        self,
        predicted_spread: float,
        n_std: float = 2.0
    ) -> dict:
        """
        基于预测价差生成交易信号

        Args:
            predicted_spread: 预测的未来价差
            n_std: 标准差倍数

        Returns:
            dict: 交易信号
        """
        opp = self.pricing_intervals.detect_arbitrage_opportunity(
            current_spread=predicted_spread,
            n_std=n_std
        )

        if not opp['has_opportunity']:
            return {
                'action': 'hold',
                'reason': 'spread_within_arbitrage_band',
                'details': opp
            }

        return {
            'action': 'execute',
            'direction': opp['direction'],
            'reason': f'arbitrage_opportunity_detected_{opp["direction"]}',
            'details': opp
        }

    def backtest(
        self,
        predicted_spreads: np.ndarray,
        window_size: int,
        n_std: float = 2.0
    ) -> dict:
        """
        回测套利策略

        Args:
            predicted_spreads: 预测的价差数组
            window_size: 窗口大小
            n_std: 标准差倍数

        Returns:
            dict: 回测结果 (Sharpe比率, 最大回撤等)
        """
        lower_bounds, upper_bounds = [], []
        for i in range(window_size, len(self.spot_price_df)):
            interval = self.pricing_intervals.calculate_arbitrage_interval(n_std)
            lower_bounds.append(interval[0])
            upper_bounds.append(interval[1])

        signals = []
        for i, pred_spread in enumerate(predicted_spreads[window_size:]):
            signal = self.generate_signal(pred_spread, n_std)
            signals.append(signal['action'])

        # 计算收益率
        spot_returns = self.spot_price_df['price'].pct_change().iloc[window_size:].values
        futures_returns = (
            self.futures_price_df['price'].iloc[window_size:].values -
            self.futures_price_df['price'].iloc[window_size - 1:-1].values
        ) / self.futures_price_df['price'].iloc[window_size - 1:-1].values

        # 策略收益: 当signal为execute时获取收益
        strategy_returns = []
        for i, signal in enumerate(signals):
            if signal == 'execute':
                # 基于方向计算收益
                direction = signals[i].get('direction', '') if isinstance(signals[i], dict) else ''
                if 'long_spot_short_futures' in str(direction):
                    strategy_returns.append(spot_returns[i] - futures_returns[i])
                elif 'long_futures_short_spot' in str(direction):
                    strategy_returns.append(futures_returns[i] - spot_returns[i])
                else:
                    strategy_returns.append(0)
            else:
                strategy_returns.append(0)

        strategy_returns = np.array(strategy_returns)

        sharpe = self._calculate_sharpe_ratio(strategy_returns)
        max_dd = self._calculate_max_drawdown(strategy_returns)

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_return': np.sum(strategy_returns),
            'n_trades': sum(1 for s in signals if s == 'execute'),
            'win_rate': np.mean(strategy_returns > 0) if len(strategy_returns) > 0 else 0
        }

    @staticmethod
    def _calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """计算年化Sharpe比率"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252)

    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """计算最大回撤"""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
