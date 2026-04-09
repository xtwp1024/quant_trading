"""
动态仓位管理器 - 波动率调整、风险平价
=======================================

根据市场波动率、策略表现、风险预算动态调整仓位
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class VolatilityRegime(Enum):
    """波动率状态"""
    LOW = "低波动"
    MEDIUM = "中等波动"
    HIGH = "高波动"
    EXTREME = "极端波动"


@dataclass
class PositionAllocation:
    """仓位分配"""
    strategy: str
    symbol: str
    base_allocation: float  # 基础分配比例
    adjusted_allocation: float  # 调整后分配
    reason: str  # 调整原因


class DynamicPositionSizer:
    """动态仓位管理器"""

    def __init__(
        self,
        total_capital: float = 1000.0,
        max_total_position: float = 0.9,
        max_single_position: float = 0.3,
        max_correlated_position: float = 0.5
    ):
        self.total_capital = total_capital
        self.max_total_position = max_total_position
        self.max_single_position = max_single_position
        self.max_correlated_position = max_correlated_position

        self.volatility_history = {}
        self.strategy_performance = {}
        self.correlation_matrix = pd.DataFrame()

    def calculate_atr(
        self,
        candles: pd.DataFrame,
        period: int = 14
    ) -> float:
        """
        计算平均真实波幅 (ATR)

        Args:
            candles: K线数据 (需要包含high, low, close列)
            period: 周期

        Returns:
            ATR值
        """
        high = candles['high']
        low = candles['low']
        close = candles['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return atr

    def calculate_volatility(
        self,
        candles: pd.DataFrame,
        method: str = 'std',
        period: int = 20
    ) -> float:
        """
        计算波动率

        Args:
            candles: K线数据
            method: 计算方法 ('std', 'atr', 'parkinson')
            period: 周期

        Returns:
            波动率 (百分比)
        """
        close = candles['close']

        if method == 'std':
            # 标准差方法
            returns = close.pct_change().dropna()
            volatility = returns.rolling(window=period).std().iloc[-1] * np.sqrt(1440)  # 年化
        elif method == 'atr':
            # ATR方法
            atr = self.calculate_atr(candles, period)
            current_price = close.iloc[-1]
            volatility = (atr / current_price) * 100
        elif method == 'parkinson':
            # Parkinson波动率 (基于高低价)
            high = candles['high']
            low = candles['low']
            hl_ratio = np.log(high / low) ** 2
            parkinson_vol = np.sqrt(hl_ratio.rolling(window=period).mean().iloc[-1] / (4 * np.log(2)))
            volatility = parkinson_vol * 100
        else:
            raise ValueError(f"Unknown method: {method}")

        return volatility

    def classify_volatility_regime(
        self,
        volatility: float,
        symbol: str = "ETH-USDT"
    ) -> VolatilityRegime:
        """
        分类波动率状态

        Args:
            volatility: 波动率值
            symbol: 交易对

        Returns:
            波动率状态
        """
        # 不同币种有不同的波动率阈值
        thresholds = {
            'BTC-USDT': {'low': 1.0, 'medium': 2.5, 'high': 4.0},
            'ETH-USDT': {'low': 1.5, 'medium': 3.0, 'high': 5.0},
            'default': {'low': 2.0, 'medium': 3.5, 'high': 6.0}
        }

        thresh = thresholds.get(symbol, thresholds['default'])

        if volatility < thresh['low']:
            return VolatilityRegime.LOW
        elif volatility < thresh['medium']:
            return VolatilityRegime.MEDIUM
        elif volatility < thresh['high']:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def adjust_position_by_volatility(
        self,
        base_position_pct: float,
        volatility: float,
        symbol: str = "ETH-USDT"
    ) -> PositionAllocation:
        """
        根据波动率调整仓位

        Args:
            base_position_pct: 基础仓位比例 (0-1)
            volatility: 当前波动率
            symbol: 交易对

        Returns:
            调整后的仓位分配
        """
        regime = self.classify_volatility_regime(volatility, symbol)

        # 波动率调整系数
        volatility_multiplier = {
            VolatilityRegime.LOW: 1.2,      # 低波动：增加仓位
            VolatilityRegime.MEDIUM: 1.0,   # 中等波动：标准仓位
            VolatilityRegime.HIGH: 0.7,    # 高波动：降低仓位
            VolatilityRegime.EXTREME: 0.4  # 极端波动：大幅降低
        }

        multiplier = volatility_multiplier[regime]
        adjusted_pct = base_position_pct * multiplier

        # 应用单仓位上限
        adjusted_pct = min(adjusted_pct, self.max_single_position)

        return PositionAllocation(
            strategy="volatility_adjusted",
            symbol=symbol,
            base_allocation=base_position_pct,
            adjusted_allocation=adjusted_pct,
            reason=f"波动率状态: {regime.value}, 调整系数: {multiplier:.1f}x"
        )

    def adjust_position_by_performance(
        self,
        base_position_pct: float,
        strategy_name: str,
        recent_returns: List[float],
        window: int = 20
    ) -> PositionAllocation:
        """
        根据策略表现调整仓位

        Args:
            base_position_pct: 基础仓位比例
            strategy_name: 策略名称
            recent_returns: 近期收益率列表
            window: 评估窗口

        Returns:
            调整后的仓位分配
        """
        if len(recent_returns) < window:
            # 数据不足，返回基础仓位
            return PositionAllocation(
                strategy=strategy_name,
                symbol="N/A",
                base_allocation=base_position_pct,
                adjusted_allocation=base_position_pct,
                reason="数据不足，使用基础仓位"
            )

        # 计算近期表现
        recent_returns = recent_returns[-window:]
        total_return = sum(recent_returns)
        win_rate = sum(1 for r in recent_returns if r > 0) / len(recent_returns)

        # 表现调整系数
        if total_return > 0.10 and win_rate > 0.6:
            # 表现优秀：增加仓位
            multiplier = 1.3
            reason = f"表现优秀 (收益率: {total_return:.1%}, 胜率: {win_rate:.1%})"
        elif total_return > 0.05 and win_rate > 0.55:
            # 表现良好：小幅增加
            multiplier = 1.1
            reason = f"表现良好 (收益率: {total_return:.1%}, 胜率: {win_rate:.1%})"
        elif total_return < -0.05 or win_rate < 0.45:
            # 表现不佳：降低仓位
            multiplier = 0.7
            reason = f"表现不佳 (收益率: {total_return:.1%}, 胜率: {win_rate:.1%})"
        elif total_return < -0.10 or win_rate < 0.40:
            # 表现很差：大幅降低
            multiplier = 0.4
            reason = f"表现很差 (收益率: {total_return:.1%}, 胜率: {win_rate:.1%})"
        else:
            # 表现一般：保持不变
            multiplier = 1.0
            reason = f"表现一般 (收益率: {total_return:.1%}, 胜率: {win_rate:.1%})"

        adjusted_pct = base_position_pct * multiplier
        adjusted_pct = min(adjusted_pct, self.max_single_position)

        return PositionAllocation(
            strategy=strategy_name,
            symbol="N/A",
            base_allocation=base_position_pct,
            adjusted_allocation=adjusted_pct,
            reason=reason
        )

    def risk_parity_allocation(
        self,
        strategies: Dict[str, Dict],
        total_risk_budget: float = 0.15
    ) -> Dict[str, PositionAllocation]:
        """
        风险平价配置 - 各策略风险贡献相等

        Args:
            strategies: 策略字典 {strategy_name: {volatility, expected_return}}
            total_risk_budget: 总风险预算 (组合波动率)

        Returns:
            各策略的仓位分配
        """
        n_strategies = len(strategies)

        if n_strategies == 0:
            return {}

        # 计算各策略风险贡献权重
        # 风险平价: weight_i ∝ 1/volatility_i
        volatilities = np.array([s['volatility'] for s in strategies.values()])
        # 防止零波动率导致除零
        volatilities = np.maximum(volatilities, 1e-10)
        inv_vols = 1.0 / volatilities
        weights = inv_vols / inv_vols.sum()

        # 计算组合波动率
        # 假设策略间相关性为0.5
        avg_correlation = 0.5
        portfolio_volatility = np.sqrt(
            np.sum(weights ** 2 * volatilities ** 2) +
            2 * avg_correlation * np.sum(
                weights[:, None] * weights[None, :] *
                volatilities[:, None] * volatilities[None, :]
            ) / 2
        )

        # 调整权重使组合波动率等于风险预算
        scaling_factor = total_risk_budget / portfolio_volatility
        scaled_weights = weights * scaling_factor

        # 应用仓位限制
        scaled_weights = np.minimum(scaled_weights, self.max_single_position)
        scaled_weights = scaled_weights / scaled_weights.sum() * total_risk_budget

        # 生成分配结果
        allocations = {}
        for i, (strategy_name, strategy_data) in enumerate(strategies.items()):
            allocations[strategy_name] = PositionAllocation(
                strategy="risk_parity",
                symbol=strategy_data.get('symbol', 'N/A'),
                base_allocation=1.0 / n_strategies,
                adjusted_allocation=scaled_weights[i],
                reason=f"风险平价: 波动率={strategy_data['volatility']:.2%}"
            )

        return allocations

    def calculate_correlation_matrix(
        self,
        returns_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        计算策略相关性矩阵

        Args:
            returns_data: 策略收益率数据 {strategy_name: returns_series}

        Returns:
            相关性矩阵
        """
        # 对齐时间序列
        df = pd.DataFrame(returns_data)
        correlation_matrix = df.corr()

        self.correlation_matrix = correlation_matrix
        return correlation_matrix

    def adjust_for_correlation(
        self,
        base_allocations: Dict[str, float],
        max_correlation: float = 0.7
    ) -> Dict[str, PositionAllocation]:
        """
        调整高相关策略的仓位

        Args:
            base_allocations: 基础分配 {strategy: allocation}
            max_correlation: 最大允许相关性

        Returns:
            调整后的分配
        """
        if self.correlation_matrix.empty:
            # 没有相关性数据，返回原分配
            return {
                name: PositionAllocation(
                    strategy="correlation_adjusted",
                    symbol="N/A",
                    base_allocation=alloc,
                    adjusted_allocation=alloc,
                    reason="无相关性数据"
                )
                for name, alloc in base_allocations.items()
            }

        adjusted = {}
        processed = set()

        for strategy in base_allocations:
            if strategy in processed:
                continue

            # 找到高相关策略组
            correlated = [strategy]
            for other in base_allocations:
                if other != strategy and other not in processed:
                    corr = self.correlation_matrix.loc[strategy, other]
                    if corr > max_correlation:
                        correlated.append(other)

            # 组内等分仓位
            total_allocation = sum(base_allocations[s] for s in correlated)
            equal_allocation = total_allocation / len(correlated)

            for s in correlated:
                adjusted[s] = PositionAllocation(
                    strategy="correlation_adjusted",
                    symbol="N/A",
                    base_allocation=base_allocations[s],
                    adjusted_allocation=equal_allocation,
                    reason=f"高相关策略组: {correlated}"
                )
                processed.add(s)

        return adjusted

    def generate_portfolio_allocation(
        self,
        strategies: Dict[str, Dict],
        current_volatilities: Dict[str, float],
        recent_returns: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        生成完整组合配置

        Args:
            strategies: 策略信息
            current_volatilities: 当前波动率
            recent_returns: 近期收益率

        Returns:
            最终仓位分配 {strategy: allocation}
        """
        allocations = {}

        for strategy_name, strategy_data in strategies.items():
            base_allocation = strategy_data.get('base_allocation', 1.0 / len(strategies))

            # 1. 波动率调整
            vol_adjustment = self.adjust_position_by_volatility(
                base_allocation,
                current_volatilities.get(strategy_name, 0.02),
                strategy_data.get('symbol', 'ETH-USDT')
            )

            # 2. 表现调整
            perf_adjustment = self.adjust_position_by_performance(
                vol_adjustment.adjusted_allocation,
                strategy_name,
                recent_returns.get(strategy_name, [])
            )

            allocations[strategy_name] = perf_adjustment.adjusted_allocation

        # 3. 相关性调整
        corr_adjustments = self.adjust_for_correlation(allocations)
        allocations = {
            name: adj.adjusted_allocation
            for name, adj in corr_adjustments.items()
        }

        # 4. 归一化到最大总仓位
        total = sum(allocations.values())
        if total > self.max_total_position:
            scale_factor = self.max_total_position / total
            allocations = {k: v * scale_factor for k, v in allocations.items()}

        return allocations


def example_usage():
    """使用示例"""

    print("="*80)
    print("动态仓位管理器 - 使用示例")
    print("="*80)

    sizer = DynamicPositionSizer(total_capital=1000)

    # 示例1: 波动率调整
    print("\n示例1: 根据波动率调整仓位")
    print("-"*40)

    base_position = 0.2  # 20%
    volatilities = [0.8, 1.5, 2.5, 4.0, 6.0]

    for vol in volatilities:
        allocation = sizer.adjust_position_by_volatility(base_position, vol)
        regime = sizer.classify_volatility_regime(vol)
        print(f"波动率: {vol:.1f}% | 状态: {regime.value:8} | "
              f"仓位: {allocation.adjusted_allocation:.1%}")

    # 示例2: 表现调整
    print("\n示例2: 根据策略表现调整仓位")
    print("-"*40)

    strategy_returns = {
        'excellent': [0.05, 0.03, 0.04, 0.06, 0.02, 0.05, 0.04, 0.03, 0.05, 0.04],
        'good': [0.03, 0.02, 0.04, -0.01, 0.03, 0.02, 0.04, 0.01, 0.03, 0.02],
        'poor': [-0.02, -0.03, 0.01, -0.04, -0.02, -0.01, -0.03, -0.02, 0.01, -0.02],
        'terrible': [-0.05, -0.04, -0.06, -0.03, -0.05, -0.04, -0.07, -0.05, -0.03, -0.06]
    }

    for strategy, returns in strategy_returns.items():
        allocation = sizer.adjust_position_by_performance(0.2, strategy, returns)
        print(f"{strategy:12} | {allocation.adjusted_allocation:.1%} | {allocation.reason}")

    # 示例3: 风险平价
    print("\n示例3: 风险平价配置")
    print("-"*40)

    strategies = {
        'Martin': {'volatility': 0.15, 'symbol': 'ETH-USDT'},
        'Grid': {'volatility': 0.08, 'symbol': 'ETH-USDT'},
        'Trend': {'volatility': 0.12, 'symbol': 'ETH-USDT'},
        'Arbitrage': {'volatility': 0.05, 'symbol': 'ETH-USDT'}
    }

    allocations = sizer.risk_parity_allocation(strategies, total_risk_budget=0.15)

    print(f"{'策略':<12} {'基础仓位':<10} {'调整后仓位':<12} {'原因'}")
    print("-"*60)

    for strategy, alloc in allocations.items():
        print(f"{strategy:<12} {alloc.base_allocation:>8.1%}   {alloc.adjusted_allocation:>10.1%}   "
              f"{alloc.reason}")

    # 示例4: 完整组合配置
    print("\n示例4: 生成完整组合配置")
    print("-"*40)

    strategies_info = {
        'Martin': {'base_allocation': 0.25, 'symbol': 'ETH-USDT'},
        'Grid': {'base_allocation': 0.25, 'symbol': 'ETH-USDT'},
        'Trend': {'base_allocation': 0.25, 'symbol': 'BTC-USDT'},
        'Arbitrage': {'base_allocation': 0.25, 'symbol': 'BNB-USDT'}
    }

    current_vols = {
        'Martin': 0.025,
        'Grid': 0.015,
        'Trend': 0.018,
        'Arbitrage': 0.010
    }

    recent_rets = {
        'Martin': [0.02, 0.03, -0.01, 0.04, 0.02],
        'Grid': [0.01, 0.02, 0.01, 0.03, 0.01],
        'Trend': [0.04, 0.05, 0.03, 0.02, 0.04],
        'Arbitrage': [0.01, 0.01, 0.02, 0.01, 0.01]
    }

    final_allocation = sizer.generate_portfolio_allocation(
        strategies_info,
        current_vols,
        recent_rets
    )

    print(f"\n最终组合配置:")
    print(f"{'策略':<12} {'配置比例':<10} {'仓位金额'}")
    print("-"*35)

    for strategy, alloc in final_allocation.items():
        amount = sizer.total_capital * alloc
        print(f"{strategy:<12} {alloc:>8.1%}   ${amount:>7.2f}")

    print(f"\n总仓位: {sum(final_allocation.values()):.1%}")
    print(f"总金额: ${sum(final_allocation.values()) * sizer.total_capital:.2f}")


if __name__ == "__main__":
    example_usage()
