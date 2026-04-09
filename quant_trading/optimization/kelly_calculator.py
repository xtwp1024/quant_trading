"""
凯利公式计算器 - 科学仓位管理
==============================

凯利公式: f* = (bp - q) / b
其中:
- b = 盈亏比 (平均盈利 / 平均亏损)
- p = 胜率 (盈利交易占比)
- q = 1 - p (败率)

凯利公式给出了最优仓位比例，可以最大化长期增长率
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path


class KellyCalculator:
    """凯利公式计算器"""

    def __init__(self):
        self.trade_history = []

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        计算凯利公式仓位比例

        Args:
            win_rate: 胜率 (0-1之间)
            avg_win: 平均盈利金额
            avg_loss: 平均亏损金额 (正数)

        Returns:
            最优仓位比例 (0-1之间)
        """
        # Validate inputs to prevent division by zero or invalid calculations
        if avg_loss <= 0:
            return 0.0

        # 盈亏比
        b = avg_win / avg_loss

        # If win/loss ratio is not positive, Kelly formula is invalid
        if b <= 0:
            return 0.0

        # 凯利公式
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # 限制在0-1之间
        kelly_fraction = max(0.0, min(1.0, kelly_fraction))

        return kelly_fraction

    def calculate_from_trades(
        self,
        trades: List[Dict],
        min_trades: int = 20
    ) -> Dict:
        """
        从交易历史计算凯利参数

        Args:
            trades: 交易历史列表
            min_trades: 最小交易次数

        Returns:
            {
                'win_rate': 胜率,
                'avg_win': 平均盈利,
                'avg_loss': 平均亏损,
                'win_loss_ratio': 盈亏比,
                'kelly_fraction': 凯利仓位,
                'half_kelly': 半凯利仓位,
                'quarter_kelly': 四分之一凯利仓位
            }
        """
        if len(trades) < min_trades:
            raise ValueError(f"交易次数不足，需要至少{min_trades}笔交易")

        # 分类盈利和亏损交易
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        if not winning_trades or not losing_trades:
            raise ValueError("需要同时有盈利和亏损交易")

        # 计算统计量
        win_rate = len(winning_trades) / len(trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades])
        avg_loss = abs(np.mean([t['pnl'] for t in losing_trades]))
        win_loss_ratio = avg_win / avg_loss

        # 凯利公式
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'kelly_fraction': kelly_fraction,
            'half_kelly': kelly_fraction / 2,
            'quarter_kelly': kelly_fraction / 4,
            'calculated_at': datetime.now().isoformat()
        }

    def calculate_position_size(
        self,
        kelly_fraction: float,
        total_capital: float,
        kelly_multiplier: float = 0.5,
        max_position: float = 0.3
    ) -> float:
        """
        计算实际仓位大小

        Args:
            kelly_fraction: 凯利公式计算的理论仓位
            total_capital: 总资金
            kelly_multiplier: 凯利乘数 (默认0.5 = 半凯利)
            max_position: 最大仓位限制

        Returns:
            实际仓位金额
        """
        # 应用凯利乘数
        adjusted_fraction = kelly_fraction * kelly_multiplier

        # 应用最大仓位限制
        adjusted_fraction = min(adjusted_fraction, max_position)

        # 计算仓位金额
        position_size = total_capital * adjusted_fraction

        return position_size

    def simulate_growth(
        self,
        initial_capital: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        kelly_fraction: float,
        n_trades: int = 100,
        n_simulations: int = 1000
    ) -> Dict:
        """
        蒙特卡洛模拟：对比不同凯利乘数的长期增长

        Args:
            initial_capital: 初始资金
            win_rate: 胜率
            avg_win_pct: 平均盈利百分比
            avg_loss_pct: 平均亏损百分比
            kelly_fraction: 凯利仓位
            n_trades: 模拟交易次数
            n_simulations: 模拟次数

        Returns:
            模拟结果
        """
        multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        results = {}

        for multiplier in multipliers:
            final_capitals = []

            for _ in range(n_simulations):
                capital = initial_capital

                for _ in range(n_trades):
                    position_size = capital * kelly_fraction * multiplier

                    # 随机决定盈利或亏损
                    if np.random.random() < win_rate:
                        # 盈利
                        capital += position_size * avg_win_pct
                    else:
                        # 亏损
                        capital -= position_size * avg_loss_pct

                    # 破产检查
                    if capital <= 0:
                        capital = 0
                        break

                final_capitals.append(capital)

            results[f'kelly_{multiplier}'] = {
                'multiplier': multiplier,
                'mean_final_capital': np.mean(final_capitals),
                'median_final_capital': np.median(final_capitals),
                'std_final_capital': np.std(final_capitals),
                'min_final_capital': np.min(final_capitals),
                'max_final_capital': np.max(final_capitals),
                'bankruptcy_rate': sum(1 for c in final_capitals if c <= 0) / n_simulations,
                'final_capitals': final_capitals
            }

        return results

    def recommend_kelly_multiplier(
        self,
        simulation_results: Dict,
        risk_tolerance: str = 'moderate'
    ) -> float:
        """
        根据模拟结果推荐凯利乘数

        Args:
            simulation_results: simulate_growth() 的返回结果
            risk_tolerance: 风险偏好 ('conservative', 'moderate', 'aggressive')

        Returns:
            推荐的凯利乘数
        """
        scores = {}

        for key, result in simulation_results.items():
            multiplier = result['multiplier']
            mean_return = (result['mean_final_capital'] - 1000) / 1000
            bankruptcy_rate = result['bankruptcy_rate']
            std_return = result['std_final_capital'] / 1000

            # 评分公式
            if risk_tolerance == 'conservative':
                # 保守：惩罚破产率
                score = mean_return - 10 * bankruptcy_rate
            elif risk_tolerance == 'moderate':
                # 中等：平衡收益和破产率
                score = mean_return - 3 * bankruptcy_rate - std_return
            else:  # aggressive
                # 激进：追求收益
                score = mean_return - bankruptcy_rate

            scores[key] = score

        best_key = max(scores, key=scores.get())
        best_multiplier = simulation_results[best_key]['multiplier']

        return best_multiplier

    def calculate_series_kelly(
        self,
        returns: pd.Series
    ) -> Dict:
        """
        使用历史收益率序列计算凯利参数

        Args:
            returns: 收益率序列 (例如: [0.05, -0.03, 0.02, ...])

        Returns:
            凯利参数字典
        """
        if len(returns) < 20:
            raise ValueError("至少需要20个收益率数据点")

        # 统计量
        mean_return = returns.mean()
        std_return = returns.std()

        # 胜率
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(positive_returns) == 0 or len(negative_returns) == 0:
            raise ValueError("需要同时有正负收益率")

        win_rate = len(positive_returns) / len(returns)
        avg_win = positive_returns.mean()
        avg_loss = abs(negative_returns.mean())

        # 凯利公式 (简化版: mean / std^2)
        if std_return <= 0:
            return {
                'error': 'Insufficient return variance for Kelly calculation (std_return <= 0)',
                'mean_return': mean_return,
                'std_return': std_return,
            }
        kelly_simple = mean_return / (std_return ** 2)

        # 凯利公式 (完整版)
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio

        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'kelly_simple': kelly_simple,
            'kelly_fraction': kelly_fraction,
            'half_kelly': kelly_fraction / 2,
            'quarter_kelly': kelly_fraction / 4
        }


def example_usage():
    """使用示例"""

    print("="*80)
    print("凯利公式计算器 - 使用示例")
    print("="*80)

    kelly = KellyCalculator()

    # 示例1: 从交易历史计算
    print("\n示例1: 从交易历史计算")
    print("-"*40)

    trades = [
        {'pnl': 150}, {'pnl': -80}, {'pnl': 200},
        {'pnl': -60}, {'pnl': 180}, {'pnl': -90},
        {'pnl': 220}, {'pnl': -70}, {'pnl': 160},
        {'pnl': -50}, {'pnl': 190}, {'pnl': -85},
        {'pnl': 210}, {'pnl': -65}, {'pnl': 175},
        {'pnl': -75}, {'pnl': 195}, {'pnl': -55},
        {'pnl': 205}, {'pnl': -95}
    ]

    result = kelly.calculate_from_trades(trades)

    print(f"总交易次数: {result['total_trades']}")
    print(f"盈利交易: {result['winning_trades']}")
    print(f"亏损交易: {result['losing_trades']}")
    print(f"胜率: {result['win_rate']:.1%}")
    print(f"平均盈利: ${result['avg_win']:.2f}")
    print(f"平均亏损: ${result['avg_loss']:.2f}")
    print(f"盈亏比: {result['win_loss_ratio']:.2f}")
    print(f"\n凯利公式结果:")
    print(f"  全凯利仓位: {result['kelly_fraction']:.1%}")
    print(f"  半凯利仓位: {result['half_kelly']:.1%}")
    print(f"  四分之一凯利: {result['quarter_kelly']:.1%}")

    # 示例2: 计算实际仓位
    print("\n\n示例2: 计算实际仓位")
    print("-"*40)

    total_capital = 1000
    kelly_fraction = result['kelly_fraction']

    for mult in [0.25, 0.5, 0.75, 1.0]:
        position = kelly.calculate_position_size(
            kelly_fraction,
            total_capital,
            kelly_multiplier=mult,
            max_position=0.3
        )
        print(f"{mult*100:.0f}%凯利: ${position:.2f} ({position/total_capital:.1%})")

    # 示例3: 模拟长期增长
    print("\n\n示例3: 模拟长期增长 (100笔交易)")
    print("-"*40)

    simulation = kelly.simulate_growth(
        initial_capital=1000,
        win_rate=result['win_rate'],
        avg_win_pct=0.15,  # 15%
        avg_loss_pct=0.08,  # 8%
        kelly_fraction=result['kelly_fraction'],
        n_trades=100,
        n_simulations=1000
    )

    print(f"{'凯利乘数':<12} {'平均最终资金':<15} {'中位数':<12} {'破产率':<10}")
    print("-"*55)

    for key, data in simulation.items():
        mult = data['multiplier']
        mean_capital = data['mean_final_capital']
        median_capital = data['median_final_capital']
        bankruptcy = data['bankruptcy_rate']
        print(f"{mult*100:.0f}%{'':8} ${mean_capital:>10.2f}   ${median_capital:>8.2f}   {bankruptcy:>6.1%}")

    # 推荐乘数
    print("\n推荐:")
    for risk in ['conservative', 'moderate', 'aggressive']:
        mult = kelly.recommend_kelly_multiplier(simulation, risk)
        print(f"  {risk.capitalize():12} -> {mult*100:.0f}% 凯利")


if __name__ == "__main__":
    example_usage()
