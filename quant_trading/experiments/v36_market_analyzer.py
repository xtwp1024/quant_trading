# -*- coding: utf-8 -*-
"""
V36 市场分析器 - 基于交易经验的策略增强

根据 量化之神 的实战经验:
1. 70%以上大盘都是冲高回落诱多
2. 高抛低吸是主要盈利模式
3. 多维度验证不能单一维度

功能:
- 冲高回落识别
- 高抛低吸信号
- 市场位置判断
- 失败率分析

Usage:
    from quant_trading.experiments.v36_market_analyzer import V36MarketAnalyzer

    analyzer = V36MarketAnalyzer()
    df = analyzer.add_market_context(df)
    df = analyzer.add_reversal_signals(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MarketContextParams:
    """市场上下文参数"""
    # 冲高回落识别
    surge_threshold: float = 0.02     # 涨幅>2%视为冲高
    pullback_ratio: float = 0.50     # 回落幅度/冲高幅度 > 50% 视为回落

    # 高抛低吸
    profit_take_ratio: float = 0.70  # 70%以上概率冲高回落
    band_width_threshold: float = 0.03  # 振幅阈值

    # 市场位置
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])


class V36MarketAnalyzer:
    """V36市场分析器

    基于实战经验的市场分析和信号增强
    """

    def __init__(self, params: MarketContextParams = None):
        self.params = params or MarketContextParams()

    def add_market_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市场上下文信息

        Args:
            df: 股票数据

        Returns:
            添加了市场上下文的DataFrame
        """
        p = self.params

        # 计算各周期MA
        for period in p.ma_periods:
            df[f'MA{period}'] = df.groupby('code')['close'].transform(
                lambda x: x.rolling(period, min_periods=1).mean()
            )

        # 大盘位置判断
        df['Above_MA5'] = df['close'] > df['MA5']
        df['Above_MA10'] = df['close'] > df['MA10']
        df['Above_MA20'] = df['close'] > df['MA20']
        df['Above_MA60'] = df['close'] > df['MA60']

        # 多头排列判断
        df['Bullish_Arrangement'] = (
            (df['MA5'] > df['MA10']) &
            (df['MA10'] > df['MA20']) &
            (df['MA20'] > df['MA60'])
        ).astype(int)

        return df

    def add_reversal_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加反转信号（冲高回落识别）

        核心逻辑:
        - 识别冲高（当日涨幅 > threshold）
        - 识别回落（冲高后下跌超过一定比例）
        - 70%以上概率是诱多

        Args:
            df: 股票数据

        Returns:
            添加了反转信号的DataFrame
        """
        p = self.params

        # 日收益率
        df['Daily_Return'] = df.groupby('code')['close'].pct_change()

        # 过去N日的最高价和涨幅
        for window in [3, 5, 10]:
            df[f'High_{window}d'] = df.groupby('code')['high'].transform(
                lambda x: x.rolling(window, min_periods=1).max().shift(1)
            )
            df[f'Return_{window}d'] = (
                (df['close'] - df[f'High_{window}d']) / df[f'High_{window}d']
            )

        # 冲高信号: 创N日新高
        df['Surge_3d'] = (df['close'] > df['High_3d']).astype(int)
        df['Surge_5d'] = (df['close'] > df['High_5d']).astype(int)

        # 回落信号: 从高点回落
        df['Pullback_From_3d'] = (
            (df['Return_3d'] < 0) &
            (df['Return_3d'] > -p.pullback_ratio)
        ).astype(int)

        # 反转向下信号（诱多）
        # 冲高后回落，可能是诱多
        df['Fakeout_Signal'] = (
            (df['Surge_3d'] == 1) &
            (df['Daily_Return'] < -0.01)
        ).astype(int)

        # 清理临时列
        for window in [3, 5, 10]:
            df.drop([f'High_{window}d', f'Return_{window}d'], axis=1, inplace=True)

        return df

    def add_band_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加震荡/高抛低吸信号

        Args:
            df: 股票数据

        Returns:
            添加了震荡信号的DataFrame
        """
        p = self.params

        # 振幅
        df['Amplitude'] = (df['high'] - df['low']) / df['low']

        # 震荡市场信号: 振幅较大但没有明显趋势
        df['Volatile_Market'] = (
            (df['Amplitude'] > p.band_width_threshold) &
            (~df['Bullish_Arrangement'].astype(bool)) &
            (df['Above_MA20'] == False)
        ).astype(int)

        # 低吸信号: 价格接近布林带下轨
        df['Near_Lower_Band'] = (
            (df['close'] < df['MA20'] * 1.02) &
            (df['Amplitude'] > p.band_width_threshold * 0.5)
        ).astype(int)

        # 高抛信号: 价格接近布林带上轨
        df['Near_Upper_Band'] = (
            (df['close'] > df['MA20'] * 0.98) &
            (df['Amplitude'] > p.band_width_threshold * 0.5)
        ).astype(int)

        return df

    def add_failure_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加失败概率估计

        基于历史数据分析:
        - 70%以上大盘都是冲高回落诱多
        - 当市场处于高位时，失败概率更高

        Args:
            df: 股票数据

        Returns:
            添加了失败概率的DataFrame
        """
        p = self.params

        # 基础失败概率（市场经验值）
        base_failure_prob = 0.70

        # 根据市场位置调整失败概率
        # 当价格在高位（接近布林带上轨）时，增加失败概率
        df['Band_Position'] = (
            (df['close'] - df['MA20']) / (df['MA20'] * 0.02)  # 布林带宽度标准化
        ).clip(-50, 50)  # 限制范围

        # 价格在MA60上方且多空头排列破坏时，高位风险
        df['HighRisk_Position'] = (
            (df['close'] > df['MA60']) &
            (~df['Bullish_Arrangement'].astype(bool))
        ).astype(int)

        # 调整后的失败概率
        df['Failure_Probability'] = (
            base_failure_prob +
            df['HighRisk_Position'] * 0.15 -  # 高位增加15%
            df['Near_Lower_Band'].astype(int) * 0.10  # 低吸减少10%
        ).clip(0.3, 0.95)  # 限制在30%-95%

        return df

    def calculate_strategy_score(
        self,
        df: pd.DataFrame,
        buy_signals: List[str] = None,
    ) -> pd.DataFrame:
        """计算策略综合评分

        综合考虑:
        - 市场位置 (多头排列加分)
        - 反转风险 (冲高回落减分)
        - 震荡风险 (高抛低吸环境)
        - 失败概率 (调整信号强度)

        Args:
            df: 股票数据
            buy_signals: 买入信号列表

        Returns:
            添加了策略评分的DataFrame
        """
        if buy_signals is None:
            buy_signals = ['Above_MA5', 'Above_MA10', 'Above_MA20']

        # 市场位置评分 (0-1)
        df['Market_Score'] = (
            df['Above_MA5'].astype(int) * 0.2 +
            df['Above_MA10'].astype(int) * 0.2 +
            df['Above_MA20'].astype(int) * 0.3 +
            df['Above_MA60'].astype(int) * 0.3
        )

        # 反转风险评分 (0-1, 越高风险越大)
        df['Reversal_Risk'] = (
            df['Fakeout_Signal'] * 0.5 +
            df['HighRisk_Position'] * 0.3 +
            df['Volatile_Market'] * 0.2
        )

        # 综合评分
        # 市场好 + 反转风险低 = 高分
        df['Strategy_Score'] = (
            df['Market_Score'] * (1 - df['Reversal_Risk']) * (1 - df['Failure_Probability'] * 0.5)
        ).clip(0, 1)

        return df

    def apply_full_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用完整市场分析

        Args:
            df: 股票数据

        Returns:
            添加了完整分析的DataFrame
        """
        df = self.add_market_context(df)
        df = self.add_reversal_signals(df)
        df = self.add_band_signals(df)
        df = self.add_failure_probability(df)
        df = self.calculate_strategy_score(df)

        return df

    def get_trading_recommendation(self, row: pd.Series) -> Dict[str, any]:
        """根据分析结果给出交易建议

        Args:
            row: 一行股票数据

        Returns:
            交易建议
        """
        recommendations = {
            'action': 'HOLD',
            'reason': '',
            'risk_level': 'MEDIUM',
            'notes': []
        }

        # 失败概率过高
        if row.get('Failure_Probability', 0) > 0.80:
            recommendations['action'] = 'AVOID'
            recommendations['reason'] = f"失败概率 {row['Failure_Probability']:.0%} 过高"
            recommendations['risk_level'] = 'HIGH'
            return recommendations

        # 高位反转信号
        if row.get('Fakeout_Signal', 0) == 1:
            recommendations['action'] = 'SELL'
            recommendations['reason'] = '识别冲高回落诱多'
            recommendations['risk_level'] = 'HIGH'
            recommendations['notes'].append('70%以上概率是诱多')
            return recommendations

        # 低吸机会
        if row.get('Near_Lower_Band', 0) == 1:
            if row.get('Above_MA20', 0) == 1:  # 但仍在MA20上方
                recommendations['action'] = 'BUY'
                recommendations['reason'] = '接近支撑位，可能反弹'
                recommendations['risk_level'] = 'LOW'
                return recommendations

        # 强势信号
        if row.get('Strategy_Score', 0) > 0.6:
            if row.get('Bullish_Arrangement', 0) == 1:
                recommendations['action'] = 'BUY'
                recommendations['reason'] = f"综合评分 {row['Strategy_Score']:.2f}，多头排列"
                recommendations['risk_level'] = 'LOW'
                return recommendations

        # 震荡市场
        if row.get('Volatile_Market', 0) == 1:
            recommendations['action'] = 'WATCH'
            recommendations['reason'] = '震荡市场，高抛低吸'
            recommendations['risk_level'] = 'MEDIUM'
            return recommendations

        return recommendations


# ===================== 测试 =====================

if __name__ == '__main__':
    from quant_trading.data.ashare_loader import load_ashare_json

    print("V36 市场分析器测试")
    print("=" * 50)

    # 加载数据
    json_path = "D:/量化交易系统/量化之神/data/ashare/all_stocks_data.json"
    df = load_ashare_json(json_path)
    print(f"加载数据: {len(df)} 条")

    # 只用前1000条测试
    df = df.head(1000)

    # 应用分析
    analyzer = V36MarketAnalyzer()
    df = analyzer.apply_full_analysis(df)

    # 统计
    print("\n分析统计:")
    print(f"  多头排列: {df['Bullish_Arrangement'].sum()}")
    print(f"  冲高信号: {df['Surge_3d'].sum()}")
    print(f"  回落信号: {df['Pullback_From_3d'].sum()}")
    print(f"  诱多信号: {df['Fakeout_Signal'].sum()}")
    print(f"  震荡市场: {df['Volatile_Market'].sum()}")
    print(f"  低吸机会: {df['Near_Lower_Band'].sum()}")

    # 示例推荐
    print("\n示例交易推荐:")
    for idx, row in df.iterrows():
        if row['Strategy_Score'] > 0:
            rec = analyzer.get_trading_recommendation(row)
            print(f"  {row['code']}: {rec['action']} - {rec['reason']}")
            if len(df) > 5:
                break

    print("\n列名:", df.columns.tolist())