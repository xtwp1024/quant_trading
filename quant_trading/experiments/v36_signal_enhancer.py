# -*- coding: utf-8 -*-
"""
V36 信号增强模块

添加三层滤波：
1. 市场情绪滤波 - 基于市场指数动量
2. 板块轮动滤波 - 600/000 看大盘，300 看创业板
3. 资金流滤波 - 基于成交量-价格相关性

Usage:
    from quant_trading.experiments.v36_signal_enhancer import V36SignalEnhancer

    enhancer = V36SignalEnhancer(params=V36Params())
    df = enhancer.add_market_sentiment(df, market_index_df)
    df = enhancer.add_sector_rotation(df, sector_etf_df)
    df = enhancer.add_money_flow(df)
    df = enhancer.apply_filters(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SignalEnhancerParams:
    """信号增强参数"""
    # 市场情绪滤波
    enable_market_filter: bool = False  # 默认关闭，等待有大盘数据时启用
    market_ma_period: int = 20        # 市场MA周期
    market_sentiment_threshold: float = -0.02  # 市场日跌幅超此值不看多

    # 板块轮动滤波
    enable_sector_filter: bool = True   # 启用板块轮动滤波
    sector_momentum_period: int = 5     # 板块动量周期(天)
    sector_lead_threshold: float = 0.01  # 领先阈值

    # 资金流滤波
    enable_money_flow: bool = True     # 启用资金流滤波
    money_flow_lookback: int = 20      # 资金流回看周期
    money_flow_threshold: float = 0.3   # 资金流入阈值

    # 指数代码映射
    index_codes: Dict[str, str] = field(default_factory=lambda: {
        "600": "sh000001",  # 上证指数
        "000": "sh000001",  # 深证成指
        "001": "sh000001",  # 也是上证
        "002": "sz399001",  # 中小板
        "300": "sz399006",  # 创业板
    })


class V36SignalEnhancer:
    """V36信号增强器

    添加三层滤波器，减少假信号：
    1. 市场情绪滤波 - 大盘不好时不入场
    2. 板块轮动滤波 - 选强势板块
    3. 资金流滤波 - 选资金流入的股票
    """

    def __init__(self, params: SignalEnhancerParams = None):
        self.params = params or SignalEnhancerParams()

    def add_market_sentiment(
        self,
        df: pd.DataFrame,
        market_index: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """添加市场情绪指标

        Args:
            df: 股票数据 (需要有 'date', 'code', 'close' 列)
            market_index: 大盘指数数据 (需要有 'date', 'code', 'close' 列)

        Returns:
            添加了市场情绪指标的DataFrame
        """
        p = self.params
        if not p.enable_market_filter:
            df['Market_Sentiment'] = 1  # 不过滤
            return df

        # 如果没有大盘数据，用简单方法：计算整体市场动量
        if market_index is None:
            # 使用市场平均涨幅作为情绪指标
            market_return = df.groupby('date')['close'].pct_change()
            market_sentiment = market_return.rolling(p.market_ma_period).mean()
            df['Market_Sentiment'] = market_sentiment.fillna(0)
        else:
            # 合并大盘数据
            index_code = list(p.index_codes.values())[0]
            index_df = market_index[market_index['code'] == index_code].copy()
            index_df = index_df.sort_values('date')

            # 计算大盘MA和动量
            index_df['Market_MA'] = index_df['close'].rolling(p.market_ma_period).mean()
            index_df['Market_Momentum'] = (
                (index_df['close'] - index_df['Market_MA']) / index_df['Market_MA']
            )

            # 大盘日收益率
            index_df['Market_Daily_Return'] = index_df['close'].pct_change()

            # 合并到股票数据
            df = df.merge(
                index_df[['date', 'Market_Momentum', 'Market_Daily_Return']],
                on='date',
                how='left'
            )
            df['Market_Momentum'] = df['Market_Momentum'].fillna(0)
            df['Market_Daily_Return'] = df['Market_Daily_Return'].fillna(0)

            # 市场情绪: 大盘在MA上方 + 日涨幅 > 阈值
            df['Market_Sentiment'] = np.where(
                (df['Market_Momentum'] > 0) &
                (df['Market_Daily_Return'] > p.market_sentiment_threshold),
                1, 0
            )

        return df

    def add_sector_rotation(
        self,
        df: pd.DataFrame,
        sector_codes: Dict[str, str] = None,
    ) -> pd.DataFrame:
        """添加板块轮动信号

        规则:
        - 600/000/001 开头: 看上证指数
        - 300/002 开头: 看创业板/中小板

        Args:
            df: 股票数据
            sector_codes: 股票-板块映射

        Returns:
            添加了板块动量指标的DataFrame
        """
        p = self.params
        if not p.enable_sector_filter:
            df['Sector_Momentum'] = 1
            return df

        # 提取股票前缀
        df['Code_Prefix'] = df['code'].astype(str).str[:3]

        # 根据前缀确定参考指数
        def get_index_code(prefix):
            if prefix.startswith('300'):
                return 'sz399006'  # 创业板
            elif prefix.startswith('002'):
                return 'sz399005'  # 中小板
            else:
                return 'sh000001'  # 上证

        df['Ref_Index'] = df['Code_Prefix'].apply(get_index_code)

        # 计算每只股票的动量
        df['Stock_MA'] = df.groupby('code')['close'].transform(
            lambda x: x.rolling(p.sector_momentum_period, min_periods=1).mean()
        )
        df['Stock_Momentum'] = (
            (df['close'] - df['Stock_MA']) / df['Stock_MA']
        )

        # 计算参考指数动量 (这里简化处理，实际需要指数数据)
        # 暂时用个股自己的动量作为板块代理
        df['Sector_Momentum'] = df['Stock_Momentum']

        return df

    def add_money_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加资金流指标

        使用成交量-价格相关性判断资金流向：
        - 价涨量增: 资金流入 (健康)
        - 价跌量增: 资金流出 (可能加速下跌)
        - 价涨量缩: 资金流出 (可能假突破)
        - 价跌量缩: 资金观望

        Args:
            df: 股票数据 (需要有 'date', 'code', 'close', 'volume' 列)

        Returns:
            添加了资金流指标的DataFrame
        """
        p = self.params
        if not p.enable_money_flow:
            df['Money_Flow'] = 1
            return df

        # 收益率
        df['Ret'] = df.groupby('code')['close'].pct_change()

        # 成交量变化率
        df['Vol_Change'] = df.groupby('code')['volume'].pct_change()

        # 资金流 = 收益 * 成交量变化 (简化)
        df['Raw_Flow'] = df['Ret'] * df['Vol_Change']

        # 移动平均平滑
        df['Money_Flow'] = df.groupby('code')['Raw_Flow'].transform(
            lambda x: x.rolling(p.money_flow_lookback, min_periods=1).mean()
        )

        # 归一化到 [-1, 1]
        df['Money_Flow'] = df['Money_Flow'].clip(-1, 1).fillna(0)

        # 资金流入信号: 资金流 > 阈值
        df['Money_Flow_Signal'] = (df['Money_Flow'] > p.money_flow_threshold).astype(int)

        # 清理临时列
        df.drop(['Ret', 'Vol_Change', 'Raw_Flow'], axis=1, inplace=True)

        return df

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用所有滤波器

        综合判断是否通过所有滤波:
        1. 市场情绪: Market_Sentiment == 1
        2. 板块动量: Sector_Momentum > 0
        3. 资金流向: Money_Flow_Signal == 1

        Returns:
            添加了 Filter_Pass 列的DataFrame
        """
        p = self.params

        # 默认都通过
        if 'Market_Sentiment' not in df.columns:
            df['Market_Sentiment'] = 1
        if 'Sector_Momentum' not in df.columns:
            df['Sector_Momentum'] = 0  # 不做板块过滤
        if 'Money_Flow_Signal' not in df.columns:
            df['Money_Flow_Signal'] = 1

        # 综合过滤
        conditions = []

        if p.enable_market_filter:
            conditions.append(df['Market_Sentiment'] == 1)

        if p.enable_sector_filter:
            conditions.append(df['Sector_Momentum'] > 0)

        if p.enable_money_flow:
            conditions.append(df['Money_Flow_Signal'] == 1)

        if conditions:
            df['Filter_Pass'] = np.logical_and.reduce(conditions).astype(int)
        else:
            df['Filter_Pass'] = 1

        return df

    def get_filter_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取滤波统计信息

        Returns:
            各滤ly通过的统计
        """
        stats = {}

        if 'Market_Sentiment' in df.columns:
            stats['market_pass_rate'] = df['Market_Sentiment'].mean()

        if 'Sector_Momentum' in df.columns:
            stats['sector_positive_rate'] = (df['Sector_Momentum'] > 0).mean()

        if 'Money_Flow_Signal' in df.columns:
            stats['money_flow_pass_rate'] = df['Money_Flow_Signal'].mean()

        if 'Filter_Pass' in df.columns:
            stats['overall_pass_rate'] = df['Filter_Pass'].mean()

        return stats


# ===================== 便捷函数 =====================

def enhance_v36_signals(
    df: pd.DataFrame,
    params: SignalEnhancerParams = None,
    market_index: pd.DataFrame = None,
) -> pd.DataFrame:
    """一站式增强V36信号

    Args:
        df: 股票数据
        params: 信号增强参数
        market_index: 大盘指数数据

    Returns:
        添加了所有滤波指标的DataFrame
    """
    enhancer = V36SignalEnhancer(params)

    df = enhancer.add_market_sentiment(df, market_index)
    df = enhancer.add_sector_rotation(df)
    df = enhancer.add_money_flow(df)
    df = enhancer.apply_filters(df)

    return df


# ===================== 测试 =====================

if __name__ == '__main__':
    from quant_trading.data.ashare_loader import load_ashare_json

    print("V36 信号增强测试")
    print("=" * 50)

    # 加载数据
    json_path = "D:/量化交易系统/量化之神/data/ashare/all_stocks_data.json"
    df = load_ashare_json(json_path)
    print(f"加载数据: {len(df)} 条")

    # 只用前100条测试
    df = df.head(1000)

    # 增强信号
    params = SignalEnhancerParams(
        enable_market_filter=True,
        enable_sector_filter=True,
        enable_money_flow=True,
    )

    df = enhance_v36_signals(df, params)
    print(f"\n增强后数据: {len(df)} 条")

    # 统计
    enhancer = V36SignalEnhancer(params)
    stats = enhancer.get_filter_stats(df)

    print("\n滤波统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2%}")

    print("\n列名:", df.columns.tolist())