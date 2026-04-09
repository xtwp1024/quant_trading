# -*- coding: utf-8 -*-
"""
缠论 (Chan Theory) Implementation

缠论是中国本土的技术分析理论，由"缠中说禅"创立。
核心概念：
- 笔 (Bi): 由相邻的顶底分型构成，中间不少于5根K线
- 线段 (XSegment): 由连续三笔构成的一段走势
- 中枢 (ZS - Central Platform): 至少三个线段的重叠区域
- 走势类型: 上涨、下跌、盘整
- 背驰 (BeiChi): 趋势力度减弱

Reference: AbuQuant TLineBu modules
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
from collections import deque

from quant_trading.signal import Signal, SignalDirection, SignalType


class KLineDirection(Enum):
    """K线方向"""
    UP = 1
    DOWN = -1
    NEUTRAL = 0


@dataclass
class FenXing:
    """分型 (Fractal) - 顶分型或底分型"""
    index: int
    timestamp: pd.Timestamp
    high: float
    low: float
    type: str  # 'top' or 'bottom'


@dataclass
class Bi:
    """笔 (Stroke) - 笔是缠论的基本构件"""
    start_fx: FenXing
    end_fx: FenXing
    direction: KLineDirection
    kline_count: int
    highs: np.ndarray
    lows: np.ndarray


@dataclass
class XSegment:
    """线段 (Segment) - 由连续三笔构成"""
    start_bi: Bi
    end_bi: Bi
    direction: KLineDirection
    peak: float  # 最高点或最低点
    trough: float
    bis: List[Bi] = field(default_factory=list)


@dataclass
class ZhongShu:
    """中枢 (Central Platform) - 至少三个线段的重叠区域"""
    start_seg: XSegment
    end_seg: XSegment
    high: float  # 中枢区间最高点
    low: float   # 中枢区间最低点
    zg: float    # 中枢上界
    zd: float    # 中枢下界
    gg: float    # 关键的上的
    dd: float    # 关键的下


@dataclass
class TrendType:
    """走势类型"""
    type: str  # '上涨', '下跌', '盘整'
    start_idx: int
    end_idx: int
    zhangfu: float  # 涨跌幅
    segments: List[XSegment] = field(default_factory=list)
    central_platforms: List[ZhongShu] = field(default_factory=list)


@dataclass
class BeiChi:
    """背驰 (Divergence) - 趋势力度减弱"""
    exists: bool
    bi1: Bi  # 第一个上涨/下跌段
    bi2: Bi  # 第二个上涨/下跌段
    type: str  # '顶背驰' or '底背驰'
    indicator: str  # 使用的指标: 'macd', 'volume', 'price'


class ChanTheoryAnalyzer:
    """
    缠论分析器

    实现缠论的核心概念：
    1. 分型识别 - 顶分型和底分型
    2. 笔构建 - 至少5根K线构成的笔
    3. 线段识别 - 三笔构成一线段
    4. 中枢识别 - 重叠区域构成中枢
    5. 走势划分 - 上涨/下跌/盘整
    6. 背驰判断 - 趋势力度对比
    """

    def __init__(
        self,
        min_bi_klines: int = 5,
        segment_min_bi: int = 3,
        require_new_high: bool = True
    ):
        """
        Args:
            min_bi_klines: 笔的最少K线数（默认5）
            segment_min_bi: 线段的最少笔数（默认3）
            require_new_high: 构建笔时是否需要创新高/新低
        """
        self.min_bi_klines = min_bi_klines
        self.segment_min_bi = segment_min_bi
        self.require_new_high = require_new_high

        self._klines: pd.DataFrame = None
        self._fenxings: List[FenXing] = []
        self._bis: List[Bi] = []
        self._segments: List[XSegment] = []
        self._central_platforms: List[ZhongShu] = []
        self._trends: List[TrendType] = []

    def load_data(self, data: pd.DataFrame) -> None:
        """
        加载K线数据

        Expected columns: open, high, low, close, volume, timestamp
        """
        self._klines = data.copy()
        if 'timestamp' not in self._klines.columns and 'date' in self._klines.columns:
            self._klines['timestamp'] = pd.to_datetime(self._klines['date'])
        elif 'timestamp' not in self._klines.columns:
            self._klines['timestamp'] = self._klines.index

        self._klines = self._klines.sort_values('timestamp').reset_index(drop=True)

    def analyze(self) -> Dict:
        """
        执行完整缠论分析

        Returns:
            包含所有分析结果的字典
        """
        if self._klines is None:
            raise ValueError("请先调用 load_data 加载数据")

        self._fenxings = self._find_fenxings()
        self._bis = self._build_bi()
        self._segments = self._build_segments()
        self._central_platforms = self._find_central_platforms()
        self._trends = self._identify_trends()

        return {
            'fenxings': self._fenxings,
            'bis': self._bis,
            'segments': self._segments,
            'central_platforms': self._central_platforms,
            'trends': self._trends
        }

    def _find_fenxings(self) -> List[FenXing]:
        """
        识别分型（顶分型和底分型）

        顶分型：中间K线高点最高，两侧K线高点依次降低
        底分型：中间K线低点最低，两侧K线低点依次升高
        """
        fenxings = []
        klines = self._klines

        for i in range(1, len(klines) - 1):
            prev = klines.iloc[i - 1]
            curr = klines.iloc[i]
            next_ = klines.iloc[i + 1]

            # 顶分型识别
            if (curr['high'] > prev['high'] and curr['high'] > next_['high'] and
                curr['high'] > prev['high'] and curr['high'] > next_['high']):
                fenxings.append(FenXing(
                    index=i,
                    timestamp=klines.iloc[i]['timestamp'],
                    high=curr['high'],
                    low=curr['low'],
                    type='top'
                ))

            # 底分型识别
            elif (curr['low'] < prev['low'] and curr['low'] < next_['low'] and
                  curr['low'] < prev['low'] and curr['low'] < next_['low']):
                fenxings.append(FenXing(
                    index=i,
                    timestamp=klines.iloc[i]['timestamp'],
                    high=curr['high'],
                    low=curr['low'],
                    type='bottom'
                ))

        return fenxings

    def _build_bi(self) -> List[Bi]:
        """
        构建笔 - 从分型构建笔

        笔的构成条件：
        1. 顶分型高于底分型
        2. 中间至少min_bi_klines根K线
        3. 需要创新高或新低（可选）
        """
        if len(self._fenxings) < 2:
            return []

        bis = []
        fenxings = self._fenxings

        i = 0
        while i < len(fenxings) - 1:
            fx1 = fenxings[i]
            fx2 = fenxings[i + 1]

            # 必须是底分型后顶分型（上涨笔）或顶分型后底分型（下跌笔）
            if fx1.type == 'bottom' and fx2.type == 'top':
                direction = KLineDirection.UP
            elif fx1.type == 'top' and fx2.type == 'bottom':
                direction = KLineDirection.DOWN
            else:
                i += 1
                continue

            # 计算笔之间的K线数量
            kline_count = fx2.index - fx1.index

            # 检查是否满足最小K线数
            if kline_count < self.min_bi_klines:
                i += 1
                continue

            # 如果需要创新高/新低
            if self.require_new_high:
                if direction == KLineDirection.UP:
                    # 上涨笔：顶分型高点需创新高
                    if len(bis) > 0 and fx2.high <= bis[-1].end_fx.high:
                        i += 1
                        continue
                else:
                    # 下跌笔：底分型低点需创新低
                    if len(bis) > 0 and fx2.low >= bis[-1].end_fx.low:
                        i += 1
                        continue

            # 获取笔的高低点数组
            highs = self._klines.loc[fx1.index:fx2.index, 'high'].values
            lows = self._klines.loc[fx1.index:fx2.index, 'low'].values

            bi = Bi(
                start_fx=fx1,
                end_fx=fx2,
                direction=direction,
                kline_count=kline_count,
                highs=highs,
                lows=lows
            )
            bis.append(bi)
            i += 1

        return bis

    def _build_segments(self) -> List[XSegment]:
        """
        构建线段 - 由连续三笔构成

        线段必须由三笔构成，且方向一致
        """
        if len(self._bis) < 3:
            return []

        segments = []
        i = 0

        while i <= len(self._bis) - 3:
            bi1 = self._bis[i]
            bi2 = self._bis[i + 1]
            bi3 = self._bis[i + 2]

            # 检查三笔方向是否一致
            if bi1.direction == bi2.direction == bi3.direction:
                direction = bi1.direction

                if direction == KLineDirection.UP:
                    # 上涨线段：找最高点作为peak
                    peak = max(bi1.highs.max(), bi2.highs.max(), bi3.highs.max())
                    trough = min(bi1.lows.min(), bi2.lows.min(), bi3.lows.min())
                else:
                    # 下跌线段：找最低点作为trough
                    trough = min(bi1.lows.min(), bi2.lows.min(), bi3.lows.min())
                    peak = max(bi1.highs.max(), bi2.highs.max(), bi3.highs.max())

                seg = XSegment(
                    start_bi=bi1,
                    end_bi=bi3,
                    direction=direction,
                    peak=peak,
                    trough=trough,
                    bis=[bi1, bi2, bi3]
                )
                segments.append(seg)
                i += 2  # 重叠一笔继续构建
            else:
                i += 1

        return segments

    def _find_central_platforms(self) -> List[ZhongShu]:
        """
        识别中枢 - 线段的重叠区域

        中枢构成条件：
        1. 至少三个线段
        2. 有重叠区域（高点与低点相交）
        """
        if len(self._segments) < 3:
            return []

        platforms = []
        i = 0

        while i <= len(self._segments) - 3:
            segs = self._segments[i:i + 3]

            # 计算重叠区域
            highs = []
            lows = []

            for seg in segs:
                highs.append(seg.peak)
                lows.append(seg.trough)

            # 重叠区间
            overlap_high = max(highs)
            overlap_low = min(lows)

            # 如果有重叠
            if overlap_high >= overlap_low:
                zg = overlap_high
                zd = overlap_low
                gg = max(highs)  # 关键的上的
                dd = min(lows)   # 关键的下的

                # 检查之前的中枢是否重叠
                if platforms and self._is_overlapping(platforms[-1], zg, zd):
                    # 扩展之前的中枢
                    prev = platforms[-1]
                    prev.end_seg = segs[-1]
                    prev.high = max(prev.high, zg)
                    prev.low = min(prev.low, zd)
                    prev.zg = max(prev.zg, zg)
                    prev.zd = min(prev.zd, zd)
                else:
                    platform = ZhongShu(
                        start_seg=segs[0],
                        end_seg=segs[-1],
                        high=zg,
                        low=zd,
                        zg=zg,
                        zd=zd,
                        gg=gg,
                        dd=dd
                    )
                    platforms.append(platform)

            i += 1

        return platforms

    def _is_overlapping(self, zs: ZhongShu, zg: float, zd: float) -> bool:
        """检查两个中枢是否重叠"""
        return not (zg < zs.zd or zd > zs.zg)

    def _identify_trends(self) -> List[TrendType]:
        """
        划分走势类型

        走势类型：
        - 上涨：高点不断抬高，低点也不断抬高
        - 下跌：高点不断降低，低点也不断降低
        - 盘整：高低点无明显趋势
        """
        if len(self._segments) < 2:
            return []

        trends = []
        current_trend = None
        start_idx = 0

        for i in range(1, len(self._segments)):
            seg_prev = self._segments[i - 1]
            seg_curr = self._segments[i]

            if current_trend is None:
                if seg_curr.direction == KLineDirection.UP:
                    current_trend = TrendType(
                        type='上涨',
                        start_idx=i - 1,
                        end_idx=i,
                        zhangfu=0,
                        segments=[seg_prev, seg_curr]
                    )
                else:
                    current_trend = TrendType(
                        type='下跌',
                        start_idx=i - 1,
                        end_idx=i,
                        zhangfu=0,
                        segments=[seg_prev, seg_curr]
                    )
            else:
                # 检查趋势是否延续
                if seg_curr.direction == KLineDirection.UP:
                    if (seg_curr.peak > seg_prev.peak and
                        seg_curr.trough > seg_prev.trough):
                        current_trend.end_idx = i
                        current_trend.segments.append(seg_curr)
                    else:
                        # 趋势结束，计算涨跌幅
                        self._finalize_trend(current_trend)
                        trends.append(current_trend)
                        current_trend = None
                else:
                    if (seg_curr.peak < seg_prev.peak and
                        seg_curr.trough < seg_prev.trough):
                        current_trend.end_idx = i
                        current_trend.segments.append(seg_curr)
                    else:
                        self._finalize_trend(current_trend)
                        trends.append(current_trend)
                        current_trend = None

        if current_trend:
            self._finalize_trend(current_trend)
            trends.append(current_trend)

        return trends

    def _finalize_trend(self, trend: TrendType) -> None:
        """计算趋势涨跌幅"""
        if len(trend.segments) >= 2:
            first = trend.segments[0]
            last = trend.segments[-1]

            if trend.type == '上涨':
                start_price = first.trough
                end_price = last.peak
            else:
                start_price = first.peak
                end_price = last.trough

            if start_price != 0:
                trend.zhangfu = (end_price - start_price) / start_price * 100

    def detect_beichi(self, bi1: Bi, bi2: Bi, method: str = 'macd') -> BeiChi:
        """
        判断背驰 - 比较两段趋势的力度

        Args:
            bi1: 第一段
            bi2: 第二段
            method: 判断方法 ('macd', 'volume', 'price')

        Returns:
            BeiChi对象
        """
        if bi1.direction != bi2.direction:
            return BeiChi(exists=False, bi1=bi1, bi2=bi2, type='', indicator=method)

        if method == 'price':
            # 价格法：比较高低点的幅度
            range1 = abs(bi1.highs.max() - bi1.lows.min())
            range2 = abs(bi2.highs.max() - bi2.lows.min())

            if bi1.direction == KLineDirection.UP:
                # 顶背驰：第二笔比第一笔弱
                exists = (bi2.end_fx.high < bi1.end_fx.high and
                         range2 < range1)
                type_ = '顶背驰' if exists else ''
            else:
                # 底背驰：第二笔比第一笔弱
                exists = (bi2.end_fx.low > bi1.end_fx.low and
                         range2 < range1)
                type_ = '底背驰' if exists else ''

        elif method == 'volume':
            # 量能法：比较成交量
            # 需要volume数据，这里简化处理
            return BeiChi(exists=False, bi1=bi1, bi2=bi2, type='', indicator=method)

        else:
            # MACD法（简化）
            return BeiChi(exists=False, bi1=bi1, bi2=bi2, type='', indicator=method)

        return BeiChi(exists=exists, bi1=bi1, bi2=bi2, type=type_, indicator=method)

    def generate_signals(self) -> List[Signal]:
        """
        基于缠论生成交易信号

        Returns:
            Signal列表
        """
        signals = []

        # 简化信号逻辑：
        # 1. 底背驰后向上突破中枢 → 买入
        # 2. 顶背驰后向下突破中枢 → 卖出

        for i, seg in enumerate(self._segments):
            if i < 2:
                continue

            # 检查背驰
            bi1 = self._segments[i - 2].bis[-1] if self._segments[i - 2].bis else None
            bi2 = self._segments[i - 1].bis[-1] if self._segments[i - 1].bis else None

            if bi1 and bi2:
                beichi = self.detect_beichi(bi1, bi2)

                if beichi.exists and self._central_platforms:
                    # 检查是否突破中枢
                    for zs in self._central_platforms:
                        if seg.direction == KLineDirection.UP:
                            if seg.peak > zs.zg:
                                signals.append(Signal(
                                    symbol='',
                                    direction=SignalDirection.LONG,
                                    strength=0.8,
                                    metadata={
                                        'type': 'chan_entry',
                                        'reason': '突破中枢上轨',
                                        'beichi': beichi.type
                                    }
                                ))
                        else:
                            if seg.trough < zs.zd:
                                signals.append(Signal(
                                    symbol='',
                                    direction=SignalDirection.SHORT,
                                    strength=0.8,
                                    metadata={
                                        'type': 'chan_entry',
                                        'reason': '跌破中枢下轨',
                                        'beichi': beichi.type
                                    }
                                ))

        return signals

    def get_summary(self) -> Dict:
        """获取缠论分析摘要"""
        return {
            '分型数量': len(self._fenxings),
            '笔数量': len(self._bis),
            '线段数量': len(self._segments),
            '中枢数量': len(self._central_platforms),
            '走势段数量': len(self._trends),
            '最新走势': self._trends[-1].type if self._trends else '未知'
        }
