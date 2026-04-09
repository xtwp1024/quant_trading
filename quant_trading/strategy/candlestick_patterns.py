# -*- coding: utf-8 -*-
"""
K线形态识别 (Candlestick Pattern Recognition)

基于日本蜡烛图理论，识别常见的反转和持续形态。

主要形态分类：
1. 单根K线形态：
   - 锤子线 (Hammer)
   - 上吊线 (Hanging Man)
   - 流星线 (Shooting Star)
   - 倒锤子线 (Inverted Hammer)

2. 双根K线形态：
   - 吞没形态 (Engulfing)
   - 乌云盖顶 (Dark Cloud Cover)
   - 刺透形态 (Piercing Line)
   - 孕线 (Harami)

3. 三根K线形态：
   - 早晨之星 (Morning Star)
   - 黄昏之星 (Evening Star)
   - 三乌鸦 (Three Black Crows)
   - 三白兵 (Three White Soldiers)

Reference: AbuQuant ABuKLUtil, candlestick analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

from quant_trading.signal import Signal, SignalDirection, SignalType


class CandlePattern(Enum):
    """K线形态类型"""

    # 单根形态
    HAMMER = 'Hammer'                       # 锤子线
    HANGING_MAN = 'Hanging Man'             # 上吊线
    SHOOTING_STAR = 'Shooting Star'        # 流星线
    INVERTED_HAMMER = 'Inverted Hammer'     # 倒锤子线
    MARUBOZU = 'Marubozu'                   # 光头光脚K线
    DOJI = 'Doji'                           # 十字星

    # 双根形态
    BULLISH_ENGULFING = 'Bullish Engulfing'     # 看涨吞没
    BEARISH_ENGULFING = 'Bearish Engulfing'     # 看跌吞没
    DARK_CLOUD_COVER = 'Dark Cloud Cover'       # 乌云盖顶
    PIERCING_LINE = 'Piercing Line'             # 刺透形态
    BULLISH_HARAMI = 'Bullish Harami'           # 看涨孕线
    BEARISH_HARAMI = 'Bearish Harami'           # 看跌孕线

    # 三根形态
    MORNING_STAR = 'Morning Star'               # 早晨之星
    EVENING_STAR = 'Evening Star'               # 黄昏之星
    THREE_WHITE_SOLDIERS = 'Three White Soldiers'  # 三白兵
    THREE_BLACK_CROWS = 'Three Black Crows'     # 三乌鸦
    THREE_METHODS = 'Three Methods'             # 三法形态

    # 持续形态
    SPINNING_TOP = 'Spinning Top'               # 纺锤线
    RISING_THREE_METHODS = 'Rising Three Methods'   # 上升三法
    FALLING_THREE_METHODS = 'Falling Three Methods'  # 下降三法


@dataclass
class CandlePatternResult:
    """K线形态识别结果"""
    pattern: CandlePattern
    bullish: bool       # 是否为看涨形态
    strength: float     # 置信度 0-1
    start_idx: int      # 起始索引
    end_idx: int        # 结束索引
    description: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class CandleStatistics:
    """K线统计信息"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    body: float          # 实体大小
    upper_shadow: float  # 上影线
    lower_shadow: float  # 下影线
    body_ratio: float    # 实体占比
    full_range: float    # 整体振幅


class CandlestickPatternAnalyzer:
    """
    K线形态分析器

    功能：
    1. 识别单根、双根、三根K线形态
    2. 计算形态强度
    3. 生成交易信号
    """

    def __init__(
        self,
        body_ratio_threshold: float = 0.6,
        shadow_ratio_threshold: float = 0.3
    ):
        """
        Args:
            body_ratio_threshold: 实体占比阈值（判定是否为光头光脚）
            shadow_ratio_threshold: 影线占比阈值
        """
        self.body_ratio_threshold = body_ratio_threshold
        self.shadow_ratio_threshold = shadow_ratio_threshold

        self._klines: pd.DataFrame = None
        self._patterns: List[CandlePatternResult] = []
        self._stats: List[CandleStatistics] = []

    def load_data(self, data: pd.DataFrame) -> None:
        """加载K线数据"""
        self._klines = data.copy()
        if 'timestamp' not in self._klines.columns and 'date' in self._klines.columns:
            self._klines['timestamp'] = pd.to_datetime(self._klines['date'])
        elif 'timestamp' not in self._klines.columns:
            self._klines['timestamp'] = self._klines.index

        self._klines = self._klines.sort_values('timestamp').reset_index(drop=True)
        self._calculate_statistics()

    def _calculate_statistics(self) -> None:
        """计算每根K线的统计信息"""
        self._stats = []

        for i in range(len(self._klines)):
            row = self._klines.iloc[i]

            candle_open = float(row['open'])
            candle_high = float(row['high'])
            candle_low = float(row['low'])
            candle_close = float(row['close'])
            volume = float(row.get('volume', 0))

            # 计算实体
            body = abs(candle_close - candle_open)

            # 计算影线
            if candle_close >= candle_open:
                # 阳线
                upper_shadow = candle_high - candle_close
                lower_shadow = candle_open - candle_low
            else:
                # 阴线
                upper_shadow = candle_high - candle_open
                lower_shadow = candle_close - candle_low

            # 计算整体范围
            full_range = candle_high - candle_low

            # 计算实体占比
            body_ratio = body / full_range if full_range > 0 else 0

            self._stats.append(CandleStatistics(
                open=candle_open,
                high=candle_high,
                low=candle_low,
                close=candle_close,
                volume=volume,
                body=body,
                upper_shadow=upper_shadow,
                lower_shadow=lower_shadow,
                body_ratio=body_ratio,
                full_range=full_range
            ))

    def analyze(self) -> Dict:
        """
        执行K线形态分析

        Returns:
            分析结果
        """
        if self._klines is None:
            raise ValueError("请先调用 load_data 加载数据")

        self._patterns = []

        # 单根形态
        self._find_single_candle_patterns()

        # 双根形态
        self._find_double_candle_patterns()

        # 三根形态
        self._find_triple_candle_patterns()

        return {
            'patterns': self._patterns,
            'total_patterns': len(self._patterns),
            'bullish_count': sum(1 for p in self._patterns if p.bullish),
            'bearish_count': sum(1 for p in self._patterns if not p.bullish)
        }

    def _find_single_candle_patterns(self) -> None:
        """识别单根K线形态"""
        for i in range(len(self._stats)):
            stat = self._stats[i]

            # 锤子线/上吊线
            if self._is_hammer_like(stat):
                is_bullish = stat.body > 0 and stat.close > stat.open
                pattern_type = CandlePattern.HAMMER if is_bullish else CandlePattern.HANGING_MAN

                self._patterns.append(CandlePatternResult(
                    pattern=pattern_type,
                    bullish=is_bullish,
                    strength=self._calculate_hammer_strength(stat),
                    start_idx=i,
                    end_idx=i,
                    description=f"{'锤子线' if is_bullish else '上吊线'}: 实体小，下影线长"
                ))

            # 流星线/倒锤子线
            if self._is_shooting_star_like(stat):
                is_bullish = stat.body < 0
                pattern_type = CandlePattern.SHOOTING_STAR if not is_bullish else CandlePattern.INVERTED_HAMMER

                self._patterns.append(CandlePatternResult(
                    pattern=pattern_type,
                    bullish=is_bullish,
                    strength=self._calculate_shooting_star_strength(stat),
                    start_idx=i,
                    end_idx=i,
                    description=f"{'流星线' if not is_bullish else '倒锤子线'}: 实体小，上影线长"
                ))

            # 十字星
            if self._is_doji(stat):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.DOJI,
                    bullish=False,
                    strength=self._calculate_doji_strength(stat),
                    start_idx=i,
                    end_idx=i,
                    description="十字星: 开收盘价接近"
                ))

            # 光头光脚
            if self._is_marubozu(stat):
                is_bullish = stat.close > stat.open
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.MARUBOZU,
                    bullish=is_bullish,
                    strength=stat.body_ratio,
                    start_idx=i,
                    end_idx=i,
                    description=f"{'光头光脚阳线' if is_bullish else '光头光脚阴线'}"
                ))

            # 纺锤线
            if self._is_spinning_top(stat):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.SPINNING_TOP,
                    bullish=False,
                    strength=0.5,
                    start_idx=i,
                    end_idx=i,
                    description="纺锤线: 实体小，上下影线适中"
                ))

    def _is_hammer_like(self, stat: CandleStatistics) -> bool:
        """判断是否为锤子线/上吊线形态"""
        if stat.full_range == 0:
            return False

        # 下影线至少是实体的2倍
        lower_ratio = stat.lower_shadow / stat.full_range
        upper_ratio = stat.upper_shadow / stat.full_range

        # 实体较小
        body_ok = stat.body_ratio < 0.4

        # 下影线长，上影线短
        return lower_ratio > 0.4 and upper_ratio < 0.15 and body_ok

    def _is_shooting_star_like(self, stat: CandleStatistics) -> bool:
        """判断是否为流星线/倒锤子线形态"""
        if stat.full_range == 0:
            return False

        upper_ratio = stat.upper_shadow / stat.full_range
        lower_ratio = stat.lower_shadow / stat.full_range

        # 上影线至少是实体的2倍
        body_ok = stat.body_ratio < 0.4

        # 上影线长，下影线短
        return upper_ratio > 0.4 and lower_ratio < 0.15 and body_ok

    def _is_doji(self, stat: CandleStatistics) -> bool:
        """判断是否为十字星"""
        if stat.full_range == 0 or stat.body == 0:
            return False

        # 实体占整体范围小于5%
        return stat.body_ratio < 0.05

    def _is_marubozu(self, stat: CandleStatistics) -> bool:
        """判断是否为光头光脚K线"""
        if stat.full_range == 0:
            return False

        # 实体占比大于阈值，且几乎没有影线
        return (stat.body_ratio > self.body_ratio_threshold and
                stat.upper_shadow / stat.full_range < 0.02 and
                stat.lower_shadow / stat.full_range < 0.02)

    def _is_spinning_top(self, stat: CandleStatistics) -> bool:
        """判断是否为纺锤线"""
        if stat.full_range == 0:
            return False

        # 实体小，上下影线差不多长
        body_ok = 0.1 < stat.body_ratio < 0.4
        shadow_balanced = (0.2 < stat.upper_shadow / stat.full_range < 0.4 and
                          0.2 < stat.lower_shadow / stat.full_range < 0.4)

        return body_ok and shadow_balanced

    def _calculate_hammer_strength(self, stat: CandleStatistics) -> float:
        """计算锤子线强度"""
        if stat.full_range == 0:
            return 0

        strength = 0.5

        # 下影线越长越强
        lower_ratio = stat.lower_shadow / stat.full_range
        strength += lower_ratio * 0.3

        # 实体越小越强
        strength += (1 - stat.body_ratio) * 0.2

        return min(1.0, strength)

    def _calculate_shooting_star_strength(self, stat: CandleStatistics) -> float:
        """计算流星线强度"""
        if stat.full_range == 0:
            return 0

        strength = 0.5

        # 上影线越长越强
        upper_ratio = stat.upper_shadow / stat.full_range
        strength += upper_ratio * 0.3

        # 实体越小越强
        strength += (1 - stat.body_ratio) * 0.2

        return min(1.0, strength)

    def _calculate_doji_strength(self, stat: CandleStatistics) -> float:
        """计算十字星强度"""
        if stat.full_range == 0:
            return 0

        # 实体越小越强
        return 1.0 - stat.body_ratio * 10

    def _find_double_candle_patterns(self) -> None:
        """识别双根K线形态"""
        for i in range(len(self._stats) - 1):
            first = self._stats[i]
            second = self._stats[i + 1]

            # 看涨吞没
            if self._is_bullish_engulfing(first, second):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.BULLISH_ENGULFING,
                    bullish=True,
                    strength=self._calculate_engulfing_strength(first, second),
                    start_idx=i,
                    end_idx=i + 1,
                    description="看涨吞没: 阴线后出现大阳线完全包裹"
                ))

            # 看跌吞没
            if self._is_bearish_engulfing(first, second):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.BEARISH_ENGULFING,
                    bullish=False,
                    strength=self._calculate_engulfing_strength(second, first),
                    start_idx=i,
                    end_idx=i + 1,
                    description="看跌吞没: 阳线后出现大阴线完全包裹"
                ))

            # 乌云盖顶
            if self._is_dark_cloud_cover(first, second):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.DARK_CLOUD_COVER,
                    bullish=False,
                    strength=self._calculate_cloud_cover_strength(first, second),
                    start_idx=i,
                    end_idx=i + 1,
                    description="乌云盖顶: 阳线后出现开盘高于前收的阴线"
                ))

            # 刺透形态
            if self._is_piercing_line(first, second):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.PIERCING_LINE,
                    bullish=True,
                    strength=self._calculate_cloud_cover_strength(second, first),
                    start_idx=i,
                    end_idx=i + 1,
                    description="刺透形态: 阴线后出现开盘低于前收的阳线"
                ))

            # 孕线
            if self._is_harami(first, second):
                is_bullish = second.close > second.open
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.BULLISH_HARAMI if is_bullish else CandlePattern.BEARISH_HARAMI,
                    bullish=is_bullish,
                    strength=0.5,
                    start_idx=i,
                    end_idx=i + 1,
                    description=f"{'看涨' if is_bullish else '看跌'}孕线: 第二根K线实体在第一根范围内"
                ))

    def _is_bullish_engulfing(self, first: CandleStatistics, second: CandleStatistics) -> bool:
        """看涨吞没: 第一根阴线，第二根阳线且完全包裹"""
        is_first_bearish = first.close < first.open
        is_second_bullish = second.close > second.open

        if not (is_first_bearish and is_second_bullish):
            return False

        # 第二根完全包裹第一根
        return (second.low <= first.low and second.high >= first.high and
                second.body > first.body)

    def _is_bearish_engulfing(self, first: CandleStatistics, second: CandleStatistics) -> bool:
        """看跌吞没: 第一根阳线，第二根阴线且完全包裹"""
        is_first_bullish = first.close > first.open
        is_second_bearish = second.close < second.open

        if not (is_first_bullish and is_second_bearish):
            return False

        return (second.low <= first.low and second.high >= first.high and
                second.body > first.body)

    def _is_dark_cloud_cover(self, first: CandleStatistics, second: CandleStatistics) -> bool:
        """乌云盖顶"""
        is_first_bullish = first.close > first.open
        is_second_bearish = second.close < second.open

        if not (is_first_bullish and is_second_bearish):
            return False

        # 第二根开盘高于第一根收盘，收盘跌破第一根实体中点
        return (second.open > first.close and
                second.close < (first.open + first.close) / 2)

    def _is_piercing_line(self, first: CandleStatistics, second: CandleStatistics) -> bool:
        """刺透形态"""
        is_first_bearish = first.close < first.open
        is_second_bullish = second.close > second.open

        if not (is_first_bearish and is_second_bullish):
            return False

        # 第二根开盘低于第一根收盘，收盘涨入第一根实体中点以上
        return (second.open < first.close and
                second.close > (first.open + first.close) / 2)

    def _is_harami(self, first: CandleStatistics, second: CandleStatistics) -> bool:
        """孕线: 第二根K线实体在第一根范围内"""
        return (second.high < first.high and second.low > first.low and
                second.body < first.body)

    def _calculate_engulfing_strength(self, inside: CandleStatistics, outside: CandleStatistics) -> float:
        """计算吞没形态强度"""
        strength = 0.6

        # 吞没程度
        range_ratio = outside.body / inside.body if inside.body > 0 else 1
        strength += min(0.2, range_ratio * 0.1)

        # 影线确认
        if inside.lower_shadow > inside.body * 0.5:
            strength += 0.1

        return min(1.0, strength)

    def _calculate_cloud_cover_strength(self, first: CandleStatistics, second: CandleStatistics) -> float:
        """计算乌云盖顶/刺透形态强度"""
        strength = 0.5

        # 穿透程度
        if first.body > 0:
            penetration = abs(second.close - first.close) / first.body
            strength += min(0.3, penetration * 0.2)

        return min(1.0, strength)

    def _find_triple_candle_patterns(self) -> None:
        """识别三根K线形态"""
        for i in range(len(self._stats) - 2):
            first = self._stats[i]
            second = self._stats[i + 1]
            third = self._stats[i + 2]

            # 早晨之星
            if self._is_morning_star(first, second, third):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.MORNING_STAR,
                    bullish=True,
                    strength=self._calculate_star_strength(second, third),
                    start_idx=i,
                    end_idx=i + 2,
                    description="早晨之星: 连续下跌后出现十字星，然后上涨"
                ))

            # 黄昏之星
            if self._is_evening_star(first, second, third):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.EVENING_STAR,
                    bullish=False,
                    strength=self._calculate_star_strength(second, third),
                    start_idx=i,
                    end_idx=i + 2,
                    description="黄昏之星: 连续上涨后出现十字星，然后下跌"
                ))

            # 三白兵
            if self._is_three_white_soldiers(first, second, third):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.THREE_WHITE_SOLDIERS,
                    bullish=True,
                    strength=0.85,
                    start_idx=i,
                    end_idx=i + 2,
                    description="三白兵: 三根连续上涨的中到大阳线"
                ))

            # 三乌鸦
            if self._is_three_black_crows(first, second, third):
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.THREE_BLACK_CROWS,
                    bullish=False,
                    strength=0.85,
                    start_idx=i,
                    end_idx=i + 2,
                    description="三乌鸦: 三根连续下跌的中到大阴线"
                ))

            # 三法形态
            if self._is_three_methods(first, second, third):
                is_rising = third.close > first.open
                self._patterns.append(CandlePatternResult(
                    pattern=CandlePattern.RISING_THREE_METHODS if is_rising else CandlePattern.FALLING_THREE_METHODS,
                    bullish=is_rising,
                    strength=0.7,
                    start_idx=i,
                    end_idx=i + 2,
                    description=f"{'上升' if is_rising else '下降'}三法"
                ))

    def _is_morning_star(self, first: CandleStatistics, second: CandleStatistics, third: CandleStatistics) -> bool:
        """早晨之星"""
        # 第一根：大阴线
        is_first_bearish = first.close < first.open and first.body_ratio > 0.5

        # 第二根：实体小（十字星或纺锤）
        is_second_small = second.body_ratio < 0.3

        # 第三根：大阳线
        is_third_bullish = third.close > third.open and third.body_ratio > 0.5

        # 第三根收盘高于第一根实体中点
        third_rises = third.close > (first.open + first.close) / 2

        return is_first_bearish and is_second_small and is_third_bullish and third_rises

    def _is_evening_star(self, first: CandleStatistics, second: CandleStatistics, third: CandleStatistics) -> bool:
        """黄昏之星"""
        # 第一根：大阳线
        is_first_bullish = first.close > first.open and first.body_ratio > 0.5

        # 第二根：实体小
        is_second_small = second.body_ratio < 0.3

        # 第三根：大阴线
        is_third_bearish = third.close < third.open and third.body_ratio > 0.5

        # 第三根收盘低于第一根实体中点
        third_falls = third.close < (first.open + first.close) / 2

        return is_first_bullish and is_second_small and is_third_bearish and third_falls

    def _is_three_white_soldiers(self, first: CandleStatistics, second: CandleStatistics,
                                 third: CandleStatistics) -> bool:
        """三白兵"""
        # 三根阳线
        if not (first.close > first.open and second.close > second.open and third.close > third.open):
            return False

        # 实体较大
        if not (first.body_ratio > 0.5 and second.body_ratio > 0.5 and third.body_ratio > 0.5):
            return False

        # 依次上涨
        return (second.close > first.close and third.close > second.close and
                # 影线短
                first.upper_shadow / first.full_range < 0.2 and
                second.upper_shadow / second.full_range < 0.2 and
                third.upper_shadow / third.full_range < 0.2)

    def _is_three_black_crows(self, first: CandleStatistics, second: CandleStatistics,
                              third: CandleStatistics) -> bool:
        """三乌鸦"""
        # 三根阴线
        if not (first.close < first.open and second.close < second.open and third.close < third.open):
            return False

        # 实体较大
        if not (first.body_ratio > 0.5 and second.body_ratio > 0.5 and third.body_ratio > 0.5):
            return False

        # 依次下跌
        return (second.close < first.close and third.close < second.close and
                first.lower_shadow / first.full_range < 0.2 and
                second.lower_shadow / second.full_range < 0.2 and
                third.lower_shadow / third.full_range < 0.2)

    def _is_three_methods(self, first: CandleStatistics, second: CandleStatistics,
                          third: CandleStatistics) -> bool:
        """三法形态"""
        # 第一根和第三根同方向，实体较大
        same_direction = ((first.close > first.open and third.close > third.open) or
                         (first.close < first.open and third.close < third.open))

        first_large = first.body_ratio > 0.5
        third_large = third.body_ratio > 0.5

        # 第二根与第一根方向相反，实体小
        second_small = second.body_ratio < 0.4
        opposite_direction = ((first.close > first.open and second.close < second.open) or
                              (first.close < first.open and second.close > second.open))

        return same_direction and first_large and third_large and second_small and opposite_direction

    def _calculate_star_strength(self, middle: CandleStatistics, third: CandleStatistics) -> float:
        """计算星形态强度"""
        strength = 0.5

        # 中间K线实体越小越强
        strength += (1 - middle.body_ratio) * 0.3

        # 第三根确认力度
        strength += third.body_ratio * 0.2

        return min(1.0, strength)

    def get_latest_patterns(self, n: int = 5) -> List[CandlePatternResult]:
        """获取最近N个形态"""
        return self._patterns[-n:] if len(self._patterns) >= n else self._patterns

    def get_patterns_by_type(self, pattern: CandlePattern) -> List[CandlePatternResult]:
        """获取指定类型的形态"""
        return [p for p in self._patterns if p.pattern == pattern]

    def generate_signals(self) -> List[Signal]:
        """
        基于K线形态生成交易信号

        Returns:
            Signal列表
        """
        signals = []

        # 获取最近的有效形态
        recent_patterns = self.get_latest_patterns(10)

        for pattern in recent_patterns:
            if pattern.strength < 0.6:
                continue

            signal_type = SignalType.ENTRY if pattern.bullish else SignalType.EXIT

            if pattern.bullish:
                signals.append(Signal(
                    symbol='',
                    direction=SignalDirection.LONG,
                    strength=pattern.strength,
                    metadata={
                        'type': 'candlestick',
                        'pattern': pattern.pattern.value,
                        'reason': pattern.description
                    }
                ))
            else:
                signals.append(Signal(
                    symbol='',
                    direction=SignalDirection.SHORT,
                    strength=pattern.strength,
                    metadata={
                        'type': 'candlestick',
                        'pattern': pattern.pattern.value,
                        'reason': pattern.description
                    }
                ))

        return signals

    def get_summary(self) -> Dict:
        """获取K线形态分析摘要"""
        pattern_counts = {}
        for p in self._patterns:
            pattern_counts[p.pattern.value] = pattern_counts.get(p.pattern.value, 0) + 1

        bullish = sum(1 for p in self._patterns if p.bullish)
        bearish = sum(1 for p in self._patterns if not p.bullish)

        return {
            '识别的形态数量': len(self._patterns),
            '看涨形态': bullish,
            '看跌形态': bearish,
            '形态分布': pattern_counts,
            '最近形态': self._patterns[-1].pattern.value if self._patterns else '无'
        }
