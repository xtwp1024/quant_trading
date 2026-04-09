# -*- coding: utf-8 -*-
"""
谐波理论 (Harmonic Pattern) Implementation

谐波理论基于艾略特波浪和斐波那契比率，识别市场中的几何价格形态。
主要形态：
- 蝴蝶型 (Butterfly)
- 螃蟹型 (Crab)
- 鲨鱼型 (Shark)
- 加特利型 (Gartley)
- 蝙蝠型 (Bat)
- 赛福型 (Cypher)

每个形态由特定的斐波那契比率定义

Reference: AbuQuant pattern recognition concepts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum

from quant_trading.signal import Signal, SignalDirection


class HarmonicPattern(Enum):
    """谐波形态类型"""
    GARTLEY = 'Gartley'
    BUTTERFLY = 'Butterfly'
    CRAB = 'Crab'
    DEEP_CRAB = 'Deep Crab'
    SHARK = 'Shark'
    CYPHER = 'Cypher'
    BAT = 'Bat'
    ALT_BAT = 'Alternate Bat'
    HALF_BAT = 'Half Bat'


@dataclass
class HarmonicPoint:
    """谐波形态的转折点"""
    index: int
    price: float
    label: str  # 'X', 'A', 'B', 'C', 'D'
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class HarmonicStructure:
    """谐波形态结构"""
    pattern: HarmonicPattern
    points: List[HarmonicPoint]  # X, A, B, C, D
    direction: str  # 'bullish' or 'bearish'
    completion_price: float  # D点价格
    reversal_price: float     # 预期反转价格
    stop_loss: float
    take_profit1: float
    take_profit2: float
    valid: bool
    fib_ratios: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # 形态置信度 0-1


@dataclass
class PatternSignal:
    """形态信号"""
    pattern: HarmonicPattern
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str


class HarmonicPatternRecognizer:
    """
    谐波形态识别器

    支持的谐波形态及其斐波那契定义：

    1. Gartley (加特利):
       - AB回撤XA的61.8%
       - BC回撤AB的38.2%或88.6%
       - CD回撤AB的127.2%或161.8%
       - AD段回撤XA的78.6%

    2. Butterfly (蝴蝶):
       - AB回撤XA的78.6%
       - BC回撤AB的38.2%或88.6%
       - CD回撤AB的161.8%或261.8%
       - D点超过X点

    3. Crab (螃蟹):
       - AB回撤XA的38.2%或61.8%
       - BC回撤AB的38.2%或88.6%
       - CD回撤AB的224%或361.8%
       - D点超过X点

    4. Bat (蝙蝠):
       - AB回撤XA的38.2%或50%
       - BC回撤AB的38.2%或88.6%
       - CD回撤AB的161.8%或268%
       - AD段回撤XA的88.6%

    5. Shark (鲨鱼):
       - AB回撤XA的113%-161.8%
       - BC回撤AB的113%-161.8%
       - CD回撤BC的161.8%
       - D点超过X点

    6. Cypher (赛福):
       - AB回撤XA的38.2%或50%
       - BC回撤AB的113%-141.4%
       - CD回撤XC的78.6%
    """

    # 各形态的斐波那契比率定义
    PATTERN_DEFINITIONS = {
        HarmonicPattern.GARTLEY: {
            'XA': [1.0],
            'AB': [0.618],
            'BC': [0.382, 0.886],
            'CD': [1.272, 1.618],
            'AD_retrace': [0.786],
            'direction': 'both'
        },
        HarmonicPattern.BUTTERFLY: {
            'XA': [1.0],
            'AB': [0.786],
            'BC': [0.382, 0.886],
            'CD': [1.618, 2.618],
            'AD_retrace': [None],  # D点超过X
            'direction': 'both'
        },
        HarmonicPattern.CRAB: {
            'XA': [1.0],
            'AB': [0.382, 0.618],
            'BC': [0.382, 0.886],
            'CD': [2.24, 3.618],
            'AD_retrace': [None],
            'direction': 'both'
        },
        HarmonicPattern.BAT: {
            'XA': [1.0],
            'AB': [0.382, 0.5],
            'BC': [0.382, 0.886],
            'CD': [1.618, 2.618],
            'AD_retrace': [0.886],
            'direction': 'both'
        },
        HarmonicPattern.SHARK: {
            'XA': [1.0],
            'AB': [1.13, 1.618],
            'BC': [1.13, 1.618],
            'CD': [1.618],
            'AD_retrace': [None],
            'direction': 'both'
        },
        HarmonicPattern.CYPHER: {
            'XA': [1.0],
            'AB': [0.382, 0.5],
            'BC': [1.13, 1.414],
            'CD': [0.786],
            'AD_retrace': [None],
            'direction': 'both'
        }
    }

    # 容许误差范围
    TOLERANCE = 0.05  # 5%容差

    def __init__(
        self,
        min_points: int = 5,
        tolerance: float = 0.05,
        require_all_cd: bool = False
    ):
        """
        Args:
            min_points: 最少转折点数
            tolerance: 斐波那契比率容差
            require_all_cd: 是否要求CD的所有比率都匹配
        """
        self.min_points = min_points
        self.tolerance = tolerance
        self.require_all_cd = require_all_cd

        self._klines: pd.DataFrame = None
        self._pivots: List[HarmonicPoint] = []
        self._patterns: List[HarmonicStructure] = []

    def load_data(self, data: pd.DataFrame) -> None:
        """加载K线数据"""
        self._klines = data.copy()
        if 'timestamp' not in self._klines.columns and 'date' in self._klines.columns:
            self._klines['timestamp'] = pd.to_datetime(self._klines['date'])
        elif 'timestamp' not in self._klines.columns:
            self._klines['timestamp'] = self._klines.index

        self._klines = self._klines.sort_values('timestamp').reset_index(drop=True)

    def analyze(self) -> Dict:
        """
        执行谐波形态分析

        Returns:
            分析结果
        """
        if self._klines is None:
            raise ValueError("请先调用 load_data 加载数据")

        self._pivots = self._find_pivots()
        self._patterns = self._find_patterns()

        return {
            'pivots': self._pivots,
            'patterns': self._patterns
        }

    def _find_pivots(self, lookback: int = 3, min_bars: int = 5) -> List[HarmonicPoint]:
        """
        找到转折点（极值点）

        Args:
            lookback: 判断极值需要的回看K线数
            min_bars: 最小间隔K线数
        """
        pivots = []
        highs = self._klines['high'].values
        lows = self._klines['low'].values

        for i in range(lookback, len(self._klines) - lookback):
            # 检查是否是高点
            is_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and highs[j] >= highs[i]:
                    is_high = False
                    break

            if is_high:
                # 检查与前一个高点的间隔
                if not pivots or (i - pivots[-1].index) >= min_bars:
                    pivots.append(HarmonicPoint(
                        index=i,
                        price=highs[i],
                        label='',
                        timestamp=self._klines.iloc[i]['timestamp']
                    ))
                elif pivots and highs[i] > pivots[-1].price:
                    # 替换前一个高点
                    pivots[-1] = HarmonicPoint(
                        index=i,
                        price=highs[i],
                        label='',
                        timestamp=self._klines.iloc[i]['timestamp']
                    )
            else:
                # 检查是否是低点
                is_low = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and lows[j] <= lows[i]:
                        is_low = False
                        break

                if is_low:
                    if not pivots or (i - pivots[-1].index) >= min_bars:
                        pivots.append(HarmonicPoint(
                            index=i,
                            price=lows[i],
                            label='',
                            timestamp=self._klines.iloc[i]['timestamp']
                        ))
                    elif pivots and lows[i] < pivots[-1].price:
                        # 替换前一个低点
                        pivots[-1] = HarmonicPoint(
                            index=i,
                            price=lows[i],
                            label='',
                            timestamp=self._klines.iloc[i]['timestamp']
                        )

        return pivots

    def _calculate_ratio(self, point1: float, point2: float, base: float) -> float:
        """计算斐波那契比率"""
        if base == 0:
            return 0
        return abs(point2 - point1) / base

    def _ratio_match(self, ratio: float, target: float) -> bool:
        """检查比率是否匹配目标值（在容差范围内）"""
        return abs(ratio - target) <= self.tolerance

    def _find_patterns(self) -> List[HarmonicStructure]:
        """
        识别谐波形态

        寻找XABCD结构的形态
        """
        patterns = []

        if len(self._pivots) < 5:
            return patterns

        # 尝试各种形态组合
        for pattern_type, definition in self.PATTERN_DEFINITIONS.items():
            patterns.extend(self._find_pattern_type(pattern_type, definition))

        # 按置信度排序
        patterns.sort(key=lambda x: x.confidence, reverse=True)

        return patterns

    def _find_pattern_type(
        self,
        pattern_type: HarmonicPattern,
        definition: Dict
    ) -> List[HarmonicStructure]:
        """查找特定类型的谐波形态"""
        patterns = []

        i = 0
        while i <= len(self._pivots) - 5:
            # 尝试以XABCD形式组合
            try:
                x = self._pivots[i]
                a = self._pivots[i + 1]
                b = self._pivots[i + 2]
                c = self._pivots[i + 3]
                d = self._pivots[i + 4]

                # 判断方向
                is_bullish = a.price > x.price and b.price < a.price and c.price > b.price
                is_bearish = a.price < x.price and b.price > a.price and c.price < b.price

                if not (is_bullish or is_bearish):
                    i += 1
                    continue

                direction = 'bullish' if is_bullish else 'bearish'

                # 计算XA距离（作为基准）
                xa_distance = abs(a.price - x.price)
                if xa_distance == 0:
                    i += 1
                    continue

                # 计算AB回撤
                ab_distance = abs(b.price - a.price)
                ab_ratio = ab_distance / xa_distance

                # 计算BC回撤
                bc_distance = abs(c.price - b.price)
                bc_ratio = bc_distance / xa_distance

                # 计算CD
                cd_distance = abs(d.price - c.price)
                cd_ratio = cd_distance / xa_distance

                # 计算AD回撤
                ad_distance = abs(d.price - x.price)
                ad_ratio = ad_distance / xa_distance

                # 验证各段比率
                fib_ratios = {
                    'AB': ab_ratio,
                    'BC': bc_ratio,
                    'CD': cd_ratio,
                    'AD': ad_ratio
                }

                valid, confidence, match_details = self._validate_pattern(
                    pattern_type, definition, fib_ratios, direction
                )

                if valid:
                    # 计算止盈止损
                    entry_price = d.price
                    stop_loss = self._calculate_stop_loss(x, d, direction, pattern_type)

                    if direction == 'bullish':
                        take_profit1 = d.price + xa_distance * 0.618
                        take_profit2 = d.price + xa_distance * 1.0
                    else:
                        take_profit1 = d.price - xa_distance * 0.618
                        take_profit2 = d.price - xa_distance * 1.0

                    pattern = HarmonicStructure(
                        pattern=pattern_type,
                        points=[x, a, b, c, d],
                        direction=direction,
                        completion_price=d.price,
                        reversal_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit1=take_profit1,
                        take_profit2=take_profit2,
                        valid=True,
                        fib_ratios=fib_ratios,
                        confidence=confidence
                    )
                    patterns.append(pattern)

            except (IndexError, KeyError):
                pass

            i += 1

        return patterns

    def _validate_pattern(
        self,
        pattern_type: HarmonicPattern,
        definition: Dict,
        fib_ratios: Dict[str, float],
        direction: str
    ) -> Tuple[bool, float, Dict]:
        """
        验证是否符合特定形态的斐波那契比率要求

        Returns:
            (是否有效, 置信度, 匹配详情)
        """
        valid_count = 0
        total_checks = 0
        match_details = {}

        # 检查AB回撤
        ab_target = definition['AB'][0]
        ab_ratio = fib_ratios['AB']

        total_checks += 1
        if self._ratio_match(ab_ratio, ab_target):
            valid_count += 1
            match_details['AB'] = True
        else:
            match_details['AB'] = False

        # 检查BC回撤
        bc_matches = [self._ratio_match(fib_ratios['BC'], target) for target in definition['BC']]
        total_checks += 1
        if any(bc_matches):
            valid_count += 1
            match_details['BC'] = True
        else:
            match_details['BC'] = False

        # 检查CD
        if self.require_all_cd:
            cd_matches = [self._ratio_match(fib_ratios['CD'], target) for target in definition['CD']]
            total_checks += 1
            if any(cd_matches):
                valid_count += 1
                match_details['CD'] = True
            else:
                match_details['CD'] = False
        else:
            # CD只要有一个匹配即可
            cd_matches = [self._ratio_match(fib_ratios['CD'], target) for target in definition['CD']]
            if any(cd_matches):
                valid_count += 1
            match_details['CD'] = any(cd_matches)
            total_checks += 1

        # 检查AD回撤
        ad_retrace = definition.get('AD_retrace', [None])[0]
        if ad_retrace is not None:
            total_checks += 1
            if self._ratio_match(fib_ratios['AD'], ad_retrace):
                valid_count += 1
                match_details['AD'] = True
            else:
                match_details['AD'] = False
        else:
            # D点应该超过X点
            if direction == 'bullish':
                is_valid_ad = fib_ratios.get('AD', 0) > 1.0
            else:
                is_valid_ad = fib_ratios.get('AD', 0) > 1.0

            match_details['AD'] = is_valid_ad
            total_checks += 1
            if is_valid_ad:
                valid_count += 1

        confidence = valid_count / total_checks if total_checks > 0 else 0
        is_valid = valid_count >= total_checks * 0.75  # 至少75%匹配

        return is_valid, confidence, match_details

    def _calculate_stop_loss(
        self,
        x: HarmonicPoint,
        d: HarmonicPoint,
        direction: str,
        pattern_type: HarmonicPattern
    ) -> float:
        """计算止损位"""
        xa_distance = abs(d.price - x.price)

        # 根据形态类型调整止损
        stop_distance = xa_distance * 0.5  # 默认止损为XA距离的50%

        if direction == 'bullish':
            return d.price - stop_distance
        else:
            return d.price + stop_distance

    def get_latest_pattern(self) -> Optional[HarmonicStructure]:
        """获取最新的有效形态"""
        if not self._patterns:
            return None

        for pattern in self._patterns:
            if pattern.valid:
                return pattern

        return None

    def get_patterns_by_type(self, pattern_type: HarmonicPattern) -> List[HarmonicStructure]:
        """获取指定类型的形态"""
        return [p for p in self._patterns if p.pattern == pattern_type]

    def generate_signals(self) -> List[Signal]:
        """
        基于谐波形态生成交易信号

        Returns:
            Signal列表
        """
        signals = []

        latest_pattern = self.get_latest_pattern()

        if latest_pattern and latest_pattern.valid:
            if latest_pattern.direction == 'bullish':
                signals.append(Signal(
                    symbol='',
                    direction=SignalDirection.LONG,
                    strength=latest_pattern.confidence,
                    price=latest_pattern.completion_price,
                    stop_loss=latest_pattern.stop_loss,
                    take_profit=latest_pattern.take_profit1,
                    metadata={
                        'type': 'harmonic',
                        'pattern': latest_pattern.pattern.value,
                        'reason': f'{latest_pattern.pattern.value}看涨形态完成'
                    }
                ))
            else:
                signals.append(Signal(
                    symbol='',
                    direction=SignalDirection.SHORT,
                    strength=latest_pattern.confidence,
                    price=latest_pattern.completion_price,
                    stop_loss=latest_pattern.stop_loss,
                    take_profit=latest_pattern.take_profit1,
                    metadata={
                        'type': 'harmonic',
                        'pattern': latest_pattern.pattern.value,
                        'reason': f'{latest_pattern.pattern.value}看跌形态完成'
                    }
                ))

        return signals

    def get_summary(self) -> Dict:
        """获取谐波形态分析摘要"""
        pattern_counts = {}
        for p in self._patterns:
            pattern_counts[p.pattern.value] = pattern_counts.get(p.pattern.value, 0) + 1

        latest = self.get_latest_pattern()

        return {
            '识别的形态数量': len(self._patterns),
            '有效形态数量': sum(1 for p in self._patterns if p.valid),
            '形态分布': pattern_counts,
            '最新形态': f"{latest.pattern.value} ({latest.direction})" if latest else '无',
            '置信度': f"{latest.confidence:.2%}" if latest else 'N/A'
        }
