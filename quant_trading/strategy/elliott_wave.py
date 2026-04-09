# -*- coding: utf-8 -*-
"""
艾略特波浪理论 (Elliott Wave Theory) Implementation

艾略特波浪理论由拉尔夫·纳尔逊·艾略特提出，认为市场价格走势遵循
五浪上涨/三浪下跌的规律。

波浪结构：
- 推动浪(Impulse): 1, 2, 3, 4, 5
- 调整浪(Correction): A, B, C

原则：
1. 浪2不能回撤浪1的100%以上
2. 浪3不能是最短的推动浪
3. 浪4不能与浪1重叠
4. 交替原则：浪2和浪4的形态交替

Reference: AbuQuant TLineBu/ABuTLWave.py, Wave analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum

from quant_trading.signal import Signal, SignalDirection


class WaveDegree(Enum):
    """波浪级别"""
    GRAND_SUPERCYCLE = 'Grand Supercycle'    # 超过百年的周期
    SUPERCYCLE = 'Supercycle'                # 50-100年
    CYCLE = 'Cycle'                           # 1-10年
    PRIMARY = 'Primary'                       # 数月到数年
    INTERMEDIATE = 'Intermediate'             # 数周到数月
    MINOR = 'Minor'                           # 数周
    MINUTE = 'Minute'                         # 数天
    MINUTTE = 'Minutte'                       # 数小时
    SUBMINUTTE = 'Subminutte'                 # 更短的周期


class WaveType(Enum):
    """波浪类型"""
    IMPULSE = 'Impulse'           # 推动浪
    CORRECTION = 'Correction'      # 调整浪
    DIAGONAL = 'Diagonal'          # 倾斜三角型


@dataclass
class Wave:
    """波浪"""
    number: int           # 波浪编号 (1-5 或 A-C)
    wave_type: WaveType   # 波浪类型
    start_idx: int        # 起始索引
    end_idx: int          # 结束索引
    start_price: float    # 起始价格
    end_price: float      # 结束价格
    high: float           # 最高点
    low: float            # 最低点
    sub_waves: List['Wave'] = field(default_factory=list)
    degree: WaveDegree = WaveDegree.PRIMARY
    fib_ratio: Optional[float] = None  # 斐波那契比率


@dataclass
class WaveStructure:
    """波浪结构"""
    waves: List[Wave]
    complete: bool           # 是否完整
    direction: str           # 'up' or 'down'
    degree: WaveDegree
    start_idx: int
    end_idx: int


@dataclass
class FibonacciLevel:
    """斐波那契回撤/扩展水平"""
    level: float
    price: float
    type: str  # 'retracement' or 'extension'


class ElliottWaveAnalyzer:
    """
    艾略特波浪分析器

    主要功能：
    1. 识别波浪结构
    2. 验证波浪规则
    3. 斐波那契比率分析
    4. 生成交易信号
    """

    # 斐波那契比率
    FIB_RATIOS = {
        'retracement': [0.236, 0.382, 0.500, 0.618, 0.764, 0.786, 1.0, 1.272, 1.618, 2.618],
        'extension': [0.618, 1.0, 1.272, 1.618, 2.0, 2.618, 3.618, 4.618]
    }

    def __init__(
        self,
        min_wave_size: int = 5,
        strict_mode: bool = True
    ):
        """
        Args:
            min_wave_size: 波浪最小K线数
            strict_mode: 严格模式，强制执行艾略特规则
        """
        self.min_wave_size = min_wave_size
        self.strict_mode = strict_mode

        self._klines: pd.DataFrame = None
        self._waves: List[Wave] = []
        self._structures: List[WaveStructure] = []
        self._pivots: List[Dict] = []

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
        执行完整艾略特波浪分析

        Returns:
            分析结果字典
        """
        if self._klines is None:
            raise ValueError("请先调用 load_data 加载数据")

        self._pivots = self._find_pivots()
        self._waves = self._identify_waves()
        self._structures = self._build_structures()

        return {
            'pivots': self._pivots,
            'waves': self._waves,
            'structures': self._structures
        }

    def _find_pivots(self, lookback: int = 3) -> List[Dict]:
        """
        找到关键的转折点（极值点）

        Args:
            lookback: 判断极值需要的前后K线数
        """
        pivots = []
        highs = self._klines['high'].values
        lows = self._klines['low'].values
        closes = self._klines['close'].values

        for i in range(lookback, len(self._klines) - lookback):
            # 检查是否是高点
            is_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and highs[j] >= highs[i]:
                    is_high = False
                    break

            if is_high:
                pivots.append({
                    'index': i,
                    'type': 'high',
                    'price': highs[i],
                    'timestamp': self._klines.iloc[i]['timestamp']
                })
            else:
                # 检查是否是低点
                is_low = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and lows[j] <= lows[i]:
                        is_low = False
                        break

                if is_low:
                    pivots.append({
                        'index': i,
                        'type': 'low',
                        'price': lows[i],
                        'timestamp': self._klines.iloc[i]['timestamp']
                    })

        return pivots

    def _identify_waves(self) -> List[Wave]:
        """
        识别波浪

        识别推动浪(1,2,3,4,5)和调整浪(A,B,C)
        """
        if len(self._pivots) < 5:
            return []

        waves = []
        i = 0

        while i < len(self._pivots) - 1:
            pivot = self._pivots[i]

            # 判断波浪方向
            if i + 1 < len(self._pivots):
                next_pivot = self._pivots[i + 1]

                if pivot['type'] == 'low' and next_pivot['type'] == 'high':
                    # 向上波浪
                    direction = 'up'
                    wave_num = self._get_next_wave_number(waves, 'up')
                    wave_type = WaveType.IMPULSE if wave_num in [1, 3, 5] else WaveType.CORRECTION
                elif pivot['type'] == 'high' and next_pivot['type'] == 'low':
                    # 向下波浪
                    direction = 'down'
                    wave_num = self._get_next_wave_number(waves, 'down')
                    wave_type = WaveType.IMPULSE if wave_num in [1, 3, 5] else WaveType.CORRECTION
                else:
                    i += 1
                    continue

                wave = Wave(
                    number=wave_num,
                    wave_type=wave_type,
                    start_idx=pivot['index'],
                    end_idx=next_pivot['index'],
                    start_price=pivot['price'],
                    end_price=next_pivot['price'],
                    high=max(pivot['price'], next_pivot['price']),
                    low=min(pivot['price'], next_pivot['price'])
                )

                # 计算斐波那契比率
                if len(waves) > 0:
                    prev = waves[-1]
                    wave_range = abs(wave.end_price - wave.start_price)
                    prev_range = abs(prev.end_price - prev.start_price)

                    if prev_range != 0:
                        wave.fib_ratio = wave_range / prev_range

                waves.append(wave)
                i += 1
            else:
                break

        return waves

    def _get_next_wave_number(self, waves: List[Wave], direction: str) -> int:
        """获取下一个波浪编号"""
        if not waves:
            return 1

        # 找到最后一个同方向波浪的编号
        last_same_dir = None
        for w in reversed(waves):
            is_up = w.start_price < w.end_price
            if (direction == 'up' and is_up) or (direction == 'down' and not is_up):
                last_same_dir = w
                break

        if last_same_dir is None:
            return 1

        # 波浪编号交替: 1,2,3,4,5 或 A,B,C
        current = last_same_dir.number

        if direction == 'up':
            # 向上波浪: 1->2->3->4->5
            if current >= 5:
                return 1
            return current + 1
        else:
            # 向下波浪: A->B->C
            if current == 'C':
                return 1
            if current == 'A':
                return ord('B')
            if current == 'B':
                return ord('C')
            return 1

    def _build_structures(self) -> List[WaveStructure]:
        """构建波浪结构"""
        if len(self._waves) < 5:
            return []

        structures = []

        # 尝试识别5浪推动+3浪调整的完整结构
        i = 0
        while i <= len(self._waves) - 8:
            wave_group = self._waves[i:i + 8]

            # 检查是否是有效的波浪结构
            impulse_waves = wave_group[:5]
            correction_waves = wave_group[5:8]

            if self._is_valid_impulse(impulse_waves):
                structure = WaveStructure(
                    waves=impulse_waves,
                    complete=True,
                    direction='up',
                    degree=WaveDegree.PRIMARY,
                    start_idx=impulse_waves[0].start_idx,
                    end_idx=impulse_waves[4].end_idx
                )
                structures.append(structure)

            if self._is_valid_correction(correction_waves):
                structure = WaveStructure(
                    waves=correction_waves,
                    complete=True,
                    direction='down',
                    degree=WaveDegree.PRIMARY,
                    start_idx=correction_waves[0].start_idx,
                    end_idx=correction_waves[2].end_idx
                )
                structures.append(structure)

            i += 1

        return structures

    def _is_valid_impulse(self, waves: List[Wave]) -> bool:
        """
        验证推动浪的有效性

        规则：
        1. 浪2不能回撤浪1的100%以上
        2. 浪3不能是最短的推动浪
        3. 浪4不能与浪1重叠
        """
        if len(waves) < 5:
            return False

        w1, w2, w3, w4, w5 = waves

        # 计算各个浪的幅度
        r1 = abs(w1.end_price - w1.start_price)
        r2 = abs(w2.end_price - w2.start_price)
        r3 = abs(w3.end_price - w3.start_price)
        r4 = abs(w4.end_price - w4.start_price)
        r5 = abs(w5.end_price - w5.start_price)

        # 规则1: 浪2回撤不能超过浪1的100%
        if w2.end_price <= w1.start_price:
            return False

        # 规则2: 浪3不能是最短的
        if r3 < r1 and r3 < r5:
            return False

        # 规则3: 浪4不能与浪1重叠
        if w4.low <= w1.high and w4.high >= w1.low:
            return False

        return True

    def _is_valid_correction(self, waves: List[Wave]) -> bool:
        """验证调整浪的有效性"""
        if len(waves) < 3:
            return False

        # 简单的锯齿形调整验证
        a, b, c = waves[:3]

        # C浪应该与A浪有相似的长度
        r_a = abs(a.end_price - a.start_price)
        r_c = abs(c.end_price - c.start_price)

        # C浪通常在A浪的61.8%-100%之间
        if r_c > r_a * 2.618:
            return False

        return True

    def calculate_fib_retracements(self, wave: Wave) -> List[FibonacciLevel]:
        """
        计算斐波那契回撤水平

        Args:
            wave: 波浪对象

        Returns:
            斐波那契回撤水平列表
        """
        if wave.start_price == wave.end_price:
            return []

        levels = []
        direction = 1 if wave.end_price > wave.start_price else -1
        wave_range = abs(wave.end_price - wave.start_price)

        for fib in self.FIB_RATIOS['retracement']:
            if direction == 1:
                # 向上波浪：从终点回撤
                price = wave.end_price - direction * wave_range * fib
            else:
                # 向下波浪：从终点回撤
                price = wave.end_price - direction * wave_range * fib

            levels.append(FibonacciLevel(
                level=fib * 100,
                price=price,
                type='retracement'
            ))

        return levels

    def calculate_fib_extensions(self, waves: List[Wave]) -> List[FibonacciLevel]:
        """
        计算斐波那契扩展水平

        Args:
            waves: 前面的波浪列表

        Returns:
            斐波那契扩展水平列表
        """
        if len(waves) < 3:
            return []

        levels = []
        w1 = waves[0]
        w2 = waves[1]

        # 以浪1和浪2为基础计算扩展
        wave_range = abs(w2.end_price - w2.start_price)

        for fib in self.FIB_RATIOS['extension']:
            price = w1.end_price + (w2.end_price - w1.start_price) + wave_range * fib

            levels.append(FibonacciLevel(
                level=fib * 100,
                price=price,
                type='extension'
            ))

        return levels

    def validate_wave_rules(self) -> List[str]:
        """
        验证波浪规则

        Returns:
            违规规则列表
        """
        violations = []

        if len(self._waves) < 5:
            return ['波浪数量不足']

        for i in range(len(self._waves) - 4):
            wave_group = self._waves[i:i + 5]

            # 规则1: 浪2回撤检查
            w1 = wave_group[0]
            w2 = wave_group[1]
            w2_retrace = abs(w2.end_price - w2.start_price) / abs(w1.end_price - w1.start_price)

            if w2_retrace > 1.0:
                violations.append(f"规则1违规: 浪{i+2}回撤超过浪{i+1}的100%")

            # 规则2: 浪3长度检查
            w3 = wave_group[2]
            r1 = abs(w1.end_price - w1.start_price)
            r3 = abs(w3.end_price - w3.start_price)

            if r3 < min(r1, abs(wave_group[4].end_price - wave_group[4].start_price)):
                violations.append(f"规则2违规: 浪{i+3}是最短的推动浪")

            # 规则3: 浪4与浪1重叠检查
            w4 = wave_group[3]
            if w4.low <= w1.high and w4.high >= w1.low:
                violations.append(f"规则3违规: 浪{i+4}与浪{i+1}重叠")

        return violations

    def generate_signals(self) -> List[Signal]:
        """
        基于艾略特波浪生成交易信号

        Returns:
            Signal列表
        """
        signals = []

        for i, structure in enumerate(self._structures):
            if not structure.complete:
                continue

            waves = structure.waves

            if structure.direction == 'up' and len(waves) >= 5:
                # 浪3突破浪1高点 -> 买入
                if waves[2].end_idx > waves[0].end_idx:
                    signals.append(Signal(
                        symbol='',
                        direction=SignalDirection.LONG,
                        strength=0.8,
                        metadata={
                            'type': 'elliott_wave',
                            'wave': 3,
                            'reason': '浪3突破浪1高点'
                        }
                    ))

                # 浪5突破浪3高点 -> 买入（趋势延续）
                if len(waves) >= 5 and waves[4].end_idx > waves[2].end_idx:
                    signals.append(Signal(
                        symbol='',
                        direction=SignalDirection.LONG,
                        strength=0.6,
                        metadata={
                            'type': 'elliott_wave',
                            'wave': 5,
                            'reason': '浪5突破浪3高点'
                        }
                    ))

            elif structure.direction == 'down' and len(waves) >= 5:
                # 向下波浪结构中的买入机会
                w2 = waves[1]
                w4 = waves[3]

                # 浪4低点买入（回调买入）
                signals.append(Signal(
                    symbol='',
                    direction=SignalDirection.LONG,
                    strength=0.5,
                    metadata={
                        'type': 'elliott_wave',
                        'wave': 4,
                        'reason': '浪4反弹'
                    }
                ))

        return signals

    def get_current_wave_position(self) -> Optional[Dict]:
        """
        获取当前所处的波浪位置

        Returns:
            当前位置信息
        """
        if not self._waves:
            return None

        last_wave = self._waves[-1]

        # 根据波浪编号和方向判断位置
        if last_wave.number in [1, 3, 5]:
            position = f"推动浪第{last_wave.number}浪"
            trend = 'up'
        elif last_wave.number in ['A', 'C']:
            position = f"调整浪第{last_wave.number}浪"
            trend = 'down'
        else:
            position = f"调整浪第{last_wave.number}浪"
            trend = 'neutral'

        return {
            'wave_number': last_wave.number,
            'position': position,
            'trend': trend,
            'fib_ratio': last_wave.fib_ratio,
            'end_price': last_wave.end_price
        }

    def get_summary(self) -> Dict:
        """获取波浪分析摘要"""
        return {
            '波浪总数': len(self._waves),
            '结构数量': len(self._structures),
            '完整结构': sum(1 for s in self._structures if s.complete),
            '当前波浪': self.get_current_wave_position()['position'] if self._waves else '无',
            '规则验证': self.validate_wave_rules() if self.strict_mode else []
        }
