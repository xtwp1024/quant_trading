"""
Harmonic Strategy — 谐波形态策略适配器
=====================================

基于 `quant_trading.strategy.harmonic.HarmonicPatternRecognizer` 的 BaseStrategy 适配器。

谐波形态类型：
- 蝴蝶型 (Butterfly)
- 螃蟹型 (Crab)
- 鲨鱼型 (Shark)
- 加特利型 (Gartley)
- 蝙蝠型 (Bat)
- 赛福型 (Cypher)

每个形态由特定的斐波那契比率定义。

Classes
-------
HarmonicStrategy
    BaseStrategy adapter for Harmonic Pattern trading.
HarmonicParams
    Strategy parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from quant_trading.signal import Signal, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext
from quant_trading.strategy.harmonic import (
    HarmonicPatternRecognizer,
    HarmonicPattern,
)


@dataclass
class HarmonicParams(StrategyParams):
    """谐波形态策略参数"""
    min_points: int = 5
    """最少转折点数"""
    tolerance: float = 0.05
    """斐波那契比率容差（5%）"""
    require_all_cd: bool = False
    """是否要求CD的所有比率都匹配"""
    entry_threshold: float = 0.6
    """形态置信度阈值（低于此值不入场）"""
    enabled_patterns: List[str] = None
    """启用哪些形态类型，默认全部启用"""

    def __post_init__(self):
        if self.enabled_patterns is None:
            self.enabled_patterns = [
                'Gartley', 'Butterfly', 'Crab', 'Bat', 'Shark', 'Cypher'
            ]


class HarmonicStrategy(BaseStrategy):
    """
    谐波形态策略 (Harmonic Pattern Strategy)

    基于谐波形态识别的技术分析策略。
    识别XABCD结构形态，在形态完成点产生交易信号。

    信号逻辑：
    - 看涨形态完成（bullish）-> 买入
    - 看跌形态完成（bearish）-> 卖出

    Parameters
    ----------
    symbol : str
        交易标的符号
    params : HarmonicParams, optional
        策略参数
    """

    name = "harmonic"

    def __init__(
        self,
        symbol: str,
        params: Optional[HarmonicParams] = None,
    ) -> None:
        super().__init__(symbol, params or HarmonicParams())
        self._recognizer = HarmonicPatternRecognizer(
            min_points=self.params.min_points,
            tolerance=self.params.tolerance,
            require_all_cd=self.params.require_all_cd,
        )
        self._signals: List[Signal] = []
        self._latest_pattern: Optional[Any] = None

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        基于谐波形态生成交易信号。

        Parameters
        ----------
        data : pd.DataFrame
            K线数据，需包含 open, high, low, close, timestamp 列

        Returns
        -------
        List[Signal]
            交易信号列表
        """
        # 加载数据到识别器
        self._recognizer.load_data(data)

        # 执行分析
        self._recognizer.analyze()

        # 获取最新有效形态
        self._latest_pattern = self._recognizer.get_latest_pattern()

        signals = []
        if self._latest_pattern and self._latest_pattern.valid:
            # 检查形态是否在启用列表中
            pattern_name = self._latest_pattern.pattern.value
            if pattern_name not in self.params.enabled_patterns:
                return signals

            # 检查置信度阈值
            if self._latest_pattern.confidence < self.params.entry_threshold:
                return signals

            # 确定方向
            direction = (
                SignalDirection.LONG
                if self._latest_pattern.direction == 'bullish'
                else SignalDirection.SHORT
            )

            # 获取当前最新K线信息
            last_row = data.iloc[-1] if len(data) > 0 else None
            timestamp = int(last_row.get("timestamp", 0)) if last_row is not None else 0
            price = float(last_row.get("close", self._latest_pattern.completion_price or 0.0)) if last_row is not None else 0.0

            signals.append(
                Signal(
                    symbol=self.symbol,
                    direction=direction,
                    price=price,
                    strength=self._latest_pattern.confidence,
                    timestamp=timestamp,
                    metadata={
                        "strategy": "harmonic",
                        "type": "harmonic_pattern",
                        "pattern": pattern_name,
                        "direction": self._latest_pattern.direction,
                        "reason": f"{pattern_name} {self._latest_pattern.direction} 形态完成",
                        "completion_price": self._latest_pattern.completion_price,
                        "stop_loss": self._latest_pattern.stop_loss,
                        "take_profit": self._latest_pattern.take_profit1,
                        "fib_ratios": self._latest_pattern.fib_ratios,
                    },
                )
            )

        self._signals = signals
        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        context: StrategyContext,
    ) -> float:
        """
        计算仓位大小。

        Parameters
        ----------
        signal : Signal
            交易信号
        context : StrategyContext
            策略上下文

        Returns
        -------
        float
            仓位大小
        """
        # 基于信号强度和置信度计算仓位
        base_size = context.available_cash * signal.strength * 0.05
        return base_size

    def on_bar(self, bar: pd.Series) -> Optional[Signal]:
        """
        处理单根K线数据。

        Parameters
        ----------
        bar : pd.Series
            K线数据

        Returns
        -------
        Optional[Signal]
            交易信号（如果有）
        """
        df = pd.DataFrame([bar])
        signals = self.generate_signals(df)
        return signals[-1] if signals else None

    def on_tick(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        处理tick数据。

        Parameters
        ----------
        tick : Dict[str, Any]
            Tick数据

        Returns
        -------
        Optional[Signal]
            交易信号（如果有）
        """
        return None

    def on_order_filled(self, order: Dict[str, Any]) -> None:
        """订单成交回调"""
        pass

    def on_position_changed(self, position: Dict[str, Any]) -> None:
        """持仓变化回调"""
        pass

    def get_required_history(self) -> int:
        """返回所需历史数据长度"""
        # 谐波形态需要足够的数据来识别XABCD结构
        return max(self.params.min_points * 10, 200)

    def get_recognizer(self) -> HarmonicPatternRecognizer:
        """获取底层识别器（用于诊断）"""
        return self._recognizer

    def get_summary(self) -> Dict[str, Any]:
        """获取谐波形态分析摘要"""
        return self._recognizer.get_summary()

    def get_latest_pattern(self) -> Optional[Any]:
        """获取最新识别的形态"""
        return self._latest_pattern

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "params": {
                "min_points": self.params.min_points,
                "tolerance": self.params.tolerance,
                "require_all_cd": self.params.require_all_cd,
                "entry_threshold": self.params.entry_threshold,
                "enabled_patterns": self.params.enabled_patterns,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarmonicStrategy":
        """从字典反序列化"""
        params = HarmonicParams(**data.get("params", {}))
        return cls(symbol=data["symbol"], params=params)
