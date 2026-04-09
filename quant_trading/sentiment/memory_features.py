# quant_trading/sentiment/memory_features.py
"""Memory Features — Working memory state and feature extraction.

记忆特征工程模块，用于：
- 跟踪最近 N 个事件 (MemoryState)
- 从历史记忆中提取时序特征 (MemoryFeatureExtractor)

这些特征用于 Random Forest 预测器的输入。

Classes:
    MemoryState: 记忆状态 dataclass
    MemoryFeatureExtractor: 记忆特征提取器
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# MemoryState
# --------------------------------------------------------------------------- #


@dataclass
class MemoryState:
    """记忆状态 — 跟踪最近 N 个事件.

    用于量化策略的 working memory，模拟人类交易员的"记忆"能力。
    只保留最近 memory_depth 个事件，自动丢弃旧数据。

    Attributes:
        events: 事件列表，每项为 dict，含 timestamp, type, content, sentiment
        sentiment_history: 过去 N 个情绪值列表
        position_history: 过去 N 个仓位值列表
        memory_depth: 记忆最大深度 (默认 10)

    Example:
        >>> memory = MemoryState(memory_depth=5)
        >>> memory.add_sentiment(0.75)
        >>> memory.add_sentiment(-0.20)
        >>> memory.add_position(0.5)
        >>> print(memory.avg_sentiment)
        0.275
        >>> print(memory.sentiment_momentum)
        0.95
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    sentiment_history: list[float] = field(default_factory=list)
    position_history: list[float] = field(default_factory=list)
    memory_depth: int = 10

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def avg_sentiment(self) -> float:
        """平均情绪.

        Returns:
            过去 N 个情绪值的算术平均，范围 [-1, +1]。
            若无情绪记录，返回 0.0。
        """
        if not self.sentiment_history:
            return 0.0
        return float(np.mean(self.sentiment_history))

    @property
    def sentiment_momentum(self) -> float:
        """情绪动量 (当前 - 过去均值).

        Returns:
            最近情绪值减去历史均值，正值表示情绪上升，负值表示下降。
            公式: latest_sentiment - avg_sentiment(历史)
        """
        if len(self.sentiment_history) < 2:
            return 0.0
        latest = self.sentiment_history[-1]
        history_mean = np.mean(self.sentiment_history[:-1])
        return float(latest - history_mean)

    @property
    def sentiment_volatility(self) -> float:
        """情绪波动率.

        Returns:
            过去 N 个情绪值的标准差，反映市场情绪的不稳定性。
        """
        if len(self.sentiment_history) < 2:
            return 0.0
        return float(np.std(self.sentiment_history))

    @property
    def position_change_rate(self) -> float:
        """持仓变化率.

        Returns:
            最新仓位与历史平均仓位的差异比例。
            用于判断是否在加仓/减仓。
        """
        if not self.position_history:
            return 0.0
        latest = self.position_history[-1]
        history_mean = np.mean(self.position_history)
        if history_mean == 0:
            return 0.0
        return float((latest - history_mean) / (abs(history_mean) + 1e-9))

    @property
    def memory_fill_ratio(self) -> float:
        """记忆填充率.

        Returns:
            已使用记忆槽位 / 总记忆槽位 [0, 1]。
            反映记忆的饱和程度。
        """
        return min(1.0, len(self.events) / max(1, self.memory_depth))

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def add_event(self, event: dict[str, Any]) -> None:
        """添加一个事件到记忆.

        Args:
            event: 事件字典，应包含 type, content, sentiment 等字段
        """
        self.events.append(event)
        self._trim()

    def add_sentiment(self, score: float) -> None:
        """添加情绪值到情绪历史.

        Args:
            score: 情绪分数 [-1, +1]
        """
        self.sentiment_history.append(float(score))
        self._trim_sentiment()

    def add_position(self, position: float) -> None:
        """添加仓位值到仓位历史.

        Args:
            position: 仓位值 (如 0.0 ~ 1.0)
        """
        self.position_history.append(float(position))
        self._trim_position()

    def clear(self) -> None:
        """清空所有记忆."""
        self.events.clear()
        self.sentiment_history.clear()
        self.position_history.clear()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _trim(self) -> None:
        """Trim events to memory_depth."""
        if len(self.events) > self.memory_depth:
            self.events = self.events[-self.memory_depth :]

    def _trim_sentiment(self) -> None:
        """Trim sentiment history to memory_depth."""
        if len(self.sentiment_history) > self.memory_depth:
            self.sentiment_history = self.sentiment_history[-self.memory_depth :]

    def _trim_position(self) -> None:
        """Trim position history to memory_depth."""
        if len(self.position_history) > self.memory_depth:
            self.position_history = self.position_history[-self.memory_depth :]


# --------------------------------------------------------------------------- #
# MemoryFeatureExtractor
# --------------------------------------------------------------------------- #


class MemoryFeatureExtractor:
    """记忆特征提取器 — 从历史数据中提取时序特征.

    用于 RF 预测器的输入特征工程。
    从 MemoryState 中提取以下特征：

    Feature vector (顺序固定):
        0. sentiment_momentum      — 情绪动量
        1. sentiment_vol           — 情绪波动率
        2. avg_sentiment            — 平均情绪
        3. position_change_rate     — 持仓变化率
        4. memory_fill_ratio        — 记忆填充率
        5. sentiment_trend          — 情绪趋势 (线性拟合斜率)
        6. sentiment_latest         — 最新情绪值
        7. position_latest          — 最新仓位

    Attributes:
        memory_depth: 记忆窗口大小 (默认 10)

    Example:
        >>> extractor = MemoryFeatureExtractor(memory_depth=10)
        >>> memory = MemoryState(memory_depth=10)
        >>> memory.add_sentiment(0.5)
        >>> memory.add_sentiment(0.7)
        >>> memory.add_position(0.3)
        >>> features = extractor.extract(memory)
        >>> print(extractor.get_feature_names())
        ['sentiment_momentum', 'sentiment_vol', ...]
    """

    FEATURE_NAMES: list[str] = [
        "sentiment_momentum",
        "sentiment_vol",
        "avg_sentiment",
        "position_change_rate",
        "memory_fill_ratio",
        "sentiment_trend",
        "sentiment_latest",
        "position_latest",
    ]

    def __init__(self, memory_depth: int = 10) -> None:
        self.memory_depth = memory_depth

    def extract(self, memory: MemoryState) -> np.ndarray:
        """提取特征向量.

        Args:
            memory: MemoryState 实例

        Returns:
            numpy array of shape (8,) containing extracted features
        """
        momentum = memory.sentiment_momentum
        vol = memory.sentiment_volatility
        avg_sent = memory.avg_sentiment
        pos_change = memory.position_change_rate
        fill_ratio = memory.memory_fill_ratio
        trend = self._sentiment_trend(memory.sentiment_history)
        latest_sent = self._latest_or_zero(memory.sentiment_history)
        latest_pos = self._latest_or_zero(memory.position_history)

        return np.array(
            [
                momentum,
                vol,
                avg_sent,
                pos_change,
                fill_ratio,
                trend,
                latest_sent,
                latest_pos,
            ],
            dtype=np.float64,
        )

    def get_feature_names(self) -> list[str]:
        """返回特征名称列表.

        Returns:
            按 extract() 输出顺序对应的特征名称
        """
        return list(self.FEATURE_NAMES)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _sentiment_trend(self, history: list[float]) -> float:
        """计算情绪趋势 (简单线性拟合斜率).

        Args:
            history: 情绪历史列表

        Returns:
            线性拟合斜率，正值表示上升趋势，负值表示下降趋势
        """
        if len(history) < 2:
            return 0.0

        x = np.arange(len(history), dtype=np.float64)
        y = np.array(history, dtype=np.float64)

        # Simple linear regression slope: cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        cov = np.mean((x - x_mean) * (y - y_mean))
        var = np.var(x)

        if var < 1e-12:
            return 0.0

        slope = cov / var
        # Normalize to roughly [-1, +1] range based on typical data scale
        return float(slope)

    @staticmethod
    def _latest_or_zero(history: list[float]) -> float:
        """Return latest value or 0 if empty."""
        return float(history[-1]) if history else 0.0
