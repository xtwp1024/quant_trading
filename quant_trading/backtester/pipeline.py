# -*- coding: utf-8 -*-
"""
Pipeline Factor Computing API
Pipeline 因子计算 API — 声明式多因子计算框架

Inspired by Zipline's Pipeline API:
    https://github.com/quantopian/zipline

Architecture:
    Pipeline
        ├── Factor (自定义因子)
        │     ├── Rank (横截面排名)
        │     ├── ZScore (Z-Score 标准化)
        │     └── Returns (收益率因子)
        ├── Filter (过滤器)
        └── Classifier (分类器)

Example:
    >>> from quant_trading.backtester.pipeline import Pipeline, Rank, ZScore, Returns
    >>> pipe = (
    ...     Pipeline()
    ...     .add_factor(Returns(window_length=20), name="returns_20d")
    ...     .add_factor(Rank(ZScore(Returns())), name="rank_zscore")
    ...     .set_screen(MyFilter())
    ... )
    >>> result = pipe.evaluate(bars)  # bars: MultiIndex DataFrame (dt, symbol)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, Optional, Union, Any
from dataclasses import dataclass, field

__all__ = [
    "Pipeline",
    "Factor",
    "Rank",
    "ZScore",
    "Returns",
    "PercentChange",
    "Filter",
    "Screen",
    "CustomFactor",
]


# ------------------------------------------------------------------
# Factor Base Class / 因子基类
# ------------------------------------------------------------------


class Factor:
    """
    因子基类 / Factor base class.

    因子是对多个标的在某个时间点的数值计算结果，通常用于排序和选股。

    Attributes:
        window_length (int): 回溯窗口长度，默认20

    Example:
        >>> class MyFactor(Factor):
        ...     def compute(self, data):
        ...         close = data["close"]
        ...         return close.pct_change()
    """

    window_safe: bool = False

    def __init__(self, window_length: int = 20) -> None:
        """
        初始化因子.

        Args:
            window_length: 窗口长度，用于计算所需的历史数据
        """
        self.window_length = window_length

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值 / Compute factor values.

        Args:
            data: 包含所需输入列的 DataFrame，(dt, symbol) 为 MultiIndex，
                  或 (symbol) 为普通列索引

        Returns:
            pd.Series: 因子值，index 为 symbol
        """
        raise NotImplementedError("Subclasses must implement compute()")

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        """使因子可调用，代理到 compute()."""
        return self.compute(data)

    def pipe(self, func: Callable[[pd.Series], pd.Series], name: str = None) -> pd.Series:
        """
        对因子输出应用管道函数 / Pipe factor through a function.

        Args:
            func: 转换函数
            name: 新因子名称

        Returns:
            pd.Series: 转换后的因子值
        """
        result = func(self._last_result if hasattr(self, "_last_result") else self.compute(self._data))
        if name:
            result.name = name
        return result

    def rank(self) -> "Rank":
        """返回横截面排名因子."""
        return Rank(self)

    def zscore(self) -> "ZScore":
        """返回横截面Z-Score标准化因子."""
        return ZScore(self)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备计算所需的数据窗口."""
        return data


# ------------------------------------------------------------------
# Built-in Factors / 内置因子
# ------------------------------------------------------------------


class Returns(Factor):
    """
    收益率因子 / Returns Factor.

    计算指定窗口内的收益率: (close[-1] - close[0]) / close[0]

    Attributes:
        window_length (int): 回溯窗口，默认2（隔日收益）

    Example:
        >>> returns = Returns(window_length=5)
        >>> result = returns.compute(bars["close"])
    """

    window_safe = True

    def __init__(self, window_length: int = 2) -> None:
        if window_length < 2:
            raise ValueError(f"Returns requires window_length >= 2, got {window_length}")
        super().__init__(window_length=window_length)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算收益率.

        Args:
            data: DataFrame with 'close' column

        Returns:
            pd.Series: 收益率，index 为 symbol
        """
        close = data["close"]
        if isinstance(close, pd.DataFrame):
            # 多标的：取最后一行
            current = close.iloc[-1] if len(close) > 0 else close.iloc[0]
            previous = close.iloc[0] if len(close) > 0 else close.iloc[0]
        else:
            current = close.iloc[-1] if len(close) > 0 else close.iloc[0]
            previous = close.iloc[0] if len(close) > 0 else close.iloc[0]

        returns = (current - previous) / previous.abs()
        returns = returns.replace([np.inf, -np.inf], np.nan)
        return returns


class PercentChange(Factor):
    """
    百分比变化因子 / Percent Change Factor.

    与 Returns 类似，但计算方式为 (new - old) / |old|
    """

    window_safe = True

    def __init__(self, window_length: int = 1) -> None:
        super().__init__(window_length=window_length + 1)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算百分比变化."""
        close = data["close"]
        if isinstance(close, pd.DataFrame):
            current = close.iloc[-1]
            previous = close.iloc[0]
        else:
            current = close.iloc[-1]
            previous = close.iloc[0]

        pct = (current - previous) / previous.abs()
        pct = pct.replace([np.inf, -np.inf], np.nan)
        return pct


class Rank(Factor):
    """
    横截面排名因子 / Cross-sectional Rank Factor.

    对因子值在横截面（同一时间点所有标的）进行排名。

    Example:
        >>> rank_returns = Rank(Returns(window_length=20))
        >>> ranked = rank_returns.compute(bars)
        >>> # 输出: 000001.XSHE 排名 5/100, 600000.XSHG 排名 1/100, ...
    """

    window_safe = True

    def __init__(self, factor: Factor = None) -> None:
        super().__init__(window_length=getattr(factor, "window_length", 1))
        self._factor = factor

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算横截面排名 (0~N-1)."""
        if self._factor is not None:
            values = self._factor.compute(data)
        else:
            values = data
            if isinstance(data, pd.DataFrame):
                values = data.iloc[-1] if len(data) > 0 else data.iloc[0]

        # rank: ascending=True 越小排名越靠前（排名1最小），pct=True 返回百分比排名
        return values.rank(ascending=True, pct=False)


class ZScore(Factor):
    """
    横截面 Z-Score 标准化因子 / Cross-sectional Z-Score Factor.

    对因子值进行 Z-Score 标准化: (x - mean) / std

    Example:
        >>> zscore_returns = ZScore(Returns(window_length=20))
        >>> standardized = zscore_returns.compute(bars)
    """

    window_safe = True

    def __init__(self, factor: Factor = None, window_length: int = 20) -> None:
        super().__init__(window_length=window_length)
        self._factor = factor

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算横截面 Z-Score."""
        if self._factor is not None:
            values = self._factor.compute(data)
        else:
            values = data
            if isinstance(data, pd.DataFrame):
                values = data.iloc[-1] if len(data) > 0 else data.iloc[0]

        mean = values.mean()
        std = values.std()

        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=values.index)

        zscore = (values - mean) / std
        zscore = zscore.replace([np.inf, -np.inf], np.nan)
        return zscore


# ------------------------------------------------------------------
# Custom Factor / 自定义因子
# ------------------------------------------------------------------


class CustomFactor(Factor):
    """
    自定义因子 / Custom Factor.

    用户可通过继承此类或直接传入函数来创建自定义因子。

    Example:
        >>> # 方式1: 继承
        ... class VWAPFactor(CustomFactor):
        ...     inputs = ["close", "volume"]
        ...     window_length = 20
        ...     def compute(self, data):
        ...         close = data["close"]
        ...         volume = data["volume"]
        ...         return (close * volume).sum() / volume.sum()
        ...
        >>> # 方式2: 函数
        ... def my_factor(data):
        ...     return data["close"].std()
        ... pipe.add_factor(my_factor, name="volatility")
    """

    inputs: list[str] = ["close"]

    def __init__(
        self,
        func: Callable[[pd.DataFrame], pd.Series] = None,
        window_length: int = 20,
        name: str = None,
    ) -> None:
        super().__init__(window_length=window_length)
        self._func = func
        self._name = name

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """执行自定义计算."""
        if self._func is not None:
            result = self._func(data)
            if self._name:
                result.name = self._name
            return result
        raise NotImplementedError("CustomFactor requires a func or subclass compute()")


# ------------------------------------------------------------------
# Filter / 过滤器
# ------------------------------------------------------------------


class Filter:
    """
    过滤器基类 / Filter base class.

    过滤器用于选择标的，返回布尔 Series。
    """

    window_safe: bool = False

    def __init__(self, window_length: int = 1) -> None:
        self.window_length = window_length

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算过滤器 (返回布尔 Series)."""
        raise NotImplementedError("Subclasses must implement compute()")

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        return self.compute(data)


class PercentileFilter(Filter):
    """
    百分位过滤器 / Percentile Filter.

    选择因子值在指定百分位以上的标的。
    """

    window_safe = True

    def __init__(self, factor: Factor, min_percentile: float = 0.0, max_percentile: float = 1.0) -> None:
        super().__init__(window_length=getattr(factor, "window_length", 1))
        self._factor = factor
        self._min = min_percentile
        self._max = max_percentile

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算百分位过滤."""
        values = self._factor.compute(data)
        lo = values.quantile(self._min)
        hi = values.quantile(self._max)
        return (values >= lo) & (values <= hi)


class TopCount(Filter):
    """
    Top-N 过滤器 / Top Count Filter.

    选择因子值排名靠前的 N 个标的。
    """

    window_safe = True

    def __init__(self, factor: Factor, n: int = 10) -> None:
        super().__init__(window_length=getattr(factor, "window_length", 1))
        self._factor = factor
        self._n = n

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """选择排名前_n的标的."""
        values = self._factor.compute(data)
        threshold = values.nlargest(self._n).min()
        return values >= threshold


# ------------------------------------------------------------------
# Screen / 筛选器
# ------------------------------------------------------------------


class Screen:
    """
    Pipeline 筛选器 / Pipeline Screen.

    用于限制 Pipeline evaluate() 返回的标的范围。
    """

    def __init__(self, filter_: Union[Filter, pd.Series, Callable]) -> None:
        self._filter = filter_

    def evaluate(self, data: pd.DataFrame, available_symbols: list) -> list:
        """评估筛选条件，返回符合条件的标的列表."""
        if callable(self._filter) and not isinstance(self._filter, (Filter, pd.Series)):
            mask = self._filter(data)
        elif isinstance(self._filter, pd.Series):
            mask = self._filter
        elif isinstance(self._filter, Filter):
            mask = self._filter.compute(data)
        else:
            return available_symbols

        if isinstance(mask, pd.DataFrame):
            mask = mask.iloc[-1] if len(mask) > 0 else mask.iloc[0]
        elif isinstance(mask, pd.Series) and isinstance(mask.index, pd.MultiIndex):
            mask = mask.xs(mask.index.get_level_values(0)[-1], level=0)

        # 确保 mask 与 available_symbols 对齐
        if isinstance(mask, pd.Series):
            return [s for s in available_symbols if s in mask.index and bool(mask.get(s, False))]
        return available_symbols


# ------------------------------------------------------------------
# Pipeline / 主管道
# ------------------------------------------------------------------


class Pipeline:
    """
    声明式多因子计算管道 / Declarative Multi-factor Pipeline.

    Pipeline 提供声明式的因子计算接口，支持：
    - add_factor: 添加因子列
    - add_column: 添加任意数据列
    - set_screen: 设置过滤条件
    - pipe: 链式调用
    - evaluate: 在数据上执行计算

    Example:
        >>> from quant_trading.backtester.pipeline import Pipeline, Rank, ZScore, Returns
        ...
        >>> # 构建 Pipeline
        ... pipe = (
        ...     Pipeline()
        ...     .add_factor(Returns(window_length=5), name="returns_5d")
        ...     .add_factor(Returns(window_length=20), name="returns_20d")
        ...     .add_factor(
        ...         lambda data: data["close"].std() / data["close"].mean(),
        ...         name="cv"
        ...     )
        ...     .set_screen(TopCount(Rank(Returns()), n=50))
        ... )
        ...
        >>> # 在数据上评估
        >>> result = pipe.evaluate(bars)
        >>> print(result.head())
    """

    def __init__(self) -> None:
        """初始化空 Pipeline."""
        self._columns: dict[str, Union[Factor, Callable, pd.Series, pd.DataFrame]] = {}
        self._screen: Optional[Screen] = None
        self._graph: list[str] = []

    def add_factor(
        self,
        factor: Union[Factor, Callable[[pd.DataFrame], pd.Series]],
        name: str,
    ) -> "Pipeline":
        """
        添加因子列 / Add a factor column.

        Args:
            factor: 因子实例或可调用对象
            name: 因子名称（输出列名）

        Returns:
            Pipeline: 返回自身以支持链式调用

        Example:
            >>> pipe.add_factor(Returns(window_length=20), name="returns_20d")
        """
        self._columns[name] = factor
        self._graph.append(f"add_factor({name})")
        return self

    def add_column(
        self,
        name: str,
        data: Union[pd.Series, pd.DataFrame],
    ) -> "Pipeline":
        """
        添加任意数据列 / Add a data column.

        Args:
            name: 列名
            data: pd.Series 或 pd.DataFrame

        Returns:
            Pipeline: 返回自身
        """
        self._columns[name] = data
        self._graph.append(f"add_column({name})")
        return self

    def set_screen(
        self,
        filter_or_func: Union[Filter, pd.Series, Callable],
    ) -> "Pipeline":
        """
        设置筛选条件 / Set screen/filter.

        Args:
            filter_or_func: Filter 实例、布尔 Series 或可调用对象

        Returns:
            Pipeline: 返回自身
        """
        self._screen = Screen(filter_or_func)
        self._graph.append("set_screen(...)")
        return self

    def pipe(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        name: str = None,
    ) -> "Pipeline":
        """
        添加管道转换步骤 / Add a pipeline transformation step.

        Args:
            func: 转换函数，接收整个结果 DataFrame，返回转换后的 DataFrame
            name: 如果提供，将结果命名为新列

        Returns:
            Pipeline: 返回自身
        """
        self._columns[name if name else f"_pipe_{len(self._graph)}"] = func
        self._graph.append(f"pipe({name or '...'})")
        return self

    def show_graph(self) -> None:
        """
        打印 Pipeline 计算图（ASCII格式）/ Print ASCII computation graph.

        Example output:
            Pipeline Graph:
            ├── add_factor(returns_5d)
            ├── add_factor(returns_20d)
            ├── add_factor(cv)
            ├── set_screen(...)
            └── evaluate()
        """
        print("Pipeline Graph:")
        for i, step in enumerate(self._graph):
            prefix = "└── " if i == len(self._graph) - 1 else "├── "
            print(f"  {prefix}{step}")
        if self._screen is not None:
            print("  └── set_screen(...) [active]")

    def evaluate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        在数据上评估 Pipeline / Evaluate pipeline on data.

        Args:
            data: MultiIndex DataFrame (dt, symbol) with OHLCV columns
                  或包含 Pipeline inputs 中指定列的 DataFrame

        Returns:
            pd.DataFrame: 因子矩阵，(dt, symbol) 为 MultiIndex，
                          列名为 add_factor 时指定的 name

        Note:
            当 data 有多个时间步时，逐时间步计算并拼接结果。
        """
        results = []

        # 确定标的列表
        if isinstance(data.index, pd.MultiIndex):
            symbols = list(data.index.get_level_values(1).unique())
        else:
            symbols = list(data.columns)

        # 逐时间步计算
        if isinstance(data.index, pd.MultiIndex):
            timestamps = sorted(data.index.get_level_values(0).unique())

            for dt in timestamps:
                bar_data = data.xs(dt, level=0)
                row_result = {}

                for col_name, factor in self._columns.items():
                    if callable(factor) and not isinstance(factor, (pd.Series, pd.DataFrame)):
                        try:
                            if isinstance(factor, Factor):
                                row_result[col_name] = factor.compute(bar_data)
                            else:
                                row_result[col_name] = factor(bar_data)
                        except Exception:
                            row_result[col_name] = pd.Series(np.nan, index=bar_data.index)
                    elif isinstance(factor, pd.Series):
                        row_result[col_name] = factor
                    elif isinstance(factor, pd.DataFrame):
                        row_result[col_name] = factor.iloc[-1] if len(factor) > 0 else factor.iloc[0]
                    else:
                        row_result[col_name] = pd.Series(np.nan, index=bar_data.index)

                # 构建当前时间步的结果
                for sym in symbols:
                    row = {"dt": dt, "symbol": sym}
                    for col_name, col_values in row_result.items():
                        if sym in col_values.index:
                            row[col_name] = col_values[sym]
                        else:
                            row[col_name] = np.nan
                    results.append(row)
        else:
            # 单时间步
            row_result = {}
            for col_name, factor in self._columns.items():
                if callable(factor) and not isinstance(factor, (pd.Series, pd.DataFrame)):
                    try:
                        if isinstance(factor, Factor):
                            row_result[col_name] = factor.compute(data)
                        else:
                            row_result[col_name] = factor(data)
                    except Exception:
                        row_result[col_name] = pd.Series(np.nan, index=data.index)
                elif isinstance(factor, pd.Series):
                    row_result[col_name] = factor
                elif isinstance(factor, pd.DataFrame):
                    row_result[col_name] = factor.iloc[-1] if len(factor) > 0 else factor.iloc[0]
                else:
                    row_result[col_name] = pd.Series(np.nan, index=data.index)

            for sym in symbols:
                row = {"symbol": sym}
                for col_name, col_values in row_result.items():
                    if sym in col_values.index:
                        row[col_name] = col_values[sym]
                    else:
                        row[col_name] = np.nan
                results.append(row)

        # 合并结果
        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)

        if "dt" in result_df.columns:
            result_df = result_df.set_index(["dt", "symbol"])
        else:
            result_df = result_df.set_index("symbol")

        # 应用 screen
        if self._screen is not None and not result_df.empty:
            if isinstance(result_df.index, pd.MultiIndex):
                available_symbols = list(result_df.index.get_level_values(1).unique())
                screened_symbols = self._screen.evaluate(data, available_symbols)
                if isinstance(result_df.index, pd.MultiIndex):
                    result_df = result_df.loc[
                        result_df.index.get_level_values(1).isin(screened_symbols)
                    ]

        return result_df

    @property
    def columns(self) -> dict:
        """返回所有列定义."""
        return dict(self._columns)
