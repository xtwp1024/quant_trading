"""
IC/IR Factor Analyzer — evaluate alpha factor predictive power.

IC (Information Coefficient) = rank correlation(predicted, actual)
IR (Information Ratio)        = mean(IC) / std(IC)

This module provides a clean IC/IR analysis framework for quantitative factor
research, including rolling IC, factor selection, and IC decay analysis.

Bilingual docstrings: Chinese first, English second.

Requires: pandas, numpy, scipy
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ttest_1samp

__all__ = ["ICAnalyzer"]

logger = logging.getLogger(__name__)


class ICAnalyzer:
    """因子IC/IR分析器 — 评估alpha因子预测能力.

    IC (Information Coefficient) = rank correlation(predicted, actual)
        衡量因子预测能力，范围[-1, 1]，越接近1预测能力越强
    IR (Information Ratio) = mean(IC) / std(IC)
        衡量IC的稳定性，IR > 0.5 表示因子具有较好的预测稳定性

    Attributes
    ----------
    method : str
        相关性计算方法，'spearman' (默认) 或 'pearson'

    Examples
    --------
    >>> analyzer = ICAnalyzer(method='spearman')
    >>> result = analyzer.compute_ic(factor_values, forward_returns)
    >>> print(f"IC={result['ic_mean']:.4f}, IR={result['ir']:.4f}")
    """

    def __init__(self, method: str = "spearman"):
        """初始化IC分析器.

        Parameters
        ----------
        method : str, optional
            相关系数计算方法，默认'spearman'（秩相关，更适合金融非正态数据）
            可选: 'spearman', 'pearson'
        """
        if method not in ("spearman", "pearson"):
            raise ValueError("method must be 'spearman' or 'pearson'")
        self.method = method

    # ------------------------------------------------------------------
    # Core IC computation
    # ------------------------------------------------------------------

    def compute_ic(
        self,
        factor_values: np.ndarray | pd.Series,
        forward_returns: np.ndarray | pd.Series,
    ) -> dict:
        """计算IC序列, 返回统计指标.

        计算因子值与未来收益之间的IC (Information Coefficient) 及相关统计量。

        Parameters
        ----------
        factor_values : np.ndarray | pd.Series
            因子值序列（横截面维度，个股或资产）
        forward_returns : np.ndarray | pd.Series
            对应的未来收益序列

        Returns
        -------
        dict
            包含以下键值的字典:
            - ic_mean  : IC序列均值
            - ic_std   : IC序列标准差
            - ir       : Information Ratio = ic_mean / ic_std
            - ic_t     : IC均值的t统计量
            - rank_ic  : 秩相关系数（与spearman方法一致）
            - p_value  : IC均值的p值

        Notes
        -----
        IC计算时会自动去除NaN值对。
        t统计量使用单样本t检验，检验IC均值是否显著不为零。
        """
        # Convert to numpy arrays, dropping NaN pairs
        fv = np.asarray(factor_values, dtype=float).ravel()
        fr = np.asarray(forward_returns, dtype=float).ravel()

        # Pair-wise drop NaN
        mask = np.isfinite(fv) & np.isfinite(fr)
        fv = fv[mask]
        fr = fr[mask]

        if len(fv) < 3:
            logger.warning("Not enough valid pairs for IC computation (n < 3)")
            return {
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "ir": np.nan,
                "ic_t": np.nan,
                "rank_ic": np.nan,
                "p_value": np.nan,
            }

        if self.method == "spearman":
            corr, rank_ic = spearmanr(fv, fr)
            rank_ic = float(rank_ic) if hasattr(rank_ic, "__iter__") else corr
        else:
            corr = pearsonr(fv, fr)[0]
            rank_ic = corr

        # Build IC time-series (single observation here, but interface is
        # consistent with compute_rolling_ic which returns a Series)
        ic_series = np.array([corr])
        ic_mean = float(np.nanmean(ic_series))
        ic_std = float(np.nanstd(ic_series, ddof=1)) if len(ic_series) > 1 else 0.0
        ir = ic_mean / (ic_std + 1e-12)

        # t-stat for IC mean
        if len(ic_series) > 1 and ic_std > 0:
            ic_t, p_value = ttest_1samp(ic_series, 0)
            ic_t = float(ic_t)
            p_value = float(p_value)
        else:
            ic_t = np.nan
            p_value = np.nan

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ir": ir,
            "ic_t": ic_t,
            "rank_ic": float(rank_ic),
            "p_value": p_value,
        }

    def compute_rolling_ic(
        self,
        factor_matrix: pd.DataFrame,
        returns: pd.Series,
        window: int = 20,
    ) -> pd.DataFrame:
        """滚动IC — 多因子矩阵.

        在时间维度上滚动计算每个因子的IC值。

        Parameters
        ----------
        factor_matrix : pd.DataFrame
            因子值矩阵，index为时间，columns为因子名
        returns : pd.Series
            未来收益序列，index应与factor_matrix的时间index对齐
        window : int, optional
            滚动窗口大小，默认20（对应月频数据的20个月）

        Returns
        -------
        pd.DataFrame
            滚动IC DataFrame，index为时间，columns为因子名

        Examples
        --------
        >>> ic_df = analyzer.compute_rolling_ic(factor_df, returns, window=20)
        >>> print(ic_df.mean())  # mean IC per factor
        """
        if not isinstance(factor_matrix, pd.DataFrame):
            raise TypeError("factor_matrix must be a pandas DataFrame")
        if not isinstance(returns, (pd.Series, pd.DataFrame)):
            raise TypeError("returns must be a pandas Series")

        # Align by index
        common_idx = factor_matrix.index.intersection(returns.index)
        if len(common_idx) == 0:
            logger.error("No common index between factor_matrix and returns")
            return pd.DataFrame()

        fm = factor_matrix.loc[common_idx]
        ret = returns.loc[common_idx]

        results = {}
        for col in fm.columns:
            ic_vals = []
            for i in range(window, len(fm)):
                window_fv = fm[col].iloc[i - window : i].values
                window_ret = ret.iloc[i - window : i].values
                # Use last IC of the window as rolling IC at time i
                ic_res = self.compute_ic(window_fv, window_ret)
                ic_vals.append(ic_res.get("rank_ic", np.nan))
            results[col] = ic_vals

        rolling_ic = pd.DataFrame(
            results, index=fm.index[window:]
        )
        return rolling_ic

    def factor_selection(
        self,
        factor_matrix: pd.DataFrame,
        returns: pd.Series,
        top_k: int = 20,
        min_ir: float = 0.5,
    ) -> list[str]:
        """基于IC/IR筛选因子.

        根据IC/IR指标对因子进行筛选和排序，保留预测能力强且稳定的因子。

        Parameters
        ----------
        factor_matrix : pd.DataFrame
            因子值矩阵，index为时间，columns为因子名
        returns : pd.Series
            未来收益序列
        top_k : int, optional
            最多保留的因子数量，默认20
        min_ir : float, optional
            IR最低阈值，默认0.5；IR < 0.5的因子通常预测不稳定

        Returns
        -------
        list[str]
            筛选后的因子名称列表，按IR降序排列

        Examples
        --------
        >>> selected = analyzer.factor_selection(factor_df, returns, top_k=15, min_ir=0.3)
        >>> print(f"Selected {len(selected)} factors")
        """
        if not isinstance(factor_matrix, pd.DataFrame):
            raise TypeError("factor_matrix must be a DataFrame")

        ic_records = []
        for col in factor_matrix.columns:
            ic_res = self.compute_ic(factor_matrix[col], returns)
            ir = ic_res["ir"]
            ic_mean = ic_res["ic_mean"]
            rank_ic = ic_res["rank_ic"]
            if np.isnan(ir):
                continue
            ic_records.append({
                "factor": col,
                "ir": ir,
                "ic_mean": ic_mean,
                "rank_ic": rank_ic,
            })

        if not ic_records:
            logger.warning("No valid IC records found")
            return []

        ic_df = pd.DataFrame(ic_records).set_index("factor")
        # Filter by min_ir and sort by |ir| descending
        ic_df["ir_abs"] = ic_df["ir"].abs()
        filtered = ic_df[ic_df["ir_abs"] >= min_ir].sort_values("ir_abs", ascending=False)

        selected = filtered.head(top_k).index.tolist()
        logger.info(
            f"Factor selection: {len(selected)}/{len(factor_matrix.columns)} "
            f"factors passed IR >= {min_ir}"
        )
        return selected

    def decay_analysis(
        self,
        factor_values: pd.DataFrame,
        returns: pd.Series,
        max_lag: int = 5,
    ) -> pd.DataFrame:
        """因子IC衰减分析 — 不同预测周期的IC.

        分析因子在不同滞后期下的IC表现，评估因子的预测周期特性。

        Parameters
        ----------
        factor_values : pd.DataFrame
            因子值DataFrame，index为时间，columns为因子名
        returns : pd.Series
            收益序列
        max_lag : int, optional
            最大滞后期数，默认5

        Returns
        -------
        pd.DataFrame
            IC衰减矩阵，columns为因子名，index为滞后期(0,1,2,...,max_lag)

        Examples
        --------
        >>> decay = analyzer.decay_analysis(factor_df, returns, max_lag=5)
        >>> print(decay)  # 查看因子IC随滞后期的衰减情况
        """
        if not isinstance(factor_values, pd.DataFrame):
            raise TypeError("factor_values must be a DataFrame")

        common_idx = factor_values.index.intersection(returns.index)
        fm = factor_values.loc[common_idx]
        ret = returns.loc[common_idx]

        decay_matrix = {}
        for lag in range(max_lag + 1):
            ic_vals = {}
            for col in fm.columns:
                fv_lag = fm[col].iloc[:-lag] if lag > 0 else fm[col]
                ret_lag = ret.iloc[lag:]
                # Align lengths
                min_len = min(len(fv_lag), len(ret_lag))
                ic_res = self.compute_ic(fv_lag.iloc[:min_len], ret_lag.iloc[:min_len])
                ic_vals[col] = ic_res.get("rank_ic", np.nan)
            decay_matrix[lag] = ic_vals

        decay_df = pd.DataFrame(decay_matrix).T
        decay_df.index.name = "lag"
        return decay_df
