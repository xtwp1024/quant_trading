"""
XGBoost Monthly Return Predictor.

XGBoost-based monthly frequency return prediction model with time-series
cross-validation, feature importance analysis, and graceful fallback when
xgboost is not installed.

Bilingual docstrings: Chinese first, English second.

Requires: xgboost, pandas, numpy, scikit-learn (all optional)
Install with: pip install xgboost scikit-learn pandas numpy
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = ["XGBoostPredictor"]

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Graceful xgboost import
# ----------------------------------------------------------------------

try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    xgb = None  # type: ignore
    logger.warning(
        "xgboost not found — XGBoostPredictor will use a simple fallback "
        "(np.linear regression) so that the class can still be instantiated. "
        "Install with: pip install xgboost"
    )

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    TimeSeriesSplit = None  # type: ignore
    StandardScaler = None  # type: ignore
    LinearRegression = None  # type: ignore


class XGBoostPredictor:
    """XGBoost月频收益预测模型.

    基于XGBoost的月频收益率预测模型，支持时序交叉验证、特征重要性分析。

    Attributes
    ----------
    n_estimators : int
        决策树数量，默认100
    max_depth : int
        单棵树最大深度，默认5（防止过拟合）
    learning_rate : float
        学习率，默认0.1
    objective : str
        优化目标，默认'reg:squarederror'（回归）
        可选: 'reg:squarederror', 'reg:logistic', 'binary:logistic'

    Examples
    --------
    >>> predictor = XGBoostPredictor(n_estimators=100, max_depth=5)
    >>> train_result = predictor.train(X_train, y_train)
    >>> predictions = predictor.predict(X_test)
    >>> importance = predictor.feature_importance()
    >>> cv_result = predictor.cross_validate(X, y, n_splits=5)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        objective: str = "reg:squarederror",
    ):
        """初始化XGBoost预测器.

        Parameters
        ----------
        n_estimators : int, optional
            弱学习器数量，默认100
        max_depth : int, optional
            树的最大深度，默认5（过大会导致过拟合）
        learning_rate : float, optional
            学习率（ shrinkage factor），默认0.1
        objective : str, optional
            目标函数，默认'reg:squarederror'
            - 'reg:squarederror': 回归MSE
            - 'reg:logistic': 回归逻辑
            - 'binary:logistic': 二分类
        """
        if not _XGB_AVAILABLE:
            logger.warning(
                "XGBoost not installed — using LinearRegression fallback. "
                "Results will differ from true XGBoost."
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective

        self._model: Any = None
        self._feature_names: Optional[list[str]] = None
        self._scaler: Any = None

        if _XGB_AVAILABLE:
            self._model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective=self.objective,
                verbosity=0,
                random_state=42,
            )
        else:
            self._model = None
            self._scaler = StandardScaler() if _SKLEARN_AVAILABLE else None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[list[str]] = None,
    ) -> dict:
        """训练XGBoost模型.

        Parameters
        ----------
        X : pd.DataFrame
            训练特征矩阵
        y : pd.Series
            目标变量（未来收益）
        feature_names : list[str], optional
            特征名称列表，用于特征重要性分析

        Returns
        -------
        dict
            训练结果，包含:
            - n_samples : 样本数量
            - n_features : 特征数量
            - feature_names : 使用的特征名称

        Notes
        -----
        训练数据会自动处理NaN值（XGBoost内置处理）。
        特征矩阵会在训练前进行标准化。
        """
        X_arr = X.fillna(0).values
        y_arr = y.fillna(0).values

        if feature_names is not None:
            self._feature_names = feature_names
        elif X.columns.tolist():
            self._feature_names = X.columns.tolist()
        else:
            self._feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

        if _XGB_AVAILABLE and self._model is not None:
            self._model.fit(
                X_arr, y_arr,
                feature_names=self._feature_names,
                verbose=False,
            )
        elif _SKLEARN_AVAILABLE and self._scaler is not None:
            X_scaled = self._scaler.fit_transform(X_arr)
            self._model = LinearRegression()
            self._model.fit(X_scaled, y_arr)
        else:
            raise RuntimeError(
                "Neither xgboost nor sklearn.linear_model is available. "
                "Please install at least one: pip install xgboost scikit-learn"
            )

        logger.info(
            f"XGBoostPredictor trained on {X_arr.shape[0]} samples, "
            f"{X_arr.shape[1]} features"
        )
        return {
            "n_samples": int(X_arr.shape[0]),
            "n_features": int(X_arr.shape[1]),
            "feature_names": self._feature_names,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """使用训练好的模型进行预测.

        Parameters
        ----------
        X : pd.DataFrame
            预测特征矩阵

        Returns
        -------
        np.ndarray
            预测值数组

        Raises
        ------
        RuntimeError
            如果模型未训练
        """
        if self._model is None:
            raise RuntimeError("Model not trained — call train() first")

        X_arr = X.fillna(0).values

        if _XGB_AVAILABLE and hasattr(self._model, "predict"):
            return self._model.predict(X_arr)
        elif _SKLEARN_AVAILABLE and self._scaler is not None:
            X_scaled = self._scaler.transform(X_arr)
            return self._model.predict(X_scaled)
        else:
            raise RuntimeError("No valid model available for prediction")

    def feature_importance(self) -> pd.Series:
        """返回特征重要性分数.

        Returns
        -------
        pd.Series
            特征重要性分数，index为特征名，values为重要性值，按降序排列

        Raises
        ------
        RuntimeError
            如果模型未训练
        """
        if self._model is None:
            raise RuntimeError("Model not trained — call train() first")

        if _XGB_AVAILABLE and hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        elif _SKLEARN_AVAILABLE and hasattr(self._model, "coef_"):
            # Use absolute coefficient values as importance for linear model
            importance = np.abs(self._model.coef_)
        else:
            raise RuntimeError("Cannot compute feature importance")

        names = self._feature_names or [f"f{i}" for i in range(len(importance))]
        series = pd.Series(importance, index=names, name="importance")
        return series.sort_values(ascending=False)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict:
        """时序交叉验证 (不要shuffle!).

        使用sklearn的TimeSeriesSplit进行时序交叉验证。
        时序交叉验证保证验证集总是出现在训练集之后，符合金融数据的时间特性。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵
        y : pd.Series
            目标变量
        n_splits : int, optional
            交叉验证折数，默认5

        Returns
        -------
        dict
            交叉验证结果，包含:
            - fold_scores : 每折的R²分数列表
            - mean_score : 平均R²分数
            - std_score : R²分数标准差
            - scores : 所有折的完整分数数组

        Examples
        --------
        >>> cv_result = predictor.cross_validate(X, y, n_splits=5)
        >>> print(f"CV R² = {cv_result['mean_score']:.4f} ± {cv_result['std_score']:.4f}")
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("sklearn required for cross_validate — pip install scikit-learn")

        if TimeSeriesSplit is None:
            raise RuntimeError("TimeSeriesSplit not available from sklearn")

        X_arr = X.fillna(0).values
        y_arr = y.fillna(0).values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_scores = []

        for train_idx, val_idx in tscv.split(X_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            if _XGB_AVAILABLE:
                fold_model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    objective=self.objective,
                    verbosity=0,
                    random_state=42,
                )
                fold_model.fit(X_train, y_train, verbose=False)
                y_pred = fold_model.predict(X_val)
            else:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)
                fold_model = LinearRegression()
                fold_model.fit(X_train_s, y_train)
                y_pred = fold_model.predict(X_val_s)

            # R² score
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            fold_scores.append(float(r2))

        fold_scores = np.array(fold_scores)
        return {
            "fold_scores": fold_scores.tolist(),
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "scores": fold_scores,
        }
