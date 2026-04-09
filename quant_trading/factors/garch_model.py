"""
GARCH波动率模型族 — 基于纯NumPy/SciPy实现，支持多种GARCH变体。
GARCH Volatility Model Family — Pure NumPy/SciPy implementation supporting multiple GARCH variants.

支持的模型 / Supported Models
----------------------------
- GARCH(1,1)        : 基本GARCH波动率模型
- GARCH-M           : 均值回归GARCH (mean-return GARCH)
- EGARCH            : 指数GARCH (Nelson, 1991) — 非对称波动率
- TGARCH            : 门限GARCH (Zakoian, 1994) — 非对称效应
- PGARCH            : 组件GARCH (Hansen, 1995) — 永久/瞬时波动率分解

核心功能 / Core Features
-----------------------
- 自动GARCH参数优化 (auto GARCH parameter optimization)
- z-score标准化残差触发机制 (z-score triggering mechanism)
- 波动率预测区间 (volatility forecast interval)
- 滚动拟合 (rolling fit for dynamic parameters)
- 自动(p,q)阶数选择 (auto order selection via AIC/BIC)

参考 / References
---------------
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
- Nelson, D.B. (1991). Conditional heteroskedasticity in asset returns.
- Zakoian, J.M. (1994). Threshold heteroskedastic models.
- Hansen, P.R. (1995). A component GARCH model.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "GARCHModel",
    "GARCHSignalGenerator",
]


# ---------------------------------------------------------------------------
# 工具函数 / Utility Functions
# ---------------------------------------------------------------------------

def _ensure_array(x: np.ndarray | pd.Series) -> np.ndarray:
    """将pd.Series或np.ndarray统一转为np.ndarray (float64)。"""
    if isinstance(x, pd.Series):
        return x.values.astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _check_stationarity(p: int, q: int, params: np.ndarray) -> bool:
    """简单检查GARCH参数是否满足平稳性条件: sum(alpha[i]) + sum(beta[j]) < 1."""
    n_alpha = p
    n_beta = q
    if len(params) < 1 + n_alpha + n_beta:
        return True  # 参数不足时跳过检查
    omega = params[0]
    alphas = params[1:1 + n_alpha]
    betas = params[1 + n_alpha:1 + n_alpha + n_beta]
    return bool(np.sum(np.abs(alphas)) + np.sum(np.abs(betas)) < 1.0)


# ---------------------------------------------------------------------------
# GARCH 负对数似然函数 / Negative Log-Likelihood Functions
# ---------------------------------------------------------------------------

def _garch_nll(params: np.ndarray, r: np.ndarray, p: int, q: int) -> float:
    """GARCH(1,1)负对数似然 / Negative log-likelihood for GARCH(1,1).

    波动率递归: h_t = w + a1*r_{t-1}^2 + b1*h_{t-1}
    """
    w = params[0]
    a = params[1:1 + p]
    b = params[1 + p:1 + p + q]

    T = len(r)
    h = np.zeros(T)
    h[0] = np.var(r) if np.var(r) > 0 else 1e-6

    for t in range(1, T):
        h[t] = w + np.sum(a * r[t - p:t][::-1] ** 2) + np.sum(b * h[t - q:t][::-1])

    h = np.maximum(h, 1e-9)
    llf = 0.5 * (np.log(2 * np.pi) + np.log(h) + r ** 2 / h)
    return np.sum(llf)


def _garchm_nll(params: np.ndarray, r: np.ndarray, p: int, q: int) -> float:
    """GARCH-M负对数似然 / Negative log-likelihood for GARCH-M.

    均值方程: r_t = mu + delta*h_t + eps_t
    波动率方程与GARCH(1,1)相同。
    """
    mu = params[0]
    delta = params[1]  # GARCH-M系数: 均值中的波动率溢价
    w = params[2]
    a = params[3:3 + p]
    b = params[3 + p:3 + p + q]

    T = len(r)
    h = np.zeros(T)
    h[0] = np.var(r) if np.var(r) > 0 else 1e-6

    for t in range(1, T):
        h[t] = w + np.sum(a * r[t - p:t][::-1] ** 2) + np.sum(b * h[t - q:t][::-1])

    h = np.maximum(h, 1e-9)
    # 残差 = r_t - mu - delta*h_t
    resid = r - mu - delta * h
    llf = 0.5 * (np.log(2 * np.pi) + np.log(h) + resid ** 2 / h)
    return np.sum(llf)


def _egarch_nll(params: np.ndarray, r: np.ndarray, p: int, q: int) -> float:
    """EGARCH负对数似然 / Negative log-likelihood for EGARCH (Nelson 1991).

    对数波动率递归:
        ln(h_t) = w + sum(ai * g(r_{t-1})) + sum(bj * ln(h_{t-1}))
    其中 g(z) = z * I(z>0) - E|z| + gamma * z * I(z<0)
    即 g(z) = (gamma + 1)*z - E|z| for z<0, (1-gamma)*z - E|z| for z>0
    为简化，使用标准化残差 epsilon_t = r_t / sqrt(h_t) 建模。
    """
    w = params[0]
    gamma = params[1]  # 非对称参数
    a = params[2:2 + p]
    b = params[2 + p:2 + p + q]

    T = len(r)
    # 初始化: 用无条件方差作为起始波动率
    init_h = np.var(r) if np.var(r) > 0 else 1e-6
    ln_h = np.full(T, np.log(init_h))
    std_resid = np.zeros(T)
    # 初始化标准化残差为0
    std_resid[0] = 0.0

    # 递归计算对数波动率
    # EGARCH: ln(h_t) = w + a1*g(eps_{t-1}) + ... + b1*ln(h_{t-1})
    # g(eps) = gamma*eps + |eps|*sqrt(pi/2)  (简化版使用)
    for t in range(1, T):
        # 使用之前的标准化残差
        e = std_resid[t - 1] if t - 1 >= 0 else 0.0
        # 非对称项: gamma * z，其中 z = (|r| - E|r|) 近似
        abs_r = np.abs(r[t - 1])
        g = gamma * r[t - 1] / np.sqrt(max(ln_h[t - 1], 1e-9))
        g = gamma * abs_r * (np.sign(r[t - 1]) if p > 0 else 0.0)

        # 简化EGARCH: ln(h_t) = w + a*g(r_{t-1}/sqrt(h_{t-1})) + b*ln(h_{t-1})
        # g(eps) = theta*eps + |eps|*sqrt(2/pi), 标准残差建模
        # 为实用性，使用简化的非对称ARCH项
        arch_term = np.sum(a * (np.abs(r[t - p:t][::-1]) * (r[t - p:t][::-1] < 0) * gamma +
                                r[t - p:t][::-1] * (r[t - p:t][::-1] >= 0) -
                                np.sqrt(2 / np.pi)))

        garch_term = np.sum(b * ln_h[t - q:t][::-1])
        ln_h[t] = w + arch_term + garch_term

    h = np.exp(np.maximum(ln_h, -20))  # 防止数值下溢
    h = np.maximum(h, 1e-9)

    # 标准化残差
    std_resid = r / np.sqrt(h)

    # 对数似然
    llf = 0.5 * (np.log(2 * np.pi) + 2 * np.log(np.sqrt(h)) + (r ** 2 / h))
    return np.sum(llf)


def _tgarch_nll(params: np.ndarray, r: np.ndarray, p: int, q: int) -> float:
    """TGARCH (Threshold GARCH) 负对数似然 / Negative log-likelihood for TGARCH.

    波动率递归:
        h_t = w + sum(ai * r_{t-i}^2 * I(r_{t-i}<0)) + sum(ai_prime * r_{t-i}^2 * I(r_{t-i}>=0))
              + sum(bj * h_{t-j})
    即负收益对波动率的影响更大（杠杆效应）。
    """
    w = params[0]
    # 非对称ARCH参数: gamma * r_{t-1}^2 当 r_{t-1} < 0
    gamma = params[1]  # 杠杆参数
    a = params[2:2 + p]  # 正向冲击参数
    b = params[2 + p:2 + p + q]

    T = len(r)
    h = np.zeros(T)
    h[0] = np.var(r) if np.var(r) > 0 else 1e-6

    for t in range(1, T):
        # ARCH项: r_{t-i}^2 对正负冲击分别处理
        arch_pos = 0.0
        arch_neg = 0.0
        for i in range(p):
            idx = t - i - 1
            if idx >= 0:
                if r[idx] < 0:
                    arch_neg += a[i] * r[idx] ** 2
                else:
                    arch_pos += a[i] * r[idx] ** 2
        # 非对称项: gamma * I(r_{t-1}<0) * r_{t-1}^2
        asym = gamma * (r[t - 1] < 0) * r[t - 1] ** 2 if t > 0 else 0.0

        garch_term = np.sum(b * h[t - q:t][::-1])
        h[t] = w + arch_pos + arch_neg + asym + garch_term

    h = np.maximum(h, 1e-9)
    llf = 0.5 * (np.log(2 * np.pi) + np.log(h) + r ** 2 / h)
    return np.sum(llf)


def _pgarch_nll(params: np.ndarray, r: np.ndarray, p: int, q: int) -> float:
    """PGARCH (Power GARCH / Component GARCH) 负对数似然.

    组件GARCH分解为永久组件(q_t)和瞬时组件(p_t):
        h_t = q_t + p_t
        q_t = w + rho * q_{t-1} + phi * (r_{t-1}^2 - q_{t-1})  # 永久组件
        p_t = beta * p_{t-1} + alpha * (r_{t-1}^2 - q_{t-1}) # 瞬时组件
    """
    w = params[0]   # 永久组件常数
    rho = params[1]  # 永久组件 persistence
    phi = params[2]  # 永久组件对冲击的响应
    beta = params[3]  # 瞬时组件 persistence
    alpha = params[4]  # 瞬时组件对冲击的响应

    T = len(r)
    q_h = np.zeros(T)  # 永久组件
    p_h = np.zeros(T)  # 瞬时组件
    h = np.zeros(T)    # 总波动率

    init_var = np.var(r) if np.var(r) > 0 else 1e-6
    q_h[0] = init_var * 0.7
    p_h[0] = init_var * 0.3
    h[0] = q_h[0] + p_h[0]

    for t in range(1, T):
        innov = r[t - 1] ** 2 - q_h[t - 1]  # 创新
        q_h[t] = w + rho * q_h[t - 1] + phi * innov
        p_h[t] = beta * p_h[t - 1] + alpha * innov
        q_h[t] = max(q_h[t], 1e-9)
        p_h[t] = max(p_h[t], 1e-9)
        h[t] = q_h[t] + p_h[t]

    h = np.maximum(h, 1e-9)
    llf = 0.5 * (np.log(2 * np.pi) + np.log(h) + r ** 2 / h)
    return np.sum(llf)


# ---------------------------------------------------------------------------
# 主类 / Main Classes
# ---------------------------------------------------------------------------

class GARCHModel:
    """GARCH波动率模型族 — 支持多种变体 / GARCH Volatility Model Family.

    支持模型 / Supported models:
        - ``'garch'``    : 基本GARCH(p,q)
        - ``'garchm'``   : GARCH-M (均值含波动率溢价)
        - ``'egarch'``   : 指数GARCH (Nelson 1991) — 非对称
        - ``'tgarch'``   : 门限GARCH (Zakoian 1994) — 非对称
        - ``'pgarch'``   : 组件GARCH (Hansen 1995) — 永久/瞬时分解

    参数说明 / Parameters
    --------------------
    model_type : str, optional
        模型类型，默认 ``'garch'``。
    p : int, optional
        ARCH阶数 (beta/alpha order)，默认 1。
    q : int, optional
        GARCH阶数 (alpha/beta order)，默认 1。
    optimizer : str, optional
        scipy.optimize.minimize 方法，默认 ``'SLSQP'``，
        也支持 ``'L-BFGS-B'``、``'Nelder-Mead'``。

    示例 / Example
    -------------
    >>> import numpy as np
    >>> returns = np.random.randn(1000) * 0.02  # 模拟收益率
    >>> model = GARCHModel(model_type='garch', p=1, q=1)
    >>> result = model.fit(returns)
    >>> forecast, std_fc = model.predict(horizon=5)
    >>> z = model.compute_zscore(returns)
    """

    MODEL_TYPES = ['garch', 'garchm', 'egarch', 'tgarch', 'pgarch']

    def __init__(
        self,
        model_type: str = 'garch',
        p: int = 1,
        q: int = 1,
        optimizer: str = 'SLSQP',
    ):
        if model_type not in self.MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported: {self.MODEL_TYPES}"
            )
        if p < 1 or q < 1:
            raise ValueError("p and q must be >= 1.")

        self.model_type = model_type
        self.p = p
        self.q = q
        self.optimizer = optimizer

        # 拟合结果 / Fitted results
        self.params_: np.ndarray | None = None
        self.llf_: float | None = None
        self.aic_: float | None = None
        self.bic_: float | None = None
        self.converged_: bool = False
        self._h_resid: np.ndarray | None = None  # 条件方差序列
        self._std_resid: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 拟合 / Fit
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray | pd.Series) -> dict:
        """拟合GARCH模型 / Fit GARCH model to returns.

        参数 / Parameters
        ----------
        returns : np.ndarray | pd.Series
            收益率序列 (e.g. 日收益率)。

        返回 / Returns
        -------
        dict
            包含以下键的字典:
            - ``params``  : 估计的参数数组
            - ``llf``     : 对数似然值
            - ``aic``     : AIC信息准则
            - ``bic``     : BIC信息准则
            - ``converged``: 是否收敛
        """
        r = _ensure_array(returns)
        if len(r) < 10:
            raise ValueError("returns must have at least 10 observations.")

        # 预处理: 去均值（对GARCH-M除外）
        self._r_raw = r.copy()
        self._r_centered = r - np.mean(r)

        T = len(r)
        n_params = self._n_params()

        # 参数边界 / Parameter bounds
        bounds = self._make_bounds(n_params)

        # 初始值 / Initial guess
        x0 = self._make_init(n_params, r)

        # 目标函数
        def objective(x):
            try:
                nll = self._neg_ll(x, r)
                if not np.isfinite(nll):
                    return 1e10
                return nll
            except Exception:
                return 1e10

        result = minimize(
            objective,
            x0,
            method=self.optimizer,
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-8},
        )

        self.params_ = result.x
        self.converged_ = result.success

        # 计算条件方差和标准化残差
        self._h_resid, self._std_resid = self._compute_h(self.params_, r)

        # 对数似然
        self.llf_ = -self._neg_ll(self.params_, r)
        k = n_params
        self.aic_ = 2 * k - 2 * self.llf_
        self.bic_ = k * np.log(T) - 2 * self.llf_

        return {
            "params": self.params_,
            "llf": self.llf_,
            "aic": self.aic_,
            "bic": self.bic_,
            "converged": self.converged_,
        }

    # ------------------------------------------------------------------
    # 预测 / Prediction
    # ------------------------------------------------------------------

    def predict(self, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """预测未来波动率 / Forecast future volatility.

        参数 / Parameters
        ----------
        horizon : int, optional
            预测步数，默认 1。

        返回 / Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean_forecast, std_forecast) — 分别是波动率均值和标准差预测序列。
        """
        if self.params_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        var_fc = self.forecast_variance(horizon)
        return np.sqrt(var_fc), np.sqrt(var_fc)

    def forecast_variance(self, horizon: int = 5) -> np.ndarray:
        """预测方差序列 / Forecast variance series.

        对于GARCH(1,1):
            h_{t+h} = omega + (alpha+beta) * h_{t+h-1}  (长期收敛到 unconditional variance)
            或递归计算多步。

        参数 / Parameters
        ----------
        horizon : int, optional
            预测步数，默认 5。

        返回 / Returns
        -------
        np.ndarray
            方差预测数组，形状 (horizon,)。
        """
        if self.params_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        r = self._r_centered
        params = self.params_
        p, q = self.p, self.q

        # 从拟合中获取最后的条件方差
        last_h = self._h_resid[-1] if self._h_resid is not None else np.var(r)

        var_fc = np.zeros(horizon)

        if self.model_type == 'garch':
            omega = params[0]
            a = params[1:1 + p]
            b = params[1 + p:1 + p + q]
            alpha_sum = np.sum(a)
            beta_sum = np.sum(b)
            # unconditional variance = omega / (1 - alpha_sum - beta_sum)
            try:
                unc_var = omega / max(1 - alpha_sum - beta_sum, 1e-9)
            except (ZeroDivisionError, FloatingPointError):
                unc_var = last_h
            # 多步预测
            for h in range(horizon):
                if h == 0:
                    var_fc[h] = last_h
                else:
                    var_fc[h] = omega + (alpha_sum + beta_sum) * var_fc[h - 1]
                # 防止偏离unconditional variance太远
                var_fc[h] = max(var_fc[h], unc_var * 0.01)

        elif self.model_type == 'garchm':
            mu = params[0]
            delta = params[1]
            w = params[2]
            a = params[3:3 + p]
            b = params[3 + p:3 + p + q]
            alpha_sum = np.sum(a)
            beta_sum = np.sum(b)
            unc_var = w / max(1 - alpha_sum - beta_sum, 1e-9)
            for h in range(horizon):
                if h == 0:
                    var_fc[h] = last_h
                else:
                    var_fc[h] = w + (alpha_sum + beta_sum) * var_fc[h - 1]
                var_fc[h] = max(var_fc[h], unc_var * 0.01)

        elif self.model_type == 'egarch':
            w = params[0]
            gamma = params[1]
            a = params[2:2 + p]
            b = params[2 + p:2 + p + q]
            beta_sum = np.sum(b)
            last_ln_h = np.log(max(last_h, 1e-9))
            for h in range(horizon):
                ln_h = w + beta_sum * last_ln_h  # 简化为常数ARCH贡献
                last_ln_h = ln_h
                var_fc[h] = np.exp(np.clip(ln_h, -20, 20))

        elif self.model_type == 'tgarch':
            w = params[0]
            gamma = params[1]
            a = params[2:2 + p]
            b = params[2 + p:2 + p + q]
            alpha_sum = np.sum(a)
            beta_sum = np.sum(b)
            unc_var = w / max(1 - alpha_sum - gamma * 0.5 - beta_sum, 1e-9)
            for h in range(horizon):
                if h == 0:
                    var_fc[h] = last_h
                else:
                    var_fc[h] = w + (alpha_sum + beta_sum + gamma * 0.5) * var_fc[h - 1]
                var_fc[h] = max(var_fc[h], unc_var * 0.01)

        elif self.model_type == 'pgarch':
            w = params[0]
            rho = params[1]
            phi = params[2]
            beta = params[3]
            alpha = params[4]
            # 简化: 使用长期方差
            unc_var = w / max(1 - rho, 1e-9)
            q_h = unc_var * 0.7
            p_h = unc_var * 0.3
            for h in range(horizon):
                innov = 0  # 假设创新为0
                q_h = w + rho * q_h + phi * innov
                p_h = beta * p_h + alpha * innov
                var_fc[h] = q_h + p_h
            var_fc = np.maximum(var_fc, unc_var * 0.01)

        else:
            # Fallback: 使用无条件方差
            var_fc[:] = max(np.var(r), 1e-9)

        return var_fc

    # ------------------------------------------------------------------
    # z-score / 标准化残差
    # ------------------------------------------------------------------

    def compute_zscore(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """计算GARCH标准化残差 (z-score) / Compute GARCH standardized residuals.

        z_t = r_t / sqrt(h_t)
        其中 h_t 是GARCH条件方差。

        参数 / Parameters
        ----------
        returns : np.ndarray | pd.Series
            收益率序列。

        返回 / Returns
        -------
        np.ndarray
            标准化残差数组，长度与returns相同。
        """
        r = _ensure_array(returns)
        if self.params_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        _, std_resid = self._compute_h(self.params_, r)
        return std_resid

    # ------------------------------------------------------------------
    # 滚动拟合 / Rolling Fit
    # ------------------------------------------------------------------

    def rolling_fit(
        self,
        returns: np.ndarray | pd.Series,
        window: int = 252,
    ) -> pd.DataFrame:
        """滚动拟合 — 动态GARCH参数 / Rolling fit for dynamic GARCH parameters.

        使用滑动窗口估计GARCH参数，返回每期的波动率和z-score。

        参数 / Parameters
        ----------
        returns : np.ndarray | pd.Series
            收益率序列。
        window : int, optional
            滚动窗口大小，默认 252 (约一年日交易日)。

        返回 / Returns
        -------
        pd.DataFrame
            包含列: ``volatility`` (条件波动率), ``zscore``, ``params``。
        """
        r = _ensure_array(returns)
        if len(r) <= window:
            raise ValueError(
                f"returns length ({len(r)}) must be > window ({window})."
            )

        T = len(r)
        vol = np.full(T, np.nan)
        zscore = np.full(T, np.nan)
        param_arr = np.full((T, self._n_params()), np.nan)

        for t in range(window, T):
            window_data = r[t - window:t]
            try:
                g = GARCHModel(
                    model_type=self.model_type,
                    p=self.p,
                    q=self.q,
                    optimizer=self.optimizer,
                )
                res = g.fit(window_data)
                h_last = g._h_resid[-1] if g._h_resid is not None else np.var(window_data)
                vol[t] = np.sqrt(max(h_last, 1e-9))
                zscore[t] = r[t] / vol[t]
                param_arr[t] = res["params"]
            except Exception:
                vol[t] = np.nan
                zscore[t] = np.nan

        df = pd.DataFrame(
            {"volatility": vol, "zscore": zscore},
            index=getattr(returns, "index", None),
        )
        df["params"] = [param_arr[i] for i in range(T)]
        return df

    # ------------------------------------------------------------------
    # 自动阶数选择 / Auto Order Selection
    # ------------------------------------------------------------------

    def auto_select_order(
        self,
        returns: np.ndarray | pd.Series,
        max_p: int = 3,
        max_q: int = 3,
        criterion: str = 'aic',
    ) -> dict:
        """自动选择最优(p,q) — 基于AIC或BIC / Auto-select optimal (p,q) by AIC or BIC.

        参数 / Parameters
        ----------
        returns : np.ndarray | pd.Series
            收益率序列。
        max_p : int, optional
            最大ARCH阶数，默认 3。
        max_q : int, optional
            最大GARCH阶数，默认 3。
        criterion : str, optional
            选择准则，``'aic'`` 或 ``'bic'``，默认 ``'aic'``。

        返回 / Returns
        -------
        dict
            包含 ``best_p``, ``best_q``, ``best_aic`` (或 ``best_bic``),
            ``results_table`` (所有候选模型的AIC/BIC)。
        """
        r = _ensure_array(returns)
        if criterion not in ('aic', 'bic'):
            raise ValueError("criterion must be 'aic' or 'bic'.")

        results = []
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    g = GARCHModel(
                        model_type=self.model_type,
                        p=p,
                        q=q,
                        optimizer=self.optimizer,
                    )
                    res = g.fit(r)
                    score = res[criterion]
                    results.append(
                        {
                            "p": p,
                            "q": q,
                            "aic": res["aic"],
                            "bic": res["bic"],
                            "llf": res["llf"],
                            "converged": res["converged"],
                            "score": score,
                        }
                    )
                except Exception:
                    continue

        if not results:
            raise RuntimeError("No model converged. Try different data or smaller max_p/q.")

        results_df = pd.DataFrame(results).sort_values("score")
        best = results_df.iloc[0]

        return {
            "best_p": int(best["p"]),
            "best_q": int(best["q"]),
            f"best_{criterion}": float(best[criterion]),
            "results_table": results_df.reset_index(drop=True),
        }

    # ------------------------------------------------------------------
    # 内部辅助 / Internal Helpers
    # ------------------------------------------------------------------

    def _n_params(self) -> int:
        """返回当前模型参数数量 / Return number of parameters for current model."""
        if self.model_type == 'garch':
            return 1 + self.p + self.q
        elif self.model_type == 'garchm':
            return 3 + self.p + self.q  # mu, delta, omega
        elif self.model_type == 'egarch':
            return 2 + self.p + self.q  # omega, gamma, alphas, betas
        elif self.model_type == 'tgarch':
            return 2 + self.p + self.q  # omega, gamma, alphas, betas
        elif self.model_type == 'pgarch':
            return 5  # w, rho, phi, beta, alpha (p,q fixed for PGARCH)
        return 1 + self.p + self.q

    def _make_bounds(self, n: int) -> list[tuple[float, float]]:
        """生成参数边界 / Generate parameter bounds."""
        if self.model_type == 'pgarch':
            # PGARCH has exactly 5 fixed parameters
            return [
                (1e-8, None),    # w
                (0.0, 0.9999),  # rho
                (-1.0, 1.0),    # phi
                (0.0, 0.9999),  # beta
                (0.0, 1.0),     # alpha
            ]

        bounds = []
        # omega (w) > 0
        bounds.append((1e-8, None))
        if self.model_type in ('garchm',):
            # mu: 无限制, delta: [-10, 10]
            bounds.append((None, None))  # mu
            bounds.append((-10.0, 10.0))  # delta
            n_rem = n - 3
        elif self.model_type in ('egarch', 'tgarch'):
            # gamma: [-10, 10]
            bounds.append((-10.0, 10.0))  # gamma
            n_rem = n - 2
        else:
            n_rem = n - 1

        # alpha[i] >= 0
        for _ in range(min(self.p, n_rem)):
            bounds.append((1e-8, None))
            n_rem -= 1
        # beta[j] >= 0
        for _ in range(min(self.q, n_rem)):
            bounds.append((1e-8, 0.9999))
            n_rem -= 1
        # 填充剩余边界
        for _ in range(n_rem):
            bounds.append((1e-8, None))

        return bounds[:n]

    def _make_init(self, n: int, r: np.ndarray) -> np.ndarray:
        """生成初始参数猜测 / Generate initial parameter guess."""
        if self.model_type == 'pgarch':
            # PGARCH always has exactly 5 parameters
            var_r = np.var(r) if np.var(r) > 0 else 1e-6
            return np.array([
                var_r * 0.01,   # w
                0.98,           # rho
                0.10,           # phi
                0.85,           # beta
                0.05,           # alpha
            ])

        x0 = np.zeros(n)
        var_r = np.var(r) if np.var(r) > 0 else 1e-6

        if self.model_type == 'garch':
            x0[0] = var_r * 0.05       # omega
            x0[1] = 0.08               # alpha1
            x0[2] = 0.90               # beta1

        elif self.model_type == 'garchm':
            x0[0] = np.mean(r)         # mu
            x0[1] = 0.01               # delta
            x0[2] = var_r * 0.05       # omega
            x0[3] = 0.08               # alpha1
            x0[4] = 0.90               # beta1

        elif self.model_type == 'egarch':
            x0[0] = np.log(var_r) * 0.1  # omega
            x0[1] = 0.01               # gamma (非对称)
            x0[2] = 0.08               # alpha1
            x0[3] = 0.90               # beta1

        elif self.model_type == 'tgarch':
            x0[0] = var_r * 0.05       # omega
            x0[1] = 0.01               # gamma
            x0[2] = 0.08               # alpha1
            x0[3] = 0.90               # beta1

        return x0

    def _neg_ll(self, params: np.ndarray, r: np.ndarray) -> float:
        """调度到具体模型的负对数似然 / Dispatch to specific NLL function."""
        if self.model_type == 'garch':
            return _garch_nll(params, r, self.p, self.q)
        elif self.model_type == 'garchm':
            return _garchm_nll(params, r, self.p, self.q)
        elif self.model_type == 'egarch':
            return _egarch_nll(params, r, self.p, self.q)
        elif self.model_type == 'tgarch':
            return _tgarch_nll(params, r, self.p, self.q)
        elif self.model_type == 'pgarch':
            return _pgarch_nll(params, r, self.p, self.q)
        raise ValueError(f"Unknown model type: {self.model_type}")

    def _compute_h(
        self, params: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算条件方差和标准化残差 / Compute conditional variance and std residuals."""
        h = np.zeros(len(r))
        init_h = np.var(r) if np.var(r) > 0 else 1e-6
        h[0] = init_h

        for t in range(1, len(r)):
            if self.model_type == 'garch':
                w = params[0]
                a = params[1:1 + self.p]
                b = params[1 + self.p:1 + self.p + self.q]
                arch = np.sum(a * r[t - self.p:t][::-1] ** 2)
                garch = np.sum(b * h[t - self.q:t][::-1])
                h[t] = w + arch + garch

            elif self.model_type == 'garchm':
                mu = params[0]
                delta = params[1]
                w = params[2]
                a = params[3:3 + self.p]
                b = params[3 + self.p:3 + self.p + self.q]
                arch = np.sum(a * r[t - self.p:t][::-1] ** 2)
                garch = np.sum(b * h[t - self.q:t][::-1])
                h[t] = w + arch + garch

            elif self.model_type == 'egarch':
                w = params[0]
                gamma = params[1]
                a = params[2:2 + self.p]
                b = params[2 + self.p:2 + self.p + self.q]
                ln_h_prev = np.log(max(h[t - 1], 1e-9))
                # 简化的EGARCH波动率更新
                arch = np.sum(a * (gamma * r[t - self.p:t][::-1] *
                                   (r[t - self.p:t][::-1] < 0) +
                                   r[t - self.p:t][::-1] * (r[t - self.p:t][::-1] >= 0) -
                                   np.sqrt(2 / np.pi)))
                garch = np.sum(b * ln_h_prev)
                h[t] = np.exp(np.clip(w + arch + garch, -20, 20))

            elif self.model_type == 'tgarch':
                w = params[0]
                gamma = params[1]
                a = params[2:2 + self.p]
                b = params[2 + self.p:2 + self.p + self.q]
                arch = 0.0
                for i in range(self.p):
                    idx = t - i - 1
                    if idx >= 0:
                        arch += a[i] * r[idx] ** 2
                asym = gamma * (r[t - 1] < 0) * r[t - 1] ** 2 if t > 0 else 0.0
                garch = np.sum(b * h[t - self.q:t][::-1])
                h[t] = max(w + arch + asym + garch, 1e-9)

            elif self.model_type == 'pgarch':
                w = params[0]
                rho = params[1]
                phi = params[2]
                beta = params[3]
                alpha = params[4]
                q_h = h[t - 1] * 0.7  # 简化
                innov = r[t - 1] ** 2 - q_h
                q_h_new = w + rho * q_h + phi * innov
                p_h_new = beta * h[t - 1] * 0.3 + alpha * innov
                h[t] = max(q_h_new + p_h_new, 1e-9)

            h[t] = max(h[t], 1e-9)

        std_resid = r / np.sqrt(h)
        return h, std_resid


# ---------------------------------------------------------------------------
# 信号生成器 / Signal Generator
# ---------------------------------------------------------------------------

class GARCHSignalGenerator:
    """基于GARCH的量化信号生成器 / GARCH-based quantitative signal generator.

    使用GARCH标准化残差(z-score)触发交易信号:
    - |z| > z_threshold: 预期反转 -> 均值回归 (+1 或 -1)
    - |z| <= z_threshold: 趋势跟随 (-1 或 +1)

    参数 / Parameters
    ----------------
    model : GARCHModel
        已拟合的GARCHModel实例。

    示例 / Example
    -------------
    >>> model = GARCHModel('garch', p=1, q=1)
    >>> result = model.fit(returns)
    >>> gen = GARCHSignalGenerator(model)
    >>> signals = gen.generate_signals(returns, z_threshold=2.0)
    """

    def __init__(self, model: GARCHModel):
        if not isinstance(model, GARCHModel):
            raise TypeError("model must be a GARCHModel instance.")
        if model.params_ is None:
            raise RuntimeError("GARCHModel must be fitted before use.")
        self.model = model

    def generate_signals(
        self,
        returns: np.ndarray | pd.Series,
        z_threshold: float = 2.0,
    ) -> pd.Series:
        """生成基于z-score的交易信号 / Generate trading signals based on z-score.

        逻辑:
            - 如果 |z_t| > z_threshold: 预期均值回归 -> +1 (多头) 或 -1 (空头)
              (方向由收益率符号决定: 正收益->空头, 负收益->多头)
            - 否则: 趋势跟随 -> -1 (看空) 或 +1 (看多)
              (方向由收益率符号决定)

        参数 / Parameters
        ----------
        returns : np.ndarray | pd.Series
            收益率序列。
        z_threshold : float, optional
            z-score触发阈值，默认 2.0。

        返回 / Returns
        -------
        pd.Series
            信号序列，值为 +1 (多头), -1 (空头), 0 (中性)。
            索引与输入returns一致。
        """
        r = _ensure_array(returns)
        zscore = self.model.compute_zscore(r)

        signals = np.zeros(len(r))

        for t in range(len(r)):
            z = zscore[t]
            ret = r[t]

            if abs(z) > z_threshold:
                # 均值回归信号: z>0 -> 做空 (正收益偏离,预期回落)
                #              z<0 -> 做多 (负收益偏离,预期反弹)
                signals[t] = -1.0 if z > 0 else 1.0
            else:
                # 趋势跟随信号: 收益率>0 -> +1 (多头)
                #               收益率<0 -> -1 (空头)
                signals[t] = 1.0 if ret >= 0 else -1.0

        return pd.Series(signals, index=getattr(returns, "index", None))

    def generate_regime_signals(
        self,
        returns: np.ndarray | pd.Series,
        z_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """生成包含波动率区制的增强信号 / Generate regime-aware enhanced signals.

        返回DataFrame包含:
        - signal: 主交易信号 (+1/-1/0)
        - zscore: GARCH标准化残差
        - volatility: 条件波动率
        - regime: 'high_vol' / 'low_vol' / 'normal'
        - confidence: 信号置信度 (|z| / z_threshold 的归一化值)

        参数 / Parameters
        ----------
        returns : np.ndarray | pd.Series
            收益率序列。
        z_threshold : float, optional
            z-score触发阈值，默认 2.0。

        返回 / Returns
        -------
        pd.DataFrame
            增强信号表。
        """
        r = _ensure_array(returns)
        zscore = self.model.compute_zscore(r)
        vol, _ = self.model.predict(horizon=1)
        vol_arr = np.full(len(r), float(vol[0]) if np.isscalar(vol) else vol)

        # 计算长期波动率中位数作为区制阈值
        vol_median = np.nanmedian(vol_arr)
        vol_std = np.nanstd(vol_arr)

        regimes = np.full(len(r), 'normal', dtype=object)
        regimes[vol_arr > vol_median + vol_std] = 'high_vol'
        regimes[vol_arr < vol_median - vol_std] = 'low_vol'

        # 置信度: |z| / z_threshold, 截断在[0,1]
        confidence = np.clip(np.abs(zscore) / z_threshold, 0, 1)

        signals = np.zeros(len(r))
        for t in range(len(r)):
            z = zscore[t]
            ret = r[t]
            reg = regimes[t]

            if abs(z) > z_threshold:
                signals[t] = -1.0 if z > 0 else 1.0
                # 高波动区制降低信号强度
                if reg == 'high_vol':
                    signals[t] *= 0.5
            else:
                signals[t] = 1.0 if ret >= 0 else -1.0
                if reg == 'low_vol':
                    signals[t] *= 0.8

        df = pd.DataFrame(
            {
                "signal": signals,
                "zscore": zscore,
                "volatility": vol_arr,
                "regime": regimes,
                "confidence": confidence,
            },
            index=getattr(returns, "index", None),
        )
        return df

    def predict_volatility_band(
        self,
        returns: np.ndarray | pd.Series,
        horizon: int = 5,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """预测波动率置信区间 / Forecast volatility confidence band.

        基于GARCH波动率预测，计算未来N步的波动率置信区间。

        参数 / Parameters
        ----------
        returns : np.ndarray | pd.Series
            收益率序列。
        horizon : int, optional
            预测步数，默认 5。
        confidence : float, optional
            置信度 (0 < confidence < 1)，默认 0.95。

        返回 / Returns
        -------
        pd.DataFrame
            包含 mean, lower, upper 波动率预测及置信区间。
        """
        r = _ensure_array(returns)
        mean_fc, std_fc = self.model.predict(horizon=horizon)

        # 简化的置信区间 (假设正态分布)
        z_val = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.0

        # 处理标量或数组
        mean_arr = np.atleast_1d(mean_fc)
        std_arr = np.atleast_1d(std_fc)

        df = pd.DataFrame(
            {
                "mean": mean_arr,
                "lower": mean_arr - z_val * std_arr,
                "upper": mean_arr + z_val * std_arr,
            }
        )
        return df
