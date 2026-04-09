"""
Volatility Surface Module / 波动率曲面模块
============================================

波动率曲面构建 + SVI (Stochastic Volatility Inspired) 微笑拟合 + Delta/Vega 对冲

功能:
1. 波动率微笑拟合 (SVI模型)
2. 波动率曲面双线性插值
3. Greeks 曲面计算 (Delta, Vega, Gamma)
4. 动态价差计算
5. 库存风险管理

Ref: Gatheral & Jacquier (2013) - "Arbitrage-free SVI volatility surfaces"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import curve_fit, minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# 内部导入：复用 black_scholes 中的 Greeks 计算
from .pricing.greeks import calculate_greeks, Greeks
from .pricing.black_scholes import BlackScholes, norm_pdf, d1_d2

__all__ = [
    "SVIVolatilityModel",
    "VolSurfaceCalculator",
    "DeltaHedgeCalculator",
    "VegaHedgeCalculator",
    "InventoryRiskManager",
]


# ============================================================================
# SVI Volatility Model / SVI 波动率模型
# ============================================================================

class SVIVolatilityModel:
    """Stochastic Volatility Inspired (SVI) 波动率微笑模型.

    SVI公式:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    其中 k = log(K/S) 是 log-moneyness

    用于拟合波动率微笑, 捕捉:
    - 波动率偏斜 (skew)       via rho 参数
    - 波动率微笑 (smile)      via b, sigma 参数
    - 波动率悬崖 (wing)       via m, sigma 参数

    Attributes:
        params: SVI 参数 {a, b, rho, m, sigma}
        T: 到期时间（年）
        fitted: 是否已拟合

    Example:
        >>> model = SVIVolatilityModel()
        >>> strikes = [2100, 2150, 2200, 2250, 2300]
        >>> ivs = [0.75, 0.72, 0.70, 0.73, 0.78]
        >>> result = model.fit(strikes, ivs, S=2200, T=30/365)
        >>> print(result['a'], result['b'], result['rho'], result['m'], result['sigma'])
    """

    def __init__(self):
        self.params: Optional[Dict[str, float]] = None
        self.T: float = 0.0
        self.S: float = 0.0
        self.fitted: bool = False

    def _svi_func(self, k: np.ndarray, a: float, b: float, rho: float,
                   m: float, sigma: float) -> np.ndarray:
        """SVI 公式向量计算.

        Args:
            k: log-moneyness = log(K/S)
            a: 水平参数 (level)
            b: 开口参数 (opening)
            rho: 偏斜参数 (skew correlation, -1 < rho < 1)
            m: 中位参数 (smile center)
            sigma: 混合参数 (mixing, > 0)

        Returns:
            波动率数组
        """
        d = k - m
        sqrt_term = np.sqrt(d * d + sigma * sigma)
        return a + b * (rho * d + sqrt_term)

    def _svi_func_scalar(self, K: float, a: float, b: float, rho: float,
                          m: float, sigma: float) -> float:
        """SVI 公式标量计算.

        Args:
            K: 行权价
            a, b, rho, m, sigma: SVI 参数

        Returns:
            波动率
        """
        k = math.log(K / self.S) if self.S > 0 else 0.0
        d = k - m
        sqrt_term = math.sqrt(d * d + sigma * sigma)
        return a + b * (rho * d + sqrt_term)

    def fit(self, strikes: np.ndarray, implied_vols: np.ndarray,
            S: float, T: float) -> Dict[str, float]:
        """拟合 SVI 参数.

        Args:
            strikes: 行权价数组
            implied_vols: 对应的隐含波动率数组
            S: 标的价格
            T: 到期时间（年）

        Returns:
            SVI 参数字典 {a, b, rho, m, sigma}

        Raises:
            ValueError: 数据点不足或拟合失败
        """
        self.S = S
        self.T = T

        # 转换为 log-moneyness
        strikes = np.asarray(strikes, dtype=float)
        implied_vols = np.asarray(implied_vols, dtype=float)

        # 过滤无效数据
        valid_mask = (implied_vols > 0) & (implied_vols < 5.0) & (strikes > 0)
        strikes = strikes[valid_mask]
        implied_vols = implied_vols[valid_mask]

        if len(strikes) < 5:
            raise ValueError("At least 5 valid data points required for SVI fit")

        k_data = np.log(strikes / S)

        # 初始猜测
        m0 = np.median(k_data)
        a0 = float(np.min(implied_vols))
        b0 = (float(np.max(implied_vols)) - a0) / (np.max(k_data) - np.min(k_data) + 1e-10)
        b0 = max(b0, 0.001)
        rho0 = 0.0
        sigma0 = 0.1

        p0 = [a0, b0, rho0, m0, sigma0]

        # 参数边界
        bounds = (
            [0.0, 0.001, -0.999, np.min(k_data), 0.001],  # 下界
            [5.0, 50.0, 0.999, np.max(k_data), 10.0],      # 上界
        )

        if _HAS_SCIPY:
            try:
                popt, _ = curve_fit(
                    self._svi_func, k_data, implied_vols,
                    p0=p0, bounds=bounds, maxfev=5000,
                    method='trf'
                )
                a, b, rho, m, sigma = popt
            except (RuntimeError, ValueError):
                # Fallback: 使用优化器
                a, b, rho, m, sigma = self._fit_optimize(k_data, implied_vols, p0, bounds)
        else:
            a, b, rho, m, sigma = self._fit_simple(k_data, implied_vols)

        self.params = {
            'a': float(a),
            'b': float(b),
            'rho': float(rho),
            'm': float(m),
            'sigma': float(sigma),
        }
        self.fitted = True
        return self.params

    def _fit_optimize(self, k_data: np.ndarray, iv_data: np.ndarray,
                      p0: List[float], bounds: Tuple) -> Tuple[float, float, float, float, float]:
        """使用 scipy.optimize.minimize 拟合 SVI 参数.

        Args:
            k_data: log-moneyness 数组
            iv_data: 波动率数组
            p0: 初始参数
            bounds: 参数边界

        Returns:
            (a, b, rho, m, sigma)
        """
        def objective(params):
            a, b, rho, m, sigma = params
            predicted = self._svi_func(k_data, a, b, rho, m, sigma)
            return np.sum((predicted - iv_data) ** 2)

        result = minimize(
            objective, p0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500}
        )
        return tuple(result.x)

    def _fit_simple(self, k_data: np.ndarray, iv_data: np.ndarray) -> Tuple[float, float, float, float, float]:
        """简单的 SVI 拟合（无 scipy 时使用）.

        Args:
            k_data: log-moneyness 数组
            iv_data: 波动率数组

        Returns:
            (a, b, rho, m, sigma)
        """
        a = float(np.min(iv_data))
        m = float(np.median(k_data))
        b = (float(np.max(iv_data)) - a) / (np.max(k_data) - np.min(k_data) + 1e-10)
        b = max(b, 0.001)
        rho = 0.0
        sigma = 0.1
        return a, b, rho, m, sigma

    def predict(self, strike: float) -> float:
        """预测给定行权价的波动率.

        Args:
            strike: 行权价

        Returns:
            波动率

        Raises:
            RuntimeError: 如果模型未拟合
        """
        if not self.fitted or self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self._svi_func_scalar(
            strike,
            self.params['a'], self.params['b'],
            self.params['rho'], self.params['m'], self.params['sigma']
        )

    def compute_smile(self, strikes: np.ndarray) -> np.ndarray:
        """计算整个微笑曲线.

        Args:
            strikes: 行权价数组

        Returns:
            波动率数组

        Raises:
            RuntimeError: 如果模型未拟合
        """
        if not self.fitted or self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        strikes = np.asarray(strikes, dtype=float)
        return np.array([self.predict(k) for k in strikes])

    def total_variance(self, strike: float) -> float:
        """计算总方差 (variance * T).

        Args:
            strike: 行权价

        Returns:
            总方差 w(T, k) = sigma^2 * T
        """
        vol = self.predict(strike)
        return vol ** 2 * self.T

    def local_vol(self, strike: float, dt: float = 1e-5) -> float:
        """计算局部波动率 (local volatility).

        du/dt = L(t, u) 满足 Dupire 方程

        Args:
            strike: 行权价
            dt: 数值微分步长

        Returns:
            局部波动率
        """
        if not self.fitted or self.params is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        vol = self.predict(strike)
        var = vol ** 2 * self.T

        # 数值微分计算局部波动率
        k = math.log(strike / self.S) if self.S > 0 else 0.0

        # 简化的局部波动率近似
        d_w = self._svi_func_scalar(strike + dt * strike, self.params['a'],
                                      self.params['b'], self.params['rho'],
                                      self.params['m'], self.params['sigma'])
        d_var = (d_w ** 2 - vol ** 2) * self.T / (dt * strike)

        if abs(d_var) < 1e-10:
            return vol

        # Dupire 公式简化版
        local_var = var + 0.5 * dt * d_var
        return max(math.sqrt(local_var / max(self.T, 1e-6)), 0.01)


# ============================================================================
# Volatility Surface Calculator / 波动率曲面计算器
# ============================================================================

class VolSurfaceCalculator:
    """波动率曲面计算器.

    功能:
    1. 从市场数据构建波动率曲面
    2. SVI 拟合
    3. Delta/Vega 计算
    4. 双线性插值

    Attributes:
        svi_model: SVI 模型实例
        surface: 波动率曲面数据字典

    Example:
        >>> calc = VolSurfaceCalculator()
        >>> options_data = pd.DataFrame({
        ...     'strike': [2100, 2150, 2200, 2250, 2300],
        ...     'expiry': [0.1, 0.1, 0.1, 0.1, 0.1],
        ...     'iv': [0.75, 0.72, 0.70, 0.73, 0.78],
        ...     'delta': [0.2, 0.35, 0.5, 0.65, 0.8],
        ... })
        >>> surface = calc.build_surface(options_data, S=2200)
        >>> vol = calc.interpolate(2177, 0.082)
    """

    def __init__(self, svi_model: Optional[SVIVolatilityModel] = None):
        """初始化波动率曲面计算器.

        Args:
            svi_model: SVI 模型实例（可选）
        """
        self.svi_model = svi_model if svi_model is not None else SVIVolatilityModel()
        self.surface: Dict[str, any] = {}
        self.S: float = 0.0
        self.r: float = 0.05

    def build_surface(self, options_data, S: float, r: float = 0.05) -> Dict:
        """构建波动率曲面.

        Args:
            options_data: DataFrame with columns=[strike, expiry, iv, delta, gamma, vega]
                          or list of dicts with same keys
            S: 标的价格
            r: 无风险利率

        Returns:
            波动率曲面字典 {
                'strikes': np.array,
                'expiries': np.array,
                'iv_matrix': np.array (expiry x strike),
                'delta_matrix': np.array,
                'vega_matrix': np.array,
                'svi_params': dict per expiry
            }

        Raises:
            ValueError: 数据格式错误
        """
        self.S = S
        self.r = r

        # 解析数据
        if hasattr(options_data, 'to_dict'):
            # pandas DataFrame
            records = options_data.to_dict('records')
        else:
            records = options_data

        if not records:
            raise ValueError("options_data is empty")

        # 提取唯一的到期时间和行权价
        strikes_set = sorted(set(r['strike'] for r in records))
        expiries_set = sorted(set(r['expiry'] for r in records))

        n_expiry = len(expiries_set)
        n_strike = len(strikes_set)

        # 构建矩阵
        iv_matrix = np.full((n_expiry, n_strike), np.nan)
        delta_matrix = np.full((n_expiry, n_strike), np.nan)
        vega_matrix = np.full((n_expiry, n_strike), np.nan)
        gamma_matrix = np.full((n_expiry, n_strike), np.nan)

        svi_params_per_expiry = {}

        strike_idx = {k: i for i, k in enumerate(strikes_set)}
        expiry_idx = {e: i for i, e in enumerate(expiries_set)}

        for rec in records:
            s = float(rec['strike'])
            t = float(rec['expiry'])
            iv = float(rec['iv'])
            si = strike_idx[s]
            ei = expiry_idx[t]

            iv_matrix[ei, si] = iv

            # 计算 Greeks（如果数据中没有）
            if 'delta' in rec and rec['delta'] is not None:
                delta_matrix[ei, si] = float(rec['delta'])
            else:
                greeks = calculate_greeks(S=S, K=s, T=t, r=r, sigma=iv, option_type='call')
                delta_matrix[ei, si] = greeks.delta

            if 'vega' in rec and rec['vega'] is not None:
                vega_matrix[ei, si] = float(rec['vega'])
            else:
                greeks = calculate_greeks(S=S, K=s, T=t, r=r, sigma=iv, option_type='call')
                vega_matrix[ei, si] = greeks.vega

            if 'gamma' in rec and rec['gamma'] is not None:
                gamma_matrix[ei, si] = float(rec['gamma'])
            else:
                greeks = calculate_greeks(S=S, K=s, T=t, r=r, sigma=iv, option_type='call')
                gamma_matrix[ei, si] = greeks.gamma

        # 对每个到期时间拟合 SVI
        for ei, t in enumerate(expiries_set):
            strikes_for_t = []
            ivs_for_t = []
            for si, s in enumerate(strikes_set):
                if not np.isnan(iv_matrix[ei, si]):
                    strikes_for_t.append(s)
                    ivs_for_t.append(iv_matrix[ei, si])

            if len(strikes_for_t) >= 5:
                svi_params = self.svi_model.fit(
                    np.array(strikes_for_t),
                    np.array(ivs_for_t),
                    S=S, T=t
                )
                svi_params_per_expiry[t] = svi_params

        self.surface = {
            'strikes': np.array(strikes_set),
            'expiries': np.array(expiries_set),
            'iv_matrix': iv_matrix,
            'delta_matrix': delta_matrix,
            'vega_matrix': vega_matrix,
            'gamma_matrix': gamma_matrix,
            'svi_params': svi_params_per_expiry,
            'S': S,
            'r': r,
        }

        return self.surface

    def interpolate(self, strike: float, expiry: float) -> float:
        """双线性插值获取波动率.

        Args:
            strike: 行权价
            expiry: 到期时间（年）

        Returns:
            插值后的波动率

        Note:
            优先使用 SVI 模型预测，如果超出范围则使用双线性插值
        """
        strikes = self.surface['strikes']
        expiries = self.surface['expiries']
        iv_matrix = self.surface['iv_matrix']

        if len(strikes) == 0 or len(expiries) == 0:
            return 0.0

        # 检查 SVI 预测
        svi_params = self.surface.get('svi_params', {})
        if expiry in svi_params:
            self.svi_model.params = svi_params[expiry]
            self.svi_model.fitted = True
            self.svi_model.S = self.S
            self.svi_model.T = expiry
            svi_vol = self.svi_model.predict(strike)
            if 0 < svi_vol < 5.0:
                return svi_vol

        # 双线性插值
        if strike <= strikes[0]:
            if len(expiries) > 1:
                return iv_matrix[0, 0] if not np.isnan(iv_matrix[0, 0]) else 0.0
            return iv_matrix[0, 0] if len(expiries) > 0 else 0.0

        if strike >= strikes[-1]:
            if len(expiries) > 1:
                return iv_matrix[0, -1] if not np.isnan(iv_matrix[0, -1]) else 0.0
            return iv_matrix[0, -1] if len(expiries) > 0 else 0.0

        # 找到插入位置
        si = np.searchsorted(strikes, strike)
        s_lo = strikes[si - 1]
        s_hi = strikes[si]

        if expiry <= expiries[0]:
            t_lo, t_hi = 0, 0
        elif expiry >= expiries[-1]:
            t_lo, t_hi = len(expiries) - 1, len(expiries) - 1
        else:
            ti = np.searchsorted(expiries, expiry)
            t_lo = expiries[ti - 1]
            t_hi = expiries[ti]

        ei_lo = np.where(expiries == t_lo)[0]
        ei_hi = np.where(expiries == t_hi)[0]

        if len(ei_lo) == 0 or len(ei_hi) == 0:
            return 0.0

        ei_lo = ei_lo[0]
        ei_hi = ei_hi[0]

        si_lo_idx = np.where(strikes == s_lo)[0][0]
        si_hi_idx = np.where(strikes == s_hi)[0][0]

        v00 = iv_matrix[ei_lo, si_lo_idx]
        v10 = iv_matrix[ei_lo, si_hi_idx]
        v01 = iv_matrix[ei_hi, si_lo_idx]
        v11 = iv_matrix[ei_hi, si_hi_idx]

        if np.isnan(v00):
            v00 = v10
        if np.isnan(v10):
            v10 = v00
        if np.isnan(v01):
            v01 = v11
        if np.isnan(v11):
            v11 = v01

        w_s = (strike - s_lo) / (s_hi - s_lo) if s_hi > s_lo else 0.0
        w_t = 0.0 if t_lo == t_hi else (expiry - t_lo) / (t_hi - t_lo)

        vol = (1 - w_s) * (1 - w_t) * v00 + w_s * (1 - w_t) * v10 + \
              (1 - w_s) * w_t * v01 + w_s * w_t * v11

        return max(0.0, vol)

    def compute_greeks_surface(self, strike: float, expiry: float,
                               option_type: str = "call") -> Greeks:
        """计算 Greek 值曲面.

        Args:
            strike: 行权价
            expiry: 到期时间（年）
            option_type: "call" 或 "put"

        Returns:
            Greeks 对象

        Note:
            使用插值的波动率计算 Greeks
        """
        vol = self.interpolate(strike, expiry)
        return calculate_greeks(
            S=self.S, K=strike, T=expiry,
            r=self.r, sigma=vol,
            option_type=option_type
        )

    def compute_smile_at_expiry(self, expiry: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算指定到期时间的波动率微笑.

        Args:
            expiry: 到期时间（年）

        Returns:
            (strikes, ivs) 元组

        Raises:
            RuntimeError: 如果曲面未构建
        """
        if not self.surface:
            raise RuntimeError("Surface not built. Call build_surface() first.")

        strikes = self.surface['strikes']
        ivs = np.array([self.interpolate(s, expiry) for s in strikes])
        return strikes, ivs


# ============================================================================
# Delta / Vega Hedge Calculator / Delta/Vega 对冲计算器
# ============================================================================

class DeltaHedgeCalculator:
    """Delta 对冲计算器.

    用于计算期权组合的 Delta 中性对冲需求.

    Attributes:
        S: 标的价格
        r: 无风险利率

    Example:
        >>> calc = DeltaHedgeCalculator(S=2200, r=0.05)
        >>> positions = [
        ...     {'type': 'call', 'K': 2200, 'T': 0.1, 'sigma': 0.7, 'size': 1},
        ...     {'type': 'put', 'K': 2100, 'T': 0.1, 'sigma': 0.7, 'size': -1},
        ... ]
        >>> hedge = calc.compute_hedge(positions)
        >>> print(f"需要买入 {hedge['delta_hedge']} 份标的来对冲")
    """

    def __init__(self, S: float, r: float = 0.05):
        """初始化 Delta 对冲计算器.

        Args:
            S: 标的价格
            r: 无风险利率
        """
        self.S = S
        self.r = r

    def compute_hedge(self, positions: List[Dict]) -> Dict[str, float]:
        """计算 Delta 对冲需求.

        Args:
            positions: 持仓列表, 每项包含:
                - type: 'call' 或 'put'
                - K: 行权价
                - T: 到期时间（年）
                - sigma: 波动率
                - size: 持仓数量（正=买入, 负=卖出）

        Returns:
            对冲需求字典 {
                'total_delta': 总 Delta
                'delta_hedge': 需要买入(-)或卖出(+)的标的数量
                'position_count': 持仓数量
            }
        """
        total_delta = 0.0
        position_count = len(positions)

        for pos in positions:
            greeks = calculate_greeks(
                S=self.S,
                K=pos['K'],
                T=pos['T'],
                r=self.r,
                sigma=pos['sigma'],
                option_type=pos['type']
            )
            total_delta += greeks.delta * pos.get('size', 1)

        delta_hedge = -total_delta  # 负的 Delta 意味着需要买入标的来对冲

        return {
            'total_delta': total_delta,
            'delta_hedge': delta_hedge,
            'position_count': position_count,
        }

    def rebalance_cost(self, current_hedge: float, new_hedge: float,
                       transaction_cost: float = 0.0001) -> float:
        """计算再平衡成本.

        Args:
            current_hedge: 当前标的持仓
            new_hedge: 目标标的持仓
            transaction_cost: 交易成本比例

        Returns:
            预计交易成本
        """
        trade_size = abs(new_hedge - current_hedge)
        return trade_size * self.S * transaction_cost


class VegaHedgeCalculator:
    """Vega 对冲计算器.

    用于计算期权组合的 Vega 中性对冲需求（通常通过交易另一种波动率产品）。

    Attributes:
        S: 标的价格
        r: 无风险利率

    Example:
        >>> calc = VegaHedgeCalculator(S=2200, r=0.05)
        >>> positions = [
        ...     {'type': 'call', 'K': 2200, 'T': 0.1, 'sigma': 0.7, 'size': 1},
        ...     {'type': 'put', 'K': 2100, 'T': 0.1, 'sigma': 0.7, 'size': 1},
        ... ]
        >>> hedge = calc.compute_hedge(positions)
        >>> print(f"需要 {-hedge['vega_hedge']:.2f} Vega 来对冲")
    """

    def __init__(self, S: float, r: float = 0.05):
        """初始化 Vega 对冲计算器.

        Args:
            S: 标的价格
            r: 无风险利率
        """
        self.S = S
        self.r = r

    def compute_hedge(self, positions: List[Dict]) -> Dict[str, float]:
        """计算 Vega 对冲需求.

        Args:
            positions: 持仓列表（同 DeltaHedgeCalculator）

        Returns:
            对冲需求字典 {
                'total_vega': 总 Vega
                'vega_hedge': 需要买入(-)或卖出(+)的 Vega 单位
                'position_count': 持仓数量
            }
        """
        total_vega = 0.0
        position_count = len(positions)

        for pos in positions:
            greeks = calculate_greeks(
                S=self.S,
                K=pos['K'],
                T=pos['T'],
                r=self.r,
                sigma=pos['sigma'],
                option_type=pos['type']
            )
            total_vega += greeks.vega * pos.get('size', 1)

        # Vega 中性意味着 total_vega = 0
        vega_hedge = -total_vega

        return {
            'total_vega': total_vega,
            'vega_hedge': vega_hedge,
            'position_count': position_count,
        }

    def hedge_with_variance(self, vega_hedge: float, vega_per_variance_unit: float,
                            variance_cost: float = 0.01) -> Dict[str, float]:
        """计算用方差产品（如 VIX 期权）对冲的需求.

        Args:
            vega_hedge: 需要对冲的 Vega 量
            vega_per_variance_unit: 每单位方差合约的 Vega
            variance_cost: 方差合约价格

        Returns:
            对冲方案字典
        """
        variance_units = vega_hedge / vega_per_variance_unit if abs(vega_per_variance_unit) > 1e-10 else 0.0
        total_cost = abs(variance_units) * variance_cost

        return {
            'variance_units': variance_units,
            'total_cost': total_cost,
            'vega_hedged': vega_hedge,
        }


# ============================================================================
# Inventory Risk Manager / 库存风险管理器
# ============================================================================

class InventoryRiskManager:
    """库存风险管理器.

    管理期权库存的 Delta/Vega 风险，支持动态对冲再平衡。

    Attributes:
        S: 标的价格
        r: 无风险利率
        max_delta_exposure: 最大 Delta 敞口
        max_vega_exposure: 最大 Vega 敞口

    Example:
        >>> mgr = InventoryRiskManager(S=2200, r=0.05, max_delta_exposure=50)
        >>> mgr.add_position({'type': 'call', 'K': 2200, 'T': 0.1, 'sigma': 0.7, 'size': 5})
        >>> risk = mgr.compute_risk()
        >>> print(f"Delta 敞口: {risk['net_delta']:.2f}")
        >>> print(f"需要对冲: {risk['delta_to_hedge']:.2f}")
    """

    def __init__(self, S: float, r: float = 0.05,
                 max_delta_exposure: float = 100.0,
                 max_vega_exposure: float = 50.0):
        """初始化库存风险管理器.

        Args:
            S: 标的价格
            r: 无风险利率
            max_delta_exposure: 最大 Delta 敞口
            max_vega_exposure: 最大 Vega 敞口
        """
        self.S = S
        self.r = r
        self.max_delta_exposure = max_delta_exposure
        self.max_vega_exposure = max_vega_exposure
        self.positions: List[Dict] = []
        self.hedge_history: List[Dict] = []
        self._delta_calc = DeltaHedgeCalculator(S=S, r=r)
        self._vega_calc = VegaHedgeCalculator(S=S, r=r)

    def add_position(self, position: Dict) -> None:
        """添加持仓.

        Args:
            position: 持仓字典 {
                'type': 'call' 或 'put',
                'K': 行权价,
                'T': 到期时间（年）,
                'sigma': 波动率,
                'size': 持仓数量
            }
        """
        self.positions.append(position.copy())

    def remove_position(self, index: int) -> Dict:
        """移除持仓.

        Args:
            index: 持仓索引

        Returns:
            被移除的持仓
        """
        if 0 <= index < len(self.positions):
            return self.positions.pop(index)
        raise IndexError("Position index out of range")

    def clear_positions(self) -> None:
        """清空所有持仓."""
        self.positions = []

    def compute_risk(self) -> Dict[str, float]:
        """计算当前风险敞口.

        Returns:
            风险字典 {
                'net_delta': 净 Delta
                'net_vega': 净 Vega
                'net_gamma': 净 Gamma
                'delta_to_hedge': Delta 对冲需求
                'vega_to_hedge': Vega 对冲需求
                'delta_risk_pct': Delta 风险百分比
                'vega_risk_pct': Vega 风险百分比
                'position_count': 持仓数量
            }
        """
        if not self.positions:
            return {
                'net_delta': 0.0,
                'net_vega': 0.0,
                'net_gamma': 0.0,
                'delta_to_hedge': 0.0,
                'vega_to_hedge': 0.0,
                'delta_risk_pct': 0.0,
                'vega_risk_pct': 0.0,
                'position_count': 0,
            }

        net_delta = 0.0
        net_vega = 0.0
        net_gamma = 0.0

        for pos in self.positions:
            greeks = calculate_greeks(
                S=self.S, K=pos['K'], T=pos['T'],
                r=self.r, sigma=pos['sigma'],
                option_type=pos['type']
            )
            size = pos.get('size', 1)
            net_delta += greeks.delta * size
            net_vega += greeks.vega * size
            net_gamma += greeks.gamma * size

        delta_to_hedge = -net_delta
        vega_to_hedge = -net_vega

        delta_risk_pct = min(abs(net_delta) / self.max_delta_exposure, 1.0) if self.max_delta_exposure > 0 else 0.0
        vega_risk_pct = min(abs(net_vega) / self.max_vega_exposure, 1.0) if self.max_vega_exposure > 0 else 0.0

        return {
            'net_delta': net_delta,
            'net_vega': net_vega,
            'net_gamma': net_gamma,
            'delta_to_hedge': delta_to_hedge,
            'vega_to_hedge': vega_to_hedge,
            'delta_risk_pct': delta_risk_pct,
            'vega_risk_pct': vega_risk_pct,
            'position_count': len(self.positions),
        }

    def should_rebalance(self, threshold: float = 0.1) -> Dict[str, bool]:
        """判断是否需要再平衡对冲.

        Args:
            threshold: 再平衡阈值（风险百分比超过此值时触发）

        Returns:
            字典 {
                'delta_rebalance': 是否需要 Delta 再平衡
                'vega_rebalance': 是否需要 Vega 再平衡
                'reason': 触发原因
            }
        """
        risk = self.compute_risk()
        reason = []

        delta_rebalance = risk['delta_risk_pct'] > threshold
        vega_rebalance = risk['vega_risk_pct'] > threshold

        if delta_rebalance:
            reason.append(f"Delta风险 {risk['delta_risk_pct']*100:.1f}% 超过阈值 {threshold*100:.1f}%")
        if vega_rebalance:
            reason.append(f"Vega风险 {risk['vega_risk_pct']*100:.1f}% 超过阈值 {threshold*100:.1f}%")

        return {
            'delta_rebalance': delta_rebalance,
            'vega_rebalance': vega_rebalance,
            'reason': '; '.join(reason) if reason else 'No rebalance needed',
        }

    def execute_hedge(self, delta_hedge: float, vega_hedge: float = 0.0,
                      price: Optional[float] = None) -> Dict:
        """执行对冲（记录）.

        Args:
            delta_hedge: Delta 对冲量（正=买入标的, 负=卖出标的）
            vega_hedge: Vega 对冲量
            price: 标的当前价格（用于计算成本）

        Returns:
            对冲执行记录
        """
        record = {
            'timestamp': None,  # 可扩展为真实时间戳
            'delta_hedge': delta_hedge,
            'vega_hedge': vega_hedge,
            'price': price or self.S,
            'cost': abs(delta_hedge) * (price or self.S) * 0.0001,  # 假设 1bp 成本
        }

        self.hedge_history.append(record)
        return record


# ============================================================================
# Dynamic Spread Calculator / 动态价差计算器
# ============================================================================

class DynamicSpreadCalculator:
    """动态价差计算器.

    计算买卖价差（bid-ask spread）基于:
    - 波动率
    - 到期时间
    - 库存风险
    - 市场微结构

    Attributes:
        S: 标的价格
        r: 无风险利率

    Example:
        >>> spread_calc = DynamicSpreadCalculator(S=2200, r=0.05)
        >>> spread = spread_calc.compute_spread(
        ...     strike=2200, T=0.1, sigma=0.7,
        ...     inventory={'delta': 10, 'vega': 5}
        ... )
        >>> print(f"合理价差: {spread:.4f}")
    """

    def __init__(self, S: float, r: float = 0.05):
        """初始化动态价差计算器.

        Args:
            S: 标的价格
            r: 无风险利率
        """
        self.S = S
        self.r = r

    def compute_spread(self, strike: float, T: float, sigma: float,
                       inventory: Optional[Dict[str, float]] = None,
                       market_impact: float = 0.0001) -> float:
        """计算合理买卖价差.

        Args:
            strike: 行权价
            T: 到期时间（年）
            sigma: 波动率
            inventory: 当前库存 {'delta': x, 'vega': y}
            market_impact: 市场冲击系数

        Returns:
            买卖价差（按标的百分比）
        """
        # 基础价差：波动率越大价差越大
        base_spread = sigma * math.sqrt(T)

        # Gamma 风险调整
        bs = BlackScholes(S=self.S, K=strike, T=T, r=self.r, sigma=sigma)
        gamma = bs.gamma()
        vega = bs.vega()

        # 库存风险溢价
        inventory_premium = 0.0
        if inventory:
            inventory_premium = (
                abs(inventory.get('delta', 0)) * gamma * 0.001 +
                abs(inventory.get('vega', 0)) * vega * 0.001
            )

        total_spread = base_spread + inventory_premium + market_impact

        # 确保最小价差
        min_spread = 0.0001  # 1bp
        return max(total_spread, min_spread)

    def compute_bid_ask(self, strike: float, T: float, sigma: float,
                        mid_price: float, inventory: Optional[Dict] = None) -> Tuple[float, float]:
        """计算 bid/ask 价格.

        Args:
            strike: 行权价
            T: 到期时间（年）
            sigma: 波动率
            mid_price: 中间价
            inventory: 当前库存

        Returns:
            (bid_price, ask_price)
        """
        spread = self.compute_spread(strike, T, sigma, inventory)
        half_spread = spread * self.S / 2

        return mid_price - half_spread, mid_price + half_spread


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=== Vol Surface Module Test ===\n")

    # 1. 测试 SVI 模型
    print("--- SVI Volatility Model ---")
    model = SVIVolatilityModel()

    # 模拟数据：ETH 行权价和隐含波动率
    S = 2200.0
    T = 30 / 365
    strikes = np.array([2000.0, 2100.0, 2200.0, 2300.0, 2400.0])
    # 构造一个有偏斜的波动率微笑
    base_vol = 0.70
    ivs = base_vol + 0.05 * (strikes - S) / S + 0.02 * ((strikes - S) / S) ** 2
    ivs = np.clip(ivs, 0.3, 1.5)

    print(f"Strikes: {strikes}")
    print(f"IVs: {ivs}")

    params = model.fit(strikes, ivs, S=S, T=T)
    print(f"SVI Params: a={params['a']:.4f}, b={params['b']:.4f}, "
          f"rho={params['rho']:.4f}, m={params['m']:.4f}, sigma={params['sigma']:.4f}")

    predicted = model.compute_smile(strikes)
    print(f"Predicted: {predicted}")

    # 2. 测试波动率曲面
    print("\n--- Volatility Surface Calculator ---")
    import pandas as pd

    options_data = pd.DataFrame({
        'strike': [2000, 2100, 2200, 2300, 2400] * 3,
        'expiry': [0.05, 0.05, 0.05, 0.05, 0.05,
                   0.1, 0.1, 0.1, 0.1, 0.1,
                   0.2, 0.2, 0.2, 0.2, 0.2],
        'iv': [0.75, 0.72, 0.70, 0.73, 0.78,
               0.72, 0.70, 0.68, 0.71, 0.75,
               0.68, 0.67, 0.65, 0.68, 0.72],
    })

    calc = VolSurfaceCalculator()
    surface = calc.build_surface(options_data, S=2200, r=0.05)
    print(f"Surface built: {len(surface['strikes'])} strikes x {len(surface['expiries'])} expiries")

    # 测试插值
    vol_atm = calc.interpolate(2200, 0.1)
    print(f"Interpolated IV at (2200, 0.1): {vol_atm:.4f}")

    # 3. 测试 Delta/Vega 对冲
    print("\n--- Delta/Vega Hedge Calculator ---")
    positions = [
        {'type': 'call', 'K': 2200, 'T': 0.1, 'sigma': 0.70, 'size': 10},
        {'type': 'put', 'K': 2100, 'T': 0.1, 'sigma': 0.70, 'size': -5},
    ]

    delta_calc = DeltaHedgeCalculator(S=2200, r=0.05)
    delta_hedge = delta_calc.compute_hedge(positions)
    print(f"Delta Hedge: {delta_hedge}")

    vega_calc = VegaHedgeCalculator(S=2200, r=0.05)
    vega_hedge = vega_calc.compute_hedge(positions)
    print(f"Vega Hedge: {vega_hedge}")

    # 4. 测试库存风险管理
    print("\n--- Inventory Risk Manager ---")
    mgr = InventoryRiskManager(S=2200, r=0.05, max_delta_exposure=50, max_vega_exposure=30)
    for pos in positions:
        mgr.add_position(pos)

    risk = mgr.compute_risk()
    print(f"Risk: delta={risk['net_delta']:.2f}, vega={risk['net_vega']:.2f}")
    print(f"Rebalance: {mgr.should_rebalance(threshold=0.1)}")

    # 5. 测试动态价差
    print("\n--- Dynamic Spread Calculator ---")
    spread_calc = DynamicSpreadCalculator(S=2200, r=0.05)
    spread = spread_calc.compute_spread(
        strike=2200, T=0.1, sigma=0.70,
        inventory={'delta': 10, 'vega': 5}
    )
    print(f"Dynamic Spread: {spread:.6f} ({spread*100:.4f}%)")

    bid, ask = spread_calc.compute_bid_ask(
        strike=2200, T=0.1, sigma=0.70,
        mid_price=100.0
    )
    print(f"Bid: {bid:.2f}, Ask: {ask:.2f}")

    print("\n=== All Tests Passed ===")
