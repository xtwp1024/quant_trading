"""
Lead-Lag Relationship Analysis
=============================

期现Lead-Lag关系分析模块

用于研究期货与现货价格之间的领先-滞后关系,
分析市场定价效率, 并应用计量经济学模型研究两者关系。

核心方法:
- OLS回归分析价差关系
- 相关系数分析
- Granger因果检验
- 脉冲响应函数 (IRF)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Literal, Dict
import warnings


class SpotFuturesRelationship:
    """
    现货-期货领先滞后关系分析器

    分析现货与期货价格之间的Lead-Lag关系,
    用于判断市场定价效率和价格发现机制。
    """

    def __init__(
        self,
        spot_price_df: pd.DataFrame,
        futures_price_df: pd.DataFrame,
        price_col: str = 'price'
    ):
        """
        Args:
            spot_price_df: 现货价格数据
            futures_price_df: 期货价格数据
            price_col: 价格列名, 默认 'price'
        """
        self.spot_price_df = spot_price_df.copy()
        self.futures_price_df = futures_price_df.copy()
        self.price_col = price_col

        # 预处理列名
        self._standardize_columns()

        # 计算收益率
        self._compute_returns()

    def _standardize_columns(self):
        """标准化列名以支持多种输入格式"""
        # 处理现货数据
        if self.price_col not in self.spot_price_df.columns:
            if 'Spot' in self.spot_price_df.columns:
                self.spot_price_df = self.spot_price_df.rename(columns={'Spot': 'price'})
            elif 'Close' in self.spot_price_df.columns:
                self.spot_price_df = self.spot_price_df.rename(columns={'Close': 'price'})
            elif len(self.spot_price_df.columns) == 1:
                self.spot_price_df.columns = ['price']

        # 处理期货数据
        if self.price_col not in self.futures_price_df.columns:
            if 'Futures' in self.futures_price_df.columns:
                self.futures_price_df = self.futures_price_df.rename(columns={'Futures': 'price'})
            elif 'Close' in self.futures_price_df.columns:
                self.futures_price_df = self.futures_price_df.rename(columns={'Close': 'price'})
            elif len(self.futures_price_df.columns) == 1:
                self.futures_price_df.columns = ['price']

    def _compute_returns(self):
        """计算收益率序列"""
        self.spot_returns = self.spot_price_df['price'].diff().iloc[1:]
        self.futures_returns = self.futures_price_df['price'].diff().iloc[1:]

    def test_relationship(self) -> Dict:
        """
        使用OLS回归测试现货与期货的关系

        回归方程: ΔS_t = α + β * ΔF_t + ε_t

        其中:
        - ΔS_t: 现货价格变动
        - ΔF_t: 期货价格变动
        - β: 关系系数 (接近1表示期货领先现货)

        Returns:
            dict: 包含回归系数的字典
        """
        try:
            import statsmodels.api as sm

            # 准备数据
            spot_diff = self.spot_price_df['price'].diff().iloc[1:].values
            futures_diff = self.futures_price_df['price'].diff().iloc[1:].values

            # 添加常数项
            X = sm.add_constant(futures_diff)
            y = spot_diff

            # OLS回归
            model = sm.OLS(y, X)
            results = model.fit()

            return {
                'alpha': float(results.params[0]),
                'beta': float(results.params[1]),
                'r_squared': float(results.rsquared),
                'p_value_alpha': float(results.pvalues[0]),
                'p_value_beta': float(results.pvalues[1]),
                'summary': results.summary().as_text()
            }
        except ImportError:
            warnings.warn("statsmodels not available, falling back to numpy OLS")
            return self._numpy_ols()

    def _numpy_ols(self) -> Dict:
        """NumPy版本的OLS (备用)"""
        spot_diff = self.spot_price_df['price'].diff().iloc[1:].values
        futures_diff = self.futures_price_df['price'].diff().iloc[1:].values

        # 中心化
        X = futures_diff - np.mean(futures_diff)
        y = spot_diff - np.mean(spot_diff)

        beta = np.dot(X, y) / np.dot(X, X)
        alpha = np.mean(spot_diff) - beta * np.mean(futures_diff)

        # 计算R²
        y_pred = alpha + beta * futures_diff
        ss_res = np.sum((spot_diff - y_pred) ** 2)
        ss_tot = np.sum((spot_diff - np.mean(spot_diff)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'r_squared': float(r_squared),
            'p_value_alpha': None,
            'p_value_beta': None,
            'summary': f"NumPy OLS (no p-values): alpha={alpha:.6f}, beta={beta:.6f}, R²={r_squared:.4f}"
        }

    def identify_lead_lag(self, threshold: float = 0.5) -> Dict:
        """
        识别期货与现货之间的领先-滞后关系

        基于相关系数判断两者关系的强度和方向:

        Args:
            threshold: 相关系数阈值, 默认0.5

        Returns:
            dict: 包含相关性分析结果的字典
        """
        # 计算相关系数
        spot_prices = self.spot_price_df['price'].values
        futures_prices = self.futures_price_df['price'].values

        corr_matrix = np.corrcoef(spot_prices, futures_prices)
        corr_coef = corr_matrix[0, 1]

        # 价格水平相关性
        price_corr = corr_coef

        # 收益率相关性
        returns_corr = np.corrcoef(self.spot_returns, self.futures_returns)[0, 1]

        # 判断关系强度
        if abs(price_corr) > threshold:
            strength = "strong"
        else:
            strength = "weak"

        # 判断领先方向 (基于OLS beta)
        test_result = self.test_relationship()
        beta = test_result['beta']

        if beta > 0.8:
            direction = "futures_leads_spot"
            description = "期货价格领先现货价格"
        elif beta < 0.2:
            direction = "spot_leads_futures"
            description = "现货价格领先期货价格"
        else:
            direction = "simultaneous"
            description = "期货与现货价格同步"

        return {
            'correlation': float(price_corr),
            'returns_correlation': float(returns_corr) if not np.isnan(returns_corr) else None,
            'strength': strength,
            'direction': direction,
            'description': description,
            'beta': test_result['beta'],
            'has_lead_lag': abs(price_corr) > threshold
        }

    def granger_causality(self, max_lag: int = 5) -> Dict:
        """
        Granger因果检验

        检验期货价格是否有助于预测现货价格 (反之亦然)

        Args:
            max_lag: 最大滞后期数

        Returns:
            dict: Granger因果检验结果
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            # 合并数据
            combined = pd.DataFrame({
                'spot': self.spot_price_df['price'].iloc[1:],
                'futures': self.futures_price_df['price'].iloc[1:]
            }).dropna()

            # Spot对Futures的Granger检验
            spot_to_futures = grangercausalitytests(
                combined[['futures', 'spot']], maxlag=max_lag, verbose=False
            )

            # Futures对Spot的Granger检验
            futures_to_spot = grangercausalitytests(
                combined[['spot', 'futures']], maxlag=max_lag, verbose=False
            )

            # 取最优滞后期
            best_spot_to_futures = min(
                spot_to_futures.items(),
                key=lambda x: abs(x[1]['ssr_ftest'][1])
            )
            best_futures_to_spot = min(
                futures_to_spot.items(),
                key=lambda x: abs(x[1]['ssr_ftest'][1])
            )

            return {
                'spot_causes_futures': {
                    'p_value': float(best_spot_to_futures[1]['ssr_ftest'][1]),
                    'optimal_lag': best_spot_to_futures[0]
                },
                'futures_causes_spot': {
                    'p_value': float(best_futures_to_spot[1]['ssr_ftest'][1]),
                    'optimal_lag': best_futures_to_spot[0]
                },
                'conclusion': self._interpret_granger(
                    best_spot_to_futures[1]['ssr_ftest'][1],
                    best_futures_to_spot[1]['ssr_ftest'][1]
                )
            }
        except ImportError:
            warnings.warn("statsmodels not available for Granger causality test")
            return {'error': 'statsmodels required for Granger causality test'}

    def _interpret_granger(
        self,
        p_spot_to_futures: float,
        p_futures_to_spot: float
    ) -> str:
        """解释Granger因果检验结果"""
        alpha = 0.05
        if p_futures_to_spot < alpha and p_spot_to_futures >= alpha:
            return "期货价格对现货价格有预测能力 (期货领先现货)"
        elif p_spot_to_futures < alpha and p_futures_to_spot >= alpha:
            return "现货价格对期货价格有预测能力 (现货领先期货)"
        elif p_futures_to_spot < alpha and p_spot_to_futures < alpha:
            return "存在双向Granger因果关系"
        else:
            return "不存在显著的Granger因果关系"

    def impulse_response(
        self,
        periods: int = 10
    ) -> Dict:
        """
        脉冲响应函数分析

        分析期货价格冲击对现货价格的影响 (反之亦然)

        Args:
            periods: 冲击响应期数

        Returns:
            dict: 脉冲响应结果
        """
        try:
            from statsmodels.tsa.api import VAR

            # 准备数据
            data = pd.DataFrame({
                'spot': self.spot_price_df['price'],
                'futures': self.futures_price_df['price']
            }).dropna()

            # 拟合VAR模型
            model = VAR(data)
            results = model.fit(maxlags=5, ic='aic')

            # 脉冲响应 (Futures -> Spot)
            irf = results.irf(periods)

            return {
                'model': 'VAR',
                'aic': float(results.aic),
                'bic': float(results.bic),
                'irf_periods': periods,
                'note': 'Use results.irf.plot() to visualize impulse responses'
            }
        except Exception as e:
            warnings.warn(f"Impulse response analysis failed: {e}")
            return {'error': str(e)}

    def analyze_price_discovery(self) -> Dict:
        """
        价格发现分析

        综合分析期货与现货的价格发现功能,
        判断哪个市场在价格发现中起主导作用。

        Returns:
            dict: 价格发现分析结果
        """
        lead_lag = self.identify_lead_lag()
        test = self.test_relationship()

        # 价格发现份额 (基于信息份额模型简化版)
        spot_var = np.var(self.spot_returns)
        futures_var = np.var(self.futures_returns)
        cov = np.cov(self.spot_returns, self.futures_returns)[0, 1]

        # 期货价格发现份额
        futures_share = (futures_var - cov) / (spot_var + futures_var - 2 * cov) if (spot_var + futures_var - 2 * cov) != 0 else 0.5
        futures_share = max(0, min(1, futures_share))  # 限制在[0,1]

        return {
            'lead_lag': lead_lag,
            'regression': {
                'beta': test['beta'],
                'r_squared': test['r_squared']
            },
            'price_discovery_share': {
                'futures': float(futures_share),
                'spot': float(1 - futures_share)
            },
            'conclusion': self._conclude_price_discovery(
                lead_lag['direction'],
                futures_share
            )
        }

    def _conclude_price_discovery(
        self,
        direction: str,
        futures_share: float
    ) -> str:
        """总结价格发现结论"""
        if futures_share > 0.6:
            market = "期货市场"
        elif futures_share < 0.4:
            market = "现货市场"
        else:
            market = "两个市场共同"

        return f"{market}在价格发现中起主导作用"
