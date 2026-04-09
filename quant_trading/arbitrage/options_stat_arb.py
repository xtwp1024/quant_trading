"""
期权统计套利模块 (Options Statistical Arbitrage Module)
=========================================================

期权统计套利核心组件：
- IVMeanReversionStrategy: 隐含波动率均值回复策略
- OptionsStatArb: 期权统计套利框架

核心逻辑：
1. IV均值回复：检测IV相对历史均值的偏离，偏高则卖出期权
2. Delta对冲：消除方向风险，保留波动率收益
3. 波动率曲面交易：不同行权价之间的套利机会
4. 统计套利信号：基于历史分布的均值回复信号

用法：
    from quant_trading.arbitrage.options_stat_arb import IVMeanReversionStrategy, OptionsStatArb

    strategy = IVMeanReversionStrategy(lookback=20, iv_threshold=0.30)
    signal = strategy.generate_signal(current_iv, iv_history)
"""

import math
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

from quant_trading.options.pricing.black_scholes import (
    BlackScholes,
    bs_price,
    bs_greeks,
    delta as bs_delta,
    gamma as bs_gamma,
    vega as bs_vega,
    implied_volatility,
)


# ============================================================================
# IV Mean Reversion Strategy
# ============================================================================

class IVMeanReversionStrategy:
    """
    隐含波动率均值回复策略 (Implied Volatility Mean Reversion Strategy)

    逻辑:
    1. 检测IV相对历史均值的偏离
    2. IV偏高 → 卖出期权 (预期回归)
    3. IV偏低 → 买入期权 (预期回归)
    4. Delta对冲消除方向风险
    5. IV回归 → 平仓获利

    策略特点:
    - 适用于波动率具有均值回复特性的市场
    - 通过Delta对冲实现市场中性的波动率策略
    - 可配置离散或连续对冲频率

    Attributes:
        lookback: 历史IV回顾窗口期（交易日）
        iv_threshold: IV偏离阈值（超过此阈值触发交易）
        hedge_freq: 对冲频率 ('discrete' 离散 或 'continuous' 连续)
        zscore_ewma_span: 计算z-score的EWMA平滑窗口
        target_delta: 目标Delta值（通常为0实现市场中立）
    """

    def __init__(
        self,
        lookback: int = 20,
        iv_threshold: float = 0.30,
        hedge_freq: Literal["discrete", "continuous"] = "continuous",
        zscore_ewma_span: int = 12,
        target_delta: float = 0.0,
    ):
        """
        初始化IV均值回复策略

        Args:
            lookback: 历史IV回顾窗口期（默认20个交易日）
            iv_threshold: IV偏离阈值，30%表示超过30%偏离时触发交易
            hedge_freq: 对冲频率，'discrete'为定期对冲，'continuous'为连续对冲
            zscore_ewma_span: 计算z-score时使用的EWMA平滑参数
            target_delta: 目标Delta值，0表示市场中立
        """
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        if iv_threshold <= 0:
            raise ValueError("iv_threshold must be positive")
        if zscore_ewma_span < 1:
            raise ValueError("zscore_ewma_span must be at least 1")

        self.lookback = lookback
        self.iv_threshold = iv_threshold
        self.hedge_freq = hedge_freq
        self.zscore_ewma_span = zscore_ewma_span
        self.target_delta = target_delta

        # 内部状态
        self._iv_history: Optional[np.ndarray] = None
        self._position: Optional[dict] = None

    def compute_iv_zscore(
        self,
        current_iv: float,
        iv_history: np.ndarray,
        use_ewma: bool = True,
    ) -> float:
        """
        计算IV的Z-score（标准分数）

        Z-score = (当前IV - 均值) / 标准差
        用于衡量当前IV相对于历史分布的偏离程度

        Args:
            current_iv: 当前隐含波动率（小数形式，如0.30表示30%）
            iv_history: 历史IV数组
            use_ewma: 是否使用EWMA计算标准差（默认True，更敏感）

        Returns:
            Z-score值：
            - > 2.0: IV显著偏高，可能做空波动率
            - < -2.0: IV显著偏低，可能做多波动率
            - 接近0: IV处于正常区间
        """
        if len(iv_history) < 2:
            return 0.0

        # 使用最近lookback个数据点
        history = iv_history[-self.lookback :]
        if len(history) < 2:
            return 0.0

        mean_iv = np.mean(history)

        if use_ewma:
            # EWMA标准差（对近期数据赋予更高权重）
            ewma = pd.Series(history).ewm(span=self.zscore_ewma_span).mean().iloc[-1]
            # 使用EWMA附近的波动作为标准差估计
            std_iv = np.std(history[-self.zscore_ewma_span :]) if len(history) >= self.zscore_ewma_span else np.std(history)
        else:
            std_iv = np.std(history)

        if std_iv < 1e-10:
            return 0.0

        zscore = (current_iv - mean_iv) / std_iv
        return float(zscore)

    def generate_signal(
        self,
        current_iv: float,
        iv_history: np.ndarray,
    ) -> Dict[str, any]:
        """
        生成交易信号

        基于IV偏离程度生成买入或卖出波动率的信号

        Args:
            current_iv: 当前隐含波动率
            iv_history: 历史IV序列

        Returns:
            信号字典:
            {
                'action': 'sell_iv' | 'buy_iv' | 'neutral',
                'size': float,          # 仓位大小
                'hedge_ratio': float,  # 对冲比率
                'zscore': float,       # 当前IV的z-score
                'mean_iv': float,      # 历史均值
                'reason': str          # 信号生成原因
            }
        """
        zscore = self.compute_iv_zscore(current_iv, iv_history)
        history = iv_history[-self.lookback :] if len(iv_history) >= self.lookback else iv_history
        mean_iv = np.mean(history) if len(history) > 0 else current_iv

        # 根据z-score生成信号
        if zscore > self.iv_threshold:
            # IV显著偏高 -> 卖出期权（做空波动率）
            size = min(zscore / self.iv_threshold, 3.0)  # 限制最大仓位
            action = "sell_iv"
            reason = f"IV zscore={zscore:.2f} exceeds threshold={self.iv_threshold}, IV={current_iv*100:.1f}% vs mean={mean_iv*100:.1f}%"

        elif zscore < -self.iv_threshold:
            # IV显著偏低 -> 买入期权（做多波动率）
            size = min(abs(zscore) / self.iv_threshold, 3.0)
            action = "buy_iv"
            reason = f"IV zscore={zscore:.2f} below threshold=-{self.iv_threshold}, IV={current_iv*100:.1f}% vs mean={mean_iv*100:.1f}%"

        else:
            # IV处于正常区间 -> 中性
            action = "neutral"
            size = 0.0
            reason = f"IV zscore={zscore:.2f} within threshold=+/-{self.iv_threshold}"

        # 对冲比率：使组合Delta接近target_delta
        hedge_ratio = 1.0 - self.target_delta  # 简单线性对冲

        return {
            "action": action,
            "size": float(size),
            "hedge_ratio": float(hedge_ratio),
            "zscore": float(zscore),
            "mean_iv": float(mean_iv),
            "reason": reason,
        }

    def compute_hedge(
        self,
        option_delta: float,
        position_size: float,
        current_hedge: float = 0.0,
    ) -> float:
        """
        计算Delta对冲数量

        用于消除期权组合的方向性风险

        Args:
            option_delta: 期权的Delta值
            position_size: 期权仓位数量（正数为多头，负数为空头）
            current_hedge: 当前标的资产的对冲仓位

        Returns:
            需要的标的资产对冲数量
            正数表示需要买入标的，负数表示需要卖出标的
        """
        # 期权组合的Delta = position_size * option_delta
        portfolio_delta = position_size * option_delta

        # 目标Delta = target_delta * |position_size|（按比例）
        target_delta_portfolio = self.target_delta * abs(position_size)

        # 需要调整的对冲量
        hedge_needed = target_delta_portfolio - portfolio_delta - current_hedge

        return float(hedge_needed)

    def compute_position_pnl(
        self,
        initial_iv: float,
        current_iv: float,
        position_size: float,
        vega: float,
        time_days: int = 0,
        theta_decay: float = 0.0,
    ) -> float:
        """
        计算仓位盈亏

        Args:
            initial_iv: 建仓时的IV
            current_iv: 当前IV
            position_size: 仓位数量
            vega: 期权Vega（每1%波动率变化的美元影响）
            time_days: 经过的交易天数
            theta_decay: 每日Theta衰减

        Returns:
            预估盈亏（美元）
        """
        # IV变化带来的盈亏
        iv_change = current_iv - initial_iv
        iv_pnl = position_size * vega * iv_change * 100  # vega是按1%计算的

        # 时间价值衰减
        theta_pnl = position_size * theta_decay * time_days

        return iv_pnl - theta_pnl

    def update_state(self, current_iv: float) -> None:
        """更新内部状态（IV历史）"""
        if self._iv_history is None:
            self._iv_history = np.array([current_iv])
        else:
            self._iv_history = np.append(self._iv_history, current_iv)

        # 保持合理的历史长度
        max_history = self.lookback * 10
        if len(self._iv_history) > max_history:
            self._iv_history = self._iv_history[-max_history:]


# ============================================================================
# Options Statistical Arbitrage Framework
# ============================================================================

class OptionsStatArb:
    """
    期权统计套利框架 (Options Statistical Arbitrage Framework)

    整合多种期权统计套利策略的统一框架：
    1. IV均值回复（基于历史IV分布）
    2. 波动率曲面套利（不同行权价之间）
    3. 期限结构套利（不同到期日之间）

    框架特点：
    - 自动扫描套利机会
    - 统一的风险管理
    - 支持多种波动率策略

    Attributes:
        iv_strategy: IV均值回复策略实例
        surface_threshold: 波动率曲面套利阈值
        term_threshold: 期限结构套利阈值
    """

    def __init__(
        self,
        iv_strategy: Optional[IVMeanReversionStrategy] = None,
        surface_threshold: float = 0.05,
        term_threshold: float = 0.03,
        min_opportunity_score: float = 0.5,
    ):
        """
        初始化期权统计套利框架

        Args:
            iv_strategy: IV均值回复策略（可选，默认创建标准配置）
            surface_threshold: 波动率曲面套利阈值（不同行权价IV差异）
            term_threshold: 期限结构套利阈值（不同到期日IV差异）
            min_opportunity_score: 最小机会评分（低于此分数的机会被过滤）
        """
        self.iv_strategy = iv_strategy or IVMeanReversionStrategy()
        self.surface_threshold = surface_threshold
        self.term_threshold = term_threshold
        self.min_opportunity_score = min_opportunity_score

        # 内部状态
        self._iv_history: Dict[str, np.ndarray] = {}  # {strike: iv_history}
        self._opportunities: List[dict] = []

    def scan_opportunities(
        self,
        options_chain: Dict[str, any],
    ) -> List[Dict[str, any]]:
        """
        扫描套利机会

        检查期权链中的各种套利机会

        Args:
            options_chain: 期权链数据，格式：
                {
                    'strike': {          # 行权价作为key
                        'iv': float,    # 隐含波动率
                        'delta': float, # Delta
                        'gamma': float, # Gamma
                        'vega': float,  # Vega
                        'price': float, # 期权价格
                        'type': str,    # 'call' 或 'put'
                        'expiry': str,  # 到期日（如 '2024-03-15'）
                        'S': float,     # 标的价格
                        'K': float,     # 行权价
                        'T': float,     # 到期时间（年）
                        'r': float,     # 无风险利率
                    },
                    ...
                }

        Returns:
            套利机会列表，每个机会包含：
            {
                'type': 'iv_mean_reversion' | 'surface' | 'term_structure',
                'strike': float,
                'action': 'buy' | 'sell',
                'size': float,
                'hedge_ratio': float,
                'score': float,         # 机会评分（0-1）
                'expected_return': float,
                'iv': float,
                'reason': str,
            }
        """
        opportunities = []

        # 1. IV均值回复机会
        iv_opportunities = self._scan_iv_reversion(options_chain)
        opportunities.extend(iv_opportunities)

        # 2. 波动率曲面套利机会
        surface_opportunities = self._scan_surface(options_chain)
        opportunities.extend(surface_opportunities)

        # 3. 期限结构套利机会
        term_opportunities = self._scan_term_structure(options_chain)
        opportunities.extend(term_opportunities)

        # 按评分排序
        opportunities.sort(key=lambda x: x["score"], reverse=True)

        # 过滤低质量机会
        opportunities = [
            op for op in opportunities if op["score"] >= self.min_opportunity_score
        ]

        self._opportunities = opportunities
        return opportunities

    def _scan_iv_reversion(
        self,
        options_chain: Dict[str, any],
    ) -> List[Dict[str, any]]:
        """扫描IV均值回复机会"""
        opportunities = []

        for strike_str, data in options_chain.items():
            strike = float(strike_str)
            iv = data.get("iv", 0)
            expiry = data.get("expiry", "")

            # 获取或初始化IV历史
            hist_key = f"{strike}_{expiry}"
            if hist_key not in self._iv_history:
                self._iv_history[hist_key] = np.array([iv])
            else:
                self._iv_history[hist_key] = np.append(
                    self._iv_history[hist_key], iv
                )

            # 限制历史长度
            if len(self._iv_history[hist_key]) > 500:
                self._iv_history[hist_key] = self._iv_history[hist_key][-500:]

            # 生成信号
            signal = self.iv_strategy.generate_signal(
                current_iv=iv,
                iv_history=self._iv_history[hist_key],
            )

            if signal["action"] != "neutral":
                # 计算机会评分
                score = min(abs(signal["zscore"]) / (self.iv_strategy.iv_threshold * 2), 1.0)

                opportunity = {
                    "type": "iv_mean_reversion",
                    "strike": strike,
                    "expiry": expiry,
                    "action": "sell" if signal["action"] == "sell_iv" else "buy",
                    "size": signal["size"],
                    "hedge_ratio": signal["hedge_ratio"],
                    "score": score,
                    "expected_return": abs(signal["zscore"]) * 0.1,  # 简化估计
                    "iv": iv,
                    "zscore": signal["zscore"],
                    "reason": signal["reason"],
                }
                opportunities.append(opportunity)

        return opportunities

    def _scan_surface(
        self,
        options_chain: Dict[str, any],
    ) -> List[Dict[str, any]]:
        """扫描波动率曲面套利机会"""
        opportunities = []

        # 按到期日分组
        by_expiry: Dict[str, List[Tuple[float, float]]] = {}  # {expiry: [(strike, iv), ...]}

        for strike_str, data in options_chain.items():
            strike = float(strike_str)
            iv = data.get("iv", 0)
            expiry = data.get("expiry", "")

            if expiry not in by_expiry:
                by_expiry[expiry] = []
            by_expiry[expiry].append((strike, iv))

        # 检查每个到期日的曲面
        for expiry, strike_ivs in by_expiry.items():
            if len(strike_ivs) < 3:
                continue

            # 按行权价排序
            strike_ivs.sort(key=lambda x: x[0])

            # 计算ATM附近的IV斜率
            S = strike_ivs[len(strike_ivs) // 2][1]  # 使用中位数行权价对应的IV作为参考
            atm_iv = None
            for strike, iv in strike_ivs:
                if abs(strike - S) < S * 0.05:  # 5%范围内视为ATM
                    atm_iv = iv
                    break

            if atm_iv is None:
                atm_iv = strike_ivs[len(strike_ivs) // 2][1]

            # 检查OTM和ITM的IV偏离
            for strike, iv in strike_ivs:
                iv_diff = iv - atm_iv

                if abs(iv_diff) > self.surface_threshold:
                    # 发现曲面套利机会
                    action = "buy" if iv_diff < 0 else "sell"
                    score = min(abs(iv_diff) / (self.surface_threshold * 2), 1.0)

                    opportunity = {
                        "type": "surface",
                        "strike": strike,
                        "expiry": expiry,
                        "action": action,
                        "size": abs(iv_diff) / self.surface_threshold,
                        "hedge_ratio": 1.0,
                        "score": score,
                        "expected_return": abs(iv_diff) * 5,
                        "iv": iv,
                        "atm_iv": atm_iv,
                        "iv_diff": iv_diff,
                        "reason": f"Surface arbitrage: strike={strike}, IV={iv*100:.1f}% vs ATM={atm_iv*100:.1f}%, diff={iv_diff*100:.1f}%",
                    }
                    opportunities.append(opportunity)

        return opportunities

    def _scan_term_structure(
        self,
        options_chain: Dict[str, any],
    ) -> List[Dict[str, any]]:
        """扫描期限结构套利机会"""
        opportunities = []

        # 按行权价分组
        by_strike: Dict[float, List[Tuple[str, float]]] = {}  # {strike: [(expiry, iv), ...]}

        for strike_str, data in options_chain.items():
            strike = float(strike_str)
            iv = data.get("iv", 0)
            expiry = data.get("expiry", "")

            if strike not in by_strike:
                by_strike[strike] = []
            by_strike[strike].append((expiry, iv))

        # 检查每个行权价的期限结构
        for strike, expiry_ivs in by_strike.items():
            if len(expiry_ivs) < 2:
                continue

            # 按到期日排序
            expiry_ivs.sort(key=lambda x: x[0])

            # 检查相邻到期日的IV差异
            for i in range(len(expiry_ivs) - 1):
                expiry1, iv1 = expiry_ivs[i]
                expiry2, iv2 = expiry_ivs[i + 1]

                # 短期IV相对于长期IV的差异
                term_diff = iv1 - iv2

                if abs(term_diff) > self.term_threshold:
                    # 发现期限结构套利机会
                    # 如果短期IV > 长期IV，说明期限结构陡峭（可能做空短期IV）
                    action = "sell_short" if term_diff > 0 else "buy_short"
                    score = min(abs(term_diff) / (self.term_threshold * 2), 1.0)

                    opportunity = {
                        "type": "term_structure",
                        "strike": strike,
                        "expiry_near": expiry1,
                        "expiry_far": expiry2,
                        "action": action,
                        "size": abs(term_diff) / self.term_threshold,
                        "hedge_ratio": 1.0,
                        "score": score,
                        "expected_return": abs(term_diff) * 3,
                        "iv_near": iv1,
                        "iv_far": iv2,
                        "term_diff": term_diff,
                        "reason": f"Term structure: near={expiry1}({iv1*100:.1f}%) vs far={expiry2}({iv2*100:.1f}%), diff={term_diff*100:.1f}%",
                    }
                    opportunities.append(opportunity)

        return opportunities

    def execute(
        self,
        opportunity: Dict[str, any],
        current_price: Optional[float] = None,
        current_iv: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        执行套利交易

        根据机会信号生成执行指令

        Args:
            opportunity: 扫描返回的机会字典
            current_price: 当前期权价格（用于验证）
            current_iv: 当前IV（用于更新状态）

        Returns:
            执行结果：
            {
                'success': bool,
                'action': str,
                'size': float,
                'strike': float,
                'estimated_cost': float,
                'hedge_instruction': dict,
                'message': str,
            }
        """
        op_type = opportunity["type"]
        strike = opportunity["strike"]
        action = opportunity["action"]
        size = opportunity["size"]

        # 生成对冲指令
        if op_type == "iv_mean_reversion":
            hedge_instruction = {
                "delta_target": 0.0,
                "hedge_frequency": self.iv_strategy.hedge_freq,
            }
        elif op_type == "surface":
            hedge_instruction = {
                "delta_target": 0.0,
                "hedge_frequency": "discrete",
                "reference_strike": strike,
            }
        else:  # term_structure
            hedge_instruction = {
                "delta_target": 0.0,
                "hedge_frequency": "discrete",
                "pair_trade": True,
            }

        # 更新IV历史状态
        if current_iv is not None and op_type == "iv_mean_reversion":
            expiry = opportunity.get("expiry", "")
            hist_key = f"{strike}_{expiry}"
            if hist_key in self._iv_history:
                self._iv_history[hist_key] = np.append(
                    self._iv_history[hist_key], current_iv
                )

        return {
            "success": True,
            "action": action,
            "size": size,
            "strike": strike,
            "estimated_cost": 0.0,  # 需要市场数据
            "hedge_instruction": hedge_instruction,
            "message": f"Execute {op_type}: {action} IV at strike={strike}",
        }

    def get_best_opportunity(self) -> Optional[Dict[str, any]]:
        """获取当前评分最高的套利机会"""
        if not self._opportunities:
            return None
        return self._opportunities[0] if self._opportunities else None

    def clear_opportunities(self) -> None:
        """清除已过期/已执行的机会"""
        self._opportunities = []


# ============================================================================
# Convenience Functions
# ============================================================================

def calculate_iv_rank(current_iv: float, iv_history: np.ndarray, lookback: int = 252) -> float:
    """
    计算IV Rank（IV在历史分布中的百分位）

    IV Rank = (当前IV - 历史最低IV) / (历史最高IV - 历史最低IV)

    Args:
        current_iv: 当前IV
        iv_history: 历史IV数组
        lookback: 回顾窗口

    Returns:
        IV Rank (0.0 - 1.0)：
        - 1.0 表示IV处于历史最高
        - 0.0 表示IV处于历史最低
        - 0.5 表示IV处于历史中位数
    """
    if len(iv_history) < 2:
        return 0.5

    history = iv_history[-lookback:] if len(iv_history) >= lookback else iv_history
    iv_min = np.min(history)
    iv_max = np.max(history)

    if iv_max <= iv_min:
        return 0.5

    rank = (current_iv - iv_min) / (iv_max - iv_min)
    return float(max(0.0, min(1.0, rank)))


def calculate_iv_percentile(current_iv: float, iv_history: np.ndarray, lookback: int = 252) -> float:
    """
    计算IV Percentile（当前IV超过历史多少比例的观测值）

    Args:
        current_iv: 当前IV
        iv_history: 历史IV数组
        lookback: 回顾窗口

    Returns:
        IV Percentile (0.0 - 1.0)
    """
    if len(iv_history) < 2:
        return 0.5

    history = iv_history[-lookback:] if len(iv_history) >= lookback else iv_history
    percentile = np.sum(history < current_iv) / len(history)
    return float(percentile)


def detect_volatility_regime(
    iv_history: np.ndarray,
    returns: np.ndarray,
    lookback: int = 20,
) -> str:
    """
    检测当前波动率环境

    环境类型:
    - 'low_vol': 低波动环境（IV < 历史的25%分位数）
    - 'normal_vol': 正常波动环境
    - 'high_vol': 高波动环境（IV > 历史的75%分位数）
    - 'spike': 波动率急涨（近期IV急剧上升）
    - 'crash': 波动率急跌（近期IV急剧下降）

    Args:
        iv_history: 历史IV序列
        returns: 标的资产收益率序列
        lookback: 检测窗口

    Returns:
        波动率环境字符串
    """
    if len(iv_history) < lookback:
        return "unknown"

    recent_iv = iv_history[-1]
    hist_iv = iv_history[-lookback:]

    # 计算分位数
    q25 = np.percentile(hist_iv, 25)
    q75 = np.percentile(hist_iv, 75)
    median_iv = np.median(hist_iv)

    # 检测IV变化率
    if len(iv_history) >= lookback:
        iv_change = (recent_iv - np.mean(hist_iv[:-1])) / (np.std(hist_iv[:-1]) + 1e-10)
    else:
        iv_change = 0.0

    # 分类
    if iv_change > 2.0:
        return "spike"
    elif iv_change < -2.0:
        return "crash"
    elif recent_iv < q25:
        return "low_vol"
    elif recent_iv > q75:
        return "high_vol"
    else:
        return "normal_vol"


# ============================================================================
# __all__ Exports
# ============================================================================

__all__ = [
    # 核心策略
    "IVMeanReversionStrategy",
    "OptionsStatArb",
    # 辅助函数
    "calculate_iv_rank",
    "calculate_iv_percentile",
    "detect_volatility_regime",
]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=== Options Statistical Arbitrage Module Test ===")
    print()

    # 测试 IVMeanReversionStrategy
    print("--- IVMeanReversionStrategy Test ---")

    # 模拟历史IV数据
    np.random.seed(42)
    iv_history = np.concatenate([
        np.random.normal(0.30, 0.05, 30),  # 正常波动率环境
        np.array([0.30, 0.32, 0.35, 0.38, 0.42, 0.45, 0.48, 0.50]),  # IV上升
    ])

    strategy = IVMeanReversionStrategy(
        lookback=20,
        iv_threshold=0.30,
        hedge_freq="discrete",
    )

    current_iv = 0.48
    signal = strategy.generate_signal(current_iv, iv_history)
    print(f"Current IV: {current_iv*100:.1f}%")
    print(f"Signal: {signal}")
    print()

    # 测试compute_hedge
    hedge = strategy.compute_hedge(option_delta=0.5, position_size=10.0)
    print(f"Required hedge (delta=0.5, position=10): {hedge}")
    print()

    # 测试 OptionsStatArb
    print("--- OptionsStatArb Test ---")

    # 构建模拟期权链
    options_chain = {}
    for strike in [95, 97, 99, 100, 101, 103, 105]:
        options_chain[str(strike)] = {
            "iv": 0.30 + abs(strike - 100) * 0.01 + np.random.normal(0, 0.02),
            "delta": bs_delta(100, strike, 30/365, 0.05, 0.30),
            "gamma": bs_gamma(100, strike, 30/365, 0.05, 0.30),
            "vega": bs_vega(100, strike, 30/365, 0.05, 0.30),
            "price": bs_price(100, strike, 30/365, 0.05, 0.30),
            "type": "call",
            "expiry": "2024-03-15",
            "S": 100,
            "K": strike,
            "T": 30/365,
            "r": 0.05,
        }

    arb = OptionsStatArb(
        iv_strategy=strategy,
        surface_threshold=0.05,
        term_threshold=0.03,
    )

    opportunities = arb.scan_opportunities(options_chain)
    print(f"Found {len(opportunities)} opportunities:")
    for op in opportunities[:3]:
        print(f"  - {op['type']}: {op['action']} at strike={op['strike']}, score={op['score']:.2f}")
        print(f"    Reason: {op['reason']}")
    print()

    # 测试辅助函数
    print("--- Helper Functions Test ---")
    print(f"IV Rank: {calculate_iv_rank(0.48, iv_history):.2f}")
    print(f"IV Percentile: {calculate_iv_percentile(0.48, iv_history):.2f}")
    print(f"Vol Regime: {detect_volatility_regime(iv_history, np.random.randn(30))}")
    print()
    print("OK")
