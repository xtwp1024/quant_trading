"""
Market Making RL Module — Stanford CS234 Inspired
基于 D:/Hive/Data/trading_repos/MARKET-MAKING-RL/ 重构

核心特性:
- LOB (Limit Order Book) 物理模拟
- Avellaneda-Stoikov 模型实现
- Gymnasium 兼容的 RL 环境
- 库存风险管理
- 最优价差计算

Market Making RL Module — Stanford CS234 Inspired
Adapted from D:/Hive/Data/trading_repos/MARKET-MAKING-RL/

Key features:
- Complete LOB (Limit Order Book) physics simulation
- Avellaneda-Stoikov model implementation
- Gymnasium-compatible RL environment
- Inventory risk management
- Optimal spread calculation
"""

import numpy as np
import heapq
from typing import Tuple, Optional, Dict, Any, NamedTuple
from dataclasses import dataclass
from gymnasium import spaces


# =============================================================================
# Data Structures / 数据结构
# =============================================================================

class LOBQuote(NamedTuple):
    """限价报价 / Limit order quote"""
    price: float
    volume: int


class LOBTrade(NamedTuple):
    """成交记录 / Trade record"""
    price: float
    volume: int
    side: str  # 'buy' or 'sell'
    timestamp: int


@dataclass
class MarketState:
    """
    市场状态 / Market state snapshot
    包含当前中价、买卖价差、库存、财富等信息
    """
    midprice: float
    best_bid: float
    best_ask: float
    inventory: int
    wealth: float
    time: float
    time_to_expiry: float


# =============================================================================
# LOBSimulator — Limit Order Book Physics / 限价订单簿物理模拟
# =============================================================================

class LOBSimulator:
    """
    限价订单簿 (LOB) 物理模拟器
    基于 Stanford CS234 CS234 MARKET-MAKING-RL 项目

    使用布朗运动模拟中价演化，Poisson 过程模拟市价订单流，
    使用堆 (heap) 实现高效订单管理。

    Limit Order Book (LOB) physics simulator.
    Simulates midprice evolution via Brownian motion and market order flow
    via Poisson processes. Uses heap-based efficient order management.

    Features:
    - Brownian motion midprice dynamics
    - Poisson market order arrival (Toke-Yoshida model)
    - Heap-based bid/ask priority queues
    - Bell-curve initial LOB population
    """

    def __init__(
        self,
        midprice: float = 533.0,
        spread: float = 10.0,
        nstocks: int = 10000,
        drift: float = 3.59e-6,
        scale: float = 2.4e-3,
        betas: Tuple[float, ...] = (7.2, -2.13, -0.8, -2.3, 0.167, -0.1),
        seed: Optional[int] = None,
    ):
        """
        初始化 LOB 模拟器 / Initialize LOB simulator

        Args:
            midprice: 初始中价 / Initial midprice
            spread: 初始买卖价差 / Initial bid-ask spread
            nstocks: 每边初始库存量 / Initial number of stocks per side
            drift: 布朗运动漂移项 / Brownian motion drift
            scale: 布朗运动尺度参数 / Brownian motion scale parameter
            betas: Toke-Yoshida 模型参数 (6个) / Toke-Yoshida model parameters (6)
            seed: 随机种子 / Random seed
        """
        self.initial_midprice = midprice
        self.initial_spread = spread
        self.initial_nstocks = nstocks
        self.drift = drift
        self.scale = scale
        self.betas = betas

        if seed is not None:
            np.random.seed(seed)

        # Order book state / 订单簿状态
        self.bids = []  # max-heap: (-price, volume)
        self.asks = []  # min-heap: (price, volume)

        # Book depth tracking / 订单簿深度追踪
        self.high_bid = 0.0
        self.nhigh_bid = 0
        self.low_ask = 0.0
        self.nlow_ask = 0

        # Midprice dynamics / 中价动力学
        self.midprice = midprice
        self.spread = spread
        self.delta_b = 0.0  # bid distance from mid
        self.delta_a = 0.0  # ask distance from mid

        # Time tracking / 时间追踪
        self.t = 0.0
        self.max_t = 1.0

        # Brownian motion model / 布朗运动模型
        self._brownian_state = 0.0

    def copy(self) -> 'LOBSimulator':
        """创建订单簿副本 / Create a copy of the order book"""
        new_lob = LOBSimulator(
            midprice=self.midprice,
            spread=self.spread,
            drift=self.drift,
            scale=self.scale,
            betas=self.betas,
        )
        new_lob.bids = self.bids.copy()
        new_lob.asks = self.asks.copy()
        new_lob.high_bid = self.high_bid
        new_lob.nhigh_bid = self.nhigh_bid
        new_lob.low_ask = self.low_ask
        new_lob.nlow_ask = self.nlow_ask
        new_lob.delta_b = self.delta_b
        new_lob.delta_a = self.delta_a
        new_lob.t = self.t
        new_lob.midprice = self.midprice
        return new_lob

    def is_empty(self) -> bool:
        """检查订单簿是否为空 (任一方无订单) / Check if book is empty (either side)"""
        return not (len(self.bids) and len(self.asks))

    def _sample_brownian_step(self) -> float:
        """采样布朗运动一步 / Sample one step of Brownian motion"""
        # Simple Brownian motion increment: N(0, scale)
        return np.random.normal(0, self.scale)

    def update_midprice(self) -> None:
        """
        更新中价 (布朗运动一步)
        Update midprice by one Brownian motion step
        """
        self.midprice += self.drift + self._sample_brownian_step()
        self._recalculate()

    def _recalculate(self) -> None:
        """
        重新计算订单簿关键指标
        Recalculate key book metrics (best bid/ask, spreads)
        """
        self.spread = 0.0
        self.delta_b = self.delta_a = 0.0
        self.low_ask = self.high_bid = 0.0
        self.nlow_ask = self.nhigh_bid = 0

        if len(self.asks):
            self.low_ask, self.nlow_ask = self.asks[0]
            self.delta_a = self.low_ask - self.midprice

            if len(self.bids):
                self.high_bid, self.nhigh_bid = -self.bids[0][0], self.bids[0][1]
                self.spread = self.low_ask - self.high_bid
                self.delta_b = self.midprice - self.high_bid

                # Clean up crossed markets / 清理交叉市场
                while self.high_bid > self.midprice and len(self.bids):
                    heapq.heappop(self.bids)
                    if not len(self.bids):
                        self.bid(self.initial_nstocks // 100, round(self.midprice, 2) - 0.01)
                        break
                    self.high_bid, self.nhigh_bid = -self.bids[0][0], self.bids[0][1]

                while self.low_ask < self.midprice and len(self.asks):
                    heapq.heappop(self.asks)
                    if not len(self.asks):
                        self.ask(self.initial_nstocks // 100, round(self.midprice, 2) + 0.01)
                        break
                    self.low_ask, self.nlow_ask = self.asks[0]

                self.spread = self.low_ask - self.high_bid
                self.delta_b = self.midprice - self.high_bid
                self.delta_a = self.low_ask - self.midprice

    def lambda_buy(self, delta: float, q: int) -> float:
        """
        计算买入强度 (Toke-Yoshida 模型)
        Compute buy intensity (market order arrival rate)

        λ(δ,q) = exp(β·[ln(1+δ), ln²(1+δ), ln(1+q), ln²(1+q), ln(1+δ+q)])

        Args:
            delta: 卖单价差 / ask distance from mid
            q: 卖单数量 / quantity at best ask

        Returns:
            买入强度 / buy intensity (Poisson rate)
        """
        if q <= 0:
            return 0.0
        beta0, beta1, beta2, beta3, beta4, beta5 = self.betas
        try:
            lam = np.exp(
                beta0
                + beta1 * np.log(1 + delta)
                + beta2 * np.log(1 + delta) ** 2
                + beta3 * np.log(1 + q)
                + beta4 * np.log(1 + q) ** 2
                + beta5 * np.log(1 + delta + q)
            )
            return max(0.0, lam)
        except (ValueError, OverflowError):
            return 0.0

    def lambda_sell(self, delta: float, q: int) -> float:
        """计算卖出强度 (对称于买入) / Compute sell intensity (symmetric to buy)"""
        return self.lambda_buy(delta, q)

    def buy(self, volume: int, maxprice: float = 0.0) -> Tuple[int, float]:
        """
        市价买入 (扫卖单堆栈)
        Market buy: consume ask orders

        Args:
            volume: 买入数量 / number of shares to buy
            maxprice: 最高买入价 (可选) / maximum acceptable price

        Returns:
            (实际买入数量, 实际支付金额) / (shares_bought, total_cost)
        """
        n_bought = 0
        total_cost = 0.0

        while volume > 0 and len(self.asks):
            if maxprice > 0 and self.asks[0][0] > maxprice:
                break

            price, n = heapq.heappop(self.asks)
            n_this = min(n, volume)
            n_bought += n_this
            total_cost += price * n_this

            if volume < n:
                heapq.heappush(self.asks, (price, n - volume))
                break
            volume -= n

        self._recalculate()
        return n_bought, total_cost

    def sell(self, volume: int, minprice: float = 0.0) -> Tuple[int, float]:
        """
        市价卖出 (扫买单堆栈)
        Market sell: consume bid orders

        Args:
            volume: 卖出数量 / number of shares to sell
            minprice: 最低卖出价 (可选) / minimum acceptable price

        Returns:
            (实际卖出数量, 实际收入金额) / (shares_sold, total_revenue)
        """
        n_sold = 0
        total_revenue = 0.0

        while volume > 0 and len(self.bids):
            if minprice > 0 and -self.bids[0][0] < minprice:
                break

            neg_price, n = heapq.heappop(self.bids)
            price = -neg_price
            n_this = min(n, volume)
            n_sold += n_this
            total_revenue += price * n_this

            if volume < n:
                heapq.heappush(self.bids, (-price, n - volume))
                break
            volume -= n

        self._recalculate()
        return n_sold, total_revenue

    def bid(self, volume: int, price: float) -> None:
        """
        添加限价买单 / Add limit buy order

        Args:
            volume: 买入数量 / number of shares
            price: 买入价格 / price per share
        """
        price = round(price, 2)
        if volume <= 0:
            return

        # If bid crosses spread, execute as market buy
        if len(self.asks) and price >= self.asks[0][0]:
            nbought, _ = self.buy(volume, maxprice=price)
            volume -= nbought
            if volume <= 0:
                return

        heapq.heappush(self.bids, (-price, volume))
        self._recalculate()

    def ask(self, volume: int, price: float) -> None:
        """
        添加限价卖单 / Add limit sell order

        Args:
            volume: 卖出数量 / number of shares
            price: 卖出价格 / price per share
        """
        price = round(price, 2)
        if volume <= 0:
            return

        # If ask crosses spread, execute as market sell
        if len(self.bids) and price <= -self.bids[0][0]:
            nsold, _ = self.sell(volume, minprice=price)
            volume -= nsold
            if volume <= 0:
                return

        heapq.heappush(self.asks, (price, volume))
        self._recalculate()

    def get_state(self) -> Tuple[int, float, int, float]:
        """
        获取当前状态元组
        Get current state tuple

        Returns:
            (n_high_bid, delta_b, n_low_ask, delta_a)
        """
        return (self.nhigh_bid, self.delta_b, self.nlow_ask, self.delta_a)

    def populate_bell_curve(self, nsteps: int = 1000) -> None:
        """
        用钟形曲线填充订单簿 (模拟初始状态)
        Populate order book with bell curve distribution

        每边按正态分布生成订单，价格围绕中价分布

        Args:
            nsteps: 生成步数 / number of generation steps
        """
        self.bids = []
        self.asks = []

        for _ in range(nsteps):
            delta_b = np.random.normal(self.spread / 2, self.spread / 4)
            delta_a = np.random.normal(self.spread / 2, self.spread / 4)
            nbid = np.random.poisson(2 * self.initial_nstocks / nsteps)
            nask = np.random.poisson(2 * self.initial_nstocks / nsteps)

            if nbid > 0:
                self.bid(nbid, self.midprice - delta_b)
            if nask > 0:
                self.ask(nask, self.midprice + delta_a)

        self._recalculate()

    def reset(
        self,
        midprice: Optional[float] = None,
        spread: Optional[float] = None,
        make_bell: bool = True,
        nsteps: int = 1000,
    ) -> None:
        """
        重置/初始化订单簿 / Reset/initialize the order book

        Args:
            midprice: 中价 (None 则用初始值) / midprice (None = use initial)
            spread: 价差 (None 则用初始值) / spread (None = use initial)
            make_bell: 是否用钟形曲线填充 / whether to populate with bell curve
            nsteps: 填充步数 / population steps
        """
        self.bids = []
        self.asks = []

        if midprice is not None:
            self.midprice = midprice
        else:
            self.midprice = self.initial_midprice

        if spread is not None:
            self.spread = spread
        else:
            self.spread = self.initial_spread

        self.t = 0.0

        if make_bell:
            self.populate_bell_curve(nsteps)
        else:
            # Simple symmetric initialization / 简单对称初始化
            n = self.initial_nstocks // 2
            self.bid(n, self.midprice - self.spread / 2)
            self.ask(n, self.midprice + self.spread / 2)

        self._recalculate()


# =============================================================================
# Avellaneda-Stoikov Model / Avellaneda-Stoikov 模型
# =============================================================================

class AvellanedaStoikovModel:
    """
    Avellaneda-Stoikov 做市商模型

    基于 2006 年 Avellaneda & Stoikov 的经典论文:
    "High-Frequency Trading in a Limit Order Book"

    核心公式:
    - 保留价格 (Reservation Price): S^r = S - q · γ · σ² · T
    - 最优价差 (Optimal Spread): δ* = γ · σ² · T + 2/γ · log(1 + γ/κ)

    Avellaneda-Stoikov Market Making Model

    Based on the 2006 Avellaneda & Stoikov paper:
    "High-Frequency Trading in a Limit Order Book"

    Core Formulas:
    - Reservation Price: S^r = S - q · γ · σ² · T
    - Optimal Spread: δ* = γ · σ² · T + 2/γ · log(1 + γ/κ)
    """

    def __init__(
        self,
        sigma: float = 1e-2,
        gamma: float = 1e-4,
        kappa: float = 1.0,
        risk_aversion: float = 1e-4,
    ):
        """
        初始化 Avellaneda-Stoikov 模型

        Args:
            sigma: 波动率 (年化) / volatility (annualized)
            gamma: 风险厌恶参数 / risk aversion parameter
            kappa: 订单流强度参数 / order flow intensity parameter
            risk_aversion: 风险厌恶系数 (同 gamma) / risk aversion (same as gamma)
        """
        self.sigma = sigma
        self.gamma = gamma
        self.kappa = kappa
        self.risk_aversion = risk_aversion

    def reservation_price(
        self,
        midprice: float,
        inventory: int,
        time_to_expiry: float,
        sigma: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> float:
        """
        计算保留价格 (无差异价格)

        保留价格是使得做市商对当前库存无差异的中价。
        当库存为正时，保留价格低于中价（卖出倾向）；
        当库存为负时，保留价格高于中价（买入倾向）。

        Compute reservation (indifference) price

        The reservation price makes the market maker indifferent to current
        inventory. Positive inventory → reservation below mid (sell bias);
        Negative inventory → reservation above mid (buy bias).

        Formula:
        S^r = S · exp((σ²/2)·T - γ·σ²·q·T/(2·κ))
        Simplified: S^r ≈ S - q · γ · σ² · T

        Args:
            midprice: 当前中价 / current midprice
            inventory: 当前库存 (q) / current inventory (q)
            time_to_expiry: 距离到期时间 (T) / time to expiry (T)
            sigma: 波动率 (覆盖默认值) / volatility (override default)
            gamma: 风险厌恶 (覆盖默认值) / risk aversion (override default)

        Returns:
            保留价格 / reservation price
        """
        sig = sigma if sigma is not None else self.sigma
        gam = gamma if gamma is not None else self.gamma

        # Full formula from task specification:
        # S^r = S * exp((σ²/2)*T - γ*σ²*q*T/(2*κ))
        exponent = (sig ** 2 / 2) * time_to_expiry - gam * (sig ** 2) * inventory * time_to_expiry / (2 * self.kappa)
        return midprice * np.exp(exponent)

    def optimal_spread(
        self,
        time_to_expiry: float,
        sigma: Optional[float] = None,
        gamma: Optional[float] = None,
        kappa: Optional[float] = None,
    ) -> float:
        """
        计算最优价差

        最优价差使得做市商期望效用最大化。
        价差随时间、波动率增加而增大。

        Compute optimal bid-ask spread

        The optimal spread maximizes expected utility.
        Spread increases with time and volatility.

        Formula:
        δ* = γ · σ² · T + 2·log(1 + γ/κ) / γ
        Simplified: δ* = γ · σ² · T + 2/γ · log(1 + γ/κ)

        Args:
            time_to_expiry: 距离到期时间 (T) / time to expiry (T)
            sigma: 波动率 (覆盖默认值) / volatility (override default)
            gamma: 风险厌恶 (覆盖默认值) / risk aversion (override default)
            kappa: 订单流强度 (覆盖默认值) / order flow intensity (override default)

        Returns:
            最优价差 / optimal spread
        """
        sig = sigma if sigma is not None else self.sigma
        gam = gamma if gamma is not None else self.gamma
        kap = kappa if kappa is not None else self.kappa

        # Formula from task specification:
        # optimal spread = γ * σ² * T + 2 * log(1 + γ/κ)
        term1 = gam * (sig ** 2) * time_to_expiry
        term2 = 2 * np.log(1 + gam / kap)

        return term1 + term2 / gam  # Normalize by gamma as in simplified formula

    def compute_quotes(
        self,
        midprice: float,
        inventory: int,
        time_to_expiry: float,
    ) -> Tuple[float, float]:
        """
        计算最优买卖报价

        基于 Avellaneda-Stoikov 模型计算最佳 bid 和 ask 价格。

        Compute optimal bid-ask quotes

        Based on Avellaneda-Stoikov model, compute best bid and ask prices.

        Args:
            midprice: 当前中价 / current midprice
            inventory: 当前库存 / current inventory
            time_to_expiry: 剩余时间 / remaining time

        Returns:
            (bid_price, ask_price) / (最优买入价, 最优卖出价)
        """
        res_price = self.reservation_price(midprice, inventory, time_to_expiry)
        spread = self.optimal_spread(time_to_expiry)

        bid_price = res_price - spread / 2
        ask_price = res_price + spread / 2

        return bid_price, ask_price


# =============================================================================
# OptimalSpreadCalculator — Spread Calculation Utility / 最优价差计算工具
# =============================================================================

class OptimalSpreadCalculator:
    """
    最优价差计算器

    基于波动率、库存、时间到期的综合价差计算。
    支持多种计算模式（简化版、完整版）。

    Optimal Spread Calculator

    Computes optimal spread based on volatility, inventory, and time to expiry.
    Supports multiple calculation modes (simplified, full).

    Usage:
        calc = OptimalSpreadCalculator(sigma=0.01, gamma=1e-4)
        spread = calc.calculate(time_to_expiry=1.0, inventory=100)
        bid, ask = calc.compute_quotes(midprice=100, inventory=100, time_to_expiry=1.0)
    """

    def __init__(
        self,
        sigma: float = 1e-2,
        gamma: float = 1e-4,
        kappa: float = 1.0,
        base_spread: float = 0.0,
    ):
        """
        初始化价差计算器 / Initialize spread calculator

        Args:
            sigma: 波动率 (年化) / volatility (annualized)
            gamma: 风险厌恶参数 / risk aversion parameter
            kappa: 订单流强度 / order flow intensity
            base_spread: 基础价差加成 / base spread addition
        """
        self.sigma = sigma
        self.gamma = gamma
        self.kappa = kappa
        self.base_spread = base_spread
        self.as_model = AvellanedaStoikovModel(sigma=sigma, gamma=gamma, kappa=kappa)

    def calculate(
        self,
        time_to_expiry: float,
        inventory: int = 0,
        volatility: Optional[float] = None,
    ) -> float:
        """
        计算最优价差

        Compute optimal spread

        Args:
            time_to_expiry: 剩余时间 (T) / time to expiry (T)
            inventory: 当前库存 / current inventory
            volatility: 实时波动率 (覆盖默认值) / real-time volatility

        Returns:
            最优价差 / optimal spread
        """
        sig = volatility if volatility is not None else self.sigma

        # Inventory-adjusted spread
        inv_factor = 1.0 + 0.1 * abs(inventory) / 1000

        spread = self.as_model.optimal_spread(
            time_to_expiry=time_to_expiry,
            sigma=sig,
            gamma=self.gamma,
            kappa=self.kappa,
        )

        return spread * inv_factor + self.base_spread

    def compute_quotes(
        self,
        midprice: float,
        inventory: int,
        time_to_expiry: float,
        volatility: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        计算最优报价对

        Compute optimal quote pair

        Args:
            midprice: 当前中价 / current midprice
            inventory: 当前库存 / current inventory
            time_to_expiry: 剩余时间 / remaining time
            volatility: 实时波动率 / real-time volatility

        Returns:
            (bid_price, ask_price, spread)
        """
        sig = volatility if volatility is not None else self.sigma

        res_price = self.as_model.reservation_price(
            midprice=midprice,
            inventory=inventory,
            time_to_expiry=time_to_expiry,
            sigma=sig,
        )

        spread = self.calculate(
            time_to_expiry=time_to_expiry,
            inventory=inventory,
            volatility=sig,
        )

        bid = res_price - spread / 2
        ask = res_price + spread / 2

        return bid, ask, spread

    def compute_inventory_penalty(self, inventory: int, max_inventory: int = 1000) -> float:
        """
        计算库存惩罚项

        库存偏离零时添加惩罚，防止过度单向持仓。

        Compute inventory penalty term

        Penalizes偏离零 inventory to prevent excessive one-sided positioning.

        Args:
            inventory: 当前库存 / current inventory
            max_inventory: 最大允许库存 / max allowed inventory

        Returns:
            惩罚项金额 / penalty amount
        """
        normalized_inv = inventory / max_inventory if max_inventory > 0 else 0
        return -self.gamma * (normalized_inv ** 2)


# =============================================================================
# MarketMakingEnv — Gymnasium-compatible RL Environment / Gymnasium兼容RL环境
# =============================================================================

class MarketMakingEnv:
    """
    Gymnasium 兼容的做市商强化学习环境

    基于 Stanford CS234 MARKET-MAKING-RL 项目设计。
    观测空间: (midprice, wealth, inventory, n_bid, delta_b, n_ask, delta_a, time_left)
    动作空间: (n_bid, delta_b, n_ask, delta_a) — 连续/离散混合

    Gymnasium-compatible Market Making RL Environment

    Observation space: (midprice, wealth, inventory, n_bid, delta_b, n_ask, delta_a, time_left)
    Action space: (n_bid, delta_b, n_ask, delta_a) — continuous/discrete hybrid

    Key features:
    - Full LOB physics simulation
    - Inventory risk management
    - Time-varying market conditions
    - Configurable reward function
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        # LOB parameters
        midprice: float = 533.0,
        spread: float = 10.0,
        nstocks: int = 10000,
        drift: float = 3.59e-6,
        scale: float = 2.4e-3,
        betas: Tuple[float, ...] = (7.2, -2.13, -0.8, -2.3, 0.167, -0.1),
        # Avellaneda-Stoikov parameters
        sigma: float = 1e-2,
        gamma: float = 1e-4,
        kappa: float = 1.0,
        # Environment parameters
        max_t: float = 1.0,
        dt: float = 1e-3,
        initial_wealth: float = 0.0,
        initial_inventory: int = 0,
        # Reward parameters
        reward_weight_wealth: float = 1.0,
        reward_weight_inventory: float = 0.1,
        # Misc
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        初始化做市商环境 / Initialize market making environment

        Args:
            midprice: 初始中价 / initial midprice
            spread: 初始价差 / initial spread
            nstocks: 每边初始订单数 / initial orders per side
            drift: 布朗运动漂移 / Brownian motion drift
            scale: 布朗运动尺度 / Brownian motion scale
            betas: Toke-Yoshida 模型参数 / Toke-Yoshida model parameters
            sigma: 波动率 / volatility
            gamma: 风险厌恶参数 / risk aversion parameter
            kappa: 订单流强度 / order flow intensity
            max_t: 最大模拟时间 / max simulation time
            dt: 时间步长 / time step
            initial_wealth: 初始财富 / initial wealth
            initial_inventory: 初始库存 / initial inventory
            reward_weight_wealth: 财富奖励权重 / wealth reward weight
            reward_weight_inventory: 库存奖励权重 / inventory reward weight
            seed: 随机种子 / random seed
            render_mode: 渲染模式 / render mode
        """
        # LOB
        self.lob = LOBSimulator(
            midprice=midprice,
            spread=spread,
            nstocks=nstocks,
            drift=drift,
            scale=scale,
            betas=betas,
            seed=seed,
        )

        # Avellaneda-Stoikov
        self.as_model = AvellanedaStoikovModel(sigma=sigma, gamma=gamma, kappa=kappa)
        self.spread_calculator = OptimalSpreadCalculator(
            sigma=sigma, gamma=gamma, kappa=kappa
        )

        # Time
        self.max_t = max_t
        self.dt = dt
        self.nt = int(max_t / dt)

        # State
        self.wealth = initial_wealth
        self.inventory = initial_inventory
        self.t = 0

        # Reward weights
        self.reward_weight_wealth = reward_weight_wealth
        self.reward_weight_inventory = reward_weight_inventory

        # Rendering
        self.render_mode = render_mode
        self._seed = seed

        # Define spaces / 定义空间
        # Observation: (midprice, wealth, inventory, n_bid, delta_b, n_ask, delta_a, time_left)
        obs_low = np.array([0, -np.inf, -1000, 0, 0, 0, 0, 0], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, 1000, 10000, 1000, 10000, 1000, max_t], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: (n_bid, delta_b, n_ask, delta_a) — all continuous approximations
        # n_bid, n_ask are rounded to int
        act_low = np.array([0, 0, 0, 0], dtype=np.float32)
        act_high = np.array([100, 100, 100, 100], dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        self._max_episode_steps = self.nt

    def seed(self, seed: int) -> None:
        """设置随机种子 / Set random seed"""
        self._seed = seed
        np.random.seed(seed)
        self.lob = LOBSimulator(
            midprice=self.lob.initial_midprice,
            spread=self.lob.initial_spread,
            nstocks=self.lob.initial_nstocks,
            drift=self.lob.drift,
            scale=self.lob.scale,
            betas=self.lob.betas,
            seed=seed,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境到初始状态 / Reset environment to initial state

        Args:
            seed: 随机种子 / random seed
            options: 重置选项 / reset options

        Returns:
            (observation, info) / (观测, 信息字典)
        """
        if seed is not None:
            self.seed(seed)

        # Reset LOB
        self.lob.reset(
            midprice=self.lob.initial_midprice,
            spread=self.lob.initial_spread,
            make_bell=True,
            nsteps=1000,
        )

        # Reset state
        self.wealth = options.get("initial_wealth", 0.0) if options else 0.0
        self.inventory = options.get("initial_inventory", 0) if options else 0
        self.t = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步环境交互 / Execute one environment step

        Args:
            action: (n_bid, delta_b, n_ask, delta_a)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Parse action / 解析动作
        n_bid = int(max(0, action[0]))
        delta_b = max(0, float(action[1]))
        n_ask = int(max(0, action[2]))
        delta_a = max(0, float(action[3]))

        # Submit limit orders / 提交限价单
        self.lob.bid(n_bid, self.lob.midprice - delta_b)
        self.lob.ask(n_ask, self.lob.midprice + delta_a)

        # Record pre-step wealth and inventory / 记录步前状态
        old_wealth = self.wealth
        old_inventory = self.inventory

        # Simulate market orders / 模拟市价单
        self._simulate_market_orders()

        # Update time / 更新时间
        self.t += 1

        # Calculate reward / 计算奖励
        reward = self._calculate_reward()

        # Check termination / 检查终止
        time_left = self.max_t - self.t * self.dt
        terminated = time_left <= 0 or self.lob.is_empty()
        truncated = False

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _simulate_market_orders(self) -> None:
        """
        模拟市价订单执行

        基于 Poisson 过程和 Toke-Yoshida 模型模拟市价买卖单。

        Simulate market order execution

        Market buy/sell based on Poisson process and Toke-Yoshida model.
        """
        self.lob.update_midprice()

        # Get current book state / 获取当前订单簿状态
        delta_a = self.lob.delta_a
        delta_b = self.lob.delta_b
        n_ask = self.lob.nlow_ask
        n_bid = self.lob.nhigh_bid

        # Compute arrival rates / 计算到达率
        lambda_buy = self.lob.lambda_buy(delta_a, n_ask)
        lambda_sell = self.lob.lambda_sell(delta_b, n_bid)

        # Sample Poisson arrivals / Poisson 采样
        n_buy = np.random.poisson(max(0, lambda_buy))
        n_sell = np.random.poisson(max(0, lambda_sell))

        # Execute market orders / 执行市价单
        n_ask_lift, cost = self.lob.buy(n_buy)
        n_bid_hit, revenue = self.lob.sell(n_sell)

        # Update wealth and inventory / 更新财富和库存
        self.wealth += revenue - cost
        self.inventory += n_bid_hit - n_ask_lift

    def _calculate_reward(self) -> float:
        """
        计算奖励

        奖励 = 财富变化 - 库存风险惩罚

        Compute reward

        Reward = wealth_change - inventory_risk_penalty
        """
        # Wealth component / 财富部分
        reward = self.reward_weight_wealth * self.wealth

        # Inventory penalty (exponential decay with time) / 库存惩罚 (指数衰减)
        time_left = self.max_t - self.t * self.dt
        inv_penalty = self.reward_weight_inventory * np.exp(-self.as_model.gamma * time_left) * abs(self.inventory)
        reward -= inv_penalty

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        获取当前观测

        观测 = (midprice, wealth, inventory, n_bid, delta_b, n_ask, delta_a, time_left)

        Get current observation

        Observation = (midprice, wealth, inventory, n_bid, delta_b, n_ask, delta_a, time_left)
        """
        time_left = max(0, self.max_t - self.t * self.dt)
        state = self.lob.get_state()
        n_bid, delta_b, n_ask, delta_a = state

        obs = np.array(
            [
                self.lob.midprice,
                self.wealth,
                self.inventory,
                n_bid,
                delta_b,
                n_ask,
                delta_a,
                time_left,
            ],
            dtype=np.float32,
        )

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """获取附加信息 / Get additional info"""
        return {
            "midprice": self.lob.midprice,
            "spread": self.lob.spread,
            "inventory": self.inventory,
            "wealth": self.wealth,
            "time": self.t * self.dt,
            "time_left": max(0, self.max_t - self.t * self.dt),
            "book_empty": self.lob.is_empty(),
        }

    def render(self) -> Optional[np.ndarray]:
        """渲染环境状态 / Render environment state"""
        if self.render_mode == "human":
            state = self.lob.get_state()
            print(
                f"t={self.t * self.dt:.4f} | "
                f"Mid={self.lob.midprice:.2f} | "
                f"Bid({state[0]})@{state[1]:.2f} | "
                f"Ask({state[2]})@{state[3]:.2f} | "
                f"W={self.wealth:.2f} I={self.inventory}"
            )
        return None

    def close(self) -> None:
        """清理环境 / Clean up environment"""
        pass


# =============================================================================
# Module Exports / 模块导出
# =============================================================================

__all__ = [
    "LOBSimulator",
    "MarketMakingEnv",
    "AvellanedaStoikovModel",
    "OptimalSpreadCalculator",
    "LOBQuote",
    "LOBTrade",
    "MarketState",
]
