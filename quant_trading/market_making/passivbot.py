"""
PassivBot加密货币做市策略 / PassivBot Cryptocurrency Market-Making Strategy

从 passivbot (https://github.com/enarjord/passivbot) 吸收并适配

核心概念 / Core Concepts:
1. Forager: 追踪价格下限, 动态调整 / Trailing stop that follows price lows
2. Grid-like spread: 等距买卖单 / Equally-spaced buy/sell orders
3. 进化算法优化参数 / Evolutionary algorithm for parameter optimization
4. 仓位管理 / Position management
5. 多交易所支持 / Multi-exchange support (REST-only)

本模块为纯Python实现, 无Rust依赖 / Pure Python, no Rust dependency.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

__all__ = [
    "PassivBotStrategy",
    "PassivBotOptimizer",
    "GridOrder",
    "ForagerState",
    "ProfitReport",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构 / Data Structures
# ---------------------------------------------------------------------------


@dataclass
class GridOrder:
    """单个网格订单 / Single grid order."""
    side: str       # 'long' or 'short'
    price: float
    qty: float


@dataclass
class ForagerState:
    """Forager追踪状态 / Forager trailing state.

    Forager是PassivBot的止损追踪机制 / Forager is PassivBot's trailing stop mechanism.
    当价格向有利方向移动时, 它会更新追踪价格下限.
    When price moves favorably, it updates the trailing price floor.
    """
    enabled: bool = False
    price_floor: float = 0.0        # 追踪价格下限 / Trailing price floor
    highest_price: float = 0.0      # 期间最高价 / Highest price since activation
    activation_price: float = 0.0   # 激活价格 / Activation price
    offset_pct: float = 0.0         # 偏移百分比 / Offset percentage


@dataclass
class ProfitReport:
    """利润报告 / Profit report."""
    total_pnl: float = 0.0           # 总盈亏 / Total PnL
    long_pnl: float = 0.0           # 多头盈亏 / Long PnL
    short_pnl: float = 0.0          # 空头盈亏 / Short PnL
    n_long_fills: int = 0           # 多头成交次数 / Long fill count
    n_short_fills: int = 0          # 空头成交次数 / Short fill count
    fees_paid: float = 0.0          # 手续费 / Fees paid
    unrealized_pnl: float = 0.0     # 未实现盈亏 / Unrealized PnL


# ---------------------------------------------------------------------------
# PassivBotStrategy
# ---------------------------------------------------------------------------


class PassivBotStrategy:
    """PassivBot加密货币做市策略 / PassivBot cryptocurrency market-making strategy.

    核心概念 / Core Concepts:
    1. Forager: 追踪价格下限, 动态调整 / Trailing stop that follows price lows
    2. Grid-like spread: 等距买卖单 / Equally-spaced buy/sell orders
    3. 进化算法优化参数 / Evolutionary algorithm for parameter optimization

    用法示例 / Usage Example:
        strategy = PassivBotStrategy(
            symbol="BTCUSDT",
            spread=0.01,           # 1% spread
            grid_spacing=0.002,    # 0.2% grid spacing
            n_grids=10,
            position_size=100.0,
            drift_mode='long',     # 'long' | 'short' | 'neutral'
        )

        # 计算网格订单 / Compute grid orders
        orders = strategy.compute_grid_orders(current_price=50000.0)

        # 更新Forager / Update forager
        new_floor = strategy.update_forager(current_price=51000.0, highest_price=52000.0)

        # 检查是否应该平仓 / Check if should close position
        should_close = strategy.should_close_position(
            current_price=48000.0,
            entry_price=50000.0,
            forager_price=49000.0
        )
    """

    def __init__(
        self,
        symbol: str,
        spread: float = 0.01,         # 1% spread / 买卖价差
        grid_spacing: float = 0.002,  # 0.2% grid spacing / 网格间距
        n_grids: int = 10,             # 网格层数 / Number of grid levels
        position_size: float = 100.0,  # 仓位大小 / Position size
        drift_mode: str = 'long',      # 'long' | 'short' | 'neutral' / 漂移模式
        price_step: float = 0.01,     # 价格最小跳动 / Price tick size
        qty_step: float = 0.0001,      # 数量最小跳动 / Quantity tick size
        fee_rate: float = 0.0004,     # 手续费率 / Fee rate (0.04%)
        forager_offset: float = 0.01,  # Forager偏移 / Forager offset (1%)
    ):
        """初始化PassivBot策略 / Initialize PassivBot strategy.

        Args:
            symbol: 交易对符号 / Trading symbol (e.g. "BTCUSDT")
            spread: 买卖价差 (小数形式) / Bid-ask spread as decimal (0.01 = 1%)
            grid_spacing: 网格间距 (小数形式) / Grid spacing as decimal (0.002 = 0.2%)
            n_grids: 网格层数 / Number of grid levels per side
            position_size: 仓位大小(名义价值) / Position size (notional value)
            drift_mode: 漂移模式 / Drift mode: 'long', 'short', or 'neutral'
            price_step: 价格最小跳动 / Minimum price tick
            qty_step: 数量最小跳动 / Minimum quantity tick
            fee_rate: 手续费率 / Maker fee rate
            forager_offset: Forager止损偏移百分比 / Forager stop-loss offset percentage
        """
        self.symbol = symbol
        self.spread = spread
        self.grid_spacing = grid_spacing
        self.n_grids = n_grids
        self.position_size = position_size
        self.drift_mode = drift_mode
        self.price_step = price_step
        self.qty_step = qty_step
        self.fee_rate = fee_rate
        self.forager_offset = forager_offset

        # 内部状态 / Internal state
        self._entry_price: float = 0.0
        self._position: float = 0.0     # 正=多头, 负=空头 / positive=long, negative=short
        self._forager_long = ForagerState()
        self._forager_short = ForagerState()
        self._balance: float = 0.0
        self._cumulative_pnl: float = 0.0
        self._fees_paid: float = 0.0
        self._long_fills: int = 0
        self._short_fills: int = 0
        self._highest_price: float = 0.0
        self._lowest_price: float = float('inf')

    # -------------------------------------------------------------------------
    # 核心方法 / Core Methods
    # -------------------------------------------------------------------------

    def compute_grid_orders(self, current_price: float) -> List[Dict]:
        """计算网格订单列表 / Compute grid orders.

        在当前价格周围生成一系列买卖订单.
        Generates a series of buy/sell orders around the current price.

        Args:
            current_price: 当前市场价格 / Current market price

        Returns:
            订单列表 / List of orders: [{'side', 'price', 'qty'}, ...]
        """
        orders: List[Dict] = []

        # 更新价格极值 / Update price extremes
        self._highest_price = max(self._highest_price, current_price)
        self._lowest_price = min(self._lowest_price, current_price)

        # 计算买卖价格 / Calculate bid and ask prices
        half_spread = current_price * self.spread / 2
        bid_price = round_to_step(current_price - half_spread, self.price_step)
        ask_price = round_to_step(current_price + half_spread, self.price_step)

        # 计算网格价格 / Calculate grid prices
        grid_prices_bid = []
        grid_prices_ask = []

        for i in range(1, self.n_grids + 1):
            offset = i * current_price * self.grid_spacing
            grid_prices_bid.append(round_to_step(bid_price - offset, self.price_step))
            grid_prices_ask.append(round_to_step(ask_price + offset, self.price_step))

        # 计算每层数量 / Calculate quantity per level
        qty_per_level = self.position_size / current_price / self.n_grids
        qty_per_level = round_to_step(qty_per_level, self.qty_step)

        # 根据漂移模式决定订单方向 / Determine order sides based on drift mode
        if self.drift_mode in ('long', 'neutral'):
            # 多头/中性模式: 放置买单 / Long/neutral: place buy orders
            for price in grid_prices_bid:
                if price > 0:
                    orders.append({
                        'side': 'long',
                        'price': price,
                        'qty': qty_per_level,
                    })

        if self.drift_mode in ('short', 'neutral'):
            # 空头/中性模式: 放置卖单 / Short/neutral: place sell orders
            for price in grid_prices_ask:
                if price > 0:
                    orders.append({
                        'side': 'short',
                        'price': price,
                        'qty': qty_per_level,
                    })

        # 如果有持仓, 添加止盈单 / If position exists, add take-profit orders
        if abs(self._position) > 1e-10:
            tp_price = self._compute_take_profit_price(current_price)
            if tp_price > 0:
                orders.append({
                    'side': 'close_long' if self._position > 0 else 'close_short',
                    'price': tp_price,
                    'qty': abs(self._position),
                })

        return orders

    def update_forager(
        self,
        current_price: float,
        highest_price: float,
    ) -> float:
        """更新Forager (追踪止损) / Update Forager (trailing stop).

        Forager追踪价格下限, 当价格向上突破后回撤时激活.
        Forager tracks the price floor, activating when price pulls back
        after breaking above the highest price.

        Args:
            current_price: 当前价格 / Current price
            highest_price: 期间最高价 / Highest price since position opened

        Returns:
            new_forager_price: 新的Forager价格下限 / New forager price floor
        """
        # 更新最高价 / Update highest price
        self._highest_price = max(self._highest_price, highest_price)

        # 初始化Forager (如果未激活) / Initialize Forager if not active
        if not self._forager_long.enabled and self._position > 0:
            # 多头持仓激活Forager / Long position activates Forager
            self._forager_long.enabled = True
            self._forager_long.activation_price = highest_price
            self._forager_long.highest_price = highest_price
            # Forager价格下限 = 最高价 - 偏移 / Forager floor = highest - offset
            self._forager_long.price_floor = round_to_step(
                highest_price * (1 - self.forager_offset),
                self.price_step
            )

        if not self._forager_short.enabled and self._position < 0:
            # 空头持仓激活Forager / Short position activates Forager
            self._forager_short.enabled = True
            self._forager_short.activation_price = highest_price
            self._forager_short.highest_price = highest_price
            self._forager_short.price_floor = round_to_step(
                highest_price * (1 + self.forager_offset),
                self.price_step
            )

        # 更新Forager状态 / Update Forager state
        if self._position > 0 and self._forager_long.enabled:
            # 多头Forager: 追踪更高的高点 / Long Forager: trail higher highs
            if current_price > self._forager_long.highest_price:
                self._forager_long.highest_price = current_price
                # 上移价格下限 / Move floor up
                self._forager_long.price_floor = round_to_step(
                    current_price * (1 - self.forager_offset),
                    self.price_step
                )

        if self._position < 0 and self._forager_short.enabled:
            # 空头Forager: 追踪更低的低点 / Short Forager: trail lower lows
            if current_price < self._forager_short.highest_price:
                self._forager_short.highest_price = current_price
                self._forager_short.price_floor = round_to_step(
                    current_price * (1 + self.forager_offset),
                    self.price_step
                )

        # 返回当前Forager价格下限 / Return current Forager price floor
        if self._position > 0:
            return self._forager_long.price_floor
        elif self._position < 0:
            return self._forager_short.price_floor
        return 0.0

    def should_close_position(
        self,
        current_price: float,
        entry_price: float,
        forager_price: float,
    ) -> bool:
        """判断是否应该平仓 / Determine if position should be closed.

        平仓条件 / Close position if:
        1. 价格触及Forager止损线 / Price hits forager stop-loss
        2. 价格回撤超过阈值 / Price retraces beyond threshold

        Args:
            current_price: 当前价格 / Current price
            entry_price: 入场价格 / Entry price
            forager_price: Forager价格下限 / Forager price floor

        Returns:
            True if should close position / 是否应该平仓
        """
        if abs(self._position) < 1e-10:
            return False

        if self._position > 0:
            # 多头: 如果价格跌破Forager下限则平仓 / Long: close if price drops below floor
            if forager_price > 0 and current_price <= forager_price:
                return True
        elif self._position < 0:
            # 空头: 如果价格涨过Forager上限则平仓 / Short: close if price rises above ceiling
            if forager_price > 0 and current_price >= forager_price:
                return True

        return False

    def compute_profit(self) -> Dict[str, Any]:
        """计算当前利润 / Compute current profit.

        Returns:
            利润字典 / Profit dictionary:
            {
                'total_pnl': float,
                'long_pnl': float,
                'short_pnl': float,
                'n_long_fills': int,
                'n_short_fills': int,
                'fees_paid': float,
                'unrealized_pnl': float,
            }
        """
        unrealized = 0.0
        if self._position > 0 and self._entry_price > 0:
            # 多头未实现盈亏 / Long unrealized PnL
            unrealized = (self._lowest_price - self._entry_price) * self._position
        elif self._position < 0 and self._entry_price > 0:
            # 空头未实现盈亏 / Short unrealized PnL
            unrealized = (self._entry_price - self._highest_price) * abs(self._position)

        return {
            'total_pnl': self._cumulative_pnl + unrealized,
            'long_pnl': self._cumulative_pnl if self._position > 0 else 0.0,
            'short_pnl': self._cumulative_pnl if self._position < 0 else 0.0,
            'n_long_fills': self._long_fills,
            'n_short_fills': self._short_fills,
            'fees_paid': self._fees_paid,
            'unrealized_pnl': unrealized,
        }

    # -------------------------------------------------------------------------
    # 持仓管理 / Position Management
    # -------------------------------------------------------------------------

    def open_position(
        self,
        side: str,
        price: float,
        qty: float,
    ) -> None:
        """开仓 / Open position.

        Args:
            side: 'long' or 'short'
            price: 成交价格 / Fill price
            qty: 成交数量 / Fill quantity
        """
        if side == 'long':
            self._position += qty
            self._long_fills += 1
        elif side == 'short':
            self._position -= qty
            self._short_fills += 1

        if self._entry_price == 0:
            self._entry_price = price
        else:
            # 计算新的平均入场价 / Calculate new average entry price
            total_value = self._entry_price * abs(self._position - qty if side == 'long' else self._position + qty) + price * qty
            self._entry_price = total_value / abs(self._position)

        # 记录手续费 / Record fee
        self._fees_paid += price * qty * self.fee_rate

    def close_position(
        self,
        price: float,
        qty: Optional[float] = None,
    ) -> float:
        """平仓 / Close position.

        Args:
            price: 成交价格 / Fill price
            qty: 平仓数量 (None=全部平) / Quantity to close (None=close all)

        Returns:
            平仓盈亏 / Close PnL
        """
        if qty is None:
            qty = abs(self._position)

        pnl = 0.0
        if self._position > 0:
            # 平多 / Close long
            pnl = (price - self._entry_price) * qty
            self._position -= qty
        elif self._position < 0:
            # 平空 / Close short
            pnl = (self._entry_price - price) * qty
            self._position += qty

        self._cumulative_pnl += pnl
        self._fees_paid += price * qty * self.fee_rate

        if abs(self._position) < 1e-10:
            self._position = 0.0
            self._entry_price = 0.0
            self._forager_long = ForagerState()
            self._forager_short = ForagerState()

        return pnl

    def reset(self) -> None:
        """重置策略状态 / Reset strategy state."""
        self._entry_price = 0.0
        self._position = 0.0
        self._forager_long = ForagerState()
        self._forager_short = ForagerState()
        self._cumulative_pnl = 0.0
        self._fees_paid = 0.0
        self._long_fills = 0
        self._short_fills = 0
        self._highest_price = 0.0
        self._lowest_price = float('inf')

    # -------------------------------------------------------------------------
    # 内部方法 / Internal Methods
    # -------------------------------------------------------------------------

    def _compute_take_profit_price(self, current_price: float) -> float:
        """计算止盈价格 / Calculate take-profit price."""
        if abs(self._position) < 1e-10:
            return 0.0

        if self._position > 0:
            # 多头止盈: 价格 > 入场价 / Long TP: price > entry
            return round_to_step(current_price * (1 + self.spread), self.price_step)
        else:
            # 空头止盈: 价格 < 入场价 / Short TP: price < entry
            return round_to_step(current_price * (1 - self.spread), self.price_step)


# ---------------------------------------------------------------------------
# PassivBotOptimizer
# ---------------------------------------------------------------------------


class PassivBotOptimizer:
    """PassivBot进化优化器 / PassivBot evolutionary optimizer.

    使用进化算法优化 / Uses evolutionary algorithm to optimize:
    - spread: 买卖价差 / Bid-ask spread
    - grid_spacing: 网格间距 / Grid spacing
    - n_grids: 网格层数 / Number of grid levels
    - position_size: 仓位大小 / Position size

    用法示例 / Usage Example:
        optimizer = PassivBotOptimizer(population_size=50)

        # 定义优化目标 / Define optimization objective
        def evaluate(params, history):
            # 运行回测并返回适应度 / Run backtest and return fitness
            returnSharpeRatio

        best_params, best_fitness = optimizer.optimize(
            n_generations=100,
            evaluate_fn=evaluate,
        )
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.1,
    ):
        """初始化进化优化器 / Initialize evolutionary optimizer.

        Args:
            population_size: 种群大小 / Population size
            mutation_rate: 突变率 / Mutation rate
            crossover_rate: 交叉率 / Crossover rate
            elite_ratio: 精英比例 / Elite ratio (top individuals pass unchanged)
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio

        # 参数边界 / Parameter bounds
        self.param_bounds = {
            'spread': (0.001, 0.05),           # 0.1% - 5%
            'grid_spacing': (0.001, 0.01),     # 0.1% - 1%
            'n_grids': (5, 50),               # 5 - 50 grids
            'position_size': (50.0, 500.0),   # 50 - 500
        }

        self._population: List[Dict[str, float]] = []
        self._fitness_cache: Dict[int, float] = {}
        self._generation: int = 0

    def fitness(self, params: Dict[str, float], history: Any = None) -> float:
        """评估适应度 / Evaluate fitness.

        子类应重写此方法实现具体的回测逻辑.
        Subclasses should override this to implement specific backtest logic.

        Args:
            params: 待评估的参数 / Parameters to evaluate
            history: 历史数据 / Historical data

        Returns:
            适应度值 (越高越好) / Fitness value (higher is better)
        """
        # 基础评估: 参数合理性得分 / Basic evaluation: parameter合理性 score
        score = 0.0

        # spread评分: 中间值最好 / spread score: middle is best
        spread = params['spread']
        spread_optimal = 0.01
        score += 1.0 - abs(spread - spread_optimal) / spread_optimal

        # grid_spacing评分: 不太大不太小 / grid_spacing score: not too large or small
        spacing = params['grid_spacing']
        spacing_optimal = 0.003
        score += 1.0 - abs(spacing - spacing_optimal) / spacing_optimal

        # n_grids评分: 适中间值 / n_grids score: moderate is best
        n_grids = int(params['n_grids'])
        n_grids_optimal = 15
        score += 1.0 - abs(n_grids - n_grids_optimal) / n_grids_optimal

        # position_size评分: 中间值最好 / position_size score: middle is best
        pos_size = params['position_size']
        pos_optimal = 200.0
        score += 1.0 - abs(pos_size - pos_optimal) / pos_optimal

        return max(0.01, score / 4.0)

    def optimize(
        self,
        n_generations: int = 100,
        evaluate_fn: Optional[callable] = None,
        history: Any = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """进化优化 / Evolutionary optimization.

        Args:
            n_generations: 迭代代数 / Number of generations
            evaluate_fn: 评估函数(params, history) -> fitness / Evaluation function
            history: 历史数据用于回测 / Historical data for backtesting
            verbose: 是否打印进度 / Print progress

        Returns:
            {best_params, best_fitness}: 最佳参数和适应度 / Best parameters and fitness
        """
        eval_fn = evaluate_fn if evaluate_fn is not None else self.fitness

        # 初始化种群 / Initialize population
        self._population = [
            self._random_individual() for _ in range(self.population_size)
        ]
        self._generation = 0

        best_individual = None
        best_fitness = float('-inf')

        for gen in range(n_generations):
            self._generation = gen

            # 评估种群 / Evaluate population
            fitness_scores = []
            for i, individual in enumerate(self._population):
                # 使用缓存 / Use cache
                cache_key = hash(tuple(sorted(individual.items())))
                if cache_key in self._fitness_cache:
                    score = self._fitness_cache[cache_key]
                else:
                    score = eval_fn(individual, history)
                    self._fitness_cache[cache_key] = score
                fitness_scores.append(score)

            # 找最佳个体 / Find best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_individual = self._population[best_idx].copy()

            if verbose and gen % 10 == 0:
                logger.info(
                    f"Generation {gen}: best_fitness={best_fitness:.4f}, "
                    f"params={best_individual}"
                )

            # 精英保留 / Elitism
            elite_size = max(1, int(self.population_size * self.elite_ratio))
            sorted_indices = sorted(
                range(len(fitness_scores)),
                key=lambda i: fitness_scores[i],
                reverse=True
            )
            elites = [self._population[i] for i in sorted_indices[:elite_size]]

            # 生成新一代 / Generate new generation
            new_population = elites.copy()

            while len(new_population) < self.population_size:
                # 选择 / Selection
                parent1 = self._tournament_select(fitness_scores)
                parent2 = self._tournament_select(fitness_scores)

                # 交叉 / Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # 突变 / Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            # 截断到种群大小 / Truncate to population size
            self._population = new_population[:self.population_size]

        return {
            'best_params': best_individual,
            'best_fitness': best_fitness,
        }

    # -------------------------------------------------------------------------
    # 进化算子 / Evolutionary Operators
    # -------------------------------------------------------------------------

    def _random_individual(self) -> Dict[str, float]:
        """生成随机个体 / Generate random individual."""
        return {
            'spread': random.uniform(*self.param_bounds['spread']),
            'grid_spacing': random.uniform(*self.param_bounds['grid_spacing']),
            'n_grids': random.randint(*self.param_bounds['n_grids']),
            'position_size': random.uniform(*self.param_bounds['position_size']),
        }

    def _tournament_select(self, fitness_scores: List[float]) -> Dict[str, float]:
        """锦标赛选择 / Tournament selection."""
        k = 3  # 锦标赛大小 / tournament size
        indices = random.sample(range(len(self._population)), min(k, len(self._population)))
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return self._population[best_idx].copy()

    def _crossover(
        self,
        parent1: Dict[str, float],
        parent2: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """均匀交叉 / Uniform crossover."""
        child1, child2 = {}, {}

        for key in parent1:
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """高斯突变 / Gaussian mutation."""
        mutated = individual.copy()

        for key, value in mutated.items():
            if random.random() < self.mutation_rate:
                if key == 'n_grids':
                    # 整数突变 / Integer mutation
                    delta = random.randint(-3, 3)
                    mutated[key] = max(
                        self.param_bounds['n_grids'][0],
                        min(self.param_bounds['n_grids'][1], value + delta)
                    )
                else:
                    # 高斯突变 / Gaussian mutation
                    bounds = self.param_bounds[key]
                    range_size = bounds[1] - bounds[0]
                    delta = random.gauss(0, range_size * 0.1)
                    mutated[key] = max(bounds[0], min(bounds[1], value + delta))

        return mutated


# ---------------------------------------------------------------------------
# 工具函数 / Utility Functions
# ---------------------------------------------------------------------------


def round_to_step(value: float, step: float) -> float:
    """四舍五入到最小跳动 / Round to nearest step."""
    if step <= 0:
        return value
    return round(value / step) * step


def calculate_profit(
    entry_price: float,
    exit_price: float,
    side: str,
    qty: float,
    fee_rate: float = 0.0004,
) -> float:
    """计算交易利润 / Calculate trade profit.

    Args:
        entry_price: 入场价格 / Entry price
        exit_price: 出场价格 / Exit price
        side: 'long' or 'short'
        qty: 数量 / Quantity
        fee_rate: 手续费率 / Fee rate

    Returns:
        净利润 / Net profit
    """
    if side == 'long':
        gross_pnl = (exit_price - entry_price) * qty
    else:
        gross_pnl = (entry_price - exit_price) * qty

    fees = (entry_price + exit_price) * qty * fee_rate
    return gross_pnl - fees


# ---------------------------------------------------------------------------
# 简化的Backtest接口 / Simple Backtest Interface
# ---------------------------------------------------------------------------


def backtest_passivbot(
    params: Dict[str, float],
    prices: List[float],
    initial_balance: float = 10000.0,
    fee_rate: float = 0.0004,
) -> float:
    """简单的PassivBot回测 / Simple PassivBot backtest.

    Args:
        params: PassivBot参数 / PassivBot parameters
        prices: 价格序列 / Price series
        initial_balance: 初始资金 / Initial balance
        fee_rate: 手续费率 / Fee rate

    Returns:
        最终收益 / Final return
    """
    strategy = PassivBotStrategy(
        symbol="TEST",
        spread=params['spread'],
        grid_spacing=params['grid_spacing'],
        n_grids=int(params['n_grids']),
        position_size=params['position_size'],
        drift_mode='neutral',
        fee_rate=fee_rate,
    )

    balance = initial_balance
    position = 0.0
    entry_price = 0.0

    for i, price in enumerate(prices):
        orders = strategy.compute_grid_orders(price)

        for order in orders:
            if order['side'] == 'long' and position == 0:
                # 开多 / Open long
                cost = order['price'] * order['qty']
                if cost <= balance:
                    balance -= cost
                    position += order['qty']
                    entry_price = order['price']
                    strategy.open_position('long', order['price'], order['qty'])

            elif order['side'] == 'short' and position == 0:
                # 开空 / Open short
                strategy.open_position('short', order['price'], order['qty'])
                position -= order['qty']
                entry_price = order['price']
                balance += order['price'] * order['qty'] - order['price'] * order['qty'] * fee_rate

            elif order['side'] == 'close_long' and position > 0:
                # 平多 / Close long
                pnl = calculate_profit(entry_price, order['price'], 'long', position, fee_rate)
                balance += position * entry_price + pnl
                position = 0.0
                entry_price = 0.0

            elif order['side'] == 'close_short' and position < 0:
                # 平空 / Close short
                pnl = calculate_profit(entry_price, order['price'], 'short', abs(position), fee_rate)
                balance += abs(position) * entry_price + pnl
                position = 0.0
                entry_price = 0.0

    # 平仓 / Close position
    if position != 0:
        final_price = prices[-1]
        side = 'long' if position > 0 else 'short'
        pnl = calculate_profit(entry_price, final_price, side, abs(position), fee_rate)
        balance += pnl

    return balance - initial_balance
