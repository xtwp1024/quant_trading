"""
Triangular Arbitrage Engine
三角套利引擎
==========================

三角套利检测引擎，支持多交易所 (Binance/Kucoin/Okx/Huobi)

核心策略:
1. 发现所有可能的三角路径 (A→B→C→A)
2. 计算每条路径的理论利润
3. 考虑手续费后筛选正利润路径
4. 执行最优路径

三角示例 (BTC基础):
    BTC → ETH → BNB → BTC
    BTC → USDT → ETH → BTC

Usage:
    from quant_trading.arbitrage.triangular import TriangularArbitrageEngine, TrianglePath, ArbitrageOpportunity

    engine = TriangularArbitrageEngine(min_profit_pct=0.001, fee_tier=0.001, exchange='binance')
    opportunities = engine.scan_opportunities()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

__all__ = [
    "TrianglePath",
    "ArbitrageOpportunity",
    "TriangularArbitrageEngine",
]


@dataclass
class TrianglePath:
    """三角路径: A→B→C→A.

    Triangular path representation for arbitrage.

    Attributes:
        base: 基础货币 (e.g., 'BTC')
        step1: 第一步货币 (e.g., 'ETH')
        step2: 第二步货币 (e.g., 'BNB')

    Example:
        >>> path = TrianglePath(base='BTC', step1='ETH', step2='BNB')
        >>> path.path
        ['BTC', 'ETH', 'BNB', 'BTC']
        >>> path.symbol1
        'BTCETH'
        >>> path.symbol2
        'ETHBNB'
        >>> path.symbol3
        'BNBBTC'
    """
    base: str   #: 基础货币 (e.g., 'BTC')
    step1: str  #: 第一步货币 (e.g., 'ETH')
    step2: str  #: 第二步货币 (e.g., 'BNB')

    @property
    def path(self) -> list[str]:
        """返回完整路径列表 [base, step1, step2, base]."""
        return [self.base, self.step1, self.step2, self.base]

    @property
    def symbol1(self) -> str:
        """第一交易对: base/step1 (e.g., 'BTCETH')."""
        return f"{self.base}{self.step1}"

    @property
    def symbol2(self) -> str:
        """第二交易对: step1/step2 (e.g., 'ETHBNB')."""
        return f"{self.step1}{self.step2}"

    @property
    def symbol3(self) -> str:
        """第三交易对: step2/base (e.g., 'BNBBTC')."""
        return f"{self.step2}{self.base}"

    @property
    def ccxt_symbols(self) -> tuple[str, str, str]:
        """返回CCXT格式的交易对元组."""
        return (
            f"{self.base}/{self.step1}",
            f"{self.step1}/{self.step2}",
            f"{self.step2}/{self.base}",
        )


@dataclass
class ArbitrageOpportunity:
    """套利机会 / Arbitrage Opportunity.

    Represents a detected triangular arbitrage opportunity with profit metrics.

    Attributes:
        triangle: 三角路径
        profit_pct: 利润率 (扣除费用后, 百分比)
        gross_profit_pct: 毛利润 (未扣费用, 百分比)
        fees: 总手续费
        volume: 可用交易量 (quote currency)
        confidence: 置信度 [0, 1]
        timestamp: 检测时间
        exchange: 交易所名称
    """
    triangle: TrianglePath
    profit_pct: float          #: 利润率 (扣除费用后, %)
    gross_profit_pct: float     #: 毛利润 (未扣费用, %)
    fees: float                 #: 总手续费
    volume: float               #: 可用交易量
    confidence: float           #: 置信度 [0, 1]
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: str = 'binance'

    def __repr__(self) -> str:
        path = f"{self.triangle.base}→{self.triangle.step1}→{self.triangle.step2}→{self.triangle.base}"
        return (
            f"ArbitrageOpportunity(path={path}, profit={self.profit_pct:.4f}%, "
            f"volume={self.volume:.2f}, confidence={self.confidence:.2f})"
        )


class TriangularArbitrageEngine:
    """三角套利引擎 / Triangular Arbitrage Engine.

    策略逻辑:
    1. 发现所有可能的三角路径
    2. 计算每条路径的理论利润
    3. 考虑手续费后筛选正利润路径
    4. 执行最优路径

    三角利润计算公式:
        假设初始金额为 1 单位 base货币:

        步骤1: base → step1
            buy step1 with base at ask price
            amount_step1 = 1 / price1

        步骤2: step1 → step2
            sell step1 for step2 at bid price
            amount_step2 = amount_step1 * price2

        步骤3: step2 → base
            sell step2 for base at bid price
            final_base = amount_step2 * price3

        毛利润: final_base - 1
        净利润: final_base - 1 - fees

    Args:
        min_profit_pct: 最小利润阈值 (默认 0.001 = 0.1%)
        fee_tier: 手续费率 (默认 0.001 = 0.1% maker fee)
        exchange: 交易所名称 (默认 'binance')
    """

    # 支持的交易所及默认手续费
    EXCHANGE_FEES: dict[str, float] = {
        'binance': 0.001,   # 0.1%
        'kucoin': 0.001,    # 0.1%
        'okx': 0.001,       # 0.1%
        'huobi': 0.002,     # 0.2%
    }

    # 跳过的不稳定交易对
    UNAVAILABLE_PAIRS: set[str] = {
        'YGG/BNB', 'RAD/BNB', 'VOXEL/BNB', 'GLMR/BNB', 'UNI/EUR',
    }

    def __init__(
        self,
        min_profit_pct: float = 0.001,
        fee_tier: float = 0.001,
        exchange: str = 'binance',
    ) -> None:
        """初始化三角套利引擎.

        Args:
            min_profit_pct: 最小利润阈值 (默认 0.001 = 0.1%)
            fee_tier: 手续费率 (默认 0.001 = 0.1% maker fee)
            exchange: 交易所名称 (默认 'binance')
        """
        if min_profit_pct < 0:
            raise ValueError("min_profit_pct must be non-negative")
        if fee_tier < 0 or fee_tier > 1:
            raise ValueError("fee_tier must be in [0, 1]")
        if exchange not in self.EXCHANGE_FEES:
            raise ValueError(f"Unsupported exchange: {exchange}. Supported: {list(self.EXCHANGE_FEES.keys())}")

        self.min_profit_pct = min_profit_pct
        self.fee_tier = fee_tier
        self.exchange = exchange

        # 内部状态
        self._markets: dict = {}
        self._tickers: dict = {}
        self._symbols_by_base: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_all_triangles(self, available_symbols: list[str]) -> list[TrianglePath]:
        """发现所有可能的三角路径.

        从可用交易对中发现所有可能的 A→B→C→A 三角路径。

        Args:
            available_symbols: 可用的交易对列表 (CCXT格式, 如 'BTC/ETH')

        Returns:
            TrianglePath 列表

        Example:
            >>> symbols = ['BTC/ETH', 'ETH/BNB', 'BNB/BTC', 'BTC/USDT', 'ETH/USDT']
            >>> engine = TriangularArbitrageEngine()
            >>> triangles = engine.find_all_triangles(symbols)
        """
        triangles: list[TrianglePath] = []

        # 构建 base -> set of symbols 索引
        symbols_by_base: dict[str, set[str]] = {}
        for symbol in available_symbols:
            if '/' not in symbol:
                continue
            base, quote = symbol.split('/')
            if base not in symbols_by_base:
                symbols_by_base[base] = set()
            symbols_by_base[base].add(symbol)

        # 对每个 quote 货币作为 base，发现三角路径
        for base, symbols in symbols_by_base.items():
            for symbol in symbols:
                if '/' not in symbol:
                    continue
                # 第一步: base/quote (buy quote with base)
                # e.g., BTC/ETH -> base=BTC, quote=ETH
                first_base, first_quote = symbol.split('/')
                if first_base != base:
                    continue

                step1 = first_quote  # e.g., ETH

                # 第二步: step1 交易对 (sell step1 for something)
                step1_symbols = symbols_by_base.get(step1, set())
                for step1_symbol in step1_symbols:
                    if '/' not in step1_symbol:
                        continue
                    if step1_symbol in self.UNAVAILABLE_PAIRS:
                        continue

                    s1_base, s1_quote = step1_symbol.split('/')
                    if s1_base != step1:
                        # step1 是 quote, base 是另一个货币
                        step2 = s1_base
                    else:
                        # step1 是 base
                        step2 = s1_quote

                    # 第三步: step2 回 base 的交易对
                    third_symbol_candidates = [
                        f"{step2}/{base}",
                        f"{base}/{step2}",
                    ]
                    third_found = False
                    for candidate in third_symbol_candidates:
                        if candidate in symbols_by_base.get(step2, set()) or \
                           candidate in symbols_by_base.get(base, set()):
                            third_found = True
                            break

                    if not third_found:
                        continue

                    # 确认第三交易对存在
                    third_symbol = f"{step2}/{base}"
                    if third_symbol not in available_symbols:
                        third_symbol = f"{base}/{step2}"
                        if third_symbol not in available_symbols:
                            continue

                    # 构建三角路径
                    triangle = TrianglePath(base=base, step1=step1, step2=step2)

                    # 验证所有三个交易对都存在
                    sym1, sym2, sym3 = triangle.ccxt_symbols
                    if sym1 not in available_symbols:
                        continue
                    if sym2 not in available_symbols:
                        continue
                    if sym3 not in available_symbols:
                        continue

                    triangles.append(triangle)

        return triangles

    def calculate_opportunity(
        self,
        triangle: TrianglePath,
        prices: dict[str, dict],
        volumes: Optional[dict[str, dict]] = None,
    ) -> ArbitrageOpportunity | None:
        """计算单个三角的套利机会.

        基于当前市场价格计算给定三角路径的套利机会。

        公式:
            1. BTC → ETH: 用BTC买ETH (付钱, 获得ETH)
            2. ETH → BNB: 用ETH买BNB
            3. BNB → BTC: 卖BNB换BTC

            初始: 1 BTC
            步骤1: 获得 ETH = 1 / price_BTCETH (ask, 买入价)
            步骤2: 获得 BNB = ETH * price_ETHBNB (bid, 卖出价)
            步骤3: 获得 BTC = BNB * price_BNBBTC (bid, 卖出价)
            最终: final_BTC - 1 = 利润

        Args:
            triangle: 三角路径
            prices: 价格字典 {symbol: {'ask': float, 'bid': float}}
                   e.g., {'BTC/ETH': {'ask': 20.0, 'bid': 19.9}, ...}
            volumes: 交易量字典 (可选) {symbol: {'ask': float, 'bid': float}}

        Returns:
            ArbitrageOpportunity 如果计算成功且利润为正，否则 None
        """
        sym1, sym2, sym3 = triangle.ccxt_symbols

        # 获取价格
        p1 = prices.get(sym1, {})
        p2 = prices.get(sym2, {})
        p3 = prices.get(sym3, {})

        ask1 = p1.get('ask')
        bid2 = p2.get('bid')
        bid3 = p3.get('bid')

        # 需要 ask1 (买入 step1 的价格) 和两个 bid (卖出价格)
        if ask1 is None or bid2 is None or bid3 is None:
            return None

        if ask1 <= 0 or bid2 <= 0 or bid3 <= 0:
            return None

        # 计算毛利润 (1 单位 base 出发)
        # 步骤1: 用 base 买 step1
        amount_step1 = 1.0 / ask1
        # 步骤2: 卖 step1 买 step2
        amount_step2 = amount_step1 * bid2
        # 步骤3: 卖 step2 买 base
        final_base = amount_step2 * bid3

        gross_profit_pct = (final_base - 1.0) * 100.0

        # 手续费: 每笔交易双向收费
        # 实际交易中:
        #   步骤1: 买 - fee on quote (base)
        #   步骤2: 卖 - fee on base (step1)
        #   步骤3: 卖 - fee on quote (step2)
        # 简化: 3笔交易, 每笔 fee_tier
        total_fees_pct = self.fee_tier * 3 * 100.0
        profit_pct = gross_profit_pct - total_fees_pct

        # 检查是否满足最小利润要求
        if profit_pct < self.min_profit_pct * 100.0:
            return None

        # 计算可用交易量
        volume = float('inf')
        if volumes:
            v1 = volumes.get(sym1, {}).get('ask', float('inf'))
            v2 = volumes.get(sym2, {}).get('bid', float('inf'))
            v3 = volumes.get(sym3, {}).get('bid', float('inf'))
            # 取最小值(流动性限制)
            volume = min(v1, v2, v3) if all(x > 0 for x in [v1, v2, v3]) else 0.0

        # 置信度: 基于利润率和交易量
        confidence = self._calculate_confidence(profit_pct, volume)

        return ArbitrageOpportunity(
            triangle=triangle,
            profit_pct=profit_pct,
            gross_profit_pct=gross_profit_pct,
            fees=total_fees_pct,
            volume=volume,
            confidence=confidence,
            timestamp=datetime.now(),
            exchange=self.exchange,
        )

    def scan_opportunities(
        self,
        prices: dict[str, dict],
        volumes: Optional[dict[str, dict]] = None,
    ) -> list[ArbitrageOpportunity]:
        """扫描所有三角路径, 返回正利润机会.

        对所有已知三角路径进行扫描，返回利润为正的机会列表。

        Args:
            prices: 价格字典 {symbol: {'ask': float, 'bid': float}}
            volumes: 交易量字典 (可选)

        Returns:
            按利润降序排列的 ArbitrageOpportunity 列表

        Example:
            >>> prices = {
            ...     'BTC/ETH': {'ask': 20.0, 'bid': 19.9},
            ...     'ETH/BNB': {'ask': 0.05, 'bid': 0.049},
            ...     'BNB/BTC': {'ask': 0.001, 'bid': 0.0009},
            ... }
            >>> engine = TriangularArbitrageEngine()
            >>> opps = engine.scan_opportunities(prices)
        """
        opportunities: list[ArbitrageOpportunity] = []

        # 从价格字典中提取所有 symbol
        available_symbols = list(prices.keys())

        # 发现所有三角路径
        triangles = self.find_all_triangles(available_symbols)

        # 计算每个三角的机会
        for triangle in triangles:
            opp = self.calculate_opportunity(triangle, prices, volumes)
            if opp is not None:
                opportunities.append(opp)

        # 按利润降序排列
        opportunities.sort(key=lambda x: x.profit_pct, reverse=True)

        return opportunities

    def execute(
        self,
        opportunity: ArbitrageOpportunity,
        capital: float,
    ) -> dict:
        """执行套利交易 (模拟/估算).

        基于套利机会估算执行结果，不进行实际交易。

        Args:
            opportunity: 套利机会
            capital: 初始资金 (base 货币)

        Returns:
            执行结果字典:
                - initial_capital: 初始资金
                - final_capital: 最终资金
                - profit: 利润
                - profit_pct: 利润百分比
                - fees: 手续费
                - trades: 交易详情列表
        """
        triangle = opportunity.triangle
        sym1, sym2, sym3 = triangle.ccxt_symbols

        # 模拟执行
        prices = {
            sym1: {'ask': capital / (capital / 1.0)},  # placeholders
            sym2: {'bid': 1.0},
            sym3: {'bid': 1.0},
        }

        # 计算交易步骤
        initial = capital
        fee_rate = self.fee_tier

        # 步骤1: 用 capital 买 step1
        step1_amount = capital / capital  # normalize
        fee1 = capital * fee_rate
        step1_received = capital - fee1

        # 步骤2: 卖 step1 买 step2
        step2_received = step1_received * (1 - fee_rate)

        # 步骤3: 卖 step2 换 base
        final = step2_received * (1 - fee_rate)
        profit = final - initial
        profit_pct = (profit / initial) * 100.0 if initial > 0 else 0.0

        trades = [
            {'step': 1, 'symbol': sym1, 'side': 'buy', 'amount': step1_amount, 'fee': fee1},
            {'step': 2, 'symbol': sym2, 'side': 'sell', 'amount': step2_received, 'fee': step1_received * fee_rate},
            {'step': 3, 'symbol': sym3, 'side': 'sell', 'amount': final, 'fee': step2_received * fee_rate},
        ]

        return {
            'initial_capital': initial,
            'final_capital': final,
            'profit': profit,
            'profit_pct': profit_pct,
            'fees': sum(t['fee'] for t in trades),
            'trades': trades,
            'opportunity': opportunity,
        }

    def estimate_profit(
        self,
        triangle: TrianglePath,
        prices: dict[str, dict],
    ) -> float:
        """估算利润 (不执行).

        快速估算给定三角路径的利润百分比，不考虑交易量限制。

        Args:
            triangle: 三角路径
            prices: 价格字典

        Returns:
            估算的利润百分比 (扣除手续费后)
        """
        opp = self.calculate_opportunity(triangle, prices)
        if opp is None:
            return 0.0
        return opp.profit_pct

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_confidence(self, profit_pct: float, volume: float) -> float:
        """计算置信度 [0, 1].

        基于利润率和交易量综合计算置信度。
        """
        if volume <= 0:
            return 0.0

        # 利润率置信度 (利润越高, 置信度越高)
        # 利润 > 0.5% -> 满分, < 0.1% -> 低分
        profit_score = min(1.0, max(0.0, (profit_pct - 0.1) / 0.4))

        # 交易量置信度 (假设 > 10000 满, < 100 低分)
        if volume > 10000:
            volume_score = 1.0
        elif volume < 100:
            volume_score = 0.1
        else:
            volume_score = 0.1 + 0.9 * (math.log10(volume) - 2) / 2  # log scale 100-10000

        return (profit_score * 0.7 + volume_score * 0.3)
