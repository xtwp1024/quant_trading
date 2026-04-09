"""
Polymarket Prediction Market Arbitrage Framework
=================================================

Polymarket 预测市场套利框架，集成 5 种套利策略:

1. 概率偏差套利 (Probability Deviation): 检测 yes/no 价格 vs 真实概率的偏差
2. 流动性挖矿 (Liquidity Provision): 高流动性市场提供价差收益
3. 资金费率套利 (Funding Rate): 跨市场资金费率差异
4. 趋势跟随 (Momentum): 跟随大资金流向
5. 均值回归 (Mean Reversion): 高波动后价格回归

典型用法:
    from quant_trading.arbitrage.polymarket import (
        PolymarketArbitrageur,
        MarketSignal,
    )
    from quant_trading.arbitrage.polymarket_api import PolymarketClient

    client = PolymarketClient()
    arb = PolymarketArbitrageur(client, min_edge=0.02, max_position=1000.0)
    markets = await client.get_markets()
    signals = arb.strategy_probability_deviation(markets)
    top = arb.rank_signals(signals)[:3]
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from quant_trading.arbitrage.polymarket_api import (
    PolymarketClient,
    CrossPlatformFeeds,
    CRYPTO_ASSETS,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class MarketSignal:
    """市场信号 / Market signal from Polymarket.

    Attributes:
        market_id: 市场 condition_id.
        question: 市场问题描述.
        yes_price: YES 代币价格 (0-1).
        no_price: NO 代币价格 (0-1) = 1 - yes_price.
        volume: 成交量 (USD).
        liquidity: 流动性 (USD).
        Sharpe: 夏普比率 (可选).
    """
    market_id: str
    question: str
    yes_price: float
    no_price: float
    volume: float
    liquidity: float
    Sharpe: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "MarketSignal":
        """从 API 字典构建 MarketSignal."""
        yes = float(data.get("yes_price", 0) or 0)
        no = float(data.get("no_price", 0) or 0)
        if no == 0:
            no = 1.0 - yes
        return cls(
            market_id=data.get("market_id", data.get("condition_id", "")),
            question=data.get("question", ""),
            yes_price=yes,
            no_price=no,
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            Sharpe=data.get("Sharpe"),
        )


@dataclass
class ArbitrageSignal:
    """套利信号 / Arbitrage opportunity signal.

    Attributes:
        strategy: 策略名称.
        market_id: 市场 ID.
        direction: 交易方向 (BUY_YES / BUY_NO / BUY_ALL).
        entry_price: 预期入场价格.
        target_price: 预期退出/结算价格.
        expected_profit_pct: 预期利润百分比 (%).
        expected_profit_usd: 预期利润 (USD).
        confidence: 置信度 (0.0-1.0).
        urgency: 紧急度 (HIGH / MEDIUM / LOW).
        token_ids: token_id 字典.
        metadata: 额外元数据.
        timestamp: 信号生成时间.
    """
    strategy: str
    market_id: str
    direction: str
    entry_price: float
    target_price: float
    expected_profit_pct: float
    expected_profit_usd: float
    confidence: float
    urgency: str = "MEDIUM"
    token_ids: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def risk_reward_ratio(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return abs(self.target_price - self.entry_price) / self.entry_price

    def __repr__(self) -> str:
        return (
            f"Signal({self.strategy}|{self.market_id[:12]}...|{self.direction}|"
            f"profit={self.expected_profit_pct:.2f}%|conf={self.confidence:.2f})"
        )


@dataclass
class PriceCandle:
    """价格 K 线 (用于动量分析)."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


# ---------------------------------------------------------------------------
# Strategy 1: Probability Deviation Arbitrage
# ---------------------------------------------------------------------------

class ProbabilityDeviationStrategy:
    """概率偏差套利 / Probability Deviation Arbitrage.

    检测 yes/no 价格 vs 真实概率的偏差.
    逻辑: 如果 yes + no != 1 (考虑费用后), 存在套利机会.

    当 YES_price + NO_price < 1 - fees 时，买入两边必然盈利.
    当 YES_price + NO_price > 1 + fees 时，卖出两边必然盈利.
    """

    FEE_PCT = 2.0  # Polymarket 手续费 %
    GAS_COST_USD = 0.007  # Polygon gas 估算

    def __init__(self, min_edge: float = 0.02):
        self.min_edge = min_edge  # 最小边缘

    def analyze(self, signal: MarketSignal) -> Optional[ArbitrageSignal]:
        """检测概率偏差套利机会."""
        yes = signal.yes_price
        no = signal.no_price

        if yes <= 0 or no <= 0:
            return None

        total_cost = yes + no
        fee = total_cost * (self.FEE_PCT / 100)
        gas = self.GAS_COST_USD * 2  # 两笔交易
        total_with_fees = total_cost + fee + gas

        # 买入两边套利
        if total_with_fees < 1.0 - self.min_edge:
            profit = 1.0 - total_with_fees
            profit_pct = (profit / total_with_fees) * 100
            max_shares = 500.0 / total_with_fees
            profit_usd = profit * max_shares

            confidence = min(0.95, 0.5 + profit_pct * 0.1)

            return ArbitrageSignal(
                strategy="probability_deviation",
                market_id=signal.market_id,
                direction="BUY_ALL",
                entry_price=total_with_fees,
                target_price=1.0,
                expected_profit_pct=profit_pct,
                expected_profit_usd=profit_usd,
                confidence=confidence,
                urgency="HIGH",
                token_ids={"YES": signal.market_id + "_YES", "NO": signal.market_id + "_NO"},
                metadata={
                    "yes_price": yes,
                    "no_price": no,
                    "total_cost": total_cost,
                    "fee": fee,
                    "gas": gas,
                    "edge": 1.0 - total_with_fees,
                },
            )

        # 卖出两边套利 (极少见)
        total_with_fees_sell = total_cost - fee
        if total_with_fees_sell > 1.0 + self.min_edge:
            profit = total_with_fees_sell - 1.0
            profit_pct = (profit / 1.0) * 100
            confidence = min(0.80, 0.3 + profit_pct * 0.1)

            return ArbitrageSignal(
                strategy="probability_deviation",
                market_id=signal.market_id,
                direction="SELL_ALL",
                entry_price=total_with_fees_sell,
                target_price=1.0,
                expected_profit_pct=profit_pct,
                expected_profit_usd=profit_pct * 5,
                confidence=confidence,
                urgency="MEDIUM",
                token_ids={"YES": signal.market_id + "_YES", "NO": signal.market_id + "_NO"},
                metadata={
                    "yes_price": yes,
                    "no_price": no,
                    "edge": total_with_fees_sell - 1.0,
                    "mode": "sell_both",
                },
            )

        return None


# ---------------------------------------------------------------------------
# Strategy 2: Liquidity Provision
# ---------------------------------------------------------------------------

class LiquidityProvisionStrategy:
    """流动性挖矿 / Liquidity Provision.

    在高流动性市场提供报价，赚取价差收益.
    逻辑: 当市场买卖价差大于正常水平时，提供流动性.
    """

    MIN_SPREAD_PCT = 1.0  # 最小价差百分比
    MAX_SPREAD_PCT = 10.0  # 最大价差 (超出可能有风险)

    def __init__(self, min_volume: float = 10000.0):
        self.min_volume = min_volume

    def analyze(self, signal: MarketSignal) -> Optional[ArbitrageSignal]:
        """检测流动性挖矿机会."""
        if signal.volume < self.min_volume:
            return None

        yes = signal.yes_price
        no = signal.no_price

        if yes <= 0 or no <= 0:
            return None

        # 计算当前价差
        spread = abs(yes - no)
        spread_pct = spread * 100

        if spread_pct < self.MIN_SPREAD_PCT or spread_pct > self.MAX_SPREAD_PCT:
            return None

        # 做市收益估算: 价差的一半作为预期收益
        expected_profit_pct = spread_pct / 2 - 0.5  # 扣除手续费
        if expected_profit_pct <= 0:
            return None

        # 确定方向
        if yes > no:
            direction = "PROVIDE_LIQUIDITY_YES"
            entry = yes
        else:
            direction = "PROVIDE_LIQUIDITY_NO"
            entry = no

        confidence = min(0.75, 0.3 + (spread_pct - self.MIN_SPREAD_PCT) * 0.05)
        position_usd = min(signal.liquidity * 0.01, 500)  # 最多500美元

        return ArbitrageSignal(
            strategy="liquidity_provision",
            market_id=signal.market_id,
            direction=direction,
            entry_price=entry,
            target_price=entry + spread / 2,
            expected_profit_pct=expected_profit_pct,
            expected_profit_usd=expected_profit_pct * position_usd / 100,
            confidence=confidence,
            urgency="MEDIUM",
            token_ids={"YES": signal.market_id + "_YES", "NO": signal.market_id + "_NO"},
            metadata={
                "spread_pct": spread_pct,
                "volume": signal.volume,
                "liquidity": signal.liquidity,
            },
        )


# ---------------------------------------------------------------------------
# Strategy 3: Funding Rate Arbitrage
# ---------------------------------------------------------------------------

class FundingRateStrategy:
    """资金费率套利 / Funding Rate Arbitrage.

    跨市场资金费率差异套利.
    逻辑: 检测相关市场间的概率差异，当偏离超过阈值时入场.
    """

    def __init__(self, min_diff: float = 0.05):
        self.min_diff = min_diff  # 最小概率差异

    def analyze(
        self,
        signal: MarketSignal,
        reference_price: Optional[float] = None,
    ) -> Optional[ArbitrageSignal]:
        """检测资金费率套利机会.

        Args:
            signal: 当前市场信号.
            reference_price: 参考价格 (如预言机价格或现货价格).
        """
        yes = signal.yes_price
        no = signal.no_price

        if yes <= 0 or no <= 0:
            return None

        implied_prob = yes
        if reference_price is not None and reference_price > 0:
            # 计算与参考价格的偏差
            deviation = abs(implied_prob - reference_price)

            if deviation < self.min_diff:
                return None

            # 方向判断
            if implied_prob > reference_price:
                direction = "BUY_NO"  # 认为概率被高估，做空
                entry = no
                target = reference_price
            else:
                direction = "BUY_YES"  # 认为概率被低估，做多
                entry = yes
                target = reference_price

            profit_pct = abs(target - entry) / entry * 100 - 2.0
            if profit_pct <= 0:
                return None

            position_usd = 200.0
            confidence = min(0.80, 0.3 + deviation * 5)

            return ArbitrageSignal(
                strategy="funding_rate",
                market_id=signal.market_id,
                direction=direction,
                entry_price=entry,
                target_price=target,
                expected_profit_pct=profit_pct,
                expected_profit_usd=profit_pct * position_usd / 100,
                confidence=confidence,
                urgency="MEDIUM",
                token_ids={"YES": signal.market_id + "_YES", "NO": signal.market_id + "_NO"},
                metadata={
                    "implied_prob": implied_prob,
                    "reference_price": reference_price,
                    "deviation": deviation,
                },
            )

        # 无参考价格时，检测 YES + NO != 1 的偏差
        total = yes + no
        deviation_from_parity = abs(total - 1.0)

        if deviation_from_parity < self.min_diff:
            return None

        # YES 价格偏低 -> 买入 YES
        if yes < 0.5:
            direction = "BUY_YES"
            entry = yes
        else:
            direction = "BUY_NO"
            entry = no

        profit_pct = deviation_from_parity * 100 - 2.0
        if profit_pct <= 0:
            return None

        position_usd = 200.0
        confidence = min(0.70, 0.2 + deviation_from_parity * 10)

        return ArbitrageSignal(
            strategy="funding_rate",
            market_id=signal.market_id,
            direction=direction,
            entry_price=entry,
            target_price=0.5,
            expected_profit_pct=profit_pct,
            expected_profit_usd=profit_pct * position_usd / 100,
            confidence=confidence,
            urgency="LOW",
            token_ids={"YES": signal.market_id + "_YES", "NO": signal.market_id + "_NO"},
            metadata={
                "total_price": total,
                "deviation_from_parity": deviation_from_parity,
            },
        )


# ---------------------------------------------------------------------------
# Strategy 4: Momentum (Trend Following)
# ---------------------------------------------------------------------------

class MomentumStrategy:
    """趋势跟随 / Trend Following Strategy.

    跟随大资金流向.
    逻辑: 检测价格动量，当趋势强劲时顺势交易.
    """

    def __init__(self, momentum_threshold: float = 0.02):
        self.momentum_threshold = momentum_threshold
        self._price_history: Dict[str, List[PriceCandle]] = {}

    def update_candle(self, market_id: str, candle: PriceCandle) -> None:
        """更新 K 线数据."""
        if market_id not in self._price_history:
            self._price_history[market_id] = []
        self._price_history[market_id].append(candle)
        # 保留最近 100 根 K 线
        if len(self._price_history[market_id]) > 100:
            self._price_history[market_id] = self._price_history[market_id][-100:]

    def analyze(self, signal: MarketSignal) -> Optional[ArbitrageSignal]:
        """检测趋势跟随机会."""
        candles = self._price_history.get(signal.market_id, [])

        if len(candles) < 5:
            return None

        # 计算动量指标
        recent = candles[-10:]
        closes = np.array([c.close for c in recent])

        roc = self._rate_of_change(closes, period=5)
        zscore = self._zscore(closes)
        rsi = self._rsi(closes)

        # 趋势确认: 动量足够强
        momentum_score = roc * 0.5 + zscore * 0.3 + (50 - rsi) * 0.02

        if abs(momentum_score) < self.momentum_threshold:
            return None

        yes = signal.yes_price

        if momentum_score > 0 and yes < 0.9:
            # 上涨趋势，买入 YES
            direction = "BUY_YES"
            entry = yes
            target = min(yes * 1.05, 0.98)
        elif momentum_score < 0 and yes > 0.1:
            # 下跌趋势，买入 NO
            direction = "BUY_NO"
            entry = 1.0 - yes
            target = max((1.0 - yes) * 1.05, 0.02)
        else:
            return None

        profit_pct = abs(target - entry) / entry * 100 - 2.0
        if profit_pct <= 0:
            return None

        confidence = min(0.75, 0.3 + abs(momentum_score) * 5)
        position_usd = 150.0

        return ArbitrageSignal(
            strategy="momentum",
            market_id=signal.market_id,
            direction=direction,
            entry_price=entry,
            target_price=target,
            expected_profit_pct=profit_pct,
            expected_profit_usd=profit_pct * position_usd / 100,
            confidence=confidence,
            urgency="MEDIUM",
            token_ids={"YES": signal.market_id + "_YES", "NO": signal.market_id + "_NO"},
            metadata={
                "roc": float(roc),
                "zscore": float(zscore),
                "rsi": float(rsi),
                "momentum_score": float(momentum_score),
                "candles_used": len(recent),
            },
        )

    @staticmethod
    def _rate_of_change(data: np.ndarray, period: int = 5) -> float:
        if len(data) < period + 1:
            return 0.0
        old = data[-period - 1]
        if old == 0:
            return 0.0
        return float(((data[-1] - old) / old) * 100)

    @staticmethod
    def _zscore(data: np.ndarray) -> float:
        if len(data) < 2:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float((data[-1] - mean) / std)

    @staticmethod
    def _rsi(data: np.ndarray, period: int = 14) -> float:
        if len(data) < period + 1:
            return 50.0
        deltas = np.diff(data[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))


# ---------------------------------------------------------------------------
# Strategy 5: Mean Reversion
# ---------------------------------------------------------------------------

class MeanReversionStrategy:
    """均值回归 / Mean Reversion Strategy.

    高波动后价格回归.
    逻辑: 当价格偏离均值过多时，预期回归.
    """

    def __init__(self, zscore_threshold: float = 2.0, lookback: int = 20):
        self.zscore_threshold = zscore_threshold
        self.lookback = lookback
        self._price_history: Dict[str, List[float]] = {}

    def update_price(self, market_id: str, price: float) -> None:
        """更新价格历史."""
        if market_id not in self._price_history:
            self._price_history[market_id] = []
        self._price_history[market_id].append(price)
        if len(self._price_history[market_id]) > self.lookback * 2:
            self._price_history[market_id] = self._price_history[market_id][-self.lookback * 2:]

    def analyze(self, signal: MarketSignal) -> Optional[ArbitrageSignal]:
        """检测均值回归机会."""
        history = self._price_history.get(signal.market_id, [])
        history.append(signal.yes_price)

        if len(history) < self.lookback:
            self._price_history[signal.market_id] = history
            return None

        lookback_data = history[-self.lookback:]
        mean = np.mean(lookback_data)
        std = np.std(lookback_data)
        current = signal.yes_price

        if std == 0:
            return None

        zscore = (current - mean) / std

        # 严重超卖 -> 买入 YES 博反弹
        if zscore < -self.zscore_threshold:
            entry = current
            target = mean
            direction = "BUY_YES"
            profit_pct = (mean - current) / current * 100 - 2.0
        # 严重超买 -> 买入 NO 博回调
        elif zscore > self.zscore_threshold:
            entry = 1.0 - current
            target = 1.0 - mean
            direction = "BUY_NO"
            profit_pct = (1.0 - mean - (1.0 - current)) / (1.0 - current) * 100 - 2.0
        else:
            return None

        if profit_pct <= 0:
            return None

        confidence = min(0.80, 0.3 + abs(zscore) * 0.1)
        position_usd = 200.0

        self._price_history[signal.market_id] = history

        return ArbitrageSignal(
            strategy="mean_reversion",
            market_id=signal.market_id,
            direction=direction,
            entry_price=entry,
            target_price=target,
            expected_profit_pct=profit_pct,
            expected_profit_usd=profit_pct * position_usd / 100,
            confidence=confidence,
            urgency="MEDIUM",
            token_ids={"YES": signal.market_id + "_YES", "NO": signal.market_id + "_NO"},
            metadata={
                "zscore": float(zscore),
                "mean": float(mean),
                "std": float(std),
                "current": current,
                "lookback": self.lookback,
            },
        )


# ---------------------------------------------------------------------------
# PolymarketArbitrageur
# ---------------------------------------------------------------------------

class PolymarketArbitrageur:
    """Polymarket 预测市场套利器 / Main arbitrage orchestrator.

    集成 5 种套利策略，统一信号生成、排名和执行:

    1. 概率偏差套利 (probability_deviation)
    2. 流动性挖矿 (liquidity_provision)
    3. 资金费率套利 (funding_rate)
    4. 趋势跟随 (momentum)
    5. 均值回归 (mean_reversion)

    Args:
        client: PolymarketClient 实例.
        min_edge: 最小套利边缘 (默认 0.02 = 2%).
        max_position: 最大持仓金额 USD (默认 1000).
    """

    # 策略优先级权重
    STRATEGY_PRIORITY = {
        "probability_deviation": 1.0,
        "liquidity_provision": 0.85,
        "funding_rate": 0.80,
        "momentum": 0.70,
        "mean_reversion": 0.75,
    }

    URGENCY_WEIGHTS = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}

    def __init__(
        self,
        client: PolymarketClient,
        min_edge: float = 0.02,
        max_position: float = 1000.0,
    ):
        self.client = client
        self.min_edge = min_edge
        self.max_position = max_position

        # 初始化 5 种策略
        self.strategy_prob = ProbabilityDeviationStrategy(min_edge=min_edge)
        self.strategy_liquidity = LiquidityProvisionStrategy()
        self.strategy_funding = FundingRateStrategy()
        self.strategy_momentum = MomentumStrategy()
        self.strategy_mean_reversion = MeanReversionStrategy()

        # 跨平台价格引用
        self.feeds = CrossPlatformFeeds()

    # -----------------------------------------------------------------------
    # Strategy Entry Points
    # -----------------------------------------------------------------------

    def strategy_probability_deviation(
        self,
        markets: List[MarketSignal],
    ) -> List[Dict]:
        """概率偏差套利 / Probability Deviation Arbitrage.

        Args:
            markets: MarketSignal 列表.

        Returns:
            套利信号字典列表.
        """
        signals = []
        for m in markets:
            sig = self.strategy_prob.analyze(m)
            if sig:
                signals.append(self._signal_to_dict(sig))
        return signals

    def strategy_liquidity_provision(
        self,
        markets: List[MarketSignal],
    ) -> List[Dict]:
        """流动性挖矿 / Liquidity Provision.

        Args:
            markets: MarketSignal 列表.

        Returns:
            套利信号字典列表.
        """
        signals = []
        for m in markets:
            sig = self.strategy_liquidity.analyze(m)
            if sig:
                signals.append(self._signal_to_dict(sig))
        return signals

    def strategy_funding_rate(
        self,
        markets: List[MarketSignal],
        reference_prices: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """资金费率套利 / Funding Rate Arbitrage.

        Args:
            markets: MarketSignal 列表.
            reference_prices: market_id -> reference_price 字典.

        Returns:
            套利信号字典列表.
        """
        signals = []
        ref = reference_prices or {}
        for m in markets:
            sig = self.strategy_funding.analyze(m, ref.get(m.market_id))
            if sig:
                signals.append(self._signal_to_dict(sig))
        return signals

    def strategy_momentum(
        self,
        markets: List[MarketSignal],
    ) -> List[Dict]:
        """趋势跟随 / Momentum (Trend Following).

        Args:
            markets: MarketSignal 列表.

        Returns:
            套利信号字典列表.
        """
        signals = []
        for m in markets:
            sig = self.strategy_momentum.analyze(m)
            if sig:
                signals.append(self._signal_to_dict(sig))
        return signals

    def strategy_mean_reversion(
        self,
        markets: List[MarketSignal],
    ) -> List[Dict]:
        """均值回归 / Mean Reversion.

        Args:
            markets: MarketSignal 列表.

        Returns:
            套利信号字典列表.
        """
        signals = []
        for m in markets:
            sig = self.strategy_mean_reversion.analyze(m)
            if sig:
                signals.append(self._signal_to_dict(sig))
        return signals

    # -----------------------------------------------------------------------
    # Signal Ranking
    # -----------------------------------------------------------------------

    def rank_signals(self, all_signals: List[Dict]) -> List[Dict]:
        """对所有策略信号进行综合排名 / Rank all strategy signals.

        综合评分 = 利润(30%) + 置信度(25%) + 紧急度(15%)
                + 策略优先级(20%) + 风险回报比(10%)

        Args:
            all_signals: 所有策略生成的信号.

        Returns:
            排序后的信号列表.
        """
        scored = []
        for sig in all_signals:
            score = self._composite_score(sig)
            scored.append((score, sig))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [sig for _, sig in scored]

    def _composite_score(self, signal: Dict) -> float:
        """计算信号综合评分."""
        profit_pct = signal.get("expected_profit_pct", 0)
        profit_score = min(profit_pct / 10, 1.0)
        confidence = signal.get("confidence", 0.5)
        urgency = signal.get("urgency", "MEDIUM")
        strategy = signal.get("strategy", "")
        entry = signal.get("entry_price", 1)
        target = signal.get("target_price", 1)

        urgency_score = self.URGENCY_WEIGHTS.get(urgency, 1.0) / 3.0
        strategy_score = self.STRATEGY_PRIORITY.get(strategy, 0.5)
        rr = abs(target - entry) / entry if entry > 0 else 0
        rr_score = min(rr * 5, 1.0)

        composite = (
            profit_score * 0.30
            + confidence * 0.25
            + urgency_score * 0.15
            + strategy_score * 0.20
            + rr_score * 0.10
        )
        return composite

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    async def execute_top_signals(
        self,
        signals: List[Dict],
        n: int = 3,
    ) -> List[Dict]:
        """执行排名最高的 N 个信号 / Execute top N ranked signals.

        Args:
            signals: 排序后的信号列表.
            n: 执行数量上限.

        Returns:
            执行结果列表.
        """
        results = []
        for sig in signals[:n]:
            try:
                result = await self._execute_signal(sig)
                results.append(result)
            except Exception as e:
                results.append({"status": "error", "message": str(e), "signal": sig})
        return results

    async def _execute_signal(self, signal: Dict) -> Dict:
        """执行单个信号."""
        direction = signal.get("direction", "")
        market_id = signal.get("market_id", "")
        entry_price = signal.get("entry_price", 0)
        amount_usd = min(self.max_position, signal.get("expected_profit_usd", 100) * 10)

        if "BUY_ALL" in direction or "SELL_ALL" in direction:
            # 双向订单
            yes_order = await self.client.place_market_order(
                market_id + "_YES", "BUY", amount_usd / 2
            )
            no_order = await self.client.place_market_order(
                market_id + "_NO", "BUY", amount_usd / 2
            )
            return {
                "status": "executed",
                "signal": signal,
                "yes_order": yes_order,
                "no_order": no_order,
            }
        elif "YES" in direction:
            order = await self.client.place_market_order(
                market_id + "_YES", "BUY", amount_usd
            )
            return {"status": "executed", "signal": signal, "order": order}
        elif "NO" in direction:
            order = await self.client.place_market_order(
                market_id + "_NO", "BUY", amount_usd
            )
            return {"status": "executed", "signal": signal, "order": order}
        else:
            return {"status": "skipped", "reason": "unknown direction", "signal": signal}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _signal_to_dict(sig: ArbitrageSignal) -> Dict:
        """ArbitrageSignal -> dict."""
        return {
            "strategy": sig.strategy,
            "market_id": sig.market_id,
            "question": getattr(sig, "question", ""),
            "direction": sig.direction,
            "entry_price": sig.entry_price,
            "target_price": sig.target_price,
            "expected_profit_pct": sig.expected_profit_pct,
            "expected_profit_usd": sig.expected_profit_usd,
            "confidence": sig.confidence,
            "urgency": sig.urgency,
            "token_ids": sig.token_ids,
            "metadata": sig.metadata,
            "timestamp": sig.timestamp,
            "risk_reward_ratio": sig.risk_reward_ratio,
        }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "MarketSignal",
    "ArbitrageSignal",
    "PriceCandle",
    "PolymarketArbitrageur",
    "ProbabilityDeviationStrategy",
    "LiquidityProvisionStrategy",
    "FundingRateStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
]
