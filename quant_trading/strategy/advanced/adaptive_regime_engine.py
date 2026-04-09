"""
Adaptive Multi-Regime Strategy Engine
====================================
Absorbed from finclaw (agents/signal_engine.py)

Adaptive strategy that automatically switches between THREE sub-strategies
based on detected market regime:

1. TREND RIDER (Bull/Strong Bull):
   - Buy breakouts, wide trailing stops, NO take profit
   - Large positions, infrequent trades
   - Academic basis: Turtle Trading (Richard Dennis, 1983)

2. MEAN REVERTER (Ranging/Sideways):
   - Buy oversold, sell overbought, tight stops
   - Small positions, frequent trades
   - Academic basis: Mean Reversion (Poterba & Summers, 1988)

3. CRISIS ALPHA (Bear/Crash):
   - Stay mostly cash, only extreme oversold bounces
   - Very small positions, very tight stops
   - Capital preservation priority

The engine auto-detects regime and switches strategies using:
- EMA alignment (8 > 21 > 50 = bull, 8 < 21 < 50 = bear)
- ADX for trend strength
- Multi-timeframe returns (5, 10, 20, 50-bar)
- Annualized volatility

Regime inertia prevents flip-flopping: requires N consecutive bars
to confirm a regime change (except CRASH which is instant).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from quant_trading.signal import Signal, SignalDirection
from quant_trading.strategy.base import BaseStrategy, StrategyParams
from quant_trading.strategy.context import StrategyContext


class MarketRegime(Enum):
    """Market regime enumeration."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    CRASH = "crash"


@dataclass
class AdaptiveRegimeParams(StrategyParams):
    """Adaptive regime engine parameters."""
    risk_per_trade: float = 0.02       # Risk per trade (fraction of capital)
    max_position_size: float = 0.60     # Maximum position size
    atr_period: int = 14                # ATR period
    donchian_period: int = 20           # Donchian channel period
    regime_inertia_bull: int = 5        # Bars to confirm bull → non-bull
    regime_inertia_bear: int = 3        # Bars to confirm bear → non-bear
    regime_inertia_neutral: int = 2     # Bars to confirm ranging → non-ranging


@dataclass
class RegimeSignalResult:
    """Internal signal result from regime engine."""
    signal: str               # buy/strong_buy/sell/strong_sell/hold
    confidence: float        # 0-1
    regime: MarketRegime
    position_size: float      # fraction of capital
    stop_loss: float          # absolute price
    take_profit: float        # absolute price
    trailing_stop_pct: float  # as fraction (0.05 = 5%)
    reasoning: str


class AdaptiveRegimeEngine(BaseStrategy):
    """
    Adaptive multi-regime signal engine.
    Automatically switches between trend-following, mean-reversion,
    and crisis-alpha based on detected market regime.
    """

    name: str = "adaptive_regime_engine"
    params: AdaptiveRegimeParams

    def __init__(
        self,
        symbol: str,
        params: Optional[AdaptiveRegimeParams] = None,
    ) -> None:
        p = params or AdaptiveRegimeParams()
        super().__init__(symbol, p)
        self._prev_regime: Optional[MarketRegime] = None
        self._regime_hold_count: int = 0
        self._prices: List[float] = []
        self._volumes: List[float] = []

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        if len(data) < 55:
            return []

        closes = data["close"].tolist()
        volumes = data["volume"].tolist() if "volume" in data else [0] * len(data)

        # Append to history
        self._prices = closes
        self._volumes = volumes

        result = self._generate(closes, volumes)
        return [self._to_signal(result, data)]

    def _generate(self, prices: list[float], volumes: list[float]) -> RegimeSignalResult:
        if len(prices) < 55:
            return RegimeSignalResult(
                signal="hold", confidence=0.3,
                regime=MarketRegime.RANGING,
                position_size=0.0, stop_loss=0.0, take_profit=0.0,
                trailing_stop_pct=0.05, reasoning="warmup",
            )

        price = prices[-1]
        atr = self._atr(prices, self.params.atr_period)

        # Step 1: Regime Detection with Inertia
        raw_regime, regime_conf = self._detect_regime(prices)
        regime = self._apply_regime_inertia(raw_regime, regime_conf)

        # Step 2: Route to appropriate sub-strategy
        if regime in (MarketRegime.STRONG_BULL, MarketRegime.BULL):
            return self._trend_rider(prices, volumes, price, atr, regime, regime_conf)
        elif regime in (MarketRegime.BEAR, MarketRegime.STRONG_BEAR, MarketRegime.CRASH):
            return self._crisis_alpha(prices, volumes, price, atr, regime, regime_conf)
        elif regime == MarketRegime.VOLATILE:
            ret_20 = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0
            if ret_20 > 0.03:
                return self._trend_rider(prices, volumes, price, atr, regime, regime_conf)
            elif ret_20 < -0.03:
                return self._crisis_alpha(prices, volumes, price, atr, regime, regime_conf)
            else:
                return self._mean_reverter(prices, volumes, price, atr, regime, regime_conf)
        else:  # RANGING
            return self._mean_reverter(prices, volumes, price, atr, regime, regime_conf)

    def calculate_position_size(self, signal: Signal, context: StrategyContext) -> float:
        return getattr(signal, 'position_size', self.params.risk_per_trade)

    # ── Regime Detection ──────────────────────────────────────────────

    def _detect_regime(self, prices: list[float]) -> tuple[MarketRegime, float]:
        """Detect market regime using multiple timeframes."""
        n = len(prices)

        ret_5 = (prices[-1] / prices[-5] - 1) if n >= 5 else 0
        ret_10 = (prices[-1] / prices[-10] - 1) if n >= 10 else 0
        ret_20 = (prices[-1] / prices[-20] - 1) if n >= 20 else 0
        ret_50 = (prices[-1] / prices[-50] - 1) if n >= 50 else 0

        # Volatility
        rets = [(prices[i] / prices[i - 1] - 1) for i in range(max(1, n - 20), n)]
        vol = self._stdev(rets) if len(rets) >= 2 else 0.02
        annualized_vol = vol * math.sqrt(252)

        # EMA alignment
        ema8 = self._ema_val(prices, 8)
        ema21 = self._ema_val(prices, 21)
        ema50 = self._ema_val(prices, 50) if n >= 50 else ema21
        bull_aligned = ema8 > ema21 > ema50
        bear_aligned = ema8 < ema21 < ema50

        # ADX
        adx = self._adx(prices, 14)

        # Decision tree
        if ret_5 < -0.07 and vol > 0.025:
            return MarketRegime.CRASH, 0.85
        if ret_20 < -0.10 and ret_50 < -0.15 and bear_aligned:
            return MarketRegime.STRONG_BEAR, 0.80
        if ret_20 < -0.05 and ret_50 < -0.05 and bear_aligned:
            return MarketRegime.BEAR, 0.75
        if ret_20 > 0.08 and ret_50 > 0.15 and bull_aligned and adx > 25:
            return MarketRegime.STRONG_BULL, 0.85
        if ret_20 > 0.03 and bull_aligned:
            return MarketRegime.BULL, 0.70
        if ret_10 > 0.05 and ret_20 > 0.02 and ema8 > ema21:
            return MarketRegime.BULL, 0.60
        if annualized_vol > 0.40:
            return MarketRegime.VOLATILE, 0.65
        return MarketRegime.RANGING, 0.50

    def _apply_regime_inertia(self, new_regime: MarketRegime, new_conf: float) -> MarketRegime:
        """Regime inertia: prevent flip-flopping."""
        if self._prev_regime is None:
            self._prev_regime = new_regime
            self._regime_hold_count = 0
            return new_regime

        if new_regime == self._prev_regime:
            self._regime_hold_count = 0
            return new_regime

        self._regime_hold_count += 1

        # Allow instant CRASH transition
        if new_regime == MarketRegime.CRASH:
            self._prev_regime = new_regime
            self._regime_hold_count = 0
            return new_regime

        # Required bars to confirm transition
        if self._prev_regime in (MarketRegime.STRONG_BULL, MarketRegime.BULL):
            required = self.params.regime_inertia_bull
        elif self._prev_regime in (MarketRegime.STRONG_BEAR, MarketRegime.BEAR):
            required = self.params.regime_inertia_bear
        else:
            required = self.params.regime_inertia_neutral

        if self._regime_hold_count >= required:
            self._prev_regime = new_regime
            self._regime_hold_count = 0

        return self._prev_regime

    # ── Sub-Strategies ────────────────────────────────────────────────

    def _trend_rider(self, prices, volumes, price, atr, regime, regime_conf) -> RegimeSignalResult:
        """Trend-following for bull markets."""
        # EMA alignment score
        ema8 = self._ema_val(prices, 8)
        ema21 = self._ema_val(prices, 21)
        ema50 = self._ema_val(prices, 50) if len(prices) >= 50 else ema21

        ema_score = 0.0
        if ema21 > 0:
            ema_score += self._clamp((ema8 / ema21 - 1) * 30, -1, 1) * 0.4
            ema_score += self._clamp((ema21 / ema50 - 1) * 20, -1, 1) * 0.3
            ema_score += self._clamp((price / ema21 - 1) * 15, -1, 1) * 0.3

        aligned = ema8 > ema21 > ema50

        # Pullback detection
        pullback_score = 0.0
        if aligned and ema21 > 0:
            dist = (price - ema21) / ema21
            if -0.03 < dist < 0.01:
                pullback_score = 0.8
            elif dist < -0.05:
                pullback_score = -0.3
            elif dist > 0.08:
                pullback_score = -0.2
            else:
                pullback_score = 0.3

        donchian_score = self._donchian_signal(prices)
        macd_score = self._macd_momentum(prices)
        vol_score = self._volume_confirm(prices, volumes)

        score = (
            ema_score * 0.30 +
            pullback_score * 0.25 +
            donchian_score * 0.25 +
            macd_score * 0.10 +
            vol_score * 0.10
        )

        buy_threshold = 0.10 if regime == MarketRegime.STRONG_BULL else 0.15
        sell_threshold = -0.30 if regime == MarketRegime.STRONG_BULL else -0.20

        if score > buy_threshold:
            signal = "strong_buy" if score > 0.30 else "buy"
        elif score < sell_threshold:
            signal = "sell"
        else:
            signal = "hold"

        confidence = min(0.50 + abs(score) * 1.5, 0.95)
        pos_size = self._trend_position_size(price, atr, confidence, regime)
        stop_loss, take_profit, trailing_pct = self._trend_stops(prices, price, atr, regime)

        return RegimeSignalResult(
            signal=signal, confidence=confidence, regime=regime,
            position_size=pos_size, stop_loss=stop_loss,
            take_profit=take_profit, trailing_stop_pct=trailing_pct,
            reasoning=f"TREND_RIDER {regime.value} score={score:+.3f}",
        )

    def _mean_reverter(self, prices, volumes, price, atr, regime, regime_conf) -> RegimeSignalResult:
        """Mean reversion for sideways markets."""
        sma20 = sum(prices[-20:]) / 20
        std20 = self._stdev(prices[-20:])

        bollinger_score = 0.0
        if std20 > 0:
            z = (price - sma20) / std20
            bollinger_score = self._clamp(-z / 2.0, -1, 1)

        rsi_val = self._rsi(prices, 14)
        rsi_score = 0.0
        if rsi_val is not None:
            if rsi_val < 25: rsi_score = 0.9
            elif rsi_val < 35: rsi_score = 0.5
            elif rsi_val < 45: rsi_score = 0.2
            elif rsi_val > 75: rsi_score = -0.9
            elif rsi_val > 65: rsi_score = -0.5
            elif rsi_val > 55: rsi_score = -0.2

        mean_dist_score = 0.0
        if sma20 > 0:
            mean_dist_score = self._clamp((sma20 - price) / sma20 * 15, -1, 1)

        stoch_score = 0.0
        if len(prices) >= 14:
            h14 = max(prices[-14:]); l14 = min(prices[-14:])
            if h14 != l14:
                k = (price - l14) / (h14 - l14)
                stoch_score = self._clamp((0.5 - k) * 2, -1, 1)

        vol_score = self._volume_confirm(prices, volumes)

        score = (
            bollinger_score * 0.30 +
            rsi_score * 0.30 +
            mean_dist_score * 0.15 +
            stoch_score * 0.15 +
            vol_score * 0.10
        )

        if score > 0.25:
            signal = "strong_buy" if score > 0.45 else "buy"
        elif score < -0.25:
            signal = "strong_sell" if score < -0.45 else "sell"
        else:
            signal = "hold"

        confidence = min(0.40 + abs(score) * 1.2, 0.90)
        pos_size = min(self.params.risk_per_trade / max((atr * 1.5) / price, 0.005), 0.25)
        pos_size *= confidence * 0.8
        pos_size = self._clamp(pos_size, 0.05, 0.25)

        stop_loss = price - atr * 1.5
        stop_loss = max(stop_loss, price * 0.94)
        risk = price - stop_loss
        take_profit = price + risk * 2.0
        trailing_pct = atr * 1.5 / price

        return RegimeSignalResult(
            signal=signal, confidence=confidence, regime=regime,
            position_size=pos_size, stop_loss=stop_loss,
            take_profit=take_profit, trailing_stop_pct=trailing_pct,
            reasoning=f"MEAN_REVERTER {regime.value} score={score:+.3f}",
        )

    def _crisis_alpha(self, prices, volumes, price, atr, regime, regime_conf) -> RegimeSignalResult:
        """Capital preservation for bear markets."""
        rsi_val = self._rsi(prices, 14)
        rsi_score = 0.0
        if rsi_val is not None:
            if rsi_val < 15: rsi_score = 1.0
            elif rsi_val < 20: rsi_score = 0.7
            elif rsi_val < 25: rsi_score = 0.3
            else: rsi_score = -0.5

        # Capitulation volume spike
        vol_spike_score = 0.0
        if volumes and len(volumes) >= 20:
            avg_vol = sum(volumes[-20:]) / 20
            if avg_vol > 0 and len(prices) >= 2:
                vol_ratio = volumes[-1] / avg_vol
                price_drop = prices[-1] / prices[-2] - 1
                if vol_ratio > 2.0 and price_drop < -0.03:
                    vol_spike_score = 0.6

        # Deep drawdown
        drawdown_score = 0.0
        if len(prices) >= 50:
            high_50 = max(prices[-50:])
            drawdown = (price / high_50) - 1
            if drawdown < -0.30: drawdown_score = 0.5
            elif drawdown < -0.20: drawdown_score = 0.3
            else: drawdown_score = -0.3

        # Short-term reversal
        reversal_score = 0.0
        if len(prices) >= 13:
            ret_10 = prices[-4] / prices[-13] - 1
            ret_3 = prices[-1] / prices[-4] - 1
            if ret_10 < -0.08 and ret_3 > 0.02:
                reversal_score = 0.6
            else:
                reversal_score = -0.3

        score = (
            rsi_score * 0.35 +
            vol_spike_score * 0.20 +
            drawdown_score * 0.25 +
            reversal_score * 0.20
        )

        if regime == MarketRegime.CRASH:
            threshold = 0.50
        elif regime == MarketRegime.STRONG_BEAR:
            threshold = 0.40
        else:
            threshold = 0.30

        if score > threshold:
            signal = "buy"
        elif score < -0.20:
            signal = "sell"
        else:
            signal = "hold"

        confidence = min(0.35 + abs(score) * 1.0, 0.80)
        pos_size = self._clamp(self.params.risk_per_trade * 0.5, 0.03, 0.15)

        stop_loss = price - atr * 1.2
        stop_loss = max(stop_loss, price * 0.95)
        take_profit = price + atr * 2.0
        trailing_pct = atr * 1.2 / price

        return RegimeSignalResult(
            signal=signal, confidence=confidence, regime=regime,
            position_size=pos_size, stop_loss=stop_loss,
            take_profit=take_profit, trailing_stop_pct=trailing_pct,
            reasoning=f"CRISIS_ALPHA {regime.value} score={score:+.3f}",
        )

    # ── Shared Factor Methods ────────────────────────────────────────

    def _donchian_signal(self, prices: list[float]) -> float:
        period = self.params.donchian_period
        if len(prices) < period + 1:
            return 0.0
        channel = prices[-(period + 1):-1]
        high = max(channel); low = min(channel)
        mid = (high + low) / 2

        if prices[-1] > high:
            strength = (prices[-1] - high) / max(high - mid, 0.001)
            return self._clamp(0.5 + strength * 0.5, 0, 1.0)
        elif prices[-1] < low:
            strength = (low - prices[-1]) / max(mid - low, 0.001)
            return self._clamp(-0.5 - strength * 0.5, -1.0, 0)
        else:
            rng = max(high - low, 0.001)
            pos = (prices[-1] - low) / rng
            return (pos - 0.5) * 0.3

    def _macd_momentum(self, prices: list[float]) -> float:
        if len(prices) < 26: return 0.0
        ema12 = self._ema_val(prices, 12)
        ema26 = self._ema_val(prices, 26)
        macd = ema12 - ema26
        if prices[-1] > 0:
            return self._clamp(macd / prices[-1] * 100 * 3, -1, 1)
        return 0.0

    def _volume_confirm(self, prices, volumes) -> float:
        if not volumes or len(volumes) < 20:
            return 0.0
        avg_vol = sum(volumes[-20:]) / 20
        if avg_vol == 0: return 0.0
        vol_ratio = volumes[-1] / avg_vol
        price_chg = (prices[-1] / prices[-2] - 1) if len(prices) >= 2 else 0
        if vol_ratio > 1.3 and price_chg > 0:
            return min(vol_ratio / 3, 1.0)
        elif vol_ratio > 1.3 and price_chg < 0:
            return -min(vol_ratio / 3, 1.0)
        return 0.0

    def _trend_position_size(self, price, atr, confidence, regime):
        if atr <= 0 or price <= 0: return 0.1
        risk_pct = (atr * 2.5) / price
        base = self.params.risk_per_trade / max(risk_pct, 0.005)
        size = base * confidence
        if regime == MarketRegime.STRONG_BULL:
            size *= 1.5
        elif regime == MarketRegime.VOLATILE:
            size *= 0.8
        return self._clamp(size, 0.10, self.params.max_position_size)

    def _trend_stops(self, prices, price, atr, regime):
        lookback = min(20, len(prices) - 1)
        swing_low = min(prices[-lookback:]) if lookback > 0 else price * 0.90

        if regime == MarketRegime.STRONG_BULL:
            mult = 3.5
        elif regime == MarketRegime.VOLATILE:
            mult = 4.0
        else:
            mult = 3.0

        atr_stop = price - atr * mult
        swing_stop = swing_low - atr * 0.5
        stop_loss = max(atr_stop, swing_stop)
        stop_loss = max(stop_loss, price * 0.82)
        take_profit = price * 10.0  # Effectively infinite

        if regime == MarketRegime.STRONG_BULL:
            trailing_pct = max(atr * 4.5 / price, 0.16)
        elif regime == MarketRegime.VOLATILE:
            trailing_pct = max(atr * 5.0 / price, 0.18)
        else:
            trailing_pct = max(atr * 3.0 / price, 0.12)

        return stop_loss, take_profit, trailing_pct

    # ── Technical Indicators ─────────────────────────────────────────

    def _ema_val(self, prices: list[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        mult = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for p in prices[period:]:
            ema = p * mult + ema * (1 - mult)
        return ema

    def _rsi(self, prices: list[float], period: int = 14) -> Optional[float]:
        if len(prices) < period + 1: return None
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        recent = deltas[-period:]
        ag = sum(max(d, 0) for d in recent) / period
        al = sum(max(-d, 0) for d in recent) / period
        if al == 0: return 100.0
        return 100 - 100 / (1 + ag / al)

    def _atr(self, prices: list[float], period: int = 14) -> float:
        if len(prices) < 2:
            return prices[-1] * 0.02 if prices else 0.02
        trs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        return sum(trs[-period:]) / min(len(trs), period)

    def _adx(self, prices: list[float], period: int = 14) -> float:
        if len(prices) < period * 2: return 20.0
        plus_dm = []; minus_dm = []; tr_vals = []
        for i in range(1, len(prices)):
            hm = prices[i] - prices[i - 1]; lm = prices[i - 1] - prices[i]
            plus_dm.append(max(hm, 0) if hm > lm else 0)
            minus_dm.append(max(lm, 0) if lm > hm else 0)
            tr_vals.append(abs(prices[i] - prices[i - 1]))

        def _smooth(vals, n):
            if not vals: return [0.0]
            result = [sum(vals[:n]) / n]
            for v in vals[n:]:
                result.append(result[-1] * (n - 1) / n + v / n)
            return result

        sp = _smooth(plus_dm[-period * 2:], period)
        sm = _smooth(minus_dm[-period * 2:], period)
        st = _smooth(tr_vals[-period * 2:], period)
        if not st or st[-1] == 0: return 20.0
        dp = 100 * sp[-1] / st[-1]; dm = 100 * sm[-1] / st[-1]
        diff = abs(dp - dm); total = dp + dm
        if total == 0: return 0.0
        return min(100 * diff / total * 1.3, 100)

    def _stdev(self, values: list[float]) -> float:
        if len(values) < 2: return 0.0
        m = sum(values) / len(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    # ── Signal Conversion ───────────────────────────────────────────

    def _to_signal(self, result: RegimeSignalResult, data: pd.DataFrame) -> Signal:
        direction_map = {
            "strong_buy": SignalDirection.LONG,
            "buy": SignalDirection.LONG,
            "sell": SignalDirection.SHORT,
            "strong_sell": SignalDirection.SHORT,
            "hold": SignalDirection.HOLD,
        }
        direction = direction_map.get(result.signal, SignalDirection.HOLD)

        return Signal(
            symbol=self.symbol,
            direction=direction,
            confidence=result.confidence,
            price=data["close"].iloc[-1],
            stop_loss=result.stop_loss if result.stop_loss > 0 else None,
            take_profit=result.take_profit if result.take_profit > 0 and result.take_profit < data["close"].iloc[-1] * 5 else None,
            trailing_stop_pct=result.trailing_stop_pct if result.trailing_stop_pct > 0 else None,
            metadata={
                "regime": result.regime.value,
                "position_size": result.position_size,
                "reasoning": result.reasoning,
            },
        )
