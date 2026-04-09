"""Technical indicator signal generators.

Provides signal generators based on technical indicators:
- RSIGenerator      — RSI overbought/oversold
- MACDGenerator     — MACD line crossover
- BollingerGenerator — Bollinger Band breakouts
- StochasticGenerator — Stochastic oscillator
- ATRGenerator      — ATR-based trend signals
- VolumeGenerator   — Volume spike signals
- MultiGenerator    — Combines multiple generators with weights

Each generator produces a stream of Signal objects from OHLCV DataFrames.

Usage
-----
```python
import pandas as pd
from quant_trading.signal.generators import RSIGenerator, MACDGenerator

df = pd.read_csv("BTC_USDT_1h.csv")
rsi_gen = RSIGenerator(period=14, oversold=30, overbought=70)
signals = rsi_gen.generate(df)
```
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Callable

import numpy as np
import pandas as pd

from quant_trading.signal.types import Signal, SignalType, SignalDirection


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SignalGenerator(ABC):
    """Abstract base for all technical indicator signal generators."""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> List[Signal]:
        """Generate signals from OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns: open, high, low, close, volume
            (or close + volume minimum).

        Returns
        -------
        List[Signal]
            Chronological list of Signal objects.
        """
        ...

    def _require_cols(self, df: pd.DataFrame, required: List[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")


# ---------------------------------------------------------------------------
# Individual generators
# ---------------------------------------------------------------------------

@dataclass
class RSIGenerator(SignalGenerator):
    """RSI-based signal generator.

    BUY  when RSI crosses below oversold threshold (oversold bounce)
    SELL when RSI crosses above overbought threshold (overbought reversal)
    """
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    symbol: str = "UNKNOWN"
    use_sma: bool = False  # Use SMA vs EMA for smoothing

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        self._require_cols(df, ["close"])
        signals = []
        close = df["close"].values

        # RSI calculation
        deltas = np.diff(close, prepend=close[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = self._rolling_mean(gains, self.period)
        avg_loss = self._rolling_mean(losses, self.period)
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)

        for i in range(1, len(rsi)):
            if np.isnan(rsi[i]) or np.isnan(rsi[i - 1]):
                continue

            # BUY: RSI crosses up through oversold
            if rsi[i - 1] <= self.oversold < rsi[i]:
                strength = min(abs(rsi[i] - self.oversold) / 30.0, 1.0)
                signals.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=self.symbol,
                        timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                        price=float(df.iloc[i]["close"]),
                        strength=float(strength),
                        reason=f"RSI oversold bounce: {rsi[i]:.1f}",
                        metadata={"rsi": float(rsi[i]), "period": self.period},
                    )
                )

            # SELL: RSI crosses down through overbought
            elif rsi[i - 1] >= self.overbought > rsi[i]:
                strength = min(abs(self.overbought - rsi[i]) / 30.0, 1.0)
                signals.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=self.symbol,
                        timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                        price=float(df.iloc[i]["close"]),
                        strength=float(strength),
                        reason=f"RSI overbought reversal: {rsi[i]:.1f}",
                        metadata={"rsi": float(rsi[i]), "period": self.period},
                    )
                )

        return signals

    def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        result = np.empty(len(arr))
        result[:] = np.nan
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1 : i + 1])
        return result


@dataclass
class MACDGenerator(SignalGenerator):
    """MACD signal generator.

    BUY  when MACD line crosses above signal line (bullish)
    SELL when MACD line crosses below signal line (bearish)
    """
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    symbol: str = "UNKNOWN"

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        self._require_cols(df, ["close"])
        signals = []
        close = df["close"].values

        # EMA calculation
        ema_fast = self._ema(close, self.fast_period)
        ema_slow = self._ema(close, self.slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, self.signal_period)

        for i in range(1, len(macd_line)):
            if np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
                continue
            if np.isnan(macd_line[i - 1]) or np.isnan(signal_line[i - 1]):
                continue

            # BUY: MACD crosses above signal
            if macd_line[i - 1] <= signal_line[i - 1] and macd_line[i] > signal_line[i]:
                macd_hist = macd_line[i] - signal_line[i]
                strength = min(abs(macd_hist) / 50.0, 1.0)
                signals.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=self.symbol,
                        timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                        price=float(df.iloc[i]["close"]),
                        strength=float(strength),
                        reason=f"MACD bullish cross: macd={macd_line[i]:.4f}",
                        metadata={"macd": float(macd_line[i]), "signal": float(signal_line[i])},
                    )
                )

            # SELL: MACD crosses below signal
            elif macd_line[i - 1] >= signal_line[i - 1] and macd_line[i] < signal_line[i]:
                macd_hist = macd_line[i - 1] - signal_line[i - 1]
                strength = min(abs(macd_hist) / 50.0, 1.0)
                signals.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=self.symbol,
                        timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                        price=float(df.iloc[i]["close"]),
                        strength=float(strength),
                        reason=f"MACD bearish cross: macd={macd_line[i]:.4f}",
                        metadata={"macd": float(macd_line[i]), "signal": float(signal_line[i])},
                    )
                )

        return signals

    def _ema(self, arr: np.ndarray, period: int) -> np.ndarray:
        result = np.empty(len(arr))
        result[:] = np.nan
        if period < 1:
            return result
        alpha = 2.0 / (period + 1)
        result[period - 1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            if not np.isnan(result[i - 1]):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result


@dataclass
class BollingerGenerator(SignalGenerator):
    """Bollinger Bands signal generator.

    BUY  when price crosses below lower band and back above
    SELL when price crosses above upper band and back below
    """
    period: int = 20
    num_std: float = 2.0
    symbol: str = "UNKNOWN"

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        self._require_cols(df, ["close"])
        signals = []
        close = df["close"].values

        mid = self._rolling_mean(close, self.period)
        std = self._rolling_std(close, self.period)
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        for i in range(self.period, len(close)):
            if np.isnan(upper[i]) or np.isnan(lower[i]):
                continue

            prev = i - 1
            price = close[i]
            prev_price = close[prev]

            # BUY: Price bounces from lower band
            if (
                prev_price <= lower[prev] < price
                or (price < lower[i] and prev_price >= lower[prev])
            ):
                if price > lower[i]:  # Recovered
                    z_score = (price - lower[i]) / (std[i] + 1e-10)
                    strength = min(abs(z_score) / 2.0, 1.0)
                    signals.append(
                        Signal(
                            type=SignalType.BUY,
                            symbol=self.symbol,
                            timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                            price=float(price),
                            strength=float(strength),
                            reason=f"Bollinger lower band bounce: price={price:.2f}",
                            metadata={"upper": float(upper[i]), "lower": float(lower[i]), "mid": float(mid[i])},
                        )
                    )

            # SELL: Price bounces from upper band
            elif (
                prev_price >= upper[prev] > price
                or (price > upper[i] and prev_price <= upper[prev])
            ):
                if price < upper[i]:  # Recovered
                    z_score = (price - upper[i]) / (std[i] + 1e-10)
                    strength = min(abs(z_score) / 2.0, 1.0)
                    signals.append(
                        Signal(
                            type=SignalType.SELL,
                            symbol=self.symbol,
                            timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                            price=float(price),
                            strength=float(strength),
                            reason=f"Bollinger upper band bounce: price={price:.2f}",
                            metadata={"upper": float(upper[i]), "lower": float(lower[i]), "mid": float(mid[i])},
                        )
                    )

        return signals

    def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        result = np.empty(len(arr))
        result[:] = np.nan
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1 : i + 1])
        return result

    def _rolling_std(self, arr: np.ndarray, window: int) -> np.ndarray:
        result = np.empty(len(arr))
        result[:] = np.nan
        for i in range(window - 1, len(arr)):
            result[i] = np.std(arr[i - window + 1 : i + 1])
        return result


@dataclass
class VolumeGenerator(SignalGenerator):
    """Volume spike signal generator.

    BUY  on volume spike + price increase
    SELL on volume spike + price decrease
    """
    period: int = 20
    spike_threshold: float = 2.0  # multiplier of avg volume
    symbol: str = "UNKNOWN"

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        self._require_cols(df, ["close", "volume"])
        signals = []
        close = df["close"].values
        volume = df["volume"].values

        avg_vol = self._rolling_mean(volume, self.period)

        for i in range(self.period, len(close)):
            if np.isnan(avg_vol[i]):
                continue

            vol_ratio = volume[i] / (avg_vol[i] + 1e-10)
            price_chg = (close[i] - close[i - 1]) / (close[i - 1] + 1e-10) if i > 0 else 0

            if vol_ratio > self.spike_threshold:
                strength = min(vol_ratio / self.spike_threshold, 1.0)

                if price_chg > 0.01:
                    signals.append(
                        Signal(
                            type=SignalType.BUY,
                            symbol=self.symbol,
                            timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                            price=float(close[i]),
                            strength=float(strength),
                            reason=f"Volume spike BUY: vol_ratio={vol_ratio:.1f}",
                            metadata={"volume": float(volume[i]), "avg_vol": float(avg_vol[i]), "vol_ratio": float(vol_ratio)},
                        )
                    )
                elif price_chg < -0.01:
                    signals.append(
                        Signal(
                            type=SignalType.SELL,
                            symbol=self.symbol,
                            timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                            price=float(close[i]),
                            strength=float(strength),
                            reason=f"Volume spike SELL: vol_ratio={vol_ratio:.1f}",
                            metadata={"volume": float(volume[i]), "avg_vol": float(avg_vol[i]), "vol_ratio": float(vol_ratio)},
                        )
                    )

        return signals

    def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        result = np.empty(len(arr))
        result[:] = np.nan
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1 : i + 1])
        return result


@dataclass
class ATRGenerator(SignalGenerator):
    """ATR-based trend signal generator.

    BUY  when price rises with ATR confirming trend
    SELL when price falls below ATR trailing stop
    """
    period: int = 14
    multiplier: float = 2.0
    symbol: str = "UNKNOWN"

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        self._require_cols(df, ["high", "low", "close"])
        signals = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.empty(len(close))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        atr = self._rolling_mean(tr, self.period)

        trailing_buy_stop = np.nan
        trailing_short_stop = np.nan

        for i in range(self.period, len(close)):
            if np.isnan(atr[i]):
                continue

            price = close[i]
            a = atr[i] * self.multiplier

            # Update trailing stops
            if np.isnan(trailing_buy_stop) or price > trailing_buy_stop:
                trailing_buy_stop = price - a
            if np.isnan(trailing_short_stop) or price < trailing_short_stop:
                trailing_short_stop = price + a

            # BUY signal: price crosses above trailing buy stop
            if not np.isnan(trailing_buy_stop) and price > trailing_buy_stop:
                signals.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=self.symbol,
                        timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                        price=float(price),
                        stop_loss=float(trailing_buy_stop),
                        reason=f"ATR trend BUY: atr={atr[i]:.4f}",
                        metadata={"atr": float(atr[i]), "stop": float(trailing_buy_stop)},
                    )
                )
                trailing_buy_stop = price - a

            # SELL signal: price crosses below trailing short stop
            elif not np.isnan(trailing_short_stop) and price < trailing_short_stop:
                signals.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=self.symbol,
                        timestamp=int(df.iloc[i]["timestamp"]) if "timestamp" in df.columns else i,
                        price=float(price),
                        stop_loss=float(trailing_short_stop),
                        reason=f"ATR trend SELL: atr={atr[i]:.4f}",
                        metadata={"atr": float(atr[i]), "stop": float(trailing_short_stop)},
                    )
                )
                trailing_short_stop = price + a

        return signals

    def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        result = np.empty(len(arr))
        result[:] = np.nan
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1 : i + 1])
        return result


# ---------------------------------------------------------------------------
# Multi-generator (weighted ensemble)
# ---------------------------------------------------------------------------

@dataclass
class MultiGenerator(SignalGenerator):
    """Combines multiple generators with weighted voting.

    Usage:
        gen = MultiGenerator([
            (RSIGenerator(14), 0.4),
            (MACDGenerator(), 0.3),
            (BollingerGenerator(), 0.3),
        ], threshold=0.5)
        signals = gen.generate(df)
    """
    generators: List[tuple] = field(default_factory=list)
    # Each item: (SignalGenerator instance, weight float)

    threshold: float = 0.5
    require_all: bool = False  # If True, require agreement across all generators

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        all_signals: List[List[Signal]] = []
        weights: List[float] = []

        for gen, weight in self.generators:
            sigs = gen.generate(df)
            all_signals.append(sigs)
            weights.append(weight)

        # Collect unique timestamps from all signals
        timestamps = set()
        for sigs in all_signals:
            for s in sigs:
                timestamps.add(s.timestamp)

        result: List[Signal] = []
        for ts in sorted(timestamps):
            buy_score = 0.0
            sell_score = 0.0

            for sigs, w in zip(all_signals, weights):
                for s in sigs:
                    if s.timestamp == ts:
                        if s.type == SignalType.BUY:
                            buy_score += w * s.strength
                        elif s.type == SignalType.SELL:
                            sell_score += w * s.strength

            total = buy_score + sell_score
            if total < 1e-10:
                continue

            buy_norm = buy_score / total
            sell_norm = sell_score / total

            if self.require_all:
                buy_ok = all(
                    any(s.timestamp == ts and s.type == SignalType.BUY for s in sigs)
                    for sigs in all_signals
                )
                sell_ok = all(
                    any(s.timestamp == ts and s.type == SignalType.SELL for s in sigs)
                    for sigs in all_signals
                )
            else:
                buy_ok = buy_norm >= self.threshold
                sell_ok = sell_norm >= self.threshold

            # Find the corresponding signal info (first matching signal)
            src_signal = None
            for sigs in all_signals:
                for s in sigs:
                    if s.timestamp == ts:
                        src_signal = s
                        break

            if buy_ok and (not sell_ok or buy_norm > sell_norm):
                result.append(
                    Signal(
                        type=SignalType.BUY,
                        symbol=src_signal.symbol if src_signal else "UNKNOWN",
                        timestamp=ts,
                        price=src_signal.price if src_signal else 0.0,
                        strength=float(buy_norm),
                        reason=f"Multi-gen BUY: {buy_score:.2f}/{total:.2f}={buy_norm:.2f}",
                        metadata={"buy_score": float(buy_score), "sell_score": float(sell_score)},
                    )
                )
            elif sell_ok and (not buy_ok or sell_norm > buy_norm):
                result.append(
                    Signal(
                        type=SignalType.SELL,
                        symbol=src_signal.symbol if src_signal else "UNKNOWN",
                        timestamp=ts,
                        price=src_signal.price if src_signal else 0.0,
                        strength=float(sell_norm),
                        reason=f"Multi-gen SELL: {sell_score:.2f}/{total:.2f}={sell_norm:.2f}",
                        metadata={"buy_score": float(buy_score), "sell_score": float(sell_score)},
                    )
                )

        return result
