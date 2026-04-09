"""TA-Lib based indicator computation engine with pandas fallback."""
from __future__ import annotations

from typing import Callable, Mapping, Sequence, Union

import numpy as np
import pandas as pd

try:
    import talib
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False


class IndicatorError(ValueError):
    """Raised when indicator computation fails."""


def _pandas_sma(values: np.ndarray, timeperiod: int = 20) -> np.ndarray:
    return pd.Series(values).rolling(window=timeperiod).mean().values


def _pandas_ema(values: np.ndarray, timeperiod: int = 20) -> np.ndarray:
    return pd.Series(values).ewm(span=timeperiod, adjust=False).mean().values


def _pandas_ma(values: np.ndarray, timeperiod: int = 20) -> np.ndarray:
    return pd.Series(values).rolling(window=timeperiod).mean().values


def _pandas_rsi(values: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    delta = pd.Series(values).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=timeperiod - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=timeperiod - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def _pandas_macd(
    values: np.ndarray,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = pd.Series(values).ewm(span=fastperiod, adjust=False).mean().values
    ema_slow = pd.Series(values).ewm(span=slowperiod, adjust=False).mean().values
    macd = ema_fast - ema_slow
    signal = pd.Series(macd).ewm(span=signalperiod, adjust=False).mean().values
    histogram = macd - signal
    return macd, signal, histogram


def _pandas_ma_type(
    values: np.ndarray, timeperiod: int = 20, matype: int = 0
) -> np.ndarray:
    if matype == 0:  # SMA
        return _pandas_sma(values, timeperiod)
    elif matype in (1, 2):  # EMA
        return _pandas_ema(values, timeperiod)
    else:  # default to SMA
        return _pandas_sma(values, timeperiod)


def _default_registry() -> dict[str, Callable[..., np.ndarray]]:
    if _HAS_TALIB:
        return {
            "sma": talib.SMA,
            "ema": talib.EMA,
            "ma": talib.MA,
            "rsi": talib.RSI,
        }
    return {
        "sma": _pandas_sma,
        "ema": _pandas_ema,
        "ma": _pandas_ma,
        "rsi": _pandas_rsi,
    }


def _to_numpy(values: Union[Sequence[float], pd.Series]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise IndicatorError("Indicator input must be one-dimensional")
    return array


class IndicatorEngine:
    """Indicator computation wrapper with a supported-indicator registry."""

    def __init__(
        self, registry: Mapping[str, Callable[..., np.ndarray]] | None = None
    ) -> None:
        source = registry or _default_registry()
        self._registry = {name.lower(): func for name, func in source.items()}
        self._has_talib = _HAS_TALIB

    def supported(self) -> list[str]:
        return sorted(self._registry.keys())

    def compute(
        self, name: str, series: Union[Sequence[float], pd.Series], **kwargs: object
    ) -> pd.Series:
        key = name.lower()
        if key not in self._registry:
            raise IndicatorError(f"Unsupported indicator: {name}")

        values = _to_numpy(series)
        output = self._registry[key](values, **kwargs)
        if isinstance(output, tuple):
            output = output[0]

        if len(output) != len(values):
            raise IndicatorError("Indicator output length does not match input length")

        if isinstance(series, pd.Series):
            return pd.Series(output, index=series.index, name=key)
        return pd.Series(output, name=key)

    def compute_ma(
        self,
        series: Union[Sequence[float], pd.Series],
        *,
        timeperiod: int = 20,
        matype: int = 0,
    ) -> pd.Series:
        values = _to_numpy(series)
        if self._has_talib:
            output = talib.MA(values, timeperiod=timeperiod, matype=matype)
        else:
            output = _pandas_ma_type(values, timeperiod, matype)
        if len(output) != len(values):
            raise IndicatorError("Indicator output length does not match input length")
        if isinstance(series, pd.Series):
            return pd.Series(output, index=series.index, name="ma")
        return pd.Series(output, name="ma")

    def compute_macd(
        self,
        series: Union[Sequence[float], pd.Series],
        *,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> pd.DataFrame:
        values = _to_numpy(series)
        if self._has_talib:
            macd, signal, hist = talib.MACD(
                values,
                fastperiod=fastperiod,
                slowperiod=slowperiod,
                signalperiod=signalperiod,
            )
        else:
            macd, signal, hist = _pandas_macd(
                values, fastperiod, slowperiod, signalperiod
            )
        if len(macd) != len(values):
            raise IndicatorError("Indicator output length does not match input length")
        frame = pd.DataFrame(
            {
                "macd": macd,
                "signal": signal,
                "histogram": hist,
            }
        )
        if isinstance(series, pd.Series):
            frame.index = series.index
        return frame
