"""
Crypto Absorption Factor
========================
Absorbed from finclaw (agents/signal_engine.py + factors/crypto/crypto_absorption.py)

High volume with small price change — orders being absorbed at support/resistance.
Logic: When volume spikes (>2x average) but price barely moves, large orders are
being absorbed by the market. This is a classic market microstructure signal:
- Absorption after downtrend = support being built = bullish
- Absorption after uptrend = resistance being hit = bearish

Output: float in [0, 1], where >0.5 = bullish absorption, <0.5 = bearish absorption
"""

from typing import Optional
import numpy as np


def compute_absorption(
    closes: list[float] | np.ndarray,
    volumes: list[float] | np.ndarray,
    idx: int,
    lookback: int = 24,
    vol_threshold: float = 2.0,
    move_threshold: float = 0.003,
) -> float:
    """
    Calculate absorption score for a single point in time.

    Parameters
    ----------
    closes : price series (list or np.ndarray)
    volumes : volume series (list or np.ndarray)
    idx : current index
    lookback : periods for average volume calculation (default 24)
    vol_threshold : volume must be > this * avg to signal absorption (default 2.0)
    move_threshold : price move must be < this fraction for absorption (default 0.3%)

    Returns
    -------
    float in [0, 1]: absorption score
        0.5 = neutral (no absorption)
        > 0.5 = bullish (absorption during downtrend = support)
        < 0.5 = bearish (absorption during uptrend = resistance)
    """
    if idx < lookback:
        return 0.5

    avg_vol = np.mean(volumes[max(0, idx - lookback):idx])
    if avg_vol <= 0:
        return 0.5

    vol_ratio = volumes[idx] / avg_vol
    if closes[idx - 1] <= 0:
        return 0.5

    pct_move = abs(closes[idx] - closes[idx - 1]) / closes[idx - 1]

    # High volume but small move = absorption
    if vol_ratio < vol_threshold or pct_move > move_threshold:
        return 0.5

    # Check trend direction leading in
    lookback_trend = min(4, idx)
    if closes[idx - lookback_trend] > 0:
        trend = (closes[idx] - closes[idx - lookback_trend]) / closes[idx - lookback_trend]
    else:
        trend = 0.0

    absorption_strength = min(vol_ratio / 6.0, 1.0)

    # Absorption after downtrend = bullish (support being built)
    # Absorption after uptrend = bearish (resistance being hit)
    if trend < -0.005:
        score = 0.5 + absorption_strength * 0.4
    elif trend > 0.005:
        score = 0.5 - absorption_strength * 0.4
    else:
        score = 0.5

    return float(max(0.0, min(1.0, score)))


def compute_absorption_series(
    closes: list[float] | np.ndarray,
    volumes: list[float] | np.ndarray,
    lookback: int = 24,
    vol_threshold: float = 2.0,
    move_threshold: float = 0.003,
) -> np.ndarray:
    """
    Compute absorption score for an entire series.

    Returns
    -------
    np.ndarray of absorption scores (one per bar)
    """
    closes = np.asarray(closes)
    volumes = np.asarray(volumes)
    n = len(closes)
    scores = np.full(n, 0.5, dtype=np.float64)

    for i in range(lookback, n):
        scores[i] = compute_absorption(
            closes, volumes, i,
            lookback=lookback,
            vol_threshold=vol_threshold,
            move_threshold=move_threshold,
        )
    return scores


class CryptoAbsorption:
    """
    Crypto absorption factor — high volume + small price move = institutional absorption.

    Output: 0-1 score. >0.5 = bullish (support being built),
    <0.5 = bearish (resistance being hit).
    """

    def __init__(
        self,
        lookback: int = 24,
        vol_threshold: float = 2.0,
        move_threshold: float = 0.003,
    ):
        self.lookback = lookback
        self.vol_threshold = vol_threshold
        self.move_threshold = move_threshold

    def evaluate(
        self,
        closes: list[float] | np.ndarray,
        volumes: list[float] | np.ndarray,
    ) -> np.ndarray:
        """Evaluate on full series."""
        return compute_absorption_series(
            closes, volumes,
            lookback=self.lookback,
            vol_threshold=self.vol_threshold,
            move_threshold=self.move_threshold,
        )

    def compute(
        self,
        closes: list[float] | np.ndarray,
        volumes: list[float] | np.ndarray,
        idx: int,
    ) -> float:
        """Compute for single index."""
        return compute_absorption(
            closes, volumes, idx,
            lookback=self.lookback,
            vol_threshold=self.vol_threshold,
            move_threshold=self.move_threshold,
        )
