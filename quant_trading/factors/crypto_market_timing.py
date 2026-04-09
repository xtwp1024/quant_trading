"""
Crypto Market Timing Score
=========================
Absorbed from finclaw (factors/crypto/crypto_market_timing_score.py)

Composite score combining:
- Trend: price vs SMA20
- Momentum: 12-bar return
- Volatility regime: recent vs long-term vol
- Volume confirmation

Output: float in [0, 1]. >0.5 = favorable market timing, <0.5 = unfavorable.
"""


def compute_timing_score(
    closes, highs, lows, volumes, idx,
    sma_period: int = 20,
    mom_period: int = 12,
    short_vol_period: int = 12,
    long_vol_period: int = 48,
    vol_avg_period: int = 24,
) -> float:
    """
    Calculate composite market timing score.

    Parameters
    ----------
    closes, highs, lows, volumes : array-like price/volume data
    idx : current index
    sma_period : SMA period for trend (default 20)
    mom_period : momentum lookback (default 12)
    short_vol_period : short vol window (default 12)
    long_vol_period : long vol window (default 48)
    vol_avg_period : volume average period (default 24)

    Returns
    -------
    float in [0, 1]: timing score. >0.5 = favorable, <0.5 = unfavorable
    """
    if idx < 50:
        return 0.5

    score = 0.0

    # 1. Trend: price vs SMA20 (0.25 weight)
    if idx >= sma_period:
        sma = sum(closes[idx - sma_period:idx]) / sma_period
        if sma > 0:
            trend = 1.0 if closes[idx - 1] > sma else 0.0
            score += trend * 0.25

    # 2. Momentum: 12-bar return (0.25 weight)
    if idx >= mom_period + 1 and closes[idx - mom_period - 1] > 0:
        mom_ret = (closes[idx - 1] - closes[idx - mom_period - 1]) / closes[idx - mom_period - 1]
        mom = 0.5 + mom_ret * 10.0
        score += max(0.0, min(1.0, mom)) * 0.25

    # 3. Volatility regime: recent vs long-term vol (0.25 weight)
    def _calc_std(start, end):
        rets = []
        for i in range(start, end):
            if i < 1 or closes[i - 1] <= 0:
                continue
            rets.append((closes[i] - closes[i - 1]) / closes[i - 1])
        if len(rets) < 2:
            return 0.0
        m = sum(rets) / len(rets)
        return (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5

    sv = _calc_std(idx - short_vol_period, idx)
    lv = _calc_std(idx - long_vol_period, idx)
    # Lower recent vol relative to long-term = favorable
    if lv > 0:
        vol_score = 0.5 + (1.0 - sv / lv) * 0.3
        score += max(0.0, min(1.0, vol_score)) * 0.25

    # 4. Volume confirmation (0.25 weight)
    if idx >= vol_avg_period:
        avg_vol = sum(volumes[idx - vol_avg_period:idx]) / vol_avg_period
        if avg_vol > 0 and idx > 0:
            vol_ratio = volumes[idx - 1] / avg_vol
            if closes[idx - 1] > closes[idx - 2]:
                v = min(vol_ratio / 2.0, 1.0)
            else:
                v = max(0.0, 1.0 - vol_ratio / 2.0)
            score += v * 0.25

    return max(0.0, min(1.0, score))


def compute_timing_series(closes, highs, lows, volumes, **kwargs) -> list:
    """Compute timing score for entire series."""
    n = len(closes)
    return [compute_timing_score(closes, highs, lows, volumes, i, **kwargs) for i in range(n)]


class CryptoMarketTiming:
    """
    Crypto market timing composite score.
    Combines trend, momentum, volatility regime, and volume for market timing assessment.
    """

    def __init__(
        self,
        sma_period: int = 20,
        mom_period: int = 12,
        short_vol_period: int = 12,
        long_vol_period: int = 48,
        vol_avg_period: int = 24,
    ):
        self.sma_period = sma_period
        self.mom_period = mom_period
        self.short_vol_period = short_vol_period
        self.long_vol_period = long_vol_period
        self.vol_avg_period = vol_avg_period

    def evaluate(self, closes, highs, lows, volumes) -> list:
        return compute_timing_series(
            closes, highs, lows, volumes,
            sma_period=self.sma_period,
            mom_period=self.mom_period,
            short_vol_period=self.short_vol_period,
            long_vol_period=self.long_vol_period,
            vol_avg_period=self.vol_avg_period,
        )

    def compute(self, closes, highs, lows, volumes, idx) -> float:
        return compute_timing_score(
            closes, highs, lows, volumes, idx,
            sma_period=self.sma_period,
            mom_period=self.mom_period,
            short_vol_period=self.short_vol_period,
            long_vol_period=self.long_vol_period,
            vol_avg_period=self.vol_avg_period,
        )
