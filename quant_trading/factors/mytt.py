"""
MyTT — Pure NumPy/Pandas Technical Indicators (40+ formulas).

Based on the open-source MyTT library (Python-Financial-Technical-Indicators-Pandas).
Zero external dependencies beyond NumPy and pandas; no Ta-Lib required.

All indicators are fully vectorised (no Python for-loops) for high performance.

Level 0 — Core helpers
    RD, RET, ABS, MAX, MIN, MA, REF, DIFF, STD, IF, SUM, HHV, LLV,
    EMA, SMA, AVEDEV, SLOPE

Level 1 — Composite helpers (built on Level 0)
    COUNT, EVERY, LAST, EXIST, BARSLAST, FORCAST, CROSS

Level 2 — Technical Indicators (built on Levels 0-1)
    MACD, KDJ, RSI, WR, BIAS, BOLL, PSY, CCI, ATR, BBI, DMI,
    TURTLES, KTN, TRIX, VR, EMV, DPO, BRAR, DMA, MTM, MASS,
    ROC, EXPMA, OBV, MFI, ASI

Example usage
-------------
    import numpy as np
    from quant_trading.factors.mytt import MACD, RSI, BOLL

    close = np.random.randn(1000).cumsum() + 100
    dif, dea, macd = MACD(close)
    k, d, j = KDJ(close, high, low)
    upper, mid, lower = BOLL(close)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

__all__ = [
    # Level 0 — core
    "RD", "RET", "ABS", "MAX", "MIN",
    "MA", "REF", "DIFF", "STD", "IF", "SUM",
    "HHV", "LLV", "EMA", "SMA", "AVEDEV", "SLOPE",
    # Level 1 — composite
    "COUNT", "EVERY", "LAST", "EXIST", "BARSLAST", "FORCAST", "CROSS",
    # Level 2 — indicators
    "MACD", "KDJ", "RSI", "WR", "BIAS", "BOLL", "PSY", "CCI",
    "ATR", "BBI", "DMI", "TURTLES", "KTN", "TRIX", "VR", "EMV",
    "DPO", "BRAR", "DMA", "MTM", "MASS", "ROC", "EXPMA", "OBV",
    "MFI", "ASI",
]


# ----------------------------------------------------------------------
# Level 0 — core tools
# ----------------------------------------------------------------------


def RD(N: np.ndarray, D: int = 3) -> np.ndarray:
    """Round to D decimal places."""
    return np.round(N, D)


def RET(S: np.ndarray, N: int = 1) -> np.ndarray:
    """Return the value N periods ago (negative index access)."""
    return np.array(S)[-N]


def ABS(S: np.ndarray) -> np.ndarray:
    """Element-wise absolute value."""
    return np.abs(S)


def MAX(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """Element-wise maximum of two arrays."""
    return np.maximum(S1, S2)


def MIN(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """Element-wise minimum of two arrays."""
    return np.minimum(S1, S2)


def MA(S: np.ndarray, N: int) -> np.ndarray:
    """Simple moving average over N periods."""
    return pd.Series(S).rolling(N).mean().values


def REF(S: np.ndarray, N: int = 1) -> np.ndarray:
    """N-period forward shift (i.e. value N periods ago)."""
    return pd.Series(S).shift(N).values


def DIFF(S: np.ndarray, N: int = 1) -> np.ndarray:
    """N-period difference: S_t - S_{t-N}."""
    return pd.Series(S).diff(N)


def STD(S: np.ndarray, N: int) -> np.ndarray:
    """N-period rolling standard deviation (population, ddof=0)."""
    return pd.Series(S).rolling(N).std(ddof=0).values


def IF(S_BOOL: np.ndarray, S_TRUE: np.ndarray, S_FALSE: np.ndarray) -> np.ndarray:
    """Vectorised if-else: S_BOOL ? S_TRUE : S_FALSE."""
    return np.where(S_BOOL, S_TRUE, S_FALSE)


def SUM(S: np.ndarray, N: int) -> np.ndarray:
    """N-period rolling sum. If N <= 0, returns cumulative sum."""
    return pd.Series(S).rolling(N).sum().values if N > 0 else pd.Series(S).cumsum().values


def HHV(S: np.ndarray, N: int) -> np.ndarray:
    """Highest high over N periods."""
    return pd.Series(S).rolling(N).max().values


def LLV(S: np.ndarray, N: int) -> np.ndarray:
    """Lowest low over N periods."""
    return pd.Series(S).rolling(N).min().values


def EMA(S: np.ndarray, N: int) -> np.ndarray:
    """Exponential moving average (alpha=2/(span+1))."""
    return pd.Series(S).ewm(span=N, adjust=False).mean().values


def SMA(S: np.ndarray, N: int, M: float = 1.0) -> np.ndarray:
    """Smoothed moving average (alpha=M/N, adjust=True)."""
    return pd.Series(S).ewm(alpha=M / N, adjust=True).mean().values


def AVEDEV(S: np.ndarray, N: int) -> np.ndarray:
    """Average absolute deviation over N periods."""
    return (
        pd.Series(S)
        .rolling(N)
        .apply(lambda x: np.abs(x - x.mean()).mean())
        .values
    )


def SLOPE(S: np.ndarray, N: int, RS: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Linear regression slope over the last N values.
    If RS=True, returns (slope, fitted_y_array); otherwise just the slope.
    """
    M = pd.Series(S[-N:])
    poly = np.polyfit(M.index, M.values, deg=1)
    Y = np.polyval(poly, M.index)
    if RS:
        return Y[1] - Y[0], Y
    return Y[1] - Y[0]


# ----------------------------------------------------------------------
# Level 1 — composite helpers
# ----------------------------------------------------------------------


def COUNT(S_BOOL: np.ndarray, N: int) -> np.ndarray:
    """Number of True values in the last N periods."""
    return SUM(S_BOOL, N)


def EVERY(S_BOOL: np.ndarray, N: int) -> np.ndarray:
    """True if ALL values were True in the last N periods."""
    R = SUM(S_BOOL, N)
    return IF(R == N, True, False)


def LAST(S_BOOL: np.ndarray, A: int, B: int) -> np.ndarray:
    """
    True if condition held continuously from A periods ago to B periods ago (A >= B).
    LAST(CLOSE>OPEN, 5, 3) — close > open continuously in the last 5 to 3 bars.
    """
    if A < B:
        A = B
    return S_BOOL[-A:-B].sum() == (A - B)


def EXIST(S_BOOL: np.ndarray, N: int = 5) -> np.ndarray:
    """True if condition was True at least once in the last N periods."""
    R = SUM(S_BOOL, N)
    return IF(R > 0, True, False)


def BARSLAST(S_BOOL: np.ndarray) -> np.ndarray:
    """
    Number of bars since the last True value.
    Returns -1 if S_BOOL has never been True.
    """
    M = np.argwhere(S_BOOL)
    return len(S_BOOL) - int(M[-1]) - 1 if M.size > 0 else -1


def FORCAST(S: np.ndarray, N: int) -> np.ndarray:
    """Linear forecast: extrapolate one period ahead using last N values."""
    K, Y = SLOPE(S, N, RS=True)
    return Y[-1] + K


def CROSS(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Detect golden/dead cross: True only on the bar where S1 crosses over S2.
    CROSS(MA(C,5), MA(C,10)) — gold cross; CROSS(MA(C,10), MA(C,5)) — dead cross.
    """
    CROSS_BOOL = IF(S1 > S2, True, False)
    return (COUNT(CROSS_BOOL > 0, 2) == 1) * CROSS_BOOL


# ----------------------------------------------------------------------
# Level 2 — technical indicators
# ----------------------------------------------------------------------


def MACD(
    CLOSE: np.ndarray, SHORT: int = 12, LONG: int = 26, M: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence/Divergence).

    Returns (DIF, DEA, MACD):
        DIF  = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
        DEA  = EMA(DIF, M)
        MACD = (DIF - DEA) * 2
    """
    DIF = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
    DEA = EMA(DIF, M)
    MACD = (DIF - DEA) * 2
    return RD(DIF), RD(DEA), RD(MACD)


def KDJ(
    CLOSE: np.ndarray,
    HIGH: np.ndarray,
    LOW: np.ndarray,
    N: int = 9,
    M1: int = 3,
    M2: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    KDJ Indicator (Stochastic variant).

    Returns (K, D, J):
        RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
        K = EMA(RSV, M1*2-1),  D = EMA(K, M2*2-1),  J = K*3 - D*2
    """
    RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    K = EMA(RSV, (M1 * 2 - 1))
    D = EMA(K, (M2 * 2 - 1))
    J = K * 3 - D * 2
    return K, D, J


def RSI(CLOSE: np.ndarray, N: int = 24) -> np.ndarray:
    """
    Relative Strength Index.

    RSI = SMA(MAX(DIF, 0), N) / SMA(ABS(DIF), N) * 100
    where DIF = CLOSE - REF(CLOSE, 1)
    """
    DIF = CLOSE - REF(CLOSE, 1)
    return RD(SMA(MAX(DIF, 0), N) / SMA(ABS(DIF), N) * 100)


def WR(
    CLOSE: np.ndarray,
    HIGH: np.ndarray,
    LOW: np.ndarray,
    N: int = 10,
    N1: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Williams %R (two periods: N and N1).

    WR(N)  = (HHV(HIGH, N)  - CLOSE) / (HHV(HIGH, N)  - LLV(LOW, N))  * 100
    WR(N1) = (HHV(HIGH, N1) - CLOSE) / (HHV(HIGH, N1) - LLV(LOW, N1)) * 100
    """
    WR = (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    WR1 = (HHV(HIGH, N1) - CLOSE) / (HHV(HIGH, N1) - LLV(LOW, N1)) * 100
    return RD(WR), RD(WR1)


def BIAS(
    CLOSE: np.ndarray, L1: int = 6, L2: int = 12, L3: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bias (Bias Ratio) — deviation of price from its N-period MA.

    BIAS(N) = (CLOSE - MA(CLOSE, N)) / MA(CLOSE, N) * 100
    """
    BIAS1 = (CLOSE - MA(CLOSE, L1)) / MA(CLOSE, L1) * 100
    BIAS2 = (CLOSE - MA(CLOSE, L2)) / MA(CLOSE, L2) * 100
    BIAS3 = (CLOSE - MA(CLOSE, L3)) / MA(CLOSE, L3) * 100
    return RD(BIAS1), RD(BIAS2), RD(BIAS3)


def BOLL(CLOSE: np.ndarray, N: int = 20, P: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.

    MID   = MA(CLOSE, N)
    UPPER = MID + STD(CLOSE, N) * P
    LOWER = MID - STD(CLOSE, N) * P
    """
    MID = MA(CLOSE, N)
    UPPER = MID + STD(CLOSE, N) * P
    LOWER = MID - STD(CLOSE, N) * P
    return RD(UPPER), RD(MID), RD(LOWER)


def PSY(CLOSE: np.ndarray, N: int = 12, M: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    PSY (Psychological Line).

    PSY  = COUNT(CLOSE > REF(CLOSE,1), N) / N * 100
    PSYMA = MA(PSY, M)
    """
    PSY = COUNT(CLOSE > REF(CLOSE, 1), N) / N * 100
    PSYMA = MA(PSY, M)
    return RD(PSY), RD(PSYMA)


def CCI(CLOSE: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, N: int = 14) -> np.ndarray:
    """
    Commodity Channel Index.

    TP = (HIGH + LOW + CLOSE) / 3
    CCI = (TP - MA(TP, N)) / (0.015 * AVEDEV(TP, N))
    """
    TP = (HIGH + LOW + CLOSE) / 3
    return (TP - MA(TP, N)) / (0.015 * AVEDEV(TP, N))


def ATR(
    CLOSE: np.ndarray, HIGH: np.ndarray, LOW: np.ndarray, N: int = 20
) -> np.ndarray:
    """
    Average True Range.

    TR = MAX(MAX(HIGH-LOW, ABS(REF(CLOSE,1)-HIGH)), ABS(REF(CLOSE,1)-LOW))
    ATR = MA(TR, N)
    """
    TR = MAX(
        MAX((HIGH - LOW), ABS(REF(CLOSE, 1) - HIGH)),
        ABS(REF(CLOSE, 1) - LOW),
    )
    return MA(TR, N)


def BBI(
    CLOSE: np.ndarray,
    M1: int = 3,
    M2: int = 6,
    M3: int = 12,
    M4: int = 20,
) -> np.ndarray:
    """
    Bull/Bear Power Index.

    BBI = (MA(CLOSE,M1) + MA(CLOSE,M2) + MA(CLOSE,M3) + MA(CLOSE,M4)) / 4
    """
    return (MA(CLOSE, M1) + MA(CLOSE, M2) + MA(CLOSE, M3) + MA(CLOSE, M4)) / 4


def DMI(
    CLOSE: np.ndarray,
    HIGH: np.ndarray,
    LOW: np.ndarray,
    M1: int = 14,
    M2: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Directional Movement Index (DMI).

    Returns (PDI, MDI, ADX, ADXR):
        PDI  = DMP * 100 / TR
        MDI  = DMM * 100 / TR
        ADX  = MA(ABS(MDI-PDI)/(PDI+MDI)*100, M2)
        ADXR = (ADX + REF(ADX, M2)) / 2
    """
    TR = SUM(
        MAX(
            MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))),
            ABS(LOW - REF(CLOSE, 1)),
        ),
        M1,
    )
    HD = HIGH - REF(HIGH, 1)
    LD = REF(LOW, 1) - LOW
    DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), M1)
    DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
    PDI = DMP * 100 / TR
    MDI = DMM * 100 / TR
    ADX = MA(ABS(MDI - PDI) / (PDI + MDI) * 100, M2)
    ADXR = (ADX + REF(ADX, M2)) / 2
    return PDI, MDI, ADX, ADXR


def TURTLES(
    HIGH: np.ndarray, LOW: np.ndarray, N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turtles trading system channel.

    UP   = HHV(HIGH, N)
    MID  = (UP + LLV(LOW, N)) / 2
    DOWN = LLV(LOW, N)
    """
    UP = HHV(HIGH, N)
    DOWN = LLV(LOW, N)
    MID = (UP + DOWN) / 2
    return UP, MID, DOWN


def KTN(
    CLOSE: np.ndarray,
    HIGH: np.ndarray,
    LOW: np.ndarray,
    N: int = 20,
    M: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keltner Channel (NT variant).

    MID   = EMA((HIGH+LOW+CLOSE)/3, N)
    ATRN  = ATR(CLOSE, HIGH, LOW, M)
    UPPER = MID + 2 * ATRN
    LOWER = MID - 2 * ATRN
    """
    MID = EMA((HIGH + LOW + CLOSE) / 3, N)
    ATRN = ATR(CLOSE, HIGH, LOW, M)
    UPPER = MID + 2 * ATRN
    LOWER = MID - 2 * ATRN
    return UPPER, MID, LOWER


def TRIX(
    CLOSE: np.ndarray, M1: int = 12, M2: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TRIX (Triple EMA).

    TR    = EMA(EMA(EMA(CLOSE, M1), M1), M1)
    TRIX  = (TR - REF(TR, 1)) / REF(TR, 1) * 100
    TRMA  = MA(TRIX, M2)
    """
    TR = EMA(EMA(EMA(CLOSE, M1), M1), M1)
    TRIX = (TR - REF(TR, 1)) / REF(TR, 1) * 100
    TRMA = MA(TRIX, M2)
    return TRIX, TRMA


def VR(CLOSE: np.ndarray, VOL: np.ndarray, M1: int = 26) -> np.ndarray:
    """
    Volume Ratio (VR).

    VR = SUM(IF(CLOSE>REF(CLOSE,1), VOL, 0), M1)
        / SUM(IF(CLOSE<=REF(CLOSE,1), VOL, 0), M1) * 100
    """
    LC = REF(CLOSE, 1)
    return SUM(IF(CLOSE > LC, VOL, 0), M1) / SUM(IF(CLOSE <= LC, VOL, 0), M1) * 100


def EMV(
    HIGH: np.ndarray,
    LOW: np.ndarray,
    VOL: np.ndarray,
    N: int = 14,
    M: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ease of Movement.

    VOLUME = MA(VOL, N) / VOL
    MID    = 100 * (HIGH + LOW - REF(HIGH + LOW, 1)) / (HIGH + LOW)
    EMV    = MA(MID * VOLUME * (HIGH - LOW) / MA(HIGH - LOW, N), N)
    MAEMV  = MA(EMV, M)
    """
    VOLUME = MA(VOL, N) / VOL
    MID = 100 * (HIGH + LOW - REF(HIGH + LOW, 1)) / (HIGH + LOW)
    EMV = MA(MID * VOLUME * (HIGH - LOW) / MA(HIGH - LOW, N), N)
    MAEMV = MA(EMV, M)
    return EMV, MAEMV


def DPO(
    CLOSE: np.ndarray, M1: int = 20, M2: int = 10, M3: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detrended Price Oscillator.

    DPO   = CLOSE - REF(MA(CLOSE, M1), M2)
    MADPO = MA(DPO, M3)
    """
    DPO = CLOSE - REF(MA(CLOSE, M1), M2)
    MADPO = MA(DPO, M3)
    return DPO, MADPO


def BRAR(
    OPEN: np.ndarray,
    CLOSE: np.ndarray,
    HIGH: np.ndarray,
    LOW: np.ndarray,
    M1: int = 26,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BR/AR (意愿/气势指标).

    AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
    BR = SUM(MAX(0, HIGH - REF(CLOSE, 1)), M1) / SUM(MAX(0, REF(CLOSE, 1) - LOW), M1) * 100
    """
    AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
    BR = SUM(MAX(0, HIGH - REF(CLOSE, 1)), M1) / SUM(MAX(0, REF(CLOSE, 1) - LOW), M1) * 100
    return AR, BR


def DMA(
    CLOSE: np.ndarray, N1: int = 10, N2: int = 50, M: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Different of Moving Averages.

    DIF   = MA(CLOSE, N1) - MA(CLOSE, N2)
    DIFMA = MA(DIF, M)
    """
    DIF = MA(CLOSE, N1) - MA(CLOSE, N2)
    DIFMA = MA(DIF, M)
    return DIF, DIFMA


def MTM(CLOSE: np.ndarray, N: int = 12, M: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Momentum.

    MTM   = CLOSE - REF(CLOSE, N)
    MTMMA = MA(MTM, M)
    """
    MTM = CLOSE - REF(CLOSE, N)
    MTMMA = MA(MTM, M)
    return MTM, MTMMA


def MASS(
    HIGH: np.ndarray,
    LOW: np.ndarray,
    N1: int = 9,
    N2: int = 25,
    M: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mass Index.

    MASS  = SUM(MA(HIGH-LOW, N1) / MA(MA(HIGH-LOW, N1), N1), N2)
    MA_MASS = MA(MASS, M)
    """
    MASS = SUM(MA(HIGH - LOW, N1) / MA(MA(HIGH - LOW, N1), N1), N2)
    MA_MASS = MA(MASS, M)
    return MASS, MA_MASS


def ROC(CLOSE: np.ndarray, N: int = 12, M: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rate of Change.

    ROC    = 100 * (CLOSE - REF(CLOSE, N)) / REF(CLOSE, N)
    MAROC  = MA(ROC, M)
    """
    ROC = 100 * (CLOSE - REF(CLOSE, N)) / REF(CLOSE, N)
    MAROC = MA(ROC, M)
    return ROC, MAROC


def EXPMA(CLOSE: np.ndarray, N1: int = 12, N2: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Double Exponential Moving Average.

    Returns (EXPMA(N1), EXPMA(N2)).
    """
    return EMA(CLOSE, N1), EMA(CLOSE, N2)


def OBV(CLOSE: np.ndarray, VOL: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume.

    OBV = SUM(IF(CLOSE>REF(CLOSE,1), VOL,
             IF(CLOSE<REF(CLOSE,1), -VOL, 0)), 0) / 10000
    """
    return SUM(
        IF(
            CLOSE > REF(CLOSE, 1),
            VOL,
            IF(CLOSE < REF(CLOSE, 1), -VOL, 0),
        ),
        0,
    ) / 10000


def MFI(
    CLOSE: np.ndarray,
    HIGH: np.ndarray,
    LOW: np.ndarray,
    VOL: np.ndarray,
    N: int = 14,
) -> np.ndarray:
    """
    Money Flow Index.

    TYP = (HIGH + LOW + CLOSE) / 3
    V1  = SUM(IF(TYP>REF(TYP,1), TYP*VOL, 0), N)
          / SUM(IF(TYP<REF(TYP,1), TYP*VOL, 0), N)
    MFI = 100 - (100 / (1 + V1))
    """
    TYP = (HIGH + LOW + CLOSE) / 3
    V1 = SUM(IF(TYP > REF(TYP, 1), TYP * VOL, 0), N) / SUM(
        IF(TYP < REF(TYP, 1), TYP * VOL, 0), N
    )
    return 100 - (100 / (1 + V1))


def ASI(
    OPEN: np.ndarray,
    CLOSE: np.ndarray,
    HIGH: np.ndarray,
    LOW: np.ndarray,
    M1: int = 26,
    M2: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulation Swing Index.

    Formula:
        LC = REF(CLOSE, 1)
        AA = ABS(HIGH - LC);  BB = ABS(LOW - LC)
        CC = ABS(HIGH - REF(LOW, 1))
        DD = ABS(LC - REF(OPEN, 1))
        R  = IF((AA>BB)&(AA>CC), AA+BB/2+DD/4, IF((BB>CC)&(BB>AA), BB+AA/2+DD/4, CC+DD/4))
        X  = (CLOSE-LC+(CLOSE-OPEN)/2+LC-REF(OPEN,1))
        SI = 16*X/R*MAX(AA,BB)
    """
    LC = REF(CLOSE, 1)
    AA = ABS(HIGH - LC)
    BB = ABS(LOW - LC)
    CC = ABS(HIGH - REF(LOW, 1))
    DD = ABS(LC - REF(OPEN, 1))
    R = IF(
        (AA > BB) & (AA > CC), AA + BB / 2 + DD / 4,
        IF((BB > CC) & (BB > AA), BB + AA / 2 + DD / 4, CC + DD / 4),
    )
    X = (CLOSE - LC + (CLOSE - OPEN) / 2 + LC - REF(OPEN, 1))
    SI = 16 * X / R * MAX(AA, BB)
    ASI = SUM(SI, M1)
    ASIT = MA(ASI, M2)
    return ASI, ASIT


__all__ = [
    "MA", "EMA", "MACD", "BOLL", "RSI", "KDJ", "WR", "BIAS", "PSY", "CCI",
    "ATR", "BBI", "DMI", "TRIX", "VR", "EMV", "DMA", "MTM", "EXPMA", "OBOV",
    "ROC", "LWR", "MFI", "ASI", "SKDJ", "OBV", "WILLR", "ADTM", "BBI",
]
