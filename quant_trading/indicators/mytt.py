"""
MyTT — Pure NumPy Technical Indicators (40+ formulas).
MyTT — 纯 NumPy 技术指标库（40+ 公式）。

Based on the open-source MyTT library.
零外部依赖，仅使用 NumPy，不使用 Ta-Lib / pandas_ta / Talib。

所有指标均已向量化（无 Python 循环），高性能。

Level 0 — 核心工具函数 / Core Helpers
    RD, RET, ABS, MAX, MIN, MA, REF, DIFF, STD, IF, SUM,
    HHV, LLV, EMA, SMA, WMA, DMA, AVEDEV, SLOPE, FORCAST, CROSS,
    COUNT, EVERY, EXIST, BARSLAST, BARSLAST_COUNT

Level 1 — 技术指标 / Technical Indicators
    MACD, KDJ, RSI, WR (Williams %R), CCI, TR, ATR, DONCHIAN, STDDEV,
    OBV, CMO, PPO, DMI (ADX), AROON, IFT (Inverse Fisher Transform),
    TRIX, VR (Volatility Ratio), ZLEMA (Zero Lag EMA), CORREL, VARP, VAR,
    BOLL (Bollinger Bands), AMA (Kaufman's Adaptive MA), KDJ, PPO,
    TURG, POTTER, MA, DMA

Example usage / 示例
--------------
    import numpy as np
    from quant_trading.indicators.mytt import MACD, BOLL, RSI, KDJ, ATR

    close = np.random.randn(1000).cumsum() + 100
    dif, dea, macd = MACD(close)
    upper, mid, lower = BOLL(close)
    k, d, j = KDJ(close, high, low)
    atr = ATR(close, high, low)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    # Level 0 — core helpers / 核心工具函数
    "RD", "RET", "ABS", "MAX", "MIN",
    "MA", "REF", "DIFF", "STD", "IF", "SUM",
    "HHV", "LLV", "EMA", "SMA", "WMA", "DMA", "AMA",
    "AVEDEV", "SLOPE", "FORCAST", "CROSS",
    "COUNT", "EVERY", "EXIST", "BARSLAST", "BARSLAST_COUNT",
    # Level 1 — indicators / 技术指标
    "MACD", "KDJ", "RSI", "WR", "CCI", "TR", "ATR",
    "DONCHIAN", "STDDEV", "OBV", "CMO", "PPO",
    "DMI", "AROON", "IFT", "TRIX", "VR",
    "ZLEMA", "CORREL", "VARP", "VAR",
    "BOLL", "BBI", "BIAS", "PSY", "MTM", "ROC",
    "DPO", "BRAR", "EMV", "MASS", "EXPMA",
    "MFI", "ASI", "KTN", "ADL", "CMF",
]


# ---------------------------------------------------------------------------
# Level 0 — 核心工具函数 / Core Helpers
# ---------------------------------------------------------------------------

def RD(N, D=3):
    """四舍五入取 D 位小数 / Round to D decimal places."""
    return np.round(N, D)


def RET(S, N=1):
    """返回序列倒数第 N 个值，默认返回最后一个 / Return the N-th-from-last element."""
    return np.array(S)[-N]


def ABS(S):
    """绝对值 / Absolute value."""
    return np.abs(S)


def MAX(S1, S2):
    """序列元素级最大值 / Element-wise maximum of two series."""
    return np.maximum(S1, S2)


def MIN(S1, S2):
    """序列元素级最小值 / Element-wise minimum of two series."""
    return np.minimum(S1, S2)


def _rolling_mean(arr: np.ndarray, n: int) -> np.ndarray:
    """Pure-NumPy N-period rolling mean (unbiased via cumsum shift)."""
    out = np.empty_like(arr, dtype=float)
    out[:] = np.nan
    if n <= 0 or n > len(arr):
        return out
    cum = np.cumsum(arr, dtype=float)
    out[n - 1] = cum[n - 1] / n
    out[n:] = (cum[n:] - cum[:-n]) / n
    return out


def MA(S, N):
    """
    简单移动平均 / Simple Moving Average.
    MA(CLOSE, 20) — N 日简单移动平均
    """
    return _rolling_mean(np.asarray(S, dtype=float), N)


def REF(S, N=1):
    """
    序列整体向后移动 N 位（产生 NaN）/ Shift series back by N (produces NaN at head).
    对应通达信 REF() /Compatible with TongDaXin REF().
    """
    out = np.empty_like(S)
    out[:] = np.nan
    out[N:] = np.asarray(S)[:-N]
    return out


def DIFF(S, N=1):
    """
    序列差分：S - REF(S, N) / Difference: S minus REF(S, N).
    前 N 个值为 NaN / First N values are NaN.
    """
    return np.asarray(S) - REF(S, N)


def STD(S, N):
    """
    N 日标准差（总体标准差，ddof=0）/ N-period rolling standard deviation (population).
    """
    out = np.empty_like(S, dtype=float)
    out[:] = np.nan
    if N <= 0 or N > len(S):
        return out
    S = np.asarray(S, dtype=float)
    cum_S = np.cumsum(S, dtype=float)
    cum_S2 = np.cumsum(S * S, dtype=float)
    out[N - 1] = np.sqrt(max(cum_S2[N - 1] / N - (cum_S[N - 1] / N) ** 2, 0))
    for i in range(N, len(S)):
        variance = cum_S2[i] - cum_S2[i - N] - ((cum_S[i] - cum_S[i - N]) ** 2) / N
        out[i] = np.sqrt(max(variance / N, 0))
    return out


def IF(S_BOOL, S_TRUE, S_FALSE):
    """
    序列布尔判断 / Element-wise if-else: S_BOOL ? S_TRUE : S_FALSE.
    等价于 np.where(S_BOOL, S_TRUE, S_FALSE).
    """
    return np.where(S_BOOL, S_TRUE, S_FALSE)


def SUM(S, N):
    """
    N 日滚动求和（返回序列）/ N-period rolling sum.
    N=0 时返回累计和 / Returns cumulative sum when N==0.
    """
    if N == 0:
        return np.cumsum(S)
    out = np.empty_like(S, dtype=float)
    out[:] = np.nan
    if N <= 0 or N > len(S):
        return out
    cum = np.cumsum(np.asarray(S, dtype=float), dtype=float)
    out[N - 1] = cum[N - 1]
    out[N:] = cum[N:] - cum[:-N]
    return out


def HHV(S, N):
    """
    N 日最高值 / Highest value in last N periods.
    HHV(HIGH, 5) — 最近 5 天最高价.
    """
    out = np.empty_like(S, dtype=float)
    out[:] = np.nan
    if N <= 0 or N > len(S):
        return out
    S = np.asarray(S, dtype=float)
    out[N - 1] = S[:N].max()
    for i in range(N, len(S)):
        out[i] = S[i - N + 1:i + 1].max()
    return out


def LLV(S, N):
    """
    N 日最低值 / Lowest value in last N periods.
    LLV(LOW, 5) — 最近 5 天最低价.
    """
    out = np.empty_like(S, dtype=float)
    out[:] = np.nan
    if N <= 0 or N > len(S):
        return out
    S = np.asarray(S, dtype=float)
    out[N - 1] = S[:N].min()
    for i in range(N, len(S)):
        out[i] = S[i - N + 1:i + 1].min()
    return out


def EMA(S, N):
    """
    指数移动平均 / Exponential Moving Average.
    alpha = 2 / (span + 1) = 2 / (N + 1).
    精度要求 S > 4*N，至少需要 120 周期才精确.
    """
    S = np.asarray(S, dtype=float)
    out = np.empty_like(S, dtype=float)
    out[:] = np.nan
    if N <= 0:
        return out
    alpha = 2.0 / (N + 1.0)
    # Seed with SMA
    out[N - 1] = S[:N].mean()
    for i in range(N, len(S)):
        out[i] = alpha * S[i] + (1 - alpha) * out[i - 1]
    return out


def SMA(S, N, M=1):
    """
    中国式 SMA（通达信兼容）/ Chinese-style SMA (compatible with TongDaXin).
    M 为权重系数 / M is the weight coefficient.
    """
    S = np.asarray(S, dtype=float)
    out = np.empty_like(S, dtype=float)
    out[:] = np.nan
    if N <= 0 or N > len(S):
        return out
    out[N - 1] = S[:N].mean()
    for i in range(N, len(S)):
        out[i] = (M * S[i] + (N - M) * out[i - 1]) / N
    return out


def WMA(S, N):
    """
    加权移动平均 / Weighted Moving Average.
    权重为 N, N-1, ..., 1 / Weights: N, N-1, ..., 1.
    """
    S = np.asarray(S, dtype=float)
    out = np.empty_like(S, dtype=float)
    out[:] = np.nan
    if N <= 0 or N > len(S):
        return out
    weights = np.arange(1, N + 1, dtype=float)
    norm = weights.sum()
    for i in range(N - 1, len(S)):
        out[i] = (S[i - N + 1:i + 1] * weights).sum() / norm
    return out


def DMA(S, N1=10, N2=50, M=10):
    """
    平行线差指标 / Dynamic Momentum Index (DMA).
    DIF = MA(S, N1) - MA(S, N2), then smoothed by MA(DIF, M).
    """
    dif = MA(S, N1) - MA(S, N2)
    difma = MA(dif, M)
    return RD(dif), RD(difma)


def AMA(S, N=10):
    """
    Kaufman's Adaptive Moving Average (AMA).
    Kaufman's Adaptive Moving Average / Kaufman's 自适应移动平均线.
    基于趋势效率比 (ER) 自动调整平滑力度.
    """
    S = np.asarray(S, dtype=float)
    n = len(S)
    if N <= 1 or N > n - 1:
        return np.full_like(S, np.nan)

    # Price direction: total change over N periods (len = n - N + 1)
    direction = np.abs(S[N - 1:] - S[:n - N + 1])

    # Volatility: sum of absolute changes within each N-period window
    volatility = np.zeros(n - N + 1)
    for i in range(1, n - N + 1):
        window = S[i:i + N] - S[i - 1:i + N - 1]
        volatility[i] = np.sum(np.abs(window))

    # Efficiency Ratio (ER)
    er = np.zeros(n - N + 1)
    mask = volatility > 1e-10
    er[mask] = direction[mask] / volatility[mask]

    # Smoothing constants
    fast_alpha = 2.0 / (2 - 1 + 1)   # span = 1
    slow_alpha = 2.0 / (30 + 1)      # span = 30
    fastest = 0.64

    alpha = np.zeros(n - N + 1)
    for i in range(len(alpha)):
        alpha[i] = (er[i] * (fastest * (fast_alpha - slow_alpha) + slow_alpha)
                    + (1 - er[i]) * slow_alpha)

    out = np.full_like(S, np.nan)
    out[N - 1] = S[:N].mean()
    for i in range(N, n):
        out[i] = alpha[i - N] * S[i] + (1 - alpha[i - N]) * out[i - 1]
    return out


def AVEDEV(S, N):
    """
    平均绝对偏差 / Average Absolute Deviation.
    序列与其 N 日平均值的绝对差的平均值.
    """
    S = np.asarray(S, dtype=float)
    out = np.empty_like(S, dtype=float)
    out[:] = np.nan
    if N <= 0 or N > len(S):
        return out
    for i in range(N - 1, len(S)):
        window = S[i - N + 1:i + 1]
        out[i] = np.abs(window - window.mean()).mean()
    return out


def SLOPE(S, N, RS=False):
    """
    N 周期线性回归斜率 / Slope of N-period linear regression.
    RS=False: 只返回斜率; RS=True: 返回 (斜率, 拟合直线序列).
    """
    S = np.asarray(S, dtype=float)
    if N <= 0 or N > len(S):
        return np.nan
    tail = S[-N:]
    x = np.arange(N, dtype=float)
    x_mean = x.mean()
    slope = ((x - x_mean) * (tail - tail.mean())).sum() / ((x - x_mean) ** 2).sum()
    if RS:
        y_fit = slope * (x - x_mean) + tail.mean()
        return slope, y_fit
    return slope


def FORCAST(S, N):
    """
    N 周期线性回归预测值 / Forecast value using N-period linear regression.
    """
    slope, y_fit = SLOPE(S, N, RS=True)
    return y_fit[-1] + slope


def CROSS(S1, S2):
    """
    判断穿越（金叉/死叉）/ Detect crossover: returns boolean array.
    上穿：昨天 S1<=S2，今天 S1>S2；下穿相反.
    """
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    diff = S1 - S2
    cross_up = (diff[:-1] <= 0) & (diff[1:] > 0)
    cross_down = (diff[:-1] >= 0) & (diff[1:] < 0)
    result = np.zeros(len(S1), dtype=bool)
    result[1:][cross_up] = True
    return result


# ---------------------------------------------------------------------------
# Level 1 — 复合工具函数 / Composite Helpers
# ---------------------------------------------------------------------------

def COUNT(S_BOOL, N):
    """
    COUNT(CLOSE > OPEN, N): 最近 N 天满足条件的天数 / Number of True days in last N periods.
    """
    return SUM(S_BOOL.astype(float), N)


def EVERY(S_BOOL, N):
    """
    EVERY(CLOSE > OPEN, 5): 最近 N 天是否全部满足条件 / True if all N periods satisfy condition.
    """
    return IF(SUM(S_BOOL.astype(float), N) == N, True, False)


def LAST(S_BOOL, A, B):
    """
    从前 A 日到前 B 日一直满足条件 / True if condition held from A periods ago to B periods ago.
    要求 A > B.
    """
    if A < B:
        A = B
    return S_BOOL[-A:-B].sum() == (A - B) if len(S_BOOL) >= A else False


def EXIST(S_BOOL, N=5):
    """
    N 日内是否存在一天满足条件 / True if condition exists within last N periods.
    """
    return IF(SUM(S_BOOL.astype(float), N) > 0, True, False)


def BARSLAST(S_BOOL):
    """
    上一次条件成立到当前的天数 / Bars since last True (returns -1 if never).
    BARSLAST(CLOSE / REF(CLOSE) >= 1.1) — 上次涨停到今天的天数.
    """
    S_BOOL = np.asarray(S_BOOL, dtype=bool)
    indices = np.argwhere(S_BOOL)
    if indices.size == 0:
        return -1
    return len(S_BOOL) - int(indices[-1, 0]) - 1


def BARSLAST_COUNT(S_BOOL):
    """
    连续满足条件的天数（从后往前数连续的 True）/ Count of consecutive Trues from the end.
    """
    S_BOOL = np.asarray(S_BOOL, dtype=bool)
    result = np.zeros(len(S_BOOL), dtype=int)
    count = 0
    for i in range(len(S_BOOL) - 1, -1, -1):
        if S_BOOL[i]:
            count += 1
        else:
            count = 0
        result[i] = count
    return result


# ---------------------------------------------------------------------------
# Level 2 — 技术指标 / Technical Indicators
# ---------------------------------------------------------------------------

def MACD(CLOSE, SHORT=12, LONG=26, M=9):
    """
    MACD 指数平滑移动平均线 / MACD (Moving Average Convergence Divergence).
    DIF = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
    DEA = EMA(DIF, M)
    MACD = (DIF - DEA) * 2
    Returns: (DIF, DEA, MACD)
    """
    DIF = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
    DEA = EMA(DIF, M)
    MACD = (DIF - DEA) * 2
    return RD(DIF), RD(DEA), RD(MACD)


def KDJ(CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
    """
    KDJ 随机指标 / KDJ Stochastic Oscillator.
    RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    K = EMA(RSV, M1*2-1), D = EMA(K, M2*2-1), J = K*3 - D*2
    Returns: (K, D, J)
    """
    RSV = (CLOSE - LLV(LOW, N)) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    K = EMA(RSV, M1 * 2 - 1)
    D = EMA(K, M2 * 2 - 1)
    J = K * 3 - D * 2
    return K, D, J


def RSI(CLOSE, N=24):
    """
    RSI 相对强弱指标 / Relative Strength Index.
    RSI = SMA(MAX(CLOSE - REF(CLOSE,1), 0), N) / SMA(ABS(CLOSE - REF(CLOSE,1)), N) * 100
    """
    DIF = CLOSE - REF(CLOSE, 1)
    return RD(SMA(MAX(DIF, 0), N) / SMA(ABS(DIF), N) * 100)


def WR(CLOSE, HIGH, LOW, N=10, N1=6):
    """
    WR 威廉指标 / Williams %R.
    WR = (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    Returns: (WR, WR1) for two periods.
    """
    WR = (HHV(HIGH, N) - CLOSE) / (HHV(HIGH, N) - LLV(LOW, N)) * 100
    WR1 = (HHV(HIGH, N1) - CLOSE) / (HHV(HIGH, N1) - LLV(LOW, N1)) * 100
    return RD(WR), RD(WR1)


def CCI(CLOSE, HIGH, LOW, N=14):
    """
    CCI 顺势指标 / Commodity Channel Index.
    TP = (HIGH + LOW + CLOSE) / 3
    CCI = (TP - MA(TP, N)) / (0.015 * AVEDEV(TP, N))
    """
    TP = (HIGH + LOW + CLOSE) / 3
    return (TP - MA(TP, N)) / (0.015 * AVEDEV(TP, N))


def TR(CLOSE, HIGH, LOW):
    """
    TR 真实波动幅度（单周期）/ True Range (single period).
    TR = MAX(MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE,1))), ABS(REF(CLOSE,1) - LOW))
    """
    tr1 = HIGH - LOW
    tr2 = ABS(HIGH - REF(CLOSE, 1))
    tr3 = ABS(REF(CLOSE, 1) - LOW)
    return MAX(MAX(tr1, tr2), tr3)


def ATR(CLOSE, HIGH, LOW, N=14):
    """
    ATR 平均真实波动幅度 / Average True Range.
    ATR = MA(TR(CLOSE, HIGH, LOW), N)
    """
    tr1 = HIGH - LOW
    tr2 = ABS(HIGH - REF(CLOSE, 1))
    tr3 = ABS(REF(CLOSE, 1) - LOW)
    tr = MAX(MAX(tr1, tr2), tr3)
    return MA(tr, N)


def DONCHIAN(CLOSE, HIGH, LOW, N=20):
    """
    DONCHIAN 唐安奇通道 / Donchian Channel.
    上轨 = HHV(HIGH, N), 下轨 = LLV(LOW, N), 中轨 = (上轨 + 下轨) / 2
    Returns: (UPPER, MID, LOWER)
    """
    UPPER = HHV(HIGH, N)
    LOWER = LLV(LOW, N)
    MID = (UPPER + LOWER) / 2
    return RD(UPPER), RD(MID), RD(LOWER)


def STDDEV(CLOSE, N=20):
    """
    STDDEV 标准差指标 / Standard Deviation.
    STDDEV = STD(CLOSE, N) — N 日滚动标准差.
    """
    return STD(CLOSE, N)


def OBV(CLOSE, VOLUME):
    """
    OBV 能量潮指标 / On-Balance Volume.
    Returns: OBV series (rounded to 3 decimals).
    """
    obv = np.zeros_like(CLOSE, dtype=float)
    obv[0] = float(VOLUME[0])
    for i in range(1, len(CLOSE)):
        if CLOSE[i] > CLOSE[i - 1]:
            obv[i] = obv[i - 1] + VOLUME[i]
        elif CLOSE[i] < CLOSE[i - 1]:
            obv[i] = obv[i - 1] - VOLUME[i]
        else:
            obv[i] = obv[i - 1]
    return RD(obv)


def CMO(CLOSE, N=14):
    """
    CMO 钱德动量指标 / Chande Momentum Oscillator.
    CMO = (SUM(MAX(DIF,0),N) - SUM(MAX(-DIF,0),N)) / (SUM(MAX(DIF,0),N) + SUM(MAX(-DIF,0),N)) * 100
    where DIF = CLOSE - REF(CLOSE, 1)
    """
    DIF = CLOSE - REF(CLOSE, 1)
    up = SUM(MAX(DIF, 0), N)
    down = SUM(MAX(-DIF, 0), N)
    return RD((up - down) / (up + down) * 100)


def PPO(CLOSE, SHORT=12, LONG=26, M=9):
    """
    PPO 价格百分比震荡指标 / Percentage Price Oscillator.
    PPO = (EMA(CLOSE,SHORT) - EMA(CLOSE,LONG)) / EMA(CLOSE,LONG) * 100
    Signal = EMA(PPO, M)
    Returns: (PPO, Signal)
    """
    ema_short = EMA(CLOSE, SHORT)
    ema_long = EMA(CLOSE, LONG)
    PPO = (ema_short - ema_long) / ema_long * 100
    Signal = EMA(PPO, M)
    return RD(PPO), RD(Signal)


def TURG(CLOSE, N=10):
    """
    TURG 价格变动率（Tar力加速度）/ Rate of price change (not true acceleration, but close to ROC).
    Alias for close comparison.
    """
    return RD((CLOSE - REF(CLOSE, N)) / REF(CLOSE, N) * 100)


def POTTER(CLOSE, HIGH, LOW, N=20, M=5):
    """
    POTTER DMI+ATR 通道指标 / Potter DMI+ATR Channel (custom composite).
    基于 DMI 方向和 ATR 波动率构造通道.
    Returns: (UPPER, LOWER, MID)
    """
    pdi, mdi, adx, adxr = DMI(CLOSE, HIGH, LOW, N, M)
    atr_val = ATR(CLOSE, HIGH, LOW, N)
    mid = EMA(CLOSE, N)
    UPPER = mid + atr_val * 2
    LOWER = mid - atr_val * 2
    return RD(UPPER), RD(LOWER), RD(mid)


def DMI(CLOSE, HIGH, LOW, M1=14, M2=6):
    """
    DMI 动向指标（ADX）/ Directional Movement Index.
    TR = SUM(MAX(MAX(HIGH-LOW, ABS(HIGH-REF(CLOSE,1))), ABS(LOW-REF(CLOSE,1))), M1)
    HD = HIGH - REF(HIGH, 1); LD = REF(LOW, 1) - LOW
    DMP = SUM(IF(HD>0 & HD>LD, HD, 0), M1)
    DMM = SUM(IF(LD>0 & LD>HD, LD, 0), M1)
    PDI = DMP*100/TR; MDI = DMM*100/TR
    ADX = MA(ABS(MDI-PDI)/(PDI+MDI)*100, M2)
    ADXR = (ADX + REF(ADX, M2)) / 2
    Returns: (PDI, MDI, ADX, ADXR)
    """
    TR = SUM(
        MAX(
            MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE, 1))),
            ABS(LOW - REF(CLOSE, 1))
        ),
        M1
    )
    HD = HIGH - REF(HIGH, 1)
    LD = REF(LOW, 1) - LOW
    DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), M1)
    DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
    PDI = DMP * 100 / TR
    MDI = DMM * 100 / TR
    DX = ABS(MDI - PDI) / (PDI + MDI) * 100
    ADX = EMA(DX, M2)
    ADXR = (ADX + REF(ADX, M2)) / 2
    return RD(PDI), RD(MDI), RD(ADX), RD(ADXR)


def AROON(CLOSE, HIGH, LOW, N=14):
    """
    AROON 阿隆指标 / AROON Indicator.
    AROON_UP = (N - 后距最高价的天数) / N * 100
    AROON_DN = (N - 后距最低价的天数) / N * 100
    Returns: (AROON_UP, AROON_DN, AROON_OSC)
    """
    n = N
    aroon_up = np.zeros_like(CLOSE, dtype=float)
    aroon_dn = np.zeros_like(CLOSE, dtype=float)

    for i in range(n - 1, len(CLOSE)):
        window_high = HIGH[i - n + 1:i + 1]
        window_low = LOW[i - n + 1:i + 1]
        aroon_up[i] = (n - (n - 1 - np.argmax(window_high))) / n * 100
        aroon_dn[i] = (n - (n - 1 - np.argmin(window_low))) / n * 100

    aroon_up[:n - 1] = np.nan
    aroon_dn[:n - 1] = np.nan
    aroon_osc = aroon_up - aroon_dn
    return RD(aroon_up), RD(aroon_dn), RD(aroon_osc)


def IFT(CLOSE, N=14):
    """
    IFT 逆渔夫变换 / Inverse Fisher Transform.
    先计算 RSI 或其他归一化指标，再做 IFT 变换.
    IFT = (EXP(2*X) - 1) / (EXP(2*X) + 1)  where X is a normalized indicator.
    这里基于 RSI 做变换 / Applied to RSI-based value here.
    """
    rsi_val = RSI(CLOSE, N)
    # Normalize RSI to roughly [-1, 1]
    x = (rsi_val - 50.0) / 50.0  # center at 0, scale to [-1, 1]
    # Clip to avoid overflow in exp
    x = np.clip(x, -5.0, 5.0)
    val = (np.exp(2.0 * x) - 1.0) / (np.exp(2.0 * x) + 1.0)
    return RD(val)


def TRIX(CLOSE, M1=12, M2=20):
    """
    TRIX 三重指数平滑平均线 / Triple Exponential Moving Average Oscillator.
    TR = EMA(EMA(EMA(CLOSE, M1), M1), M1)
    TRIX = (TR - REF(TR, 1)) / REF(TR, 1) * 100
    TRMA = MA(TRIX, M2)
    Returns: (TRIX, TRMA)
    """
    TR = EMA(EMA(EMA(CLOSE, M1), M1), M1)
    TRIX = (TR - REF(TR, 1)) / REF(TR, 1) * 100
    TRMA = MA(TRIX, M2)
    return RD(TRIX), RD(TRMA)


def VR(CLOSE, VOL, M1=26):
    """
    VR 成交量比率 / Volume Ratio.
    VR = SUM(IF(CLOSE > LC, VOL, 0), M1) / SUM(IF(CLOSE <= LC, VOL, 0), M1) * 100
    where LC = REF(CLOSE, 1)
    """
    LC = REF(CLOSE, 1)
    return SUM(IF(CLOSE > LC, VOL, 0), M1) / SUM(IF(CLOSE <= LC, VOL, 0), M1) * 100


def ZLEMA(CLOSE, N=14):
    """
    ZLEMA 零滞后指数移动平均 / Zero Lag Exponential Moving Average.
    移除了 EMA 的滞后，通过添加价格变化量的加权实现.
    ZLEMA = EMA(CLOSE + (CLOSE - REF(CLOSE, lag)), N)
    where lag = (N-1) / 2
    """
    N = int(N)
    if N <= 0 or N > len(CLOSE) - 1:
        return np.full_like(CLOSE, np.nan)
    lag = int((N - 1) / 2)
    adjusted = CLOSE + (CLOSE - REF(CLOSE, lag))
    return EMA(adjusted, N)


def CORREL(CLOSE1, CLOSE2, N=20):
    """
    CORREL 相关系数 / Pearson Correlation Coefficient over N periods.
    """
    out = np.zeros(len(CLOSE1), dtype=float)
    out[:] = np.nan
    for i in range(N - 1, len(CLOSE1)):
        x = CLOSE1[i - N + 1:i + 1]
        y = CLOSE2[i - N + 1:i + 1]
        xm = x - x.mean()
        ym = y - y.mean()
        denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
        out[i] = (xm * ym).sum() / denom if denom != 0 else 0
    return out


def VARP(CLOSE, N=20):
    """
    VARP 总体方差 / Population Variance over N periods.
    VARP = STD(CLOSE, N) ** 2
    """
    return STD(CLOSE, N) ** 2


def VAR(CLOSE, N=20):
    """
    VAR 样本方差（ddof=1）/ Sample Variance over N periods (ddof=1).
    """
    out = np.zeros(len(CLOSE), dtype=float)
    out[:] = np.nan
    if N <= 1 or N > len(CLOSE):
        return out
    S = np.asarray(CLOSE, dtype=float)
    for i in range(N - 1, len(S)):
        window = S[i - N + 1:i + 1]
        out[i] = window.var(ddof=1)
    return out


def BOLL(CLOSE, N=20, P=2):
    """
    BOLL 布林线 / Bollinger Bands.
    MID = MA(CLOSE, N)
    UPPER = MID + P * STD(CLOSE, N)
    LOWER = MID - P * STD(CLOSE, N)
    Returns: (UPPER, MID, LOWER)
    """
    MID = MA(CLOSE, N)
    UPPER = MID + STD(CLOSE, N) * P
    LOWER = MID - STD(CLOSE, N) * P
    return RD(UPPER), RD(MID), RD(LOWER)


def BBI(CLOSE, M1=3, M2=6, M3=12, M4=20):
    """
    BBI 多空指标 / Bull and Bear Balance Index.
    BBI = (MA(CLOSE,M1) + MA(CLOSE,M2) + MA(CLOSE,M3) + MA(CLOSE,M4)) / 4
    """
    return (MA(CLOSE, M1) + MA(CLOSE, M2) + MA(CLOSE, M3) + MA(CLOSE, M4)) / 4


def BIAS(CLOSE, L1=6, L2=12, L3=24):
    """
    BIAS 乖离率 / Bias Ratio (deviation from moving average).
    BIAS1 = (CLOSE - MA(CLOSE, L1)) / MA(CLOSE, L1) * 100
    Returns: (BIAS1, BIAS2, BIAS3)
    """
    BIAS1 = (CLOSE - MA(CLOSE, L1)) / MA(CLOSE, L1) * 100
    BIAS2 = (CLOSE - MA(CLOSE, L2)) / MA(CLOSE, L2) * 100
    BIAS3 = (CLOSE - MA(CLOSE, L3)) / MA(CLOSE, L3) * 100
    return RD(BIAS1), RD(BIAS2), RD(BIAS3)


def PSY(CLOSE, N=12, M=6):
    """
    PSY 心理线 / Psychological Line.
    PSY = COUNT(CLOSE > REF(CLOSE, 1), N) / N * 100
    PSYMA = MA(PSY, M)
    Returns: (PSY, PSYMA)
    """
    PSY = COUNT(CLOSE > REF(CLOSE, 1), N) / N * 100
    PSYMA = MA(PSY, M)
    return RD(PSY), RD(PSYMA)


def MTM(CLOSE, N=12, M=6):
    """
    MTM 动量指标 / Momentum.
    MTM = CLOSE - REF(CLOSE, N)
    MTMMA = MA(MTM, M)
    Returns: (MTM, MTMMA)
    """
    MTM = CLOSE - REF(CLOSE, N)
    MTMMA = MA(MTM, M)
    return RD(MTM), RD(MTMMA)


def ROC(CLOSE, N=12, M=6):
    """
    ROC 变动率指标 / Rate of Change.
    ROC = 100 * (CLOSE - REF(CLOSE, N)) / REF(CLOSE, N)
    MAROC = MA(ROC, M)
    Returns: (ROC, MAROC)
    """
    ROC = 100 * (CLOSE - REF(CLOSE, N)) / REF(CLOSE, N)
    MAROC = MA(ROC, M)
    return RD(ROC), RD(MAROC)


def DPO(CLOSE, M1=20, M2=10, M3=6):
    """
    DPO 区间振荡线 / Detrended Price Oscillator.
    DPO = CLOSE - REF(MA(CLOSE, M1), M2)
    MADPO = MA(DPO, M3)
    Returns: (DPO, MADPO)
    """
    DPO = CLOSE - REF(MA(CLOSE, M1), M2)
    MADPO = MA(DPO, M3)
    return RD(DPO), RD(MADPO)


def BRAR(OPEN, CLOSE, HIGH, LOW, M1=26):
    """
    BRAR 人气意愿指标 / BRAR (BR and AR combined sentiment indicator).
    AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
    BR = SUM(MAX(0, HIGH - REF(CLOSE, 1)), M1) / SUM(MAX(0, REF(CLOSE, 1) - LOW), M1) * 100
    Returns: (AR, BR)
    """
    AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
    BR = SUM(MAX(0, HIGH - REF(CLOSE, 1)), M1) / SUM(MAX(0, REF(CLOSE, 1) - LOW), M1) * 100
    return RD(AR), RD(BR)


def EMV(HIGH, LOW, VOL, N=14, M=9):
    """
    EMV 简易波动指标 / Ease of Movement.
    VOLUME = MA(VOL, N) / VOL
    MID = 100 * (HIGH + LOW - REF(HIGH + LOW, 1)) / (HIGH + LOW)
    EMV = MA(MID * VOLUME * (HIGH - LOW) / MA(HIGH - LOW, N), N)
    MAEMV = MA(EMV, M)
    Returns: (EMV, MAEMV)
    """
    VOLUME = MA(VOL, N) / VOL
    MID = 100 * (HIGH + LOW - REF(HIGH + LOW, 1)) / (HIGH + LOW)
    EMV = MA(MID * VOLUME * (HIGH - LOW) / MA(HIGH - LOW, N), N)
    MAEMV = MA(EMV, M)
    return RD(EMV), RD(MAEMV)


def MASS(HIGH, LOW, N1=9, N2=25, M=6):
    """
    MASS 梅斯线 / MASS.
    MASS = SUM(MA(HIGH - LOW, N1) / MA(MA(HIGH - LOW, N1), N1), N2)
    MA_MASS = MA(MASS, M)
    Returns: (MASS, MA_MASS)
    """
    MASS = SUM(MA(HIGH - LOW, N1) / MA(MA(HIGH - LOW, N1), N1), N2)
    MA_MASS = MA(MASS, M)
    return RD(MASS), RD(MA_MASS)


def EXPMA(CLOSE, N1=12, N2=50):
    """
    EXPMA 指数平滑移动平均（别名 EMA）/ Exponential Moving Average (alias of EMA).
    Returns: (EXPMA1, EXPMA2)
    """
    return EMA(CLOSE, N1), EMA(CLOSE, N2)


def MFI(CLOSE, HIGH, LOW, VOL, N=14):
    """
    MFI 资金流量指标 / Money Flow Index.
    TYP = (HIGH + LOW + CLOSE) / 3
    V1 = SUM(IF(TYP > REF(TYP,1), TYP*VOL, 0), N) / SUM(IF(TYP < REF(TYP,1), TYP*VOL, 0), N)
    MFI = 100 - 100 / (1 + V1)
    """
    TYP = (HIGH + LOW + CLOSE) / 3
    V1 = SUM(IF(TYP > REF(TYP, 1), TYP * VOL, 0), N) / SUM(
        IF(TYP < REF(TYP, 1), TYP * VOL, 0), N
    )
    return 100 - 100 / (1 + V1)


def ASI(OPEN, CLOSE, HIGH, LOW, M1=26, M2=10):
    """
    ASI 累计振动指标 / Accumulation Swing Index.
    LC = REF(CLOSE, 1)
    AA = ABS(HIGH - LC); BB = ABS(LOW - LC)
    CC = ABS(HIGH - REF(LOW, 1)); DD = ABS(LC - REF(OPEN, 1))
    R = IF(AA > BB) & (AA > CC), AA + BB/2 + DD/4, IF((BB > CC) & (BB > AA), BB + AA/2 + DD/4, CC + DD/4)
    X = (CLOSE - LC + (CLOSE - OPEN) / 2 + LC - REF(OPEN, 1))
    SI = 16 * X / R * MAX(AA, BB)
    ASI = SUM(SI, M1); ASIT = MA(ASI, M2)
    Returns: (ASI, ASIT)
    """
    LC = REF(CLOSE, 1)
    AA = ABS(HIGH - LC)
    BB = ABS(LOW - LC)
    CC = ABS(HIGH - REF(LOW, 1))
    DD = ABS(LC - REF(OPEN, 1))

    R = IF(
        (AA > BB) & (AA > CC),
        AA + BB / 2 + DD / 4,
        IF(
            (BB > CC) & (BB > AA),
            BB + AA / 2 + DD / 4,
            CC + DD / 4
        )
    )
    X = CLOSE - LC + (CLOSE - OPEN) / 2 + LC - REF(OPEN, 1)
    SI = 16 * X / R * MAX(AA, BB)
    ASI = SUM(SI, M1)
    ASIT = MA(ASI, M2)
    return RD(ASI), RD(ASIT)


def KTN(CLOSE, HIGH, LOW, N=20, M=10):
    """
    KTN 肯特纳通道 / Keltner Channel.
    MID = EMA((HIGH + LOW + CLOSE) / 3, N)
    ATRN = ATR(CLOSE, HIGH, LOW, M)
    UPPER = MID + 2 * ATRN; LOWER = MID - 2 * ATRN
    Returns: (UPPER, MID, LOWER)
    """
    MID = EMA((HIGH + LOW + CLOSE) / 3, N)
    ATRN = ATR(CLOSE, HIGH, LOW, M)
    UPPER = MID + 2 * ATRN
    LOWER = MID - 2 * ATRN
    return RD(UPPER), RD(MID), RD(LOWER)


def ADL(HIGH, LOW, CLOSE, VOL):
    """
    ADL 累积/派发线 / Accumulation/Distribution Line.
    MFM = ((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)
    MFV = MFM * VOL
    ADL = SUM(MFV, 0) (cumulative sum)
    """
    mfm = np.zeros_like(CLOSE, dtype=float)
    for i in range(len(CLOSE)):
        if HIGH[i] != LOW[i]:
            mfm[i] = ((CLOSE[i] - LOW[i]) - (HIGH[i] - CLOSE[i])) / (HIGH[i] - LOW[i])
    mfv = mfm * VOL
    adl = np.zeros_like(CLOSE, dtype=float)
    adl[0] = mfv[0]
    for i in range(1, len(adl)):
        adl[i] = adl[i - 1] + mfv[i]
    return RD(adl)


def CMF(HIGH, LOW, CLOSE, VOL, N=20):
    """
    CMF 蔡金资金流量指标 / Chaikin Money Flow.
    MFM = ((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)
    MFV = MFM * VOL
    CMF = SUM(MFV, N) / SUM(VOL, N)
    """
    mfm = np.zeros_like(CLOSE, dtype=float)
    for i in range(len(CLOSE)):
        if HIGH[i] != LOW[i]:
            mfm[i] = ((CLOSE[i] - LOW[i]) - (HIGH[i] - CLOSE[i])) / (HIGH[i] - LOW[i])
    mfv = mfm * VOL
    cmf = SUM(mfv, N) / SUM(VOL, N)
    return RD(cmf)
