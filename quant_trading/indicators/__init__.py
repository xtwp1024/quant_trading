"""
quant_trading.indicators — Technical indicator library.
量化交易系统技术指标模块。

Pure NumPy implementations (zero external dependencies beyond NumPy).
所有指标均为纯 NumPy 实现。
"""

from quant_trading.indicators.mytt import (
    # Level 0 — core helpers
    RD, RET, ABS, MAX, MIN,
    MA, REF, DIFF, STD, IF, SUM,
    HHV, LLV, EMA, SMA, WMA, DMA, AMA,
    AVEDEV, SLOPE, FORCAST, CROSS,
    COUNT, EVERY, EXIST, BARSLAST, BARSLAST_COUNT,
    # Level 1 — indicators
    MACD, KDJ, RSI, WR, CCI, TR, ATR,
    DONCHIAN, STDDEV, OBV, CMO, PPO,
    DMI, AROON, IFT, TRIX, VR,
    ZLEMA, CORREL, VARP, VAR,
    BOLL, BBI, BIAS, PSY, MTM, ROC,
    DPO, BRAR, EMV, MASS, EXPMA,
    MFI, ASI, KTN, ADL, CMF,
    TURG, POTTER,
)

__all__ = [
    # Level 0
    "RD", "RET", "ABS", "MAX", "MIN",
    "MA", "REF", "DIFF", "STD", "IF", "SUM",
    "HHV", "LLV", "EMA", "SMA", "WMA", "DMA", "AMA",
    "AVEDEV", "SLOPE", "FORCAST", "CROSS",
    "COUNT", "EVERY", "EXIST", "BARSLAST", "BARSLAST_COUNT",
    # Level 1
    "MACD", "KDJ", "RSI", "WR", "CCI", "TR", "ATR",
    "DONCHIAN", "STDDEV", "OBV", "CMO", "PPO",
    "DMI", "AROON", "IFT", "TRIX", "VR",
    "ZLEMA", "CORREL", "VARP", "VAR",
    "BOLL", "BBI", "BIAS", "PSY", "MTM", "ROC",
    "DPO", "BRAR", "EMV", "MASS", "EXPMA",
    "MFI", "ASI", "KTN", "ADL", "CMF",
    "TURG", "POTTER",
]
