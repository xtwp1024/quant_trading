# -*- coding: utf-8 -*-
"""
V36 策略配置文件

存储经过优化的V36策略参数

Generated: 2026-04-09
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ===================== A股参数 =====================

@dataclass
class V36AStockConfig:
    """V36 A股配置"""

    # 风控参数
    stop_loss: float = -0.07      # 止损 -7%
    take_profit: float = 0.20     # 止盈 +20%
    time_stop: int = 6             # 时间止损 6天
    slippage: float = 0.001        # 滑点 0.1%

    # 持仓管理
    max_pos: int = 8               # 最大持仓数
    max_sector: int = 4            # 最大持仓板块数
    account_drawdown_limit: float = -0.08  # 账户回撤限制 -8%

    # 因子参数
    bb_period: int = 20            # 布林带周期
    vol_ma_short: int = 5         # 成交量短周期
    vol_ma_long: int = 20          # 成交量长周期
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])

    # Vol Ratio 阈值
    vol_ratio_threshold: float = 0.7  # 当前值

    # 买点开关
    use_stabilization: bool = True      # 企稳买点
    use_support_bounce: bool = True     # 回踩支撑买点
    use_ma5_bounce: bool = False        # 强势回踩5日线（拖累策略，建议关闭）


# ===================== Crypto参数 =====================

@dataclass
class V36CryptoConfig:
    """V36 Crypto配置"""

    # 风控参数
    stop_loss: float = -0.03      # 止损 -3%
    take_profit: float = 0.10     # 止盈 +10%
    time_stop: int = 24            # 时间止损 24小时
    slippage: float = 0.002        # 滑点 0.2%

    # 持仓管理
    max_pos: int = 4               # 最大持仓数
    account_drawdown_limit: float = -0.10  # 账户回撤限制 -10%

    # 因子参数
    bb_period: int = 20            # 布林带周期
    vol_ma_short: int = 5         # 成交量短周期
    vol_ma_long: int = 20          # 成交量长周期

    # Vol Ratio 阈值
    vol_ratio_threshold: float = 0.7


# ===================== 优化后的参数 =====================

# A股优化后的参数（基于50次Optuna试验，Trial 12最佳）
V36_ASTOCK_OPTIMIZED = {
    "stop_loss": -0.074,       # -7.4% (原: -7%)
    "take_profit": 0.315,      # +31.5% (原: +20%)
    "time_stop": 3,             # 3天 (原: 6天)
    "vol_ratio_threshold": 1.197,  # (原: 0.7)
}

# 50次试验详情:
# - 最佳评分: 0.6043
# - 胜率: 51.1% (目标: 60%)
# - 盈亏比: 1.50 (目标: 1.5, 已达标!)
# - 最大回撤: 87.8% (数据问题导致)
# - 平均收益: 1061.2%

# AntiOverfit 验证结果（基于1只股票，数据不足）:
# - PBO: 50% (边界，需要更多数据)
# - DSR: 1.0 > 0.95 (通过)
# - SPA p: 0.0000 < 0.05 (通过)
# - 综合: 未通过验证（需完整数据）

# Crypto优化后的参数（50次Optuna试验，Binance真实数据，Trial 41最佳）
V36_CRYPTO_OPTIMIZED = {
    "stop_loss": -0.063,       # -6.3%
    "take_profit": 0.072,      # +7.2%
    "time_stop": 14,            # 14小时
    "vol_ratio_threshold": 1.411,
}

# Binance真实数据50次试验详情:
# - 最佳评分: 0.5367
# - 胜率: 39.1% (目标: 55%, 未达标)
# - 盈亏比: 1.34 (目标: 1.3, 已达标!)
# - 最大回撤: 83.5% (2022-2023加密熊市导致)
# - 数据: 10个交易对 x 500天日线

# 注意: 胜率未达标原因分析
# 1. 2022-2023加密熊市期间止损频繁触发
# 2. 止盈参数保守(+7.2%)导致牛市收益被截断
# 3. 需要在牛市数据上重新验证


# ===================== 股票池 =====================

V36_STOCK_POOL = {
    "603803": "通信-CPO",
    "603499": "通信算力",
    "603222": "通信趋势",
    "000586": "通信妖股",
    "601869": "通信光缆",
    "300499": "光模块",
    "000062": "电子",
    "002902": "PCB",
    "002384": "PCB",
    "300602": "5G",
    "002364": "电力",
    "000601": "电力",
    "683339": "电网",
    "300933": "电网",
    "002156": "半导体",
    "300042": "存储",
    "300476": "PCB",
    "002645": "稀土",
    "002756": "特钢",
    "002424": "锗业",
}


# ===================== 成功标准 =====================

V36_SUCCESS_CRITERIA = {
    "win_rate": 0.60,       # 胜率 > 60%
    "profit_loss_ratio": 1.5,  # 盈亏比 > 1.5
    "max_drawdown": 0.15,   # 最大回撤 < 15%
}


# ===================== 买点评估结果 =====================

BUY_POINT_EVALUATION = {
    "stabilization": {
        "name": "企稳",
        "win_rate": 0.451,
        "profit_loss_ratio": 1.21,
        "n_trades": 452,
        "recommendation": "keep",  # keep, remove, enhance
        "note": "基础买点，表现良好",
    },
    "support_bounce": {
        "name": "回踩支撑",
        "win_rate": 0.750,
        "profit_loss_ratio": 0.81,
        "n_trades": 4,
        "recommendation": "verify",  # 需要更多数据验证
        "note": "样本太少，结论不可靠",
    },
    "ma5_bounce": {
        "name": "强势回踩5日线",
        "win_rate": 0.388,
        "profit_loss_ratio": 1.16,
        "n_trades": 116,
        "recommendation": "remove",  # 拖累策略
        "note": "拖累总收益，建议移除",
    },
}


# ===================== Phase 4: 信号增强参数 =====================

SIGNAL_ENHANCER_PARAMS = {
    # 市场情绪滤波 (默认关闭，需要大盘数据)
    "enable_market_filter": False,
    "market_ma_period": 20,
    "market_sentiment_threshold": -0.02,

    # 板块轮动滤波 (默认开启)
    "enable_sector_filter": True,
    "sector_momentum_period": 5,

    # 资金流滤波 (默认开启)
    "enable_money_flow": True,
    "money_flow_lookback": 20,
    "money_flow_threshold": 0.3,
}

# 指数代码映射
INDEX_CODES = {
    "600": "sh000001",  # 上证指数
    "000": "sh000001",  # 深证成指
    "001": "sh000001",  # 也是上证
    "002": "sz399001",  # 中小板
    "300": "sz399006",  # 创业板
}
