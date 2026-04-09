# 回测/策略框架仓库吸收分析报告

**扫描日期**: 2026-03-30
**数据源**: `D:/Hive/Data/trading_repos/`
**匹配关键词**: backtest, backtesting, strategy-framework, algorithmic-trading, 策略, 回测, backtrader, zipline
**匹配仓库总数**: ~90个

---

## 仓库分析汇总表

| 仓库名 | 策略/框架类型 | 核心功能 | 互补模块 | 代码质量 | 优先级 |
|--------|--------------|---------|---------|---------|--------|
| **SuiteTrading v2** | 向量化回测+FSM仓位+Optuna优化 | 63+ bt/sec, CSCV+DSR+Hansen SPA防过拟合, 12指标引擎, 双跑者架构, 647测试 | `backtester`, `risk`, `optimization` | 9.5/10 | **HIGH** |
| **AbuQuant** | 中国版全栈量化 | 缠论/艾略特波浪/谐波, 18496种策略, A股/美股/港股/期货/期权/加密 | `strategy`, `ml` | 9/10 | **HIGH** |
| **backtrader (WebUI版)** | Backtrader+FastAPI+React | AI驱动, Monaco Editor策略编辑, CCXT+IBKR双适配器, Walk-Forward优化 | `backtester` | 9/10 | **HIGH** |
| **zipline** | 事件驱动回测框架 | Pythonic事件驱动, PyData生态, 内置MA/线性回归, Sklearn/Scipy支持 | `backtester`, `data` | 8.5/10 | **HIGH** |
| **trading-strategy** | DEX量化框架 | 区块链数据+回测+执行, SushiSwap/QuickSwap/PancakeSwap, 多链 | `data`, `strategy` | 8/10 | **HIGH** |
| **algo-backtester (QFAB)** | 装饰器框架 | MultiIndicator/SingleIndicator装饰器, Hyper-parameter调优 | `indicators` | 7.5/10 | MEDIUM |
| **ai-trading-prototype-backtester** | 情绪分析回测 | Binance Vision数据, 情绪CSV驱动, HTML可视化 | `sentiment` | 6.5/10 | MEDIUM |
| **Plutus** | 信号式回测框架 | Yahoo Finance, MACrossoverStrategy等, SignalStrategy接口 | `strategy` | 6/10 | MEDIUM |
| **PyBacktesting** | 艾略特波浪+遗传算法 | Elliott Wave建模, Fibonacci入场/出场, 遗传算法优化 | `optimize` | 6.5/10 | MEDIUM |
| **portfolio-backtester** | 组合回测+再平衡 | SMA/EMA/MACD, CAGR/Sharpe/Sortino, 杠杆ETF | `portfolio` | 6/10 | MEDIUM |
| **strategy_powerbacktest** | Futu OpenAPI | Futu实时/历史, 1m-1d多周期, SQLite/CSV | `broker` | 7/10 | MEDIUM |
| **Quantitative-Trading-Strategy-Based-on-ML** | ML量化策略 | IC/IR因子挖掘, XGBoost预测, HS300风控 | `factors`, `ml` | 7/10 | MEDIUM |
| **Quantopian_Pairs_Trader** | 配对交易 | KPSS/ADF平稳性, OU半衰期, Hurst指数, Kalman滤波 | `pairs` | 6.5/10 | LOW |
| **Backtesting-RSI-Algo** | Zipline+Pyfolio | RSI超买/超卖, Quandl数据 | `strategy` | 5.5/10 | LOW |
| **backtrader-backtests** | Backtrader示例 | StochasticSR, BollBand+ADX | `strategy` | 6/10 | LOW |
| **stock_backtester** | MACD策略 | Yahoo Finance, MACD交叉 | `strategy` | 5.5/10 | LOW |
| **mkbacktester** | FX向量化 | MT5格式CSV, ta库 | `data` | 5/10 | LOW |
| **algorithm-component-library** | 代码片段集合 | 可组合算法片段 | `components` | 4/10 | LOW |

---

## HIGH 优先级详细分析

### 1. SuiteTrading v2
**路径**: `D:/Hive/Data/trading_repos/AutomationTrading-Strategy-Backtesting-Suite/suitetrading/`
- **核心**: 63+ bt/sec向量化回测, 双跑者架构(FSM/Simple)
- **防过拟合**: CSCV + Deflated Sharpe + Hansen SPA测试
- **仓位FSM**: 6种仓位移位 × 6种止损 × 6种风险原型
- **指标**: 12个(6自定义Pine复制 + 6 TA-Lib)
- **优化**: Optuna贝叶斯 + Walk-Forward + 多核并行

### 2. AbuQuant
**路径**: `D:/Hive/Data/trading_repos/abu/`
- **核心**: 缠论/艾略特波浪/谐波理论/K线形态
- **规模**: 18,496种量化策略
- **市场**: A股/美股/港股/期货/期权/加密货币

### 3. backtrader (WebUI版)
**路径**: `D:/Hive/Data/trading_repos/backtrader/`
- **核心**: FastAPI+React全栈, Monaco Editor在线编辑
- **适配器**: CCXT(币安/OKX/Bybit) + IBKR
- **安全**: 策略沙箱隔离(subprocess/docker), Fernet加密

### 4. zipline
**路径**: `D:/Hive/Data/trading_repos/zipline/`
- **核心**: Pythonic事件驱动, PyData生态集成
- **范式**: initialize/handle_data

### 5. trading-strategy
**路径**: `D:/Hive/Data/trading_repos/trading-strategy/`
- **核心**: DEX数据+回测+执行, Trading Strategy Protocol
- **支持**: SushiSwap/QuickSwap/PancakeSwap, 多链

---

## 互补性分析

| 现有短板 | 最佳互补 | 吸收价值 |
|----------|----------|----------|
| 缺乏Walk-Forward/防过拟合 | SuiteTrading v2 (CSCV/DSR/SPA) | 极高 |
| 缺乏FSM仓位管理 | SuiteTrading v2 (Position FSM) | 极高 |
| 指标库有限 | SuiteTrading v2 (12指标+Pine复制) | 高 |
| 缠论/波浪/谐波缺失 | AbuQuant | 高 |
| zipline集成 | backtrader WebUI版 | 高 |
| DEX/Chain数据 | trading-strategy | 中 |

---

## 吸收建议

### 立即吸收 (HIGH)
1. **SuiteTrading v2** → 防过拟合 + FSM仓位管理系统
2. **AbuQuant** → 缠论/波浪/谐波理论
3. **backtrader WebUI版** → 全栈回测平台
4. **trading-strategy** → DEX数据支持

---

**分析日期**: 2026-03-30
