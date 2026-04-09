# 时间序列/预测/统计建模仓库分析报告

**扫描日期**: 2026-03-30
**数据源**: `D:/Hive/Data/trading_repos/`
**匹配关键词**: time-series, time_series, forecasting, arima, garch, var, state-space, kalman-filter, hidden-markov, kalman, hmm, 统计建模

---

## 匹配仓库列表

| 仓库名 | 模型类型 | 核心功能 | 互补模块 | 代码质量 | 优先级 |
|--------|----------|----------|----------|---------|--------|
| **KalmanBOT_ICASSP23** | KalmanNet (NN增强Kalman) | 配对交易、对冲比率动态估计、滚动预测 | `factors` 配对因子, `risk` 波动率 | 8/10 | **HIGH** |
| **IBApi-GARCH-CrackSpreadTrading-Algo** | GARCH | 能源价差交易、GARCH波动率预测、z-score触发 | `risk` 波动率模型 | 7/10 | **HIGH** |
| **Hybrid-LSTM-GARCHE-model-for-NASDAQ** | LSTM+GARCH+SARIMA | NASDAQ预测、波动率建模、季节性 | `factors` 预测因子 | 6/10 | **HIGH** |
| **Advanced-Algorithmic-Trading** (chapter-time-series) | HMM+Kalman | 市场机制识别、配对交易动态对冲 | `factors` 市场状态, `risk` | 7/10 | **HIGH** |
| **AI-Powered-Energy-Algorithmic-Trading** | HMM+Neural Networks | 能源股票交易、Black-Litterman组合 | `factors` 预测因子 | 6/10 | MEDIUM |
| **Quantropy** | ADF/KPSS/Phillips-Perron | 平稳性检验、单位根检验 | `data` 预处理 | 6/10 | MEDIUM |
| **Financial-analytics** | GARCH | 波动率建模(Streamlit可视化) | `risk` 波动率模型 | 5/10 | MEDIUM |
| **stock-pairs-trading-python** | Kalman+Cointegration | 配对交易、协整检验、z-score策略 | `factors` 配对因子 | 7/10 | MEDIUM |
| **RegimeSwitchingMomentumStrategy** | HMM (GaussianHMM) | 市场机制检测、动量策略 | `factors` 市场状态 | 6/10 | MEDIUM |
| **LLM-TradeBot** (prophet) | Prophet+LightGBM | 加密货币预测、Binance数据 | `factors` 预测 | 6/10 | LOW |
| **Stock-market-forecasting** | LSTM+RF | S&P500日内预测 | `factors` 预测 | 6/10 | LOW |
| **TradingNeuralNetwork** (arma.py) | ARMA | 时间序列基础模型 | `data` 处理 | 5/10 | LOW |

---

## HIGH 优先级详细分析

### 1. KalmanBOT_ICASSP23
**路径**: `D:/Hive/Data/trading_repos/KalmanBOT_ICASSP23/`
- **核心**: KalmanNet (GRU架构神经网络增强卡尔曼滤波)
- **亮点**: 状态空间+深度学习融合，对冲比率在线学习
- **学术出处**: ICASSP 2023

### 2. IBApi-GARCH-CrackSpreadTrading-Algo
**路径**: `D:/Hive/Data/trading_repos/IBApi-GARCH-CrackSpreadTrading-Algo/`
- **核心**: GARCH波动率驱动策略，原油/天然气价差
- **亮点**: 自动GARCH参数优化，z-score触发，IB API执行

### 3. Advanced-Algorithmic-Trading
**路径**: `D:/Hive/Data/trading_repos/Advanced-Algorithmic-Trading/`
- **核心**: HMM市场机制识别 + Kalman Filter配对交易
- **亮点**: 系统性教程代码，三个关键章节

---

## 互补性分析

| 现有模块 | 可吸收的模型/功能 |
|---------|------------------|
| `factors` | Kalman Filter对冲比率、HMM市场状态、GARCH波动率、ARIMA预测 |
| `risk` | GARCH波动率模型、协整检验、状态空间风险 |
| `data` | ADF/KPSS平稳性检验 |
| `execution` | IB API集成模式 |

---

## 吸收建议

### HIGH优先级
1. **KalmanBOT_ICASSP23** - 学术级状态空间+神经网络融合
2. **IBApi-GARCH-CrackSpreadTrading-Algo** - 成熟GARCH实现
3. **Advanced-Algorithmic-Trading** - 系统性HMM/Kalman教程

### MEDIUM优先级
4. **stock-pairs-trading-python** - 简洁Kalman配对交易
5. **RegimeSwitchingMomentumStrategy** - HMM市场状态检测
6. **Quantropy** - 平稳性检验工具集

---

**分析日期**: 2026-03-30
