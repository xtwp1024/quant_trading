# ML/神经网络/预测模型仓库吸收分析报告

**扫描日期**: 2026-03-30
**数据源**: `D:/Hive/Data/trading_repos/`
**匹配关键词**: ml, lstm, rnn, transformer, neural-network, deep-learning, xgboost, prediction, forecast
**匹配仓库总数**: 36个

---

## 仓库分析汇总表

| 仓库名 | 模型类型 | 核心功能 | 互补模块 | 代码质量 | 优先级 |
|--------|----------|----------|----------|----------|--------|
| **HFT-price-prediction** | LightGBM+RF | HFT限单簿价格变动分类(76-78%), 遗传算法特征选择 | `factors.high_freq` | 8/10 | **HIGH** |
| **Quantitative-Trading-Strategy-Based-on-Machine-Learning** | XGBoost | HS300月频预测, IC/IR因子挖掘, 均值方差风控 | `factors.alpha_evaluator` | 9/10 | **HIGH** |
| **RL-DeepQLearning-Trading** | Double DQN | 33个技术指标+LOB状态, Experience replay | `rl_trader` | 8/10 | **HIGH** |
| **Hybrid-LSTM-GARCHE-model-for-NASDAQ** | LSTM+GARCH+SARIMA | NASDAQ波动率预测, Azure Streamlit可视化 | `factors` 波动率 | 7/10 | **HIGH** |
| **PyTorch-DDPG-Stock-Trading** | DDPG (Actor-Critic) | SH50股票5秒级高频, PyTorch/GPU, OU噪声 | `rl_trader` | 8/10 | **HIGH** |
| **StockPredictionRNN** | LSTM RNN | NYSE OpenBook高频价格预测 | `factors.high_freq` | 7/10 | **HIGH** |
| **DPML** | Dual-Process Meta-Learning | ECML PKDD 2022: 交易量预测双过程元学习 | `factors` 交易量因子 | 9/10 | **HIGH** |
| **AI-Powered-Energy-Algorithmic-Trading** | HMM+Neural Network | 能源股83%回报, Sharpe 0.77, Black-Litterman | `portfolio` 组合优化 | 9/10 | **HIGH** |
| **XGTraderAGI** | XGBoost | 加密货币技术指标信号, 贝叶斯超参优化 | `factors` XGBoost增强 | 8/10 | **HIGH** |
| **Stock-Price-Prediction-LSTM-and-Trading-Strategy-RL** | LSTM+RL | 中国股票LSTM预测+强化学习交易 | `rl` 预测+决策闭环 | 7/10 | **HIGH** |
| **value-based-deep-reinforcement-learning** | Double DQN | PyTorch单股票价值型DRL | `rl_trader` | 7/10 | **HIGH** |
| **TradingNeuralNetwork** | CNN/TDNN/RNN/LSTM | S&P500多模型预测, ARMA/ANN对比 | `factors` 多模型框架 | 8/10 | **HIGH** |
| **ML_HFT** | ML (LOB动力学) | 高频限单簿价格变动预测 | `factors.high_freq` | 7/10 | **HIGH** |
| **Stock-market-forecasting** | Prophet+Keras LSTM | 日频+分钟级预测, 配对交易 | `factors` 预测 | 7/10 | **HIGH** |
| **Algo-Trading-ML-Model** | Random Forest | 技术指标特征分类买卖信号 | `factors` RF增强 | 7/10 | MEDIUM |
| **ML-SuperTrend-MT5** | K-means+SuperTrend | K-means动态SuperTrend参数优化, MT5 | `factors` 技术指标 | 8/10 | MEDIUM |
| **Machine_learning_trading_algorithm** | Random Forest | S&P500 30日涨跌预测 | `factors` 选股 | 7/10 | MEDIUM |
| **Mastering-Algorithmic-Trading-with-Deep-Learning** | LSTM | 多时间框架LSTM, MQL5 EA集成 | `factors` LSTM框架 | 7/10 | MEDIUM |
| **LSTM-Algorithmic-Trading-Bot** | LSTM | PyTorch BTC预测+布林带/随机指标 | `factors` LSTM | 6/10 | MEDIUM |
| **LSTM-Crypto-Price-Prediction** | LSTM RNN | BTC技术指标(MACD/RSI/DPO)预测 | `factors` 加密货币 | 6/10 | MEDIUM |
| **MachineLearningStocks** | Random Forest | Yfinance基本面特征+预测 | `factors` 特征工程 | 6/10 | MEDIUM |
| **Keras-Neuro-Evolution-Trading-Bot** | Neuro-Evolution (GA) | Keras神经进化, 适应度归一化选择 | `optimization` 进化算法 | 6/10 | MEDIUM |
| **stock-trading-ml** | Keras LSTM | 技术指标预测, alpha_vantage数据 | `factors` 预测 | 6/10 | MEDIUM |
| **quarterly-earnings-machine-learning-algo** | ML | 财报季盈利预测, SEC财报解析 | `factors` 基本面 | 6/10 | MEDIUM |
| **Stock_Support_Resistance_ML** | K-means | K-means识别支撑阻力位 | `factors` 技术分析 | 6/10 | MEDIUM |
| **trading-price-prediction** | Neural Network | 股票/外汇/加密预测 | `factors` 预测 | 5/10 | MEDIUM |

---

## HIGH 优先级详细分析

### 1. HFT-price-prediction
- **模型**: LightGBM + Random Forest ensemble
- **亮点**: 76-78%价格变动分类, 遗传算法特征选择, LOB微观结构特征

### 2. Quantitative-Trading-Strategy-Based-on-Machine-Learning
- **模型**: XGBoost分类
- **亮点**: IC/IR/相关性分析, 月频预测, 11.54%年化

### 3. DPML (Dual-Process Meta-Learning)
- **模型**: 双过程元学习
- **亮点**: ECML PKDD 2022论文, 交易量预测

### 4. AI-Powered-Energy-Algorithmic-Trading
- **模型**: HMM + Neural Network + Black-Litterman
- **亮点**: 83%回报, Sharpe 0.77, 双重alpha系统

---

## 吸收建议

### 立即吸收 (HIGH + 质量>=8)
1. **Quantitative-Trading-Strategy-Based-on-Machine-Learning** → 因子挖掘流程
2. **HFT-price-prediction** → 高频特征工程
3. **RL-DeepQLearning-Trading** → DQN增强
4. **ML-SuperTrend-MT5** → 技术指标库

### 长期吸收
5. **PyTorch-DDPG-Stock-Trading** → DDPG参考
6. **TradingNeuralNetwork** → 多模型对比框架

---

**分析日期**: 2026-03-30
