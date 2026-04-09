# Market Making / HFT / Liquidity 相关仓库吸收分析报告

**扫描日期**: 2026-03-30
**数据源**: `D:/Hive/Data/trading_repos/`
**匹配关键词**: market-mak, market_mak, marketmak, hft, high-freq, order-book, orderbook, liquidity, spread, bid-ask, avellaneda, passivbot, grid-trading, TWAP, VWAP
**匹配仓库总数**: 39 个

---

## 执行摘要

本次扫描发现 **39 个** 与 Market Making、高频交易(HFT)、订单簿分析、流动性提供相关的仓库。其中：

- **VERY HIGH (9个)**: 核心做市商策略，值得深入研究吸收
- **HIGH (9个)**: 重要策略实现，互补性强
- **MEDIUM (12个)**: 有价值组件，可选择性吸收
- **LOW (9个)**: 质量较低、已过时或研究性质

---

## 详细分析表

| 仓库名 | 策略类型 | 核心功能 | 互补模块 | 代码质量 | 优先级 |
|--------|----------|----------|----------|----------|--------|
| **JaxMARL-HFT** | Multi-Agent RL (PPO/IPPO), JAX | GPU加速高频交易多智能体RL框架，LOBSTER数据，4096+并行环境 | market_making, HFT, connectors | 9/10 | **VERY HIGH** |
| **MARKET-MAKING-RL** | PPO | HFT做市商RL智能体，完整限单订单簿模拟，reward设计精细 | market_making | 9/10 | **VERY HIGH** |
| **market-making-algorithm-optiver** | Black-Scholes MM | Optiver竞赛第4名，Delta hedging，动态bid/ask定价 | strategy.options_market_making | 9/10 | **VERY HIGH** |
| **options-market-making** | 期权做市 | Delta/Vega对冲，波动率微笑拟合，统计套利 | strategy.options_mm | 8/10 | **VERY HIGH** |
| **polymarket-market-maker-bot** | Polymarket CLOB MM | 生产级Polymarket做市，库存管理，WebSocket实时订单簿，Prometheus监控 | strategy.prediction_market_mm | 9/10 | **VERY HIGH** |
| **py_polymarket_hft_mm** | Polymarket HFT | Python高频预测市场做市，CPU affinity控制，GC管理，BPS阈值 | strategy.prediction_market_mm | 7/10 | **VERY HIGH** |
| **DRL_for_Active_High_Frequency_Trading** | PPO | 主动高频交易DRL框架，Intel股票LOBSTER数据，arXiv论文 | HFT, market_making | 8/10 | HIGH |
| **high-frequency-factors** | 高频因子 | 中国A股39个高频因子，逐笔委托/成交数据 | data.factor_calculator | 7/10 | HIGH |
| **OctoBot-Market-Making** | 多交易所MM | 15+交易所做市，参考价格套利保护，Docker化 | strategy.exchange_mm | 8/10 | HIGH |
| **passivbot** | 加密货币MM | Rust+Python混合，追踪止损/Forager，进化算法优化 | strategy.crypto_mm | 8/10 | HIGH |
| **HFTSimulator** | 交易模拟器 | LSTM+ODE-LSTM策略，真实市场数据回测 | backtest.market_simulator | 8/10 | HIGH |
| **HFT-strategies** | HFT策略集 | C++策略，Kalman Filter/RL/LSTM，IEX数据解析 | strategy.hft_strategies | 7/10 | HIGH |
| **avellaneda-stoikov** | Avellaneda-Stoikov | 2008经典论文实现，inventory-based MM | strategy.microstructure | 7/10 | HIGH |
| **hft-avellaneda** | Avellaneda-Stoikov | A-S论文实现，1000次模拟验证 | strategy.market_making | 7/10 | HIGH |
| **alpha-agent** | Alpha Factors, ML | 101因子+LSTM/XGBoost/Prophet，多智能体协作 | factors, backtester | 8/10 | HIGH |
| **grid_trading_bot** | Grid Trading | 现代模块化网格交易引擎，异步架构，CCXT，Grafana | strategy.grid_trading | 8/10 | MEDIUM |
| **coinbase-exchange-order-book** | Order Book | Coinbase订单簿，WebSocket | data.orderbook_processor | 6/10 | MEDIUM |
| **HFTE** | 市场模拟器 | 订单簿市场模拟器，粒子滤波器 | backtest.simulator | 7/10 | MEDIUM |
| **HFT-Prediction** | ML价格预测 | Keras MLP/RNN预测HFT价格变动 | ml.price_predictor | 6/10 | MEDIUM |

---

## Very High 优先级详细分析

### 1. JaxMARL-HFT
- GPU加速多智能体RL框架 (IPPO, PPO)
- JAX-LOB模拟器 + LOBSTER真实市场数据
- 三类智能体：做市商、执行代理、方向交易
- 支持4096+并行环境，ICAIF 2025论文

### 2. MARKET-MAKING-RL
- Stanford CS234课程项目
- 完整限价订单簿(LOB)物理实现
- Avellaneda-Stoikov + Mykland-Zhang + Toke-Yoshida理论
- CARA utility reward函数

### 3. market-making-algorithm-optiver
- Optiver竞赛第4名方案
- Black-Scholes期权定价 + Delta hedging
- 动态价差调整

### 4. polymarket-market-maker-bot
- 生产级Polymarket CLOB做市商
- 完整库存管理 + 风险控制
- WebSocket实时订单簿 + Prometheus监控

### 5. py_polymarket_hft_mm
- Python高频预测市场做市
- CPU affinity控制 + GC管理优化
- BPS阈值检测

---

## 优先级汇总

### VERY HIGH (9个) - 立即深入研究:
1. JaxMARL-HFT
2. MARKET-MAKING-RL
3. market-making-algorithm-optiver
4. options-market-making
5. polymarket-market-maker-bot
6. py_polymarket_hft_mm

### HIGH (9个) - 纳入吸收计划:
7. avellaneda-stoikov
8. DRL_for_Active_High_Frequency_Trading
9. hft-avellaneda
10. HFTSimulator
11. HFT-strategies
12. high-frequency-factors
13. OctoBot-Market-Making
14. passivbot

### MEDIUM (12个) - 选择性吸收:
15. coinbase-exchange-order-book
16. example-hftish
17. HFTE
18. HFT-Prediction
19. high_frequency_trading
20. Label-Unbalance-in-High-Frequency-Trading
21. grid_trading_bot
22. ML_HFT
23. passivbot_binance_isolated_margin_legacy

### LOW (9个) - 暂不推荐:
24. BisonHFT, BTCChina-MarketMaker, ChannelBreakOutHFT, CryptoHFT, ftx_grid_trading_bot, hft, HFT-Bot-Binance, hft-ext, HFTrader, hftrap, High-frequency-Pairs-trading, Leveraged-grid-trading-bot, ML-HFT

---

**分析日期**: 2026-03-30
