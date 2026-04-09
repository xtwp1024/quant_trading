# 技术分析仓库吸收分析报告

**扫描日期**: 2026-03-30
**数据源**: `D:/Hive/Data/trading_repos/`
**匹配关键词**: technical-analysis, indicator, chart-pattern, candlestick, moving-average, macd, rsi, bollinger, fibonacci, pattern, 技术分析, 技术指标, 均线
**匹配仓库总数**: 50+个

---

## 仓库分析表

| 仓库名 | 指标类型 | 核心功能 | 互补模块 | 代码质量 | 优先级 |
|--------|----------|----------|----------|----------|--------|
| **Python-Financial-Technical-Indicators-Pandas (MyTT)** | 40+指标 | 纯NumPy/Pandas实现(MACD/RSI/BOLL/KDJ/CCI/ATR/DMI/TRIX/VR/EMV/OBV等), 零依赖 | `factors.technical_indicators` | 8.5/10 | **HIGH** |
| **stock-indicators-python** | 50+指标 | 专业开源库, .NET高性能核心, 完整测试 | `factors.stock_indicators` | 9/10 | **HIGH** |
| **polymarket-crypto-toolkit** | EMA/SMA/RSI/MACD/Bollinger | 现代化架构, 模块化可插拔 | `factors.polymarket_adapters` | 8/10 | **HIGH** |
| **AutomationTrading-Strategy-Backtesting-Suite** | 15个复杂指标 | Pine Script v5转换, Firestorm/SSL/WaveTrend/Squeeze/FiboMAI等 | `strategies.indicator_catalog` | 8/10 | **HIGH** |
| **freqtrade** | 全套TA指标 | 专业级加密框架, 15+交易所, FreqAI ML优化 | `execution.freqtrade_adapter` | 9/10 | **HIGH** |
| **backtrader** | 通用量化框架 | CCXT/IBKR双适配器, 策略沙箱, 多语言i18n | `backtester.backtrader_engine` | 8.5/10 | **HIGH** |
| **hummingbot** | 蜡烛/K线数据 | 15+交易所蜡烛数据馈送, WebSocket实时 | `data.hummingbot_feeds` | 8/10 | **HIGH** |
| **OctoBot** | TA评估器 | RSI/MACD/EMA及蜡烛形态识别, 完整回测 | `bots.octobot_integration` | 8/10 | MEDIUM |
| **ChartScanAI** | 图表形态识别 | YOLOv8深度学习蜡烛图形态识别 | `ml.chart_pattern_recognizer` | 7/10 | MEDIUM |
| **Hunter** | RSI/MACD/Bollinger/VWAP | 多因子质量评分, TechnicalExitManager | `risk.hunter_style` | 7.5/10 | MEDIUM |
| **MACD_BTCUSD** | MACD | QuantConnect 1小时MACD, 978%收益 | `strategies.macd` | 7/10 | MEDIUM |
| **MeanReversionAlgo** | RSI/ROC/均线 | Larry Connors均值回归, RSI 2周期反弹 | `strategies.mean_reversion` | 7/10 | MEDIUM |
| **ReversionSys** | RSI/ROC | Amibroker+Python+IB API均值回归 | `strategies.reversion` | 7/10 | MEDIUM |
| **binance-technical-algorithm** | MACD/RSI/威廉%R | 币安技术分析, 动态订单大小 | `signals.binance_ta` | 6.5/10 | MEDIUM |
| **stock_market_indicators** | 20+基础指标 | 小型库EMA/MACD/OBV/ATR/Bollinger/Chaikin/RSI等 | `factors.basic_indicators` | 6/10 | LOW |

---

## HIGH 优先级详细分析

### 1. MyTT (Python-Financial-Technical-Indicators-Pandas)
- **核心**: 纯Python实现40+技术指标, 零依赖
- **指标**: MACD/RSI/BOLL/KDJ/WR/BIAS/PSY/CCI/ATR/BBI/DMI/TRIX/VR/EMV/DMA/MTM/EXPMA/OBV
- **价值**: 可直接整合到 `factors` 模块, 替代Ta-Lib

### 2. stock-indicators-python
- **核心**: 专业开源库, 50+指标, .NET高性能
- **价值**: 作为 `stock_indicators` 依赖库

### 3. freqtrade
- **核心**: 成熟生产级框架, 15+交易所, FreqAI ML
- **价值**: 可作为 `execution` 层或吸收其指标

### 4. backtrader
- **核心**: CCXT/IBKR双适配器, 策略沙箱隔离
- **价值**: 回测引擎集成

### 5. hummingbot
- **核心**: 15+交易所统一蜡烛数据接口
- **价值**: 吸收 `CandlesFactory` 架构到 `data` 模块

---

## 吸收建议

### 立即吸收
1. **MyTT** → 纯Python零依赖指标 → `factors.technical`
2. **freqtrade** → 生产级框架 → `execution` 层
3. **backtrader** → 回测引擎 → `backtester`
4. **hummingbot** → 交易所数据接口 → `data`

### 选择性吸收
5. **ChartScanAI** → YOLOv8形态识别 → `ml`
6. **Hunter** → TechnicalExitManager → `risk`

---

**分析日期**: 2026-03-30
