# Risk Management / 风险管理 仓库吸收分析报告

**扫描日期**: 2026-03-30
**数据源**: `D:/Hive/Data/trading_repos/`
**匹配关键词**: risk, portfolio, position-sizing, stop-loss, take-profit, kelly, var, cvar, sharpe, sortino, drawdown, risk-parity, position-sizing, sizing, hedge, hedging

---

## 仓库分析表

| 仓库名 | 风控类型 | 核心功能 | 互补模块 | 代码质量 | 优先级 |
|--------|----------|----------|----------|----------|--------|
| **OptionSuite/riskManager** | 期权风控/希腊值管理 | Put Vertical/Strangle风控; 持仓周期管理(50%止盈/21天/到期); 5种策略类型 | `options/risk_manager.py` - 期权特异性风控 | 8/10 | **HIGH** |
| **PortfolioStrategyBacktestUS** | 组合风控/波动率门限 | Markowitz均值方差; Risk-Parity; 波动率门限(短期>长期均值则50%减仓) | `maverick_risk.py` - 波动率门限逻辑 | 8/10 | **HIGH** |
| **Hedge_Fund_Agents** | 多Agent风控 | 9种投资Agent; 独立Risk Manager Agent; 20%单仓限制 | `five_force/agents/` - AI2_Risk模块 | 7/10 | **HIGH** |
| **five_factors_riskControl** | 量化风控/XGBoost | 5因子模型; HS300波动率突破风控(>长期均值则减仓) | `five_force` 模块 | 7/10 | **HIGH** |
| **Portfolio-Management-ActorCriticRL** | RL组合管理 | A2C/DDPG/PPO三种RL算法; Buy&Hold对比 | `rl/env_stocktrading_llm_risk.py` - RL训练框架 | 7/10 | **HIGH** |
| **crypto_hedge_scalping_bot** | 加密对冲/网格风控 | 5网格对冲(4多+1空); 无止损但空头对冲实现盈亏平衡 | `maverick_risk.py` 止盈止损模块 | 7/10 | MEDIUM |
| **KalmanBOT_ICASSP23/HedgeRatio** | 配对交易/对冲比率 | Kalman滤波估算对冲比率; 统计套利 | 对冲策略模块 | 7/10 | MEDIUM |
| **kiwano_portfolio** | 组合管理框架 | Backtest/Livetest/Livetrade三模式; 组合+策略双层架构 | 组合管理模块 | 6/10 | MEDIUM |
| **portfolio-backtester** | 组合分析/风控指标 | Sharpe/Sortino/Max Drawdown; 定期Rebalance | `risk_metrics.py` - 已含类似实现 | 6/10 | MEDIUM |
| **Ctrl-Alt-DefeatTheMarket/risk** | 交易风控/头寸管理 | 仓位清理; 库存调整; 20-lot防御性报价; Sharpe/Drawdown | 风控规则引擎 | 6/10 | MEDIUM |
| **Learn-Algorithmic-Trading/Chapter6** | 风控指标库 | Sharpe/Sortino/Max Drawdown; 周期性损益分析 | `risk_metrics.py` - 已覆盖 | 5/10 | LOW |
| **HFT-strategies/quant/risk_measure** | 高频风控度量 | 周/月损益分布; Drawdown追踪; 夏普/索提诺 | `risk_metrics.py` - 已覆盖 | 6/10 | LOW |

---

## HIGH 优先级详细分析

### 1. OptionSuite/riskManager
**路径**: `D:/Hive/Data/trading_repos/OptionSuite/`
- **核心**: PutVerticalRiskManagement, StrangleRiskManagement
- **策略类型**: HOLD_TO_EXPIRATION, CLOSE_AT_50_PERCENT, CLOSE_AT_50_PERCENT_OR_21_DAYS, CLOSE_AT_21_DAYS
- **可借鉴**: 持仓周期检测逻辑, Greeks组合风控

### 2. PortfolioStrategyBacktestUS
**路径**: `D:/Hive/Data/trading_repos/PortfolioStrategyBacktestUS/`
- **核心**: Markowitz均值方差, Risk-Parity, SEV选股, 波动率门限(短期std>bound则减仓50%)
- **可借鉴**: 波动率门限风控规则, A股HS300特殊处理

### 3. five_factors_riskControl
**路径**: `D:/Hive/Data/trading_repos/five_factors_riskControl/`
- **核心**: XGBoost月收益预测, 波动率突破风控(HS300短期>长期均值减仓)
- **可借鉴**: 与PortfolioStrategyBacktestUS互补, 针对A股

### 4. Hedge_Fund_Agents
**路径**: `D:/Hive/Data/trading_repos/Hedge_Fund_Agents/`
- **核心**: 9种投资风格Agent, 独立Risk Manager Agent(20%单仓上限)
- **可借鉴**: 多Agent风控架构设计

---

## 现有系统已覆盖的风控模块

| 模块 | 功能 | 评分 |
|------|------|------|
| `risk/risk_metrics.py` | Sharpe, Sortino, VaR, CVaR, Calmar, Max Drawdown, Win Rate, Profit Factor | 9/10 |
| `risk/maverick_risk.py` | Kelly仓位计算, ATR/支撑/摆动低点止损, 组合风险 | 8/10 |
| `risk/matilda_risk.py` | LPM/HPM, Kappa比率, Omega, Rolling Sharpe, Roy's Safety First | 8/10 |
| `options/risk_manager.py` | 期权风险管理 | - |

---

## 吸收建议

**HIGH优先级**:
1. OptionSuite风控策略类型系统(持仓周期管理枚举)
2. PortfolioStrategyBacktestUS波动率门限逻辑
3. five_factors_riskControl的HS300波动率风控

**MEDIUM优先级**:
4. crypto_hedge_scalping_bot的无止损对冲网格逻辑
5. KalmanBOT的对冲比率Kalman滤波

**LOW优先级**:
6. Learn-Algorithmic-Trading的简单风控指标(已覆盖)
7. HFT-strategies的损益分布分析(已覆盖)

---

**分析日期**: 2026-03-30
