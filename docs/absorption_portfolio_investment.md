# Portfolio/Investment/资产管理仓库吸收报告

**扫描日期**: 2026-03-30
**数据源**: `D:/Hive/Data/trading_repos/`
**匹配关键词**: portfolio, investment, asset-allocation, markowitz, rebalance, 组合, 资产管理, mean-variance, efficient-frontier

---

## 仓库分析汇总表

| 仓库名 | 优化方法 | 核心功能 | 互补模块 | 代码质量 | 优先级 |
|--------|----------|----------|----------|----------|--------|
| **FinRL-DAPO-SR** | DAPO (PPO变体) + LLM情感/风险信号 | DRL股票交易, 2025 IEEE FinRL Contest第2名 | `rl/`, `signal/`, `factors/` | 8/10 | **HIGH** |
| **PortfolioStrategyBacktestUS** | Vast Portfolio Selection + Markowitz + Momentum | 大规模资产筛选+组合配置回测 (SP500 2018-2020) | `backtest/` 因子选股, `risk/` HRP | 7.5/10 | **HIGH** |
| **Markowitzify** | NCO(非凸/全局搜索), Markowitz, Hurst, Monte Carlo | 轻量级组合优化库 + 技术分析 | `risk/portfolio_optimizer.py` 全局优化 | 7/10 | MEDIUM |
| **hiquant** | MACD/SMA/KDJ指标, 基金评估 | A股/港股/美股数据采集, 量化框架 | `data/` A股数据 | 6.5/10 | MEDIUM |
| **Portfolio-Management-ActorCriticRL** | A2C, DDPG, PPO | DRL组合管理 vs Buy&Hold | `rl/` DRL参考 | 6.5/10 | MEDIUM |
| **ws-rebalancer** | 迭代目标再平衡 | WealthSimple组合再平衡CLI | `risk/` 再平衡 | 7/10 | LOW |
| **portfolio-backtester** | 指标信号+定期再平衡 | 股票组合回测工具 | `backtest/`, `risk/` | 6/10 | LOW |
| **kiwano_portfolio** | 双层策略架构 | Binance加密货币组合管理 | `exchanges/` | 5.5/10 | LOW |
| **PortfolioTrackingIntro** | 配置驱动IB API执行 | MySQL组合追踪+自动化交易 | `execution/` | 5/10 | LOW |
| **AI4Finance-Foundation** | DQN, DDPG (已被FinRL超越) | DRL股票交易 (NeurIPS 2018) | `rl/` 参考(已过时) | 5/10 | LOW |

---

## HIGH 优先级详细分析

### 1. FinRL-DAPO-SR
**路径**: `D:/Hive/Data/trading_repos/FinRL-DAPO-SR/`
- **核心**: DAPO算法(2025 PPO改进) + LLM情感/风险信号
- **亮点**: DAPO比标准Actor-Critic更适合组合交易, LLM信号集成

### 2. PortfolioStrategyBacktestUS
**路径**: `D:/Hive/Data/trading_repos/PortfolioStrategyBacktestUS/`
- **核心**: Vast Portfolio Selection (回归法) + Markowitz Mean-Variance
- **亮点**: 两阶段框架(大规模筛选→精选组合配置), 解决高维协方差估计问题

---

## 现有系统已覆盖

| 模块 | 能力 |
|------|------|
| `risk/portfolio_optimizer.py` | Markowitz, Max Sharpe, Min Variance, Risk Parity, Black-Litterman |
| `risk/matilda_portfolio.py` | HRP (Lopez de Prado), PMPT (Sortino), Efficient Frontier |
| `risk/risk_metrics.py` | Sharpe, Sortino, VaR/CVaR, Calmar, Max Drawdown |
| `portfolio/optimizers.py` | MeanVariance, HRP, BlackLitterman, CVaR, RiskParity |

---

## 关键发现

1. **已有扎实Markowitz基础**: Mean-Variance, HRP, PMPT, Black-Litterman, Efficient Frontier均已实现
2. **最稀缺: DRL组合优化**: 现有 `rl/` 偏通用交易, 缺乏专门的DRL组合权重优化
3. **大规模候选集筛选薄弱**: Vast Portfolio Selection的回归法可补充高维筛选
4. **A股/港股数据缺失**: hiquant提供了补充

---

## 吸收建议

### HIGH
1. **FinRL-DAPO-SR** → DAPO算法 + LLM信号 → `rl/`, `factors/`
2. **PortfolioStrategyBacktestUS** → Vast Portfolio Selection → `backtest/` 选股模块

### MEDIUM
3. **Markowitzify** → NCO全局优化 → `risk/portfolio_optimizer.py`
4. **hiquant** → A股/港股数据 → `data/`

---

**分析日期**: 2026-03-30
