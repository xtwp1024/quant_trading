# LLM/Agent/Autonomous AI 仓库吸收分析报告

**扫描日期**: 2026-03-30
**数据源**: `D:/Hive/Data/trading_repos/`
**匹配关键词**: llm, gpt, langchain, llamaindex, agent, autonomous, chatgpt, claude, gemini, openai, agentic
**匹配仓库总数**: 18个

---

## 仓库汇总表

| 仓库名 | 架构类型 | 核心功能 | 互补模块 | 代码质量 | 优先级 |
|--------|----------|----------|----------|----------|--------|
| **Fully-Autonomous-Polymarket-AI-Trading-Bot** | Multi-Model Ensemble (GPT-4o/Claude/Gemini) | Polymarket预测，15+风险检查，fractional Kelly，多模型校准 | `agent` 风险控制, decision校准 | 9.5/10 | **HIGH** |
| **LLM-TradeBot** | Multi-Agent + Adversarial Decision | 加密期货做市，多智能体协作(DECISION/RISK/REFLECTION)，四层策略过滤器 | `agent.decision`, `agent.reflection` | 9.0/10 | **HIGH** |
| **TradingAgents-AShare** | Multi-Agent (15个专业Agent) | A股智能投研，多空辩论，红蓝对抗，Docker部署 | `agent.debate`, `research` | 8.5/10 | **HIGH** |
| **AgentQuant** | LLM-driven Research (Gemini 2.5) | 自主量化研究，Walk-Forward验证，regime检测，消融实验 | `agent.planner`, `agent.research` | 8.5/10 | **HIGH** |
| **FinMem-LLM-StockTrading** | LLM + Layered Memory | 分层记忆架构(ICLR/AAAI论文)，Profiling/Memory/Decision | `agent.memory`, `agent.decision` | 8.5/10 | **HIGH** |
| **Hedge_Fund_Agents** | Multi-Agent (Buffett/Graham/Ackman风格) | 模拟传奇投资者风格，多角度分析 | `agent.diversity` | 8.0/10 | MEDIUM |
| **alpha-agent** | Multi-Agent + 101 Alphas | Alpha因子选择，LSTM/XGBoost预测 | `factors.alpha_selection` | 8.0/10 | MEDIUM |
| **ai-broker-investing-agent** | Multi-Agent (Bull vs Bear Debate) | 股票+预测市场，Alpaca/TradingView集成 | `agent.debate` | 8.0/10 | MEDIUM |
| **Stockagent** | Multi-Agent LLM Framework | 真实世界模拟，防泄露，外部因素分析 | `agent.simulation` | 8.0/10 | MEDIUM |
| **PolyMarket-AI-agent-trading** | LangChain + RAG + FastAPI | Polymarket AI交易，RAG增强研究 | `agent.research_augmented` | 7.5/10 | MEDIUM |
| **LLM-Trading-Lab** | Single LLM (ChatGPT) | 6个月实盘实验，40页评估论文 | `agent.evaluation` | 7.5/10 | MEDIUM |
| **deep-q-trading-agent** | Deep Q-Learning + Transfer | RL交易，NumQ/NumDReg架构 | `rl` | 7.0/10 | LOW |
| **LLM-Stock-Manager** | GPT-3.5 + Real-time | TradingView webhook，邮件告警 | `data.pipeline` | 6.5/10 | LOW |
| **ChatGPT-Trading-Bot** | GPT-3.5 + TradingView | PineScript生成，Heroku部署 | `strategy.code_generation` | 6.0/10 | LOW |
| **ChatGPT-Trading-Bot-for-KuCoin** | GPT-3.5 + KuCoin | 价格预测，ccxt集成 | `execution.exchange` | 5.5/10 | LOW |
| **Trading-Bot-AI-ChatGPT** | GPT + ML (RF/NN/LSTM) | 多模型预测，币安现货 | `model.ensemble` | 6.0/10 | LOW |
| **trading-bot-gpt** | Minimal GPT | 基础GPT bot | - | 4.0/10 | LOW |
| **StockGPT-Public** | GPT-based Q&A | 股票投顾问答 | `agent.qa` | 5.0/10 | LOW |

---

## HIGH 优先级详细分析

### 1. Fully-Autonomous-Polymarket-AI-Trading-Bot
**路径**: `D:/Hive/Data/trading_repos/Fully-Autonomous-Polymarket-AI-Trading-Bot/`
- 多模型集成: GPT-4o 40% / Claude 3.5 Sonnet 35% / Gemini 1.5 Pro 25%
- 15+独立风险检查机制
- Fractional Kelly仓位管理 (7个调整因子)
- 三重dry-run安全门，SQLite WAL + SHA-256审计

### 2. LLM-TradeBot
**路径**: `D:/Hive/Data/trading_repos/LLM-TradeBot/`
- 对抗决策框架: SymbolSelector -> DataSync -> QuantAnalyst -> RegimeDetector -> DecisionCore -> RiskAudit
- 四层策略过滤器 (Trend -> AI Filter -> Setup -> Trigger)
- 支持8个LLM厂商 (DeepSeek/OpenAI/Claude/Qwen/Gemini/Kimi/MiniMax/GLM)

### 3. TradingAgents-AShare
**路径**: `D:/Hive/Data/trading_repos/TradingAgents-AShare/`
- 15个专业智能体: 6分析师 + 2研究员 + 3风控 + 4交易员
- 红蓝对抗辩论机制 (Claim-driven structured debate)
- Docker一键部署

### 4. AgentQuant
**路径**: `D:/Hive/Data/trading_repos/AgentQuant/`
- Gemini 2.5 Flash驱动自主量化研究
- Walk-Forward验证 + 消融实验
- 市场regime检测 (Bull/Bear/Crisis)

### 5. FinMem-LLM-StockTrading
**路径**: `D:/Hive/Data/trading_repos/FinMem-LLM-StockTrading/`
- 分层记忆架构: Profiling/Memory/Decision
- 可调认知跨度
- ICLR Workshop / AAAI论文实现

---

## 吸收建议

### 高优先级吸收模块
1. **多模型集成框架** - 三模型并行 + Platt calibration → `agent.multi_llm_ensemble`
2. **对抗辩论架构** - Bull/Bear多角度 → `agent.debate`
3. **分层记忆系统** - Profiling/Memory/Decision → `agent.finmem_layer`
4. **自主量化研究** - Walk-Forward验证 → `agent.research`

---

**分析日期**: 2026-03-30
