# Production Trading Guide

> **量化之神** V36策略生产环境实盘交易指南

## 目录

1. [概述](#概述)
2. [安全检查清单](#安全检查清单)
3. [配置步骤](#配置步骤)
4. [启动交易](#启动交易)
5. [监控面板](#监控面板)
6. [紧急程序](#紧急程序)
7. [熔断机制](#熔断机制)
8. [常见问题](#常见问题)

---

## 概述

本文档说明如何在生产环境中安全运行V36量化交易策略。

### 交易模式

| 模式 | 说明 | 风险等级 |
|------|------|---------|
| `dry_run` | 干跑模式，仅监控和信号，不交易 | 零风险 |
| `paper` | Paper交易，模拟执行，不真实下单 | 极低 |
| `live` | 实盘交易，真金白银 | 高 |

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    EmergencyStopSystem                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Circuit     │  │ Kill        │  │ Alert       │          │
│  │ Breaker     │  │ Switch      │  │ Manager     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     RiskManager                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Position    │  │ Loss        │  │ Trade       │          │
│  │ Limits       │  │ Limits       │  │ Approval    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   LiveTradingEngine                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ V36Signal   │  │ Executor     │  │ Dashboard   │          │
│  │ Generator   │  │              │  │             │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Binance                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 安全检查清单

在启动实盘交易前，必须完成以下所有检查项：

### 必需的环境变量

```bash
# API凭证 (必需)
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_api_secret_here"

# 生产模式 (必需)
export PRODUCTION_MODE="true"

# 可选: 告警通知
export TELEGRAM_BOT_TOKEN="your_telegram_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."

# 数据库路径 (可选)
export TRADING_DB_PATH="data/production_trading.db"
```

### 配置验证

- [ ] API密钥已设置且有效
- [ ] 生产模式已启用 (`PRODUCTION_MODE=true`)
- [ ] 风险限制已确认
- [ ] 告警通知已配置 (强烈推荐)
- [ ] 数据库路径已确认
- [ ] 日志路径已确认

### 风险参数确认

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|---------|------|
| `max_position_size` | $1000 | $500-$2000 | 单币种最大持仓 |
| `max_total_exposure` | 60% | 40%-70% | 总仓位上限 |
| `max_daily_loss` | $500 | 资本5% | 日最大亏损 |
| `max_single_loss` | $100 | 资本1% | 单笔最大亏损 |
| `circuit_break_after_losses` | 3 | 3-5 | 熔断连续亏损次数 |
| `stop_loss_pct` | -3% | -3%~-5% | 止损比例 |
| `take_profit_pct` | +7% | +5%~+10% | 止盈比例 |

---

## 配置步骤

### 1. 创建Binance API密钥

1. 登录 [Binance](https://www.binance.com)
2. 进入 API Management
3. 创建新API密钥
4. 设置权限: **只开启 futures 交易权限**
5. 保存 API Key 和 Secret

> **安全警告**: 不要开启提现权限!

### 2. 设置环境变量

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加:

export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export PRODUCTION_MODE="true"

# 可选: Telegram通知
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF"
export TELEGRAM_CHAT_ID="987654321"
```

### 3. 配置Telegram告警 (推荐)

1. 在 Telegram 中搜索 @BotFather
2. 创建新机器人，保存 token
3. 在 Telegram 中搜索 @userinfobot 获取你的 Chat ID

### 4. 验证配置

```bash
cd D:/量化交易系统/量化之神/quant_trading
python -c "from config.production import get_production_config, print_config_summary; print_config_summary(get_production_config())"
```

---

## 启动交易

### 模式1: 干跑模式 (推荐首先)

不真实交易，仅监控信号:

```bash
cd D:/量化交易系统/量化之神/quant_trading
python execution/run_live_trading.py --mode dry_run --symbols BTCUSDT ETHUSDT --capital 10000
```

### 模式2: Paper交易

模拟交易，使用真实市场数据:

```bash
python execution/run_live_trading.py --mode paper --symbols BTCUSDT --capital 10000
```

### 模式3: 实盘交易 (最后一步)

```bash
# 确保已完成所有安全检查清单
python execution/run_live_trading.py --mode live --symbols BTCUSDT --capital 10000
```

### 带审批的大额交易模式

```bash
python execution/run_live_trading.py --mode live --approval-required --large-trade-threshold 500 --symbols BTCUSDT
```

---

## 监控面板

运行后会显示实时Dashboard:

```
======================================================================
  量化之神 Production Trading Dashboard  |  2026-04-10 12:00:00
======================================================================

  [账户]  初始: $10,000.00  |  当前: $10,250.00  |  回报: +2.50%
  [P&L]   日盈亏: +$50.00  |  总盈亏: +$250.00  |  最大回撤: $100.00
  [交易]  总交易: 15  |  胜率: 60.0%  |  连续胜/负: 3/1

  [持仓]  (2/4)
    BTCUSDT: LONG 0.05 @ $50,000  |  P&L: +$25.00 (+0.50%)
    ETHUSDT: LONG 0.3 @ $3,000  |  P&L: +$15.00 (+0.50%)

----------------------------------------------------------------------
```

### 日志文件

所有交易记录保存在:
- 数据库: `data/production_trading.db`
- 日志: `logs/production_trading.log`

---

## 紧急程序

### 立即停止所有交易

**方法1**: 在运行终端按 `Ctrl+C`

**方法2**: 使用Kill Switch (如果已武装):

```python
# 在Python中
from execution.emergency_stop import EmergencyStopSystem
emergency = EmergencyStopSystem(config, executor, alert_manager)
emergency.arm_kill_switch()
# 输入 "KILL" 命令将平仓
emergency.execute_kill_switch("Manual stop")
```

### 恢复交易

1. 分析触发原因
2. 检查市场状况
3. 修复问题
4. 重启交易引擎

```bash
# 重启
python execution/run_live_trading.py --mode live --symbols BTCUSDT
```

### 常见紧急情况

| 情况 | 处理方式 |
|------|---------|
| 熔断触发 | 检查异常日志，等待冷却后自动恢复 |
| Kill Switch | 分析原因，手动确认后重启 |
| API连接失败 | 检查网络，查看日志重试 |
| 价格异常波动 | 等待市场稳定，监控系统 |

---

## 熔断机制

### 自动熔断触发条件

1. **价格异常**: 5分钟内价格变化超过5%
2. **连续亏损**: 3次连续亏损
3. **日亏损超限**: 日亏损超过$500
4. **API限流**: 请求超过1200次/分钟
5. **连接失败**: 连续3次API调用失败

### 熔断后的行为

1. 自动平仓所有持仓
2. 发送告警通知
3. 停止开新仓位
4. 等待冷却期 (60秒)
5. 自动进入恢复模式

### 查看熔断记录

```bash
sqlite3 data/production_trading.db
sqlite> SELECT * FROM circuit_trips;
```

---

## 回测验证

在实盘交易前，必须运行生产回测:

```bash
cd D:/量化交易系统/量化之神/quant_trading
python experiments/production_backtest.py --symbols BTCUSDT ETHUSDT --start 2024-01-01 --end 2024-12-31
```

### 准入标准

| 指标 | 要求 | 说明 |
|------|------|------|
| 胜率 | > 55% | 历史胜率 |
| 盈亏比 | > 1.3 | 平均盈利/平均亏损 |
| 最大回撤 | < 15% | 历史最大回撤 |
| 夏普比率 | > 1.0 | 风险调整后收益 |
| 交易次数 | > 10 | 足够的样本 |

---

## 常见问题

### Q: 熔断触发了怎么办?

A: 检查 `logs/production_trading.log` 中的异常记录。常见原因:
- 市场剧烈波动
- 网络连接不稳定
- 策略连续亏损

### Q: 如何调整风险参数?

A: 编辑 `config/production.py` 或设置环境变量:

```bash
export MAX_POSITION_SIZE=2000
export MAX_DAILY_LOSS=1000
```

### Q: 可以同时运行多个交易对吗?

A: 可以，但建议先从单个交易对开始:

```bash
python execution/run_live_trading.py --mode paper --symbols BTCUSDT
```

确认稳定后再增加:

```bash
python execution/run_live_trading.py --mode paper --symbols BTCUSDT ETHUSDT SOLUSDT
```

### Q: Kill Switch如何工作?

A:
1. 需要先武装: `emergency.arm_kill_switch()`
2. 执行终止: 调用 `execute_kill_switch()`
3. 系统会平掉所有持仓并发送告警

### Q: 如何备份交易记录?

A: 数据库已自动保存所有交易:

```bash
cp data/production_trading.db data/production_trading_$(date +%Y%m%d).db
```

---

## 联系方式

- 紧急情况: 检查日志 `logs/production_trading.log`
- 技术支持: 参考代码注释或提交Issue

---

*最后更新: 2026-04-10*
