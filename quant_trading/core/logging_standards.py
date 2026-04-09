# -*- coding: utf-8 -*-
"""
日志系统规范 - Logging Standards

统一日志级别使用，提升日志可读性和可维护性。
"""

from enum import Enum
from typing import Optional
from .logger import logger


class LogLevel(Enum):
    """日志级别定义"""

    DEBUG = 10      # 调试信息：开发时使用，生产环境关闭
    INFO = 20       # 普通信息：正常操作记录
    WARNING = 30    # 警告信息：需要注意但不影响运行
    ERROR = 40      # 错误信息：操作失败但系统可继续
    CRITICAL = 50   # 严重错误：系统可能无法继续运行


class LoggingStandard:
    """
    Titan V13 日志使用标准

    规范日志级别使用，确保日志清晰有用。
    """

    # ==================== 日志级别使用指南 ====================

    @staticmethod
    def guide() -> str:
        """返回日志使用指南"""
        return """
# Titan V13 日志使用规范

## 日志级别定义

### DEBUG (10) - 调试信息
**用途**: 开发调试，生产环境应关闭
**场景**:
- 变量值查看
- 函数入口/出口跟踪
- 详细计算过程
- 临时调试输出

**示例**:
```python
logger.debug(f"当前持仓: {position}")
logger.debug(f"计算结果: {result}, 耗时: {elapsed}ms")
```

**生产环境**: 应设置为 INFO 级别或更高

---

### INFO (20) - 普通信息 (默认推荐)
**用途**: 正常操作记录，系统运行状态
**场景**:
- ✅ 系统启动/关闭
- ✅ 配置加载成功
- ✅ 连接建立成功
- ✅ 交易订单记录
- ✅ 策略信号生成
- ✅ 定时任务执行
- ✅ 数据保存完成

**示例**:
```python
logger.info("✅ 系统启动完成")
logger.info(f"📊 市场数据 | {symbol}: {price}")
logger.info(f"💰 下单成功 | {symbol} {side} {amount}")
logger.info("✅ 配置加载成功")
```

**何时使用**:
- 所有正常操作
- 用户需要了解的正常流程
- 定期状态记录

---

### WARNING (30) - 警告信息
**用途**: 需要注意但不影响系统运行
**场景**:
- ⚠️ 配置值可能导致问题
- ⚠️ 性能下降（但仍可用）
- ⚠️ 重试操作
- ⚠️ 使用默认值而非配置
- ⚠️ 资源使用偏高
- ⚠️ 功能即将弃用

**示例**:
```python
logger.warning(f"⚠️ 高杠杆警告({lev}x)，建议 ≤ 20x")
logger.warning(f"⚠️ API响应延迟: {latency}ms")
logger.warning("⚠️ 使用默认配置，未找到config.yaml")
logger.warning("⚠️ 数据库连接池接近上限")
```

**何时使用**:
- 不影响功能但需要关注
- 降级服务但仍在运行
- 配置不合理但系统继续

---

### ERROR (40) - 错误信息
**用途**: 操作失败但系统可继续
**场景**:
- ❌ 订单执行失败
- ❌ API调用失败
- ❌ 数据库写入失败
- ❌ 文件读写失败
- ❌ 验证失败

**示例**:
```python
logger.error(f"❌ 订单执行失败: {symbol} - {reason}")
logger.error(f"❌ 数据库连接失败: {error}")
logger.error(f"❌ 配置验证失败: {errors}")
logger.error(f"❌ API调用失败: {url} - {status_code}")
```

**何时使用**:
- 单个操作失败
- 可恢复的错误
- 不需要立即人工干预

---

### CRITICAL (50) - 严重错误
**用途**: 系统可能无法继续
**场景**:
- 🛑 系统崩溃风险
- 🛑 每日亏损超限
- 🛑 熔断器触发
- 🛑 关键依赖缺失
- 🛑 数据库完全不可用

**示例**:
```python
logger.critical("🛑 达到每日亏损限额！系统已停止")
logger.critical("🛑 熔断器触发，停止所有交易")
logger.critical("🛑 数据库连接完全失败")
logger.critical("🛑 关键配置缺失，无法启动")
```

**何时使用**:
- 需要立即人工干预
- 系统无法继续运行
- 可能导致数据丢失或资金损失

---

## 模块特定日志规范

### 交易模块 (modules/executor.py, modules/trading.py)
```python
# ✅ INFO: 正常交易流程
logger.info(f"💰 下单 | {symbol} {side} {amount} @{price}")
logger.info(f"✅ 订单成交 | ID: {order_id}, 价格: {exec_price}")

# ⚠️ WARNING: 交易问题但可继续
logger.warning(f"⚠️ 流动性不足，分批执行: {chunks}批")
logger.warning(f"⚠️ 滑点较大: {slippage}%")

# ❌ ERROR: 订单失败
logger.error(f"❌ 订单失败 | {symbol} {side} - {reason}")
logger.error(f"❌ 余额不足 | 需要 {required}, 可用 {available}")
```

### 风险管理模块 (modules/risk.py)
```python
# ✅ INFO: 风险检查通过
logger.info(f"✅ 风险检查通过 | {symbol} {side}")

# ⚠️ WARNING: 风险警告
logger.warning(f"⚠️ 杠杆过高 | {current}x > {limit}x")
logger.warning(f"⚠️ 接近日亏限额 | {current}% / {max}%")

# 🛑 CRITICAL: 触发风控
logger.critical(f"🛑 日亏限额触发 | {loss}% > {max}%")
logger.critical(f"🛑 熔断器激活 | 停止所有交易")
```

### 数据库模块 (core/database.py)
```python
# ✅ INFO: 数据库操作成功
logger.info("✅ PostgreSQL 连接池已创建")
logger.info(f"✅ 交易已记录 | ID: {trade_id}")
logger.info("✅ 数据库表结构已初始化")

# ⚠️ WARNING: 数据库警告
logger.warning(f"⚠️ 连接池使用率偏高 | {size}/{max}")
logger.warning("⚠️ 查询耗时较长 | 1000ms+")

# ❌ ERROR: 数据库错误
logger.error(f"❌ 查询执行失败 | {error}")
logger.error(f"❌ 事务回滚 | {reason}")
```

### 市场数据模块 (modules/market.py)
```python
# ✅ INFO: 市场数据更新
logger.info(f"📊 行情更新 | {symbol}: {price}")

# ⚠️ WARNING: 数据质量警告
logger.warning(f"⚠️ 陈旧数据 | {symbol} 延迟 {age}ms")
logger.warning("⚠️ 交易所API延迟 | 500ms+")

# ❌ ERROR: 数据获取失败
logger.error(f"❌ 行情获取失败 | {symbol}: {error}")
```

---

## 日志格式规范

### 推荐格式
```python
# 使用表情符号标识类型
logger.info(f"✅ 成功描述 | 关键信息")
logger.warning(f"⚠️ 警告描述 | 详细信息")
logger.error(f"❌ 错误描述 | 失败原因")

# 使用竖线分隔多个信息
logger.info(f"💰 下单 | {symbol} | {side} | {amount} | @{price}")

# 使用冒号表示层级
logger.info(f"✅ 模块: {message}")
logger.info(f"  └─ 子操作: {detail}")
```

### 避免的格式
```python
# ❌ 避免：过于简略
logger.debug("ok")
logger.info("done")

# ❌ 避免：无意义的DEBUG
logger.debug(f"var = {var}")  # 生产环境不应有

# ❌ 避免：混乱的表情符号
logger.info("🔥💰🚀💎 order executed")

# ❌ 避免：过长的堆栈跟踪（除非调试）
import traceback
logger.error(traceback.format_exc())  # 使用简短描述即可
```

---

## 性能考虑

### 日志级别与性能
- **DEBUG**: 每条日志 ~0.1ms，仅开发时使用
- **INFO**: 每条日志 ~0.05ms，生产环境推荐
- **WARNING**: 每条日志 ~0.05ms
- **ERROR**: 每条日志 ~0.1ms，包含错误详情

### 高频日志优化
```python
# ❌ 避免：高频DEBUG日志
async def process_ticks():
    for tick in ticks:
        logger.debug(f"Tick: {tick}")  # 每秒可能数百条

# ✅ 推荐：定期统计
async def process_ticks():
    count = 0
    for tick in ticks:
        count += 1
        if count % 1000 == 0:
            logger.info(f"📊 已处理 {count} 个tick")
```

---

## 结构化日志（可选扩展）

### JSON格式日志（未来）
```python
import json

class StructuredLogger:
    """结构化日志（可选）"""

    def log(self, level, event, **kwargs):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'event': event,
            **kwargs
        }
        logger.info(json.dumps(log_entry))

# 使用
logger.log('INFO', 'ORDER_PLACED',
          symbol='ETH/USDT',
          side='buy',
          amount=100)
```

---

## 最佳实践总结

1. **生产环境使用 INFO 级别**
   - 正常操作：INFO
   - 需要关注：WARNING
   - 操作失败：ERROR
   - 系统威胁：CRITICAL

2. **日志消息清晰有用**
   - 包含关键参数（symbol, price, amount）
   - 使用表情符号区分类型
   - 简洁但完整

3. **避免日志污染**
   - 不在高频循环中记录 DEBUG
   - 不过度记录成功信息（定期统计）
   - 不在异常中打印完整堆栈

4. **异步日志注意**
   - 使用异步日志库（如果需要）
   - 避免阻塞事件循环
   - 批量写入提升性能

---

**标准版本**: v1.0
**适用系统**: Titan V13
**生效日期**: 2026-02-12
"""

    # ==================== 日志装饰器 ====================

    @staticmethod
    def log_entry_exit(func):
        """记录函数入口和出口（仅用于DEBUG）"""
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if logger.level <= LogLevel.DEBUG.value:
                logger.debug(f"→ 进入 {func.__name__}")
            result = await func(*args, **kwargs)
            if logger.level <= LogLevel.DEBUG.value:
                logger.debug(f"← 离开 {func.__name__}")
            return result
        return wrapper

    # ==================== 日志验证 ====================

    @staticmethod
    def validate_log_level(level: str) -> LogLevel:
        """
        验证日志级别是否有效

        Args:
            level: 日志级别字符串 (DEBUG/INFO/WARNING/ERROR/CRITICAL)

        Returns:
            LogLevel 枚举值

        Raises:
            ValueError: 无效的日志级别
        """
        level_map = {
            'DEBUG': LogLevel.DEBUG,
            'INFO': LogLevel.INFO,
            'WARNING': LogLevel.WARNING,
            'ERROR': LogLevel.ERROR,
            'CRITICAL': LogLevel.CRITICAL
        }

        if level.upper() not in level_map:
            raise ValueError(f"无效的日志级别: {level}")

        return level_map[level.upper()]


# ==================== 使用示例 ====================

def example_usage():
    """日志使用示例"""

    # ✅ 正确：正常操作使用 INFO
    logger.info("✅ 系统启动完成")
    logger.info("📊 市场数据获取成功")

    # ⚠️ 正确：需要注意使用 WARNING
    logger.warning("⚠️ 使用默认配置")
    logger.warning("⚠️ API延迟较高")

    # ❌ 错误：使用 ERROR
    logger.error("❌ 订单执行失败")

    # 🛑 严重：使用 CRITICAL
    logger.critical("🛑 系统无法继续运行")

    # ❌ 避免：正常操作使用 DEBUG
    logger.debug("配置加载完成")  # 应该用 INFO
