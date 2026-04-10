# -*- coding: utf-8 -*-
"""
Emergency Stop System
====================

紧急止损系统 - 监控异常、自动平仓、告警通知

功能:
1. 异常监控 - 价格异常、成交量异常、连接异常
2. 自动平仓 - 触发熔断时自动平掉所有持仓
3. Kill Switch - 手动终止交易的开关
4. 告警通知 - Telegram/Slack/邮件通知

Usage:
    emergency = EmergencyStopSystem(config, executor, alert_manager)
    emergency.start()
"""

import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import threading
import json

logger = logging.getLogger("EmergencyStop")


class EmergencyLevel(Enum):
    """紧急级别"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CircuitState(Enum):
    """熔断状态"""
    NORMAL = "normal"
    TRIPPED = "tripped"
    RECOVERING = "recovering"


@dataclass
class AnomalyEvent:
    """异常事件"""
    timestamp: int
    level: EmergencyLevel
    event_type: str
    symbol: str
    details: str
    value: float
    threshold: float


@dataclass
class PositionSnapshot:
    """持仓快照"""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    pnl_pct: float


class AlertManager:
    """告警管理器"""

    def __init__(self, config):
        self.config = config
        self.telegram_enabled = config.alerts.telegram_enabled
        self.slack_enabled = config.alerts.slack_enabled

    def send_alert(
        self,
        level: EmergencyLevel,
        title: str,
        message: str,
        data: Dict = None
    ) -> None:
        """发送告警"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{level.value.upper()}] {title}\n{message}"

        if data:
            full_message += f"\n\nData:\n{json.dumps(data, indent=2)}"

        logger.log(
            logging.ERROR if level in [EmergencyLevel.CRITICAL, EmergencyLevel.EMERGENCY]
            else logging.WARNING,
            full_message
        )

        # 发送Telegram
        if self.telegram_enabled:
            self._send_telegram(full_message)

        # 发送Slack
        if self.slack_enabled:
            self._send_slack(level, title, message, data)

    def _send_telegram(self, message: str) -> None:
        """发送Telegram消息"""
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.config.alerts.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.config.alerts.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Telegram发送失败: {e}")

    def _send_slack(self, level: EmergencyLevel, title: str, message: str, data: Dict) -> None:
        """发送Slack消息"""
        try:
            import requests
            color = {
                EmergencyLevel.NORMAL: "good",
                EmergencyLevel.WARNING: "warning",
                EmergencyLevel.CRITICAL: "danger",
                EmergencyLevel.EMERGENCY: "danger"
            }.get(level, "warning")

            payload = {
                "attachments": [{
                    "color": color,
                    "title": title,
                    "text": message,
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in (data or {}).items()
                    ],
                    "ts": time.time()
                }]
            }
            requests.post(self.config.alerts.slack_webhook_url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Slack发送失败: {e}")


class EmergencyStopSystem:
    """
    紧急止损系统

    监控并响应:
    - 价格剧烈波动
    - 连续亏损
    - 异常交易量
    - API连接问题
    - 风控阈值突破
    """

    def __init__(
        self,
        config: Any,
        executor: Any,
        alert_manager: AlertManager = None
    ):
        self.config = config
        self.executor = executor
        self.alerts = alert_manager or AlertManager(config)

        # 状态
        self.circuit_state = CircuitState.NORMAL
        self.emergency_active = False
        self.kill_switch_armed = False

        # 监控数据
        self.anomaly_log: List[AnomalyEvent] = []
        self.circuit_trip_count = 0
        self.last_circuit_trip_time = 0

        # 持仓快照
        self.positions: Dict[str, PositionSnapshot] = {}

        # 回调函数
        self.on_emergency: Optional[Callable] = None
        self.on_circuit_trip: Optional[Callable] = None

        # 线程锁
        self._lock = threading.Lock()

        # 数据库
        self.db_path = getattr(config, 'database_path', 'data/emergency_stop.db')
        self._init_database()

        # 配置
        self.circuit_config = config.risk.circuit_breaker
        self.emergency_config = config.risk.emergency_stop

        logger.info("EmergencyStopSystem 初始化完成")
        logger.info(f"  Circuit Breaker: {self.circuit_config.enabled}")
        logger.info(f"  Emergency Stop: {self.emergency_config.enabled}")
        logger.info(f"  Kill Switch: {self.emergency_config.kill_switch_enabled}")

    def _init_database(self) -> None:
        """初始化数据库"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                level TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT,
                details TEXT,
                value REAL,
                threshold REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS circuit_trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                reason TEXT NOT NULL,
                positions_closed INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emergency_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT,
                actions_taken TEXT
            )
        """)
        conn.commit()
        conn.close()

    def record_anomaly(
        self,
        level: EmergencyLevel,
        event_type: str,
        symbol: str,
        details: str,
        value: float,
        threshold: float
    ) -> None:
        """记录异常事件"""
        event = AnomalyEvent(
            timestamp=int(time.time() * 1000),
            level=level,
            event_type=event_type,
            symbol=symbol,
            details=details,
            value=value,
            threshold=threshold
        )

        with self._lock:
            self.anomaly_log.append(event)

            # 只保留最近1000条
            if len(self.anomaly_log) > 1000:
                self.anomaly_log = self.anomaly_log[-1000:]

        # 保存到数据库
        self._save_anomaly_event(event)

        # 发送告警
        if level in [EmergencyLevel.CRITICAL, EmergencyLevel.EMERGENCY]:
            self.alerts.send_alert(
                level,
                f"异常事件: {event_type}",
                details,
                {
                    "symbol": symbol,
                    "value": f"{value:.4f}",
                    "threshold": f"{threshold:.4f}"
                }
            )

    def _save_anomaly_event(self, event: AnomalyEvent) -> None:
        """保存异常事件到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO anomaly_events
                (timestamp, level, event_type, symbol, details, value, threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp,
                event.level.value,
                event.event_type,
                event.symbol,
                event.details,
                event.value,
                event.threshold
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"保存异常事件失败: {e}")

    def update_positions(self, positions: Dict[str, PositionSnapshot]) -> None:
        """更新持仓快照"""
        with self._lock:
            self.positions = positions.copy()

    def check_price_anomaly(
        self,
        symbol: str,
        current_price: float,
        previous_price: float,
        threshold_pct: float = 0.05
    ) -> bool:
        """
        检查价格异常

        Args:
            symbol: 交易对
            current_price: 当前价格
            previous_price: 之前价格
            threshold_pct: 阈值百分比

        Returns:
            True if anomaly detected
        """
        if previous_price == 0:
            return False

        change_pct = abs(current_price - previous_price) / previous_price

        if change_pct > threshold_pct:
            self.record_anomaly(
                EmergencyLevel.CRITICAL if change_pct > threshold_pct * 2
                else EmergencyLevel.WARNING,
                "PRICE_SPIKE",
                symbol,
                f"价格变化 {change_pct:.2%} 超过阈值 {threshold_pct:.2%}",
                change_pct,
                threshold_pct
            )
            return True

        return False

    def check_volume_anomaly(
        self,
        symbol: str,
        current_volume: float,
        average_volume: float,
        threshold_multiplier: float = 5.0
    ) -> bool:
        """
        检查成交量异常

        Args:
            symbol: 交易对
            current_volume: 当前成交量
            average_volume: 平均成交量
            threshold_multiplier: 阈值倍数

        Returns:
            True if anomaly detected
        """
        if average_volume == 0:
            return False

        ratio = current_volume / average_volume

        if ratio > threshold_multiplier:
            self.record_anomaly(
                EmergencyLevel.WARNING,
                "VOLUME_SPIKE",
                symbol,
                f"成交量是平均的 {ratio:.1f}倍",
                ratio,
                threshold_multiplier
            )
            return True

        return False

    def check_loss_limits(self, daily_pnl: float, consecutive_losses: int) -> EmergencyLevel:
        """
        检查亏损限制

        Args:
            daily_pnl: 日盈亏
            consecutive_losses: 连续亏损次数

        Returns:
            EmergencyLevel indicating current risk level
        """
        max_daily_loss = self.config.risk.loss.max_daily_loss
        max_consecutive = self.config.risk.loss.max_consecutive_losses

        # 检查连续亏损
        if consecutive_losses >= max_consecutive:
            self.record_anomaly(
                EmergencyLevel.EMERGENCY,
                "CONSECUTIVE_LOSSES",
                "N/A",
                f"连续亏损 {consecutive_losses} 次达到限制",
                consecutive_losses,
                max_consecutive
            )
            return EmergencyLevel.EMERGENCY

        # 检查日亏损
        if abs(daily_pnl) > max_daily_loss:
            self.record_anomaly(
                EmergencyLevel.CRITICAL,
                "DAILY_LOSS_LIMIT",
                "N/A",
                f"日亏损 {daily_pnl:.2f} 超过限制 {max_daily_loss:.2f}",
                abs(daily_pnl),
                max_daily_loss
            )
            return EmergencyLevel.CRITICAL

        # 检查接近限制
        loss_ratio = abs(daily_pnl) / max_daily_loss if max_daily_loss > 0 else 0
        if loss_ratio > 0.8:
            return EmergencyLevel.WARNING

        return EmergencyLevel.NORMAL

    def check_circuit_breaker(self) -> bool:
        """
        检查是否应该触发熔断

        Returns:
            True if circuit breaker should trip
        """
        if not self.circuit_config.enabled:
            return False

        if self.circuit_state == CircuitState.TRIPPED:
            # 检查是否在冷却期
            cooldown = self.circuit_config.retry_cooldown_seconds
            if time.time() - self.last_circuit_trip_time < cooldown:
                return False
            # 进入恢复期
            self.circuit_state = CircuitState.RECOVERING
            logger.info("Circuit Breaker 进入恢复期")
            return False

        # 检查最近异常事件
        recent_anomalies = [
            e for e in self.anomaly_log
            if e.timestamp > (time.time() - 300) * 1000  # 最近5分钟
            and e.level in [EmergencyLevel.CRITICAL, EmergencyLevel.EMERGENCY]
        ]

        if len(recent_anomalies) >= self.circuit_config.max_retry_attempts:
            self._trip_circuit_breaker(
                f"5分钟内 {len(recent_anomalies)} 次严重异常"
            )
            return True

        return False

    def _trip_circuit_breaker(self, reason: str) -> None:
        """触发熔断"""
        logger.warning(f"CIRCUIT BREAKER TRIPPED: {reason}")

        self.circuit_state = CircuitState.TRIPPED
        self.circuit_trip_count += 1
        self.last_circuit_trip_time = time.time()

        # 保存记录
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO circuit_trips (timestamp, reason)
                VALUES (?, ?)
            """, (int(time.time() * 1000), reason))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"保存熔断记录失败: {e}")

        # 发送告警
        if self.config.risk.circuit_breaker.enabled:
            self.alerts.send_alert(
                EmergencyLevel.EMERGENCY,
                "熔断触发!",
                reason,
                {
                    "circuit_trip_count": self.circuit_trip_count,
                    "positions_affected": len(self.positions)
                }
            )

        # 触发回调
        if self.on_circuit_trip:
            try:
                self.on_circuit_ttrip(reason)
            except Exception as e:
                logger.error(f"Circuit trip callback failed: {e}")

    def arm_kill_switch(self) -> None:
        """Arm the kill switch"""
        self.kill_switch_armed = True
        logger.warning("Kill Switch 已武装! 输入 KILL 命令将终止所有交易")

    def disarm_kill_switch(self) -> None:
        """Disarm the kill switch"""
        self.kill_switch_armed = False
        logger.info("Kill Switch 已解除")

    async def execute_kill_switch(self, reason: str = "Manual kill switch") -> Dict[str, Any]:
        """
        执行Kill Switch - 平掉所有持仓

        Args:
            reason: 终止原因

        Returns:
            Dict with results
        """
        if not self.kill_switch_armed:
            logger.warning("Kill Switch 未武装，忽略执行请求")
            return {"success": False, "reason": "Kill switch not armed"}

        if not self.emergency_config.kill_switch_enabled:
            logger.warning("Kill Switch 未启用")
            return {"success": False, "reason": "Kill switch not enabled"}

        logger.critical(f"EXECUTING KILL SWITCH: {reason}")

        results = {
            "success": True,
            "reason": reason,
            "timestamp": int(time.time() * 1000),
            "positions_closed": 0,
            "orders_placed": [],
            "errors": []
        }

        # 平掉所有持仓
        positions_to_close = list(self.positions.items())

        for symbol, pos in positions_to_close:
            try:
                # 市价平仓
                side = "sell" if pos.side.lower() == "long" else "buy"
                order = await self.executor.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="market",
                    quantity=pos.quantity,
                    price=pos.current_price
                )
                results["orders_placed"].append(order)
                results["positions_closed"] += 1

                logger.info(f"Kill Switch: 平仓 {symbol} {pos.quantity} @ {pos.current_price}")

            except Exception as e:
                logger.error(f"Kill Switch: 平仓 {symbol} 失败: {e}")
                results["errors"].append(f"{symbol}: {str(e)}")

        # 保存事件
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO emergency_events
                (timestamp, event_type, details, actions_taken)
                VALUES (?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                "KILL_SWITCH",
                reason,
                json.dumps(results)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"保存紧急事件失败: {e}")

        # 发送告警
        self.alerts.send_alert(
            EmergencyLevel.EMERGENCY,
            "Kill Switch 执行!",
            f"所有持仓已平仓\n原因: {reason}",
            {
                "positions_closed": results["positions_closed"],
                "errors": len(results["errors"])
            }
        )

        self.emergency_active = True

        return results

    async def close_all_positions(self, reason: str = "Emergency close") -> Dict[str, Any]:
        """
        关闭所有持仓

        Args:
            reason: 平仓原因

        Returns:
            Dict with results
        """
        logger.warning(f"Closing all positions: {reason}")

        results = {
            "success": True,
            "reason": reason,
            "timestamp": int(time.time() * 1000),
            "positions_closed": 0,
            "errors": []
        }

        positions_to_close = list(self.positions.items())

        for symbol, pos in positions_to_close:
            try:
                side = "sell" if pos.side.lower() == "long" else "buy"
                order = await self.executor.place_order(
                    symbol=symbol,
                    side=side,
                    order_type="market",
                    quantity=pos.quantity,
                    price=pos.current_price
                )

                if order.status.value == "filled":
                    results["positions_closed"] += 1
                    logger.info(f"平仓 {symbol}: {pos.quantity} @ {pos.current_price}")

            except Exception as e:
                logger.error(f"平仓 {symbol} 失败: {e}")
                results["errors"].append(f"{symbol}: {str(e)}")
                results["success"] = False

        # 保存事件
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO emergency_events
                (timestamp, event_type, details, actions_taken)
                VALUES (?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                "CLOSE_ALL",
                reason,
                json.dumps({"closed": results["positions_closed"], "errors": results["errors"]})
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"保存紧急事件失败: {e}")

        return results

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "circuit_state": self.circuit_state.value,
            "emergency_active": self.emergency_active,
            "kill_switch_armed": self.kill_switch_armed,
            "circuit_trip_count": self.circuit_trip_count,
            "anomaly_count_5m": len([
                e for e in self.anomaly_log
                if e.timestamp > (time.time() - 300) * 1000
            ]),
            "positions_tracked": len(self.positions),
            "last_trip_time": self.last_circuit_trip_time
        }

    def get_anomaly_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取异常摘要"""
        cutoff = (time.time() - hours * 3600) * 1000

        recent = [e for e in self.anomaly_log if e.timestamp > cutoff]

        by_type = {}
        by_level = {}

        for e in recent:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            by_level[e.level.value] = by_level.get(e.level.value, 0) + 1

        return {
            "period_hours": hours,
            "total_anomalies": len(recent),
            "by_type": by_type,
            "by_level": by_level
        }


# ===================== 主函数 =====================

async def main():
    """测试Emergency Stop System"""
    from config.production import ProductionConfig

    config = ProductionConfig()

    # 模拟executor
    class MockExecutor:
        async def place_order(self, **kwargs):
            class MockOrder:
                status = type('obj', (object,), {'value': 'filled'})()
            return MockOrder()

    executor = MockExecutor()
    alert_manager = AlertManager(config)
    emergency = EmergencyStopSystem(config, executor, alert_manager)

    # 测试告警
    print("Testing alert system...")
    emergency.alerts.send_alert(
        EmergencyLevel.WARNING,
        "Test Alert",
        "This is a test message"
    )

    # 测试异常记录
    emergency.record_anomaly(
        EmergencyLevel.CRITICAL,
        "PRICE_SPIKE",
        "BTCUSDT",
        "价格下跌10%",
        0.10,
        0.05
    )

    # 测试状态
    print("\nEmergency Stop Status:")
    print(emergency.get_status())

    # 测试熔断
    for i in range(5):
        emergency.record_anomaly(
            EmergencyLevel.CRITICAL,
            "TEST",
            "ETHUSDT",
            f"Test anomaly {i}",
            1.0,
            0.5
        )

    emergency.check_circuit_breaker()
    print("\nAfter anomalies:")
    print(emergency.get_status())


if __name__ == "__main__":
    asyncio.run(main())
