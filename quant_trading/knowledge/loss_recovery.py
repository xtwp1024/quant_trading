"""
Loss Recovery Manager - 损失复利机制核心模块
核心理念：从每次亏损中学习，防止重复犯错

损失复利公式：
1. 记录每笔亏损交易
2. 分析亏损原因（分类）
3. 制定针对性改进措施
4. 跟踪改进效果
5. 将经验转化为风控规则
"""

import json
import os
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger("LossRecovery")


class LossReason(Enum):
    """亏损原因分类"""
    MARKET_TIMING = "入场时机不当"
    POSITION_SIZING = "仓位管理不当"
    STOP_LOSS = "止损设置不当"
    EMOTIONAL = "情绪化交易"
    FOMO = "FOMO追涨杀跌"
    NEWS_EVENT = "突发事件影响"
    TECHNICAL = "技术分析错误"
    OTHER = "其他原因"


class Priority(Enum):
    """优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# 默认风控规则
DEFAULT_RULES: Dict[str, Any] = {
    "maxSingleLoss": 100.0,
    "maxDailyLoss": 500.0,
    "maxDailyTrades": 10,
    "cooldownAfterLoss": 1800000,  # 30分钟
    "requiredAnalysis": True,
    "autoStopRules": {
        "consecutiveLosses": 3,
        "drawdownLimit": 0.15
    }
}


@dataclass
class TradeRecord:
    """交易记录"""
    id: str
    timestamp: int
    datetime: str
    symbol: str
    type: str  # 'long' or 'short'
    entryPrice: float
    exitPrice: float
    quantity: float
    pnl: float
    pnlPercent: float
    reason: str
    lossReason: Optional[str] = None
    analyzed: bool = False


@dataclass
class LossAnalysis:
    """亏损分析数据"""
    totalLoss: float = 0.0
    lossCount: int = 0
    reasons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    improvementActions: List[Dict[str, Any]] = field(default_factory=list)


class LossRecoveryManager:
    """损失复利管理器"""

    def __init__(self, data_dir: str = "./data") -> None:
        self.data_dir = data_dir
        self.trade_log_path = os.path.join(data_dir, "trade_log.json")
        self.loss_analysis_path = os.path.join(data_dir, "loss_analysis.json")
        self.recovery_rules_path = os.path.join(data_dir, "recovery_rules.json")

        self._ensure_data_dir()
        self._init_data()

    def _ensure_data_dir(self) -> None:
        """确保数据目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)

    def _init_data(self) -> None:
        """初始化数据文件"""
        # 交易日志
        if not os.path.exists(self.trade_log_path):
            self._save_data(self.trade_log_path, [])

        # 损失分析
        if not os.path.exists(self.loss_analysis_path):
            self._save_data(self.loss_analysis_path, asdict(LossAnalysis()))

        # 复利规则
        if not os.path.exists(self.recovery_rules_path):
            self._save_data(self.recovery_rules_path, DEFAULT_RULES)

    def _load_data(self, filepath: str) -> Any:
        """读取数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            logger.warning(f"读取文件失败 {filepath}: {e}")
            return None

    def _save_data(self, filepath: str, data: Any) -> bool:
        """保存数据"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except IOError as e:
            logger.error(f"保存文件失败 {filepath}: {e}")
            return False

    def _generate_id(self) -> str:
        """生成唯一ID"""
        return f"trade_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

    def record_trade(self, trade: Dict[str, Any]) -> TradeRecord:
        """
        记录交易

        Args:
            trade: 交易对象，包含 symbol, type, entryPrice, exitPrice,
                   quantity, pnl, pnlPercent, reason
        """
        trades = self._load_data(self.trade_log_path) or []

        trade_record = TradeRecord(
            id=self._generate_id(),
            timestamp=int(time.time() * 1000),
            datetime=datetime.now().isoformat(),
            symbol=trade.get("symbol", "UNKNOWN"),
            type=trade.get("type", "long"),
            entryPrice=trade.get("entryPrice", 0),
            exitPrice=trade.get("exitPrice", 0),
            quantity=trade.get("quantity", 0),
            pnl=trade.get("pnl", 0),
            pnlPercent=trade.get("pnlPercent", 0),
            reason=trade.get("reason", "未指定"),
            lossReason=None,
            analyzed=False
        )

        trades.append(asdict(trade_record))
        self._save_data(self.trade_log_path, trades)

        # 如果是亏损，自动触发分析流程
        if trade_record.pnl < 0:
            self.analyze_loss(asdict(trade_record))

        return trade_record

    def _classify_loss_reason(self, trade: Dict[str, Any]) -> str:
        """分类亏损原因"""
        pnl_percent = trade.get("pnlPercent", 0)
        reason = trade.get("reason", "")

        # 简单分类逻辑
        if pnl_percent < -2:
            return LossReason.POSITION_SIZING.value
        elif not reason or reason == "未指定":
            return LossReason.MARKET_TIMING.value
        elif "追" in reason or "恐慌" in reason:
            return LossReason.FOMO.value
        else:
            return LossReason.TECHNICAL.value

    def _generate_improvement(self, loss_reason: str) -> Dict[str, str]:
        """生成改进建议"""
        improvements: Dict[str, Dict[str, str]] = {
            LossReason.MARKET_TIMING.value: {
                "action": "等待明确的信号再入场，避免震荡市交易",
                "priority": Priority.HIGH.value
            },
            LossReason.POSITION_SIZING.value: {
                "action": "降低单笔交易仓位，控制在总资金的2%以内",
                "priority": Priority.HIGH.value
            },
            LossReason.STOP_LOSS.value: {
                "action": "重新评估止损位设置，使用技术位而非固定百分比",
                "priority": Priority.HIGH.value
            },
            LossReason.EMOTIONAL.value: {
                "action": "强制冷静期，亏损后休息1小时再交易",
                "priority": Priority.CRITICAL.value
            },
            LossReason.FOMO.value: {
                "action": "制定交易计划并严格执行，不因价格波动而改变",
                "priority": Priority.HIGH.value
            },
            LossReason.NEWS_EVENT.value: {
                "action": "关注重大新闻事件，提前做好风控",
                "priority": Priority.MEDIUM.value
            },
            LossReason.TECHNICAL.value: {
                "action": "复盘分析错误的技术指标使用方法",
                "priority": Priority.MEDIUM.value
            },
            LossReason.OTHER.value: {
                "action": "详细记录交易过程，寻找规律",
                "priority": Priority.LOW.value
            }
        }

        return improvements.get(loss_reason, improvements[LossReason.OTHER.value])

    def analyze_loss(self, trade: Dict[str, Any]) -> None:
        """分析亏损交易"""
        logger.info(f"开始分析亏损交易: ID={trade.get('id')}, 金额={trade.get('pnl')} USDT ({trade.get('pnlPercent', 0):.2f}%)")

        # 加载历史分析数据
        analysis = self._load_data(self.loss_analysis_path) or asdict(LossAnalysis())

        # 更新统计
        analysis["totalLoss"] = analysis.get("totalLoss", 0) + abs(trade.get("pnl", 0))
        analysis["lossCount"] = analysis.get("lossCount", 0) + 1

        # 分类亏损原因
        loss_reason = self._classify_loss_reason(trade)
        trade["lossReason"] = loss_reason

        # 统计各类原因
        if loss_reason not in analysis["reasons"]:
            analysis["reasons"][loss_reason] = {
                "count": 0,
                "totalLoss": 0,
                "occurrences": []
            }

        analysis["reasons"][loss_reason]["count"] += 1
        analysis["reasons"][loss_reason]["totalLoss"] += abs(trade.get("pnl", 0))
        analysis["reasons"][loss_reason]["occurrences"].append({
            "tradeId": trade.get("id"),
            "timestamp": trade.get("timestamp"),
            "loss": trade.get("pnl")
        })

        # 生成改进建议
        improvement = self._generate_improvement(loss_reason)
        analysis["improvementActions"].append({
            "timestamp": int(time.time() * 1000),
            "tradeId": trade.get("id"),
            "reason": loss_reason,
            "action": improvement["action"],
            "priority": improvement["priority"]
        })

        # 保存分析结果
        self._save_data(self.loss_analysis_path, analysis)

        # 更新交易记录
        trades = self._load_data(self.trade_log_path) or []
        trades = [t if t.get("id") != trade.get("id") else {**t, "lossReason": loss_reason, "analyzed": True} for t in trades]
        self._save_data(self.trade_log_path, trades)

        # 显示分析报告
        self._display_loss_analysis(trade, loss_reason, improvement)

        # 检查是否需要调整风控规则
        self._adjust_risk_rules(analysis)

    def _display_loss_analysis(self, trade: Dict[str, Any], loss_reason: str, improvement: Dict[str, str]) -> None:
        """显示亏损分析报告"""
        logger.info("=" * 50)
        logger.info(f"亏损交易分析报告: ID={trade.get('id')}")
        logger.info(f"交易对: {trade.get('symbol')}, 类型: {'做多' if trade.get('type') == 'long' else '做空'}")
        logger.info(f"亏损: {trade.get('pnl')} USDT ({trade.get('pnlPercent', 0):.2f}%)")
        logger.info(f"亏损原因: {loss_reason}, 优先级: {improvement['priority'].upper()}")
        logger.info(f"改进建议: {improvement['action']}")
        logger.info("=" * 50)

    def _adjust_risk_rules(self, analysis: Dict[str, Any]) -> None:
        """根据亏损分析调整风控规则"""
        rules = self._load_data(self.recovery_rules_path) or DEFAULT_RULES.copy()

        for reason, data in analysis.get("reasons", {}).items():
            if data.get("count", 0) >= 3:
                logger.warning(f"检测到频繁亏损模式: {reason} ({data['count']}次)")

                if "仓位" in reason:
                    rules["maxSingleLoss"] = max(50, rules["maxSingleLoss"] * 0.8)
                    logger.warning(f"调整规则: 单笔最大亏损降至 {rules['maxSingleLoss']} USDT")
                elif "情绪" in reason or "FOMO" in reason:
                    rules["cooldownAfterLoss"] = min(7200000, rules["cooldownAfterLoss"] * 1.5)
                    logger.warning(f"调整规则: 亏损后冷却时间增至 {rules['cooldownAfterLoss'] / 60000} 分钟")

        self._save_data(self.recovery_rules_path, rules)

    def check_trade_allowed(self, proposed_trade: Dict) -> Dict[str, Any]:
        """
        检查是否允许交易

        Returns:
            Dict with 'allowed' (bool) and 'reason' (str)
        """
        trades = self._load_data(self.trade_log_path) or []
        rules = self._load_data(self.recovery_rules_path) or DEFAULT_RULES.copy()
        analysis = self._load_data(self.loss_analysis_path) or {}

        # 检查1: 单笔亏损限制
        if proposed_trade.get("potentialLoss", 0) < -rules.get("maxSingleLoss", 100):
            return {
                "allowed": False,
                "reason": f"超过单笔最大亏损限制 ({rules['maxSingleLoss']} USDT)"
            }

        # 检查2: 单日亏损限制
        today = datetime.now().date().isoformat()
        today_trades = [t for t in trades if t.get("datetime", "").startswith(today)]
        today_loss = sum(t.get("pnl", 0) for t in today_trades if t.get("pnl", 0) < 0)

        if today_loss < -rules.get("maxDailyLoss", 500):
            return {
                "allowed": False,
                "reason": f"已达到单日最大亏损限制 ({rules['maxDailyLoss']} USDT)"
            }

        # 检查3: 单日交易次数
        if len(today_trades) >= rules.get("maxDailyTrades", 10):
            return {
                "allowed": False,
                "reason": f"已达到单日最大交易次数 ({rules['maxDailyTrades']}次)"
            }

        # 检查4: 亏损后冷却期
        if trades:
            last_trade = trades[-1]
            if last_trade.get("pnl", 0) < 0:
                time_since_loss = int(time.time() * 1000) - last_trade.get("timestamp", 0)
                cooldown = rules.get("cooldownAfterLoss", 1800000)
                if time_since_loss < cooldown:
                    remaining_min = (cooldown - time_since_loss) // 60000 + 1
                    return {
                        "allowed": False,
                        "reason": f"亏损后冷却期，还需等待 {remaining_min} 分钟"
                    }

        # 检查5: 连续亏损
        consecutive_losses = rules.get("autoStopRules", {}).get("consecutiveLosses", 3)
        recent_trades = trades[-consecutive_losses:] if len(trades) >= consecutive_losses else trades
        if len(recent_trades) >= consecutive_losses and all(t.get("pnl", 0) < 0 for t in recent_trades):
            return {
                "allowed": False,
                "reason": f"连续亏损 {consecutive_losses} 次，强制停止交易"
            }

        return {"allowed": True}

    def generate_recovery_report(self) -> Dict[str, Any]:
        """生成损失复利报告"""
        trades = self._load_data(self.trade_log_path) or []
        analysis = self._load_data(self.loss_analysis_path) or {}
        rules = self._load_data(self.recovery_rules_path) or DEFAULT_RULES.copy()

        profit_trades = [t for t in trades if t.get("pnl", 0) > 0]
        loss_trades = [t for t in trades if t.get("pnl", 0) < 0]

        report = {
            "summary": {
                "totalTrades": len(trades),
                "profitTrades": len(profit_trades),
                "lossTrades": len(loss_trades),
                "totalPnl": sum(t.get("pnl", 0) for t in trades),
                "totalLoss": analysis.get("totalLoss", 0),
                "winRate": f"{(len(profit_trades) / len(trades) * 100):.2f}%" if trades else "0%"
            },
            "topLossReasons": sorted(
                [
                    {
                        "reason": reason,
                        "count": data.get("count", 0),
                        "totalLoss": data.get("totalLoss", 0),
                        "avgLoss": data.get("totalLoss", 0) / data.get("count", 1) if data.get("count", 0) > 0 else 0
                    }
                    for reason, data in analysis.get("reasons", {}).items()
                ],
                key=lambda x: x["count"],
                reverse=True
            )[:5],
            "recentImprovements": analysis.get("improvementActions", [])[-10:],
            "currentRules": rules
        }

        return report

    def display_recovery_report(self) -> None:
        """显示损失复利报告"""
        report = self.generate_recovery_report()

        logger.info("=" * 50)
        logger.info("损失复利分析报告")
        logger.info("=" * 50)
        logger.info(f"总交易: {report['summary']['totalTrades']}, 盈利: {report['summary']['profitTrades']}, 亏损: {report['summary']['lossTrades']}")
        logger.info(f"胜率: {report['summary']['winRate']}, 总盈亏: {report['summary']['totalPnl']:.2f} USDT")

        logger.info("主要亏损原因 (Top 5):")
        for i, item in enumerate(report["topLossReasons"]):
            logger.info(f"  {i + 1}. {item['reason']}: 次数={item['count']}, 平均亏损={item['avgLoss']:.2f} USDT")

        logger.info("最近改进建议:")
        for i, item in enumerate(report["recentImprovements"][-5:]):
            logger.info(f"  {i + 1}. [{item.get('priority', 'N/A').upper()}] {item.get('action', '')}")

        rules = report["currentRules"]
        logger.info(f"当前风控规则: 单笔最大亏损={rules.get('maxSingleLoss', 100)} USDT, "
                   f"单日最大亏损={rules.get('maxDailyLoss', 500)} USDT, "
                   f"冷却时间={rules.get('cooldownAfterLoss', 1800000) / 60000} 分钟")
        logger.info("=" * 50)


# 导出主要类
__all__ = ["LossRecoveryManager", "LossReason", "Priority", "TradeRecord"]
