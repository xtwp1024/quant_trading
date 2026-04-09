#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETH Long Runner - 主调度器 for ETH Long Running Mode.
ETH 长程运作模式主调度器

功能:
- 定时循环入口
- Tick (15分钟): 感知 → 技术指标 → Consensus → Signal
- Cycle (1小时): + Research → Debate → Manager
- Review (4小时): 对比signal vs 价格, 更新Memory
- Dream (每日): 30天SDE模拟, 完整流程
- Learn (每日 Dream后): 权重调整
- Report (每日): 日报输出
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
import os
# 使用os.path获取绝对路径，避免Path与中文路径的问题
_script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root_str = _script_dir if _script_dir else "D:/量化交易系统/量化之神"
sys.path.insert(0, project_root_str)
project_root = project_root_str  # 兼容字符串使用

from quant_trading.agent.eth_cycle_brain import EthCycleBrain, CycleResult
from quant_trading.agent.technical_perception import TechnicalPerception
from quant_trading.agent.memory_bank import MemoryBank, DecisionRecord
from quant_trading.agent.view_evolover import ViewEvolver
from quant_trading.agent.review_engine import ReviewEngine
from quant_trading.agent.dream_scheduler import DreamScheduler
from quant_trading.agent.learning_loop import LearningLoop
from quant_trading.agent.report_generator import ReportGenerator, TradingSignal

logger = logging.getLogger("ETHLongRunner")


class ETHLongRunner:
    """
    ETH 长程运作模式主调度器

    调度各组件周期性运行:
    - Tick (15分钟): 技术感知 + 快速决策
    - Cycle (1小时): 完整认知循环
    - Review (4小时): 复盘
    - Dream (每日): 梦境推演
    - Learn (每日): 权重学习
    - Report (每日): 报告生成
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 路径配置
        default_root = project_root_str
        data_dir_cfg = self.config.get("data_dir")
        reports_dir_cfg = self.config.get("reports_dir")
        self.data_dir = Path(data_dir_cfg) if data_dir_cfg else Path(default_root) / "data"
        self.reports_dir = Path(reports_dir_cfg) if reports_dir_cfg else Path(default_root) / "reports"

        # 初始化组件
        self.memory_bank = MemoryBank(str(self.data_dir / "memory_bank.db"))
        self.view_evolver = ViewEvolver(self.memory_bank)
        self.review_engine = ReviewEngine(self.memory_bank)
        self.learning_loop = LearningLoop(self.memory_bank)
        self.dream_scheduler = DreamScheduler()
        self.report_generator = ReportGenerator(str(self.reports_dir))
        self.cycle_brain = EthCycleBrain(memory_bank=self.memory_bank)

        # 状态
        self.running = False
        self.last_tick_time: Optional[datetime] = None
        self.last_cycle_time: Optional[datetime] = None
        self.last_review_time: Optional[datetime] = None
        self.last_dream_time: Optional[datetime] = None
        self.last_report_time: Optional[datetime] = None

        # 信号历史
        self.signal_history: List[TradingSignal] = []

        # 间隔配置 (秒)
        self.TICK_INTERVAL = self.config.get("tick_interval", 900)  # 15分钟
        self.CYCLE_INTERVAL = self.config.get("cycle_interval", 3600)  # 1小时
        self.REVIEW_INTERVAL = self.config.get("review_interval", 14400)  # 4小时
        self.REPORT_INTERVAL = self.config.get("report_interval", 86400)  # 24小时

        # 标志
        self._shutdown_requested = False

    def start(self):
        """启动长程运作"""
        logger.info("=" * 60)
        logger.info("[START] ETH Long Runner 启动")
        logger.info("=" * 60)

        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True

        # 创建事件循环
        try:
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            logger.info("收到键盘中断")
        finally:
            self.stop()

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}, 请求关闭...")
        self._shutdown_requested = True
        self.running = False

    async def _main_loop(self):
        """主事件循环"""
        logger.info("[FEED] 进入主事件循环...")

        while self.running and not self._shutdown_requested:
            now = datetime.now()

            try:
                # 获取市场数据
                ohlcv, market_data = await self._fetch_market_data()
                if ohlcv is None:
                    logger.warning("获取市场数据失败，10分钟后重试")
                    await asyncio.sleep(600)
                    continue

                # Tick 检查 (15分钟)
                if self._should_run_tick():
                    await self._run_tick(ohlcv, market_data)

                # Cycle 检查 (1小时)
                if self._should_run_cycle():
                    await self._run_cycle(ohlcv, market_data)

                # Review 检查 (4小时)
                if self._should_run_review():
                    await self._run_review(market_data)

                # Dream 检查 (每日收盘后，约20:00 UTC)
                if self._should_run_dream():
                    await self._run_dream(market_data)

                # Report 检查 (每日)
                if self._should_run_report():
                    await self._run_report()

                # 更新最后运行时间
                self.last_tick_time = now

                # 休眠直到下一个检查点
                await asyncio.sleep(60)  # 每分钟检查一次

            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                await asyncio.sleep(60)

    def _should_run_tick(self) -> bool:
        """是否应该运行Tick"""
        if self.last_tick_time is None:
            return True
        elapsed = (datetime.now() - self.last_tick_time).total_seconds()
        return elapsed >= self.TICK_INTERVAL

    def _should_run_cycle(self) -> bool:
        """是否应该运行Cycle"""
        if self.last_cycle_time is None:
            return True
        elapsed = (datetime.now() - self.last_cycle_time).total_seconds()
        return elapsed >= self.CYCLE_INTERVAL

    def _should_run_review(self) -> bool:
        """是否应该运行Review"""
        if self.last_review_time is None:
            return True
        elapsed = (datetime.now() - self.last_review_time).total_seconds()
        return elapsed >= self.REVIEW_INTERVAL

    def _should_run_dream(self) -> bool:
        """是否应该运行Dream"""
        if self.last_dream_time is None:
            # 首次运行，检查是否在收盘后 (20:00 UTC)
            return datetime.now().hour >= 20
        elapsed = (datetime.now() - self.last_dream_time).total_seconds()
        return elapsed >= 86400  # 至少24小时

    def _should_run_report(self) -> bool:
        """是否应该运行Report"""
        if self.last_report_time is None:
            return True
        elapsed = (datetime.now() - self.last_report_time).total_seconds()
        return elapsed >= self.REPORT_INTERVAL

    async def _fetch_market_data(self) -> tuple:
        """获取市场数据"""
        try:
            # 使用Binance API获取ETH数据
            import requests

            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": "ETHUSDT", "interval": "1h", "limit": 200}

            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            ohlcv = np.array([[
                float(x[1]),  # open
                float(x[2]),  # high
                float(x[3]),  # low
                float(x[4]),  # close
                float(x[5]),  # volume
            ] for x in data])

            current_price = ohlcv[-1][3]
            market_data = {
                "price": current_price,
                "volume": ohlcv[-1][4],
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"[FEED] 获取市场数据成功: {len(ohlcv)} 条K线, 价格=${current_price:.2f}")
            return ohlcv, market_data

        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return None, None

    async def _run_tick(self, ohlcv, market_data):
        """运行Tick (快速感知)"""
        logger.info("[TICK] Tick: 快速感知...")

        perception = TechnicalPerception()
        indicators = perception.compute(ohlcv)

        if indicators is None:
            return

        # 生成快速信号
        signals = perception.generate_signals(indicators)
        signal, strength = perception.signal_to_direction(signals)

        price = market_data["price"]

        # 计算简单目标
        if signal == "BUY":
            target = price * 1.02
            stop = price * 0.98
        elif signal == "SELL":
            target = price * 0.98
            stop = price * 1.02
        else:
            target = price
            stop = price

        ts = TradingSignal(
            timestamp=datetime.now().isoformat(),
            signal=signal,
            strength=strength,
            price=price,
            target_zone=(price * 0.99, target),
            stop_loss=stop,
            score=0.0,
            risk_level="LOW"
        )

        self.signal_history.append(ts)
        logger.info(f"[TICK] Tick完成: signal={signal}, strength={strength:.2f}")

    async def _run_cycle(self, ohlcv, market_data):
        """运行完整认知循环"""
        logger.info("[CYCLE] Cycle: 完整认知循环...")

        # 运行认知大脑
        result = await self.cycle_brain.run_cycle(ohlcv, market_data)

        if result is None:
            logger.error("认知循环失败")
            return

        # 记录决策
        decision = DecisionRecord(
            id=None,
            timestamp=result.timestamp,
            price=result.price,
            signal=result.signal,
            strength=result.signal_strength,
            score=result.consensus_score,
            indicators={
                "rsi": result.indicators.rsi,
                "macd_hist": result.indicators.macd_hist,
                "trend": result.indicators.trend,
                "atr": result.indicators.atr,
            },
            research_summary=f"Bull={result.research.bull_score:.2f}, Bear={result.research.bear_score:.2f}" if result.research else "",
            debate_summary=f"Winner={result.debate.winner}" if result.debate else ""
        )
        decision_id = self.memory_bank.save_decision(decision)

        # 记录观点
        self.view_evolver.record_view(
            view_type=result.signal,
            confidence=result.signal_strength,
            price=result.price,
            bull_score=result.debate.bull_score if result.debate else 0,
            bear_score=result.debate.bear_score if result.debate else 0,
            consensus_score=result.consensus_score
        )

        # 保存信号
        ts = TradingSignal(
            timestamp=result.timestamp,
            signal=result.signal,
            strength=result.signal_strength,
            price=result.price,
            target_zone=result.target_zone,
            stop_loss=result.stop_loss,
            score=result.consensus_score,
            risk_level=result.risk_level
        )
        self.signal_history.append(ts)

        # 输出信号提醒
        alert = self.report_generator.generate_signal_alert(ts, {
            "rsi": result.indicators.rsi,
            "macd_hist": result.indicators.macd_hist,
            "trend": result.indicators.trend,
        })
        print(alert)

        self.last_cycle_time = datetime.now()
        logger.info(f"[CYCLE] Cycle完成: signal={result.signal}, score={result.consensus_score:.3f}")

    async def _run_review(self, market_data):
        """运行复盘"""
        logger.info("[REVIEW] Review: 复盘...")

        current_price = market_data["price"]
        results = self.review_engine.review_decisions(current_price, lookback_hours=4)

        if results:
            report = self.review_engine.generate_review_report(results)
            print(report)

        self.last_review_time = datetime.now()
        logger.info(f"[REVIEW] Review完成: 复盘了{len(results)}个决策")

    async def _run_dream(self, market_data):
        """运行梦境推演"""
        logger.info("[DREAM] Dream: 梦境推演...")

        price = market_data["price"]

        dream_result = await self.dream_scheduler.run_dream(
            initial_price=price,
            duration_days=30,
            volatility=0.2,
            drift=0.0
        )

        logger.info(f"[DREAM] Dream完成: {dream_result.summary}")

        # 运行学习
        learning_result = self.learning_loop.learn(lookback_days=7)
        if not learning_result.get("skipped"):
            logger.info(f"[BRAIN] 学习完成: win_rate={learning_result.get('overall_win_rate', 0):.1%}")

        self.last_dream_time = datetime.now()

    async def _run_report(self):
        """生成报告"""
        logger.info("[REPORT] Report: 生成报告...")

        today = datetime.now()
        indicators = self.cycle_brain.perception.compute if hasattr(self.cycle_brain, 'perception') else None

        # 获取观点演化数据
        view_evolution = self.view_evolver.get_view_evolution(days=7)

        # 获取性能数据
        performance = self.review_engine.calculate_signal_performance(days=7)

        # 生成日报
        report_path = self.report_generator.generate_daily_report(
            date=today,
            signals=self.signal_history[-100:],  # 最近100个信号
            consensus_score=self.signal_history[-1].score if self.signal_history else 0,
            indicators={
                "rsi": self.signal_history[-1].score if self.signal_history else 50,
                "trend": "unknown",
            },
            view_evolution=view_evolution,
            performance=performance
        )

        logger.info(f"[REPORT] 报告已生成: {report_path}")

        # 检查是否需要生成周报
        if today.weekday() == 0:  # 周一
            week_start = today - timedelta(days=7)
            self.report_generator.generate_weekly_report(
                week_start=week_start,
                week_end=today,
                signals=self.signal_history[-1000:],  # 保留更多历史
                performance=performance,
                view_evolution=view_evolution,
                weight_changes=self.learning_loop.weight_history[-10:]
            )

        self.last_report_time = datetime.now()

    def stop(self):
        """停止长程运作"""
        logger.info("[STOP] ETH Long Runner 停止")
        self.running = False


def main():
    """入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ETH Long Runner")
    parser.add_argument("--tick-interval", type=int, default=900, help="Tick间隔(秒)")
    parser.add_argument("--cycle-interval", type=int, default=3600, help="Cycle间隔(秒)")
    parser.add_argument("--data-dir", type=str, help="数据目录")
    parser.add_argument("--reports-dir", type=str, help="报告目录")

    args = parser.parse_args()

    config = {
        "tick_interval": args.tick_interval,
        "cycle_interval": args.cycle_interval,
        "data_dir": args.data_dir,
        "reports_dir": args.reports_dir,
    }

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    runner = ETHLongRunner(config)
    runner.start()


if __name__ == "__main__":
    main()
