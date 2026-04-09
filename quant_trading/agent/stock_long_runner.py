#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Long Runner - A股长程运作模式主调度器
A股股票池周期性持续分析

功能:
- Tick (30分钟): 股票池快速扫描, 更新分类
- Cycle (2小时): 完整分析, 包括:
  - 各类别信号检测
  - 企稳候选检查 (弱势股)
  - 趋势候选检查 (趋势股)
- Review (每日收盘后): 复盘今日分类变化, 更新策略
- Dream (每日收盘后22:00): 30天SDE模拟, 推演股票池演化
- Report (每日收盘后): 生成日报
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root_str = os.path.normpath(os.path.join(_script_dir, "..")) if _script_dir else "D:/量化交易系统/量化之神"
sys.path.insert(0, project_root_str)

from quant_trading.signal.stock_pool import (
    StockPoolManager,
    StockPoolClassifier,
    AStockDataProvider,
    PoolType,
)

logger = logging.getLogger("StockLongRunner")


class StockLongRunner:
    """
    A股长程运作模式主调度器

    调度各组件周期性运行:
    - Tick (30分钟): 股票池快速扫描
    - Cycle (2小时): 完整分析
    - Review (每日收盘后): 复盘
    - Dream (每日22:00): 梦境推演
    - Report (每日收盘后): 报告生成
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
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.data_provider = AStockDataProvider()
        self.pool_manager = StockPoolManager()

        # 股票池配置
        _default_stocks = [
            "000001", "600519", "300750", "601318",
            "600036", "000002", "601166", "600900",
            "300059", "600030", "601888", "002594"
        ]
        self.stock_list = self.config.get("stock_list") or _default_stocks
        self.lookback_days = self.config.get("lookback_days", 250)

        # 状态
        self.running = False
        self.last_tick_time: Optional[datetime] = None
        self.last_cycle_time: Optional[datetime] = None
        self.last_review_time: Optional[datetime] = None
        self.last_dream_time: Optional[datetime] = None
        self.last_report_time: Optional[datetime] = None

        # 历史记录
        self.pool_history: List[Dict[str, Any]] = []
        self.signal_history: List[Dict[str, Any]] = []

        # 间隔配置 (秒)
        self.TICK_INTERVAL = self.config.get("tick_interval", 1800)  # 30分钟
        self.CYCLE_INTERVAL = self.config.get("cycle_interval", 7200)  # 2小时
        self.REVIEW_INTERVAL = self.config.get("review_interval", 43200)  # 12小时
        self.REPORT_INTERVAL = self.config.get("report_interval", 86400)  # 24小时

        # 标志
        self._shutdown_requested = False

        # A股交易时间判断
        self._market_open = time(9, 30)
        self._market_close = time(15, 0)
        self._lunch_start = time(11, 30)
        self._lunch_end = time(13, 0)

    def start(self):
        """启动长程运作"""
        logger.info("=" * 60)
        logger.info("[START] Stock Long Runner 启动")
        logger.info(f"监控股票数量: {len(self.stock_list)}")
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"报告目录: {self.reports_dir}")
        logger.info("=" * 60)

        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True

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

    def _is_market_time(self) -> bool:
        """判断是否在交易时间内"""
        now = datetime.now()
        current_time = now.time()

        # 周末判断
        if now.weekday() >= 5:  # 周六、周日
            return False

        # 午餐时间判断
        if self._lunch_start <= current_time <= self._lunch_end:
            return False

        # 正常交易时间
        if self._market_open <= current_time <= self._market_close:
            return True

        return False

    async def _main_loop(self):
        """主事件循环"""
        logger.info("[FEED] 进入主事件循环...")

        while self.running and not self._shutdown_requested:
            now = datetime.now()

            try:
                # 判断是否在交易时间
                is_trading = self._is_market_time()

                if is_trading:
                    logger.info(f"[FEED] 当前处于交易时间 ({now.strftime('%H:%M:%S')})")
                else:
                    logger.info(f"[FEED] 当前处于非交易时间 ({now.strftime('%H:%M:%S')})")

                # 获取市场数据
                symbols_ohlcv, volume_ratios = await self._fetch_market_data()

                if symbols_ohlcv:
                    # 更新股票池
                    self.pool_manager.update_pool(symbols_ohlcv, volume_ratios)
                    logger.info(f"[FEED] 股票池更新成功: {len(symbols_ohlcv)} 只股票")

                    # Tick 检查 (30分钟) - 交易时间内更频繁
                    tick_interval = 900 if is_trading else self.TICK_INTERVAL  # 交易时间15分钟
                    if self._should_run_tick(tick_interval):
                        await self._run_tick(symbols_ohlcv, volume_ratios)

                    # Cycle 检查 (2小时)
                    if self._should_run_cycle():
                        await self._run_cycle(symbols_ohlcv, volume_ratios)
                else:
                    logger.warning("[FEED] 未能获取市场数据")

                # Review 检查 (每日收盘后约16:00)
                if self._should_run_review():
                    await self._run_review()

                # Dream 检查 (每日22:00)
                if self._should_run_dream():
                    await self._run_dream()

                # Report 检查 (每日收盘后约17:00)
                if self._should_run_report():
                    await self._run_report()

                # 休眠直到下一个检查点
                await asyncio.sleep(300)  # 每5分钟检查一次

            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                await asyncio.sleep(300)

    def _should_run_tick(self, interval: int) -> bool:
        """是否应该运行Tick"""
        if self.last_tick_time is None:
            return True
        elapsed = (datetime.now() - self.last_tick_time).total_seconds()
        return elapsed >= interval

    def _should_run_cycle(self) -> bool:
        """是否应该运行Cycle"""
        if self.last_cycle_time is None:
            return True
        elapsed = (datetime.now() - self.last_cycle_time).total_seconds()
        return elapsed >= self.CYCLE_INTERVAL

    def _should_run_review(self) -> bool:
        """是否应该运行Review (每日16:00-17:00)"""
        if self.last_review_time is None:
            return datetime.now().hour in [16, 17]
        elapsed = (datetime.now() - self.last_review_time).total_seconds()
        return elapsed >= self.REVIEW_INTERVAL and datetime.now().hour in [16, 17]

    def _should_run_dream(self) -> bool:
        """是否应该运行Dream (每日22:00)"""
        if self.last_dream_time is None:
            return datetime.now().hour >= 22
        elapsed = (datetime.now() - self.last_dream_time).total_seconds()
        return elapsed >= 86400 and datetime.now().hour >= 22

    def _should_run_report(self) -> bool:
        """是否应该运行Report (每日17:00)"""
        if self.last_report_time is None:
            return datetime.now().hour == 17
        elapsed = (datetime.now() - self.last_report_time).total_seconds()
        return elapsed >= self.REPORT_INTERVAL and datetime.now().hour == 17

    async def _fetch_market_data(self) -> tuple:
        """获取市场数据"""
        try:
            symbols_ohlcv, volume_ratios = self.data_provider.fetch_batch(
                self.stock_list,
                days=self.lookback_days
            )
            logger.info(f"[FEED] 获取市场数据: {len(symbols_ohlcv)}/{len(self.stock_list)} 只股票")
            return symbols_ohlcv, volume_ratios
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return {}, {}

    async def _run_tick(self, symbols_ohlcv: Dict[str, np.ndarray], volume_ratios: Dict[str, float]):
        """运行Tick (快速扫描)"""
        logger.info("[TICK] Tick: 股票池快速扫描...")

        summary = self.pool_manager.get_pool_summary()

        # 记录历史
        self._record_snapshot(summary)

        # 输出简要信息
        for pool_name, data in summary.items():
            count = data["count"]
            if count > 0:
                stocks = data["symbols"][:3]  # 只显示前3个
                logger.info(f"[TICK] {pool_name}类({count}只): {', '.join(stocks)}" +
                           ("..." if count > 3 else ""))

        # 弱势企稳候选检查
        candidates = self.pool_manager.get_stabilization_candidates()
        if candidates:
            logger.info(f"[TICK] 发现 {len(candidates)} 只企稳候选股票:")
            for c in candidates[:3]:
                logger.info(f"       - {c['symbol']}: {c['signal']}")

        self.last_tick_time = datetime.now()
        logger.info("[TICK] Tick完成")

    async def _run_cycle(self, symbols_ohlcv: Dict[str, np.ndarray], volume_ratios: Dict[str, float]):
        """运行完整分析周期"""
        logger.info("[CYCLE] Cycle: 完整分析...")

        summary = self.pool_manager.get_pool_summary()

        # 生成完整报告
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"A股股票池分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_lines.append("=" * 60)

        for pool_name, data in summary.items():
            report_lines.append(f"\n【{pool_name}类】({data['count']}只)")
            if data["details"]:
                for detail in data["details"][:5]:  # 最多显示5个
                    chars = self.pool_manager.classifier.classifications.get(detail["symbol"])
                    if chars:
                        c = chars.characteristics
                        report_lines.append(
                            f"  {detail['symbol']}: "
                            f"波动率={c.get('volatility', 0):.1f}%, "
                            f"RSI={c.get('rsi', 0):.1f}, "
                            f"信号={detail.get('signal', '-')}"
                        )

        # 弱势企稳候选
        candidates = self.pool_manager.get_stabilization_candidates()
        if candidates:
            report_lines.append(f"\n【弱势企稳候选】({len(candidates)}只)")
            for c in candidates:
                report_lines.append(f"  {c['symbol']}: {c['signal']} - {c['reason'][:40]}")

        # 趋势股票
        trend_stocks = self.pool_manager.get_trend_stocks()
        if trend_stocks:
            report_lines.append(f"\n【趋势股票】({len(trend_stocks)}只)")
            for s in trend_stocks[:5]:
                report_lines.append(
                    f"  {s['symbol']}: {s['signal']} "
                    f"(动量={s['momentum']:+.1f}%, ADX={s['adx']:.1f})"
                )

        report_text = "\n".join(report_lines)
        print(report_text)

        # 保存到文件
        report_file = self.reports_dir / f"cycle_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"[CYCLE] 报告已保存: {report_file}")

        self.last_cycle_time = datetime.now()
        logger.info("[CYCLE] Cycle完成")

    async def _run_review(self):
        """运行复盘"""
        logger.info("[REVIEW] Review: 复盘今日分类...")

        if len(self.pool_history) < 2:
            logger.info("[REVIEW] 历史数据不足，跳过复盘")
            return

        # 分析分类变化
        latest = self.pool_history[-1]
        previous = self.pool_history[-2] if len(self.pool_history) >= 2 else latest

        changes = []
        for pool_type in ["做T", "趋势", "弱势"]:
            latest_symbols = set(latest.get(pool_type, []))
            previous_symbols = set(previous.get(pool_type, []))

            added = latest_symbols - previous_symbols
            removed = previous_symbols - latest_symbols

            if added:
                changes.append(f"{pool_type}新增: {', '.join(added)}")
            if removed:
                changes.append(f"{pool_type}减少: {', '.join(removed)}")

        if changes:
            logger.info("[REVIEW] 分类变化:")
            for change in changes:
                logger.info(f"       {change}")
        else:
            logger.info("[REVIEW] 分类无变化")

        # 保存复盘记录
        review_file = self.data_dir / f"review_{datetime.now().strftime('%Y%m%d')}.json"
        import json
        with open(review_file, "w", encoding="utf-8") as f:
            json.dump({
                "date": datetime.now().isoformat(),
                "changes": changes,
                "latest_pool": latest,
                "pool_count": len(self.pool_history)
            }, f, ensure_ascii=False, indent=2)

        self.last_review_time = datetime.now()
        logger.info("[REVIEW] Review完成")

    async def _run_dream(self):
        """运行梦境推演"""
        logger.info("[DREAM] Dream: 梦境推演 (30天SDE模拟)...")

        # 对每只股票进行简单的蒙特卡洛模拟
        results = {}
        for symbol, ohlcv in list(self.pool_manager.classifier.classifications.items()):
            if len(ohlcv) < 20:
                continue

            chars = ohlcv.characteristics
            volatility = chars.get("volatility", 2.0) / 100
            current_price = ohlcv.characteristics.get("price_position", 50)

            # 简化的几何布朗运动模拟
            np.random.seed(hash(symbol) % 2**32)
            n_sims = 100
            n_days = 30
            dt = 1/252
            mu = 0.0001  # 漂移
            sigma = volatility * np.sqrt(dt)

            sim_returns = np.random.normal(mu, sigma, (n_sims, n_days))
            sim_prices = np.exp(np.cumsum(sim_returns, axis=1))
            sim_prices = sim_prices * current_price  # 归一化

            # 计算上涨/下跌概率
            final_prices = sim_prices[:, -1]
            up_count = np.sum(final_prices > current_price)
            prob_up = up_count / n_sims

            # 模拟结果摘要
            results[symbol] = {
                "prob_up": prob_up,
                "expected_return": np.mean(final_prices / current_price - 1),
                "max_loss": np.min(np.min(sim_prices, axis=1) / current_price - 1),
                "max_gain": np.max(np.max(sim_prices, axis=1) / current_price - 1),
                "pool_type": ohlcv.pool_type.value
            }

        # 输出模拟结果
        logger.info(f"[DREAM] 模拟完成: {len(results)} 只股票")
        for symbol, result in sorted(results.items(), key=lambda x: x[1]["prob_up"], reverse=True)[:5]:
            logger.info(
                f"       {symbol}({result['pool_type']}): "
                f"上涨概率={result['prob_up']:.1%}, "
                f"期望收益={result['expected_return']:+.1%}"
            )

        # 保存梦境结果
        dream_file = self.data_dir / f"dream_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        import json
        with open(dream_file, "w", encoding="utf-8") as f:
            json.dump({
                "date": datetime.now().isoformat(),
                "results": results
            }, f, ensure_ascii=False, indent=2)

        self.last_dream_time = datetime.now()
        logger.info(f"[DREAM] Dream完成: {dream_file}")

    async def _run_report(self):
        """生成日报"""
        logger.info("[REPORT] Report: 生成日报...")

        summary = self.pool_manager.get_pool_summary()

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"A股股票池日报 - {datetime.now().strftime('%Y-%m-%d')}")
        report_lines.append("=" * 60)

        # 总体概况
        total = sum(d["count"] for d in summary.values())
        report_lines.append(f"\n【总体概况】监控股票: {total}只")

        for pool_name, data in summary.items():
            report_lines.append(f"  - {pool_name}类: {data['count']}只")

        # 详细分类
        for pool_name, data in summary.items():
            report_lines.append(f"\n【{pool_name}类】")
            if data["details"]:
                for detail in data["details"]:
                    report_lines.append(f"  {detail['symbol']}: {detail.get('signal', '-')}")
            else:
                report_lines.append("  (无)")

        # 弱势企稳候选
        candidates = self.pool_manager.get_stabilization_candidates()
        if candidates:
            report_lines.append(f"\n【弱势企稳候选】({len(candidates)}只)")
            for c in candidates:
                report_lines.append(f"  {c['symbol']}: {c['signal']}")

        # 趋势股票
        trend_stocks = self.pool_manager.get_trend_stocks()
        if trend_stocks:
            report_lines.append(f"\n【趋势股票】({len(trend_stocks)}只)")
            for s in trend_stocks[:5]:
                report_lines.append(
                    f"  {s['symbol']}: {s['signal']} (动量={s['momentum']:+.1f}%)"
                )

        # 操作建议
        report_lines.append("\n【操作建议】")
        scalping_count = summary.get("做T", {}).get("count", 0)
        trend_count = summary.get("趋势", {}).get("count", 0)
        watch_count = summary.get("弱势", {}).get("count", 0)

        if trend_count > scalping_count:
            report_lines.append("  - 当前趋势类股票较多，可关注趋势跟踪策略")
        elif scalping_count > trend_count:
            report_lines.append("  - 当前做T类股票较多，可关注高抛低吸机会")
        else:
            report_lines.append("  - 各类股票分布均衡，谨慎观望为主")

        if candidates:
            report_lines.append(f"  - {len(candidates)}只弱势股出现企稳信号，可适当关注")

        report_lines.append("\n" + "=" * 60)

        report_text = "\n".join(report_lines)
        print(report_text)

        # 保存日报
        report_file = self.reports_dir / f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"[REPORT] 日报已保存: {report_file}")

        self.last_report_time = datetime.now()

    def _record_snapshot(self, summary: Dict[str, Any]):
        """记录股票池快照"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "做T": summary.get("做T", {}).get("symbols", []),
            "趋势": summary.get("趋势", {}).get("symbols", []),
            "弱势": summary.get("弱势", {}).get("symbols", []),
        }
        self.pool_history.append(snapshot)

        # 保持历史记录在100条以内
        if len(self.pool_history) > 100:
            self.pool_history = self.pool_history[-100:]

    def stop(self):
        """停止长程运作"""
        logger.info("[STOP] Stock Long Runner 停止")
        self.running = False


def main():
    """入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Stock Long Runner - A股长程运作模式")
    parser.add_argument("--stock-list", nargs="*", help="股票代码列表")
    parser.add_argument("--tick-interval", type=int, default=1800, help="Tick间隔(秒), 默认30分钟")
    parser.add_argument("--cycle-interval", type=int, default=7200, help="Cycle间隔(秒), 默认2小时")
    parser.add_argument("--lookback-days", type=int, default=250, help="历史数据天数")
    parser.add_argument("--data-dir", type=str, help="数据目录")
    parser.add_argument("--reports-dir", type=str, help="报告目录")
    parser.add_argument("--once", action="store_true", help="单次运行后退出")

    args = parser.parse_args()

    config = {
        "stock_list": args.stock_list,
        "tick_interval": args.tick_interval,
        "cycle_interval": args.cycle_interval,
        "lookback_days": args.lookback_days,
        "data_dir": args.data_dir,
        "reports_dir": args.reports_dir,
    }

    # 配置日志 - 确保reports_dir有默认值
    _reports_dir = config.get("reports_dir") or os.path.join(project_root_str, "reports")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                Path(_reports_dir) / f"runner_{datetime.now().strftime('%Y%m%d')}.log",
                encoding="utf-8"
            )
        ]
    )

    runner = StockLongRunner(config)

    if args.once:
        # 单次运行模式
        asyncio.run(runner._fetch_market_data())
        symbols_ohlcv, volume_ratios = runner.data_provider.fetch_batch(
            runner.stock_list,
            runner.lookback_days
        )
        if symbols_ohlcv:
            runner.pool_manager.update_pool(symbols_ohlcv, volume_ratios)
            asyncio.run(runner._run_cycle(symbols_ohlcv, volume_ratios))
    else:
        runner.start()


if __name__ == "__main__":
    main()
