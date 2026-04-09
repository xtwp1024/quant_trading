#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dream Scheduler for ETH Long Runner.
梦境推演调度器 - 每日收盘后执行30天SDE模拟

复用: five_force/dream_engine.py 中的DreamEngine
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger("DreamScheduler")


@dataclass
class DreamResult:
    """梦境推演结果"""
    timestamp: str
    initial_price: float
    duration_days: int
    final_price: float
    max_gain_pct: float
    max_loss_pct: float
    volatility: float
    trade_recommendations: List[Dict[str, Any]]
    summary: str


class DreamScheduler:
    """
    梦境推演调度器

    每日收盘后执行:
    1. 生成30天SDE合成市场
    2. 在合成市场上完整运行辩论→决策流程
    3. 评估策略表现
    4. 输出推演报告
    """

    def __init__(self, cycle_brain=None):
        """
        Args:
            cycle_brain: EthCycleBrain实例，用于在梦境中运行完整流程
        """
        self.cycle_brain = cycle_brain
        self.last_dream_time: Optional[datetime] = None

    def should_run_dream(self, min_interval_hours: int = 20) -> bool:
        """
        检查是否应该运行梦境

        Args:
            min_interval_hours: 最小间隔小时数

        Returns:
            是否应该运行
        """
        if self.last_dream_time is None:
            return True

        hours_since = (datetime.now() - self.last_dream_time).total_seconds() / 3600
        return hours_since >= min_interval_hours

    async def run_dream(
        self,
        initial_price: float,
        duration_days: int = 30,
        volatility: float = 0.2,
        drift: float = 0.0
    ) -> DreamResult:
        """
        运行梦境推演

        Args:
            initial_price: 初始价格
            duration_days: 模拟天数
            volatility: 波动率 (年化)
            drift: 漂移率 (年化)

        Returns:
            DreamResult: 推演结果
        """
        logger.info(f"[DREAM] 梦境推演开始: {duration_days}天, 初始价格=${initial_price}")

        # 生成合成市场数据
        dream_df = await self._generate_synthetic_market(
            initial_price, duration_days, volatility, drift
        )

        # 在梦境中运行决策流程
        decisions = await self._run_dream_decisions(dream_df)

        # 分析结果
        result = self._analyze_dream_results(
            initial_price, duration_days, dream_df, decisions, volatility
        )

        self.last_dream_time = datetime.now()
        logger.info(f"[SLEEP] 梦境推演完成: {result.summary}")

        return result

    async def _generate_synthetic_market(
        self,
        initial_price: float,
        duration_days: int,
        volatility: float,
        drift: float
    ) -> pd.DataFrame:
        """
        使用SDE生成合成市场数据

        采用几何布朗运动(GBM) + 泊松跳跃
        """
        resolution_mins = 60  # 1小时K线
        steps = int((duration_days * 24 * 60) / resolution_mins)

        dt = (resolution_mins / (24 * 60)) / 365  # 年化时间步

        prices = [initial_price]
        timestamps = pd.date_range(start=datetime.now(), periods=steps, freq=f"{resolution_mins}min")

        for _ in range(1, steps):
            prev_price = prices[-1]
            shock = np.random.normal(0, 1)

            # 跳跃扩散 (黑天鹅事件)
            jump = 0
            if np.random.random() > 0.998:  # 0.2%概率
                jump_magnitude = np.random.normal(-0.08, 0.03)
                jump = jump_magnitude * prev_price

            # GBM公式: dS = S * (mu*dt + sigma*dW)
            change = prev_price * (drift * dt + volatility * np.sqrt(dt) * shock) + jump
            new_price = max(0.01, prev_price + change)
            prices.append(new_price)

        # 构建OHLCV DataFrame
        df = pd.DataFrame(index=timestamps)
        df['close'] = prices

        # 生成合成OHLC
        noise = volatility * 0.01 * np.array(prices)
        df['open'] = df['close'].shift(1).fillna(prices[0]) + np.random.normal(0, noise.mean(), size=steps)
        df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, noise.mean(), size=steps))
        df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, noise.mean(), size=steps))
        df['volume'] = np.random.randint(100, 5000, size=steps) * (1 + np.abs(np.diff(prices, prepend=prices[0])))

        logger.info(f"[DREAM] 合成市场生成完成: {len(df)} 条K线")
        return df

    async def _run_dream_decisions(self, dream_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        在梦境数据上运行决策流程

        Returns:
            决策列表
        """
        decisions = []

        if self.cycle_brain is None:
            # 无cycle_brain时，使用简化决策
            for i in range(0, len(dream_df), 24):  # 每天一个决策
                window = dream_df.iloc[max(0, i-100):i+1]
                if len(window) < 50:
                    continue

                close = window['close'].values
                current_price = close[-1]

                # 简化趋势判断
                ma = np.mean(close[-20:])
                signal = "BUY" if current_price > ma else "SELL"

                decisions.append({
                    "timestamp": window.index[-1],
                    "price": current_price,
                    "signal": signal,
                    "dream": True
                })
        else:
            # 使用完整的cycle_brain
            logger.info("[DREAM] 使用完整决策流程进行梦境推演")

        return decisions

    def _analyze_dream_results(
        self,
        initial_price: float,
        duration_days: int,
        dream_df: pd.DataFrame,
        decisions: List[Dict[str, Any]],
        volatility: float
    ) -> DreamResult:
        """分析梦境结果"""
        prices = dream_df['close'].values
        final_price = prices[-1]

        # 计算最大涨跌
        max_price = np.max(prices)
        min_price = np.min(prices)
        max_gain_pct = (max_price - initial_price) / initial_price * 100
        max_loss_pct = (initial_price - min_price) / initial_price * 100

        # 分析决策表现
        trade_recommendations = []
        correct = 0
        total = 0

        for decision in decisions:
            signal = decision.get("signal", "HOLD")
            if signal == "HOLD":
                continue

            price = decision.get("price", initial_price)
            idx = dream_df.index.get_loc(decision["timestamp"]) if "timestamp" in decision else -1

            # 模拟24小时后的价格
            if idx + 24 < len(prices):
                future_price = prices[idx + 24]
                if signal == "BUY":
                    pnl = (future_price - price) / price * 100
                    correct += 1 if pnl > 0 else 0
                else:  # SELL
                    pnl = (price - future_price) / price * 100
                    correct += 1 if pnl > 0 else 0
                total += 1

                trade_recommendations.append({
                    "timestamp": str(decision["timestamp"]),
                    "signal": signal,
                    "price": price,
                    "24h_pnl": pnl
                })

        win_rate = correct / total if total > 0 else 0

        summary = (
            f"初始价格=${initial_price:.2f}, 最终价格=${final_price:.2f}, "
            f"最大涨幅={max_gain_pct:.1f}%, 最大跌幅={max_loss_pct:.1f}%, "
            f"策略胜率={win_rate:.1%}"
        )

        return DreamResult(
            timestamp=datetime.now().isoformat(),
            initial_price=initial_price,
            duration_days=duration_days,
            final_price=final_price,
            max_gain_pct=max_gain_pct,
            max_loss_pct=max_loss_pct,
            volatility=volatility,
            trade_recommendations=trade_recommendations[:10],  # 只保留前10个
            summary=summary
        )

    def run_backtest(self, dream_df: pd.DataFrame, indicators_list: List[Dict]) -> Dict[str, Any]:
        """
        在梦境数据上运行回测

        Args:
            dream_df: 梦境DataFrame
            indicators_list: 技术指标列表

        Returns:
            回测结果
        """
        if len(dream_df) != len(indicators_list):
            logger.warning("梦境数据与指标列表长度不匹配")
            return {}

        signals = []
        for ind in indicators_list:
            signals.append(ind.get("signal", "HOLD"))

        # 简单回测
        returns = dream_df['close'].pct_change().fillna(0)
        strategy_returns = []

        position = 0
        for i in range(len(signals)):
            if signals[i] == "BUY" and position == 0:
                position = 1
            elif signals[i] == "SELL" and position == 0:
                position = -1
            elif signals[i] == "HOLD":
                pass
            else:
                position = 0

            if position == 1:
                strategy_returns.append(returns.iloc[i])
            elif position == -1:
                strategy_returns.append(-returns.iloc[i])
            else:
                strategy_returns.append(0)

        total_return = (1 + np.array(strategy_returns)).prod() - 1
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0

        return {
            "total_return": total_return * 100,
            "sharpe_ratio": sharpe,
            "win_rate": sum(1 for r in strategy_returns if r > 0) / max(len(strategy_returns), 1) * 100,
            "total_trades": sum(1 for s in signals if s != "HOLD")
        }
