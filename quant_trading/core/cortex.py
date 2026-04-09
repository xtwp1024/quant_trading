# -*- coding: utf-8 -*-

from .logger import logger
from typing import Any, Dict
import time
import json
import os
from pathlib import Path

class TitanCortex:
    """
    TitanCortex: 元认知风险控制器 (Meta-Cognitive Risk Controller).
    基因: FinRL (性能监控), Warp-Drive (自愈) (Genes: FinRL, Warp-Drive).
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        # HIGH: 验证必要的配置项是否存在，防止运行时KeyError
        self._risk_config = self.config.get('risk', {})
        self._max_daily_loss_pct = self._risk_config.get('max_daily_loss_pct', 0.1)  # 默认10%
        self._max_slippage_pct = self._risk_config.get('max_slippage_pct', 0.01)  # 默认1%
        self.strategy_health: Dict[str, float] = {} # strategy_name -> health_score (0.0 to 1.0)
        self.persist_file = Path(__file__).parent.parent / "data" / "cortex_health.json"
        self.load_state()
        logger.info("🛡️ [TitanCortex] 元认知皮层已初始化 (Meta-Cognitive Cortex Initialized).")

    def update_health(self, strategy_name: str, pnl: float, slippage: float) -> None:
        """
        贝叶斯启发的健康评分更新 (Bayesian-inspired health score update).
        """
        if strategy_name not in self.strategy_health:
            self.strategy_health[strategy_name] = 1.0 # 初始健康 (Start healthy)

        # 亏损和高滑点的惩罚 (Penalty for loss and high slippage)
        penalty = 0.0
        if pnl < 0:
            # HIGH: 使用预先验证的配置值，避免KeyError
            penalty += abs(pnl) / self._max_daily_loss_pct
        if slippage > self._max_slippage_pct:
            # DYNAMIC PENALTY: Penalty scales with how much we exceeded the slippage limit
            # Base penalty 0.05 + Excess ratio
            excess = (slippage - self._max_slippage_pct) / self._max_slippage_pct
            penalty += 0.05 + (excess * 0.1)
            
        # 衰减和恢复逻辑 (Decay and recover logic)
        self.strategy_health[strategy_name] = max(0.0, min(1.0, self.strategy_health[strategy_name] - penalty + 0.01))
        self.save_state()
        
        status = "HEALTHY" if self.strategy_health[strategy_name] > 0.7 else "WARN" if self.strategy_health[strategy_name] > 0.4 else "CRITICAL"
        logger.info(f"🛡️ [TitanCortex] 状态 [{strategy_name}]: {status} (得分: {self.strategy_health[strategy_name]:.2f})")
        
        if self.strategy_health[strategy_name] < 0.3:
            self.trigger_apoptosis(strategy_name)

    def trigger_apoptosis(self, strategy_name: str) -> None:
        """
        策略凋亡：优雅地关闭失败的策略 (Strategy Apoptosis: Graceful shutdown of a failing strategy).
        """
        logger.critical(f"💀 [TITAN CORTEX] {strategy_name} 触发凋亡。边缘恶化 (APOPTOSIS TRIGGERED. Edge deteriorated).")
        # 在真实系统中，这将发出事件以杀死策略 (In a real system, this would emit an event to kill the strategy)
        # 对于模拟，我们仅记录为元事件 (For simulation, we just log it as a meta-event)

    def self_heal(self) -> None:
        """
        尝试在市场平静时期恢复健康状态 (Attempt to restore healthy states during market quiet times).
        """
        for strat in self.strategy_health:
            if self.strategy_health[strat] < 1.0:
                self.strategy_health[strat] = min(1.0, self.strategy_health[strat] + 0.05)
                logger.info(f"🩹 [TitanCortex] 正在自愈 {strat}... 健康度: {self.strategy_health[strat]:.2f}")
        self.save_state()

    def save_state(self) -> None:
        """
        Persist health scores to disk.
        """
        try:
            self.persist_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_file, 'w', encoding='utf-8') as f:
                json.dump(self.strategy_health, f, indent=4)
        except Exception as e:
            logger.error(f"❌ [TitanCortex] Failed to save state: {e}")

    def load_state(self) -> None:
        """
        Load health scores from disk.
        """
        if self.persist_file.exists():
            try:
                with open(self.persist_file, 'r', encoding='utf-8') as f:
                    self.strategy_health = json.load(f)
                logger.info(f"💾 [TitanCortex] Loaded health states for {len(self.strategy_health)} strategies.")
            except Exception as e:
                logger.error(f"❌ [TitanCortex] Failed to load state: {e}")
