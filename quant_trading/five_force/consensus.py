import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger("ConsensusEngine")

@dataclass
class Vote:
    unit: str
    direction: float # 1.0 (BUY) to -1.0 (SELL)
    confidence: float # 0.0 to 1.0
    weight: float = 1.0

class ConsensusEngine:
    """
    Stage 6: Market Fusion.
    Aggregates "Thoughts" from all Cognitive Units to form a unified Decision.
    """
    def __init__(self):
        # Weights for each Unit (can be dynamic)
        self.weights = {
            "AlphaUnit(α)": 1.5, # Pattern Resonator (Primary Signal)
            "BetaUnit(β)": 2.0,  # Risk Warden (Veto Power)
            "GammaUnit(γ)": 1.0, # Reflexivity
            "DeltaUnit(δ)": 0.8, # Multiverse
            "EpsilonUnit(ε)": 0.5 # Time Crystal
        }
        self.votes: List[Vote] = []

    def cast_vote(self, unit_name, signal_type, confidence=1.0):
        direction = 0.0
        if signal_type == "BUY": direction = 1.0
        elif signal_type == "SELL": direction = -1.0
        
        weight = self.weights.get(unit_name, 1.0)
        
        vote = Vote(unit_name, direction, confidence, weight)
        self.votes.append(vote)
        logger.debug(f"🗳️ Vote Cast: {unit_name} -> {signal_type} ({confidence:.2f})")

    def fuse(self):
        if not self.votes:
            return 0.0, "Neutral"

        total_score = 0.0
        total_weight = 0.0
        
        for v in self.votes:
            score = v.direction * v.confidence * v.weight
            total_score += score
            total_weight += v.weight
            
        final_score = total_score / total_weight if total_weight > 0 else 0
        
        # Clean up for next tick
        self.votes.clear()
        
        # Interpret
        decision = "HOLD"
        if final_score > 0.3: decision = "BUY"
        elif final_score < -0.3: decision = "SELL"
        
        return final_score, decision

    def adjust_weight(self, unit: str, delta: float):
        """
        动态调整权重 (用于LearningLoop)

        Args:
            unit: 单元名称
            delta: 权重变化量
        """
        old_weight = self.weights.get(unit, 1.0)
        new_weight = max(0.1, min(3.0, old_weight + delta))
        self.weights[unit] = new_weight
        logger.info(f"⚖️ 权重调整: {unit} {old_weight:.2f} → {new_weight:.2f} (delta={delta:+.2f})")

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.weights.copy()

    def reset_weights(self):
        """重置权重到默认值"""
        self.weights = {
            "AlphaUnit(α)": 1.5,
            "BetaUnit(β)": 2.0,
            "GammaUnit(γ)": 1.0,
            "DeltaUnit(δ)": 0.8,
            "EpsilonUnit(ε)": 0.5
        }
        logger.info("⚖️ 权重已重置为默认值")
