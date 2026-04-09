import numpy as np
import logging

logger = logging.getLogger("RLAllocator")

class RLAllocator:
    """
    AI-4: The RL Allocator (Contextual Gradient Bandit).
    Learns optimal weights based on Market Regime (Context) and Realized PnL (Reward).
    """
    def __init__(self, strategies: list[str], alpha: float = 0.1):
        """
        params:
            strategies: list of strategy names [str]
            alpha: learning rate
        """
        self.strategies = strategies
        self.alpha = alpha
        
        # Contexts: CALM, CHAOS, CRISIS
        self.contexts = ["CALM", "CHAOS", "CRISIS"]
        
        # Preferences H(s, a)
        # Shape: (num_contexts, num_strategies)
        # Initialize with 0.0 (Equal probability start)
        self.preferences = np.zeros((len(self.contexts), len(strategies)))
        
        # Action Probabilities Policy(s)
        self.policy = np.zeros_like(self.preferences)
        self.update_policy() # Init probabilities
        
        # Historical Reward baseline for baseline subtraction (reduce variance)
        self.avg_reward = 0.0

    def _get_context_idx(self, context_str):
        if context_str not in self.contexts:
            return 0 # Default to CALM
        return self.contexts.index(context_str)

    def update_policy(self):
        """
        Softmax: pi(a|s) = exp(H(s,a)) / sum(exp(H(s,b)))
        """
        for i in range(len(self.contexts)):
            exps = np.exp(self.preferences[i])
            self.policy[i] = exps / np.sum(exps)

    def allocate(self, grok_scores: dict[str, float], market_regime: str = "CALM") -> dict[str, float]:
        """
        Selects weights based on current Policy + Grok Score Heuristic.
        Mixing RL with Grok Scores speeds up convergence.
        
        returns: {strat_name: weight_float}
        """
        ctx_idx = self._get_context_idx(market_regime)
        rl_weights = self.policy[ctx_idx]
        
        final_weights = {}
        total_weight = 0.0
        
        for i, strat in enumerate(self.strategies):
            # Hybrid: 50% RL Policy + 50% Grok Score (Deep Ranking)
            # This is "Guided RL"
            grok_score = grok_scores.get(strat, 0.0)
            
            # RL Weight
            w_rl = rl_weights[i]
            
            # Combined
            w_combined = (w_rl * 0.7) + (grok_score * 0.3)
            
            # Hard Constraint: If Grok says 0 (Meltdown), we respect it
            if grok_score < 0.1:
                w_combined = 0.0
                
            final_weights[strat] = w_combined
            total_weight += w_combined
            
        # Normalize
        if total_weight > 0:
            for k in final_weights:
                final_weights[k] = round(final_weights[k] / total_weight, 4)
        else:
            # Fallback to equal
             for k in final_weights: final_weights[k] = 0.0
             
        return final_weights

    def update(self, market_regime: str, actions_weights: dict[str, float], reward: float):
        """
        Gradient Bandit Update.
        H_{t+1}(a) = H_t(a) + alpha * (R - R_avg) * (1 - pi(a))  if action taken
        H_{t+1}(b) = H_t(b) - alpha * (R - R_avg) * pi(b)       if action NOT taken
        
        Here, our "Action" is a vector (portfolio weights), not a single arm pull.
        We approximate by updating all arms proportional to their weight contribution.
        """
        ctx_idx = self._get_context_idx(market_regime)
        
        # Update baseline reward
        self.avg_reward += 0.1 * (reward - self.avg_reward) # Moving average
        
        baseline_diff = reward - self.avg_reward
        
        for i, strat in enumerate(self.strategies):
            prob = self.policy[ctx_idx][i]
            weight_used = actions_weights.get(strat, 0.0)
            
            # Gradient Ascent Logic adapted for Portfolio
            # If we used this strategy heavily (weight > prob?), and reward is good, boost it.
            # Simplified: H = H + alpha * (R - R_bar) * (weight - prob)
            # This reinforces choices that deviated from the mean policy in a good direction.
            
            gradient = (weight_used - prob) * baseline_diff
            self.preferences[ctx_idx][i] += self.alpha * gradient
            
        # Re-calc policy
        self.update_policy()
        
        logger.info(f"🎓 RL Update ({market_regime}): Reward={reward:.4f} | Top Pref: {self.strategies[np.argmax(self.preferences[ctx_idx])]}")

if __name__ == "__main__":
    # Test
    strats = ["A", "B", "C"]
    rl = RLAllocator(strats)
    w = rl.allocate({"A":0.8, "B":0.5, "C":0.2}, "CHAOS")
    print(f"Weights: {w}")
    rl.update("CHAOS", w, reward=0.05) # Good reward
    w2 = rl.allocate({"A":0.8, "B":0.5, "C":0.2}, "CHAOS")
    print(f"Weights Post-Update: {w2}")
