import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("PortfolioAllocator")

class PortfolioAllocator:
    """
    AI-4: The Allocator.
    Responsibility: "Candidate Isolation" and Entropy Maximization.
    Prevents "Crowded Trades" by penalizing high-correlation strategies.
    """
    def __init__(self):
        pass

    def allocate(self, strategies, grok_scores, market_regime="CALM"):
        """
        Calculates optimal weights for a portfolio of strategies.
        
        params:
            strategies: list of strategy objects (must have .name)
            grok_scores: dict {strategy_name: score_float}
            market_regime: str "CALM" | "CHAOS" | "CRISIS"
            
        returns:
            weights: dict {strategy_name: percentage_float}
        """
        # 1. Filter by Regime (AI-5 Logic Injection)
        # If CHAOS, we only allow Elite strategies (Score > 0.8)
        eligible_strategies = []
        
        for strat in strategies:
            score = grok_scores.get(strat.name, 0)
            if market_regime in ["CHAOS", "CRISIS"]:
                if score < 0.8:
                    logger.info(f"🚫 MELTDOWN: {strat.name} cut due to {market_regime} (Score {score:.2f} < 0.8)")
                    continue
            if score < 0.4: # Base floor
                continue
            eligible_strategies.append(strat)
            
        if not eligible_strategies:
            logger.warning("No strategies survived regime check. Cash is King.")
            return {s.name: 0.0 for s in strategies}

        # 2. Candidate Isolation (Correlation Check)
        # Simulation: We generate a dummy correlation matrix for demonstration.
        # In real life, we would use strat.history to calc correlation.
        
        n = len(eligible_strategies)
        names = [s.name for s in eligible_strategies]
        
        # Simulating a correlation matrix where GeneticAlpha might correlate with OmegaPoint
        # For demo, Random correlation between -0.2 and 0.9
        np.random.seed(42) # Fixed seed for demo stability
        corr_matrix = np.random.uniform(-0.2, 0.9, (n, n))
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Symmetric
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        
        # 3. Penalize High Correlation (Entropy Maximization)
        # We adjust the "Effective Score" based on uniqueness.
        final_weights = {}
        
        for i, strat in enumerate(eligible_strategies):
            base_score = grok_scores.get(strat.name, 0.5)
            
            # Calculate Uniqueness: 1 - average correlation with others
            avg_corr = np.mean(corr_matrix[i, :])
            uniqueness = 1.0 - max(0, avg_corr - 0.2) # Simple penalty if corr > 0.2
            
            # Boost Uniqueness importance
            effective_score = base_score * (1.0 + uniqueness)
            
            final_weights[strat.name] = effective_score
            
        # 4. Normalize to 1.0 (Capital Allocation)
        total_score = sum(final_weights.values())
        if total_score > 0:
            for k in final_weights:
                final_weights[k] = round(final_weights[k] / total_score, 4)
        else:
             return {s.name: 0.0 for s in strategies}
             
        # Add zeroes for rejected strategies
        for s in strategies:
            if s.name not in final_weights:
                final_weights[s.name] = 0.0
                
        return final_weights

if __name__ == "__main__":
    # Test Bench
    class MockStrat:
        def __init__(self, name): self.name = name
        
    s1 = MockStrat("TrendFollower_A")
    s2 = MockStrat("TrendFollower_B") # Correlated
    s3 = MockStrat("MeanReversion_C") # Uncorrelated
    
    allocator = PortfolioAllocator()
    scores = {
        "TrendFollower_A": 0.9,
        "TrendFollower_B": 0.88,
        "MeanReversion_C": 0.7  # Lower score but unique
    }
    
    print("--- regime: CALM ---")
    w = allocator.allocate([s1, s2, s3], scores, "CALM")
    print(w)
    
    print("\n--- regime: CHAOS (Meltdown) ---")
    w = allocator.allocate([s1, s2, s3], scores, "CHAOS")
    print(w)
