import numpy as np
import logging

logger = logging.getLogger("GrokScorer")

class GrokScorer:
    """
    The Brain of the Legion.
    Uses a Numpy-based Self-Attention mechanism to evaluate strategy quality.
    Inspired by: X (Twitter) Recommendation Algorithm (Deep Ranking).
    
    Why Self-Attention?
    - Captures non-linear relationships between disparate metrics.
    - E.g. High Return + High Drawdown might be bad, but High Return + Low Correlation is good.
    - Traditional linear weighting cannot capture this context.
    """
    def __init__(self, embedding_dim=16, num_heads=4):
        self.d_model = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Initialize Weights (Random for now, in a real system these would be learned/loaded)
        # Weight matrices for Q, K, V
        self.W_q = np.random.randn(self.d_model, self.d_model) * 0.01
        self.W_k = np.random.randn(self.d_model, self.d_model) * 0.01
        self.W_v = np.random.randn(self.d_model, self.d_model) * 0.01
        
        # Dense Layer Weights
        self.W_out = np.random.randn(self.d_model, 1) * 0.01
        
        # Bias (Heuristic to ensure decent starting scores)
        self.bias = 0.5 

    def predict(self, metrics_dict):
        """
        Deep Ranking Prediction.
        params:
            metrics_dict: {'fuzz_score': 90.0, 'attack_score': 70.0, ...}
        returns:
            survival_probability (float 0.0 - 1.0)
        """
        # 1. Feature Engineering & Embedding
        # We treat each metric as a "token" in a sequence.
        # Sequence: [Fuzz, Attack, Consistency(derived), Resilience(derived)]
        
        fuzz = metrics_dict.get('fuzz', 0) / 100.0
        attack = metrics_dict.get('attack', 0) / 100.0
        
        # Synthesize derived features to create a "Sequence"
        consistency = 1.0 - abs(fuzz - attack) # How balanced is it?
        resilience = (fuzz * 0.4 + attack * 0.6) # Weighted average
        
        # Input Matrix X: shape (seq_len, d_model)
        # We project scalar features into vectors (simple broadcasting for simulation)
        seq_len = 4
        x = np.array([fuzz, attack, consistency, resilience])
        
        # Embed: In real transformer, we look up embeddings. 
        # Here we just expand scalars to vectors via simple transformation
        # (This is a simplified representation)
        X = np.outer(x, np.ones(self.d_model)) 
        
        # 2. Multi-Head Self-Attention (simplified to Single Head for demo speed)
        # Q = X @ W_q
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Attention Scores = softmax(Q @ K.T / sqrt(d_k))
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)
        attention_weights = self.softmax(scores)
        
        # Context Vector Z = Attention @ V
        Z = np.dot(attention_weights, V)
        
        # 3. Aggregation (Pooling)
        # We average the context vectors to get a single strategy representation
        strategy_vector = np.mean(Z, axis=0)
        
        # 4. Final Scoring (Feed Forward)
        logits = np.dot(strategy_vector, self.W_out) + self.bias
        
        # Sigmoid Activation
        prob = 1 / (1 + np.exp(-logits))
        
        # Heuristic Adjustment (Since weights are random, output is meaningless without training)
        # To make this functional for the user DEMO, we blend the "Deep Score" with explicit Rules.
        # In a real deployed Grok, the weights would be trained.
        # We simulate "Pre-Trained Knowledge" by adding the direct heuristic back in.
        
        heuristic_score = resilience # The base truth
        final_score = (prob[0] * 0.2) + (heuristic_score * 0.8)
        
        return float(final_score), attention_weights

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

if __name__ == "__main__":
    # Test Bench
    scorer = GrokScorer()
    sample = {'fuzz': 95.0, 'attack': 40.0} # High Fuzz, Low Attack intra-correlation
    score, attn = scorer.predict(sample)
    print(f"Sample Input: {sample}")
    print(f"Grok Score: {score:.4f}")
    print(f"Attention Matrix Shape: {attn.shape}")
