"""
Compound scoring: combines importance and recency into a single priority score.

This is the core prioritization mechanism of FinMem. The compound score
determines the ordering of memories within a layer, which entries survive
cleanup, and which qualify for jumping to other memory layers.

Compound scoring formula (preserved exactly from FinMem):
    1. partial_score = recency_score + importance_score / 100
       (scales importance to [0, 1] range and adds recency)
    2. final_score  = similarity_score + partial_score
       (similarity from vector search is added when querying)

The final score blends three signals:
    - recency_score:    how fresh the memory is (0 to 1)
    - importance_score: how valuable the memory is (capped at 100)
    - similarity_score: semantic relevance to query (from vector search)
"""


class CompoundScorer:
    """
    Computes compound scores from recency and importance.

    The compound score is the primary ranking signal within each memory layer.
    """

    def recency_and_importance_score(
        self, recency_score: float, importance_score: float
    ) -> float:
        """
        Combine recency and importance into a partial compound score.

        Formula: recency_score + importance_score / 100
        (importance is capped at 100 before scaling)

        Args:
            recency_score:    Score in [0, 1], where 1 = most recent.
            importance_score: Score in [0, ~100], where higher = more important.

        Returns:
            Partial compound score in [0, 2].
        """
        importance_score = min(importance_score, 100)
        return recency_score + importance_score / 100

    def merge_score(
        self, similarity_score: float, recency_and_importance: float
    ) -> float:
        """
        Merge semantic similarity with recency+importance for final ranking.

        Formula: similarity_score + recency_and_importance

        Used after vector search to produce the final query result ranking.

        Args:
            similarity_score:      Cosine similarity from vector search.
            recency_and_importance: Output of recency_and_importance_score().

        Returns:
            Final compound score for ranking.
        """
        return similarity_score + recency_and_importance

    def compute_full(
        self,
        recency_score: float,
        importance_score: float,
        similarity_score: float = 0.0,
    ) -> float:
        """
        Compute the full compound score in one call.

        Convenience wrapper: recency_and_importance_score + merge_score.
        """
        partial = self.recency_and_importance_score(recency_score, importance_score)
        return self.merge_score(similarity_score, partial)
