
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - CORRELATION - %(levelname)s - %(message)s')
logger = logging.getLogger("CorrelationManager")

class CorrelationManager:
    """
    Calculates signal similarity between strategies.
    Ensures Operation Phoenix maintains diversity.
    """
    def __init__(self):
        self.reference_masks = {} # {name: buy_mask_series}

    def add_reference(self, name: str, buy_mask: pd.Series):
        """Adds a verified elite's signal series for comparison."""
        self.reference_masks[name] = buy_mask

    def check_redundancy(self, name: str, new_buy_mask: pd.Series, threshold=0.70) -> Dict[str, Any]:
        """
        Checks if the new strategy is too similar to any existing reference.
        Returns result dict with 'is_redundant' and 'max_corr'.
        """
        if not self.reference_masks:
            return {"is_redundant": False, "max_corr": 0.0, "reason": "No references"}

        max_corr = 0.0
        most_similar = None

        for ref_name, ref_mask in self.reference_masks.items():
            # Align lengths if necessary
            s1 = ref_mask.astype(int)
            s2 = new_buy_mask.astype(int)
            
            # If signals are all zeros, correlation will be NaN. Treat as zero (not redundant but useless)
            if s1.sum() == 0 or s2.sum() == 0:
                correlation = 0.0
            elif s1.equals(s2):
                return {"is_redundant": True, "max_corr": 1.0, "similar_to": ref_name}
            else:
                correlation = s1.corr(s2)
                if np.isnan(correlation):
                    correlation = 0.0

            if correlation > max_corr:
                max_corr = correlation
                most_similar = ref_name

        is_redundant = max_corr > threshold
        return {
            "is_redundant": is_redundant,
            "max_corr": max_corr,
            "similar_to": most_similar
        }

    def generate_full_matrix_report(self, masks: Dict[str, pd.Series]) -> pd.DataFrame:
        """Generates a full correlation matrix for audit."""
        df = pd.DataFrame(masks).astype(int)
        return df.corr()
