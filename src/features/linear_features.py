"""
Linear feature extractor for potential function-approximation extensions.
"""

import numpy as np
from typing import Tuple


class LinearFeatureExtractor:
    """Converts a state tuple into a normalised feature vector."""

    def __init__(self, shelf_life: int, max_inventory: int):
        self.D = shelf_life
        self.N_max = max_inventory
        # features: normalised bucket fills + total inventory + freshness ratio
        self.num_features = self.D + 2

    def __call__(self, state: Tuple[int, ...]) -> np.ndarray:
        feats = np.zeros(self.num_features)
        total = sum(state)
        for i, n in enumerate(state):
            feats[i] = n / self.N_max  # normalised fill
        feats[self.D] = total / (self.N_max * self.D)  # total fill ratio
        feats[self.D + 1] = (
            state[-1] / total if total > 0 else 0.0
        )  # freshness ratio
        return feats
