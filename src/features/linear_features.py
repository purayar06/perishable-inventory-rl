"""
Linear feature extractor for function-approximation agents.
"""

import numpy as np
from typing import Tuple


class LinearFeatureExtractor:
    """Converts a state tuple into a normalised feature vector.

    Features (D + 3 total):
        [0..D-1]  normalised bucket fills  n_i / N_max
        [D]       total inventory ratio    sum(n_i) / (N_max * D)
        [D+1]     near-expiry units        n_0 / N_max  (items about to expire)
        [D+2]     average remaining shelf-life (normalised to [0,1])
    """

    def __init__(self, shelf_life: int, max_inventory: int):
        self.D = shelf_life
        self.N_max = max_inventory
        self.num_features = self.D + 3

    def __call__(self, state: Tuple[int, ...]) -> np.ndarray:
        feats = np.zeros(self.num_features)
        total = sum(state)
        for i, n in enumerate(state):
            feats[i] = n / self.N_max  # normalised fill
        feats[self.D] = total / (self.N_max * self.D)  # total inventory ratio
        feats[self.D + 1] = state[0] / self.N_max  # near-expiry units
        # average remaining shelf-life: sum((i+1)*n_i) / total / D
        feats[self.D + 2] = (
            sum((i + 1) * state[i] for i in range(self.D)) / max(total, 1) / self.D
        )
        return feats
