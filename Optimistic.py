from Greedy import Greedy
import Utils
import numpy as np

class Optimistic(Greedy):

    def __init__(self, n_arms, dist_name, initial_val):
        self.initial_val = initial_val
        super().__init__(n_arms, dist_name)
        self.label = "optimistic"

    def reset(self):
        super().reset()
        self.estimated_values = np.full(self.n_arms, self.initial_val, dtype=float) # fill with 1
