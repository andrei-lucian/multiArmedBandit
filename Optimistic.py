from Greedy import Greedy
import numpy as np

class Optimistic(Greedy):

    def __init__(self, n_arms, distribution, dist_name):
        super().__init__(n_arms, distribution, dist_name)
        self.label = "optimistic"

    def reset(self):
        if self.dist_name == "bernoulli":
            self.estimated_values = np.ones(self.n_arms) # fill with 1
        else:
            self.estimated_values = np.full(self.n_arms, 11.0)

        self.iteration = 0
        self.rewards = np.zeros(self.n_arms)
        self.arm_pulls = np.zeros(self.n_arms) # fill with 0 
        self.total = 0

    