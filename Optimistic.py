from Greedy import Greedy
import Utils
import numpy as np

class Optimistic(Greedy):

    def __init__(self, n_arms, dist_name, initial_val):
        self.initial_val = initial_val
        super().__init__(n_arms, dist_name)
        self.label = "optimistic"

    def reset(self):
        self.iteration = 0
        self.rewards = np.zeros(self.n_arms)
        self.arm_pulls = np.zeros(self.n_arms) # fill with 0 
        self.total = 0
        self.generated_reward = 0
        self.estimated_values = np.full(self.n_arms, self.initial_val, dtype=float) # fill with 1
        if self.dist_name == "bernoulli":
            self.distribution = Utils.bernoulliDis(self.n_arms)
        else:
            self.distribution = Utils.gaussianDis(self.n_arms)    