from Greedy import Greedy
import numpy as np

class Optimistic(Greedy):

    def __init__(self, n_arms, distribution, dist_name):
        super().__init__(n_arms, distribution, dist_name)
    
    def reset(self):
        print("reset from Optimistic class called")
        if self.dist_name == "bernoulli":
            self.rewards = np.ones(self.n_arms) # fill with 0 
        else:
            self.rewards = np.full(self.n_arms, 11)

        self.estimated_values = np.zeros(self.n_arms)
        self.arm_pulls = np.zeros(self.n_arms) # fill with 0 
        self.total = 0

    