from Greedy import Greedy
import random
from random import randrange
import numpy as np

class EpsilonGreedy(Greedy):

    def __init__(self, n_arms, distribution, dist_name, eps):
        super().__init__(n_arms, distribution, dist_name)
        self.epsilon = eps
    
    def choose_arm(self): # return index of chosen bandit
        if self.iteration == 0:
            return randrange(self.n_arms) # pick random at first    
        
        else:
            epsilon_random = random.random() # generate random val for epsilon

            if self.dist_name == "bernoulli": # bernoulli
                self.estimated_values = np.divide(self.rewards, self.arm_pulls, out=np.zeros_like(self.rewards), where=self.arm_pulls!=0) # ratio of value 
            
            if epsilon_random < self.epsilon:
                return np.random.choice(np.flatnonzero(self.estimated_values == self.estimated_values.max())) # max val
            else: 
                return randrange(self.n_arms) # random val 
