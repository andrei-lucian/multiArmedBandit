from Greedy import Greedy
import random
from random import randrange
import numpy as np

class EpsilonGreedy(Greedy):

    def __init__(self, n_arms, dist_name, eps):
        super().__init__(n_arms, dist_name)
        self.epsilon = eps
        self.label = "epsilon greedy, eps = " + str(eps)
    
    def choose_arm(self): # return index of chosen bandit
        if self.iteration == 0:
            return randrange(self.n_arms) # pick random at first    
        
        else:
            epsilon_random = random.random() # generate random val for epsilon
            
            if epsilon_random < self.epsilon:
                return np.random.choice(np.flatnonzero(self.estimated_values == self.estimated_values.max())) # max val
            else: 
                return randrange(self.n_arms) # random val 
