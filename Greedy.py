from Agent import Agent
import numpy as np
from random import randrange

class Greedy(Agent):

  def __init__(self, n_arms, distribution, dist_name):
    super().__init__(n_arms, distribution, dist_name)
    print(n_arms, dist_name)

  def choose_arm(self): # return index of chosen bandit
    if self.iteration == 0:
      return randrange(self.n_arms) # pick random at first 
    
    else:
        if self.dist_name == "bernoulli": # bernoulli
            self.estimated_values = np.divide(self.rewards, self.arm_pulls, out=np.zeros_like(self.rewards), where=self.arm_pulls!=0) # ratio of value 
    
        return np.random.choice(np.flatnonzero(self.estimated_values == self.estimated_values.max())) # max val



