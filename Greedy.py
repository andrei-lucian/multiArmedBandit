from Agent import Agent
import numpy as np
from random import randrange

class Greedy(Agent):

  def __init__(self, n_arms, dist_name):
    super().__init__(n_arms, dist_name)
    self.label = "greedy"

  def choose_arm(self): # return index of chosen bandit
    if self.iteration == 0:
      return randrange(self.n_arms) # pick random at first 
    else:
      return np.random.choice(np.flatnonzero(self.estimated_values == self.estimated_values.max())) # max val



