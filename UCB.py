from Agent import Agent
import numpy as np
import math

class UCB(Agent):

  def __init__(self, n_arms, dist_name, c):
    super().__init__(n_arms, dist_name)
    self.label = "UCB, c = " + str(c)
    self.c = c

  def reset(self): # reset function
    super().reset()
    self.ucb_values = np.zeros(self.n_arms)

  def choose_arm(self): # return index of chosen bandit
    if self.iteration < self.n_arms:
      return self.iteration
      
    else:
      return self.increment_ucb() # max val

  def increment_ucb(self):
    for arm in range(self.n_arms):
      bonus = self.c * math.sqrt((math.log(self.iteration)) / float(self.arm_pulls[arm])) if self.arm_pulls[arm] != 0 else 0
      self.ucb_values[arm] = self.estimated_values[arm] + bonus
    #print(self.ucb_values)
    return np.random.choice(np.flatnonzero(self.ucb_values == self.ucb_values.max()))