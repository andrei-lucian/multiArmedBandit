from Agent import Agent
import numpy as np
import math

class SoftMax(Agent):

  def __init__(self, n_arms, dist_name, tau):
    super().__init__(n_arms, dist_name)
    self.tau = tau
    self.label = "softmax, tau = " + str(tau)

  def reset(self): # reset function
    super().reset()
    self.probabilities = np.zeros(self.n_arms)
    #print("estimated values at start: ", self.estimated_values)

  def choose_arm(self): # return index of chosen bandit
    if self.iteration < self.n_arms:
      return self.iteration
      
    else:
      self.calculate_probs()
      return np.random.choice(self.n_arms, p=self.probabilities) # max val
      
  def calculate_probs(self):
   # Calculate Softmax probabilities based on each round
    z = sum([math.exp(v / self.tau) for v in self.estimated_values])
    self.probabilities = [math.exp(v / self.tau) / z for v in self.estimated_values]