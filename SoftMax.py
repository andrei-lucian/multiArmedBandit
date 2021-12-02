from Agent import Agent
import numpy as np
import math
import random

class SoftMax(Agent):

  def __init__(self, n_arms, distribution, dist_name, tau):
    super().__init__(n_arms, distribution, dist_name)
    self.tau = tau

  def reset(self): # reset function
    super().reset()
    print("reset from SoftMax class called")
    self.probabilities = np.zeros(self.n_arms)
    #print("estimated values at start: ", self.estimated_values)

  def choose_arm(self): # return index of chosen bandit
    if self.iteration < self.n_arms:
      return self.iteration
      
    else:
        if self.dist_name == "bernoulli": # bernoulli
            self.estimated_values = np.divide(self.rewards, self.arm_pulls, out=np.zeros_like(self.rewards), where=self.arm_pulls!=0) # ratio of value 
        probs = self.calculate_probs()
        return self.arm_selection(probs) # max val
    
  # Arm selection based on Softmax probability
  def arm_selection(self, probs):
    z = random.random()
    cum_prob = 0.0
    
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        
        if cum_prob > z:
            return i
    return len(probs) - 1
      
  def calculate_probs(self):
   # Calculate Softmax probabilities based on each round
    z = sum([math.exp(v / self.tau) for v in self.estimated_values])
    probs = [math.exp(v / self.tau) / z for v in self.estimated_values]
    # Use categorical_draw to pick arm
    return probs