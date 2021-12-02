import abc
import random
import numpy as np

class Agent:
    def __init__(self, n_arms, distribution, dist_name):
        self.dist_name = dist_name
        assert (self.dist_name == "bernoulli") or (self.dist_name == "gaussian"), "dist_name should be gaussian or bernoulli"
        self.n_arms = n_arms
        self.distribution = distribution
        self.reset()

    def get_label(self):
        return self.label
    
    def reset(self):
        self.iteration = 0
        self.estimated_values = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms) # fill with 0 
        self.arm_pulls = np.zeros(self.n_arms) # fill with 0 
        self.total = 0
    
    @abc.abstractmethod
    def choose_arm(self):
        return

    def update(self):
        chosen_arm = self.choose_arm()
        print("chosen arm: ", chosen_arm)
        val = self.distribution[chosen_arm]
        self.arm_pulls[chosen_arm] += 1

        if self.dist_name == "bernoulli": # bernoulli
            rand = random.random() # random val for bernoulli chance
            reward = 1 if rand < val else 0 
            print("reward gained: ", reward)
            if reward:
                self.rewards[chosen_arm] += 1 # increment successes 
                self.total = self.rewards.sum()
        else:
            self.estimated_values[chosen_arm] = val
            print("reward gained: ", val)
            self.total += val
        self.debug_prints()
        self.iteration += 1
        return self.total, chosen_arm # return total reward and index

    def debug_prints(self):
        print("iteration: ", self.iteration)
        print("arm pulls: ", self.arm_pulls)
        print("estimated values: ", self.estimated_values)
        print("rewards: ", self.rewards)
        print("total reward: ", self.total)
        print("\n")
