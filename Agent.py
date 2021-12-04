import abc
import random
import numpy as np
import Utils

class Agent:
    def __init__(self, n_arms, dist_name):
        self.dist_name = dist_name
        assert (self.dist_name == "bernoulli") or (self.dist_name == "gaussian"), "dist_name should be gaussian or bernoulli"
        self.n_arms = n_arms
        self.generated_reward = 0
        self.chosen_arm = None
        self.reset()

    def get_label(self):
        return self.label
    
    def reset(self):
        self.iteration = 0
        self.estimated_values = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)
        self.arm_pulls = np.zeros(self.n_arms) 
        self.total = 0
        self.generated_reward = 0
        if self.dist_name == "bernoulli":
            self.distribution = Utils.bernoulliDis(self.n_arms)
        else: 
            self.distribution = Utils.gaussianDis(self.n_arms)
    
    @abc.abstractmethod
    def choose_arm(self):
        return

    def update(self):
        self.estimated_values = np.divide(self.rewards, self.arm_pulls, out=np.zeros_like(self.rewards), where=self.arm_pulls!=0) # ratio of value 
        self.chosen_arm = self.choose_arm()
        #print("chosen arm: ", chosen_arm)
        val = self.distribution[self.chosen_arm]
        self.arm_pulls[self.chosen_arm] += 1

        if self.dist_name == "bernoulli": # bernoulli
            rand = random.random() # random val for bernoulli chance
            self.generated_reward = 1 if rand < val else 0 
            #print("reward gained: ", reward)
            if self.generated_reward:
                self.rewards[self.chosen_arm] += 1 # increment successes 
                self.total = self.rewards.sum()
        else:
            self.generated_reward = float(np.random.normal(val, 1.5, 1))
            self.rewards[self.chosen_arm] += self.generated_reward
            #print("reward gained: ", generated_reward)
            self.total += self.generated_reward
        #self.debug_prints()
        self.iteration += 1
        return self.total, self.chosen_arm # return total reward and index

    def debug_prints(self):
        print("iteration: ", self.iteration)
        print("arm pulls: ", self.arm_pulls)
        print("estimated values: ", self.estimated_values)
        print("rewards: ", self.rewards)
        print("total reward: ", self.total)
        print("\n")
