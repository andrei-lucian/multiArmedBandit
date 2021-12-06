from Agent import Agent
import numpy as np

class ActionPreferences(Agent):

    def __init__(self, n_arms, dist_name, alpha):
        super().__init__(n_arms, dist_name)
        self.label = "Action Preferences, alpha = " + str(alpha)
        self.alpha = alpha

    def reset(self): # reset function
        super().reset()
        self.H = np.zeros(self.n_arms)

    def choose_arm(self): # return index of chosen bandit
        if self.iteration < self.n_arms:
            return self.iteration
        
        else:
            self.update_H(self.chosen_arm)
            return np.random.choice(self.n_arms, p=self.probabilities) # max val

    def update_H(self, chosen_arm):
        self.probabilities = np.exp(self.H)/np.sum(np.exp(self.H), axis=0)
        current_mean_reward = self.total/self.arm_pulls.sum()
        arm_bonus = self.alpha*(self.generated_reward - current_mean_reward)*(1-self.probabilities[chosen_arm])
        self.H[chosen_arm] += arm_bonus

        for arm in range(self.n_arms):
            if arm != chosen_arm:
                rest_bonus = self.alpha*(self.generated_reward - current_mean_reward)*(self.probabilities[arm])
                self.H[arm] += rest_bonus