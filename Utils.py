import random 
import numpy as np
import pandas as pd 
import seaborn as sns

def find_best_bandit(agent):
    d = np.asarray(agent.distribution)
    return np.random.choice(np.flatnonzero(d == d.max()))

def bernoulliDis(n_bandits):
    return [random.uniform(0, 1) for _ in range(n_bandits)]

def gaussianDis(n_bandits):
    return [float(np.random.normal(5, 1.5, 1)) for _ in range(n_bandits)]

def run_experiment(agent, time_steps, repetitions):

  all_rewards = np.zeros((repetitions, time_steps)) # 2d array to keep track of cumulative reward
  best_bandit = find_best_bandit(agent) # find the index of the best bandit
  best_action_counter = np.zeros(repetitions) # keep track of the number of times the best bandit is selected

  for rep in range(repetitions): # repetitions
    
    for step in range(time_steps): # time steps 
      current_total_reward, chosen_index = agent.update() # get current reward and chosen index of agent 
      all_rewards[rep][step] = current_total_reward
      if chosen_index == best_bandit: # increment counter if best action is chosen 
        best_action_counter[rep] += 1 
    agent.reset()

  best_action_freq = best_action_counter*100/time_steps # percentage of time steps the best action was chosen
  mean_best_action_freq = best_action_freq.mean()
  mean_reward = all_rewards.mean(axis=0) # mean reward over all repetitions
  print("Mean percentage of times that the best action is taken:", mean_best_action_freq)
  return best_action_freq, mean_reward

def plot_learning(agents, labels, time_steps, repetitions):
  agents_rewards = []
  for i, agent in enumerate(agents):
    best_action_freq, mean_reward = run_experiment(agent, time_steps, repetitions)
    agents_rewards.append(mean_reward)
  x = np.linspace(0, time_steps, time_steps, dtype=int)
  agents_rewards = np.array(agents_rewards)
  agents_rewards = agents_rewards.swapaxes(0,1)
  df = pd.DataFrame(data = agents_rewards, columns=labels)
  sns.set(rc={'figure.figsize':(10,7)})
  fig = sns.lineplot(data=df)
  fig.set(xlabel='Episode', ylabel='Total reward')