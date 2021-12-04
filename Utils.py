import random 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

def find_best_bandit(agent):
    d = np.asarray(agent.distribution)
    return np.random.choice(np.flatnonzero(d == d.max()))

def bernoulliDis(n_bandits):
    return [random.uniform(0, 1) for _ in range(n_bandits)]

def gaussianDis(n_bandits):
    return [float(np.random.normal(5, 1.5, 1)) for _ in range(n_bandits)]

def run_experiment(agent, time_steps, repetitions):

  all_rewards = np.zeros((repetitions, time_steps)) # 2d array to keep track of cumulative reward
  best_action_counter = 0 # keep track of the number of times the best bandit is selected
  best_action_percentage = np.zeros((repetitions, time_steps))

  for rep in range(repetitions): # repetitions
    best_bandit = find_best_bandit(agent) # find the index of the best bandit

    for step in range(time_steps): # time steps 
      current_total_reward, chosen_index = agent.update() # get current reward and chosen index of agent 
      all_rewards[rep][step] = current_total_reward/(step+1)
      if chosen_index == best_bandit: # increment counter if best action is chosen 
        best_action_counter += 1 
      best_action_percentage[rep][step] = (best_action_counter*100)/(step+1)
      
    best_action_counter = 0
    agent.reset()

  mean_reward = all_rewards.mean(axis=0) # mean reward over all repetitions
  mean_percentage = best_action_percentage.mean(axis=0)
  print(best_action_percentage)
  print(mean_percentage)
  return mean_reward, mean_percentage

def concatenate_experiments(agents, time_steps, repetitions):
  labels = []
  agents_rewards = []
  agents_percentages = []

  for agent in agents:
    labels.append(agent.get_label())
    mean_reward, mean_percentage = run_experiment(agent, time_steps, repetitions)
    agents_rewards.append(mean_reward)
    agents_percentages.append(mean_percentage)

  agents_rewards = np.array(agents_rewards)
  agents_rewards = agents_rewards.swapaxes(0,1)
  agents_rewards = pd.DataFrame(data = agents_rewards, columns=labels)

  agents_percentages = np.array(agents_percentages)
  agents_percentages = agents_percentages.swapaxes(0,1)
  agents_percentages = pd.DataFrame(data = agents_percentages, columns=labels)

  return agents_rewards, agents_percentages

def plot_learning(rewards, percentages, fig_name):
  path = os.path.abspath(os.getcwd())
  path = os.path.join(path, 'plots\\')   
  if not os.path.isdir(path):
    os.makedirs(path)
  
  r_path = os.path.join(path, fig_name + "_r")
  p_path = os.path.join(path, fig_name + "_p")

  sns.set(rc={'figure.figsize':(10,7)})
  reward_plot = sns.lineplot(data=rewards)
  reward_plot.set(xlabel='Episode', ylabel='Total reward')
  reward_plot.get_figure().savefig(r_path)
  reward_plot.get_figure().clf()

  percentage_plot = sns.lineplot(data=percentages)
  percentage_plot.set(xlabel='Episode', ylabel='Percentage of times the best action is chosen')
  percentage_plot.get_figure().savefig(p_path)
  percentage_plot.get_figure().clf()
