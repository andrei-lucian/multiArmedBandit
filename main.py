from Agent import Agent
from Greedy import Greedy
from EpsilonGreedy import EpsilonGreedy
from Optimistic import Optimistic
from UCB import UCB
from SoftMax import SoftMax
from ActionPreferences import ActionPreferences
import Utils
import argparse

print('For help run program with flag -h \n')

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--n_bandits', type=int, help='<int> The number of bandits to use.')
parser.add_argument('--distribution', type=str, help='<string> Type "gaussian" or "bernoulli".')
parser.add_argument('--time_steps', type=int, help='<int> The number of time steps to run each repetition for.')
parser.add_argument('--repetitions', type=int, help='<int> The number of repetitions of each experiment.')
parser.add_argument('--experiment_name', type=str, help='<string> The name that the .png file will be saved as (in the "plots" folder). Note that using the same name will overwrite the previous file.')

args = parser.parse_args()
n_bandits = args.n_bandits
distribution = args.distribution
time_steps = args.time_steps
repetitions = args.repetitions
fig_name = args.experiment_name

if distribution == "bernoulli":
    epsilon = 0.15
    c = 1
    optimistic_val = 1
    tau = 0.1
    alpha = 0.5
elif distribution == "gaussian":
    epsilon = 0.15
    c = 1
    optimistic_val = 20
    tau = 0.2
    alpha = 0.1

params = n_bandits, distribution

g = Greedy(*params)
e = EpsilonGreedy(*params, epsilon)
u =  UCB(*params, c)
o = Optimistic(*params, optimistic_val)
s = SoftMax(*params, tau)
a = ActionPreferences(*params, alpha)
agents = [g, e, u, o, s, a]

reward, percentage = Utils.concatenate_experiments(agents, time_steps, repetitions)
Utils.plot_learning(reward, percentage, fig_name)
