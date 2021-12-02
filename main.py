from Agent import Agent
from Greedy import Greedy
from EpsilonGreedy import EpsilonGreedy
from Optimistic import Optimistic
from UCB import UCB
from SoftMax import SoftMax
import Utils

b = Utils.bernoulliDis(5)
print(b)
g = Utils.gaussianDis(5)
print(g)
params = 5, g, "gaussian"

o = Optimistic(*params)
print("\n")

s = SoftMax(*params, 3)

agents = [o]
r, p = Utils.concatenate_experiments(agents, 20, 1)
Utils.plot_learning(r,p)