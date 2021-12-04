from Agent import Agent
from Greedy import Greedy
from EpsilonGreedy import EpsilonGreedy
from Optimistic import Optimistic
from UCB import UCB
from SoftMax import SoftMax
from ActionPreferences import ActionPreferences
import Utils


# g = Utils.gaussianDis(5)
# print(g)
params = 5, "gaussian"
# params = 5, g, "gaussian"

# g = Greedy(*params)
# e = EpsilonGreedy(*params, 0.8)
# u =  UCB(*params, 1)
# o = Optimistic(*params)
# s = SoftMax(*params, 0.3)
# a = ActionPreferences(*params, 0.1)

u1 = UCB(*params, 0.1)
u2 = UCB(*params, 0.5)
u3 = UCB(*params, 1)
u4 = UCB(*params, 3)
u5 = UCB(*params, 5)

e1 = EpsilonGreedy(*params, 0.99)
e2 = EpsilonGreedy(*params, 0.9)
e3 = EpsilonGreedy(*params, 1)
e4 = EpsilonGreedy(*params, 0.85)
e5 = EpsilonGreedy(*params, 0.75)
e6 = EpsilonGreedy(*params, 0.7)

agents = [u1, u2, u3, u4, u5]
#agents = [e1, e2, e3, e4, e5, e6]
# agents = [a]
r, p = Utils.concatenate_experiments(agents, 2000, 10)
Utils.plot_learning(r,p, "eps 3 runs")