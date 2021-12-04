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
params = 5, "bernoulli"
# params = 5, g, "gaussian"

# g = Greedy(*params)
# e = EpsilonGreedy(*params, 0.8)
# u =  UCB(*params, 1)
# o = Optimistic(*params)
# s = SoftMax(*params, 0.3)
# a = ActionPreferences(*params, 0.1)

# u1 = UCB(*params, 1.5)
# u2 = UCB(*params, 0.5)
# u3 = UCB(*params, 1)
# u4 = UCB(*params, 3)
# u5 = UCB(*params, 2)

# e1 = EpsilonGreedy(*params, 0.99)
# e2 = EpsilonGreedy(*params, 0.9)
# e3 = EpsilonGreedy(*params, 1)
# e4 = EpsilonGreedy(*params, 0.85)
# e5 = EpsilonGreedy(*params, 0.75)
# e6 = EpsilonGreedy(*params, 0.7)
# e7 = EpsilonGreedy(*params, 0.95)

s1 = SoftMax(*params , 0.1) 
s2 = SoftMax(*params , 0.5) 
s3 = SoftMax(*params , 0.2) 
s4 = SoftMax(*params , 0.05) 
s5 = SoftMax(*params , 0.3) 
s6 = SoftMax(*params , 0.8) 

a1 = ActionPreferences(*params, 0.1)
a2 = ActionPreferences(*params, 0.5)
a3 = ActionPreferences(*params, 1)
a4 = ActionPreferences(*params, 1.5)
a5 = ActionPreferences(*params, 3)
a6 = ActionPreferences(*params, 5)

# agents = [u1, u2, u3, u4, u5]
#agents = [e1, e2, e3, e4, e5, e6, e7]
agents = [s1, s2, s3, s4, s5, s6]
agents1 = [a1, a2, a3, a4, a5, a6]
# r, p = Utils.concatenate_experiments(agents, 2000, 100)
# Utils.plot_learning(r,p, "softmax 100 runs")

x, y = Utils.concatenate_experiments(agents1, 2000, 100)
Utils.plot_learning(x,y, "preferences 100 runs")