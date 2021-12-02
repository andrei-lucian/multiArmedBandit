from Agent import Agent
from Greedy import Greedy
from EpsilonGreedy import EpsilonGreedy
from Optimistic import Optimistic
from UCB import UCB
from SoftMax import SoftMax
import Utils

b = Utils.bernoulliDis(5)
print(b)
params = 5, b, "bernoulli"
# a = Agent(5, b, "bernoulli")
# g = Greedy(5, b, "bernoulli")

# for _ in range(5):  
#     g.update()

# e = EpsilonGreedy(5, b, "bernoulli", 0.9)
o = Optimistic(*params)
s = SoftMax(*params, 3)