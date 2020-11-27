import numpy as np
from cart_pole_q_learning_agent import CartPoleQLearningAgent

np.random.seed(0)

agent = CartPoleQLearningAgent()

print('Training agent')
agent.train()

print('Saving agent learned data')
agent.save()