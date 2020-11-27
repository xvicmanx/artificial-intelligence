import numpy as np
from cart_pole_deep_q_learning_agent import CartPoleDeepQLearningAgent

np.random.seed(0)

agent = CartPoleDeepQLearningAgent()

print('Training agent')
agent.train()

print('Saving agent learned data')
agent.save()