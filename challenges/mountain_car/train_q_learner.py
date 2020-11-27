import numpy as np
from mountain_car_q_learning_agent import MountainCarQLearningAgent

np.random.seed(0)

agent = MountainCarQLearningAgent()

print('Training agent')
agent.train()

print('Saving agent learned data')
agent.save()