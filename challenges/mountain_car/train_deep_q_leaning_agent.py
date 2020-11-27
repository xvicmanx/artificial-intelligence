import numpy as np
from mountain_car_deep_q_learning_agent import MountainCarDeepQLearningAgent

np.random.seed(0)

agent = MountainCarDeepQLearningAgent()

print('Training agent')
agent.train()

print('Saving agent learned data')
agent.save()