import numpy as np
from mountain_car import MountainCar
from q_learner import QLearner
from deep_q_learner import DeepQLearner
from helpers import show_result

np.random.seed(0)
environment = MountainCar()

model = DeepQLearner(environment)
# model.train()

# print('Saving model')
# model.save('MountainCarDeepQLearner.sav')
model.load('MountainCarDeepQLearner.sav')


# model = QLearner(environment)
# model.train()

# model.save('MountainCarQLearner.sav')
# model.load('MountainCarQLearner.sav')

show_result(environment, model)