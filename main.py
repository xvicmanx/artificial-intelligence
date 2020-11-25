import numpy as np
from mountain_car import MountainCar
from q_learner import QLearner
from helpers import show_result

np.random.seed(0)
environment = MountainCar()
model = QLearner(environment)
model.train()

show_result(environment, model)