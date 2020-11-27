import os.path
from environment import MountainCarEnvironment
from q_learner import QLearner

dirname = os.path.dirname(__file__)
persisted_models_dirname = 'data'

class MountainCarQLearningAgent(QLearner):
  def __init__(self):
    super().__init__(
      MountainCarEnvironment(),
      persisted_models_dirname + '/' + 'mountain_car_q_learner.sav'
    )