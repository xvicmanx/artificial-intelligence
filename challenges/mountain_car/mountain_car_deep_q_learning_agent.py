import os.path
from environment import MountainCarEnvironment
from deep_q_learner import DeepQLearner

dirname = os.path.dirname(__file__)
persisted_models_dirname = 'data'

class MountainCarDeepQLearningAgent(DeepQLearner):
  def __init__(self):
    super().__init__(
      MountainCarEnvironment(),
      persisted_models_dirname + '/' + 'mountain_car_deep_q_learner.sav'
    )