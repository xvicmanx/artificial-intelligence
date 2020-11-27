import os.path
from mountain_car_environment import MountainCarEnvironment
from deep_q_learner import DeepQLearner

persisted_models_dirname = os.path.dirname(__file__) + '/data'

class MountainCarDeepQLearningAgent(DeepQLearner):
  def __init__(self):
    super().__init__(
      MountainCarEnvironment(),
      persisted_models_dirname + '/' + 'mountain_car_deep_q_learner.sav'
    )