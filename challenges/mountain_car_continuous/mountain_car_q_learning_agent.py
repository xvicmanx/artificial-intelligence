import os.path
from mountain_car_environment import MountainCarEnvironment
from common.q_learner import QLearner

persisted_models_dirname = os.path.dirname(__file__) + '/data'

class MountainCarQLearningAgent(QLearner):
  def __init__(self):
    super().__init__(
      MountainCarEnvironment(),
      persisted_models_dirname + '/' + 'mountain_car_q_learner.sav'
      persisted_models_dirname + '/' + 'mountain_car_q_learner_reward_plot.png',
    )