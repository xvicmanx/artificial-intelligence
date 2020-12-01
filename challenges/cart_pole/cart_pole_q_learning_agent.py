import os.path
from cart_pole_environment import CartPoleEnvironment
from common.q_learner import QLearner

persisted_models_dirname = os.path.dirname(__file__) + '/data'

class CartPoleQLearningAgent(QLearner):
  def __init__(self):
    super().__init__(
      CartPoleEnvironment(),
      persisted_models_dirname + '/' + 'cart_pole_q_learning_agent.sav',
      persisted_models_dirname + '/' + 'cart_pole_q_learner_reward_plot.png',
      episodes = 10000,
      iterations = 200,
      exploration_rate = 0.02,
      discount_factor = 0.9,
    )