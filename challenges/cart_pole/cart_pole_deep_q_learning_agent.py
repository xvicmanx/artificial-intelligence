import os.path
from cart_pole_environment import CartPoleEnvironment
from deep_q_learner import DeepQLearner

persisted_models_dirname = os.path.dirname(__file__) + '/data'

class CartPoleDeepQLearningAgent(DeepQLearner):
  def __init__(self):
    super().__init__(
      CartPoleEnvironment(),
      persisted_models_dirname + '/' + 'cart_pole_deep_q_learning_agent.sav',
      episodes = 3000,
      discount_factor = 0.99
    )