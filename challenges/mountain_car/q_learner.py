import numpy as np
from helpers import pick_action
from agent import Agent

class QLearner(Agent):
  min_learning_rate = 0.001

  def __init__(
    self,
    environment,
    agent_persist_file_path,
    learning_rate = 1.0,
    discount_factor = 1.0,
    episodes = 100,
    iterations = 10000,
    exploration_rate = 0.02
  ):
    super().__init__(environment)

    states = environment.get_states_dimension()
    actions = environment.get_number_of_actions()

    self.__agent_persist_file_path = agent_persist_file_path
    self.__learning_rate = learning_rate
    self.__discount_factor = discount_factor
    self.__episodes = episodes
    self.__iterations = iterations
    self.__exploration_rate = exploration_rate
    self.__q_values = np.zeros(states + (actions, ))
    self.__policy = None
    

  def train(self):
    q = self.__q_values
    env = self._environment
    df = self.__discount_factor
    initial_learning_rate = self.__learning_rate

    for e in range(self.__episodes):
      state = env.get_initial_state()

      print('Episode ' + str(e + 1))
      
      lr = max(
        QLearner.min_learning_rate,
        initial_learning_rate * (1 - e / self.__episodes),
      )

      for i in range(self.__iterations):
        action = self.__pick_action(state)
        observation, reward, done, _ = env.apply_action(action)
        new_state = env.get_state(observation)
        q_new = reward + df * np.max(q[new_state])
        q[state][action] = (1 - lr) * q[state][action] + lr * q_new
        state = new_state

        if done:
          break

    self.__policy = self.__get_policy()

  """Gets the best action from the observation using the learned policy
  """
  def get_action(self, observation):
    state = self._environment.get_state(observation)
    return self.__policy[state]

  """Save a serialized version of the model to a file
  """
  def save(self):
    self._save_object(self.__q_values, self.__agent_persist_file_path)

  """Loads the serialized version of the model from file
  """
  def load(self):
    self.__q_values = self._load_object(self.__agent_persist_file_path)
    self.__policy = self.__get_policy()

  def __pick_action(self, state):
    return pick_action(
      self.__q_values[state],
      self._environment.get_number_of_actions(),
      self.__exploration_rate,
    )

  def __get_policy(self):
    return np.argmax(
      self.__q_values,
      axis = len(self._environment.get_states_dimension()),
    )

# Q(s, a) = (1 - learning_rate) * Q(s, a) + learning_rate * Q_new(s, a)
# Q_new(s, a) = reward(s, a) + discount_factor * max_a' Q(s', a') 
