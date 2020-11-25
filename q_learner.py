import numpy as np
import gym
from helpers import softmax

class QLearner:
  min_learning_rate = 0.001

  def __init__(
    self,
    environment,
    learning_rate = 1.0,
    discount_factor = 1.0,
    episodes = 100,
    iterations = 10000,
    randomness = 0.02
  ):
    states = environment.get_states_dimension()
    actions = environment.get_number_of_actions()

    self.__learning_rate = learning_rate
    self.__discount_factor = discount_factor
    self.__q_values = np.zeros(states + (actions, ))
    self.__environment = environment
    self.__episodes = episodes
    self.__iterations = iterations
    self.__randomness = randomness
    

  def train(self):
    q = self.__q_values
    env = self.__environment
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

    return np.argmax(q, axis = len(env.get_states_dimension()))


  """Picks the next action based on a probability distribution created
    from the q-values or randomly.
    The selection would depend on the value of the randomness.
  """
  def __pick_action(self, state):
    q = self.__q_values
    env = self.__environment
    actions = env.get_number_of_actions()

    should_pick_random = np.random.uniform(0, 1) < self.__randomness

    if should_pick_random:
      return np.random.choice(actions)

    return np.random.choice(actions, p = softmax(q[state]))


# Q(s, a) = (1 - learning_rate) * Q(s, a) + learning_rate * Q_new(s, a)
# Q_new(s, a) = reward(s, a) + discount_factor * max_a' Q(s', a') 
