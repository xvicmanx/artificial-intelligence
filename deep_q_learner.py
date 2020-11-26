import numpy as np
import random
import gym
import joblib
from helpers import softmax

from sklearn.neural_network import MLPRegressor

class DeepQLearner:
  def __init__(
    self,
    environment,
    episodes = 10,
    iterations = 1000,
    randomness = 0.2,
    discount_factor = 0.9,
    batch_size = 100,
    memory_size = 100000,
  ):
    self.__environment = environment
    self.__episodes = episodes
    self.__iterations = iterations
    self.__randomness = randomness
    self.__discount_factor = discount_factor
    self.__model = None
    self.__memory = None
    self.__batch_size = batch_size
    self.__memory_size = memory_size
    
  def train(self):
    env = self.__environment
    model = self.__init_model(env)
  
    for e in range(self.__episodes):
      observation = env.get_initial_observation()

      print('Episode ' + str(e + 1))

      for i in range(self.__iterations):
        print('Episode ' + str(e + 1), 'Iter ' + str(i + 1))

        action = self.__pick_action(observation)
        new_observation, reward, done, _ = env.apply_action(action)

        self.__store_in_memory((
          observation,
          new_observation,
          action,
          reward,
        ))

        if len(self.__memory) > self.__batch_size:
          self.__batch_train()
        
        observation = new_observation

        if done:
          print('Done', reward)
          break

    return model

  """Gets the best action from the observation using the learned policy
  """
  def get_action(self, observation):
    q_values = self.__model.predict([observation])
    return np.argmax(q_values)

  """Save a serialized version of the model to a file
  """
  def save(self, file_path):
    self.__save_object(self.__model, file_path)

  """Loads the serialized version of the model from file
  """
  def load(self, file_path):
    self.__model = self.__load_object(file_path)

  def __save_object(self, obj, filepath):
    joblib.dump(obj, filepath)
  
  def __load_object(self, file_path):
    return joblib.load(file_path)

  def __init_model(self, env):
    actions = env.get_number_of_actions()
    model = MLPRegressor(
      random_state = 0,
      max_iter = 1,
      warm_start = True,
    )
    self.__model = model
    self.__memory = []

    model.fit(
      [env.get_initial_observation()],
      np.random.rand(1, actions),
    )

    return model

  def __store_in_memory(sample):
    self.__memory.append(sample)

    if len(self.__memory) > self.__memory_size:
      del self.__memory[0]

  def __batch_train(self):
    model = self.__model
    df = self.__discount_factor
    samples = random.sample(self.__memory, self.__batch_size)
    
    for sample in samples:
      observation, new_observation, action, reward = sample
      target = reward + df * np.max(model.predict([new_observation])[0])
      outputs = model.predict([observation])
      outputs[0][action] = target
      model.fit([observation], outputs)

  """Picks the next action based on a probability distribution created
    from the q-values or randomly.
    The selection would depend on the value of the randomness. (action selection policy: epsilon greedy)
  """
  def __pick_action(self, observation):
    env = self.__environment
    actions = env.get_number_of_actions()

    should_pick_random = np.random.uniform(0, 1) < self.__randomness

    if should_pick_random:
      return np.random.choice(actions)

    q_values = self.__model.predict([observation])

    return np.random.choice(actions, p = softmax(q_values[0]))
