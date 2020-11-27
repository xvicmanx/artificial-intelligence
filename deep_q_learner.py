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
    episodes = 700,
    iterations = 200,
    exploration_rate = 1.0,
    min_exploration_rate = 0.02,
    exploration_rate_decay = 0.99,
    discount_factor = 0.5,
    train_interval = 50,
    batch_size = 200,
    memory_size = 20000,
  ):
    self.__environment = environment
    self.__episodes = episodes
    self.__iterations = iterations
    self.__exploration_rate = exploration_rate
    self.__min_exploration_rate = min_exploration_rate
    self.__exploration_rate_decay = exploration_rate_decay
    self.__discount_factor = discount_factor
    self.__model = None
    self.__memory = None
    self.__batch_size = batch_size
    self.__memory_size = memory_size
    self.__train_interval = train_interval
    
  def train(self):
    env = self.__environment
    model = self.__init_model(env)
  
    count = 0
    for e in range(self.__episodes):
      observation = env.get_initial_observation()
      total_reward = 0.0

      print('Episode ' + str(e + 1))

      for i in range(self.__iterations):
        print('Episode ' + str(e + 1), 'Iter ' + str(i + 1))

        action = self.__pick_action(observation)
        new_observation, reward, done, _ = env.apply_action(action)
        total_reward += reward
  
        self.__store_in_memory((
          observation,
          new_observation,
          action,
          reward,
          done,
        ))

        if len(self.__memory) > self.__batch_size and count % self.__train_interval == 0:
          self.__batch_train()
        
        observation = new_observation

        count += 1

        if done:
          print('Done', reward)
          break

      print('Episode ' + str(e + 1), 'Total reward', total_reward)
      self.__exploration_rate = max(
        self.__min_exploration_rate,
        self.__exploration_rate * self.__exploration_rate_decay
      )

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
      hidden_layer_sizes = (20, 20),
    )
    self.__model = model
    self.__memory = []

    model.fit(
      [env.get_initial_observation()],
      np.random.rand(1, actions),
    )

    return model

  def __store_in_memory(self, sample):
    self.__memory.append(sample)

    if len(self.__memory) > self.__memory_size:
      del self.__memory[0]

  def __batch_train(self):
    model = self.__model
    df = self.__discount_factor
    samples = random.sample(self.__memory, self.__batch_size)
    
    observations, new_observations, actions, rewards, dones = list(zip(*samples))
    targets = rewards + df * np.max(model.predict(new_observations), axis = 1)
    outputs = model.predict(observations)

    for i in range(len(actions)):
      outputs[i][actions[i]] = rewards[i] if dones[i] else targets[i]

    model.partial_fit(observations, outputs)

  """Picks the next action based on a probability distribution created
    from the q-values or randomly.
    The selection would depend on the value of the exploration_rate. (action selection policy: epsilon greedy)
  """
  def __pick_action(self, observation):
    env = self.__environment
    actions = env.get_number_of_actions()

    should_pick_random = np.random.uniform(0, 1) < self.__exploration_rate

    if should_pick_random:
      return np.random.choice(actions)

    q_values = self.__model.predict([observation])

    return np.random.choice(actions, p = softmax(q_values[0]))
