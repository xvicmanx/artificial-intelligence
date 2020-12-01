import numpy as np
from helpers import pick_action
from agent import Agent


# Q(s, a) = (1 - learning_rate) * Q(s, a) + learning_rate * Q_new(s, a)
# Q_new(s, a) = reward(s, a) + discount_factor * max_a' Q(s', a') 


class QLearner(Agent):
  min_learning_rate = 0.001

  def __init__(
    self,
    environment,
    agent_persist_file_path,
    reward_plot_file_path,
    learning_rate = 1.0,
    discount_factor = 1.0,
    episodes = 100,
    iterations = 10000,
    exploration_rate = 0.5,
    min_exploration_rate = 0.02,
    exploration_rate_decay = 0.998,
  ):
    """An agent that learns from the environment using the Deep Q Learning algorithm

    Args:
        environment (Environment): Environment to interacts with
        agent_persist_file_path (string): Path to persist the learned model
        reward_plot_file_path (string): Path to store the reward plot figure
        learning_rate (float, optional): Learning rate. Defaults to 1.0.
        discount_factor (float, optional): Discount factor. Defaults to 0.5.
        episodes (int, optional): Number of episodes to run. Defaults to 100.
        iterations (int, optional): Maximum number of iterations per episode. Defaults to 10000.
        exploration_rate (float, optional): Exploration rate. Defaults to 0.02.
    """    
    super().__init__(environment)

    states = environment.get_states_dimension()
    actions = environment.get_number_of_actions()

    self.__agent_persist_file_path = agent_persist_file_path
    self.__reward_plot_file_path = reward_plot_file_path
    self.__learning_rate = learning_rate
    self.__discount_factor = discount_factor
    self.__episodes = episodes
    self.__iterations = iterations
    self.__exploration_rate = exploration_rate
    self.__min_exploration_rate = min_exploration_rate
    self.__exploration_rate_decay = exploration_rate_decay
    self.__q_values = np.zeros(states + (actions, ))
    self.__policy = None
    self.__total_rewards = []
    

  def train(self):
    """Trains the agent using the q learning algorithm
    """
    q = self.__q_values
    env = self._environment
    df = self.__discount_factor
    initial_learning_rate = self.__learning_rate
    self.__total_rewards = []
    max_reward = None
    for e in range(self.__episodes):
      state = env.get_initial_state()
      total_reward = 0.0
      
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

        total_reward += reward

        if done:
          break

      self.__total_rewards.append(total_reward)

      if max_reward is None:
        max_reward = total_reward
      else:
        max_reward = max(max_reward, total_reward)
      
      self.__exploration_rate = max(
        self.__min_exploration_rate,
        self.__exploration_rate * self.__exploration_rate_decay
      )
      print(
        'Episode ' + str(e + 1),
        ', Max reward ' + str(max_reward),
        ', Total reward ' + str(total_reward),
        ', Exploration rate ' + str(self.__exploration_rate), 
        )


    self.__policy = self.__get_policy()

  def get_action(self, observation):
    """Gets the best action from the observation using the learned policy

    Args:
        observation (object): Observation from the environment

    Returns:
        integer: selected action
    """    
    state = self._environment.get_state(observation)
    return self.__policy[state]

  def save(self):
    """Save a serialized version of the model to a file
    """    
    self._save_object(self.__q_values, self.__agent_persist_file_path)
    self._save_reward_plot(self.__total_rewards, self.__reward_plot_file_path)

  def load(self):
    """Loads the serialized version of the model from file
    """
    self.__q_values = self._load_object(self.__agent_persist_file_path)
    self.__policy = self.__get_policy()

  def __pick_action(self, state):
    """Picks the next action based on a probability distribution created
    from the q-values or randomly.
    The selection would depend on the value of the exploration_rate. (action selection policy: epsilon greedy)

    Args:
        observation (object): Observation from the environment

    Returns:
        integer: Picked action
    """
    return pick_action(
      self.__q_values[state],
      self._environment.get_number_of_actions(),
      self.__exploration_rate,
    )

  def __get_policy(self):
    """Gets the policy from the Q values table

    Returns:
        array: The policy learned
    """    
    return np.argmax(
      self.__q_values,
      axis = len(self._environment.get_states_dimension()),
    )

