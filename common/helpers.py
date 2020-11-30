import numpy as np


def make_discrete(observation, space, intervals):
  """Make space continuous values discrete

  Args:
      observation (array): Observation from environment
      space (object): Observation space
      intervals (int): dimension of the grid where the continues values will be divided.

  Returns:
      tuple: The discrete representation of the values
  """  
  lv = space.low
  hv = space.high
  iz = (hv - lv) / intervals
    
  items = []
  for i in range(len(hv)):
    items.append(int((observation[i] - lv[i]) / iz[i]))

  return tuple(items)

def softmax(vect):
  """Converts a vector to probability representation using Softmax function

  Args:
      vect (array): vector to convert

  Returns:
      array: vector converted to probability representation
  """  
  values = np.exp(vect)
  return values / np.sum(values)


def pick_action(values, actions, exploration_rate):
  """Picks the next action based on a probability distribution created
  from the values passed or randomly.
  The selection would depend on the value of the exploration rate. (action selection policy: epsilon greedy)

  Args:
      values (array): vector of values
      actions (integer): number of actions that can be taken
      exploration_rate (float): Exploration rate (randomness)

  Returns:
      integer: Selected action
  """  
  should_pick_random = np.random.uniform(0, 1) < exploration_rate

  if should_pick_random:
    return np.random.choice(actions)

  return np.random.choice(actions, p = softmax(values))


def show_result(
  environment,
  agent,
  iterations = 10000,
  show_iterations = False
):
  """Runs one episode of the agent in the environment

  Args:
      environment (Environment): Environment the agent interacts with
      agent (Agent): Agent that will interact with the environment
      iterations (int, optional): Maximum number of iterations to take. Defaults to 10000.
      show_iterations (bool, optional): To show the interation count. Defaults to False.
  """  
  observation = environment.get_initial_observation()

  for i in range(iterations):
    if show_iterations:
      print('Iteration', i + 1)

    environment.render()
    action = agent.get_action(observation)
    observation, _, done, _ = environment.apply_action(action)
    
    if done:
      break