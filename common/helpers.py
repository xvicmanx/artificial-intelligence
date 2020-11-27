import numpy as np

"""Make space continuous values discrete
"""
def make_discrete(observation, space, intervals):
  lv = space.low
  hv = space.high
  iz = (hv - lv) / intervals
    
  items = []
  for i in range(len(hv)):
    items.append(int((observation[i] - lv[i]) / iz[i]))

  return tuple(items)

def softmax(vect):
  values = np.exp(vect)
  return values / np.sum(values)

"""Picks the next action based on a probability distribution created
  from the values passed or randomly.
  The selection would depend on the value of the exploration rate. (action selection policy: epsilon greedy)
"""
def pick_action(values, actions, exploration_rate):
  should_pick_random = np.random.uniform(0, 1) < exploration_rate

  if should_pick_random:
    return np.random.choice(actions)

  return np.random.choice(actions, p = softmax(values))

"""Runs one episode of the agent in its environment
"""
def show_result(
  environment,
  agent,
  iterations = 10000,
  show_iterations = False
):
  observation = environment.get_initial_observation()

  for i in range(iterations):
    if show_iterations:
      print('Iteration', i + 1)

    environment.render()
    action = agent.get_action(observation)
    observation, _, done, _ = environment.apply_action(action)
    
    if done:
      break