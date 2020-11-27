import numpy as np

def softmax(vect):
  values = np.exp(vect)
  return values / np.sum(values)

def make_discrete(observation, space, intervals):
  lv = space.low
  hv = space.high
  iz = (hv - lv) / intervals
    
  items = []
  for i in range(len(hv)):
    items.append(int((observation[i] - lv[i]) / iz[i]))

  return tuple(items)

def show_result(
  environment,
  model,
  iterations = 10000,
  show_iterations = False
):
  observation = environment.get_initial_observation()

  for i in range(iterations):
    if show_iterations:
      print('Iteration', i + 1)

    environment.render()
    action = model.get_action(observation)
    observation, _, done, _ = environment.apply_action(action)
    
    if done:
      break