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
  policy,
  iterations = 10000,
):
  state = environment.get_initial_state()
  for i in range(iterations):
    environment.render()
    action = policy[state]
    observation, _, done, _ = environment.apply_action(action)
    state = environment.get_state(observation)
    
    if done:
      break