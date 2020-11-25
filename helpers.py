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