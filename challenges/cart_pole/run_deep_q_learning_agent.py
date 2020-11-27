from cart_pole_deep_q_learning_agent import CartPoleDeepQLearningAgent

agent = CartPoleDeepQLearningAgent()

print('Loading agent learned data')
agent.load()

print('Running a test episode')
agent.run_test_episode()
