from cart_pole_q_learning_agent import CartPoleQLearningAgent

agent = CartPoleQLearningAgent()

print('Loading agent learned data')
agent.load()

print('Running a test episode')
agent.run_test_episode()
