from mountain_car_q_learning_agent import MountainCarQLearningAgent

agent = MountainCarQLearningAgent()

print('Loading agent learned data')
agent.load()

print('Running a test episode')
agent.run_test_episode(iterations = 200)
