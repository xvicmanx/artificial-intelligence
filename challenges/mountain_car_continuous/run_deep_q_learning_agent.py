from mountain_car_deep_q_learning_agent import MountainCarDeepQLearningAgent

agent = MountainCarDeepQLearningAgent()

print('Loading agent learned data')
agent.load()

print('Running a test episode')
agent.run_test_episode()
