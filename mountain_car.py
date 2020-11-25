import numpy as np
import gym

class MountainCar:
  intervals = 40
  env_name = 'MountainCar-v0'
  
  def __init__(self):
    self.__environment = gym.make(MountainCar.env_name).env
    self.__environment.seed(0)

  def get_initial_state(self):
    return self.get_state(self.__environment.reset())

  def apply_action(self, action):
    return self.__environment.step(action)

  def get_state(self, observation):
    space = self.__environment.observation_space
    lowest_values = space.low
    highest_values = space.high
    interval_sizes = (highest_values - lowest_values) / MountainCar.intervals
    
    return (
      int((observation[0] - lowest_values[0]) / interval_sizes[0]),
      int((observation[1] - lowest_values[1]) / interval_sizes[1]),
    )

  def get_states_dimension(self):
    return (MountainCar.intervals, MountainCar.intervals)

  def get_number_of_actions(self):
    return self.__environment.action_space.n

  def render(self):
    self.__environment.render()
