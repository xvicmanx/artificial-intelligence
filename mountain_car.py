import numpy as np
import gym
from helpers import make_discrete

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
    return make_discrete(
      observation,
      self.__environment.observation_space,
      MountainCar.intervals,
    )

  def get_states_dimension(self):
    return (MountainCar.intervals, MountainCar.intervals)

  def get_number_of_actions(self):
    return self.__environment.action_space.n

  def render(self):
    self.__environment.render()
