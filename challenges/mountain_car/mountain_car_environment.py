import numpy as np
import gym
from common.helpers import make_discrete

class MountainCarEnvironment:
  intervals = 40
  env_name = 'MountainCar-v0'
  
  def __init__(self):
    self.__environment = gym.make(MountainCarEnvironment.env_name).env
    self.__environment.seed(0)

  def get_initial_state(self):
    return self.get_state(self.get_initial_observation())

  def get_initial_observation(self):
    return self.__environment.reset()

  def apply_action(self, action):
    return self.__environment.step(action)

  def get_state(self, observation):
    return make_discrete(
      observation,
      self.__environment.observation_space,
      MountainCarEnvironment.intervals,
    )

  def get_states_dimension(self):
    return (
      MountainCarEnvironment.intervals,
      MountainCarEnvironment.intervals,
    )

  def get_number_of_actions(self):
    return self.__environment.action_space.n

  def render(self):
    self.__environment.render()
