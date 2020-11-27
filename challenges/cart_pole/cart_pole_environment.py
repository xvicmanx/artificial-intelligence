import numpy as np
import gym
from common.helpers import make_discrete

class CartPoleEnvironment:
  intervals = 9
  env_name = 'CartPole-v1'
  
  def __init__(self):
    self.__environment = gym.make(CartPoleEnvironment.env_name).env
    self.__environment.seed(0)

  def get_initial_state(self):
    return self.get_state(self.get_initial_observation())

  def get_initial_observation(self):
    return self.__environment.reset()

  def apply_action(self, action):
    return self.__environment.step(action)

  def get_state(self, observation):
    values = make_discrete(
      observation,
      self.__environment.observation_space,
      CartPoleEnvironment.intervals,
    )

    velocity_state = 0
    if observation[1] > 0:
      velocity_state = 1
    elif observation[1] < 0:
      velocity_state = 2

    angular_velocity_state = 0
    if observation[3] > 0:
      angular_velocity_state = 1
    elif observation[3] < 0:
      angular_velocity_state = 2

    return (
      values[0],
      velocity_state,
      values[2],
      angular_velocity_state
    )

  def get_states_dimension(self):
    return (
      # Cart position
      CartPoleEnvironment.intervals,
      # Cart Velocity
      3,
      # Pole Angle
      CartPoleEnvironment.intervals,
      # Pole Angular Velocity
      3,
    )

  def get_number_of_actions(self):
    return self.__environment.action_space.n

  def render(self):
    self.__environment.render()
