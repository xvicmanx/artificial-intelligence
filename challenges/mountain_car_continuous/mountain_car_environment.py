import numpy as np
import gym
from common.helpers import make_discrete
from common.environment import Environment

class MountainCarEnvironment(Environment):
  intervals = 40
  number_of_actions = 3
  env_name = 'MountainCarContinuous-v0'
  
  def __init__(self):
    super().__init__()
    self.__environment = gym.make(MountainCarEnvironment.env_name).env
    self.__environment.seed(0)

  def get_initial_state(self):
    """Gets the initial state of an environment

    Returns:
        tuple: A tuple of of numbers representing the state
    """
    return self.get_state(self.get_initial_observation())

  def get_initial_observation(self):
    """Gets the initial observation of an environment

    Returns:
        object: Initial observation
    """
    return self.__environment.reset()

  def apply_action(self, action):
    """Applies an action to an environment

    Args:
        action (integer): Id of action to apply
    """
    space = self.__environment.action_space
    lv = space.low
    hv = space.high
    size = (hv[0] - lv[0]) / (self.number_of_actions - 1)
    value = action * size + lv[0]
    return self.__environment.step([value])

  def get_state(self, observation):
    """Maps an observation to state

    Args:
        observation (object): Observation from the environment

    Returns:
        tuple: A tuple of of numbers representing the state
    """
    return make_discrete(
      observation,
      self.__environment.observation_space,
      MountainCarEnvironment.intervals,
    )

  def get_states_dimension(self):
    """Gets the dimension of the state

    Returns:
        tuple: A tuple of integer that contains the number of values for dimension
    """
    return (
      MountainCarEnvironment.intervals,
      MountainCarEnvironment.intervals,
    )

  def get_number_of_actions(self):
    """Gets the number of actions that can be applied to the environment

    Returns:
        integer: Number of actions that can be applied
    """
    return MountainCarEnvironment.number_of_actions

  def render(self):
    """Renders the environment
    """
    self.__environment.render()
