import numpy as np
import gym
from common.helpers import make_discrete
from common.environment import Environment

class CartPoleEnvironment(Environment):
  intervals = 9
  env_name = 'CartPole-v1'
  
  def __init__(self):
    super().__init__()
    self.__environment = gym.make(CartPoleEnvironment.env_name).env
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
    return self.__environment.step(action)

  def get_state(self, observation):
    """Maps an observation to state

    Args:
        observation (object): Observation from the environment

    Returns:
        tuple: A tuple of of numbers representing the state
    """
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
    """Gets the dimension of the state

    Returns:
        tuple: A tuple of integer that contains the number of values for dimension
    """
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
    """Gets the number of actions that can be applied to the environment

    Returns:
        integer: Number of actions that can be applied
    """
    return self.__environment.action_space.n

  def render(self):
    """Renders the environment
    """
    self.__environment.render()
