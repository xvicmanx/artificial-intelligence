class Environment:
  """An abstraction of the environment that an agent interacts with.
  """  

  def get_initial_state(self):
    """Gets the initial state of an environment

    Returns:
        tuple: A tuple of of numbers representing the state
    """
    raise Exception('Method not implemented')

  def get_initial_observation(self):
    """Gets the initial observation of an environment

    Returns:
        object: Initial observation
    """
    raise Exception('Method not implemented')

  def apply_action(self, action):
    """Applies an action to an environment

    Args:
        action (integer): Id of action to apply
    """
    raise Exception('Method not implemented')

  def get_state(self, observation):
    """Maps an observation to state

    Args:
        observation (object): Observation from the environment

    Returns:
        tuple: A tuple of of numbers representing the state
    """
    raise Exception('Method not implemented')

  def get_states_dimension(self):
    """Gets the dimension of the state

    Returns:
        tuple: A tuple of integer that contains the number of values for dimension
    """    
    raise Exception('Method not implemented')

  def get_number_of_actions(self):
    """Gets the number of actions that can be applied to the environment

    Returns:
        integer: Number of actions that can be applied
    """
    raise Exception('Method not implemented')  

  def render(self):
    """Renders the environment
    """
    raise Exception('Method not implemented')
