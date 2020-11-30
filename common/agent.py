import joblib
from helpers import show_result

class Agent:  
  def __init__(self, environment):
    """Agent is an abstraction of an agent that interacts with an environment,
      learn from it and improve its performance using reinforcement learning.

    Args:
        environment (Environment): the environment the agent will interact with
    """    
    self._environment = environment

  def train(self):
    """Trains the agent
    """    
    raise Exception('Method not implemented')

  def get_action(self, observation):
    """Gets the best action from the observation

    Args:
        observation (object): Environment observation

    Returns:
        integer: action
    """    
    raise Exception('Method not implemented')

  def save(self):
    """Save a serialized version of the agent to files
    """    
    raise Exception('Method not implemented')

  def load(self):
    """Loads the serialized version of the agent from files

    Returns:
        object: unserialized object
    """    
    raise Exception('Method not implemented')

  def run_test_episode(
    self,
    iterations = 10000,
    show_iterations = True,
  ):
    """Runs one episode of the agent in its environment

    Args:
        iterations (int, optional): The number of maximum iterations to take. Defaults to 10000.
        show_iterations (bool, optional): To display the iterations count. Defaults to True.
    """    
    show_result(
      self._environment,
      self,
      iterations,
      show_iterations,
    )

  def _save_object(self, obj, filepath):
    """Serializes an object and stores it to a file

    Args:
        obj (string): Object to serialize
        filepath (string): path to store the serialized object
    """    
    joblib.dump(obj, filepath)
  
  def _load_object(self, file_path):
    """Loads a serialized object from a file

    Args:
        file_path (string): path of the file the object is stored

    Returns:
        object: the unserialized object
    """    
    return joblib.load(file_path)
