import joblib
from helpers import show_result

class Agent:
  def __init__(self, environment):
    self._environment = environment

  """Trains the agent
  """
  def train(self):
    raise Exception('Method not implemented')

  """Gets the best action from the observation
  """
  def get_action(self, observation):
    raise Exception('Method not implemented')

  """Save a serialized version of the agent to files
  """
  def save(self):
    raise Exception('Method not implemented')

  """Loads the serialized version of the agent from files
  """
  def load(self):
    raise Exception('Method not implemented')

  """Runs one episode of the agent in its environment
  """
  def run_test_episode(
    self,
    iterations = 10000,
    show_iterations = True,
  ):
    show_result(
      self._environment,
      self,
      iterations,
      show_iterations,
    )

  def _save_object(self, obj, filepath):
    joblib.dump(obj, filepath)
  
  def _load_object(self, file_path):
    return joblib.load(file_path)
