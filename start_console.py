import code
from common.helpers import print_gym_environment_details

class PlaygroundConsole(code.InteractiveConsole):
  def start(self):
    try:
      self.interact('Starting playground console')
    except:
      print('Closing playground console')

console = PlaygroundConsole(locals())
console.start()
