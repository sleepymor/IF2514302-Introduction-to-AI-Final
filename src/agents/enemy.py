import numpy as np
class EnemyAgent():
  def __init__(self, env):
    self.env = env

  def action(self):
    x = np.random.randint(0, self.env.width)
    y = np.random.randint(0, self.env.height)

    return x, y