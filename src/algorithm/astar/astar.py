import math
class AStar: 
  def __init__(self, env):
    self.env = env

  def heuristic(self, pos, goal): 
    euclidean = math.dist(pos, goal)
    return euclidean

  def get_neighbors(self, node): 
    print(node)
    return


  def search(self, start, goal): return 
