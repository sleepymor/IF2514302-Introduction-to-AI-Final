import math
# from algorithm.astar.astar import AStar 
from algorithm.alpha_minmax.minmax import Minimax
from environment.environment import TacticalEnvironment
# class EnemyAgent():
#   def __init__(self, env: TacticalEnvironment):
#     self.env = env
#     self.move_range = 2

#   def action(self):
#     a_star = AStar(env=self.env)

#     start = tuple(self.env.enemy_pos)
#     goal = tuple(self.env.player_pos)

#     path = a_star.search(start, goal)

#     if path is None or len(path) <= 1:
#       return start
    
#     path = path[1:]

#     index = min(self.move_range - 1, len(path) - 1)

#     next_tile = path[index]
#     # enemy_range = self.env.get_move_range(self.env.enemy_pos)

#     # if next_step in enemy_range:
#     #   return next_step
    
#     # best_tile = None
#     # best_dist = float("inf")

#     # for tile in enemy_range:
#     #   euclidean_dist = math.dist(tile, next_step)
#     #   if euclidean_dist < best_dist:
#     #     best_dist = euclidean_dist
#     #     best_tile = tile

#     return next_tile

class EnemyAgent:
    # Enemy Agent menggunakan Minimax untuk mengejar player.
    
    def __init__(self, env: TacticalEnvironment):
        self.env = env
        self.algorithm = Minimax(max_depth=3, use_improvements=True)

    def action(self):
        
        best_move = self.algorithm.search(self.env)
        
        
        valid_moves = self.env.get_move_range(self.env.enemy_pos, move_range=2)
        if best_move not in valid_moves:
            print("⚠️  Minimax returned invalid move, using fallback")
            player_pos = tuple(self.env.player_pos)
            best_move = min(valid_moves, 
                          key=lambda pos: abs(pos[0]-player_pos[0]) + abs(pos[1]-player_pos[1]), 
                          default=tuple(self.env.enemy_pos))
        
        print(f"Minimax enemy chose: {best_move}")
        print(f"Enemy position: {tuple(self.env.enemy_pos)}")
        
        return best_move