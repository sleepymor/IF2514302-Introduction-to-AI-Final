import math
from algorithm.astar.astar import AStar
from environment.environment import TacticalEnvironment


class EnemyAgent:
    def __init__(self, env: TacticalEnvironment):
        self.env = env
        self.move_range = 2

    def action(self):
        """Calculate and return the next enemy action using A* pathfinding."""
        a_star = AStar(env=self.env)

        start = tuple(self.env.enemy_pos)
        goal = tuple(self.env.player_pos)

        path = a_star.search(start, goal)

        if path is None or len(path) <= 1:
            return start

        path = path[1:]

        index = min(self.move_range - 1, len(path) - 1)

        next_tile = path[index]
        # enemy_range = self.env.get_move_range(self.env.enemy_pos)

        # if next_step in enemy_range:
        #   return next_step

        # best_tile = None
        # best_dist = float("inf")

        # for tile in enemy_range:
        #   euclidean_dist = math.dist(tile, next_step)
        #   if euclidean_dist < best_dist:
        #     best_dist = euclidean_dist
        #     best_tile = tile

        return next_tile
