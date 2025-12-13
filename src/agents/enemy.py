from algorithm.astar.astar import AStar
from environment.environment import TacticalEnvironment


class EnemyAgent:
    """Enemy AI agent using A* pathfinding algorithm."""

    def __init__(self, env: TacticalEnvironment):
        self.env = env
        self.move_range = 2
        # last computed path (list of (x,y) tuples) for visualization
        self.last_path = []

    def action(self):
        """Calculate and return the next enemy action using A* pathfinding."""
        a_star = AStar(env=self.env)

        start = tuple(self.env.enemy_pos)
        goal = tuple(self.env.player_pos)

        path = a_star.search(start, goal)

        # store for external visualization (main loop can call peek_path too)
        self.last_path = [] if path is None else list(path)
        try:
            # also attach to env for renderer convenience
            self.env.enemy_intent_path = self.last_path
        except Exception:
            pass

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

    def peek_path(self):
        """Return the computed A* path from enemy to player without consuming state.

        This method allows the main process (UI) to request the latest path
        for visualization even if the actual action is computed in a worker.
        """
        a_star = AStar(env=self.env)
        start = tuple(self.env.enemy_pos)
        goal = tuple(self.env.player_pos)
        path = a_star.search(start, goal)
        path = [] if path is None else list(path)
        # do not mutate env state here
        return path