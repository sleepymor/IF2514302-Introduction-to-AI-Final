"""Enemy AI agent implementation.

Provides enemy agent using A* pathfinding to hunt the player.
"""

from algorithm.astar.astar import AStar
from environment.environment import TacticalEnvironment


class EnemyAgent:
    """Enemy AI agent using A* pathfinding algorithm.

    The enemy agent uses A* pathfinding to find the shortest path to the player
    and moves towards it with a configurable move range per turn.

    Attributes:
        env: TacticalEnvironment instance
        move_range: Number of tiles the enemy can move per turn
        last_path: Last computed path for visualization purposes
    """

    def __init__(self, env: TacticalEnvironment):
        """Initialize enemy agent.

        Args:
            env: TacticalEnvironment instance containing game state
        """
        self.env = env
        self.move_range = 2
        self.last_path = []

    def action(self) -> tuple:
        """Calculate and return the next enemy action using A* pathfinding.

        Computes the shortest path to the player using A* algorithm and moves
        towards the player within the configured move_range.

        Returns:
            Tuple of (x, y) coordinates for the next tile to move to
        """
        a_star = AStar(env=self.env)

        start = tuple(self.env.enemy_pos)
        goal = tuple(self.env.player_pos)

        # Find path to player
        path = a_star.search(start, goal)

        # Store path for visualization
        self.last_path = [] if path is None else list(path)
        try:
            self.env.enemy_intent_path = self.last_path
        except Exception:
            pass

        # No path found or at goal
        if path is None or len(path) <= 1:
            return start

        # Remove starting position from path
        path = path[1:]

        # Move up to move_range tiles towards player
        index = min(self.move_range - 1, len(path) - 1)
        next_tile = path[index]

        return next_tile

    def peek_path(self) -> list:
        """Return the computed A* path from enemy to player.

        This method allows visualization of the enemy's intended path without
        consuming or modifying any game state. Can be called by the UI to
        display enemy pathfinding for debugging.

        Returns:
            List of (x, y) tuples representing the path from enemy to player,
            or empty list if no path exists
        """
        a_star = AStar(env=self.env)
        start = tuple(self.env.enemy_pos)
        goal = tuple(self.env.player_pos)
        path = a_star.search(start, goal)
        return [] if path is None else list(path)
