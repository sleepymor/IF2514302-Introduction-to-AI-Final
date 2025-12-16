"""A* pathfinding algorithm for grid-based environments.

Uses Manhattan distance heuristic and standard A* search with f = g + h scoring.
Finds shortest path from start to goal while avoiding walls.
"""

import heapq

from algorithm.astar.node import Node
from environment.environment import TacticalEnvironment


class AStar:
    """A* pathfinder for tactical grid environments."""

    def __init__(self, env: TacticalEnvironment):
        """
        Initialize A* pathfinder with a tactical environment.

        Args:
            env: The TacticalEnvironment containing grid, walls, and bounds.
        """
        self.env = env
        self.counter = 0

    def heuristic(self, pos, goal):
        """
        Calculate Manhattan distance heuristic.

        Manhattan distance is suitable for 4-directional (up/down/left/right)
        movement on a grid.

        Args:
            pos: Current position as (x, y) tuple
            goal: Goal position as (x, y) tuple

        Returns:
            int: Manhattan distance |x1 - x2| + |y1 - y2|
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def get_neighbors(self, node):
        """
        Get all valid neighboring tiles from a node.

        Neighbors must satisfy:
        - Within grid bounds
        - Not blocked by walls
        - 4-directional movement (up, down, left, right)

        Args:
            node: Current Node being expanded

        Returns:
            list: List of accessible neighboring Node objects
        """
        x, y = node.position
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        neighbors = []

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds and walls
            if self.env.in_bounds(nx, ny) and not self.env.is_blocked(nx, ny):
                neighbor_node = Node(parent=node, position=(nx, ny))
                neighbors.append(neighbor_node)

        return neighbors

    def search(self, start, goal):
        """
        Perform A* search from start to goal.

        Uses standard A* with:
        - Open set: Priority queue of nodes to explore (by f-score)
        - Closed set: Already explored nodes
        - g: Cost from start to current node
        - h: Heuristic estimate from current to goal
        - f: g + h (total estimated cost)

        Args:
            start: Starting position as (x, y) tuple
            goal: Goal position as (x, y) tuple

        Returns:
            list: Path from start to goal (including both endpoints)
                  Returns None if no path exists
        """
        # Initialize start node
        start_node = Node(parent=None, position=start)
        goal_node = Node(parent=None, position=goal)

        # Open set: priority queue ordered by f-score
        open_heap = []
        self.counter = 0

        heapq.heappush(open_heap, (0, self.counter, start_node))
        self.counter += 1

        # Track nodes in open set by position for quick lookup
        open_set = {start: start_node}
        closed_set = set()

        # Main A* loop
        while open_heap:
            # Get node with lowest f-score
            _, _, current = heapq.heappop(open_heap)
            closed_set.add(current.position)

            # Goal reached - reconstruct path
            if current == goal_node:
                path = []
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1]

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                # Skip if already explored
                if neighbor.position in closed_set:
                    continue

                # Calculate tentative g-score (cost from start)
                tentative_g = current.g + 1

                # Update if we found a better path to this neighbor
                if (
                    neighbor.position not in open_set
                    or tentative_g < open_set[neighbor.position].g
                ):
                    # Update neighbor's scores
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor.position, goal)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current

                    # Add/update in open set
                    open_set[neighbor.position] = neighbor
                    heapq.heappush(open_heap, (neighbor.f, self.counter, neighbor))
                    self.counter += 1

        # No path found
        return None
