import heapq
from environment.environment import TacticalEnvironment
from algorithm.astar.node import Node


class AStar:
    def __init__(self, env: TacticalEnvironment):
        """
        Initializes the A* pathfinder with a given environment.

        Args:
            env (TacticalEnvironment): The grid environment used for pathfinding.
                                       Contains methods like in_bounds() and is_blocked().
        """
        self.env = env

    def heuristic(self, pos, goal):
        """
        Computes the Manhattan distance heuristic between two points.

        Manhattan distance is used because movement is limited to 4-directional grid steps.

        Args:
            pos (tuple[int, int]): The (x, y) coordinate of the current node.
            goal (tuple[int, int]): The (x, y) coordinate of the goal node.

        Returns:
            int: Manhattan distance |x1 - x2| + |y1 - y2|.
        """
        manhattan_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        return manhattan_dist

    def get_neighbors(self, node):
        """
        Returns all valid neighboring nodes for a given node.

        Movement allowed:
            - (1, 0)  → right
            - (-1, 0) → left
            - (0, 1)  → down
            - (0, -1) → up

        A neighbor is valid only if:
            - It is inside grid bounds.
            - It is not blocked by a wall.

        Args:
            node (Node): The current node being expanded.

        Returns:
            list[Node]: A list of accessible neighboring Node objects.
        """
        (x, y) = node.position

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if self.env.in_bounds(nx, ny) and not self.env.is_blocked(nx, ny):
                neighbors_node = Node(parent=node, position=(nx, ny))
                neighbors.append(neighbors_node)

        return neighbors

    def search(self, start, goal):
        """
        Performs A* search from start to goal in the tactical environment.

        Standard A* implementation with:
            - Open set (priority queue)
            - Closed set (visited nodes)
            - g = cost from start
            - h = heuristic estimate to goal
            - f = g + h

        The algorithm stops when:
            - The goal is reached.
            - The open set is empty (no path exists).

        Args:
            start (tuple[int, int]): Starting coordinate (x, y).
            goal (tuple[int, int]): Goal coordinate (x, y).

        Returns:
            list[tuple[int, int]] | None:
                Full path from start to goal, including both endpoints.
                Returns None if no valid path is found.
        """
        start_node = Node(None, start)
        goal_node = Node(None, goal)

        open_heap = []
        self.counter = 0

        heapq.heappush(open_heap, (0, self.counter, start_node))
        self.counter += 1

        open_set = {start: start_node}
        closed_set = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            closed_set.add(current.position)

            if current == goal_node:
                path = []
                while current is not None:
                    path.append(current.position)  # kamu salah tulis posistion
                    current = current.parent
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                if neighbor.position in closed_set:
                    continue

                tentative_g = current.g + 1

                if (
                    neighbor.position not in open_set
                    or tentative_g < open_set[neighbor.position].g
                ):

                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor.position, goal)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current

                    open_set[neighbor.position] = neighbor

                    heapq.heappush(open_heap, (neighbor.f, self.counter, neighbor))
                    self.counter += 1

        return None
