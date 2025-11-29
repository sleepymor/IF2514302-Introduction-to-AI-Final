import math
import random
from algorithm.astar.astar import AStar
from environment.environment import TacticalEnvironment


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    Represents a game state and stores search statistics.
    """

    def __init__(self, state: TacticalEnvironment, parent=None, action=None):
        """
        Initialize a new MCTS node.

        Args:
            state: Current game state (TacticalEnvironment)
            parent: Parent node in search tree
            action: Action taken to reach this state from parent
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0  # Total reward accumulated

        # Get legal actions for CURRENT turn (player or enemy)
        action_list = list(self._get_legal_actions())
        random.shuffle(action_list)
        self.untried_actions = action_list

    def _get_legal_actions(self):
        """
        Get all legal actions for the current player.

        Returns:
            set: Legal move positions for current turn
        """
        # Check whose turn it is
        if self.state.turn == "player":
            return self.state.get_move_range(self.state.player_pos)
        elif self.state.turn == "enemy":
            return self.state.get_move_range(self.state.enemy_pos, move_range=3)
        else:
            return set()

    def add_child(self, child_node):
        """Add a child node and remove its action from untried actions"""
        self.children.append(child_node)
        if child_node.action in self.untried_actions:
            self.untried_actions.remove(child_node.action)

    def ucb_score(self, c=1.4):
        """
        UCB1 formula for balancing exploration and exploitation.

        Args:
            c: Exploration constant

        Returns:
            float: UCB score for this node
        """
        if self.visits == 0:
            return float("inf")  # Unvisited nodes have highest priority

        # Exploitation: average reward
        exploitation = self.wins / self.visits

        # Exploration: bonus for less-visited nodes
        if self.parent and self.parent.visits > 0:
            exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        else:
            exploration = 0

        return exploitation + exploration

    def best_child(self, c=1.4):
        """
        Select child with highest UCB score.

        Args:
            c: Exploration constant

        Returns:
            MCTSNode: Child with best UCB score
        """
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.ucb_score(c))

    def is_terminal(self):
        """
        Check if state is terminal (win/lose).

        Returns:
            bool: True if terminal state
        """
        is_term, _ = self.state.is_terminal()
        return is_term

    def is_fully_expanded(self):
        """
        Check if all possible actions have been tried.

        Returns:
            bool: True if no untried actions remain
        """
        return len(self.untried_actions) == 0

    @property
    def depth(self):
        """
        Calculate depth of node in tree.

        Returns:
            int: Depth from root
        """
        depth = 0
        node = self
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def __repr__(self):
        """String representation for debugging"""
        return (
            f"MCTSNode(action={self.action}, visits={self.visits}, "
            f"wins={self.wins:.2f}, turn={self.state.turn})"
        )
