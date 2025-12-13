import math
import random
from environment.environment import TacticalEnvironment


class MCTSNode:
    """
    Node in a Monte Carlo Tree Search (MCTS) tree.

    Responsibilities:
    - Store game state
    - Maintain parent / children relationships
    - Track visit statistics
    - Provide UCB-based selection utilities
    """

    def __init__(
        self,
        state: TacticalEnvironment,
        parent: "MCTSNode | None" = None,
        action=None,
    ):
        self.state = state
        self.parent = parent
        self.action = action

        # Tree structure
        self.children: list[MCTSNode] = []

        # MCTS statistics
        self.visits: int = 0
        self.total_wins: float = 0.0

        # Expansion bookkeeping
        self.untried_actions = self._init_untried_actions()

    # ------------------------------------------------------------------
    # Initialization & Tree Management
    # ------------------------------------------------------------------
    def _init_untried_actions(self):
        """Initialize and shuffle all legal actions from this state."""
        actions = list(self.state.get_valid_actions(unit="current"))
        random.shuffle(actions)
        return actions

    def add_child(self, child: "MCTSNode"):
        """Attach a child node and mark its action as tried."""
        self.children.append(child)
        if child.action in self.untried_actions:
            self.untried_actions.remove(child.action)

    # ------------------------------------------------------------------
    # MCTS Properties
    # ------------------------------------------------------------------
    def is_terminal(self) -> bool:
        """Return True if this node represents a terminal game state."""
        terminal, _ = self.state.is_terminal()
        return terminal

    def is_fully_expanded(self) -> bool:
        """Return True if all legal actions have been expanded."""
        return not self.untried_actions

    @property
    def depth(self) -> int:
        """Depth of this node from the root."""
        depth = 0
        current = self
        while current:
            depth += 1
            current = current.parent
        return depth

    # ------------------------------------------------------------------
    # UCB / Selection Logic
    # ------------------------------------------------------------------
    def ucb_score(self, exploration_constant: float = 1.4) -> float:
        """
        Compute Upper Confidence Bound (UCB1) score.

        UCB = exploitation + exploration
        """
        if self.visits == 0:
            return float("inf")

        exploitation = self.total_wins / self.visits

        if self.parent is None:
            return exploitation

        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.4) -> "MCTSNode":
        """Select child with highest UCB score."""
        return max(
            self.children,
            key=lambda child: child.ucb_score(exploration_constant),
        )

    # ------------------------------------------------------------------
    # Reward Evaluation (Heuristic)
    # ------------------------------------------------------------------
    def evaluate(self) -> float:
        """
        Evaluate current state from the player's perspective.

        Terminal:
        - +1.0 : goal reached
        -  0.0 : caught / trap

        Non-terminal:
        - heuristic based on distance to goal and enemy
        """
        player_pos = tuple(self.state.player_pos)
        enemy_pos = tuple(self.state.enemy_pos)
        goal_pos = self.state.goal

        # Terminal outcomes
        if player_pos == goal_pos:
            return 1.0
        if player_pos == enemy_pos or player_pos in self.state.traps:
            return 0.0

        # Heuristic shaping
        dist_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        dist_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )

        goal_score = 1.0 / (dist_goal + 1)
        danger_penalty = 1.0 / (dist_enemy + 1)

        return max(0.0, goal_score - 0.5 * danger_penalty)
