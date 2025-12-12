import random

from environment.environment import TacticalEnvironment
from algorithm.minimax.minimaxnode import MinimaxNode
from utils.logger import Logger


class MinimaxSearch:
    """
    Minimax search algorithm with normalized scoring.

    Implements standard Minimax algorithm with depth-limited search.
    Evaluation scores are normalized to [0.0, 1.0] to match MCTS
    reward system for fair comparison.

    Attributes:
        max_depth (int): Maximum search depth
        log (Logger): Logger instance for debug output
    """

    def __init__(self, max_depth: int = 3):
        """
        Initialize Minimax search.

        Args:
            max_depth: Maximum search depth (default: 3)
                      Recommended values:
                      - depth=2: Fast but weak (~100ms)
                      - depth=3: Balanced (~500ms)
                      - depth=4: Strong but slow (~2-3s)
        """
        self.max_depth = max_depth
        self.log = Logger("Minimax")

    def search(self, state: TacticalEnvironment) -> tuple:
        """
        Execute Minimax search to find best action.

        Starting from current state, evaluates all possible moves
        up to max_depth and returns the action with highest minimax value.

        Args:
            state: Current game state

        Returns:
            tuple: Best action as (x, y) coordinates, or None if no legal moves
        """
        legal_actions = list(state.get_valid_actions(unit="current"))

        if not legal_actions:
            self.log.warning("No legal actions available!")
            return None

        self.log.info(f"Minimax searching (depth={self.max_depth})...")

        best_val = -float("inf")
        best_action = None

        # ========== EVALUATE ALL ROOT ACTIONS ==========
        # Iterate through all possible first moves
        for action in legal_actions:
            # Clone state and apply action
            next_state = state.clone()
            next_state.step(action, simulate=True)

            # Call min_value since enemy moves next
            val = self.min_value(next_state, depth=1)

            # Log evaluation for debugging
            self.log.debug(f"Action {action} → score: {val:.4f}")

            # Track best action
            if val > best_val:
                best_val = val
                best_action = action

        self.log.info(
            f"Best action: {best_action} with normalized score: {best_val:.4f}"
        )

        return best_action

    def max_value(self, state: TacticalEnvironment, depth: int) -> float:
        """
        Maximizing player's turn (player trying to maximize score).

        Recursively evaluates all possible player moves and returns
        the maximum achievable score at current depth.

        Args:
            state: Current game state
            depth: Current search depth

        Returns:
            float: Maximum normalized score achievable [0.0, 1.0]
        """
        # Create node for terminal check and evaluation
        node = MinimaxNode(state)

        # ========== BASE CASE: TERMINAL OR MAX DEPTH ==========
        if depth >= self.max_depth or node.is_terminal():
            return node.evaluate()

        # ========== RECURSIVE CASE: MAXIMIZE ==========
        v = -float("inf")
        legal_actions = node.get_legal_actions()

        # Try all possible player moves
        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action, simulate=True)

            # Recurse to enemy's turn (minimize)
            v = max(v, self.min_value(next_state, depth + 1))

        return v

    def min_value(self, state: TacticalEnvironment, depth: int) -> float:
        """
        Minimizing player's turn (enemy trying to minimize score).

        Recursively evaluates all possible enemy moves and returns
        the minimum achievable score at current depth.

        Args:
            state: Current game state
            depth: Current search depth

        Returns:
            float: Minimum normalized score achievable [0.0, 1.0]
        """
        # Create node for terminal check and evaluation
        node = MinimaxNode(state)

        # ========== BASE CASE: TERMINAL OR MAX DEPTH ==========
        if depth >= self.max_depth or node.is_terminal():
            return node.evaluate()

        # ========== RECURSIVE CASE: MINIMIZE ==========
        v = float("inf")
        legal_actions = node.get_legal_actions()

        # Try all possible enemy moves
        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action, simulate=True)

            # Recurse to player's turn (maximize)
            v = min(v, self.max_value(next_state, depth + 1))

        return v
