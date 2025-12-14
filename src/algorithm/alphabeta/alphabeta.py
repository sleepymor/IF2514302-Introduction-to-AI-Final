import random

from algorithm.alphabeta.alphabetanode import AlphaBetaNode
from utils.logger import Logger


class AlphaBetaSearch:
    """
    Alpha-Beta Pruning search algorithm.

    This implementation:
    - Uses depth-limited minimax with alpha-beta pruning
    - Returns scores normalized to [0.0, 1.0] to match MCTS semantics
    - Tracks number of visited nodes for analysis/debugging
    """

    def __init__(self, max_depth: int = 3):
        """
        Initialize Alpha-Beta search.

        Args:
            max_depth (int): Maximum search depth for the minimax tree.
        """
        self.max_depth = max_depth
        self.log = Logger("AlphaBeta")
        self._nodes_visited = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, state):
        """
        Run Alpha-Beta search from the given game state.

        The root is treated as a MAX node (player's turn).

        Args:
            state: Current TacticalEnvironment state.

        Returns:
            tuple:
                - best_action: Action with the highest evaluated value
                - meta (dict):
                    * nodes_visited (int): Number of expanded nodes
                    * win_probability (float): Normalized score in [0.0, 1.0]
        """
        alpha = -float("inf")
        beta = float("inf")

        best_value = -float("inf")
        best_action = None
        self._nodes_visited = 0

        legal_actions = list(state.get_valid_actions(unit="current"))
        if not legal_actions:
            return None, {
                "nodes_visited": 0,
                "win_probability": 0.0,
            }

        # Root-level expansion (MAX player)
        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action)

            value = self._min_value(next_state, alpha, beta, depth=1)

            self.log.info(f"Action {action} -> raw score: {value:.3f}")

            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, best_value)

        win_probability = self._normalize_score(best_value)

        self.log.info(
            f"Best action: {best_action}, raw score: {best_value:.3f}, win_prob: {win_probability:.2f}"
        )

        meta = {
            "nodes_visited": self._nodes_visited,
            "win_probability": win_probability,
        }

        return best_action, meta

    # ------------------------------------------------------------------
    # Alpha-Beta Core (Minimax)
    # ------------------------------------------------------------------
    def _max_value(self, state, alpha, beta, depth):
        """
        Evaluate a MAX node in the minimax tree.

        Args:
            state: Current game state
            alpha (float): Best already-explored MAX value
            beta (float): Best already-explored MIN value
            depth (int): Current depth in the search tree

        Returns:
            float: Best achievable value from this state
        """
        node = AlphaBetaNode(state)
        self._nodes_visited += 1

        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        value = -float("inf")
        legal_actions = list(state.get_valid_actions(unit="current"))

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action)

            value = max(
                value,
                self._min_value(next_state, alpha, beta, depth + 1),
            )

            if value >= beta:
                # Beta cutoff
                return value

            alpha = max(alpha, value)

        return value

    def _min_value(self, state, alpha, beta, depth):
        """
        Evaluate a MIN node in the minimax tree.

        Args:
            state: Current game state
            alpha (float): Best already-explored MAX value
            beta (float): Best already-explored MIN value
            depth (int): Current depth in the search tree

        Returns:
            float: Worst-case value assuming optimal opponent play
        """
        node = AlphaBetaNode(state)
        self._nodes_visited += 1

        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        value = float("inf")
        legal_actions = list(state.get_valid_actions(unit="current"))

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action)

            value = min(
                value,
                self._max_value(next_state, alpha, beta, depth + 1),
            )

            if value <= alpha:
                # Alpha cutoff
                return value

            beta = min(beta, value)

        return value

    # ------------------------------------------------------------------
    # Scoring Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        """
        Normalize a raw Alpha-Beta evaluation score into [0.0, 1.0].

        This allows Alpha-Beta outputs to be directly comparable with
        MCTS win_probability values.

        Assumes:
            raw_score is approximately in the range [-1.0, 1.0]

        Args:
            raw_score (float): Heuristic value from evaluation function

        Returns:
            float: Normalized win probability
        """
        raw_score = max(-1.0, min(1.0, raw_score))
        return 0.5 * (raw_score + 1.0)
