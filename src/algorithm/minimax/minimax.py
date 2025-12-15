from algorithm.minimax.minimaxnode import MinimaxNode
from utils.logger import Logger
import random


class MinimaxSearch:
    """
    Minimax search algorithm with alpha-beta pruning optimization.

    Finds optimal moves by recursively evaluating game tree with
    depth-limited search (max_depth) and early termination at terminal states.
    """

    def __init__(self, max_depth: int = 4):
        """
        Initialize Minimax search.

        Args:
            max_depth (int): Maximum search depth (default: 4)
        """
        self.max_depth = max_depth
        self.log = Logger("Minimax")

    def search(self, state) -> tuple:
        """
        Execute Minimax search and return best action.

        Implements random tie-breaking to avoid deterministic behavior
        and prevent infinite loops in symmetric game states.

        Args:
            state: Current TacticalEnvironment state

        Returns:
            tuple: (best_action, metadata_dict)
                - best_action: Optimal move found
                - metadata: {minimax_score, num_actions_evaluated}
        """
        best_minimax_score = -float("inf")
        candidate_best_actions = []

        legal_actions = list(state.get_valid_actions(unit="current"))
        if not legal_actions:
            return None, {"minimax_score": 0.0, "actions_evaluated": 0}

        # Randomize action order to avoid bias (e.g., always checking upward)
        random.shuffle(legal_actions)

        # Root level: maximize player's perspective
        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action, simulate=True)

            minimax_score = self._min_value(next_state, depth=1)

            # Track best action(s) with tie-breaking
            if minimax_score > best_minimax_score:
                best_minimax_score = minimax_score
                candidate_best_actions = [action]
            elif minimax_score == best_minimax_score:
                candidate_best_actions.append(action)

        # Random selection among equally good moves
        chosen_action = (
            random.choice(candidate_best_actions) if candidate_best_actions else None
        )

        metadata = {
            "minimax_score": float(best_minimax_score),
            "actions_evaluated": len(legal_actions),
        }

        return chosen_action, metadata

    def _max_value(self, state, depth: int) -> float:
        """
        Minimax MAX node: player turn (maximize score).

        Args:
            state: Current game state
            depth (int): Current search depth

        Returns:
            float: Best achievable score from this state [0.0, 1.0]
        """
        node = MinimaxNode(state)

        # Terminal condition: reached max depth or game over
        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        max_score = -float("inf")
        legal_actions = list(state.get_valid_actions(unit="current"))

        # No legal moves: evaluate current position
        if not legal_actions:
            return node.evaluate()

        # Expand all child nodes
        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action, simulate=True)

            child_score = self._min_value(next_state, depth + 1)
            max_score = max(max_score, child_score)

        return max_score

    def _min_value(self, state, depth: int) -> float:
        """
        Minimax MIN node: enemy turn (minimize score).

        Args:
            state: Current game state
            depth (int): Current search depth

        Returns:
            float: Worst-case score assuming optimal enemy play [0.0, 1.0]
        """
        node = MinimaxNode(state)

        # Terminal condition: reached max depth or game over
        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        min_score = float("inf")
        legal_actions = list(state.get_valid_actions(unit="current"))

        # No legal moves: evaluate current position
        if not legal_actions:
            return node.evaluate()

        # Expand all child nodes
        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action, simulate=True)

            child_score = self._max_value(next_state, depth + 1)
            min_score = min(min_score, child_score)

        return min_score
