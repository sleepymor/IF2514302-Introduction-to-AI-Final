from environment.environment import TacticalEnvironment
import math
import random


class AlphaBetaNode:
    """Node abstraction for Alpha-Beta search with heuristic evaluation."""

    def __init__(
        self,
        state: TacticalEnvironment,
        parent: "AlphaBetaNode | None" = None,
        action=None,
    ):
        """Initialize an Alpha-Beta search node."""
        self.state = state
        self.parent = parent
        self.action = action

    def evaluate(self) -> float:
        """
        Evaluate the current state from the player's perspective.

        Returns:
            float: Evaluation score where:
                - +inf: Guaranteed win (goal reached)
                - -inf: Guaranteed loss (caught or trapped)
                - Otherwise: Unbounded heuristic value
        """
        player_pos = tuple(self.state.player_pos)
        enemy_pos = tuple(self.state.enemy_pos)
        goal_pos = self.state.goal

        # Terminal states
        if player_pos == goal_pos:
            return float("inf")

        if player_pos == enemy_pos or player_pos in self.state.traps:
            return float("-inf")

        # Non-terminal heuristic evaluation
        cumulative_score = 0.0

        # Goal proximity penalty
        manhattan_dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(
            player_pos[1] - goal_pos[1]
        )
        cumulative_score -= manhattan_dist_to_goal * 100

        # Enemy proximity penalty (danger zone)
        manhattan_dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )
        if manhattan_dist_to_enemy <= 3 and manhattan_dist_to_enemy > 0:
            cumulative_score -= 5000.0 / (manhattan_dist_to_enemy**2)

        # Mobility bonus
        available_legal_actions = list(self.get_legal_actions())
        num_legal_moves = len(available_legal_actions)
        cumulative_score += num_legal_moves * 10

        # Stochastic noise for tie-breaking
        stochastic_noise = random.uniform(0.0, 0.5)
        cumulative_score += stochastic_noise

        # Near-zero stabilization
        if -0.1 < cumulative_score < 0.1:
            cumulative_score += 0.3

        return cumulative_score

    def get_legal_actions(self):
        """Return all legal actions available to the current player."""
        return self.state.get_valid_actions(unit="current")

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        is_term, _ = self.state.is_terminal()
        return is_term
