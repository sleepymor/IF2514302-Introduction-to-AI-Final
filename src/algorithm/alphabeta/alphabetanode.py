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
            float: Evaluation score in range [-1.0, 1.0] where:
                - 1.0: Guaranteed win (goal reached)
                - -1.0: Guaranteed loss (caught or trapped)
                - Otherwise: Heuristic value based on goal proximity and safety
        """
        player_pos = tuple(self.state.player_pos)
        enemy_pos = tuple(self.state.enemy_pos)
        goal_pos = self.state.goal

        # Terminal states
        if player_pos == goal_pos:
            return 1.0  # WIN

        if player_pos == enemy_pos or player_pos in self.state.traps:
            return -1.0  # LOSS

        # Non-terminal heuristic evaluation
        cumulative_score = 0.0

        # Goal proximity BONUS (closer = better) - PRIMARY OBJECTIVE
        manhattan_dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(
            player_pos[1] - goal_pos[1]
        )
        # Normalize distance to goal (max distance on 30x15 grid is ~45)
        max_distance = self.state.width + self.state.height
        goal_bonus = (max_distance - manhattan_dist_to_goal) / max_distance
        cumulative_score += goal_bonus * 0.8  # 80% weight on goal proximity

        # Enemy proximity penalty (danger zone) - avoid being close to enemy
        manhattan_dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )
        if manhattan_dist_to_enemy <= 5 and manhattan_dist_to_enemy > 0:
            # Heavy penalty for being too close
            enemy_penalty = -1.0 / (manhattan_dist_to_enemy + 1.0)
            cumulative_score += enemy_penalty * 0.2  # 20% weight on enemy avoidance
        else:
            # Safe distance bonus
            cumulative_score += 0.05

        # Clamp to [-1.0, 1.0] range for consistent behavior
        cumulative_score = max(-1.0, min(1.0, cumulative_score))

        return cumulative_score

    def get_legal_actions(self):
        """Return all legal actions available to the current player."""
        return self.state.get_valid_actions(unit="current")

    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        is_term, _ = self.state.is_terminal()
        return is_term
