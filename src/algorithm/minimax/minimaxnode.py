from environment.environment import TacticalEnvironment


class MinimaxNode:
    """Node representation for Minimax algorithm with normalized [0.0, 1.0] evaluation."""

    def __init__(
        self,
        state: TacticalEnvironment,
        parent: "MinimaxNode | None" = None,
        action=None,
    ):
        """Initialize a Minimax node."""
        self.state = state
        self.parent = parent
        self.action = action

    def evaluate(self) -> float:
        """
        Evaluate current state with goal proximity as primary objective.

        Returns:
            float: Evaluation score in range [-1.0, 1.0].
                - 1.0: Goal reached (win)
                - -1.0: Caught or hit trap (loss)
                - Otherwise: Heuristic based on distance to goal and safety
        """
        player_pos = tuple(self.state.player_pos)
        enemy_pos = tuple(self.state.enemy_pos)
        goal_pos = tuple(self.state.goal)

        # Terminal states
        if player_pos == goal_pos:
            return 1.0  # WIN

        if player_pos == enemy_pos or player_pos in self.state.traps:
            return -1.0  # LOSS

        # Goal proximity scoring - PRIMARY OBJECTIVE (80% weight)
        manhattan_dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(
            player_pos[1] - goal_pos[1]
        )
        max_board_distance = float(self.state.width + self.state.height)
        goal_closeness = 1.0 - (manhattan_dist_to_goal / max_board_distance)
        goal_closeness = max(0.0, min(1.0, goal_closeness))

        cumulative_score = goal_closeness * 0.8

        # Enemy avoidance penalty - SECONDARY OBJECTIVE (20% weight)
        manhattan_dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )

        # Graduated penalty based on enemy distance
        if manhattan_dist_to_enemy <= 1:
            enemy_penalty_factor = 0.3  # Severe danger
        elif manhattan_dist_to_enemy == 2:
            enemy_penalty_factor = 0.6
        elif manhattan_dist_to_enemy == 3:
            enemy_penalty_factor = 0.8
        elif manhattan_dist_to_enemy == 4:
            enemy_penalty_factor = 0.9
        else:
            enemy_penalty_factor = 1.0  # Safe

        # Apply enemy avoidance as 20% of score
        enemy_penalty = (enemy_penalty_factor - 1.0) * 0.2
        cumulative_score += enemy_penalty

        # Normalize to [-1.0, 1.0]
        cumulative_score = max(-1.0, min(1.0, cumulative_score))
        return cumulative_score

    def get_legal_actions(self) -> list:
        """Return all legal actions from current state."""
        return list(self.state.get_valid_actions(unit="current"))

    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        is_term, _ = self.state.is_terminal()
        return is_term

    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        """Normalize raw score to [0.0, 1.0]."""
        raw_score = max(-1.0, min(1.0, raw_score))
        return 0.5 * (raw_score + 1.0)
