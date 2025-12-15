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
        Evaluate current state with normalized score in [0.0, 1.0].

        Returns:
            float: Evaluation combining goal proximity, enemy avoidance,
                   mobility, and trap penalties.
        """
        player_pos = tuple(self.state.player_pos)
        enemy_pos = tuple(self.state.enemy_pos)
        goal_pos = tuple(self.state.goal)

        # Terminal states
        if player_pos == goal_pos:
            return float("inf")

        if player_pos == enemy_pos or player_pos in self.state.traps:
            return float("-inf")

        # Goal proximity scoring
        manhattan_dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(
            player_pos[1] - goal_pos[1]
        )
        max_board_distance = float(self.state.width + self.state.height)
        goal_closeness = 1.0 - (manhattan_dist_to_goal / max_board_distance)
        goal_closeness = max(0.0, min(1.0, goal_closeness))

        cumulative_score = goal_closeness * 0.6

        # Enemy avoidance penalty (distance-based)
        manhattan_dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )

        if manhattan_dist_to_enemy <= 1:
            penalty_factor = 0.3
        elif manhattan_dist_to_enemy == 2:
            penalty_factor = 0.6
        elif manhattan_dist_to_enemy == 3:
            penalty_factor = 0.8
        else:
            penalty_factor = 1.0

        cumulative_score *= penalty_factor

        # Mobility bonus
        num_legal_moves = len(self.get_legal_actions())
        mobility_bonus = (num_legal_moves / 8.0) * 0.1
        cumulative_score += mobility_bonus

        # Trap proximity penalty
        if self.state.traps:
            min_trap_distance = min(
                abs(player_pos[0] - trap[0]) + abs(player_pos[1] - trap[1])
                for trap in self.state.traps
            )
            if min_trap_distance <= 1:
                cumulative_score *= 0.9

        # Normalize to [0.0, 1.0]
        normalized_score = self._normalize_score(cumulative_score)
        return normalized_score

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
