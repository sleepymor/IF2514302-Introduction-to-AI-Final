import math
import random
from environment.environment import TacticalEnvironment


class MinimaxNode:
    """
    Node representation for Minimax algorithm.

    Uses normalized evaluation logic consistent with MCTS:
    - Higher scores (+) for moves closer to goal
    - Lower scores (-) for proximity to enemy
    - Scores normalized to [0.0, 1.0] range for consistency

    Attributes:
        state (TacticalEnvironment): Current game state
        parent (MinimaxNode): Parent node in tree
        action (tuple): Action that led to this state
        WIN_SCORE (float): Score for winning state (1.0)
        LOSE_SCORE (float): Score for losing state (0.0)
    """

    # Normalized terminal scores matching MCTS
    WIN_SCORE = 1.0
    LOSE_SCORE = 0.0

    def __init__(self, state: TacticalEnvironment, parent=None, action=None):
        """
        Initialize Minimax node.

        Args:
            state: Current game state
            parent: Parent node (default: None)
            action: Action that led to this state (default: None)
        """
        self.state = state
        self.parent = parent
        self.action = action

    def evaluate(self) -> float:
        """
        Evaluate current state with normalized scoring [0.0, 1.0].

        Evaluation components:
        1. Terminal states (win/loss): 1.0 or 0.0
        2. Goal proximity: Weighted by closeness to goal
        3. Enemy avoidance: Penalty for proximity to enemy
        4. Mobility bonus: Reward for having more legal moves

        Returns:
            float: Normalized evaluation score in [0.0, 1.0]
        """
        player_pos = tuple(self.state.player_pos)
        enemy_pos = tuple(self.state.enemy_pos)
        goal_pos = tuple(self.state.goal)

        # ========== 1. TERMINAL STATE CHECK ==========
        # Win condition
        if player_pos == goal_pos:
            return self.WIN_SCORE

        # Loss conditions (caught by enemy or stepped on trap)
        if player_pos == enemy_pos or player_pos in self.state.traps:
            return self.LOSE_SCORE

        # ========== 2. GOAL PROXIMITY SCORING ==========
        # Calculate Manhattan distance to goal
        dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(
            player_pos[1] - goal_pos[1]
        )

        # Maximum possible distance (diagonal of map)
        max_dist = float(self.state.width + self.state.height)

        # Normalize goal closeness to [0.0, 1.0]
        # 1.0 = at goal, 0.0 = maximum distance away
        goal_closeness = 1.0 - (dist_to_goal / max_dist)
        goal_closeness = max(0.0, min(1.0, goal_closeness))

        # Base score from goal proximity (weight: 0.6)
        score = goal_closeness * 0.6

        # ========== 3. ENEMY AVOIDANCE PENALTY ==========
        # Calculate Manhattan distance to enemy
        dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )

        # Apply distance-based penalty (matching MCTS rollout_reward logic)
        if dist_to_enemy <= 1:
            # Adjacent to enemy: severe penalty
            penalty_factor = 0.3
        elif dist_to_enemy == 2:
            # Within attack range: moderate penalty
            penalty_factor = 0.6
        elif dist_to_enemy == 3:
            # Nearby: light penalty
            penalty_factor = 0.8
        else:
            # Safe distance: no penalty
            penalty_factor = 1.0

        score *= penalty_factor

        # ========== 4. MOBILITY BONUS ==========
        # Reward having more legal moves (exploration capability)
        num_legal_moves = len(self.get_legal_actions())
        max_possible_moves = 8.0  # Maximum possible moves in grid

        # Normalize mobility to [0.0, 1.0] and add small bonus
        mobility_bonus = (num_legal_moves / max_possible_moves) * 0.1
        score += mobility_bonus

        # ========== 5. TRAP PROXIMITY PENALTY ==========
        # Penalize positions near traps
        if self.state.traps:
            min_trap_dist = min(
                abs(player_pos[0] - trap[0]) + abs(player_pos[1] - trap[1])
                for trap in self.state.traps
            )

            # Small penalty if very close to trap
            if min_trap_dist <= 1:
                score *= 0.9

        # Clamp final score to [0.0, 0.99] (only true win = 1.0)
        return max(0.0, min(0.99, score))

    def get_legal_actions(self) -> list:
        """
        Get all legal actions from current state.

        Returns:
            list: List of valid action tuples (x, y)
        """
        return list(self.state.get_valid_actions(unit="current"))

    def is_terminal(self) -> bool:
        """
        Check if current state is terminal.

        Returns:
            bool: True if state is terminal, False otherwise
        """
        is_term, _ = self.state.is_terminal()
        return is_term
