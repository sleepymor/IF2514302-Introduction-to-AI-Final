import numpy as np
import random
import math

from environment.environment import TacticalEnvironment
from algorithm.mcts.mctsnode import MCTSNode
from algorithm.astar.astar import AStar
from utils.logger import Logger


class MCTS:
    """Monte Carlo Tree Search algorithm implementation."""

    def __init__(self, iterations=1500, exploration_constant=1.4, max_sim_depth=100):
        """
        Initialize MCTS.

        Args:
            iterations: Number of MCTS iterations to perform
            exploration_constant: UCB1 exploration parameter (typically sqrt(2) ≈ 1.4)
            max_sim_depth: Maximum depth for simulation rollouts
        """
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_sim_depth = max_sim_depth
        self.log = Logger("MCTS")

    def search(self, node):
        """
        Execute MCTS to find best action.

        Args:
            node: Current game state

        Returns:
            Best action found by MCTS
        """
        root = MCTSNode(state=node)

        # Check for immediate win
        valid_actions = list(root.state.get_valid_actions(unit="current"))
        for action in valid_actions:
            temp_state = root.state.clone()
            temp_state.step(action, simulate=True)
            if tuple(temp_state.player_pos) == temp_state.goal:
                self.log.info(f"Immediate win found: {action}")
                return action

        # Main MCTS loop
        for _ in range(self.iterations):
            # 1. Selection: traverse tree using UCB1
            selected = self._selection(root)

            # 2. Expansion: add new child node
            expanded = self._expansion(selected)

            # 3. Simulation: random rollout from new node
            reward = self._simulation(expanded)

            # 4. Backpropagation: update statistics
            self._backpropagation(expanded, reward)

        # Return action with most visits (most robust choice)
        if not root.children:
            return random.choice(valid_actions)

        best_child = max(root.children, key=lambda c: c.visits)
        win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0

        self.log.info(
            f"Best action: {best_child.action}, visits: {best_child.visits}, "
            f"win_rate: {win_rate:.2f}"
        )

        return best_child.action

    def _selection(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCB1 until reaching expandable node.

        Args:
            node: Starting node (usually root)

        Returns:
            Node to expand
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node

            # Select child with highest UCB1 value
            node = node.best_child(c=self.exploration_constant)
        return node

    def _expansion(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: add one untried child node.

        Args:
            node: Node to expand

        Returns:
            Newly created child node, or node itself if terminal
        """
        if node.is_terminal() or not node.untried_actions:
            return node

        # Pick random untried action
        action = random.choice(node.untried_actions)

        # Create new state
        new_state = node.state.clone()
        new_state.step(action, simulate=True)

        # Create and add child node
        child = MCTSNode(new_state, parent=node, action=action)
        node.add_child(child)

        return child

    def _simulation(self, node: MCTSNode) -> float:
        """
        Simulation phase: perform random rollout from node.

        Args:
            node: Starting node for simulation

        Returns:
            Reward value (0.0 to 1.0)
        """
        state = node.state.clone()

        # Check if already terminal
        is_term, reason = state.is_terminal()
        if is_term:
            return self.rollout_reward(state, reason)

        # Rollout with simple policies
        for _ in range(self.max_sim_depth):
            if state.turn == "player":
                action = self._player_policy(state)
            else:  # enemy turn
                action = self._enemy_policy(state)

            if action is None:
                break

            state.step(action, simulate=True)

            # Check terminal condition
            is_term, reason = state.is_terminal()
            if is_term:
                return self.rollout_reward(state, reason)

        # Simulation depth limit reached, use heuristic
        return self.rollout_reward(state, reason=None)

    def _player_policy(self, state: TacticalEnvironment) -> tuple:
        """
        Simple player policy for simulation: move toward goal.

        Args:
            state: Current game state

        Returns:
            Action to take
        """
        legal = list(state.get_valid_actions(unit="current"))
        if not legal:
            return None

        # Mix of greedy (70%) and random (30%) for exploration
        if random.random() < 0.7:
            goal = state.goal
            best_action = min(
                legal, key=lambda a: abs(a[0] - goal[0]) + abs(a[1] - goal[1])
            )
            return best_action
        else:
            return random.choice(legal)

    def _enemy_policy(self, state: TacticalEnvironment, reason: str) -> tuple:
        """
        Evaluate terminal or non-terminal state.

        Args:
            state: Game state to evaluate
            reason: Terminal reason if state is terminal

        Returns:
            Reward value (0.0 to 1.0)
        """

        if reason == "goal":
            return 1.0  # Win
        if reason in ["trap", "caught"]:
            return 0.0  # Loss

        # Heuristic for non-terminal states
        player_pos = tuple(state.player_pos)
        enemy_pos = tuple(state.enemy_pos)
        goal_pos = state.goal

        # Distance to goal (normalized)
        dist_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        max_dist = state.width + state.height
        goal_score = 1.0 - (dist_goal / max_dist)

        # Distance to enemy (safety bonus)
        dist_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )

        # Penalty for being too close to enemy
        if dist_enemy <= 2:
            safety_penalty = 0.5  # High danger
        elif dist_enemy <= 4:
            safety_penalty = 0.8  # Medium danger
        else:
            safety_penalty = 1.0  # Safe

        # Combined score (weighted toward goal progress)
        score = 0.7 * goal_score + 0.3 * safety_penalty

        return max(0.0, min(0.99, score))  # Keep below 1.0 for non-terminal

    def fast_enemy_policy(self, state):
        """Fast enemy policy for simulation."""
        enemy_pos = tuple(state.enemy_pos)
        player_pos = tuple(state.player_pos)  # <--- PERBAIKAN 1: Pastikan Tuple

        # Ambil range 2
        legal_moves = list(state.get_move_range(state.enemy_pos, move_range=2))

        # 1. KILL INSTANT
        # Sekarang pengecekan ini AKURAT karena tipe datanya sama (Tuple vs Set of Tuples)
        if player_pos in legal_moves:
            return player_pos

        # 2. CHASE (Greedy)
        best_move = enemy_pos
        min_dist = float("inf")
        random.shuffle(legal_moves)

        for move in legal_moves:
            dist = abs(move[0] - player_pos[0]) + abs(move[1] - player_pos[1])
            if dist < min_dist:
                min_dist = dist
                best_move = move

        return best_move

    def rollout_reward(self, state, reason):
        """Calculate reward for rollout."""
        # 1. Menang/Kalah Mutlak
        if reason == "goal":
            return 1.0
        if reason == "trap" or reason == "caught":
            return 0.0

        # 2. Heuristic Score (0.0 - 0.9)
        player_pos = tuple(state.player_pos)  # Pastikan Tuple
        enemy_pos = tuple(state.enemy_pos)
        goal_pos = state.goal

        dist_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        max_dist = state.width + state.height

        # Base Score: Progres ke Goal (Max 0.8)
        score = 0.1 + 0.7 * (1.0 - (dist_goal / max_dist))

        # --- PERBAIKAN 2: Penalti Jarak Aman (Social Distancing) ---
        # Jika jarak ke musuh <= 3 langkah, kurangi nilai heuristiknya.
        # Ini memaksa AI memilih jalur yang TIDAK MEPET musuh,
        # meskipun jalur mepet itu lebih dekat ke goal.
        dist_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )

        if dist_enemy <= 2:
            score *= 0.1  # HANCURKAN skornya jika sangat dekat (Zona Maut)
        elif dist_enemy <= 3:
            score *= 0.5  # Potong skor 50% jika agak dekat (Zona Waspada)

        return max(0.0, min(0.9, score))

    def _backpropagation(self, node: MCTSNode, reward: float) -> None:
        """Backpropagation phase of MCTS."""
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
