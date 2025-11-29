import numpy as np
import random
import math
from environment.environment import TacticalEnvironment
from algorithm.mcts.mctsnode import MCTSNode
from algorithm.astar.astar import AStar
from utils.logger import Logger

log = Logger("MCTS")


class MCTS:
    def __init__(self, iterations=50, exploration_constant=1.4, max_sim_depth=40):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_sim_depth = max_sim_depth

    def search(self, node: TacticalEnvironment) -> tuple:
        root = MCTSNode(state=node)

        successes = 0
        failures = 0

        for i in range(self.iterations):
            # Selection: traverse tree using UCB

            selected = self._selection(root)

            # Expansion: add new child if not terminal
            expanded = self._expansion(selected)

            # Simulation: rollout from new node
            result = self._simulation(expanded)

            # Backpropagation: update statistics
            self._backpropagation(expanded, result)

            # Tracking success rate
            if result > 0:
                successes += 1
            elif result < 0:
                failures += 1

        # log.info(
        #     f"Iterations: {self.iterations}, Successes: {successes}, Failures: {failures}"
        # )
        # log.info(f"Root visits: {root.visits}, Root children: {len(root.children)}")

        if not root.children:
            log.error("No children generated! Check expansion logic.")
            return None

        best_child = max(
            root.children, key=lambda child: child.visits
        )  # ← Gunakan visits!

        avg_reward = best_child.wins / best_child.visits if best_child.visits > 0 else 0
        # log.info(
        #     f"Best: {best_child.action} | Visits: {best_child.visits} | "
        #     f"Avg Reward: {avg_reward:.2f}"
        # )

        return best_child.action

    def _selection(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCB until reaching
        unexpanded node or terminal state
        """
        while not node.is_terminal():

            if not node.children:
                return node

            if not node.is_fully_expanded():
                return node

            node = node.best_child(c=self.exploration_constant)

            if node is None:
                break

        return node

    def _expansion(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: add one new child node for an untried action
        """
        if node.is_terminal() or not node.untried_actions:
            return node

        action = random.choice(node.untried_actions)

        new_state = node.state.clone()
        new_state.step(action, simulate=True)

        child = MCTSNode(new_state, parent=node, action=action)
        node.add_child(child)

        return child

    def _simulation(self, node: MCTSNode) -> float:
        """
        Simulation phase: play out game randomly from current state
        Returns reward from perspective of current player
        """
        state = node.state.clone()

        for step in range(self.max_sim_depth):
            is_term, reason = state.is_terminal()
            if is_term:
                break

            if state.turn == "player":
                legal = list(state.get_move_range(state.player_pos))
                if not legal:
                    break  # No legal moves, game over

                action = self._heuristic_rollout_policy(state, legal)
                if action is None:  # Safety check
                    action = random.choice(legal)

                result = state.step(action, simulate=True)

                # Check if step caused terminal state
                if result and result[0]:
                    break

            elif state.turn == "enemy":
                enemy_action = self._enemy_policy(state)

                result = state.step(enemy_action, simulate=True)

                # Check if step caused terminal state
                if result and result[0]:
                    break

        return self._rollout_reward(state)

    def _heuristic_rollout_policy(
        self, state: TacticalEnvironment, legal_moves: list
    ) -> tuple:
        """
        Memilih gerakan 'ringan' yang cerdas, bukan acak.
        Tujuannya: 1. Mendekati Goal, 2. Menjauhi Musuh.
        """
        if not legal_moves:
            return None  # No moves available

        enemy_pos = tuple(state.enemy_pos)
        goal_pos = state.goal

        best_move = None
        best_score = -float("inf")

        for move in legal_moves:
            # Skip walls and traps (these shouldn't be in legal_moves anyway)
            if move in state.walls or move in state.traps:
                continue

            # Calculate score
            dist_goal = abs(move[0] - goal_pos[0]) + abs(move[1] - goal_pos[1])

            dist_enemy = abs(move[0] - enemy_pos[0]) + abs(move[1] - enemy_pos[1])

            score = -dist_goal
            # score = (dist_enemy * 1.0) - (dist_goal * 2.0)

            # Penalty if too close to enemy
            if dist_enemy <= 1:
                score -= 3.0

            if score > best_score:
                best_score = score
                best_move = move

        # Safety check
        if best_move is None and legal_moves:
            best_move = random.choice(legal_moves)

        return best_move

    def _enemy_policy(self, state: TacticalEnvironment) -> tuple:
        enemy_moves = list(state.get_move_range(state.enemy_pos))

        if not enemy_moves:
            return tuple(state.enemy_pos)

        # Use A* to find path to player
        a_star = AStar(env=state)
        start = tuple(state.enemy_pos)
        goal = tuple(state.player_pos)
        path = a_star.search(start, goal)

        # No path found or path too short
        if not path or len(path) <= 1:
            return random.choice(enemy_moves)

        # Try to move along the path
        path = path[1:]  # Remove current position

        # Try positions along the path (closer first)
        for next_pos in path[:2]:  # Check first 3 steps
            if next_pos in enemy_moves:
                return next_pos

        # Fallback: random legal move
        return random.choice(enemy_moves)

    def _rollout_reward(self, state: TacticalEnvironment) -> float:
        """
        Compute the rollout reward from the player's perspective.
        """
        player_position = tuple(state.player_pos)
        enemy_position = tuple(state.enemy_pos)

        if player_position == state.goal:
            return 1.0  # Win condition

        if player_position == enemy_position or player_position in state.traps:
            return -1.0  # Loss condition

        # Stable heuristic reward
        dist_goal = abs(player_position[0] - state.goal[0]) + abs(
            player_position[1] - state.goal[1]
        )
        dist_enemy = abs(player_position[0] - enemy_position[0]) + abs(
            player_position[1] - enemy_position[1]
        )

        # Base score
        goal_score = -0.05 * dist_goal

        if dist_enemy <= 1:
            enemy_penalty = -0.5  # Heavy penalty for being next to enemy
        elif dist_enemy <= 2:
            enemy_penalty = -0.2  # Medium penalty
        else:
            enemy_penalty = 0.02 * dist_enemy  # Small bonus for being far

        player_legal_moves = state.get_move_range(player_position)
        mobility_bonus = 0.01 * len(player_legal_moves)

        score = goal_score + enemy_penalty + mobility_bonus
        # score = (
        #     -0.05 * dist_goal  # semakin dekat goal semakin bagus
        #     + 0.02 * dist_enemy  # semakin jauh dari musuh semakin bagus
        # )
        return max(score, -0.9)

    def _backpropagation(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase: update node statistics up to root
        """
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
