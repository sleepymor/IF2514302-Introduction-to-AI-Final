import numpy as np
import random
import math

from algorithm.mcts.mctsnode import MCTSNode
from algorithm.astar.astar import AStar
from utils.logger import Logger


class MCTS:
    """Monte Carlo Tree Search algorithm implementation."""

    def __init__(self, iterations=2000, exploration_constant=1.4, max_sim_depth=500):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_sim_depth = max_sim_depth
        self.log = Logger("MCTS")

    def search(self, node):
        """Execute MCTS search algorithm."""
        root_state = MCTSNode(state=node).state
        valid_actions = list(root_state.get_valid_actions(unit="current"))

        for action in valid_actions:
            temp_state = root_state.clone()
            temp_state.step(action, simulate=True)
            if tuple(temp_state.player_pos) == temp_state.goal:
                self.log.info(f"Instant Win found at {action}!")
                return action

        root = MCTSNode(state=node)

        for _ in range(self.iterations):
            selected = self.selection(root)
            expanded = self.expansion(selected)
            reward = self.simulation(expanded)
            self.backpropagation(expanded, reward)

        if not root.children:
            chosen = random.choice(valid_actions)
            return chosen, {"nodes_visited": 0, "win_probability": 0.0}

        best_child = max(root.children, key=lambda child: child.visits)
        mcts_action = best_child.action

        win_rate = (
            best_child.total_wins / best_child.visits if best_child.visits > 0 else 0.0
        )
        self.log.info(
            f"MCTS suggests: {mcts_action}, visits: {best_child.visits}, win_rate: {win_rate:.2f}"
        )

        # Count expanded nodes for reporting
        def count_nodes(n):
            total = 1
            for c in n.children:
                total += count_nodes(c)
            return total

        nodes = count_nodes(root)

        enemy_pos = tuple(root_state.enemy_pos)

        dist_to_enemy = abs(mcts_action[0] - enemy_pos[0]) + abs(
            mcts_action[1] - enemy_pos[1]
        )
        if dist_to_enemy <= 2:
            self.log.warning(
                f"DANGER! MCTS chose a suicide move {mcts_action} (Dist {dist_to_enemy}). Overriding..."
            )

            best_safe_action = None
            max_dist = -1

            for action in valid_actions:
                manhattan_dist = abs(action[0] - enemy_pos[0]) + abs(
                    action[1] - enemy_pos[1]
                )
                if manhattan_dist > 2:
                    if manhattan_dist > max_dist:
                        max_dist = manhattan_dist
                        best_safe_action = action

            if best_safe_action:
                self.log.info(
                    f"Safety Override: Switched to {best_safe_action} (Dist {max_dist})"
                )
                return best_safe_action
            else:
                self.log.warning("No safe moves available. Accepting fate.")

        return mcts_action, {"nodes_visited": nodes, "win_probability": float(win_rate)}

    def selection(self, node: MCTSNode) -> MCTSNode:
        """Select phase of MCTS."""
        while not node.is_terminal():
            if not node.children:
                return node
            if not node.is_fully_expanded():
                return node
            node = node.best_child(exploration_constant=self.exploration_constant)
        return node

    def expansion(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase of MCTS."""
        if node.is_terminal() or not node.untried_actions:
            return node

        action = random.choice(node.untried_actions)
        new_state = node.state.clone()
        new_state.step(action, simulate=True)

        child = MCTSNode(new_state, parent=node, action=action)
        node.add_child(child)
        return child

    def simulation(self, node: MCTSNode) -> float:
        """Simulation phase of MCTS."""
        state = node.state.clone()

        is_term, reason = state.is_terminal()
        if is_term:
            reward = self.rollout_reward(state, reason)
            return MCTSNode._normalize_score(reward)

        for _ in range(self.max_sim_depth):
            if state.turn == "player":

                legal = list(state.get_valid_actions(unit="current"))
                if not legal:
                    break

                action = random.choice(legal)
                if random.random() < 0.7:
                    best_dist = float("inf")
                    goal = state.goal
                    for act in legal:
                        dist = abs(act[0] - goal[0]) + abs(act[1] - goal[1])
                        if dist < best_dist:
                            best_dist = dist
                            action = act

                state.step(action, simulate=True)

            elif state.turn == "enemy":
                enemy_action = self.fast_enemy_policy(state)
                state.step(enemy_action, simulate=True)

            is_term, reason = state.is_terminal()
            if is_term:
                reward = self.rollout_reward(state, reason)
                return MCTSNode._normalize_score(reward)

        reward = self.rollout_reward(state, reason=None)
        return MCTSNode._normalize_score(reward)

    def fast_enemy_policy(self, state):
        """Fast enemy policy for simulation."""
        enemy_pos = tuple(state.enemy_pos)
        player_pos = tuple(state.player_pos)

        legal_moves = list(state.get_move_range(state.enemy_pos, move_range=2))

        if player_pos in legal_moves:
            return player_pos

        legal_moves.sort(key=lambda m: abs(m[0] - player_pos[0]) + abs(m[1] - player_pos[1]))

        return legal_moves[0] if legal_moves else enemy_pos

    def rollout_reward(self, state, reason):
        """Calculate reward for rollout."""
        if reason == "goal":
            return 1.0
        if reason == "trap" or reason == "caught":
            return 0.0

        player_pos = tuple(state.player_pos)
        enemy_pos = tuple(state.enemy_pos)
        goal_pos = state.goal

        dist_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        max_dist = state.width + state.height

        score = 0.1 + 0.7 * (1.0 - (dist_goal / max_dist))

        dist_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )

        if dist_enemy <= 2:
            return 0.0
        elif dist_enemy <= 3:
            score *= 0.5

        return max(0.0, min(0.9, score))

    def backpropagation(self, node: MCTSNode, reward: float):
        """Backpropagation phase of MCTS."""
        while node is not None:
            node.visits += 1
            node.total_wins += reward
            node = node.parent
