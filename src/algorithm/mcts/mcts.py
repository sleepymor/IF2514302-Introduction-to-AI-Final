import numpy as np
import random
import math

from algorithm.mcts.mctsnode import MCTSNode
from algorithm.astar.astar import AStar
from utils.logger import Logger


class MCTS:
    """Monte Carlo Tree Search algorithm implementation."""

    def __init__(self, iterations=3000, exploration_constant=1.4, max_sim_depth=200):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_sim_depth = max_sim_depth
        self.log = Logger("MCTS")

    def search(self, node):
        """Execute MCTS search algorithm."""
        # 1. Cek Instant Win di Root
        root_state = MCTSNode(state=node).state
        valid_actions = list(root_state.get_valid_actions(unit='current'))

        # Optimasi: Jika bisa menang sekarang, langsung ambil!
        for action in valid_actions:
            temp_state = root_state.clone()
            temp_state.step(action, simulate=True)
            if tuple(temp_state.player_pos) == temp_state.goal:
                self.log.info(f"Instant Win found at {action}!")
                return action

        root = MCTSNode(state=node)

        # 2. Loop MCTS
        for i in range(self.iterations):
            selected = self.selection(root)
            expanded = self.expansion(selected)
            reward = self.simulation(expanded)
            self.backpropagation(expanded, reward)

        # 3. Pilih Langkah Terbaik MCTS
        if not root.children:
            return random.choice(valid_actions)

        best_child = max(root.children, key=lambda c: c.visits)
        mcts_action = best_child.action

        win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0
        self.log.info(f"MCTS suggests: {mcts_action}, visits: {best_child.visits}, win_rate: {win_rate:.2f}")

        # --- 4. SAFETY OVERRIDE (JARING PENGAMAN) ---
        # Jika MCTS menyarankan langkah bunuh diri (masuk range musuh), KITA TOLAK.

        enemy_pos = tuple(root_state.enemy_pos)

        # Hitung jarak langkah MCTS ke musuh
        dist_to_enemy = abs(mcts_action[0] - enemy_pos[0]) + abs(mcts_action[1] - enemy_pos[1])

        # Jika langkah tersebut masuk range serangan musuh (<= 2)
        if dist_to_enemy <= 2:
            self.log.warning(f"DANGER! MCTS chose a suicide move {mcts_action} (Dist {dist_to_enemy}). Overriding...")

            # Cari langkah alternatif yang Paling Aman (Paling jauh dari musuh)
            best_safe_action = None
            max_dist = -1

            for action in valid_actions:
                d = abs(action[0] - enemy_pos[0]) + abs(action[1] - enemy_pos[1])
                # Pastikan langkah alternatif ini lebih aman dari pilihan MCTS
                if d > 2:
                    if d > max_dist:
                        max_dist = d
                        best_safe_action = action

            # Jika ketemu langkah aman, pakai itu
            if best_safe_action:
                self.log.info(f"Safety Override: Switched to {best_safe_action} (Dist {max_dist})")
                return best_safe_action
            else:
                self.log.warning("No safe moves available. Accepting fate.")

        return mcts_action

    def selection(self, node: MCTSNode) -> MCTSNode:
        """Select phase of MCTS."""
        while not node.is_terminal():
            if not node.children:
                return node
            if not node.is_fully_expanded():
                return node
            node = node.best_child(c=self.exploration_constant)
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
            return self.rollout_reward(state, reason)

        for _ in range(self.max_sim_depth):
            if state.turn == "player":
                # Kebijakan Player: Campuran Cerdas & Random
                legal = list(state.get_valid_actions(unit='current'))
                if not legal:
                    break

                # 70% coba maju ke goal, 30% random (eksplorasi)
                action = random.choice(legal)
                if random.random() < 0.7:
                    best_dist = float('inf')
                    goal = state.goal
                    # Cari langkah terdekat ke goal secara Manhattan
                    for a in legal:
                        d = abs(a[0] - goal[0]) + abs(a[1] - goal[1])
                        if d < best_dist:
                            best_dist = d
                            action = a

                state.step(action, simulate=True)

            elif state.turn == "enemy":
                # Kebijakan Musuh
                enemy_action = self.fast_enemy_policy(state)
                state.step(enemy_action, simulate=True)

            is_term, reason = state.is_terminal()
            if is_term:
                return self.rollout_reward(state, reason)

        return self.rollout_reward(state, reason=None)

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
        min_dist = float('inf')
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
        dist_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])

        if dist_enemy <= 2:
            score *= 0.1  # HANCURKAN skornya jika sangat dekat (Zona Maut)
        elif dist_enemy <= 3:
            score *= 0.5  # Potong skor 50% jika agak dekat (Zona Waspada)

        return max(0.0, min(0.9, score))

    def backpropagation(self, node: MCTSNode, reward: float):
        """Backpropagation phase of MCTS."""
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent