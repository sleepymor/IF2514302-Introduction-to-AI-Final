import numpy as np
import random
import math
from environment.environment import TacticalEnvironment
from algorithm.mcts.mctsnode import MCTSNode
from algorithm.astar.astar import AStar
from utils.logger import Logger


class MCTS:
    """Monte Carlo Tree Search algorithm implementation."""

    def __init__(self, iterations=2500, exploration_constant=2.0, max_sim_depth=150, verbose=True):
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_sim_depth = max_sim_depth
        self.log = Logger("MCTS")
        self.verbose = verbose

    def search(self, node):
        """Execute MCTS search algorithm."""
        # 1. Cek Instant Win di Root
        root_state = MCTSNode(state=node).state
        valid_actions = list(root_state.get_valid_actions(unit="current"))

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

        # Select best child by win rate (exploitation), but require minimum visits for stability
        min_visits = max(1, root.visits // len(root.children) // 2)  # at least 25% of fair share
        candidate_children = [c for c in root.children if c.visits >= min_visits]
        
        if not candidate_children:
            candidate_children = root.children
        
        best_child = max(candidate_children, key=lambda c: c.wins / (c.visits + 1) if c.visits > 0 else 0)
        mcts_action = best_child.action

        if self.verbose:
            win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0
            self.log.info(
                f"MCTS suggests: {mcts_action}, visits: {best_child.visits}, win_rate: {win_rate:.2f}"
            )

        # --- 4. SAFETY OVERRIDE (JARING PENGAMAN) ---
        # Jika MCTS menyarankan langkah bunuh diri (masuk range musuh), KITA TOLAK.
        enemy_pos = tuple(root_state.enemy_pos)

        # Hitung jarak langkah MCTS ke musuh (aksi adalah koordinat target)
        try:
            dist_to_enemy = abs(mcts_action[0] - enemy_pos[0]) + abs(
                mcts_action[1] - enemy_pos[1]
            )
        except Exception:
            dist_to_enemy = float("inf")

        # Jika langkah tersebut masuk range serangan musuh (<= 2), override
        if dist_to_enemy <= 2:
            if self.verbose:
                self.log.warning(
                    f"DANGER! MCTS chose a risky move {mcts_action} (Dist {dist_to_enemy}). Overriding..."
                )

            # Cari langkah alternatif yang Paling Aman (Paling jauh dari musuh)
            best_safe_action = None
            max_dist = -1

            for action in valid_actions:
                d = abs(action[0] - enemy_pos[0]) + abs(action[1] - enemy_pos[1])
                # Prefer any action that is strictly safer than the MCTS suggestion
                if d > dist_to_enemy and d > max_dist:
                    max_dist = d
                    best_safe_action = action

            # Jika ketemu langkah aman, pakai itu
            if best_safe_action:
                if self.verbose:
                    self.log.info(
                        f"Safety Override: Switched to {best_safe_action} (Dist {max_dist})"
                    )
                return best_safe_action
            else:
                # Kalau tidak ada langkah yang lebih aman, pilih opsi paling jauh dari musuh
                fallback = max(valid_actions, key=lambda a: abs(a[0] - enemy_pos[0]) + abs(a[1] - enemy_pos[1]))
                if self.verbose:
                    self.log.warning(f"No strictly safer move found; fallback to {fallback}")
                return fallback
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
                player_action = self._player_policy(state)
                state.step(player_action, simulate=True)

            elif state.turn == "enemy":
                enemy_action = self._enemy_policy(state)
                state.step(enemy_action, simulate=True)

            is_term, reason = state.is_terminal()
            if is_term:
                return self.rollout_reward(state, reason)

        return self.rollout_reward(state, reason=None)

    def _player_policy(self, state: TacticalEnvironment):
        goal = tuple(state.goal)
        player = tuple(state.player_pos)
        enemy = tuple(state.enemy_pos)

        legal_moves = list(state.get_valid_actions(unit="current"))
        if not legal_moves:
            return None

        # Direct finish if reachable
        if goal in legal_moves:
            return goal

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Epsilon-greedy: 85% greedy (A*/manhattan), 15% random exploration
        eps = 0.15
        if random.random() < eps:
            return random.choice(legal_moves)

        # Greedy: use A* path's next-step if possible (deterministic, stronger guidance)
        try:
            astar = AStar(state)
            path = astar.search(player, goal)
            if path and len(path) > 1:
                next_step = path[1]
                # Check if next step is safe (not on trap, not too close to enemy)
                if next_step in legal_moves:
                    dist_to_enemy_next = manhattan(next_step, enemy)
                    if next_step not in state.traps and dist_to_enemy_next > 1:
                        return next_step
        except Exception:
            pass

        # Fallback: Manhattan greedy with multi-factor scoring
        best_score = float("-inf")
        best_move = None
        
        for move in legal_moves:
            # Avoid traps strongly
            if move in state.traps:
                continue
            
            dist_to_goal = manhattan(move, goal)
            dist_to_enemy = manhattan(move, enemy)
            
            # Score: minimize goal distance (primary), maximize enemy distance (secondary)
            score = -dist_to_goal * 10.0 + dist_to_enemy * 1.0
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is not None:
            return best_move
        
        # Last resort: pick safest of remaining
        return max(legal_moves, key=lambda m: manhattan(m, enemy))

    def _enemy_policy(self, state):
        """Fast enemy policy for simulation."""
        enemy_pos = tuple(state.enemy_pos)
        player_pos = tuple(state.player_pos)

        legal_moves = list(state.get_move_range(state.enemy_pos, move_range=2))

        if player_pos in legal_moves:
            return player_pos

        best_move = enemy_pos
        min_dist = float("inf")

        for move in legal_moves:
            dist = abs(move[0] - player_pos[0]) + abs(move[1] - player_pos[1])
            if dist < min_dist:
                min_dist = dist
                best_move = move

        return best_move

    def rollout_reward(self, state, reason):
        """Calculate reward for rollout."""
        # Normalize rewards to [0.0, 1.0]
        # Win
        if reason == "goal":
            return 1.0

        # Loss (caught or trap)
        if reason == "trap" or reason == "caught":
            return 0.0

        # Timeout / Survival: compute a partial score based on closeness to goal
        player_pos = tuple(state.player_pos)
        enemy_pos = tuple(state.enemy_pos)
        goal_pos = tuple(state.goal)

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        dist_goal = manhattan(player_pos, goal_pos)
        # max possible manhattan in grid
        max_dist = float(state.width + state.height)
        # closeness ratio: 0 (far) .. 1 (at goal)
        closeness = 1.0 - (dist_goal / max_dist)
        closeness = max(0.0, min(1.0, closeness))

        # Map closeness to [0.25, 0.95] so timeouts provide meaningful signal
        base_score = 0.25 + 0.7 * closeness

        # Apply gentle repulsion if final pos is too close to enemy
        dist_enemy = manhattan(player_pos, enemy_pos)
        # Gentler penalties the closer we are:
        if dist_enemy <= 1:
            penalty_factor = 0.5
        elif dist_enemy == 2:
            penalty_factor = 0.7
        elif dist_enemy == 3:
            penalty_factor = 0.85
        else:
            penalty_factor = 1.0

        score = base_score * penalty_factor

        # clamp to [0.0, 0.99]
        return max(0.0, min(0.99, score))

    def backpropagation(self, node: MCTSNode, reward: float):
        """Backpropagation phase of MCTS."""
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
