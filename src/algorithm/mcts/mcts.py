import numpy as np
import random
import math
from environment.environment import TacticalEnvironment
from algorithm.mcts.mctsnode import MCTSNode
from algorithm.astar.astar import AStar
from utils.logger import Logger

log = Logger("MCTS")


class MCTS: 
  def __init__(self, iterations=50, exploration_constant=1.4, max_sim_depth=400):
    self.iterations = iterations
    self.exploration_constant = exploration_constant
    self.max_sim_depth = max_sim_depth
    

  def search(self, node): 
    root_state = MCTSNode(state=node).state
    valid_actions = root_state.get_valid_actions(unit='current')
    
    for action in valid_actions:
        # Simulasikan 1 langkah
        temp_state = root_state.clone()
        temp_state.step(action, simulate=True)
        
        # Jika aksi ini mencapai goal, AMBIL SEGERA!
        if tuple(temp_state.player_pos) == temp_state.goal:
            log.info(f"Instant Win found at {action}!")
            return action
          
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
        enemy_action = self.enemy_policy(state)
        state.step(enemy_action, simulate=True)
    
    return self.rollout_reward(state)
  
  # punya ibnu
  def _heuristic_rollout_policy(self, state, legal_moves):
    """
    Memilih gerakan 'ringan' yang cerdas, bukan acak.
    Tujuannya: 1. Mendekati Goal, 2. Menjauhi Musuh.
    """
    player_pos = tuple(state.player_pos)
    enemy_pos = tuple(state.enemy_pos)
    goal_pos = state.goal
    
    # Hitung jarak saat ini ke goal (sebelum bergerak)
    current_dist_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
    
    best_move = None
    best_score = -float('inf')

    # Urutkan moves agar deterministik jika score sama, atau acak untuk variasi
    random.shuffle(legal_moves) 

    for move in legal_moves:
        if move in state.traps:
            continue
            
        dist_goal = abs(move[0] - goal_pos[0]) + abs(move[1] - goal_pos[1])
        dist_enemy = abs(move[0] - enemy_pos[0]) + abs(move[1] - enemy_pos[1])
        
        # --- LOGIKA SKOR BARU: MOMENTUM ---
        score = 0
        
        # 1. Base Score: Jarak ke Goal
        score -= dist_goal * 3.0 
        
        # 2. Momentum Bonus (KUNCI ANTI-OSCILLATION)
        # Jika gerakan ini membuat kita LEBIH DEKAT ke goal dibanding posisi sekarang, beri bonus!
        if dist_goal < current_dist_goal:
            score += 5.0  # Insentif besar untuk MAJU
        else:
            score -= 2.0  # Penalti kecil untuk MUNDUR/DIAM

        # 3. Safety Logic (Tetap dipertahankan agar tidak mati konyol)
        if dist_enemy <= 1:
            score -= 100.0 # SANGAT BAHAYA (Instant Death zone)
        elif dist_enemy <= 2:
            score -= 10.0  # Bahaya
        else:
            score += dist_enemy * 0.5 # Sedikit bonus jika jauh dari musuh
            
        if score > best_score:
            best_score = score
            best_move = move
            
    if best_move is None:
        return random.choice(legal_moves)
        
    return best_move

  def enemy_policy(self, state):
    enemy_moves = list(state.get_move_range(state.enemy_pos, move_range=2))
    player_pos_tuple = tuple(state.player_pos)
    
    # Jika posisi player ada dalam jangkauan gerak musuh -> SERANG!
    if player_pos_tuple in enemy_moves:
        return player_pos_tuple
      
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

    # 1. Terminal States
    if player_position == state.goal:
        return 10.0  # BERIKAN REWARD MASIF UNTUK MENANG (Bukan cuma 1.0)
    if player_position == enemy_position or player_position in state.traps:
        return -10.0 # BERIKAN PENALTI MASIF UNTUK KALAH

    # 2. Heuristic Calculation
    dist_goal = abs(player_position[0] - state.goal[0]) + abs(player_position[1] - state.goal[1])
    dist_enemy = abs(player_position[0] - enemy_position[0]) + abs(player_position[1] - enemy_position[1])

    score = 0
    
    # --- LOGIKA BARU: Agresif tapi Waspada ---
    
    # Fokus Utama: Semakin dekat goal, nilai semakin tinggi secara eksponensial
    # Ini mendorong AI mengambil langkah terjauh (5 langkah) dibanding pendek (3 langkah)
    goal_reward = 10.0 / (dist_goal + 1) 
    
    # Penalti musuh
    enemy_penalty = 0
    if dist_enemy <= 1:
        enemy_penalty = 20.0 # SANGAT BAHAYA (sebelahan) -> LARI!
    elif dist_enemy <= 2:
        enemy_penalty = 5.0  # Bahaya, hindari jika bisa
    elif dist_enemy <= 3:
        enemy_penalty = 1.0  # Waspada, tapi jangan putar balik kalau goal dekat

    score = goal_reward - enemy_penalty

    return score


  def backpropagation(self, node: MCTSNode, reward: float): 
    """
      Backpropagation phase: update node statistics up to root
    """
    while node is not None:
      node.visits += 1
      node.wins += reward
      node = node.parent

