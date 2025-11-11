import numpy as np
import random
import math
from algorithm.mcts.mctsnode import MCTSNode
from algorithm.astar.astar import AStar
from utils.logger import Logger
log = Logger("MCTS")

class MCTS: 
  def __init__(self, iterations=50, exploration_constant=1.4, max_sim_depth=40):
    self.iterations = iterations
    self.exploration_constant = exploration_constant
    self.max_sim_depth = max_sim_depth
    

  def search(self, node): 
    root = MCTSNode(state=node)

    for _ in range(self.iterations):
      # Selection: traverse tree using UCB
      selected = self.selection(root)

      # Expansion: add new child if not terminal
      expanded = self.expansion(selected)
    
      # Simulation: rollout from new node
      result = self.simulation(expanded)
      # log.info(f"Simulation result: {result}")

      # Backpropagation: update statistics
      self.backpropagation(expanded, result)

    best_child = max(root.children, key=lambda c: c.wins / c.visits)

    log.info(f"Best action: {best_child.action}, visits: {best_child.visits} win rate: {best_child.wins/best_child.visits:.2%}")

    return best_child.action

      

  def selection(self, node:MCTSNode) -> MCTSNode:  
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
    return node

  def expansion(self, node: MCTSNode) -> MCTSNode: 
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

  def simulation(self, node: MCTSNode) -> float:
    """
      Simulation phase: play out game randomly from current state
      Returns reward from perspective of current player
    """
    state = node.state.clone()

    # TODO: Simulation
    for _ in range(self.max_sim_depth):

      if self.is_terminal_state(state):
        break

      if state.turn == "player":
        legal = list(state.get_move_range(state.player_pos))
        if not legal:
          break

        action = self._heuristic_rollout_policy(state, legal)
        state.step(action, simulate=True)

      elif state.turn == "enemy":
        enemy_action = self.enemy_policy(state)
        state.step(enemy_action, simulate=True)
    
    return self.rollout_reward(state)
  
  def _heuristic_rollout_policy(self, state, legal_moves):
    """
    Memilih gerakan 'ringan' yang cerdas, bukan acak.
    Tujuannya: 1. Mendekati Goal, 2. Menjauhi Musuh.
    """
    player_pos = tuple(state.player_pos)
    enemy_pos = tuple(state.enemy_pos)
    goal_pos = state.goal

    best_move = None
    best_score = -float('inf') # Kita ingin memaksimalkan skor

    for move in legal_moves:
        # Hitung 'skor' untuk setiap gerakan yang mungkin
        
        # 1. Seberapa dekat ke goal? (Kita ingin ini sekecil mungkin)
        dist_goal = abs(move[0] - goal_pos[0]) + abs(move[1] - goal_pos[1])
        
        # 2. Seberapa jauh dari musuh? (Kita ingin ini sebesar mungkin)
        dist_enemy = abs(move[0] - enemy_pos[0]) + abs(move[1] - enemy_pos[1])

        # Bobot: Kita anggap menjauhi musuh 2x lebih penting daripada mendekati goal
        # (Angka 2.0 ini bisa di-tuning oleh Ibnu!)
        score = (dist_enemy * 1.0) - (dist_goal * 2.0)

        if score > best_score:
            best_score = score
            best_move = move

    # Jika karena alasan tertentu tidak ada gerakan terbaik (misal semua skor sama)
    # kita kembali ke gerakan acak untuk menghindari error.
    if best_move is None:
        return random.choice(legal_moves)
        
    return best_move
  
  def enemy_policy(self, state):
    a_star = AStar(env=state)

    start = tuple(state.enemy_pos)
    goal = tuple(state.player_pos)

    path = a_star.search(start, goal)

    if path is None or len(path) <= 1:
        return start
    
    path = path[1:]

    index = min(2, len(path) - 1)

    next_tile = path[index]

    return next_tile


  def is_terminal_state(self, state):
    player_position = tuple(state.player_pos)
    enemy_position = tuple(state.enemy_pos)

    if player_position == state.goal:
        return True
    if player_position == enemy_position:
        return True
    if player_position in state.traps:
        return True
    return False

  def rollout_reward(self, state):
    """Reward kuat agar player menghindari enemy dan mendekati goal."""
    player_position = tuple(state.player_pos)
    enemy_position = tuple(state.enemy_pos)

    if player_position == state.goal:
        return 1.0
    if player_position == enemy_position or player_position in state.traps:
        return -1.0

    # Reward heuristik stabil
    dist_goal = abs(player_position[0] - state.goal[0]) + abs(player_position[1] - state.goal[1])
    dist_enemy = abs(player_position[0] - enemy_position[0]) + abs(player_position[1] - enemy_position[1])

    score = (
        -0.05 * dist_goal +   # semakin dekat goal semakin bagus
        +0.01 * dist_enemy    # semakin jauh dari musuh semakin bagus
    )
    return score

  def backpropagation(self, node: MCTSNode, reward: float): 
    """
      Backpropagation phase: update node statistics up to root
    """
    while node is not None:
      node.visits += 1
      node.wins += reward
      node = node.parent

