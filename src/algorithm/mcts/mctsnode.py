import math
import random
from algorithm.astar.astar import AStar
from environment.environment import TacticalEnvironment

class MCTSNode: 
  """
    Node in the Monte Carlo Tree Search tree.
    Represents a game state and stores search statistics.
  """
  def __init__(self, state: TacticalEnvironment, parent=None, action=None):
    """
      Initialize a new MCTS node.
      
      Args:
          state: Current game state (TacticalEnvironment)
          parent: Parent node in search tree
          action: Action taken to reach this state from parent
    """
    self.state = state 
    self.parent = parent
    self.action = action
    self.children = []
    self.visits = 0 
    self.wins = 0 

    # Ambil daftar aksi yang legal
    actions_list = list(self.get_legal_action())
    
    # ACAK URUTANNYA SEBELUM DISIMPAN
    random.shuffle(actions_list) 
    
    self.untried_actions = actions_list

  def _get_legal_actions(self):
    """
      Get all legal actions for the current player.
      Delegates to environment's get_valid_actions method.
      
      Returns:
          list: Legal move positions for current turn
    """
    return self.state.get_valid_actions(unit='current')

  def add_child(self, child_node):
    self.children.append(child_node)

    if child_node.action in self.untried_actions:
      self.untried_actions.remove(child_node.action)
  
  def ucb_score(self, c=1.4): 
    """
      UCB1 formula for balancing exploration and exploitation
    """ 

    if self.visits == 0:
      return float("inf")
    
    if self.parent is None:
      return float("-inf")
    
    
    exploitation = self.wins / self.visits
    exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)

    # print(f"get ucb: {exploitation}, {exploration}")
    return exploitation + exploration
  
  def best_child(self, c=1.4):
    """
      Select child with highest UCB score
    """
    return max(self.children, key=lambda child: child.ucb_score(c))
  
  def is_terminal(self): 
    """
      Check if state is terminal (win/lose)
    """
    player_position = tuple(self.state.player_pos)
    enemy_position = tuple(self.state.enemy_pos)

    # Win condition
    if player_position == self.state.goal:
      return True

    # Lose condition
    if player_position == enemy_position:
      return True
    if player_position in self.state.traps:
      return True
    
    return False
  
  def get_result(self):
    """
      Return reward from current player's perspective
      +1 for win, -1 for loss, 0 for ongoing
    """
    player_position = tuple(self.state.player_pos)
    enemy_position = tuple(self.state.enemy_pos)

    # Win condition - reached goal
    if player_position == self.state.goal:
        return 1.0
    
    # Lose conditions
    if player_position == enemy_position:
        return -1.0
    if player_position in self.state.traps:
        return -1.0
    
    # Heuristic for non-terminal states (distance to goal)
    goal_dist = abs(player_position[0] - self.state.goal[0]) + abs(player_position[1] - self.state.goal[1])
    enemy_dist = abs(player_position[0] - enemy_position[0]) + abs(player_position[1] - enemy_position[1])
    
    # Small bonus/penalty based on distance
    return 0.1 * (1.0 / (goal_dist + 1) - 0.5 / (enemy_dist + 1))

  def get_legal_action(self): 
    """
      Get all legal moves from current state
    """
    if self.state.turn == 'player': 
      moves = self.state.get_move_range(self.state.player_pos)
      return list(moves)
    elif self.state.turn == 'enemy':
      a_star = AStar(env=self.state)

      start = tuple(self.state.enemy_pos)
      goal = tuple(self.state.player_pos)

      path = a_star.search(start, goal)

      if path is None or len(path) <= 1:
          return start
      
      path = path[1:]

      index = min(2, len(path) - 1)

      next_tile = path[index]

      # If enemy moves more than one tile per turn, select accordingly (indexing)
      # but env.enemy has move_range 3 in your EnemyAgent, adapt if necessary
      return next_tile
    
    return []

  def is_fully_expanded(self):
    """
      Check if all possible actions have been tried
    """ 
    return len(self.untried_actions) == 0

  @property
  def depth(self):
    """
      Calculate depth of node in tree
    """
    depth = 0
    node = self
    while node is not None:
      depth += 1
      node = node.parent
    return depth