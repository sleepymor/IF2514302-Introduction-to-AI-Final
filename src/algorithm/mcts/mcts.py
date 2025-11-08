import numpy as np
import random
import math
from algorithm.mcts.mctsnode import MCTSNode
from utils.logger import Logger
log = Logger("MCTS")

class MCTS: 
  def __init__(self, iterations=50, exploration_constant=1.4, max_sim_depth=50):
    self.iterations = iterations
    self.exploration_constant = exploration_constant
    self.max_sim_depth = max_sim_depth
    

  def search(self, node): 
    root = MCTSNode(state=node)

    for i in range(self.iterations):
      # Selection: traverse tree using UCB
      selected = self.selection(root)

      # Expansion: add new child if not terminal
      if not selected.is_terminal():
          expanded = self.expansion(selected)
          # log.info(f"Expanded new child with action: {expanded.action}")
      else:
          expanded = selected
  
      # Simulation: rollout from new node
      result = self.simulation(expanded)
      # log.info(f"Simulation result: {result}")

      # Backpropagation: update statistics
      self.backpropagation(expanded, result)

    best_child = max(root.children, key=lambda c: c.visits)
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
    if not node.untried_actions:
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

    final_node = MCTSNode(state)
    return final_node.get_result()

    
  def backpropagation(self, node: MCTSNode, reward: float): 
    """
      Backpropagation phase: update node statistics up to root
    """
    while node is not None:
      node.visits += 1
      node.wins += reward
      node = node.parent

