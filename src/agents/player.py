import numpy as np
from algorithm.mcts.mcts import MCTS
from environment.environment import TacticalEnvironment
class PlayerAgent:
    """
      A simple Player Agent that selects a valid move within the allowed
      movement range. This agent chooses one of the available tiles at random.

      Attributes
      ----------
      env : TacticalEnvironment
            Reference to the game environment, used to query player position
            and valid movement tiles.
    """
    def __init__(self, env: TacticalEnvironment):
        self.env = env

    def action(self):
        state_copy = self.env.clone()

        # mcts search
        mcts = MCTS(iterations=1000, exploration_constant=2, max_sim_depth=100)
        best_move = mcts.search(state_copy)
        return best_move     