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
        mcts = MCTS(iterations=300, exploration_constant=2000, max_sim_depth=65)
        best_move = mcts.search(state_copy)

        print("[DEBUG] Chosen move by MCTS:", best_move)
        print("[DEBUG] Real player_pos:", tuple(self.env.player_pos))
        print("[DEBUG] Real enemy_pos:", tuple(self.env.enemy_pos))
        print("[DEBUG] Is chosen move in real move_range?:", best_move in self.env.get_move_range(self.env.player_pos))


        return best_move     