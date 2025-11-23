
from algorithm.mcts.mcts import MCTS
from algorithm.alpha_minmax.alphabeta import AlphaBeta
from environment.environment import TacticalEnvironment
# class PlayerAgent:
#     """
#       A simple Player Agent that selects a valid move within the allowed
#       movement range. This agent chooses one of the available tiles at random.

#       Attributes
#       ----------
#       env : TacticalEnvironment
#             Reference to the game environment, used to query player position
#             and valid movement tiles.
#     """
#     def __init__(self, env: TacticalEnvironment):
#         self.env = env

#     def action(self):
#         state_copy = self.env.clone()

#         # mcts search
#         mcts = MCTS(iterations=300, exploration_constant=1.4, max_sim_depth=65)
#         best_move = mcts.search(state_copy)

#         print("Chosen move by MCTS:", best_move)
#         print("Real player_pos:", tuple(self.env.player_pos))
#         print("Real enemy_pos:", tuple(self.env.enemy_pos))
#         print("Is chosen move in real move_range?:", best_move in self.env.get_move_range(self.env.player_pos))


#         return best_move     

class PlayerAgent:
    
    # Player Agent menggunakan Alpha-Beta Pruning untuk memilih move optimal.

    def __init__(self, env: TacticalEnvironment):
        self.env = env
        self.algorithm = AlphaBeta(max_depth=4, use_improvements=True)

    def action(self):
        best_move = self.algorithm.search(self.env)

        valid_moves = self.env.get_move_range(self.env.player_pos)
        if best_move not in valid_moves:
            print("AlphaBeta returned invalid move, using fallback")
            best_move = tuple(self.env.player_pos)  
        
        print(f"AlphaBeta chose: {best_move}")
        print(f"Player position: {tuple(self.env.player_pos)}")
        print(f"Valid moves: {len(valid_moves)} tiles")
        
        return best_move