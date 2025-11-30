import math
from environment.environment import TacticalEnvironment
from utils.logger import Logger

# Import fungsi evaluate dari file sebelah (minimaxnode.py)
from algorithm.minimax.minimaxnode import heuristic_evaluate 

log = Logger("Minimax")

class MinimaxSearch:
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        log.info(f"MinimaxSearch initialized with max_depth={self.max_depth}")

    def search(self, initial_state: TacticalEnvironment):
        best_action = None
        best_score = -math.inf
        
        legal_actions = list(initial_state.get_valid_actions(unit='current'))

        # Loop level root
        for action in legal_actions:
            new_state = initial_state.clone()
            new_state.step(action, simulate=True)
            
            # Panggil rekursif
            score = self._minimax(new_state, self.max_depth - 1, False)
            
            if score > best_score:
                best_score = score
                best_action = action
                
        log.info(f"Best action found: {best_action} with score: {best_score}")
        return best_action

    def _minimax(self, current_state: TacticalEnvironment, depth: int, is_maximizing_player: bool):
        is_term, _ = current_state.is_terminal()
        
        # --- BASE CASE ---
        if depth == 0 or is_term:
            # Panggil fungsi dari minimaxnode.py
            return heuristic_evaluate(current_state, depth=depth)

        # --- RECURSIVE CASE ---
        legal_actions = list(current_state.get_valid_actions(unit='current'))

        if is_maximizing_player:
            max_eval = -math.inf
            for action in legal_actions:
                new_state = current_state.clone()
                new_state.step(action, simulate=True)
                eval_val = self._minimax(new_state, depth - 1, False)
                max_eval = max(max_eval, eval_val)
            return max_eval
            
        else:
            min_eval = math.inf
            for action in legal_actions:
                new_state = current_state.clone()
                new_state.step(action, simulate=True)
                eval_val = self._minimax(new_state, depth - 1, True)
                min_eval = min(min_eval, eval_val)
            return min_eval