import math
from environment.environment import TacticalEnvironment
# Pastikan import ini mengarah ke file node yang baru kita buat di atas
from algorithm.minimax.minimaxnode import heuristic_evaluate 
from utils.logger import Logger

log = Logger("Minimax")

class MinimaxSearch:
    
    def __init__(self, max_depth=4):
        """
        Inisialisasi Minimax.
        Note: Minimax lebih lambat dari AlphaBeta, jadi default depth
        mungkin perlu lebih kecil agar tidak nge-lag.
        """
        self.max_depth = max_depth
        log.info(f"MinimaxSearch initialized with max_depth={self.max_depth}")

    def search(self, initial_state: TacticalEnvironment):
        """
        Mencari aksi terbaik menggunakan Minimax standar.
        """
        best_action = None
        best_score = -math.inf
        
        legal_actions = list(initial_state.get_valid_actions(unit='current'))

        # Loop level root (Maximizing Player)
        for action in legal_actions:
            new_state = initial_state.clone()
            new_state.step(action, simulate=True)
            
            # Panggil rekursif _minimax (tanpa alpha/beta)
            # Giliran berikutnya adalah Musuh (False)
            score = self._minimax(new_state, self.max_depth - 1, False)
            
            log.info(f"Action {action} evaluated -> score: {score}")

            if score > best_score:
                best_score = score
                best_action = action
                
        log.info(f"Best action found: {best_action} with score: {best_score}")
        return best_action

    def _minimax(self, current_state: TacticalEnvironment, depth: int, is_maximizing_player: bool):
        """
        Fungsi rekursif Minimax tanpa pruning.
        """
        is_term, _ = current_state.is_terminal()
        
        # --- BASE CASE ---
        if depth == 0 or is_term:
            return heuristic_evaluate(current_state)

        # --- RECURSIVE CASE ---
        
        legal_actions = list(current_state.get_valid_actions(unit='current'))

        if is_maximizing_player:
            # Giliran PLAYER (Cari nilai Tertinggi)
            max_eval = -math.inf
            for action in legal_actions:
                new_state = current_state.clone()
                new_state.step(action, simulate=True)
                
                eval_val = self._minimax(new_state, depth - 1, False)
                max_eval = max(max_eval, eval_val)
            return max_eval
            
        else:
            # Giliran ENEMY (Cari nilai Terendah untuk Player)
            min_eval = math.inf
            for action in legal_actions:
                new_state = current_state.clone()
                new_state.step(action, simulate=True)
                
                eval_val = self._minimax(new_state, depth - 1, True)
                min_eval = min(min_eval, eval_val)
            return min_eval