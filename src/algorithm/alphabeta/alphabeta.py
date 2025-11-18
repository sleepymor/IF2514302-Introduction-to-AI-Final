import math
from environment.environment import TacticalEnvironment
from algorithm.alphabeta.alphabetanode import heuristic_evaluate # <- Impor heuristik
from utils.logger import Logger

log = Logger("AlphaBeta")

class AlphaBetaSearch:
    
    def __init__(self, max_depth=6):
        """
        Inisialisasi pencarian Alpha-Beta.
        
        Args:
            max_depth (int): Seberapa "jauh" AI akan melihat ke masa depan.
                             Nilai yang lebih tinggi lebih pintar, tetapi lebih lambat.
                             Nilai 6-8 biasanya awal yang baik.
        """
        self.max_depth = max_depth
        log.info(f"AlphaBetaSearch initialized with max_depth={self.max_depth}")

    def search(self, initial_state: TacticalEnvironment):
        """
        Mencari aksi terbaik dari state saat ini.
        Ini adalah fungsi utama yang akan Anda panggil dari PlayerAgent.

        Args:
            initial_state (TacticalEnvironment): Kondisi environment saat ini.

        Returns:
            tuple: Aksi (x, y) terbaik untuk diambil.
        """
        
        # Kita (Player) adalah 'Maximizing Player', kita ingin skor tertinggi.
        best_action = None
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        legal_actions = list(initial_state.get_valid_actions(unit='current'))

        # Acak urutan aksi untuk hasil yang lebih bervariasi jika skornya sama
        # random.shuffle(legal_actions) 
        
        # Loop manual di level pertama (root) untuk bisa melacak AKSI terbaik
        for action in legal_actions:
            # 1. Buat state baru hasil dari aksi
            new_state = initial_state.clone()
            new_state.step(action, simulate=True)
            
            # 2. Panggil _alphabeta untuk state baru tersebut.
            #    Giliran berikutnya adalah 'Minimizing Player' (Musuh).
            score = self._alphabeta(new_state, self.max_depth - 1, alpha, beta, False)
            
            log.info(f"Action {action} evaluated with score: {score}")

            # 3. Update skor dan aksi terbaik
            if score > best_score:
                best_score = score
                best_action = action
            
            # Update alpha di root
            alpha = max(alpha, best_score)
                
        log.info(f"Best action found: {best_action} with score: {best_score}")
        return best_action

    def _alphabeta(self, current_state: TacticalEnvironment, depth: int, 
                   alpha: float, beta: float, is_maximizing_player: bool):
        """
        Fungsi rekursif inti dari Alpha-Beta Pruning.
        """
        
        is_term, _ = current_state.is_terminal()
        
        # --- BASE CASE ---
        # Jika kita sudah mencapai kedalaman maksimum atau permainan berakhir
        if depth == 0 or is_term:
            # Kembalikan evaluasi statis dari state ini
            return heuristic_evaluate(current_state)

        # --- RECURSIVE CASE ---

        # 2. Maximizing Player (Giliran PLAYER)
        if is_maximizing_player:
            value = -math.inf
            legal_actions = list(current_state.get_valid_actions(unit='current'))
            
            for action in legal_actions:
                new_state = current_state.clone()
                new_state.step(action, simulate=True)
                
                # Panggil rekursif untuk giliran Minimizing Player
                value = max(value, self._alphabeta(new_state, depth - 1, alpha, beta, False))
                
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # *** Beta Cutoff (Pruning) ***
            return value

        # 3. Minimizing Player (Giliran ENEMY)
        else: # (is_minimizing_player)
            value = math.inf
            legal_actions = list(current_state.get_valid_actions(unit='current'))

            for action in legal_actions:
                new_state = current_state.clone()
                new_state.step(action, simulate=True)
                
                # Panggil rekursif untuk giliran Maximizing Player
                value = min(value, self._alphabeta(new_state, depth - 1, alpha, beta, True))
                
                beta = min(beta, value)
                if alpha >= beta:
                    break  # *** Alpha Cutoff (Pruning) ***
            return value