import random
from environment.environment import TacticalEnvironment
from algorithm.mcts.mcts import MCTS
from algorithm.alphabeta.alphabeta import AlphaBetaSearch
# --- TAMBAHAN 1: Import Minimax ---
from algorithm.minimax.minimax import MinimaxSearch 
from utils.logger import Logger

class PlayerAgent:
    
    def __init__(self, env: TacticalEnvironment, algorithm="MCTS"):
        self.env = env
        self.algorithm_choice = algorithm
        self.log = Logger("PlayerAgent")

        # --- Parameter Algoritma ---
        mcts_iterations = 50
        mcts_sim_depth = 40
        alphabeta_max_depth = 6 
        minimax_max_depth = 4 # Minimax biasanya lebih berat, depth dikurangi sedikit

        self.log.info("Initializing MCTS algorithm...")
        self.mcts_search = MCTS(iterations=mcts_iterations, max_sim_depth=mcts_sim_depth)
        
        self.log.info("Initializing AlphaBetaSearch algorithm...")
        self.alphabeta_search = AlphaBetaSearch(max_depth=alphabeta_max_depth)

        # --- TAMBAHAN 2: Inisialisasi Minimax ---
        self.log.info("Initializing MinimaxSearch algorithm...")
        self.minimax_search = MinimaxSearch(max_depth=minimax_max_depth)

        # --- Konfirmasi Pilihan Algoritma ---
        if self.algorithm_choice == "MCTS":
            self.log.info(f"--- PlayerAgent using: MCTS ---")
        elif self.algorithm_choice == "ALPHABETA":
            self.log.info(f"--- PlayerAgent using: AlphaBeta ---")
        elif self.algorithm_choice == "MINIMAX": # <-- Tambahan Log
            self.log.info(f"--- PlayerAgent using: Minimax ---")
        else:
            self.log.warning(f"Unknown algorithm '{self.algorithm_choice}'. Defaulting to MCTS.")
            self.algorithm_choice = "MCTS"


    def action(self) -> tuple:
        current_state = self.env.clone()
        best_action = None

        # --- Logika Pemilihan Algoritma ---
        if self.algorithm_choice == "MCTS":
            self.log.info("MCTS is thinking...")
            best_action = self.mcts_search.search(current_state)
        
        elif self.algorithm_choice == "ALPHABETA":
            self.log.info("AlphaBeta is thinking...")
            best_action = self.alphabeta_search.search(current_state)

        # --- TAMBAHAN 3: Panggil Minimax ---
        elif self.algorithm_choice == "MINIMAX":
            self.log.info("Minimax is thinking...")
            best_action = self.minimax_search.search(current_state)

        # --- Fallback ---
        if best_action is None:
            # ... (kode fallback sama seperti sebelumnya)
            legal_actions = list(self.env.get_valid_actions(unit='current'))
            if legal_actions:
                best_action = random.choice(legal_actions)
            else:
                best_action = self.env.player_pos
        
        self.log.info(f"Chosen action: {best_action}")
        return best_action