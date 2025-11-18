import random
from environment.environment import TacticalEnvironment
from algorithm.mcts.mcts import MCTS
from algorithm.alphabeta.alphabeta import AlphaBetaSearch
from utils.logger import Logger

class PlayerAgent:
    
    # PERBAIKANNYA ADA DI SINI:
    # Perhatikan bagaimana __init__ sekarang menerima 'algorithm="MCTS"'
    def __init__(self, env: TacticalEnvironment, algorithm="MCTS"):
        """
        Inisialisasi PlayerAgent.
        
        Args:
            env (TacticalEnvironment): Referensi ke environment game utama.
            algorithm (str): Algoritma yang akan digunakan ("MCTS" atau "ALPHABETA").
        """
        self.env = env
        self.algorithm_choice = algorithm
        self.log = Logger("PlayerAgent")

        # --- Parameter Algoritma (Bisa di-tuning) ---
        mcts_iterations = 50
        mcts_sim_depth = 40
        alphabeta_max_depth = 6 

        # --- Inisialisasi KEDUA Algoritma ---
        
        self.log.info("Initializing MCTS algorithm...")
        self.mcts_search = MCTS(
            iterations=mcts_iterations, 
            max_sim_depth=mcts_sim_depth
        )
        
        self.log.info("Initializing AlphaBetaSearch algorithm...")
        self.alphabeta_search = AlphaBetaSearch(
            max_depth=alphabeta_max_depth
        )

        # --- Konfirmasi Pilihan Algoritma ---
        if self.algorithm_choice == "MCTS":
            self.log.info(f"--- PlayerAgent akan menggunakan: MCTS (Iterasi: {mcts_iterations}) ---")
        elif self.algorithm_choice == "ALPHABETA":
            self.log.info(f"--- PlayerAgent akan menggunakan: AlphaBeta (Depth: {alphabeta_max_depth}) ---")
        else:
            self.log.warning(f"Pilihan algoritma '{self.algorithm_choice}' tidak dikenali. Menggunakan MCTS sebagai default.")
            self.algorithm_choice = "MCTS"


    def action(self) -> tuple:
        """
        Memutuskan aksi terbaik berdasarkan algoritma yang dipilih.
        Metode ini dipanggil oleh main.py di setiap giliran player.
        """
        
        current_state = self.env.clone()
        best_action = None

        # --- Logika Pemilihan Algoritma ---
        
        if self.algorithm_choice == "MCTS":
            self.log.info("MCTS is thinking...")
            best_action = self.mcts_search.search(current_state)
        
        elif self.algorithm_choice == "ALPHABETA":
            self.log.info("AlphaBeta is thinking...")
            best_action = self.alphabeta_search.search(current_state)

        # --- Fallback (Jaga-jaga) ---
        if best_action is None:
            self.log.error(f"Algorithm {self.algorithm_choice} gagal menemukan aksi. Mengambil langkah acak.")
            legal_actions = list(self.env.get_valid_actions(unit='current'))
            if legal_actions:
                best_action = random.choice(legal_actions)
            else:
                best_action = self.env.player_pos # Diam di tempat
        
        self.log.info(f"Chosen action: {best_action}")
        return best_action