import random

from environment.environment import TacticalEnvironment
from algorithm.mcts.mcts import MCTS
from algorithm.alphabeta.alphabeta import AlphaBetaSearch
from algorithm.minimax.minimax import MinimaxSearch
from utils.logger import Logger


class PlayerAgent:
    """Player AI agent that supports multiple search algorithms."""

    def __init__(self, env: TacticalEnvironment, algorithm="MCTS", benchmark_mode=False):
        self.env = env
        self.algorithm_choice = algorithm
        self.benchmark_mode = benchmark_mode
        self.log = Logger("PlayerAgent")

        # --- Parameter Algoritma ---
        if benchmark_mode:
            # Reduced parameters for faster benchmarking
            mcts_iterations = 800
            mcts_sim_depth = 50
            alphabeta_max_depth = 4
            minimax_max_depth = 3
        else:
            # Full parameters for normal play
            mcts_iterations = 1500
            mcts_sim_depth = 80
            alphabeta_max_depth = 6
            minimax_max_depth = 4

        self.log.info(f"Initializing MCTS (iterations={mcts_iterations}, depth={mcts_sim_depth})...")
        self.mcts_search = MCTS(iterations=mcts_iterations, max_sim_depth=mcts_sim_depth)

        self.log.info(f"Initializing AlphaBetaSearch (depth={alphabeta_max_depth})...")
        self.alphabeta_search = AlphaBetaSearch(max_depth=alphabeta_max_depth)

        self.log.info(f"Initializing MinimaxSearch (depth={minimax_max_depth})...")
        self.minimax_search = MinimaxSearch(max_depth=minimax_max_depth)

        # --- Konfirmasi Pilihan Algoritma ---
        if self.algorithm_choice == "MCTS":
            self.log.info(f"--- PlayerAgent using: MCTS ---")
        elif self.algorithm_choice == "ALPHABETA":
            self.log.info(f"--- PlayerAgent using: AlphaBeta ---")
        elif self.algorithm_choice == "MINIMAX":
            self.log.info(f"--- PlayerAgent using: Minimax ---")
        else:
            self.log.warning(f"Unknown algorithm '{self.algorithm_choice}'. Defaulting to MCTS.")
            self.algorithm_choice = "MCTS"

    def action(self) -> tuple:
        """Execute player action based on selected algorithm."""
        current_state = self.env.clone()
        best_action = None

        # --- Logika Pemilihan Algoritma ---
        if self.algorithm_choice == "MCTS":
            self.log.info("MCTS is thinking...")
            best_action = self.mcts_search.search(current_state)

        elif self.algorithm_choice == "ALPHABETA":
            self.log.info("AlphaBeta is thinking...")
            best_action = self.alphabeta_search.search(current_state)

        elif self.algorithm_choice == "MINIMAX":
            self.log.info("Minimax is thinking...")
            best_action = self.minimax_search.search(current_state)

        # --- Fallback ---
        if best_action is None:
            legal_actions = list(self.env.get_valid_actions(unit='current'))
            if legal_actions:
                best_action = random.choice(legal_actions)
            else:
                best_action = self.env.player_pos

        self.log.info(f"Chosen action: {best_action}")
        return best_action