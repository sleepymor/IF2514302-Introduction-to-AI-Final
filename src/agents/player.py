import random
import time

from environment.environment import TacticalEnvironment
from algorithm.mcts.mcts import MCTS
from algorithm.alphabeta.alphabeta import AlphaBetaSearch
from algorithm.minimax.minimax import MinimaxSearch
from utils.logger import Logger


class PlayerAgent:
    """Player AI agent that supports multiple search algorithms."""

    def __init__(self, env: TacticalEnvironment, algorithm="MCTS", benchmark_mode=False):
        self.env = env
        self.algorithm_choice = (algorithm or "MCTS").upper()
        self.benchmark_mode = benchmark_mode
        self.log = Logger("PlayerAgent")

        # --- Parameter Algoritma ---
        if benchmark_mode:
            # Reduced parameters for faster benchmarking
            self.mcts_iterations = 200
            self.mcts_sim_depth = 6
            self.alphabeta_max_depth = 4
            self.minimax_max_depth = 3
        else:
            # Full parameters for normal play
            self.mcts_iterations = 200
            self.mcts_sim_depth = 6
            self.alphabeta_max_depth = 6
            self.minimax_max_depth = 4

        # Do not instantiate all searches up-front; create only the selected one.
        self.mcts_search = None
        self.alphabeta_search = None
        self.minimax_search = None

        if self.algorithm_choice == "MCTS":
            self.log.info(f"Initializing MCTS (iterations={self.mcts_iterations}, depth={self.mcts_sim_depth})...")
            self.mcts_search = MCTS(iterations=self.mcts_iterations, max_sim_depth=self.mcts_sim_depth)
            self.log.info(f"--- PlayerAgent using: MCTS ---")

        elif self.algorithm_choice == "ALPHABETA" or self.algorithm_choice == "ALPHA-BETA":
            self.log.info(f"Initializing AlphaBetaSearch (depth={self.alphabeta_max_depth})...")
            self.alphabeta_search = AlphaBetaSearch(max_depth=self.alphabeta_max_depth)
            self.log.info(f"--- PlayerAgent using: AlphaBeta ---")

        elif self.algorithm_choice == "MINIMAX":
            self.log.info(f"Initializing MinimaxSearch (depth={self.minimax_max_depth})...")
            self.minimax_search = MinimaxSearch(max_depth=self.minimax_max_depth)
            self.log.info(f"--- PlayerAgent using: Minimax ---")

        else:
            self.log.warning(f"Unknown algorithm '{self.algorithm_choice}'. Defaulting to MCTS.")
            self.algorithm_choice = "MCTS"
            self.log.info(f"Initializing MCTS (iterations={self.mcts_iterations}, depth={self.mcts_sim_depth})...")
            self.mcts_search = MCTS(iterations=self.mcts_iterations, max_sim_depth=self.mcts_sim_depth)

    def action(self) -> tuple:
        """Execute player action based on selected algorithm."""
        current_state = self.env.clone()
        best_action = None
        metadata = {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0}

        # Lazily initialize if something changed externally
        start_time = time.time()
        if self.algorithm_choice == "MCTS":
            if self.mcts_search is None:
                self.log.info(f"Lazy-initializing MCTS (iterations={self.mcts_iterations}, depth={self.mcts_sim_depth})...")
                self.mcts_search = MCTS(iterations=self.mcts_iterations, max_sim_depth=self.mcts_sim_depth)
            self.log.info("MCTS is thinking...")
            result = self.mcts_search.search(current_state)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                best_action, meta = result
                metadata.update(meta)
            else:
                best_action = result

        elif self.algorithm_choice == "ALPHABETA":
            if self.alphabeta_search is None:
                self.log.info(f"Lazy-initializing AlphaBetaSearch (depth={self.alphabeta_max_depth})...")
                self.alphabeta_search = AlphaBetaSearch(max_depth=self.alphabeta_max_depth)
            self.log.info("AlphaBeta is thinking...")
            result = self.alphabeta_search.search(current_state)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                best_action, meta = result
                metadata.update(meta)
            else:
                best_action = result

        elif self.algorithm_choice == "MINIMAX":
            if self.minimax_search is None:
                self.log.info(f"Lazy-initializing MinimaxSearch (depth={self.minimax_max_depth})...")
                self.minimax_search = MinimaxSearch(max_depth=self.minimax_max_depth)
            self.log.info("Minimax is thinking...")
            result = self.minimax_search.search(current_state)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                best_action, meta = result
                metadata.update(meta)
            else:
                best_action = result

        # --- Fallback ---
        if best_action is None:
            legal_actions = list(self.env.get_valid_actions(unit='current'))
            if legal_actions:
                best_action = random.choice(legal_actions)
            else:
                best_action = self.env.player_pos
                
        # Benchmark mode: return immediately without extra metadata        
        if self.benchmark_mode:
            return best_action
        
        # finalize thinking time
        end_time = time.time()
        metadata["thinking_time"] = end_time - start_time

        self.log.info(f"Chosen action: {best_action} (meta={metadata})")
        return best_action, metadata