"""Player AI agent with support for multiple search algorithms.

Supports MCTS, AlphaBeta, and Minimax algorithms with configurable parameters.
Can operate in benchmark mode (minimal overhead) or interactive mode (with metadata).
"""

import random
import time

from algorithm.astar.astar import AStar
from algorithm.alphabeta.alphabeta import AlphaBetaSearch
from algorithm.mcts.mcts import MCTS
from algorithm.minimax.minimax import MinimaxSearch
from environment.environment import TacticalEnvironment
from utils.logger import Logger


class PlayerAgent:
    """Player AI agent that supports multiple search algorithms.

    Attributes:
        env: TacticalEnvironment instance
        algorithm_choice: Selected algorithm ("MCTS", "ALPHABETA", "MINIMAX")
        benchmark_mode: If True, returns only action (no metadata)
        mcts_iterations: MCTS search iterations
        mcts_sim_depth: MCTS simulation depth
        alphabeta_max_depth: AlphaBeta search depth
        minimax_max_depth: Minimax search depth
    """

    # Default parameters for benchmark mode
    DEFAULT_MCTS_ITERATIONS = 200
    DEFAULT_MCTS_SIM_DEPTH = 5
    DEFAULT_ALPHABETA_DEPTH = 4
    DEFAULT_MINIMAX_DEPTH = 3

    # Default parameters for interactive mode
    INTERACTIVE_MCTS_ITERATIONS = 200
    INTERACTIVE_MCTS_SIM_DEPTH = 50
    INTERACTIVE_ALPHABETA_DEPTH = 5
    INTERACTIVE_MINIMAX_DEPTH = 4

    def __init__(
        self,
        env: TacticalEnvironment,
        algorithm: str = "MCTS",
        benchmark_mode: bool = False,
        mcts_iterations: int = None,
        mcts_sim_depth: int = None,
        alphabeta_depth: int = None,
        minimax_depth: int = None,
    ):
        """Initialize player agent with selected algorithm.

        Args:
            env: TacticalEnvironment instance
            algorithm: Algorithm choice ("MCTS", "ALPHABETA", "MINIMAX")
            benchmark_mode: If True, optimize for speed and minimal output
            mcts_iterations: MCTS iterations (overrides default)
            mcts_sim_depth: MCTS simulation depth (overrides default)
            alphabeta_depth: AlphaBeta depth (overrides default)
            minimax_depth: Minimax depth (overrides default)
        """
        self.env = env
        self.algorithm_choice = (algorithm or "MCTS").upper()
        self.benchmark_mode = benchmark_mode
        self.log = Logger("PlayerAgent")

        # Initialize algorithm parameters
        self._init_algorithm_parameters(
            benchmark_mode,
            mcts_iterations,
            mcts_sim_depth,
            alphabeta_depth,
            minimax_depth,
        )

        if mcts_sim_depth is not None:
            self.mcts_sim_depth = mcts_sim_depth
        elif benchmark_mode:
            self.mcts_sim_depth = 5
        else:
            self.mcts_sim_depth = 5

        if alphabeta_depth is not None:
            self.alphabeta_max_depth = alphabeta_depth
        elif benchmark_mode:
            self.alphabeta_max_depth = 4
        else:
            self.alphabeta_max_depth = 5

        if minimax_depth is not None:
            self.minimax_max_depth = minimax_depth
        elif benchmark_mode:
            self.minimax_max_depth = 3
        else:
            self.minimax_max_depth = 4

        # Do not instantiate all searches up-front; create only the selected one.
        self.mcts_search = None
        self.alphabeta_search = None
        self.minimax_search = None

        # Create selected algorithm
        self._init_selected_algorithm()

    def _init_algorithm_parameters(
        self,
        benchmark_mode: bool,
        mcts_iterations: int,
        mcts_sim_depth: int,
        alphabeta_depth: int,
        minimax_depth: int,
    ) -> None:
        """Initialize algorithm parameters based on mode and overrides.

        Args:
            benchmark_mode: Benchmark mode flag
            mcts_iterations: MCTS iterations override
            mcts_sim_depth: MCTS depth override
            alphabeta_depth: AlphaBeta depth override
            minimax_depth: Minimax depth override
        """
        # Set MCTS parameters
        if mcts_iterations is not None:
            self.mcts_iterations = mcts_iterations
        elif benchmark_mode:
            self.mcts_iterations = self.DEFAULT_MCTS_ITERATIONS
        else:
            self.mcts_iterations = self.INTERACTIVE_MCTS_ITERATIONS

        if mcts_sim_depth is not None:
            self.mcts_sim_depth = mcts_sim_depth
        elif benchmark_mode:
            self.mcts_sim_depth = self.DEFAULT_MCTS_SIM_DEPTH
        else:
            self.mcts_sim_depth = self.INTERACTIVE_MCTS_SIM_DEPTH

        # Set AlphaBeta parameters
        if alphabeta_depth is not None:
            self.alphabeta_max_depth = alphabeta_depth
        elif benchmark_mode:
            self.alphabeta_max_depth = self.DEFAULT_ALPHABETA_DEPTH
        else:
            self.alphabeta_max_depth = self.INTERACTIVE_ALPHABETA_DEPTH

        # Set Minimax parameters
        if minimax_depth is not None:
            self.minimax_max_depth = minimax_depth
        elif benchmark_mode:
            self.minimax_max_depth = self.DEFAULT_MINIMAX_DEPTH
        else:
            self.minimax_max_depth = self.INTERACTIVE_MINIMAX_DEPTH

    def _init_selected_algorithm(self) -> None:
        """Initialize the selected search algorithm."""
        if self.algorithm_choice == "MCTS":
            self.log.info(
                f"Initializing MCTS (iterations={self.mcts_iterations}, "
                f"depth={self.mcts_sim_depth})..."
            )
            self.mcts_search = MCTS(
                iterations=self.mcts_iterations,
                max_sim_depth=self.mcts_sim_depth,
            )
            self.log.info("--- PlayerAgent using: MCTS ---")

        elif self.algorithm_choice in ["ALPHABETA", "ALPHA-BETA"]:
            self.log.info(
                f"Initializing AlphaBetaSearch (depth={self.alphabeta_max_depth})..."
            )
            self.alphabeta_search = AlphaBetaSearch(max_depth=self.alphabeta_max_depth)
            self.log.info("--- PlayerAgent using: AlphaBeta ---")

        elif self.algorithm_choice == "MINIMAX":
            self.log.info(
                f"Initializing MinimaxSearch (depth={self.minimax_max_depth})..."
            )
            self.minimax_search = MinimaxSearch(max_depth=self.minimax_max_depth)
            self.log.info("--- PlayerAgent using: Minimax ---")

        else:
            self.log.warning(
                f"Unknown algorithm '{self.algorithm_choice}'. Defaulting to MCTS."
            )
            self.algorithm_choice = "MCTS"
            self.mcts_search = MCTS(
                iterations=self.mcts_iterations,
                max_sim_depth=self.mcts_sim_depth,
            )

    def action(self) -> tuple:
        """Execute player action based on selected algorithm.

        Returns:
            In benchmark mode: action only
            In interactive mode: (action, metadata) tuple where metadata includes:
                - nodes_visited: Number of nodes explored
                - thinking_time: Time spent deciding (seconds)
                - win_probability: Estimated win probability (if available)
        """
        current_state = self.env.clone()
        best_action = None
        metadata = {
            "nodes_visited": 0,
            "thinking_time": 0.0,
            "win_probability": 0.0,
        }

        start_time = time.time()

        # Execute selected algorithm
        if self.algorithm_choice == "MCTS":
            best_action, metadata = self._execute_mcts(current_state, metadata)
        elif self.algorithm_choice == "ALPHABETA":
            best_action, metadata = self._execute_alphabeta(current_state, metadata)
        elif self.algorithm_choice == "MINIMAX":
            best_action, metadata = self._execute_minimax(current_state, metadata)

        # Fallback to random action if no valid action found
        if best_action is None:
            best_action = self._get_fallback_action()

        # Record thinking time
        end_time = time.time()
        metadata["thinking_time"] = end_time - start_time

        # Return based on mode
        if self.benchmark_mode:
            return best_action

        self.log.info(f"Chosen action: {best_action} (meta={metadata})")
        return best_action, metadata

    def _execute_mcts(self, state, metadata: dict) -> tuple:
        """Execute MCTS algorithm.

        Args:
            state: Current game state
            metadata: Metadata dictionary to update

        Returns:
            (action, metadata) tuple
        """
        if self.mcts_search is None:
            self.log.info(
                f"Lazy-initializing MCTS (iterations={self.mcts_iterations}, "
                f"depth={self.mcts_sim_depth})..."
            )
            self.mcts_search = MCTS(
                iterations=self.mcts_iterations,
                max_sim_depth=self.mcts_sim_depth,
            )

        self.log.info("MCTS is thinking...")
        result = self.mcts_search.search(state)

        action = self._extract_action_and_metadata(result, metadata)
        return action, metadata

    def _execute_alphabeta(self, state, metadata: dict) -> tuple:
        """Execute AlphaBeta algorithm.

        Args:
            state: Current game state
            metadata: Metadata dictionary to update

        Returns:
            (action, metadata) tuple
        """
        if self.alphabeta_search is None:
            self.log.info(
                f"Lazy-initializing AlphaBetaSearch "
                f"(depth={self.alphabeta_max_depth})..."
            )
            self.alphabeta_search = AlphaBetaSearch(max_depth=self.alphabeta_max_depth)

        self.log.info("AlphaBeta is thinking...")
        result = self.alphabeta_search.search(state)

        action = self._extract_action_and_metadata(result, metadata)
        return action, metadata

    def _execute_minimax(self, state, metadata: dict) -> tuple:
        """Execute Minimax algorithm.

        Args:
            state: Current game state
            metadata: Metadata dictionary to update

        Returns:
            (action, metadata) tuple
        """
        if self.minimax_search is None:
            self.log.info(
                f"Lazy-initializing MinimaxSearch (depth={self.minimax_max_depth})..."
            )
            self.minimax_search = MinimaxSearch(max_depth=self.minimax_max_depth)

        self.log.info("Minimax is thinking...")
        result = self.minimax_search.search(state)

        action = self._extract_action_and_metadata(result, metadata)
        return action, metadata

    @staticmethod
    def _extract_action_and_metadata(result, metadata: dict):
        """Extract action from algorithm result (handles both formats).

        Args:
            result: Either plain action or (action, metadata) tuple
            metadata: Metadata dictionary to update

        Returns:
            Extracted action
        """
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
        ):
            action, meta = result
            metadata.update(meta)
            return action
        return result

    def _get_fallback_action(self):
        """Get fallback action when no valid action from algorithm.

        Returns:
            Valid action or player position as fallback
        """
        legal_actions = list(self.env.get_valid_actions(unit="current"))
        if legal_actions:
            return random.choice(legal_actions)
        return self.env.player_pos

    def peek_path_to_goal(self) -> list:
        """Calculate path from player to goal using A* (for visualization).

        Returns:
            List of (x, y) tuples representing path, or empty list if no path exists
        """
        a_star = AStar(env=self.env)
        start = tuple(self.env.player_pos)
        goal = tuple(self.env.goal)

        path = a_star.search(start, goal)
        return [] if path is None else list(path)
