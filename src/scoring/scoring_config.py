"""Configuration for benchmark scoring system.

This module defines the ScoringConfig class which allows customization of:
- Number of test seeds and runs per seed
- Algorithm selection and parameters
- Environment configuration (grid size, obstacles, etc.)
- Game parameters (max turns, etc.)
"""

from typing import List, Optional


class ScoringConfig:
    """Configuration container for benchmark scoring."""

    def __init__(
        self,
        num_seeds: int = 100,
        tests_per_seed: int = 5,
        algorithms: Optional[List[str]] = None,
        grid_width: int = 15,
        grid_height: int = 10,
        num_walls: int = 10,
        num_traps: int = 5,
        max_turns: int = 500,
        mcts_iterations: int = 200,
        mcts_sim_depth: int = 5,
        alphabeta_depth: int = 5,
        minimax_depth: int = 4,
    ):
        """Initialize scoring configuration.

        Args:
            num_seeds: Number of different random seeds to test.
            tests_per_seed: Number of runs per seed
                (total tests = num_seeds * tests_per_seed).
            algorithms: List of algorithm names to test. Defaults to all three.
            grid_width: Grid width for environment.
            grid_height: Grid height for environment.
            num_walls: Number of walls in environment.
            num_traps: Number of traps in environment.
            max_turns: Maximum turns before force-stopping a game.
            mcts_iterations: Number of iterations for MCTS algorithm.
            mcts_sim_depth: Maximum simulation depth for MCTS.
            alphabeta_depth: Maximum depth for AlphaBeta search.
            minimax_depth: Maximum depth for Minimax search.
        """
        self.num_seeds = num_seeds
        self.tests_per_seed = tests_per_seed
        self.algorithms = algorithms or ["MCTS", "ALPHABETA", "MINIMAX"]

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_walls = num_walls
        self.num_traps = num_traps
        self.max_turns = max_turns

        self.mcts_iterations = mcts_iterations
        self.mcts_sim_depth = mcts_sim_depth
        self.alphabeta_depth = alphabeta_depth
        self.minimax_depth = minimax_depth

    @property
    def total_tests(self) -> int:
        """Calculate total number of tests."""
        return self.num_seeds * self.tests_per_seed

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of all configuration parameters.
        """
        return {
            "num_seeds": self.num_seeds,
            "tests_per_seed": self.tests_per_seed,
            "algorithms": self.algorithms,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "num_walls": self.num_walls,
            "num_traps": self.num_traps,
            "max_turns": self.max_turns,
            "mcts_iterations": self.mcts_iterations,
            "mcts_sim_depth": self.mcts_sim_depth,
            "alphabeta_depth": self.alphabeta_depth,
            "minimax_depth": self.minimax_depth,
            "total_tests": self.total_tests,
        }
