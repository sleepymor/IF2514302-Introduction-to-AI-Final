"""Episode execution for benchmark.

Executes single episodes with specified algorithms and environments.
Isolated for clean multiprocessing support.
"""

from typing import Tuple

from benchmark_data import BenchmarkConfig


class EpisodeRunner:
    """Executes single episodes with specified algorithm and environment.

    Isolated class for clean multiprocessing support.
    Each episode:
    1. Creates environment with specified seed
    2. Initializes player and enemy agents
    3. Runs game loop until terminal or max moves
    4. Returns outcome and move count
    """

    @staticmethod
    def run_single_episode(
        config: BenchmarkConfig, algorithm: str, seed: int, episode_index: int = 0
    ) -> Tuple[str, int]:
        """Execute single episode with specified algorithm and seed.

        Args:
            config: BenchmarkConfig with game parameters
            algorithm: Algorithm choice ("MCTS", "ALPHABETA", "MINIMAX")
            seed: Random seed for reproducible environments
            episode_index: Episode number for tracking

        Returns:
            Tuple of (outcome, move_count) where:
            - outcome: "goal", "trap", "caught", or "timeout"
            - move_count: Number of moves taken
        """
        import sys
        from pathlib import Path
        import os

        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Suppress pygame welcome message
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

        from agents.enemy import EnemyAgent
        from agents.player import PlayerAgent
        from environment.environment import TacticalEnvironment

        # Create environment
        env = TacticalEnvironment(
            width=config.grid_width,
            height=config.grid_height,
            num_walls=config.num_walls,
            num_traps=config.num_traps,
            seed=seed,
        )
        player_agent = PlayerAgent(env, algorithm=algorithm, benchmark_mode=True)
        enemy_agent = EnemyAgent(env)

        # Run game loop
        move_count = 0
        while True:
            # Check move limit
            if move_count >= config.max_moves_per_episode:
                return "timeout", move_count

            # Get actions
            if env.turn == "player":
                action = EpisodeRunner._extract_action(player_agent.action())
            else:
                action = EpisodeRunner._extract_action(enemy_agent.action())

            # Step environment
            is_terminal, reason = env.step(action, simulate=True)
            move_count += 1

            if is_terminal:
                return reason, move_count

    @staticmethod
    def _extract_action(result):
        """Extract action from result (handles both tuple and plain formats).

        Args:
            result: Either plain action or (action, metadata) tuple

        Returns:
            Extracted action value
        """
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
        ):
            return result[0]
        return result

    @staticmethod
    def worker(
        args: Tuple[BenchmarkConfig, str, int, int],
    ) -> Tuple[str, int, int, int]:
        """Multiprocessing worker function.

        Args:
            args: Tuple of (config, algorithm, seed, episode_index)

        Returns:
            Tuple of (outcome, seed, episode_index, move_count)
        """
        config, algorithm, seed, episode_index = args
        outcome, move_count = EpisodeRunner.run_single_episode(
            config, algorithm, seed, episode_index
        )
        return outcome, seed, episode_index, move_count
