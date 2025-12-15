"""Headless evaluation of AI agents without pygame visualization.

Runs episodic evaluations of player and enemy agents in TacticalEnvironment,
collecting outcome statistics for performance analysis.

Uses multiprocessing to run episodes in parallel for faster evaluation.

Architecture:
- Configuration parameters at module level for testability
- Episode runner isolated for multiprocessing
- Results aggregator handles statistics
"""

import logging
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Dict, Tuple

from tqdm import tqdm

from agents.enemy import EnemyAgent
from agents.player import PlayerAgent
from environment.environment import TacticalEnvironment

# Suppress logging during evaluation for clean progress bar display
logging.disable(logging.CRITICAL)


# =============================================================================
# CONFIGURATION - Testable parameters
# =============================================================================


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable configuration for headless evaluation."""

    grid_width: int
    grid_height: int
    num_walls: int
    num_traps: int
    environment_seed: int
    player_algorithm: str
    num_episodes: int
    progress_interval: int
    num_processes: int = 4  # Number of parallel processes


def create_evaluation_config() -> EvaluationConfig:
    """Create evaluation configuration from testable parameters.

    Returns:
        EvaluationConfig with all evaluation parameters.
    """
    # =========================================================================
    # TESTABLE PARAMETERS - Modify for different evaluation scenarios
    # =========================================================================
    GRID_WIDTH = 30  # Match benchmark.py for fair comparison
    GRID_HEIGHT = 15  # Match benchmark.py for fair comparison
    NUM_WALLS = 125  # Match benchmark.py for fair comparison
    NUM_TRAPS = 20  # Match benchmark.py for fair comparison
    ENVIRONMENT_SEED = (
        None  # None = random seed each episode, or set fixed seed (e.g., 32)
    )
    PLAYER_ALGORITHM = "MCTS"  # "MCTS", "ALPHABETA", or "MINIMAX"
    NUM_EPISODES = 15
    PROGRESS_INTERVAL = 5
    # Use all available CPU cores (or limit with os.cpu_count())
    NUM_PROCESSES = min(os.cpu_count() or 4, 8)
    # =========================================================================

    return EvaluationConfig(
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        num_walls=NUM_WALLS,
        num_traps=NUM_TRAPS,
        environment_seed=ENVIRONMENT_SEED,
        player_algorithm=PLAYER_ALGORITHM,
        num_episodes=NUM_EPISODES,
        progress_interval=PROGRESS_INTERVAL,
        num_processes=NUM_PROCESSES,
    )


# =============================================================================
# EPISODE RUNNER - Single game execution (for multiprocessing)
# =============================================================================


def _run_episode_worker(args: Tuple[EvaluationConfig, int]) -> str:
    """Worker function for multiprocessing pool.

    Args:
        args: Tuple of (config, episode_index)

    Returns:
        Terminal reason: "goal", "trap", or "caught".
    """
    config, episode_index = args
    return run_single_episode(config, episode_index)


def run_single_episode(config: EvaluationConfig, episode_index: int = 0) -> str:
    """Execute single episode and return terminal reason.

    Args:
        config: EvaluationConfig with game parameters.
        episode_index: Index of episode (used for seed generation if random mode).

    Returns:
        Terminal reason: "goal", "trap", or "caught".
    """
    # Determine seed: use fixed seed or generate random seed per episode
    seed = (
        config.environment_seed
        if config.environment_seed is not None
        else episode_index
    )

    # Create environment and agents
    env = TacticalEnvironment(
        width=config.grid_width,
        height=config.grid_height,
        num_walls=config.num_walls,
        num_traps=config.num_traps,
        seed=seed,
    )
    player_agent = PlayerAgent(
        env, algorithm=config.player_algorithm, benchmark_mode=True
    )
    enemy_agent = EnemyAgent(env)

    # Run game loop until terminal state
    while True:
        # Get action from current player
        if env.turn == "player":
            result = player_agent.action()
            # Handle both (action, metadata) tuple and plain action formats
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[1], dict)
            ):
                action = result[0]
            else:
                action = result
        else:
            result = enemy_agent.action()
            # Handle both (action, metadata) tuple and plain action formats
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[1], dict)
            ):
                action = result[0]
            else:
                action = result

        # Step environment and check for terminal condition
        is_terminal, reason = env.step(action, simulate=True)

        if is_terminal:
            return reason


# =============================================================================
# RESULTS AGGREGATION - Statistics collection
# =============================================================================


@dataclass
class EvaluationResults:
    """Container for evaluation statistics."""

    outcomes: Dict[str, int] = field(
        default_factory=lambda: {"goal": 0, "trap": 0, "caught": 0}
    )

    def record_outcome(self, reason: str) -> None:
        """Record episode outcome.

        Args:
            reason: Terminal reason ("goal", "trap", or "caught").
        """
        self.outcomes[reason] += 1

    def get_win_rate(self, total_episodes: int) -> float:
        """Calculate win rate percentage.

        Args:
            total_episodes: Total number of episodes evaluated.

        Returns:
            Win rate as decimal (0.0 to 1.0).
        """
        return self.outcomes["goal"] / total_episodes if total_episodes > 0 else 0.0


def display_results_summary(
    config: EvaluationConfig, results: EvaluationResults
) -> None:
    """Display evaluation results in formatted summary.

    Args:
        config: EvaluationConfig used for evaluation.
        results: EvaluationResults with collected statistics.
    """
    total = config.num_episodes
    win_rate = results.get_win_rate(total)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Episodes: {total}")
    print(f"Algorithm: {config.player_algorithm}")
    print(f"Environment Seed: {config.environment_seed}")
    print()
    print(
        f"Goal Reached:  {results.outcomes['goal']:>3}  ({results.outcomes['goal']/total:>5.1%})"
    )
    print(
        f"Hit Trap:      {results.outcomes['trap']:>3}  ({results.outcomes['trap']/total:>5.1%})"
    )
    print(
        f"Caught:        {results.outcomes['caught']:>3}  ({results.outcomes['caught']/total:>5.1%})"
    )
    print()
    print(f"Win Rate:      {win_rate:>5.1%}")
    print("=" * 50 + "\n")


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================


def run_evaluation(config: EvaluationConfig) -> EvaluationResults:
    """Run complete evaluation suite using parallel processing.

    Args:
        config: EvaluationConfig with evaluation parameters.

    Returns:
        EvaluationResults with collected statistics.
    """
    results = EvaluationResults()

    seed_mode = (
        "random"
        if config.environment_seed is None
        else f"fixed (seed={config.environment_seed})"
    )
    print(
        f"Starting evaluation: {config.num_episodes} episodes of {config.player_algorithm}"
    )
    print(f"Seed mode: {seed_mode}")
    print(f"Using {config.num_processes} parallel processes...\n")

    # Create list of work items with episode indices
    work_items = [(config, i) for i in range(config.num_episodes)]

    # Run all episodes in parallel using multiprocessing pool
    with Pool(processes=config.num_processes) as pool:
        outcomes = tqdm(
            pool.imap_unordered(_run_episode_worker, work_items),
            total=config.num_episodes,
            desc="Evaluation Progress",
            unit="episode",
        )

        # Collect results
        for outcome in outcomes:
            results.record_outcome(outcome)

    return results


def main() -> None:
    """Execute headless evaluation."""
    # Create configuration
    config = create_evaluation_config()

    # Run evaluation
    results = run_evaluation(config)

    # Display results
    display_results_summary(config, results)


if __name__ == "__main__":
    main()
