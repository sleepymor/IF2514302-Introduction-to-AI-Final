"""Benchmark runner for AI algorithm evaluation.

Evaluates multiple AI algorithms on the same environment and seed,
collecting performance metrics and exporting results to CSV.

Architecture:
- BenchmarkConfig: Immutable configuration for reproducibility
- Episode runner isolated from evaluation logic
- Results aggregator handles statistics for each algorithm
- CSV export for easy analysis
"""

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logging.disable(logging.CRITICAL)

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.enemy import EnemyAgent
from agents.player import PlayerAgent
from environment.environment import TacticalEnvironment
from tqdm import tqdm

# =============================================================================
# CONFIGURATION - Testable parameters
# =============================================================================


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration for benchmark evaluation."""

    grid_width: int
    grid_height: int
    num_walls: int
    num_traps: int
    environment_seed: int
    algorithms: List[str]
    num_episodes: int


def create_benchmark_config() -> BenchmarkConfig:
    """Create benchmark configuration from testable parameters.

    Returns:
        BenchmarkConfig with all benchmark parameters.
    """
    # =========================================================================
    # TESTABLE PARAMETERS - Modify for different evaluation scenarios
    # =========================================================================
    GRID_WIDTH = 30  # Match run_headless.py for fair comparison
    GRID_HEIGHT = 15  # Match run_headless.py for fair comparison
    NUM_WALLS = 125  # Match run_headless.py for fair comparison
    NUM_TRAPS = 20  # Match run_headless.py for fair comparison
    ENVIRONMENT_SEED = 32
    ALGORITHMS = ["MCTS", "ALPHABETA", "MINIMAX"]  # All algorithms to evaluate
    NUM_EPISODES = 10
    # =========================================================================

    return BenchmarkConfig(
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        num_walls=NUM_WALLS,
        num_traps=NUM_TRAPS,
        environment_seed=ENVIRONMENT_SEED,
        algorithms=ALGORITHMS,
        num_episodes=NUM_EPISODES,
    )


# =============================================================================
# EPISODE RUNNER - Single game execution
# =============================================================================


def run_single_episode(config: BenchmarkConfig, algorithm: str) -> str:
    """Execute single episode with specified algorithm.

    Args:
        config: BenchmarkConfig with game parameters.
        algorithm: Algorithm to use ("MCTS", "ALPHABETA", or "MINIMAX").

    Returns:
        Terminal reason: "goal", "trap", or "caught".
    """
    # Create environment and agents
    env = TacticalEnvironment(
        width=config.grid_width,
        height=config.grid_height,
        num_walls=config.num_walls,
        num_traps=config.num_traps,
        seed=config.environment_seed,
    )
    player_agent = PlayerAgent(env, algorithm=algorithm, benchmark_mode=True)
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
class BenchmarkResults:
    """Container for benchmark statistics."""

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
    config: BenchmarkConfig, results: Dict[str, BenchmarkResults]
) -> None:
    """Display benchmark results in formatted summary.

    Args:
        config: BenchmarkConfig used for evaluation.
        results: Dictionary mapping algorithm name to BenchmarkResults.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Episodes per Algorithm: {config.num_episodes}")
    print(f"Environment Seed: {config.environment_seed}")
    print(f"Grid Size: {config.grid_width}x{config.grid_height}")
    print()

    for algo in config.algorithms:
        algo_results = results[algo]
        total = config.num_episodes
        win_rate = algo_results.get_win_rate(total)

        print(f"{algo}:")
        print(
            f"  Goal Reached:  {algo_results.outcomes['goal']:>3}  ({algo_results.outcomes['goal']/total:>5.1%})"
        )
        print(
            f"  Hit Trap:      {algo_results.outcomes['trap']:>3}  ({algo_results.outcomes['trap']/total:>5.1%})"
        )
        print(
            f"  Caught:        {algo_results.outcomes['caught']:>3}  ({algo_results.outcomes['caught']/total:>5.1%})"
        )
        print(f"  Win Rate:      {win_rate:>5.1%}\n")

    print("=" * 70 + "\n")


# =============================================================================
# EXPORT - CSV export functionality
# =============================================================================


def export_to_csv(
    config: BenchmarkConfig, results: Dict[str, BenchmarkResults]
) -> None:
    """Export benchmark results to CSV file.

    Args:
        config: BenchmarkConfig used for evaluation.
        results: Dictionary mapping algorithm name to BenchmarkResults.
    """
    output_dir = Path(__file__).parent.parent.parent / "data" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"benchmark_results_{timestamp}.csv"

    print("[EXPORTING] Writing results to CSV...")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            [
                "Algorithm",
                "Total Episodes",
                "Goal Reached",
                "Goal Rate (%)",
                "Hit Trap",
                "Trap Rate (%)",
                "Caught by Enemy",
                "Caught Rate (%)",
                "Win Rate (%)",
            ]
        )

        # Write data for each algorithm
        for algo in config.algorithms:
            algo_results = results[algo]
            total = config.num_episodes
            win_rate = algo_results.get_win_rate(total)

            writer.writerow(
                [
                    algo,
                    total,
                    algo_results.outcomes["goal"],
                    f"{algo_results.outcomes['goal']/total*100:.1f}",
                    algo_results.outcomes["trap"],
                    f"{algo_results.outcomes['trap']/total*100:.1f}",
                    algo_results.outcomes["caught"],
                    f"{algo_results.outcomes['caught']/total*100:.1f}",
                    f"{win_rate*100:.1f}",
                ]
            )

    print(f"✓ Results saved to: {filepath}\n")


# =============================================================================
# BENCHMARK EXECUTION - Main evaluation loop
# =============================================================================


def run_benchmark(config: BenchmarkConfig) -> Dict[str, BenchmarkResults]:
    """Run complete benchmark suite evaluating all algorithms.

    Args:
        config: BenchmarkConfig with evaluation parameters.

    Returns:
        Dictionary mapping algorithm name to BenchmarkResults.
    """
    results = {algo: BenchmarkResults() for algo in config.algorithms}

    print(f"\nStarting benchmark: {config.num_episodes} episodes per algorithm...\n")

    # Evaluate each algorithm
    for algo in tqdm(config.algorithms, desc="Algorithms", unit="algorithm"):
        algo_progress = tqdm(
            range(config.num_episodes),
            desc=f"  {algo}",
            unit="episode",
            leave=False,
        )

        for _ in algo_progress:
            # Run episode and record outcome
            outcome = run_single_episode(config, algo)
            results[algo].record_outcome(outcome)

    return results


# =============================================================================
# MAIN ENTRY POINT - Orchestrates pipeline
# =============================================================================


def main() -> None:
    """Execute complete benchmark suite.

    Orchestrates configuration, execution, analysis, and export.
    Each step is isolated for clarity and testability.
    """
    print("\n" + "=" * 70)
    print("TACTICAL AI BENCHMARK SUITE")
    print("=" * 70 + "\n")

    # Configuration
    config = create_benchmark_config()
    print("Configuration:")
    print(f"  Grid Size: {config.grid_width}x{config.grid_height}")
    print(f"  Obstacles: Walls={config.num_walls}, Traps={config.num_traps}")
    print(f"  Environment Seed: {config.environment_seed}")
    print(f"  Algorithms: {', '.join(config.algorithms)}")
    print(f"  Episodes per Algorithm: {config.num_episodes}\n")

    # Execution
    results = run_benchmark(config)

    # Display results
    display_results_summary(config, results)

    # Export
    export_to_csv(config, results)
    print("✓ Benchmark Complete!")


if __name__ == "__main__":
    main()
