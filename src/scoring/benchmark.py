"""Benchmark runner for AI algorithm evaluation.

Evaluates multiple AI algorithms on the same environment and seed,
collecting performance metrics and exporting results to multiple formats.

Uses multiprocessing to run episodes in parallel for faster evaluation.

Modular Architecture:
- benchmark_data: Configuration and results data structures
- episode_runner: Episode execution for multiprocessing
- results_displayer: Console output formatting
- csv_exporter: CSV export (aggregate and per-seed)
- excel_exporter: Excel export (multi-sheet)
- benchmark: Main orchestration and execution
"""

import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict

from tqdm import tqdm

# Import modular components
from benchmark_data import BenchmarkConfig, BenchmarkResults
from csv_exporter import CSVExporter
from episode_runner import EpisodeRunner
from excel_exporter import ExcelExporter
from results_displayer import ResultsDisplayer

logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import load_config

# =============================================================================
# CONFIGURATION
# =============================================================================


def create_benchmark_config() -> BenchmarkConfig:
    """Create benchmark configuration from testable parameters.

    MODIFY these parameters to change evaluation scenarios:
    - ENVIRONMENT_SEEDS: List of seeds for environment variation
    - EPISODES_PER_SEED: Episodes per seed (more = more reliable but slower)
    - ALGORITHMS: Which algorithms to compare
    - Grid and obstacle parameters for map difficulty

    Returns:
        BenchmarkConfig with all benchmark parameters.
    """
    # =========================================================================
    # TESTABLE PARAMETERS - Modify for different evaluation scenarios
    # =========================================================================

    config = load_config("configs/config.yaml")

    GRID_WIDTH = config["benchmark"]["grid_width"]
    GRID_HEIGHT = config["benchmark"]["grid_height"]
    NUM_WALLS = config["benchmark"]["num_walls"]
    NUM_TRAPS = config["benchmark"]["num_traps"]
    ENVIRONMENT_SEEDS = config["benchmark"]["environment_seed"]
    ALGORITHMS = config["benchmark"]["player_algorithms"]
    EPISODES_PER_SEED = config["benchmark"]["num_episodes"]
    MAX_MOVES = config["benchmark"]["max_moves"]
    NUM_PROCESSES = config["benchmark"]["num_processes"]
    # =========================================================================

    return BenchmarkConfig(
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        num_walls=NUM_WALLS,
        num_traps=NUM_TRAPS,
        environment_seeds=ENVIRONMENT_SEEDS,
        algorithms=ALGORITHMS,
        episodes_per_seed=EPISODES_PER_SEED,
        max_moves_per_episode=MAX_MOVES,
        num_processes=NUM_PROCESSES,
    )


# =============================================================================
# BENCHMARK ORCHESTRATION
# =============================================================================


class BenchmarkOrchestrator:
    """Main orchestrator for complete benchmark execution.

    Coordinates:
    1. Configuration creation
    2. Parallel episode execution
    3. Results aggregation
    4. Display and export
    """

    @staticmethod
    def run(config: BenchmarkConfig) -> Dict[str, BenchmarkResults]:
        """Execute complete benchmark suite.

        Args:
            config: BenchmarkConfig with evaluation parameters

        Returns:
            Algorithm name -> BenchmarkResults mapping
        """
        results = {algo: BenchmarkResults() for algo in config.algorithms}

        total_episodes = len(config.environment_seeds) * config.episodes_per_seed
        print(
            f"\nStarting benchmark: {len(config.environment_seeds)} seeds × "
            f"{config.episodes_per_seed} episodes per algorithm"
        )
        print(f"Seeds: {config.environment_seeds}")
        print(f"Total episodes per algorithm: {total_episodes}")
        print(f"Using {config.num_processes} parallel processes...\n")

        # Run each algorithm
        for algo in tqdm(config.algorithms, desc="Algorithms", unit="algorithm"):
            BenchmarkOrchestrator._run_algorithm(config, algo, results)

        return results

    @staticmethod
    def _run_algorithm(
        config: BenchmarkConfig, algo: str, results: Dict[str, BenchmarkResults]
    ) -> None:
        """Run all episodes for a single algorithm.

        Args:
            config: BenchmarkConfig with evaluation parameters
            algo: Algorithm name to evaluate
            results: Results dictionary to populate
        """
        # Prepare work items
        algo_work = [
            (config, algo, seed, ep_idx)
            for seed in config.environment_seeds
            for ep_idx in range(config.episodes_per_seed)
        ]

        # Execute in parallel
        with Pool(processes=config.num_processes) as pool:
            algo_outcomes = tqdm(
                pool.imap_unordered(EpisodeRunner.worker, algo_work),
                total=len(algo_work),
                desc=f"  {algo}",
                unit="ep",
                leave=False,
            )

            for outcome, seed, episode_idx, move_count in algo_outcomes:
                results[algo].record_outcome(outcome, seed, episode_idx, move_count)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """Execute complete benchmark suite.

    Pipeline:
    1. Configuration
    2. Execution
    3. Display
    4. Export to CSV and Excel
    """
    print("\n" + "=" * 70)
    print("TACTICAL AI BENCHMARK SUITE")
    print("=" * 70 + "\n")

    # Configuration
    config = create_benchmark_config()
    total_episodes = len(config.environment_seeds) * config.episodes_per_seed

    print("Configuration:")
    print(f"  Grid Size: {config.grid_width}x{config.grid_height}")
    print(f"  Obstacles: Walls={config.num_walls}, Traps={config.num_traps}")
    print(f"  Seeds to Test: {config.environment_seeds}")
    print(f"  Episodes per Seed: {config.episodes_per_seed}")
    print(f"  Max Moves per Episode: {config.max_moves_per_episode}")
    print(f"  Total Episodes per Algorithm: {total_episodes}")
    print(f"  Algorithms: {', '.join(config.algorithms)}\n")

    # Execution
    results = BenchmarkOrchestrator.run(config)

    # Display
    ResultsDisplayer.display_summary(config, results)

    # Export
    CSVExporter.export_all(config, results)
    ExcelExporter.export(config, results)
    print("✓ Benchmark Complete!")


if __name__ == "__main__":
    main()
