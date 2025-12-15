"""Benchmark runner for AI algorithm evaluation.

Executes parallel benchmark tests across multiple seeds and algorithms,
collecting performance metrics and exporting results for analysis.

Architecture:
- BenchmarkConfig: Immutable configuration for reproducibility
- execute_game_worker: Worker process function for multiprocessing
- Result analysis and export pipeline: Separate concerns for clean flow
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import List

logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from results_analyzer import ResultsAnalyzer
from results_exporter import ResultsExporter
from scoring_config import ScoringConfig
from test_runner import GameResult, TestRunner
from tqdm import tqdm

# =============================================================================
# CONFIGURATION - Immutable structure for reproducibility
# =============================================================================


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration object for benchmark execution.

    Ensures consistency across multiprocessing workers and enables
    easy parameter validation.
    """

    test_seeds: List[int]
    num_runs_per_seed: int
    algorithms: List[str]
    environment_config: ScoringConfig
    num_processes: int = 6


def create_benchmark_config() -> BenchmarkConfig:
    """Create benchmark configuration from testable parameters.

    All test parameters are defined here for easy modification without
    searching through function calls. Configuration is immutable to ensure
    consistency across multiprocessing workers.

    Returns:
        BenchmarkConfig with all parameters for benchmark run.
    """
    # =========================================================================
    # TESTABLE PARAMETERS - Modify these for different test scenarios
    # =========================================================================

    TEST_SEEDS = [8]  # Specific seed for reproducible testing
    NUM_RUNS_PER_SEED = 5
    ALGORITHMS = ["MCTS", "ALPHABETA", "MINIMAX"]
    NUM_PROCESSES = 6

    # Environment matches main.py for fair comparison
    GRID_WIDTH = 30
    GRID_HEIGHT = 15
    NUM_WALLS = 125
    NUM_TRAPS = 20
    MAX_TURNS = 300

    # Algorithm-specific hyperparameters
    MCTS_ITERATIONS = 200
    MCTS_SIM_DEPTH = 5
    ALPHABETA_DEPTH = 5
    MINIMAX_DEPTH = 4

    # =========================================================================

    environment_config = ScoringConfig(
        num_seeds=len(TEST_SEEDS),
        tests_per_seed=NUM_RUNS_PER_SEED,
        algorithms=ALGORITHMS,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        num_walls=NUM_WALLS,
        num_traps=NUM_TRAPS,
        max_turns=MAX_TURNS,
        mcts_iterations=MCTS_ITERATIONS,
        mcts_sim_depth=MCTS_SIM_DEPTH,
        alphabeta_depth=ALPHABETA_DEPTH,
        minimax_depth=MINIMAX_DEPTH,
    )

    return BenchmarkConfig(
        test_seeds=TEST_SEEDS,
        num_runs_per_seed=NUM_RUNS_PER_SEED,
        algorithms=ALGORITHMS,
        environment_config=environment_config,
        num_processes=NUM_PROCESSES,
    )


# =============================================================================
# WORKER FUNCTION - Isolated for multiprocessing cleanliness
# =============================================================================


def execute_game_worker(work_item: tuple) -> GameResult:
    """Execute single benchmark game in worker process.

    Isolates test execution in separate process. Arguments are passed as tuple
    to work with multiprocessing.Pool.imap_unordered.

    Args:
        work_item: Tuple of (algorithm, seed, test_num, environment_config).

    Returns:
        GameResult with game outcome and metrics.
    """
    algorithm, seed, test_num, environment_config = work_item
    test_runner = TestRunner(environment_config, benchmark_mode=True)
    return test_runner.run_single_game(algorithm, seed, test_num)


# =============================================================================
# REPORTING - Separate display logic from analysis
# =============================================================================


def _format_algorithm_stats(algorithm_name: str, stats: dict) -> str:
    """Format statistics for single algorithm into display string.

    Extracts formatting logic to reduce repetition and improve maintainability.

    Args:
        algorithm_name: Name of algorithm being reported.
        stats: Statistics dictionary from analyzer.

    Returns:
        Formatted string for console display.
    """
    win_stat = f"{stats['Win Rate (%)']}% ({stats['Wins']} wins)"
    trap_stat = f"{stats['Deaths by Trap']} ({stats['Death by Trap Rate (%)']}%)"
    enemy_stat = f"{stats['Deaths by Enemy']} ({stats['Death by Enemy Rate (%)']}%)"
    timeout_stat = f"{stats['Timeouts']} ({stats['Timeout Rate (%)']}%)"

    return (
        f"\n{algorithm_name}:\n"
        f"  Total Tests: {stats['Total Tests']}\n"
        f"  Win Rate: {win_stat}\n"
        f"  Deaths by Trap: {trap_stat}\n"
        f"  Deaths by Enemy: {enemy_stat}\n"
        f"  Timeouts: {timeout_stat}\n"
        f"  Errors: {stats['Errors']}\n"
        f"  Average Moves to Win: {stats['Average Moves to Win']}\n"
        f"  Average Turns: {stats['Average Turns']}"
    )


def display_results_summary(analyzer: ResultsAnalyzer) -> None:
    """Display benchmark results in formatted summary.

    Responsible only for presentation. Analysis is done by ResultsAnalyzer.

    Args:
        analyzer: ResultsAnalyzer with computed statistics.
    """
    stats_by_algorithm = analyzer.get_stats_by_algorithm()

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)

    for algorithm in sorted(stats_by_algorithm.keys()):
        formatted = _format_algorithm_stats(algorithm, stats_by_algorithm[algorithm])
        print(formatted)

    print("\n" + "=" * 90 + "\n")


def display_configuration(config: BenchmarkConfig) -> None:
    """Display benchmark configuration to console.

    Args:
        config: BenchmarkConfig to display.
    """
    env = config.environment_config
    total_tests = (
        len(config.test_seeds) * config.num_runs_per_seed * len(config.algorithms)
    )

    print("Configuration:")
    print(f"  Test Seeds: {config.test_seeds}")
    print(f"  Runs per Seed: {config.num_runs_per_seed}")
    print(f"  Algorithms: {', '.join(config.algorithms)}")
    print(f"  Total Tests: {total_tests}")
    print(f"  Grid Size: {env.grid_width}x{env.grid_height}")
    print(f"  Obstacles: Walls={env.num_walls}, Traps={env.num_traps}")
    print(f"  Max Turns: {env.max_turns}")
    print(f"  MCTS: iterations={env.mcts_iterations}, depth={env.mcts_sim_depth}")
    print(f"  AlphaBeta: depth={env.alphabeta_depth}")
    print(f"  Minimax: depth={env.minimax_depth}\n")


# =============================================================================
# BENCHMARK EXECUTION - Clear separation of concerns
# =============================================================================


def create_work_items(config: BenchmarkConfig) -> List[tuple]:
    """Generate all work items for parallel execution.

    Creates Cartesian product of algorithms × seeds × runs.

    Args:
        config: BenchmarkConfig defining test parameters.

    Returns:
        List of (algorithm, seed, test_num, config) tuples.
    """
    work_items = []
    for algorithm in config.algorithms:
        for seed in config.test_seeds:
            for test_num in range(config.num_runs_per_seed):
                work_items.append(
                    (algorithm, seed, test_num, config.environment_config)
                )
    return work_items


def run_benchmark_tests(config: BenchmarkConfig) -> List[GameResult]:
    """Execute all benchmark tests in parallel.

    Uses multiprocessing.Pool for parallel execution. Gracefully handles
    missing tqdm library for progress tracking.

    Args:
        config: BenchmarkConfig with test parameters.

    Returns:
        List of GameResult from all tests.
    """
    work_items = create_work_items(config)
    total_tests = len(work_items)

    print(
        f"[RUNNING] Starting {total_tests} tests with {config.num_processes} processes..."
    )

    results = []
    with Pool(processes=config.num_processes) as process_pool:
        # Create iterator - use tqdm if available for progress tracking
        iterator = tqdm(
            process_pool.imap_unordered(execute_game_worker, work_items),
            total=total_tests,
            desc="Progress",
        )

        results = list(iterator)

    return results


def export_results(results: List[GameResult]) -> None:
    """Export benchmark results to Excel file.

    Args:
        results: List of GameResult from benchmark.
    """
    print("[EXPORTING] Preparing results export...")
    exporter = ResultsExporter()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.xlsx"
    exporter.export_to_excel(results, filename)

    print(f"✓ Results saved to: {exporter.output_dir}")


# =============================================================================
# MAIN ENTRY POINT - Orchestrates pipeline
# =============================================================================


def main() -> None:
    """Execute complete benchmark suite.

    Orchestrates configuration, execution, analysis, and export.
    Each step is isolated for clarity and testability.
    """
    print("\n" + "=" * 90)
    print("TACTICAL AI BENCHMARK SUITE")
    print("=" * 90 + "\n")

    # Configuration
    config = create_benchmark_config()
    display_configuration(config)

    # Execution
    results = run_benchmark_tests(config)

    # Analysis and reporting
    print("\n[ANALYZING] Processing results...")
    analyzer = ResultsAnalyzer(results)
    display_results_summary(analyzer)

    # Export
    export_results(results)
    print("✓ Benchmark Complete!")


if __name__ == "__main__":
    main()
