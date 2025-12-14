"""
Optimized benchmark runner with multiprocessing and reduced logging.
Entry point for running the complete benchmark suite faster.
"""

import sys
import os
import logging
from pathlib import Path
from multiprocessing import Pool

logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring_config import ScoringConfig
from test_runner import TestRunner, GameResult
from results_analyzer import ResultsAnalyzer
from results_exporter import ResultsExporter
from utils.logger import Logger
from datetime import datetime

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class OptimizedTestRunner(TestRunner):
    """Optimized test runner with faster algorithm parameters."""

    def __init__(self, config: ScoringConfig, benchmark_mode=True):
        super().__init__(config, benchmark_mode=benchmark_mode)
        if benchmark_mode:
            self._reduce_algorithm_params()

    def _reduce_algorithm_params(self):
        """Reduce algorithm parameters for faster benchmarking."""
        self.config.mcts_iterations = 200
        self.config.mcts_sim_depth = 5
        self.config.alphabeta_depth = 4
        self.config.minimax_depth = 3


def run_single_game_worker(args):
    """Worker function for multiprocessing."""
    algorithm, seed, test_num, config = args
    runner = OptimizedTestRunner(config, benchmark_mode=True)
    return runner.run_single_game(algorithm, seed, test_num)


def print_results_summary(analyzer: ResultsAnalyzer):
    """Print a nice summary of results."""
    stats = analyzer.get_stats_by_algorithm()

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)

    for algo, stat in sorted(stats.items()):
        print(f"\n{algo}:")
        print(f"  Total Tests: {stat['Total Tests']}")
        print(f"  Win Rate: {stat['Win Rate (%)']}% ({stat['Wins']} wins)")
        print(
            f"  Deaths by Trap: {stat['Deaths by Trap']} ({stat['Death by Trap Rate (%)']}%)"
        )
        print(
            f"  Deaths by Enemy: {stat['Deaths by Enemy']} ({stat['Death by Enemy Rate (%)']}%)"
        )
        print(f"  Timeouts: {stat['Timeouts']} ({stat['Timeout Rate (%)']}%)")
        print(f"  Errors: {stat['Errors']}")
        print(f"  Average Moves to Win: {stat['Average Moves to Win']}")
        print(f"  Average Turns: {stat['Average Turns']}")

    print("\n" + "=" * 90 + "\n")


def main():
    """Main entry point for optimized benchmarking."""

    print("\n" + "=" * 90)
    print("TACTICAL AI BENCHMARK (OPTIMIZED)")
    print("=" * 90 + "\n")

    config = ScoringConfig(
        num_seeds=5,
        tests_per_seed=100,
        algorithms=["MCTS", "ALPHABETA", "MINIMAX"],
        grid_width=30,
        grid_height=15,
        num_walls=125,
        num_traps=20,
        max_turns=300,
    )

    print(f"Configuration:")
    print(f"  Algorithms: {', '.join(config.algorithms)}")
    print(f"  Seeds: {config.num_seeds}")
    print(f"  Tests per seed: {config.tests_per_seed}")
    print(f"  Total tests: {config.total_tests}")
    print(f"  Grid size: {config.grid_width}x{config.grid_height}")
    print(f"  Max turns per game: {config.max_turns}")
    print(f"  Mode: OPTIMIZED (No logging, reduced parameters)\n")

    if not HAS_TQDM:
        print("Note: Install tqdm for progress bars: pip install tqdm\n")

    work_items = []
    for algorithm in config.algorithms:
        for seed in range(config.num_seeds):
            for test_num in range(config.tests_per_seed):
                work_items.append((algorithm, seed, test_num, config))

    print(f"[RUNNING] Starting {len(work_items)} tests with multiprocessing...")

    results = []
    num_processes = 6

    with Pool(processes=num_processes) as pool:
        iterator = (
            tqdm(
                pool.imap_unordered(run_single_game_worker, work_items),
                total=len(work_items),
                desc="Progress",
            )
            if HAS_TQDM
            else pool.imap_unordered(run_single_game_worker, work_items)
        )

        for result in iterator:
            results.append(result)

    # Analyze results
    print("\n[ANALYZING] Processing results...")
    analyzer = ResultsAnalyzer(results)
    print_results_summary(analyzer)

    # Export to Excel
    print("[EXPORTING] Preparing export...")
    exporter = ResultsExporter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exporter.export_to_excel(results, f"benchmark_results_{timestamp}.xlsx")

    print(f"✓ Benchmark Complete!")
    print(f"✓ Results saved to: {exporter.output_dir}")


if __name__ == "__main__":
    main()
