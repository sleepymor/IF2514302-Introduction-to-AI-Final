"""CSV export functionality for benchmark results.

Exports aggregate and per-seed results to CSV format.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

from benchmark_data import BenchmarkConfig, BenchmarkResults


class CSVExporter:
    """Exports benchmark results to CSV format."""

    OUTPUT_DIR = None  # Lazily initialized

    @classmethod
    def _ensure_output_dir(cls) -> Path:
        """Ensure output directory exists.

        Returns:
            Path to output directory
        """
        if cls.OUTPUT_DIR is None:
            cls.OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "results"
            cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return cls.OUTPUT_DIR

    @classmethod
    def export_all(
        cls, config: BenchmarkConfig, results: Dict[str, BenchmarkResults]
    ) -> None:
        """Export results to both aggregate and per-seed CSV files.

        Args:
            config: BenchmarkConfig used for evaluation
            results: Algorithm name -> BenchmarkResults mapping
        """
        cls.export_to_csv(config, results)
        cls.export_detailed_per_seed_csv(config, results)

    @classmethod
    def export_to_csv(
        cls, config: BenchmarkConfig, results: Dict[str, BenchmarkResults]
    ) -> None:
        """Export aggregate results to CSV.

        Single CSV file with one row per algorithm showing overall statistics.

        Args:
            config: BenchmarkConfig used for evaluation
            results: Algorithm name -> BenchmarkResults mapping
        """
        output_dir = cls._ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"benchmark_results_{timestamp}.csv"

        print("[EXPORTING] Writing results to CSV...")

        total_episodes = len(config.environment_seeds) * config.episodes_per_seed

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Algorithm",
                    "Total Episodes",
                    "Seeds Tested",
                    "Episodes per Seed",
                    "Max Moves",
                    "Goal Reached",
                    "Goal Rate (%)",
                    "Hit Trap",
                    "Trap Rate (%)",
                    "Caught by Enemy",
                    "Caught Rate (%)",
                    "Timeout",
                    "Timeout Rate (%)",
                    "Win Rate (%)",
                    "Avg Moves",
                ]
            )

            # Data rows
            for algo in config.algorithms:
                algo_results = results[algo]
                win_rate = algo_results.get_win_rate(total_episodes)
                avg_moves = algo_results.get_average_moves()

                writer.writerow(
                    [
                        algo,
                        total_episodes,
                        len(config.environment_seeds),
                        config.episodes_per_seed,
                        config.max_moves_per_episode,
                        algo_results.outcomes.get("goal", 0),
                        f"{algo_results.outcomes.get('goal', 0)/total_episodes*100:.1f}",
                        algo_results.outcomes.get("trap", 0),
                        f"{algo_results.outcomes.get('trap', 0)/total_episodes*100:.1f}",
                        algo_results.outcomes.get("caught", 0),
                        f"{algo_results.outcomes.get('caught', 0)/total_episodes*100:.1f}",
                        algo_results.outcomes.get("timeout", 0),
                        f"{algo_results.outcomes.get('timeout', 0)/total_episodes*100:.1f}",
                        f"{win_rate*100:.1f}",
                        f"{avg_moves:.1f}",
                    ]
                )

        print(f"✓ Results saved to: {filepath}\n")

    @classmethod
    def export_detailed_per_seed_csv(
        cls, config: BenchmarkConfig, results: Dict[str, BenchmarkResults]
    ) -> None:
        """Export per-seed results to CSV.

        One row per algorithm-seed combination with per-seed statistics.

        Args:
            config: BenchmarkConfig used for evaluation
            results: Algorithm name -> BenchmarkResults mapping
        """
        output_dir = cls._ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"benchmark_detailed_per_seed_{timestamp}.csv"

        print("[EXPORTING] Writing detailed per-seed results to CSV...")

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Algorithm",
                    "Seed",
                    "Episodes",
                    "Goal Reached",
                    "Goal Rate (%)",
                    "Hit Trap",
                    "Trap Rate (%)",
                    "Caught",
                    "Caught Rate (%)",
                    "Timeout",
                    "Timeout Rate (%)",
                    "Win Rate (%)",
                    "Avg Moves",
                ]
            )

            # Data rows
            for algo in config.algorithms:
                algo_results = results[algo]
                for seed in config.environment_seeds:
                    if seed in algo_results.outcomes_by_seed:
                        seed_outcomes = algo_results.outcomes_by_seed[seed]
                        total = config.episodes_per_seed
                        win_rate = (
                            seed_outcomes.get("goal", 0) / total * 100
                            if total > 0
                            else 0
                        )
                        avg_moves = algo_results.get_seed_average_moves(seed)

                        writer.writerow(
                            [
                                algo,
                                seed,
                                total,
                                seed_outcomes.get("goal", 0),
                                f"{seed_outcomes.get('goal', 0)/total*100:.1f}",
                                seed_outcomes.get("trap", 0),
                                f"{seed_outcomes.get('trap', 0)/total*100:.1f}",
                                seed_outcomes.get("caught", 0),
                                f"{seed_outcomes.get('caught', 0)/total*100:.1f}",
                                seed_outcomes.get("timeout", 0),
                                f"{seed_outcomes.get('timeout', 0)/total*100:.1f}",
                                f"{win_rate:.1f}",
                                f"{avg_moves:.1f}",
                            ]
                        )

        print(f"✓ Detailed per-seed results saved to: {filepath}\n")
