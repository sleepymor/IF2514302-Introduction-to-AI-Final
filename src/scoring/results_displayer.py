"""Results display formatting for benchmark.

Provides human-readable console output of benchmark results.
"""

from typing import Dict

from benchmark_data import BenchmarkConfig, BenchmarkResults


class ResultsDisplayer:
    """Formats and displays benchmark results in console.

    Provides human-readable summary of algorithm performance.
    """

    @staticmethod
    def display_summary(
        config: BenchmarkConfig, results: Dict[str, BenchmarkResults]
    ) -> None:
        """Display formatted results summary.

        Args:
            config: BenchmarkConfig used for evaluation
            results: Algorithm name -> BenchmarkResults mapping
        """
        total_episodes = len(config.environment_seeds) * config.episodes_per_seed

        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total Episodes per Algorithm: {total_episodes}")
        print(
            f"  ({len(config.environment_seeds)} seeds × {config.episodes_per_seed} episodes)"
        )
        print(f"Seeds Tested: {config.environment_seeds}")
        print(f"Grid Size: {config.grid_width}x{config.grid_height}\n")

        for algo in config.algorithms:
            algo_results = results[algo]
            win_rate = algo_results.get_win_rate(total_episodes)
            avg_moves = algo_results.get_average_moves()

            print(f"\n{algo}:")
            print(
                f"  Goal Reached:  {algo_results.outcomes.get('goal', 0):>3}  ({algo_results.outcomes.get('goal', 0)/total_episodes:>5.1%})"
            )
            print(
                f"  Hit Trap:      {algo_results.outcomes.get('trap', 0):>3}  ({algo_results.outcomes.get('trap', 0)/total_episodes:>5.1%})"
            )
            print(
                f"  Caught:        {algo_results.outcomes.get('caught', 0):>3}  ({algo_results.outcomes.get('caught', 0)/total_episodes:>5.1%})"
            )
            print(
                f"  Timeout:       {algo_results.outcomes.get('timeout', 0):>3}  ({algo_results.outcomes.get('timeout', 0)/total_episodes:>5.1%})"
            )
            print(f"  Win Rate:      {win_rate:>5.1%}")
            print(f"  Avg Moves:     {avg_moves:>5.1f}\n")

        print("=" * 70 + "\n")
