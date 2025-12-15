"""Benchmark data structures and results tracking.

Defines:
- BenchmarkConfig: Configuration dataclass
- BenchmarkResults: Results aggregation and statistics
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class BenchmarkConfig:
    """Immutable configuration for benchmark evaluation.

    Attributes:
        grid_width: Map width in cells
        grid_height: Map height in cells
        num_walls: Number of wall obstacles
        num_traps: Number of trap obstacles
        environment_seeds: List of seeds to test for environment variation
        algorithms: List of algorithm names to evaluate
        episodes_per_seed: Number of episodes per seed per algorithm
        max_moves_per_episode: Max moves before auto-terminating episode
        num_processes: Number of parallel CPU processes
    """

    grid_width: int
    grid_height: int
    num_walls: int
    num_traps: int
    environment_seeds: List[int]
    algorithms: List[str]
    episodes_per_seed: int
    max_moves_per_episode: int = 300
    num_processes: int = 4


@dataclass
class BenchmarkResults:
    """Container for episode results and statistics.

    Tracks outcomes at three levels:
    1. Aggregate: Total counts across all episodes
    2. Per-seed: Breakdown by environment seed
    3. Episode details: Individual episode information

    Attributes:
        outcomes: Aggregate outcome counts
        outcomes_by_seed: Per-seed outcome breakdown
        episode_details: Individual episode records with move counts
    """

    outcomes: Dict[str, int] = field(
        default_factory=lambda: {"goal": 0, "trap": 0, "caught": 0, "timeout": 0}
    )
    outcomes_by_seed: Dict[int, Dict[str, int]] = field(default_factory=dict)
    episode_details: List[Dict] = field(default_factory=list)

    def record_outcome(
        self, reason: str, seed: int = None, episode: int = None, move_count: int = 0
    ) -> None:
        """Record an episode outcome.

        Updates aggregate counts, per-seed counts, and detailed records.

        Args:
            reason: Terminal reason ("goal", "trap", "caught", "timeout")
            seed: Environment seed for this episode
            episode: Episode index for tracking
            move_count: Number of moves in this episode
        """
        # Update aggregate counts
        self.outcomes[reason] = self.outcomes.get(reason, 0) + 1

        # Update per-seed counts
        if seed is not None:
            if seed not in self.outcomes_by_seed:
                self.outcomes_by_seed[seed] = {
                    "goal": 0,
                    "trap": 0,
                    "caught": 0,
                    "timeout": 0,
                }
            self.outcomes_by_seed[seed][reason] = (
                self.outcomes_by_seed[seed].get(reason, 0) + 1
            )

        # Record episode-level details
        if seed is not None and episode is not None:
            self.episode_details.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "outcome": reason,
                    "moves": move_count,
                }
            )

    def get_win_rate(self, total_episodes: int) -> float:
        """Calculate win rate (goals reached / total episodes).

        Args:
            total_episodes: Total number of episodes

        Returns:
            Win rate as decimal (0.0 to 1.0)
        """
        return self.outcomes["goal"] / total_episodes if total_episodes > 0 else 0.0

    def get_average_moves(self) -> float:
        """Calculate average moves across all episodes.

        Returns:
            Average move count per episode
        """
        if not self.episode_details:
            return 0.0
        total_moves = sum(ep["moves"] for ep in self.episode_details)
        return total_moves / len(self.episode_details)

    def get_seed_average_moves(self, seed: int) -> float:
        """Calculate average moves for specific seed.

        Args:
            seed: Environment seed

        Returns:
            Average move count for this seed's episodes
        """
        seed_episodes = [ep for ep in self.episode_details if ep["seed"] == seed]
        if not seed_episodes:
            return 0.0
        total_moves = sum(ep["moves"] for ep in seed_episodes)
        return total_moves / len(seed_episodes)
