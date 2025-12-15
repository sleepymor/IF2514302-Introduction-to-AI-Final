"""Headless evaluation of AI agents without pygame visualization.

Runs episodic evaluations of player and enemy agents in TacticalEnvironment,
collecting outcome statistics for performance analysis.

Architecture:
- Configuration parameters at module level for testability
- Episode runner isolated from evaluation logic
- Results aggregator handles statistics
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

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
    environment_seed: int
    player_algorithm: str
    num_episodes: int
    progress_interval: int


def create_evaluation_config() -> EvaluationConfig:
    """Create evaluation configuration from testable parameters.

    Returns:
        EvaluationConfig with all evaluation parameters.
    """
    # =========================================================================
    # TESTABLE PARAMETERS - Modify for different evaluation scenarios
    # =========================================================================
    GRID_WIDTH = 15
    GRID_HEIGHT = 10
    ENVIRONMENT_SEED = 8
    PLAYER_ALGORITHM = "MCTS"  # "MCTS", "ALPHABETA", or "MINIMAX"
    NUM_EPISODES = 15
    PROGRESS_INTERVAL = 5
    # =========================================================================

    return EvaluationConfig(
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        environment_seed=ENVIRONMENT_SEED,
        player_algorithm=PLAYER_ALGORITHM,
        num_episodes=NUM_EPISODES,
        progress_interval=PROGRESS_INTERVAL,
    )


# =============================================================================
# EPISODE RUNNER - Single game execution
# =============================================================================


def run_single_episode(config: EvaluationConfig) -> str:
    """Execute single episode and return terminal reason.

    Args:
        config: EvaluationConfig with game parameters.

    Returns:
        Terminal reason: "goal", "trap", or "caught".
    """
    # Create environment and agents
    env = TacticalEnvironment(
        width=config.grid_width,
        height=config.grid_height,
        seed=config.environment_seed,
    )
    player_agent = PlayerAgent(env, algorithm=config.player_algorithm)
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
    """Run complete evaluation suite.

    Args:
        config: EvaluationConfig with evaluation parameters.

    Returns:
        EvaluationResults with collected statistics.
    """
    results = EvaluationResults()

    print(f"Starting evaluation: {config.num_episodes} episodes...\n")

    for episode_num in tqdm(
        range(config.num_episodes),
        desc="Evaluation Progress",
        unit="episode",
    ):
        # Run episode and record outcome
        outcome = run_single_episode(config)
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
