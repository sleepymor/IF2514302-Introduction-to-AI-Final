import random
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# CRITICAL: Disable logging early before any logger is created
logging.disable(logging.CRITICAL)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.environment import TacticalEnvironment
from agents.enemy import EnemyAgent
from agents.player import PlayerAgent
from utils.logger import Logger
from scoring_config import ScoringConfig

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class GameResult:
    """Data class for a single game result."""
    
    def __init__(self):
        self.seed = None
        self.algorithm = None
        self.test_num = None
        self.result = None
        self.moves = 0
        self.total_turns = 0
        self.total_actions = 0


class TestRunner:
    """Runs benchmark tests for AI algorithms."""
    
    def __init__(self, config: ScoringConfig, benchmark_mode=False):
        """Initialize test runner."""
        self.config = config
        self.benchmark_mode = benchmark_mode
        self.logger = Logger("TestRunner")
        self.results: List[GameResult] = []
        
        if benchmark_mode:
            # Already disabled at module level, but ensure it
            logging.disable(logging.CRITICAL)
            Logger.set_benchmark_mode(True)

    def run_single_game(
        self,
        algorithm: str,
        seed: int,
        test_num: int,
    ) -> GameResult:
        """
        Run a single game with specified algorithm and seed.
        """
        result = GameResult()
        result.seed = seed
        result.algorithm = algorithm
        result.test_num = test_num

        try:
            env = TacticalEnvironment(
                width=self.config.grid_width,
                height=self.config.grid_height,
                num_walls=self.config.num_walls,
                num_traps=self.config.num_traps,
                seed=seed,
                use_assets=False,
            )

            player_agent = PlayerAgent(
                env, algorithm=algorithm, benchmark_mode=self.benchmark_mode
            )
            enemy_agent = EnemyAgent(env)

            turn_count = 0
            max_turns = self.config.max_turns
            game_over = False
            end_reason = None

            while turn_count < max_turns and not game_over:
                if env.turn == "player":
                    action = player_agent.action()
                    is_terminal, reason = env.step(action, simulate=True)

                    if is_terminal:
                        game_over = True
                        end_reason = reason
                        result.moves += 1
                        result.total_actions += 1
                    else:
                        result.moves += 1
                        result.total_actions += 1

                if not game_over and env.turn == "enemy":
                    enemy_moves = env.get_move_range(env.enemy_pos)

                    best_move = env.enemy_pos
                    min_dist = float("inf")

                    for move in enemy_moves:
                        dist = abs(move[0] - env.player_pos[0]) + abs(
                            move[1] - env.player_pos[1]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            best_move = move

                    is_terminal, reason = env.step(best_move, simulate=True)
                    result.total_actions += 1

                    if is_terminal:
                        game_over = True
                        end_reason = reason

                turn_count += 1

            if end_reason == "goal":
                result.result = "win"
            elif end_reason == "trap":
                result.result = "trap"
            elif end_reason == "caught":
                result.result = "caught"
            else:
                result.result = "timeout"

            result.total_turns = turn_count

        except Exception as e:
            self.logger.error(f"Error running game: {str(e)}")
            result.result = "error"
            result.moves = 0

        return result

    def run_benchmark(self, verbose: bool = True) -> List[GameResult]:
        """
        Run complete benchmark suite with progress bars.

        Args:
            verbose: Whether to print progress

        Returns:
            List[GameResult]: All results from benchmark
        """
        total_tests = self.config.total_tests

        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting Benchmark Suite")
            print(f"{'='*70}")
            print(f"Algorithms: {', '.join(self.config.algorithms)}")
            print(f"Seeds: {self.config.num_seeds}")
            print(f"Tests per seed: {self.config.tests_per_seed}")
            print(f"Total tests: {total_tests}")
            print(f"{'='*70}\n")

        # Test each algorithm
        algo_iterator = (
            tqdm(self.config.algorithms, desc="Algorithms", position=0)
            if HAS_TQDM
            else self.config.algorithms
        )

        for algorithm in algo_iterator:
            if verbose and HAS_TQDM:
                algo_iterator.set_description(f"Algorithm: {algorithm}")
            elif verbose:
                print(f"\n[ALGORITHM: {algorithm}]")

            # Test each seed with progress bar
            seed_iterator = (
                tqdm(
                    range(self.config.num_seeds),
                    desc=f"  Seeds",
                    position=1,
                    leave=False,
                )
                if HAS_TQDM
                else range(self.config.num_seeds)
            )

            for seed in seed_iterator:
                # Run tests_per_seed games for this seed with nested progress bar
                test_iterator = (
                    tqdm(
                        range(self.config.tests_per_seed),
                        desc=f"    Tests/Seed",
                        position=2,
                        leave=False,
                    )
                    if HAS_TQDM
                    else range(self.config.tests_per_seed)
                )

                for test_num in test_iterator:
                    result = self.run_single_game(algorithm, seed, test_num)
                    self.results.append(result)

                    if HAS_TQDM:
                        test_iterator.update(1)

        if verbose:
            print(f"\n{'='*70}")
            print(f"Benchmark Complete!")
            print(f"{'='*70}\n")

        return self.results
