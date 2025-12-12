"""
Configuration file for benchmark scoring system.
Customize number of seeds, tests per seed, and algorithm parameters.
"""

class ScoringConfig:
    """Configuration for benchmark scoring."""
    
    def __init__(
        self,
        num_seeds: int = 100,
        tests_per_seed: int = 5,
        algorithms: list = None,
        grid_width: int = 15,
        grid_height: int = 10,
        num_walls: int = 10,
        num_traps: int = 5,
        max_turns: int = 500,
    ):
        """
        Initialize scoring configuration.
        
        Args:
            num_seeds: Number of different random seeds to test
            tests_per_seed: Number of runs per seed (total tests = num_seeds * tests_per_seed)
            algorithms: List of algorithm names to test
            grid_width: Grid width for environment
            grid_height: Grid height for environment
            num_walls: Number of walls in environment
            num_traps: Number of traps in environment
            max_turns: Maximum turns before force-stopping a game
        """
        self.num_seeds = num_seeds
        self.tests_per_seed = tests_per_seed
        self.algorithms = algorithms or ["MCTS", "ALPHABETA", "MINIMAX"]
        
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_walls = num_walls
        self.num_traps = num_traps
        self.max_turns = max_turns
    
    @property
    def total_tests(self) -> int:
        """Calculate total number of tests."""
        return self.num_seeds * self.tests_per_seed
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'num_seeds': self.num_seeds,
            'tests_per_seed': self.tests_per_seed,
            'algorithms': self.algorithms,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'num_walls': self.num_walls,
            'num_traps': self.num_traps,
            'max_turns': self.max_turns,
            'total_tests': self.total_tests,
        }
