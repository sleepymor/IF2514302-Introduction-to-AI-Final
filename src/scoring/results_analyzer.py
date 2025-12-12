"""
Analyzer for benchmark results.
Calculates statistics and metrics.
"""

from typing import Dict, List
from collections import defaultdict
from test_runner import GameResult


class ResultsAnalyzer:
    """Analyzes benchmark test results."""
    
    def __init__(self, results: List[GameResult]):
        """
        Initialize analyzer.
        
        Args:
            results: List of GameResult objects
        """
        self.results = results
    
    def get_stats_by_algorithm(self) -> Dict[str, Dict]:
        """
        Calculate statistics for each algorithm.
        
        Returns:
            Dictionary with stats for each algorithm
        """
        stats_by_algo = defaultdict(lambda: {
            'total_tests': 0,
            'wins': 0,
            'traps': 0,
            'caught': 0,
            'timeout': 0,
            'errors': 0,
            'total_moves': 0,
            'total_turns': 0,
        })

        # Mapping from runner result label -> stats key
        result_to_key = {
            'win': 'wins',
            'trap': 'traps',
            'caught': 'caught',
            'timeout': 'timeout',
            'error': 'errors',
        }
        
        for result in self.results:
            algo = result.algorithm
            stats_by_algo[algo]['total_tests'] += 1

            # Safely map result label to the corresponding stats key
            res_label = result.result if result.result is not None else 'error'
            key = result_to_key.get(res_label, None)

            if key is None:
                # If unknown result label, count as 'errors' and continue
                stats_by_algo[algo]['errors'] += 1
            else:
                stats_by_algo[algo][key] += 1

            stats_by_algo[algo]['total_moves'] += result.moves
            stats_by_algo[algo]['total_turns'] += result.total_turns
        
        # Calculate derived stats
        final_stats = {}
        for algo, stats in stats_by_algo.items():
            if stats['total_tests'] > 0:
                final_stats[algo] = {
                    'Algorithm': algo,
                    'Total Tests': stats['total_tests'],
                    'Wins': stats['wins'],
                    'Win Rate (%)': round(100 * stats['wins'] / stats['total_tests'], 2),
                    'Deaths by Trap': stats['traps'],
                    'Death by Trap Rate (%)': round(100 * stats['traps'] / stats['total_tests'], 2),
                    'Deaths by Enemy': stats['caught'],
                    'Death by Enemy Rate (%)': round(100 * stats['caught'] / stats['total_tests'], 2),
                    'Timeouts': stats['timeout'],
                    'Timeout Rate (%)': round(100 * stats['timeout'] / stats['total_tests'], 2),
                    'Errors': stats['errors'],
                    'Average Moves to Win': round(stats['total_moves'] / stats['total_tests'], 2),
                    'Average Turns': round(stats['total_turns'] / stats['total_tests'], 2),
                }
        
        return final_stats
    
    def get_stats_by_seed(self, algorithm: str = None) -> Dict[int, Dict]:
        """
        Get statistics by seed.
        
        Args:
            algorithm: If specified, only stats for this algorithm
            
        Returns:
            Dictionary with stats for each seed
        """
        stats_by_seed = defaultdict(lambda: {
            'wins': 0,
            'traps': 0,
            'caught': 0,
            'timeout': 0,
            'total': 0,
            'total_moves': 0,
        })

        # Use same mapping here for consistency
        result_to_key = {
            'win': 'wins',
            'trap': 'traps',
            'caught': 'caught',
            'timeout': 'timeout',
            'error': None,  # we don't record error counts per-seed in this structure
        }
        
        for result in self.results:
            if algorithm and result.algorithm != algorithm:
                continue
            
            seed = result.seed
            stats_by_seed[seed]['total'] += 1

            res_label = result.result if result.result is not None else 'error'
            key = result_to_key.get(res_label, None)
            if key:
                stats_by_seed[seed][key] += 1

            stats_by_seed[seed]['total_moves'] += result.moves
        
        # Calculate percentages
        final_stats = {}
        for seed, stats in sorted(stats_by_seed.items()):
            if stats['total'] > 0:
                final_stats[seed] = {
                    'Seed': seed,
                    'Total Tests': stats['total'],
                    'Wins': stats['wins'],
                    'Win Rate (%)': round(100 * stats['wins'] / stats['total'], 2),
                    'Deaths by Trap': stats['traps'],
                    'Deaths by Enemy': stats['caught'],
                    'Timeouts': stats['timeout'],
                    'Avg Moves': round(stats['total_moves'] / stats['total'], 2),
                }
        
        return final_stats
    
    def get_raw_data(self) -> List[Dict]:
        """
        Get all raw data as list of dictionaries.
        
        Returns:
            List of dictionaries with all test data
        """
        data = []
        for result in self.results:
            data.append({
                'Algorithm': result.algorithm,
                'Seed': result.seed,
                'Test #': result.test_num,
                'Result': result.result,
                'Moves': result.moves,
                'Total Turns': result.total_turns,
                'Total Actions': result.total_actions,
            })
        return data