# test_alphabeta_vs_minimax.py
import pygame
from environment.environment import TacticalEnvironment
from agents.player import PlayerAgent
from agents.enemy import EnemyAgent

class AlphaBetaVsMinimaxTest:
    def __init__(self, width=10, height=8, seed=42):
        pygame.init()
        self.env = TacticalEnvironment(width=width, height=height, seed=seed)
        self.screen = pygame.display.set_mode((width * 40, height * 40))
        self.clock = pygame.time.Clock()
        
        self.player_agent = PlayerAgent(self.env)
        self.enemy_agent = EnemyAgent(self.env)
        
        self.stats = {
            'player_wins': 0,
            'enemy_wins': 0,
            'total_games': 0,
            'steps_per_game': []
        }
        
    def run_single_game(self):
        """Run satu game sampai selesai"""
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            if self.env.turn == "player":
                action = self.player_agent.action()
                self.env.step(action)
                steps += 1
                
            elif self.env.turn == "enemy":
                action = self.enemy_agent.action()
                self.env.step(action)
                steps += 1
            
            
            terminal, reason = self.env.is_terminal()
            if terminal:
                self.stats['total_games'] += 1
                self.stats['steps_per_game'].append(steps)
                
                if reason == "goal":
                    self.stats['player_wins'] += 1
                    print(f"GAME {self.stats['total_games']}: AlphaBeta WON in {steps} steps!")
                else:
                    self.stats['enemy_wins'] += 1  
                    print(f"GAME {self.stats['total_games']}: Minimax WON in {steps} steps! ({reason})")
                
                return reason, steps
            
            # Visualization
            self.screen.fill((20, 20, 30))
            self.env.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(3)
        
        print(f"GAME {self.stats['total_games']}: DRAW after {max_steps} steps")
        self.stats['total_games'] += 1
        self.stats['steps_per_game'].append(max_steps)
        return "draw", max_steps
    
    def run_multiple_games(self, num_games=10):
        """Run multiple games untuk statistik"""
        print("AlphaBeta vs Minimax - Performance Test")
        print("=" * 50)
        
        for game_num in range(num_games):
            print(f"\n--- Game {game_num + 1} ---")
            self.env.reset()
            self.run_single_game()
            
            win_rate = (self.stats['player_wins'] / self.stats['total_games']) * 100
            avg_steps = sum(self.stats['steps_per_game']) / len(self.stats['steps_per_game'])
            
            print(f"Stats: AlphaBeta {self.stats['player_wins']}-{self.stats['enemy_wins']} " 
                  f"({win_rate:.1f}% win rate), Avg steps: {avg_steps:.1f}")
        
        # Final statistics
        print("\n" + "=" * 50)
        print("FINAL RESULTS:")
        print(f"AlphaBeta (Player) Wins: {self.stats['player_wins']}")
        print(f"Minimax (Enemy) Wins: {self.stats['enemy_wins']}") 
        print(f"Draws: {self.stats['total_games'] - self.stats['player_wins'] - self.stats['enemy_wins']}")
        print(f"Win Rate: {(self.stats['player_wins']/self.stats['total_games'])*100:.1f}%")
        print(f"Average Steps: {sum(self.stats['steps_per_game'])/len(self.stats['steps_per_game']):.1f}")
        
        pygame.quit()

if __name__ == "__main__":
    tester = AlphaBetaVsMinimaxTest(width=10, height=8, seed=42)
    tester.run_multiple_games(num_games=5)