from algorithm.mcts.mcts import MCTS
from environment.environment import TacticalEnvironment

def test_mcts():
    env = TacticalEnvironment(width=30, height=10, num_walls=10, num_traps=5, seed=42)
    
    print(f"Initial state:")
    print(f"  Player: {env.player_pos}")
    print(f"  Enemy: {env.enemy_pos}")
    print(f"  Goal: {env.goal}")
    print(f"  Distance to goal: {abs(env.player_pos[0] - env.goal[0]) + abs(env.player_pos[1] - env.goal[1])}")
    print(f"  Distance to enemy: {abs(env.player_pos[0] - env.enemy_pos[0]) + abs(env.player_pos[1] - env.enemy_pos[1])}")
    
    mcts = MCTS(iterations=100, exploration_constant=1.4, max_sim_depth=50)
    action = mcts.search(env)
    
    print(f"\nRecommended action: {action}")

# if __name__ == "__main__":
test_mcts()
