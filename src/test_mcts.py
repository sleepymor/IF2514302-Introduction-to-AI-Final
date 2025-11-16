from utils.config_loader import load_config
from algorithm.mcts.mcts import MCTS
from environment.environment import TacticalEnvironment

def test_mcts():
    config = load_config()

    env = TacticalEnvironment(**config["environment"])
    
    print(f"Initial state:")
    print(f"  Player: {env.player_pos}")
    print(f"  Enemy: {env.enemy_pos}")
    print(f"  Goal: {env.goal}")
    print(f"  Distance to goal: {abs(env.player_pos[0] - env.goal[0]) + abs(env.player_pos[1] - env.goal[1])}")
    print(f"  Distance to enemy: {abs(env.player_pos[0] - env.enemy_pos[0]) + abs(env.player_pos[1] - env.enemy_pos[1])}")
    
    mcts = MCTS(**config["mcts"])
    action = mcts.search(env)
    
    print(f"\nRecommended action: {action}")

# if __name__ == "__main__":
test_mcts()
