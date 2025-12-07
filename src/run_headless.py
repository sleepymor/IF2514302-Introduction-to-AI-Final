from environment.environment import TacticalEnvironment
from agents.player import PlayerAgent
from agents.enemy import EnemyAgent


def run_episode(seed=None):
    env = TacticalEnvironment(width=15, height=10, seed=32)
    player = PlayerAgent(env, algorithm="MCTS")
    enemy = EnemyAgent(env)

    while True:
        if env.turn == "player":
            action = player.action()
        else:
            action = enemy.action()

        terminal, reason = env.step(action, simulate=True)

        if terminal:
            return reason  # goal / trap / caught


def evaluate(n=5):
    stats = {"goal": 0, "trap": 0, "caught": 0}
    for i in range(n):
        result = run_episode(seed=32)
        stats[result] += 1

        if (i + 1) % 5 == 0:
            print(f"Sim {i+1}/{n}")

    print("\nRESULTS:")
    print(f"Goal reached : {stats['goal']}")
    print(f"Traps hit    : {stats['trap']}")
    print(f"Caught       : {stats['caught']}")
    print(f"Win rate     : {stats['goal']/n:.2%}")


if __name__ == "__main__":
    evaluate()
