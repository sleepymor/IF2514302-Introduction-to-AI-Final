from environment.environment import TacticalEnvironment
from agents.enemy import EnemyAgent
from agents.player import PlayerAgent
from utils.logger import Logger
import pygame
import random


def main():
    """
    Tactical Grid-base Game Loop

    This function initialize the game enviroment, loads agents, handles
    """
    pygame.init()
    env = TacticalEnvironment(width=15, height=10, seed=32)

    random.seed(None)

    screen = pygame.display.set_mode((env.width * 40, env.height * 40))
    clock = pygame.time.Clock()

    log = Logger("game")

    # --- PERUBAHAN DI SINI ---
    # Kita tambahkan parameter algorithm="ALPHABETA"
    playerAgent = PlayerAgent(env, algorithm="MCTS")
    # -------------------------

    enemyAgent = EnemyAgent(env)

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if env.turn == "player":
            action_x, action_y = playerAgent.action()
            env.step((action_x, action_y))
            # log.info("Player moved

        elif env.turn == "enemy":
            action_x, action_y = enemyAgent.action()
            env.step((action_x, action_y))
            # log.info("Enemy moved!")

        # Di dalam loop main.py atau test_mcts.py
        print(f"Player: {env.player_pos}, Enemy: {env.enemy_pos}")

        screen.fill((20, 20, 30))
        env.draw(screen)
        pygame.display.flip()

        clock.tick(120)

    pygame.quit()


if __name__ == "__main__":
    main()
