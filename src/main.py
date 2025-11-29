from environment.environment import TacticalEnvironment
from agents.enemy import EnemyAgent
from agents.player import PlayerAgent
from utils.logger import Logger
import pygame


def main():
    """
    Tactical Grid-base Game Loop

    This function initialize the game enviroment, loads agents, handles
    """
    pygame.init()
    env = TacticalEnvironment(width=15, height=10, seed=32)

    screen = pygame.display.set_mode((env.width * 40, env.height * 40))
    clock = pygame.time.Clock()

    playerAgent = PlayerAgent(env, algorithm="ALPHABETA")

    enemyAgent = EnemyAgent(env)

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     if env.turn == "player":
            #         playerAgent.action()
            #         env.step((0, 3))
            #         # env.step((1, 4))

            #     elif env.turn == "enemy":
            #         env.step((6, 1))

        if env.turn == "player":
            action_x, action_y = playerAgent.action()
            env.step((action_x, action_y))

        elif env.turn == "enemy":
            action_x, action_y = enemyAgent.action()
            env.step((action_x, action_y))

        screen.fill((20, 20, 30))
        env.draw(screen)
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
