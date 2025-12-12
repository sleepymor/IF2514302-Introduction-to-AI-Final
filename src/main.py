from environment.environment import TacticalEnvironment
from agents.enemy import EnemyAgent
from agents.player import PlayerAgent
from utils.logger import Logger
import pygame
import random

TILE_SIZE = 40


def main():
    pygame.init()

    env = TacticalEnvironment(width=15, height=10, seed=32)
    random.seed(None)

    screen = pygame.display.set_mode((env.width * TILE_SIZE, env.height * TILE_SIZE))
    pygame.display.set_caption("Tactical AI: Minimax/AlphaBeta")
    clock = pygame.time.Clock()

    log = Logger("MainGame")

    # Ganti algorithm="MINIMAX" atau "ALPHABETA" atau "MCTS" sesuai kebutuhan
    playerAgent = PlayerAgent(env, algorithm="MINIMAX")
    enemyAgent = EnemyAgent(env)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # if event.type == pygame.MOUSEBUTTONDOWN:
        if env.turn == "player":
            action_x, action_y = playerAgent.action()
            env.step((action_x, action_y))

        elif env.turn == "enemy":
            action_x, action_y = enemyAgent.action()
            env.step((action_x, action_y))

        print(f"Player: {env.player_pos}, Enemy: {env.enemy_pos}")

        screen.fill((20, 20, 30))
        env.draw(screen)
        pygame.display.flip()

        clock.tick(1)

    pygame.quit()


if __name__ == "__main__":
    main()
