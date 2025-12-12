import pygame
import random

from environment.environment import TacticalEnvironment
from agents.enemy import EnemyAgent
from agents.player import PlayerAgent
from utils.logger import Logger


def main():
    """Main game loop."""
    # 1. Inisialisasi Pygame dan Environment
    pygame.init()

    # Setup Environment (bisa ubah seed=None untuk acak total)
    env = TacticalEnvironment(width=15, height=10, seed=32)
    random.seed(None)  # Reset random seed global agar AI tidak deterministik aneh

    # Setup Layar
    TILE_SIZE = 40  # Pastikan sama dengan di environment.py
    screen = pygame.display.set_mode((env.width * TILE_SIZE, env.height * TILE_SIZE))
    pygame.display.set_caption("Tactical AI: Minimax/AlphaBeta")
    clock = pygame.time.Clock()

    log = Logger("MainGame")

    # 2. Inisialisasi Agent
    # Ganti algorithm="MINIMAX" atau "ALPHABETA" atau "MCTS" sesuai kebutuhan
    playerAgent = PlayerAgent(env, algorithm="MINIMAX")
    enemyAgent = EnemyAgent(env)

    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Variabel untuk menampung hasil langkah (default: tidak terminal)
        step_result = (False, None)

        # --- Logika Giliran (Turn) ---
        if env.turn == "player":
            # 1. Minta Agent memilih langkah
            action_x, action_y = playerAgent.action()

            # 2. Eksekusi langkah di environment
            # env.step sekarang mengembalikan (is_terminal, reason)
            step_result = env.step((action_x, action_y))

            # log.info(f"Player moved to {(action_x, action_y)}")

        elif env.turn == "enemy":
            # 1. Minta Enemy memilih langkah
            action_x, action_y = enemyAgent.action()

            # 2. Eksekusi langkah
            step_result = env.step((action_x, action_y))

            # log.info("Enemy moved!")

        # --- Logika Game Over / Reset (PENTING) ---
        # Kita cek hasil dari step tadi
        is_terminal, reason = step_result

        if is_terminal:
            # Tampilkan pesan berdasarkan alasan terminal
            if reason == "goal":
                print("\n>>> VICTORY! You reached the goal. <<<\n")
                log.info("Result: WIN (Goal reached)")
            elif reason == "trap":
                print("\n>>> DEFEAT! You hit a trap. <<<\n")
                log.info("Result: LOSS (Hit trap)")
            elif reason == "caught":
                print("\n>>> DEFEAT! The enemy caught you. <<<\n")
                log.info("Result: LOSS (Caught by enemy)")

            # Beri jeda sebentar agar user bisa melihat posisi terakhir
            pygame.display.flip()
            pygame.time.delay(1000)  # Jeda 1 detik (1000 ms)

            # Lakukan Reset Environment
            env.reset()
            print("--- Game Reset ---\n")

        # --- Drawing (Gambar ke Layar) ---
        screen.fill((20, 20, 30))  # Warna Background
        env.draw(screen)
        pygame.display.flip()

        # Batasi FPS (biar tidak terlalu cepat/panas)
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()