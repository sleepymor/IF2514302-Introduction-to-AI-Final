import pygame
import random
import os
import time
from multiprocessing import Pool
from typing import Any, Dict, Tuple

from environment.environment import TacticalEnvironment
from agents.enemy import EnemyAgent
from agents.player import PlayerAgent
from utils.logger import Logger

# Worker function must be importable at module top-level (for Windows spawn)
def run_agent_action_worker(args: Tuple[str, str, Dict[str, Any]]):
    """
    Worker that receives (agent_type, algorithm_choice, env_snapshot)
    Reconstructs a TacticalEnvironment from the snapshot, creates the agent,
    runs the agent.action() and returns the action.
    """
    agent_type, algorithm_choice, snap = args

    # Reconstruct environment (avoid pygame assets)
    env = TacticalEnvironment(
        width=snap["width"],
        height=snap["height"],
        num_walls=snap.get("num_walls", 0),
        num_traps=snap.get("num_traps", 0),
        seed=None,
        use_assets=False,
    )
    env.grid = snap["grid"]
    env.player_pos = list(snap["player_pos"])
    env.enemy_pos = list(snap["enemy_pos"])
    env.goal = snap["goal"]
    env.walls = set(snap["walls"])
    env.traps = set(snap["traps"])
    env.turn = snap["turn"]

    # Run the chosen agent
    if agent_type == "player":
        agent = PlayerAgent(env, algorithm=algorithm_choice)
    else:
        agent = EnemyAgent(env)

    try:
        action = agent.action()
    except Exception:
        # Ensure worker returns a reasonable fallback action on error
        action = tuple(env.player_pos) if agent_type == "player" else tuple(env.enemy_pos)

    # Normalize to tuple
    if isinstance(action, list):
        action = tuple(action)
    return action


def make_env_snapshot(env: TacticalEnvironment) -> dict:
    """
    Create a small, picklable snapshot of the environment state for workers.
    Avoids passing pygame Surfaces or font objects.
    """
    return {
        "width": env.width,
        "height": env.height,
        "num_walls": env.num_walls,
        "num_traps": env.num_traps,
        "grid": env.grid,
        "player_pos": list(env.player_pos),
        "enemy_pos": list(env.enemy_pos),
        "goal": env.goal,
        "walls": list(env.walls),
        "traps": list(env.traps),
        "turn": env.turn,
    }


def main():
    """Main game loop with multiprocessing AI workers."""
    # Initialize Pygame
    pygame.init()

    # Setup Environment (seed can be None to randomize)
    env = TacticalEnvironment(width=15, height=10, seed=32)
    random.seed(None)

    # Screen setup
    TILE_SIZE = 40
    screen = pygame.display.set_mode((env.width * TILE_SIZE, env.height * TILE_SIZE))
    pygame.display.set_caption("Tactical AI: Minimax/AlphaBeta (multiprocessing)")
    clock = pygame.time.Clock()

    log = Logger("MainGame")

    # Agents (we keep objects here for config, but worker will reconstruct env and agent)
    playerAgent = PlayerAgent(env, algorithm="MINIMAX")
    enemyAgent = EnemyAgent(env)

    # Multiprocessing pool (create once)
    cpu_count = os.cpu_count() or 2
    pool = Pool(processes=min(cpu_count, 4))  # cap to reasonable number

    # Pending async jobs storage
    pending_future = None
    pending_owner = None
    pending_algorithm = None

    running = True
    try:
        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # If there's no pending AI job and it's a turn, start one
            if pending_future is None:
                if env.turn == "player":
                    # Start player's thinking in a worker
                    snap = make_env_snapshot(env)
                    args = ("player", playerAgent.algorithm_choice, snap)
                    pending_future = pool.apply_async(run_agent_action_worker, (args,))
                    pending_owner = "player"
                    pending_algorithm = playerAgent.algorithm_choice
                    log.info("Started player worker...")
                elif env.turn == "enemy":
                    snap = make_env_snapshot(env)
                    args = ("enemy", "", snap)
                    pending_future = pool.apply_async(run_agent_action_worker, (args,))
                    pending_owner = "enemy"
                    pending_algorithm = ""
                    log.info("Started enemy worker...")

            # If we have a pending job, poll it (non-blocking)
            if pending_future is not None:
                if pending_future.ready():
                    try:
                        action = pending_future.get(timeout=1)
                    except Exception:
                        # If something went wrong, fallback to no-op movement
                        if pending_owner == "player":
                            action = tuple(env.player_pos)
                        else:
                            action = tuple(env.enemy_pos)

                    # Execute the action in the environment (simulate=False so gameplay resets properly)
                    env.step(action, simulate=True)

                    # Clear pending
                    pending_future = None
                    pending_owner = None
                    pending_algorithm = None

                    # After step, check terminal and reset if necessary
                    is_terminal, reason = env.is_terminal()
                    if is_terminal:
                        if reason == "goal":
                            print("\n>>> VICTORY! You reached the goal. <<<\n")
                            log.info("Result: WIN (Goal reached)")
                        elif reason == "trap":
                            print("\n>>> DEFEAT! You hit a trap. <<<\n")
                            log.info("Result: LOSS (Hit trap)")
                        elif reason == "caught":
                            print("\n>>> DEFEAT! The enemy caught you. <<<\n")
                            log.info("Result: LOSS (Caught by enemy)")
                        # Brief pause so user can see final state
                        pygame.display.flip()
                        pygame.time.delay(1000)
                        env.reset()
                        print("--- Game Reset ---\n")

            # Drawing
            screen.fill((20, 20, 30))
            env.draw(screen)
            pygame.display.flip()

            # Cap FPS
            clock.tick(30)

    finally:
        # Clean shutdown: close pool
        try:
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
        pygame.quit()


if __name__ == "__main__":
    main()