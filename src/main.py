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
        result = agent.action()
    except Exception:
        # Ensure worker returns a reasonable fallback action on error
        result = (tuple(env.player_pos), {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0}) if agent_type == "player" else (tuple(env.enemy_pos), {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0})

    # Normalize to tuple
    # result may be (action, metadata) or just action
    metadata = {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0}
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        action, meta = result
        if isinstance(meta, dict):
            metadata.update(meta)
    else:
        action = result

    if isinstance(action, list):
        action = tuple(action)

    return (action, metadata)


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
    env = TacticalEnvironment(width=20, height=15, seed=42, num_walls=75, num_traps=10)
    random.seed(None)

    # Screen setup
    TILE_SIZE = 40
    screen = pygame.display.set_mode((env.width * TILE_SIZE, env.height * TILE_SIZE))
    pygame.display.set_caption("Tactical AI")
    clock = pygame.time.Clock()

    log = Logger("MainGame")

    # Agents (we keep objects here for config, but worker will reconstruct env and agent)
    playerAgent = PlayerAgent(env, algorithm="MCTS",)
    enemyAgent = EnemyAgent(env)

    # Multiprocessing pool (create once)
    cpu_count = os.cpu_count() or 4
    pool = Pool(processes=min(cpu_count, 4))  # cap to reasonable number

    # Pending async jobs storage
    pending_future = None
    pending_owner = None
    pending_algorithm = None

    running = True
    paused = False
    step_requested = False
    try:
        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Toggle pause
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"Paused: {paused}")
                    # Step one turn when paused
                    elif event.key == pygame.K_n:
                        if paused:
                            step_requested = True
                    # Reset environment
                    elif event.key == pygame.K_r:
                        env.reset()
                        # clear agent caches
                        for attr in ("mcts_search", "alphabeta_search", "minimax_search"):
                            if hasattr(playerAgent, attr):
                                setattr(playerAgent, attr, None)
                        print("Environment reset")
                    # Algorithm switching: 1=MCTS, 2=AlphaBeta, 3=Minimax
                    elif event.key == pygame.K_1:
                        playerAgent.algorithm_choice = "MCTS"
                        # reset lazy-inits so new algorithm will initialize
                        playerAgent.mcts_search = None
                        playerAgent.alphabeta_search = None
                        playerAgent.minimax_search = None
                        print("Switched algorithm -> MCTS")
                    elif event.key == pygame.K_2:
                        playerAgent.algorithm_choice = "ALPHABETA"
                        playerAgent.mcts_search = None
                        playerAgent.alphabeta_search = None
                        playerAgent.minimax_search = None
                        print("Switched algorithm -> AlphaBeta")
                    elif event.key == pygame.K_3:
                        playerAgent.algorithm_choice = "MINIMAX"
                        playerAgent.mcts_search = None
                        playerAgent.alphabeta_search = None
                        playerAgent.minimax_search = None
                        print("Switched algorithm -> Minimax")

            # If there's no pending AI job and it's a turn, start one
            # When paused, don't start jobs unless user requested a single step
            can_start_job = not paused or step_requested
            if pending_future is None and can_start_job:
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
                        result = pending_future.get(timeout=1)
                    except Exception:
                        # If something went wrong, fallback to no-op movement
                        if pending_owner == "player":
                            result = (tuple(env.player_pos), {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0})
                        else:
                            result = (tuple(env.enemy_pos), {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0})

                    # result is (action, metadata)
                    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                        action, metadata = result
                    else:
                        action = result
                        metadata = {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0}

                    # Execute the action in the environment (simulate=False so gameplay resets properly)
                    env.step(action, simulate=True)

                    # attach metadata for HUD
                    # keep separate slots so enemy metadata doesn't overwrite player stats
                    if pending_owner == "player":
                        env.ai_metadata_player = metadata
                        env.ai_metadata = metadata
                    else:
                        env.ai_metadata_enemy = metadata

                    # If we were paused stepping, consume the step request now
                    if paused and step_requested:
                        step_requested = False

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
            # Update environment HUD data
            env.active_algorithm_name = playerAgent.algorithm_choice

            # pass demo state for HUD
            env.paused = paused
            env.step_requested = step_requested
            env.current_fps = clock.get_fps()

            # For visualization, ask the local enemy agent for its path and attach it
            try:
                env.enemy_intent_path = enemyAgent.peek_path()
            except Exception:
                env.enemy_intent_path = getattr(env, "enemy_intent_path", None)

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