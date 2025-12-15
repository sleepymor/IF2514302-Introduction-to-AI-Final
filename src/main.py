"""Main game loop for Tactical AI with pygame visualization and multiprocessing.

Combines real-time game visualization with AI decision-making in separate
worker processes. Supports multiple algorithms (MCTS, AlphaBeta, Minimax)
with interactive controls and pause/step functionality for analysis.

Architecture:
- Worker functions isolated for multiprocessing
- Configuration parameters centralized for testability
- Event handling separated by input type (mouse, keyboard)
- Game loop orchestrates: events → AI decisions → rendering
"""

import os
import random
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Any, Dict, Optional, Tuple

import pygame

from agents.enemy import EnemyAgent
from agents.player import PlayerAgent
from environment.environment import TacticalEnvironment
from utils.logger import Logger


# =============================================================================
# CONFIGURATION - Immutable for consistency and testability
# =============================================================================


@dataclass(frozen=True)
class GameConfig:
    """Immutable configuration for game execution.

    Centralizes all game parameters for easy modification and testability.
    """

    environment_seed: Optional[int]
    player_algorithm: str
    grid_width: int
    grid_height: int
    num_walls: int
    num_traps: int
    tile_size: int
    max_processes: int


def create_game_config() -> GameConfig:
    """Create game configuration from testable parameters.

    All configurable parameters are defined here for easy modification
    without searching through the codebase.

    Returns:
        GameConfig with all game parameters.
    """
    # =========================================================================
    # TESTABLE PARAMETERS - Modify these to change game behavior
    # =========================================================================
    ENVIRONMENT_SEED = 32  # Specific seed or None for random
    # Change default algorithm here: "MCTS", "ALPHABETA", or "MINIMAX"
    PLAYER_ALGORITHM = "MCTS"
    GRID_WIDTH = 30
    GRID_HEIGHT = 15
    NUM_WALLS = 125
    NUM_TRAPS = 20
    TILE_SIZE = 40
    MAX_PROCESSES = 6  # Multiprocessing pool size
    # =========================================================================

    return GameConfig(
        environment_seed=ENVIRONMENT_SEED,
        player_algorithm=PLAYER_ALGORITHM,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        num_walls=NUM_WALLS,
        num_traps=NUM_TRAPS,
        tile_size=TILE_SIZE,
        max_processes=MAX_PROCESSES,
    )


def run_agent_action_worker(args: Tuple[str, str, Dict[str, Any]]):
    """Worker that executes agent action in separate process.

    Reconstructs environment from snapshot, creates agent, runs decision,
    and returns action with metadata.

    Args:
        args: Tuple of (agent_type, algorithm_choice, env_snapshot).
              - agent_type: "player" or "enemy"
              - algorithm_choice: Algorithm name for player
              - env_snapshot: Environment state dict

    Returns:
        Tuple[Tuple, Dict]: (action, metadata) for the agent's turn.
    """
    agent_type, algorithm_choice, snap = args

    # Reconstruct environment from snapshot (avoid pygame assets in workers)
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

    # Create and run agent
    if agent_type == "player":
        agent = PlayerAgent(env, algorithm=algorithm_choice)
    else:
        agent = EnemyAgent(env)

    try:
        result = agent.action()
    except Exception:
        # Fallback on error
        result = (
            (
                tuple(env.player_pos),
                {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0},
            )
            if agent_type == "player"
            else (
                tuple(env.enemy_pos),
                {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0},
            )
        )

    # Normalize result to (action, metadata) format
    metadata = {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0}
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        action, meta = result
        if isinstance(meta, dict):
            metadata.update(meta)
    else:
        action = result

    # Ensure action is tuple
    if isinstance(action, list):
        action = tuple(action)

    return (action, metadata)


def make_env_snapshot(env: TacticalEnvironment) -> dict:
    """Create picklable snapshot of environment for multiprocessing worker.

    Captures all state without pygame Surfaces or unpicklable objects.

    Args:
        env: TacticalEnvironment to snapshot.

    Returns:
        Dict with all environment state data.
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


# =============================================================================
# EVENT HANDLING - Separated by input type for clarity
# =============================================================================


def handle_quit_event(running: bool) -> bool:
    """Handle quit event.

    Args:
        running: Current running state.

    Returns:
        Updated running state.
    """
    return False


def handle_mouse_button_down(
    event: pygame.event.Event, env: TacticalEnvironment
) -> None:
    """Handle mouse button down for HUD dragging.

    Args:
        event: Pygame mouse button down event.
        env: Game environment.
    """
    if event.button == 1 and getattr(env, "hud_visible", True):
        hud_rect = getattr(env, "hud_rect", None)
        if hud_rect and hud_rect.collidepoint(event.pos):
            env.hud_dragging = True
            mx, my = event.pos
            env.hud_drag_offset = (env.hud_pos[0] - mx, env.hud_pos[1] - my)


def handle_mouse_button_up(event: pygame.event.Event, env: TacticalEnvironment) -> None:
    """Handle mouse button up to stop HUD dragging.

    Args:
        event: Pygame mouse button up event.
        env: Game environment.
    """
    if event.button == 1:
        env.hud_dragging = False


def handle_mouse_motion(
    event: pygame.event.Event, env: TacticalEnvironment, screen: pygame.Surface
) -> None:
    """Handle mouse motion for HUD dragging.

    Args:
        event: Pygame mouse motion event.
        env: Game environment.
        screen: Pygame display surface.
    """
    if getattr(env, "hud_dragging", False):
        mx, my = event.pos
        offx, offy = env.hud_drag_offset
        new_x = mx + offx
        new_y = my + offy

        # Clamp HUD to screen bounds
        sw, sh = screen.get_size()
        hr = getattr(env, "hud_rect", pygame.Rect(0, 0, 0, 0))
        hw, hh = hr.width, hr.height
        new_x = max(0, min(new_x, sw - hw))
        new_y = max(0, min(new_y, sh - hh))
        env.hud_pos[0] = new_x
        env.hud_pos[1] = new_y


def reset_algorithm_caches(player_agent: PlayerAgent) -> None:
    """Reset all algorithm search caches.

    Args:
        player_agent: Player agent whose caches to reset.
    """
    for attr in ("mcts_search", "alphabeta_search", "minimax_search"):
        if hasattr(player_agent, attr):
            setattr(player_agent, attr, None)


def switch_player_algorithm(player_agent: PlayerAgent, algorithm_name: str) -> None:
    """Switch player algorithm and reset caches.

    Args:
        player_agent: Player agent to reconfigure.
        algorithm_name: Name of algorithm to switch to.
    """
    player_agent.algorithm_choice = algorithm_name
    reset_algorithm_caches(player_agent)
    print(f"Switched algorithm → {algorithm_name}")


def handle_keyboard_input(
    event: pygame.event.Event,
    player_agent: PlayerAgent,
    env: TacticalEnvironment,
) -> Tuple[bool, bool]:
    """Handle keyboard input for game controls.

    Args:
        event: Pygame keyboard event.
        player_agent: Player agent for algorithm switching.
        env: Game environment for reset.

    Returns:
        Tuple of (paused, step_requested).
    """
    paused = False
    step_requested = False

    if event.key == pygame.K_SPACE:
        paused = True  # Signal to toggle
    elif event.key == pygame.K_n:
        step_requested = True
    elif event.key == pygame.K_r:
        env.reset()
        reset_algorithm_caches(player_agent)
        print("Environment reset")
    elif event.key == pygame.K_1:
        switch_player_algorithm(player_agent, "MCTS")
    elif event.key == pygame.K_2:
        switch_player_algorithm(player_agent, "ALPHABETA")
    elif event.key == pygame.K_3:
        switch_player_algorithm(player_agent, "MINIMAX")
    elif event.key == pygame.K_h:
        env.hud_visible = not getattr(env, "hud_visible", True)
        print(f"HUD visible: {env.hud_visible}")

    return paused, step_requested


def process_events(
    running: bool,
    paused: bool,
    player_agent: PlayerAgent,
    env: TacticalEnvironment,
    screen: pygame.Surface,
) -> Tuple[bool, bool, bool]:
    """Process all pygame events.

    Args:
        running: Current running state.
        paused: Current paused state.
        player_agent: Player agent for controls.
        env: Game environment.
        screen: Pygame display surface.

    Returns:
        Tuple of (running, paused, step_requested).
    """
    step_requested = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            handle_mouse_button_down(event, env)
        elif event.type == pygame.MOUSEBUTTONUP:
            handle_mouse_button_up(event, env)
        elif event.type == pygame.MOUSEMOTION:
            handle_mouse_motion(event, env, screen)
        elif event.type == pygame.KEYDOWN:
            pause_toggle, step = handle_keyboard_input(event, player_agent, env)
            if pause_toggle:
                paused = not paused
                print(f"Paused: {paused}")
            step_requested = step

    return running, paused, step_requested


# =============================================================================
# AI DECISION MAKING - Isolated logic for clarity
# =============================================================================


def submit_agent_job(
    env: TacticalEnvironment,
    player_agent: PlayerAgent,
    pool: Pool,
    log: Logger,
) -> Tuple[Any, str]:
    """Submit agent decision job to worker pool.

    Args:
        env: Game environment.
        player_agent: Player agent for algorithm info.
        pool: Multiprocessing pool.
        log: Logger instance.

    Returns:
        Tuple of (pending_future, pending_owner).
    """
    snap = make_env_snapshot(env)

    if env.turn == "player":
        args = ("player", player_agent.algorithm_choice, snap)
        pending_future = pool.apply_async(run_agent_action_worker, (args,))
        log.info("Started player worker...")
        return pending_future, "player"

    elif env.turn == "enemy":
        args = ("enemy", "", snap)
        pending_future = pool.apply_async(run_agent_action_worker, (args,))
        log.info("Started enemy worker...")
        return pending_future, "enemy"

    return None, None


def process_worker_result(
    env: TacticalEnvironment,
    pending_owner: str,
    result: Tuple,
) -> Dict[str, Any]:
    """Process completed worker job result.

    Args:
        env: Game environment.
        pending_owner: "player" or "enemy".
        result: Result tuple from worker.

    Returns:
        Metadata dictionary.
    """
    # Parse result (action, metadata)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        action, metadata = result
    else:
        action = result
        metadata = {"nodes_visited": 0, "thinking_time": 0.0, "win_probability": 0.0}

    # Execute action
    env.step(action, simulate=True)

    # Store metadata for HUD
    if pending_owner == "player":
        env.ai_metadata_player = metadata
        env.ai_metadata = metadata
    else:
        env.ai_metadata_enemy = metadata

    return metadata


def handle_game_terminal(env: TacticalEnvironment, log: Logger) -> None:
    """Handle game over condition.

    Args:
        env: Game environment.
        log: Logger instance.
    """
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

        # Show final game state briefly
        pygame.display.flip()
        pygame.time.delay(1000)
        env.reset()
        print("--- Game Reset ---\n")


# =============================================================================
# RENDERING - Display and HUD updates
# =============================================================================


def update_hud_data(
    env: TacticalEnvironment,
    player_agent: PlayerAgent,
    clock: pygame.time.Clock,
    paused: bool,
    step_requested: bool,
) -> None:
    """Update HUD data before rendering.

    Args:
        env: Game environment.
        player_agent: Player agent for algorithm info.
        clock: Pygame clock for FPS.
        paused: Current paused state.
        step_requested: Current step request state.
    """
    env.active_algorithm_name = player_agent.algorithm_choice
    env.paused = paused
    env.step_requested = step_requested
    env.current_fps = clock.get_fps()


def render_frame(
    env: TacticalEnvironment, screen: pygame.Surface, clock: pygame.time.Clock
) -> None:
    """Render game frame.

    Args:
        env: Game environment.
        screen: Pygame display surface.
        clock: Pygame clock for framerate control.
    """
    screen.fill((20, 20, 30))
    env.draw(screen)
    pygame.display.flip()
    clock.tick(30)  # 30 FPS cap


# =============================================================================
# MAIN GAME LOOP
# =============================================================================


def main():
    """Main game loop with multiprocessing AI workers."""
    # Create configuration
    config = create_game_config()

    # Initialize Pygame
    pygame.init()

    # Setup Environment
    env = TacticalEnvironment(
        width=config.grid_width,
        height=config.grid_height,
        seed=config.environment_seed,
        num_walls=config.num_walls,
        num_traps=config.num_traps,
    )
    random.seed(None)

    # Setup Display
    screen_width = env.width * config.tile_size
    screen_height = env.height * config.tile_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Tactical AI - Game Loop")
    clock = pygame.time.Clock()

    log = Logger("MainGame")

    # Initialize Agents
    player_agent = PlayerAgent(env, algorithm=config.player_algorithm)
    enemy_agent = EnemyAgent(env)

    # Setup Multiprocessing Pool
    cpu_count = os.cpu_count() or 4
    pool = Pool(processes=min(cpu_count, config.max_processes))

    # Game State Variables
    pending_future = None
    pending_owner = None
    running = True
    paused = False
    step_requested = False

    # Main Game Loop
    try:
        while running:
            # Event Processing
            running, paused, step_request = process_events(
                running, paused, player_agent, env, screen
            )
            step_requested = step_request or step_requested

            # AI Decision Making
            can_start_job = not paused or step_requested
            if pending_future is None and can_start_job:
                pending_future, pending_owner = submit_agent_job(
                    env, player_agent, pool, log
                )

            # Process Completed Jobs
            if pending_future is not None and pending_future.ready():
                try:
                    result = pending_future.get(timeout=1)
                except Exception:
                    # Fallback on error
                    result = (
                        (
                            tuple(env.player_pos),
                            {
                                "nodes_visited": 0,
                                "thinking_time": 0.0,
                                "win_probability": 0.0,
                            },
                        )
                        if pending_owner == "player"
                        else (
                            tuple(env.enemy_pos),
                            {
                                "nodes_visited": 0,
                                "thinking_time": 0.0,
                                "win_probability": 0.0,
                            },
                        )
                    )

                # Process result and update environment
                process_worker_result(env, pending_owner, result)

                # Clear step request if we were paused
                if paused and step_requested:
                    step_requested = False

                # Clear pending job
                pending_future = None
                pending_owner = None

                # Check for game over
                handle_game_terminal(env, log)

            # Rendering
            update_hud_data(env, player_agent, clock, paused, step_requested)

            # Get enemy pathfinding for visualization
            try:
                env.enemy_intent_path = enemy_agent.peek_path()
            except Exception:
                env.enemy_intent_path = getattr(env, "enemy_intent_path", None)
                
            try:
                env.player_intent_path = player_agent.peek_path_to_goal()
            except Exception:
                env.player_intent_path = getattr(env, "player_intent_path", None)

            render_frame(env, screen, clock)

    finally:
        # Cleanup: Close multiprocessing pool and pygame
        try:
            pool.close()
            pool.join()
        except Exception:
            pool.terminate()
        pygame.quit()


if __name__ == "__main__":
    main()
