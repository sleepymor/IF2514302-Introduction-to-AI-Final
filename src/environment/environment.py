# src/env/environment.py
import os
import copy
import pygame
import random
from collections import deque
from environment.generator import generate_environment

BG_COLOR = (20, 20, 30)
GRID_COLOR = (50, 50, 70)
MOVE_RANGE_COLOR = (80, 80, 200, 100)
ENEMY_MOVE_RANGE_COLOR = (70, 70, 150, 100)
TILE_SIZE = 40
FPS = 30


class TacticalEnvironment:
    """
    Turn-based tactical game environment with grid-based movement.
    Manages player, enemy, walls, traps, and goal placement.
    """

    def __init__(
        self,
        width=10,
        height=10,
        num_walls=10,
        num_traps=5,
        seed=None,
        use_assets=False,
        tile_dir="assets/tiles",
        char_dir="assets/characters",
    ):
        """
        Initialize the tactical environment.

          Args:
              width: Grid width in tiles
              height: Grid height in tiles
              num_walls: Number of wall obstacles to generate
              num_traps: Number of trap tiles to generate
              seed: Random seed for environment generation (None for random)
              use_assets: Whether to load image assets or use colored rectangles
              tile_dir: Directory path for tile textures
              char_dir: Directory path for character textures
        """
        self.width = width
        self.height = height
        self.num_walls = num_walls
        self.num_traps = num_traps
        self.seed = seed
        self.use_assets = use_assets
        self.tile_dir = tile_dir
        self.char_dir = char_dir

        self.textures = self.load_textures() if use_assets else None

        self.coordinate_font = (
            pygame.font.SysFont("consolas", 12) if pygame.font.get_init() else None
        )

        # Initialize game state
        self.reset()

    def load_textures(self):
        """
        Load all texture assets from disk with fallback to solid colors.
        Only called once during initialization to optimize performance.

        Returns:
            dict: Dictionary mapping entity names to pygame Surfaces
        """

        def load_image(folder, name, fallback_color, size=(TILE_SIZE, TILE_SIZE)):
            """
            Load a single image with fallback to colored surface.

            Args:
                folder: Directory containing the image
                name: Filename of the image
                fallback_color: RGB tuple to use if image not found
                size: Target size for the image

            Returns:
                pygame.Surface: Loaded and scaled image or colored surface
            """
            path = os.path.join(folder, name)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.scale(img, size)
            else:
                surf = pygame.Surface(size)
                surf.fill(fallback_color)
                return surf

        return {
            "player": load_image(self.char_dir, "player.png", (80, 180, 255)),
            "enemy": load_image(self.char_dir, "enemy.png", (255, 80, 80)),
            "goal": load_image(self.tile_dir, "goal.png", (80, 255, 120)),
            "wall": load_image(self.tile_dir, "wall.png", (100, 100, 100)),
            "trap": load_image(self.tile_dir, "trap.png", (200, 50, 200)),
        }

    def reset(self):
        """
        Reset the environment to initial state.
        Generates a new map layout and resets turn to player.
        """
        (
            self.grid,
            self.player_pos,
            self.enemy_pos,
            self.goal,
            self.walls,
            self.traps,
        ) = generate_environment(
            self.width, self.height, self.num_walls, self.num_traps, self.seed
        )
        self.turn = "player"

        self._cached_player_moves = None
        self._cached_enemy_moves = None

        if self.seed is not None:
            random.seed(None)

    def in_bounds(self, x, y):
        """
        Check if coordinates are within grid boundaries.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            bool: True if position is within grid bounds
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x, y):
        """
        Check if a tile is blocked by a wall.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            bool: True if position contains a wall
        """
        return (x, y) in self.walls

    # def get_move_range(self, pos, move_range=3):
    #   """
    #   Calculate all valid tiles within movement range using Manhattan distance.

    #   Args:
    #       pos: Current position as (x, y) tuple
    #       move_range: Maximum movement distance (default 3 for player, 2 for enemy)

    #   Returns:
    #       set: Set of (x, y) tuples representing reachable tiles
    #   """
    #   x, y = pos
    #   tiles = set()

    #   for dx in range(-move_range, move_range + 1):
    #     for dy in range(-move_range, move_range + 1):
    #       nx, ny = x + dx, y + dy
    #       manhattan_dist = abs(dx) + abs(dy)

    #       if self.in_bounds(nx, ny) and manhattan_dist <= move_range and not self.is_blocked(nx, ny):
    #         tiles.add((nx, ny))

    #       # euclidean_dist = math.dist((0, 0), (dx, dy))

    #       # if self.in_bounds(nx, ny) and euclidean_dist <= move_range and not self.is_blocked(nx, ny):
    #       #   tiles.add((nx, ny))

    #   tiles.discard(tuple(pos))
    #   return tiles

    def get_move_range(self, pos, move_range=3):

        queue = deque([(pos, 0)])
        visited = {tuple(pos)}

        reachable = set()

        while queue:
            (x, y), dist = queue.popleft()

            if dist > move_range:
                continue

            reachable.add((x, y))

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                if not self.in_bounds(nx, ny):
                    continue

                if (nx, ny) in visited:
                    continue

                if self.is_blocked(nx, ny):
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

        reachable.remove(tuple(pos))  # remove origin if you want
        return reachable

    def move_unit(self, pos, target):
        """
        Attempt to move a unit to target position.
        Validates that target is in bounds and not blocked.

        Args:
            pos: Current position as list [x, y]
            target: Target position as tuple (x, y)

        Returns:
            list: New position [x, y] if valid, otherwise original position
        """
        if not self.in_bounds(*target) or self.is_blocked(*target):
            return pos
        return list(target)

    def is_terminal(self):
        """
        Check if the game is in a terminal state (win/loss).

        Returns:
            tuple: (is_terminal: bool, reason: str or None)
        """
        if tuple(self.player_pos) == self.goal:
            return (True, "goal")
        elif tuple(self.player_pos) in self.traps:
            return (True, "trap")
        elif tuple(self.enemy_pos) == tuple(self.player_pos):
            return (True, "caught")
        return (False, None)

    def step(self, action, simulate=False):
        """
        Execute one turn of the game.
        Handles player/enemy movement and state transitions.

        Args:
          action: Target position tuple (x, y) or None to skip turn
          simulate: If True, don't auto-reset on terminal states (for MCTS/AI)
                    If False, reset immediately on win/loss (for gameplay)

        Returns:
          tuple or None: (is_terminal, reason) if simulate=True, else None
        """
        self._cached_player_moves = None
        self._cached_enemy_moves = None

        if self.turn == "player":
            # Player action
            if action is not None:
                move_tiles = self.get_move_range(self.player_pos)
                if action in move_tiles:
                    self.player_pos = self.move_unit(self.player_pos, action)

            is_terminal, reason = self.is_terminal()
            if is_terminal:
                if not simulate:
                    if reason == "goal":
                        print("You reached the goal! Resetting...")
                    elif reason == "trap":
                        print("You hit a trap! Resetting...")
                    self.reset()
                    return
                else:
                    return (True, reason)

            self.turn = "enemy"

        elif self.turn == "enemy":
            # Enemy action
            if action is not None:
                enemy_moves_tiles = self.get_move_range(self.enemy_pos)
                if action in enemy_moves_tiles:
                    self.enemy_pos = self.move_unit(self.enemy_pos, action)

            is_terminal, reason = self.is_terminal()
            if is_terminal:
                if not simulate:
                    print("Enemy caught you! Resetting...")
                    self.reset()
                    return
                else:
                    return (True, reason)

            self.turn = "player"

        if simulate:
            return (False, None)

    def get_valid_actions(self, unit="current"):
        """
        Get all valid actions for a unit (useful for AI/MCTS).

        Args:
            unit: 'current' (current turn), 'player', or 'enemy'

        Returns:
            set: Set of valid move positions
        """
        if unit == "current":
            unit = self.turn

        if unit == "player":
            return self.get_move_range(self.player_pos, move_range=3)
        elif unit == "enemy":
            return self.get_move_range(self.enemy_pos, move_range=3)

        return set()

    def draw(self, screen):
        """
        Render the entire game state to screen.
        Draws grid, tiles, movement ranges, and units.

        Args:
          screen: pygame Surface to draw on
        """

        # Draw grid and tiles
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)

                # Draw grid lines
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

                # Draw coordinate labels
                if self.coordinate_font:
                    text = self.coordinate_font.render(
                        f"{x},{y}", True, (100, 100, 150)
                    )
                    screen.blit(text, (x * TILE_SIZE + 2, y * TILE_SIZE + 2))

                if (x, y) in self.walls:
                    if self.use_assets:
                        screen.blit(self.textures["wall"], rect.topleft)
                    else:
                        pygame.draw.rect(screen, (100, 100, 100), rect)

                elif (x, y) in self.traps:
                    if self.use_assets:
                        screen.blit(self.textures["trap"], rect.topleft)
                    else:
                        pygame.draw.rect(screen, (200, 50, 200), rect)

                elif (x, y) == self.goal:
                    if self.use_assets:
                        screen.blit(self.textures["goal"], rect.topleft)
                    else:
                        pygame.draw.rect(screen, (80, 255, 120), rect)

        # Player movement range
        for mx, my in self.get_move_range(self.player_pos, move_range=2):
            rect = pygame.Rect(mx * TILE_SIZE, my * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            surf.fill(MOVE_RANGE_COLOR)
            screen.blit(surf, rect.topleft)

        # Enemy movement range
        for ex, ey in self.get_move_range(self.enemy_pos, move_range=2):
            rect = pygame.Rect(ex * TILE_SIZE, ey * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            surf.fill(ENEMY_MOVE_RANGE_COLOR)
            screen.blit(surf, rect.topleft)

        if self.use_assets:
            screen.blit(
                self.textures["player"],
                (self.player_pos[0] * TILE_SIZE, self.player_pos[1] * TILE_SIZE),
            )
            screen.blit(
                self.textures["enemy"],
                (self.enemy_pos[0] * TILE_SIZE, self.enemy_pos[1] * TILE_SIZE),
            )
        else:
            pygame.draw.rect(
                screen,
                (80, 180, 255),
                (
                    self.player_pos[0] * TILE_SIZE,
                    self.player_pos[1] * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE,
                ),
            )
            pygame.draw.rect(
                screen,
                (255, 80, 80),
                (
                    self.enemy_pos[0] * TILE_SIZE,
                    self.enemy_pos[1] * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE,
                ),
            )

    def clone(self):
        """
        Create a deep copy of the environment.
        Useful for MCTS simulations and game tree search.

        Returns:
          TacticalEnvironment: Independent copy of current state
        """

        cloned = TacticalEnvironment(
            width=self.width,
            height=self.height,
            num_walls=self.num_walls,
            num_traps=self.num_traps,
            seed=None,
            use_assets=False,
        )

        cloned.grid = copy.deepcopy(self.grid)
        cloned.player_pos = copy.deepcopy(self.player_pos)
        cloned.enemy_pos = copy.deepcopy(self.enemy_pos)
        cloned.goal = copy.deepcopy(self.goal)
        cloned.walls = copy.deepcopy(self.goal)
        cloned.traps = copy.deepcopy(self.traps)

        cloned.turn = copy.deepcopy(self.turn)
        cloned._cached_player_moves = None
        cloned._cached_enemy_moves = None

        return cloned
