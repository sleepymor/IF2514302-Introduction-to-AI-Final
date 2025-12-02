# src/environment/environment.py
import os
import copy
import pygame
import random
from collections import deque
from environment.generator import generate_environment

BG_COLOR = (20, 20, 30)
GRID_COLOR = (50, 50, 70)
MOVE_RANGE_COLOR = (80, 80, 200, 100)
ENEMY_MOVE_RANGE_COLOR = (150, 26, 26, 100)
TILE_SIZE = 40
FPS = 60


class TacticalEnvironment:
    """
    Turn-based tactical game environment with grid-based movement.
    Manages player, enemy, walls, traps, and goal placement.
    """
    
    def __init__(self, width=10, height=10, num_walls=10, num_traps=5, seed=None, use_assets=False, tile_dir="assets/tiles", char_dir="assets/characters"):
        self.width = width
        self.height = height

        self.num_walls = num_walls
        self.num_traps = num_traps
        self.seed = seed

        self.use_assets = use_assets
        self.tile_dir = tile_dir
        self.char_dir = char_dir

        # Animation draw positions
        self.player_draw_pos = None
        self.enemy_draw_pos = None


        self.turn_counter = 0

        self.textures = self.load_textures() if use_assets else None

        self.coordinate_font = pygame.font.SysFont("consolas", 12) if pygame.font.get_init() else None

        # Initialize game state
        self.reset()

    def load_textures(self):
        def load_image(folder, name, fallback_color, size=(TILE_SIZE, TILE_SIZE)):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.scale(img, size)
            else:
                surf = pygame.Surface(size)
                surf.fill(fallback_color)
                return surf

        return {
            'player': load_image(self.char_dir, 'player.png', (80, 180, 255)),
            'enemy': load_image(self.char_dir, 'enemy.png', (255, 80, 80)),
            'goal': load_image(self.tile_dir, 'goal.png', (80, 255, 120)),
            'wall': load_image(self.tile_dir, 'wall.png', (100, 100, 100)),
            'trap': load_image(self.tile_dir, 'trap.png', (200, 50, 200)),
        }

    def reset(self):
        """
        Reset the environment to initial state.
        """
        (self.grid, self.player_pos, self.enemy_pos, self.goal, 
         self.walls, self.traps) = generate_environment(
             self.width, self.height, self.num_walls, self.num_traps, self.seed
        )
        self.turn = 'player'

        self._cached_player_moves = None
        self._cached_enemy_moves = None
        
        if self.seed is not None:
            random.seed(None)

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x, y):
        return (x, y) in self.walls

    def get_move_range(self, pos, move_range=3):
        """
        Calculate all valid tiles within movement range using BFS.
        """
        queue = deque([(pos, 0)])
        visited = {tuple(pos)}
        reachable = set()

        while queue:
            (x, y), dist = queue.popleft()

            if dist > move_range:
                continue

            reachable.add((x, y))

            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x + dx, y + dy

                if not self.in_bounds(nx, ny):
                    continue

                if (nx, ny) in visited:
                    continue

                if self.is_blocked(nx, ny):
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

        reachable.remove(tuple(pos))  # remove origin
        return reachable


    def move_unit(self, pos, target):
        if not self.in_bounds(*target) or self.is_blocked(*target):
            return pos
        return list(target)

    def is_terminal(self):
        """
        Check if the game is in a terminal state (win/loss).
        """
        if tuple(self.player_pos) == self.goal:
            return (True, "goal")
        elif tuple(self.player_pos) in self.traps:
            return (True, "trap")
        elif tuple(self.enemy_pos) == tuple(self.player_pos):
            return (True, "caught")
        return (False, None)

    def spawn_trap(self):
        empty_tiles = [
          (x, y) 
          for x in range(self.width) 
          for y in range(self.height)
          if (x, y) != tuple(self.player_pos) 
          and (x, y) != tuple(self.enemy_pos) 
          and (x, y) != self.goal 
          and (x, y) not in self.walls 
          and (x, y) not in self.traps
        ]
        
        if not empty_tiles:
           return
        
        new_trap = random.choice(empty_tiles)
        self.traps.add(new_trap)
          

          
        

    def step(self, action, simulate=False):
        """
        Execute one turn of the game.
        Handles player/enemy movement and state transitions.
        
        UPDATE: Menghapus auto-reset dan print spam agar kompatibel dengan AI Simulation.
        """
        self._cached_player_moves = None
        self._cached_enemy_moves = None

        if self.turn == 'player':
            # Player action
            if action is not None:
                move_tiles = self.get_move_range(self.player_pos)
                if tuple(action) in move_tiles:
                    self.player_pos = self.move_unit(self.player_pos, action)

            # Cek Terminal setelah bergerak
            is_terminal, reason = self.is_terminal()
            if is_terminal:
                # LANGSUNG RETURN STATUS, JANGAN RESET DI SINI
                return (True, reason)

            self.turn = 'enemy'


        elif self.turn == 'enemy':
            # Enemy action
            if action is not None:
                enemy_moves_tiles = self.get_move_range(self.enemy_pos)
                if tuple(action) in enemy_moves_tiles:
                    self.enemy_pos = self.move_unit(self.enemy_pos, action)

            # Cek Terminal setelah bergerak
            is_terminal, reason = self.is_terminal()
            if is_terminal:
                # LANGSUNG RETURN STATUS, JANGAN RESET DI SINI
                return (True, reason)
            
            self.turn = 'player'
        
<<<<<<< HEAD
        # Jika game belum selesai
=======
        Returns:
          tuple or None: (is_terminal, reason) if simulate=True, else None
      """
      self._cached_player_moves = None
      self._cached_enemy_moves = None

      if self.turn == 'player' and action is not None and not simulate:
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

        self.turn = 'enemy'

      elif self.turn == 'enemy':
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
          
        self.turn = 'player'

        self.turn_counter += 1

        if self.turn_counter % 3 == 0:
            print("Trap spawned on turn", self.turn_counter)
            self.spawn_trap()
      return

      if simulate: 
>>>>>>> 698d9f507b39de653514052a5ca4174be31f9edf
        return (False, None)
    
    def get_valid_actions(self, unit='current'):
        if unit == 'current':
            unit = self.turn
        
        if unit == 'player':
            return self.get_move_range(self.player_pos, move_range=3)
        elif unit == 'enemy':
            return self.get_move_range(self.enemy_pos, move_range=3)
        
        return set()


    def draw(self, screen):
        """
        Render the entire game state to screen.
        """
        # Draw grid and tiles
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)

<<<<<<< HEAD
                # Draw grid lines
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

                # Draw coordinate labels
                if self.coordinate_font:
                    text = self.coordinate_font.render(f"{x},{y}", True, (100, 100, 150))
                    screen.blit(text, (x * TILE_SIZE + 2, y * TILE_SIZE + 2))
                
                if (x, y) in self.walls:
                    if self.use_assets:
                        screen.blit(self.textures['wall'], rect.topleft)
                    else:
                        pygame.draw.rect(screen, (100, 100, 100), rect)

                elif (x, y) in self.traps:
                    if self.use_assets:
                        screen.blit(self.textures['trap'], rect.topleft)
                    else:
                        pygame.draw.rect(screen, (200, 50, 200), rect)
                    
                elif (x, y) == self.goal:
                    if self.use_assets:
                        screen.blit(self.textures['goal'], rect.topleft)
                    else:
                        pygame.draw.rect(screen, (80, 255, 120), rect)

        # Player movement range
        for mx, my in self.get_move_range(self.player_pos):
            rect = pygame.Rect(mx*TILE_SIZE, my*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            surf.fill(MOVE_RANGE_COLOR)
            screen.blit(surf, rect.topleft)

        # Enemy movement range
        for ex, ey in self.get_move_range(self.enemy_pos, move_range=2):
            rect = pygame.Rect(ex*TILE_SIZE, ey*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            surf.fill(ENEMY_MOVE_RANGE_COLOR)
            screen.blit(surf, rect.topleft)
=======
        # Draw grid and tiles
      for y in range(self.height):
          for x in range(self.width):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)

            # Draw grid lines
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

            # Draw coordinate labels
            if self.coordinate_font:
              text = self.coordinate_font.render(f"{x},{y}", True, (100, 100, 150))
              screen.blit(text, (x * TILE_SIZE + 2, y * TILE_SIZE + 2))
            

            if (x, y) in self.walls:
              if self.use_assets:
                screen.blit(self.textures['wall'], rect.topleft)
              else:
                pygame.draw.rect(screen, (100, 100, 100), rect)

            elif (x, y) in self.traps:
              if self.use_assets:
                screen.blit(self.textures['trap'], rect.topleft)
              else:
                pygame.draw.rect(screen, (200, 50, 200), rect)
              
            elif (x, y) == self.goal:
              if self.use_assets:
                screen.blit(self.textures['goal'], rect.topleft)
              else:
                pygame.draw.rect(screen, (80, 255, 120), rect)
>>>>>>> 698d9f507b39de653514052a5ca4174be31f9edf

        if self.use_assets:
            screen.blit(self.textures['player'], (self.player_pos[0]*TILE_SIZE, self.player_pos[1]*TILE_SIZE))
            screen.blit(self.textures['enemy'], (self.enemy_pos[0]*TILE_SIZE, self.enemy_pos[1]*TILE_SIZE))
        else:
            pygame.draw.rect(screen, (80, 180, 255), (self.player_pos[0]*TILE_SIZE, self.player_pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(screen, (255, 80, 80), (self.enemy_pos[0]*TILE_SIZE, self.enemy_pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE))

    def clone(self):
        """
        Create a deep copy of the environment.
        """
        # 1. Shallow copy
        cloned = copy.copy(self)

        # 2. Deep copy data dinamis
        cloned.player_pos = list(self.player_pos)
        cloned.enemy_pos = list(self.enemy_pos)
        
        # 3. Reset cache di klon
        cloned._cached_player_moves = None
        cloned._cached_enemy_moves = None

        # Copy turn
        cloned.turn = self.turn

        return cloned