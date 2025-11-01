# src/env/environment.py
import pygame
import os
from environment.generator import generate_environment

BG_COLOR = (20, 20, 30)
GRID_COLOR = (50, 50, 70)
MOVE_RANGE_COLOR = (80, 80, 200, 100)
TILE_SIZE = 40
FPS = 30

class TacticalEnvironment:
    def __init__(self, width=10, height=10, num_walls=10, num_traps=5, seed=None, use_assets=False, tile_dir="assets/tiles", char_dir="assets/characters"):
        self.width = width
        self.height = height
        self.num_walls = num_walls
        self.num_traps = num_traps
        self.seed = seed
        self.use_assets = use_assets
        self.tile_dir = tile_dir
        self.char_dir = char_dir
        self.textures = self.load_textures() if use_assets else None
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
        (self.grid, self.player_pos, self.enemy_pos, self.goal, 
         self.walls, self.traps) = generate_environment(
            self.width, self.height, self.num_walls, self.num_traps, self.seed
        )
        self.turn = 'player'

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x, y):
        return (x, y) in self.walls

    def get_move_range(self, pos, move_range=3):
        x, y = pos
        tiles = set()
        for dx in range(-move_range, move_range + 1):
            for dy in range(-move_range, move_range + 1):
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny) and abs(dx) + abs(dy) <= move_range and not self.is_blocked(nx, ny):
                    tiles.add((nx, ny))
        return tiles

    def move_unit(self, pos, target):
        if not self.in_bounds(*target) or self.is_blocked(*target):
            return pos
        return list(target)

    def step(self, action):
        if self.turn == 'player':
            move_tiles = self.get_move_range(self.player_pos)
            if action in move_tiles:
                self.player_pos = self.move_unit(self.player_pos, action)

                if tuple(self.player_pos) == self.goal:
                    print("You reached the goal! Resetting...")
                    self.reset()
                elif tuple(self.player_pos) in self.traps:
                    print("You hit a trap! Resetting...")
                    self.reset()

                self.turn = 'enemy'

        elif self.turn == 'enemy':
            ex, ey = self.enemy_pos
            px, py = self.player_pos
            if ex < px: ex += 1
            elif ex > px: ex -= 1
            if ey < py: ey += 1
            elif ey > py: ey -= 1
            if not self.is_blocked(ex, ey):
                self.enemy_pos = [ex, ey]

            if tuple(self.enemy_pos) == tuple(self.player_pos):
                print("Enemy caught you! Resetting...")
                self.reset()

            self.turn = 'player'

    def draw(self, screen):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

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

        for mx, my in self.get_move_range(self.player_pos):
            rect = pygame.Rect(mx*TILE_SIZE, my*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            surf.fill(MOVE_RANGE_COLOR)
            screen.blit(surf, rect.topleft)

        if self.use_assets:
            screen.blit(self.textures['player'], (self.player_pos[0]*TILE_SIZE, self.player_pos[1]*TILE_SIZE))
            screen.blit(self.textures['enemy'], (self.enemy_pos[0]*TILE_SIZE, self.enemy_pos[1]*TILE_SIZE))
        else:
            pygame.draw.rect(screen, (80, 180, 255), (self.player_pos[0]*TILE_SIZE, self.player_pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(screen, (255, 80, 80), (self.enemy_pos[0]*TILE_SIZE, self.enemy_pos[1]*TILE_SIZE, TILE_SIZE, TILE_SIZE))
