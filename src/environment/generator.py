import random
import math

def generate_environment(width, height, num_walls, num_traps, seed=None):
    if seed is not None:
        random.seed(seed)

    grid = [[None for _ in range(width)] for _ in range(height)]
    player_pos = [0, 0]

    walls = set()
    for _ in range(num_walls):
        x, y = random.randrange(width), random.randrange(height)
        if (x, y) != tuple(player_pos):
            walls.add((x, y))

    traps = set()
    for _ in range(num_traps):
        x, y = random.randrange(width), random.randrange(height)
        if (x, y) != tuple(player_pos) and (x, y) not in walls:
            traps.add((x, y))

    # Goal far from player
    while True:
        gx, gy = random.randrange(width), random.randrange(height)
        dist = math.dist(player_pos, (gx, gy))
        if dist > max(width, height) / 2 and (gx, gy) not in walls:
            goal = (gx, gy)
            break

    # Enemy random position
    while True:
        ex, ey = random.randrange(width), random.randrange(height)
        if (ex, ey) not in walls and (ex, ey) != tuple(player_pos) and (ex, ey) != goal:
            enemy_pos = [ex, ey]
            break
        
    random.seed(None)

    return grid, player_pos, enemy_pos, goal, walls, traps