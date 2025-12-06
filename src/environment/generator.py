import random
import math


def generate_environment(width, height, num_walls, num_traps, seed=None):
    """
    Generate Tactical Grid-based Environment

    Steps:

    1. Use Random Seed if one not Picked
        -If user provide seed value, random number generator will initialize with the seed given.
        This seed ensure reproducible map generation.

    2. Set Player Position at [0,0]
        -The Player Will Always spawn at top-left corner of the grid.
        This coordinate is fixed and cant be overwritten by walls, trap, goal, or
        enemy placement.

    3. Build Empty Grid
        - A 2D list (height x width) filled with " " is created. Each cell
        starts as empty, and other objects (Walls, Traps, Goal, Player, Enemy)
        will later replace these default values.
        - The grid uses the format grid[y][x], meaning row-major order.

    4. Place Walls
        - Walls are placed randomly on valid coordinates.
        - A coordinate is considered valid if:
            It is inside the grid
            It is not already occupied by the player or another wall
        - Walls are stored in a set for quick lookup and to avoid duplicates.
        - These act as obstacles and block movement.

    5. Place Traps
        - Traps are also placed randomly with similar rules:
            Cannot overlap with the player
            Cannot overlap with walls
        - Traps use a set to prevent duplicates.
        - These tiles typically cause damage or penalties in the game.

    6. Place Goals
        - The goal is placed at a location far from the player's starting point.
        - Distance is checked using math.dist() (Euclidean distance).
        - The goal cannot overlap walls, the player, or traps.
        - Ensures the player has to navigate through the grid to reach it.

    7. Place Enemy
        - The enemy is placed last, ensuring no conflict with:
            Player starting position
            Walls
            Traps
            Goal
       - Enemy position is stored as a list [x, y] for mutability (movement).


    This set of Code will Return:
    grid = 2D List
    player_pos = list
    enemy_post = list
    goal = tuple
    walls = set of tuple
    traps = set of tuple

    How this Script Works:

    """

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

    while True:
        gx, gy = random.randrange(width), random.randrange(height)
        dist = math.dist(player_pos, (gx, gy))
        if dist > max(width, height) / 2 and (gx, gy) not in walls:
            goal = (gx, gy)
            break

    while True:
        ex, ey = random.randrange(width), random.randrange(height)
        if (ex, ey) not in walls and (ex, ey) != tuple(player_pos) and (ex, ey) != goal:
            enemy_pos = [ex, ey]
            break

    random.seed(None)

    return grid, player_pos, enemy_pos, goal, walls, traps
