import random
import math

def carve_path(start, goal, noise=0.4):
    x, y = start
    gx, gy = goal
    path = {(x, y)}

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while (x, y) != (gx, gy):
        moves = []

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if manhattan((nx, ny), goal) <= manhattan((x, y), goal):
                moves.append((dx, dy))

        # --- Add noise ---
        if random.random() < noise:
            dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        else:
            dx, dy = random.choice(moves)

        x += dx
        y += dy
        path.add((x, y))

    return path


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
    start = tuple(player_pos)

    # --- Place Goal ---
    while True:
        gx, gy = random.randrange(width), random.randrange(height)
        if math.dist(start, (gx, gy)) > 0.6 * math.hypot(width, height):
            goal = (gx, gy)
            break

    # --- Place Enemy FIRST (FAR FROM PLAYER) ---
    min_enemy_dist = 0.4 * math.hypot(width, height)

    while True:
        ex, ey = random.randrange(width), random.randrange(height)
        if (
            (ex, ey) != start
            and (ex, ey) != goal
            and math.dist((ex, ey), start) >= min_enemy_dist
        ):
            enemy_pos = [ex, ey]
            break


    enemy = tuple(enemy_pos)

    # --- Carve Mandatory Paths ---
    path_player_goal = carve_path(start, goal)
    path_enemy_player = carve_path(enemy, start)
    path_enemy_goal = carve_path(enemy, goal)

    protected_path = (
        path_player_goal
        | path_enemy_player
        | path_enemy_goal
    )

    forbidden_positions = set(protected_path)
    forbidden_positions.add(start)
    forbidden_positions.add(goal)
    forbidden_positions.add(enemy)

    # --- Place Walls ---
    walls = set()
    while len(walls) < num_walls:
        x, y = random.randrange(width), random.randrange(height)
        if (x, y) not in forbidden_positions:
            walls.add((x, y))

    # --- Place Traps ---
    traps = set()
    while len(traps) < num_traps:
        x, y = random.randrange(width), random.randrange(height)
        if (
            (x, y) not in forbidden_positions
            and (x, y) not in walls
        ):
            traps.add((x, y))

    random.seed(None)
    return grid, player_pos, enemy_pos, goal, walls, traps