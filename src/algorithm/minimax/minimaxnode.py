from environment.environment import TacticalEnvironment

# --- Skor Definitif ---
WIN_SCORE = 1_000_000
LOSE_SCORE = -1_000_000

def heuristic_evaluate(state: TacticalEnvironment) -> float:
    """
    Fungsi evaluasi statis untuk Minimax.
    Sama seperti AlphaBeta, kita menilai dari perspektif PLAYER (Maximizer).
    """
    player_pos = tuple(state.player_pos)
    enemy_pos = tuple(state.enemy_pos)
    goal_pos = state.goal

    # 1. Cek Terminal (Menang/Kalah)
    if player_pos == goal_pos:
        return WIN_SCORE
    
    if player_pos == enemy_pos or player_pos in state.traps:
        return LOSE_SCORE

    # 2. Heuristik (Jarak)
    # Semakin dekat ke goal = Bagus
    dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
    goal_score = (state.width + state.height - dist_to_goal) * 10 

    # Semakin jauh dari musuh = Bagus
    dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
    enemy_score = dist_to_enemy * 5

    return goal_score + enemy_score