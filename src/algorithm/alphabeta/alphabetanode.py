from environment.environment import TacticalEnvironment
import math

# --- Skor Definitif ---
# Gunakan angka yang sangat besar untuk menang/kalah
# agar nilainya selalu mengalahkan skor heuristik.
WIN_SCORE = 1_000_000
LOSE_SCORE = -1_000_000

def heuristic_evaluate(state: TacticalEnvironment) -> float:
    """
    Static evaluation function (heuristic) untuk sebuah game state.
    Fungsi ini *selalu* mengembalikan skor dari perspektif PLAYER.

    - Skor tinggi = Bagus untuk Player
    - Skor rendah = Bagus untuk Enemy (Buruk untuk Player)
    """
    player_pos = tuple(state.player_pos)
    enemy_pos = tuple(state.enemy_pos)
    goal_pos = state.goal

    # 1. Cek kondisi Terminal (Menang/Kalah)
    # Ini adalah kondisi paling penting dan harus dievaluasi terlebih dahulu.

    if player_pos == goal_pos:
        return WIN_SCORE  # Player menang

    if player_pos == enemy_pos or player_pos in state.traps:
        return LOSE_SCORE  # Player kalah

    # 2. Cek kondisi Non-Terminal (Heuristik)
    # Jika permainan belum berakhir, kita perkirakan skornya.

    # A. Skor Jarak ke Goal (Player)
    # Semakin dekat ke goal, semakin tinggi skornya.
    dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
    # (state.width + state.height) adalah jarak terjauh yang mungkin
    # Kita balik nilainya: (jarak_maks - jarak_sekarang)
    goal_score = (state.width + state.height - dist_to_goal) * 10  # Bobot: 10

    # B. Skor Jarak ke Musuh (Player)
    # Semakin jauh dari musuh, semakin tinggi skornya.
    dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
    
    if dist_to_enemy == 0:
        return LOSE_SCORE # Seharusnya sudah ditangkap di atas, tapi untuk keamanan

    enemy_score = dist_to_enemy * 5  # Bobot: 5

    # Skor total adalah gabungan dari keduanya
    # AI akan mencoba memaksimalkan nilai ini.
    heuristic_score = goal_score + enemy_score

    return heuristic_score