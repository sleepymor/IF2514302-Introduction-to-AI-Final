import math
import random
from environment.environment import TacticalEnvironment

# --- Skor Definitif ---
WIN_SCORE = 1_000_000
LOSE_SCORE = -1_000_000

def heuristic_evaluate(state: TacticalEnvironment, depth: int = 0) -> float:
    player_pos = tuple(state.player_pos)
    enemy_pos = tuple(state.enemy_pos)
    goal_pos = state.goal

    # --- 1. Terminal State ---
    if player_pos == goal_pos:
        # Bonus besar untuk kedalaman (depth) agar lari sekencang mungkin
        return WIN_SCORE + (depth * 5000)
    
    if player_pos == enemy_pos or player_pos in state.traps:
        return LOSE_SCORE

    score = 0.0
    
    # --- 2. LOGIKA BARU: Jarak ke Goal (Obsesif) ---
    dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
    
    # Kita beri bobot SANGAT BESAR (200) agar dia tidak peduli rintangan kecil
    score -= dist_to_goal * 200 

    # --- 3. Rasa Takut (Hanya jika kritis) ---
    dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
    
    # UBAH DISINI: Hanya takut jika jarak <= 2 (Benar-benar dekat)
    # Jika jarak 3 atau 4, AI akan mengabaikan musuh demi mencapai goal.
    if dist_to_enemy <= 2:
        score -= 10000.0 / (dist_to_enemy + 0.1)

    # --- 4. Anti-Stuck (Random lebih besar) ---
    # Perbesar nilai random agar dia mau mencoba langkah "aneh" saat buntu
    score += random.uniform(0, 2.0)

    return score