from environment.environment import TacticalEnvironment

class MinimaxNode:
    """
    Node untuk algoritma Minimax.
    Menggunakan logika evaluasi yang sama dengan AlphaBeta:
    - Poin PLUS (+) besar jika mendekat ke Goal.
    - Poin MINUS (-) besar jika dekat dengan Enemy.
    """
    
    WIN_SCORE = 1_000_000
    LOSE_SCORE = -1_000_000

    def __init__(self, state: TacticalEnvironment, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

    def evaluate(self) -> float:
        player_pos = tuple(self.state.player_pos)
        enemy_pos = tuple(self.state.enemy_pos)
        goal_pos = self.state.goal

        # --- 1. Cek Terminal State (Menang/Kalah) ---
        if player_pos == goal_pos:
            return self.WIN_SCORE
        if player_pos == enemy_pos or player_pos in self.state.traps:
            return self.LOSE_SCORE

        score = 0.0
        
        # --- 2. LOGIKA UTAMA: Semakin Dekat Goal = Semakin Tinggi Skor (+) ---
        dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        
        # Hitung jarak terjauh yang mungkin (Diagonal Peta)
        max_dist = self.state.width + self.state.height
        
        # Rumus: (Jarak Maksimal - Jarak Sekarang) * Bobot
        # Ini memastikan gerakan mendekati goal selalu memberi tambahan poin positif.
        score += (max_dist - dist_to_goal) * 20 

        # --- 3. Hindari Musuh (Repulsion Field) ---
        dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
        safe_dist_enemy = max(dist_to_enemy, 0.5)
        
        # Pengurangan nilai drastis jika musuh terlalu dekat
        score -= 2000.0 / safe_dist_enemy

        # --- 4. Bonus Mobilitas ---
        # Agar AI tidak terjebak di jalan buntu
        num_legal_moves = len(self.get_legal_actions())
        score += num_legal_moves * 15

        return score

    def get_legal_actions(self):
        return self.state.get_valid_actions(unit='current')

    def is_terminal(self):
        is_term, _ = self.state.is_terminal()
        return is_term