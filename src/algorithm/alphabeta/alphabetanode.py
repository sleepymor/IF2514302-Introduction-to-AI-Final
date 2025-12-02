from environment.environment import TacticalEnvironment

class AlphaBetaNode:
    """
    Node Alpha-Beta yang diperbaiki dengan logika 'Repulsion Field'.
    Player akan mencoba menghindari musuh dari jarak jauh dan mencari jalan aman.
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

        # --- 1. Cek Menang/Kalah ---
        if player_pos == goal_pos:
            return self.WIN_SCORE
        
        # Hanya kalah jika TERTANGKAP (posisi sama).
        # Jangan anggap kalah jika cuma dekat, atau AI akan menyerah duluan.
        if player_pos == enemy_pos or player_pos in self.state.traps:
            return self.LOSE_SCORE

        score = 0.0
        
        # --- 2. Fokus Utama: Jarak ke Goal (Bobot DIBESARKAN) ---
        # Kita naikkan drastis dari 20 ke 100 per langkah.
        # Ini memaksa AI untuk berpikir "Maju itu sangat menguntungkan".
        dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        score -= dist_to_goal * 100 

        # --- 3. Rasa Takut Musuh (DIBATASI) ---
        dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
        
        # LOGIKA BARU: "Zona Aman"
        # Jika jarak musuh > 3 langkah, ABAIKAN musuh. Fokus lari ke goal!
        # Ini mencegah AI ketakutan dari jarak jauh yang bikin dia mundur-mundur.
        if dist_to_enemy <= 3:
            # Jika sangat dekat (<= 3), baru beri penalti besar agar menghindar
            # Pakai kuadrat agar makin dekat makin panik
            score -= 5000.0 / (dist_to_enemy ** 2)

        # --- 4. Bonus Eksplorasi Kecil ---
        # Beri sedikit poin plus untuk opsi langkah yang banyak (agar tidak mojok)
        num_legal_moves = len(self.get_legal_actions())
        score += num_legal_moves * 10

        # --- 5. Anti-Looping (Noise) ---
        # Tambahkan nilai random yang sangat kecil (misal 0.1 - 0.9).
        # Tujuannya: Jika ada dua langkah yang nilainya sama persis (bikin bingung),
        # noise ini akan memilihkan salah satu agar AI tidak macet/bolak-balik.
        import random
        score += random.uniform(0, 0.5)

        return score

    def get_legal_actions(self):
        return self.state.get_valid_actions(unit='current')

    def is_terminal(self):
        is_term, _ = self.state.is_terminal()
        return is_term