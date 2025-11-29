from environment.environment import TacticalEnvironment


class AlphaBetaNode:
    """
    Node Alpha-Beta yang diperbaiki dengan logika 'Repulsion Field'.
    Player akan mencoba menghindari musuh dari jarak jauh dan mencari jalan aman.
    """

    WIN_SCORE = 1
    LOSE_SCORE = -1

    def __init__(self, state: TacticalEnvironment, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

    def evaluate(self) -> float:
        player_pos = tuple(self.state.player_pos)
        enemy_pos = tuple(self.state.enemy_pos)
        goal_pos = self.state.goal

        # --- 1. Terminal State Check ---
        if player_pos == goal_pos:
            return self.WIN_SCORE
        if player_pos == enemy_pos or player_pos in self.state.traps:
            return self.LOSE_SCORE

        score = 0.0

        # --- 2. Jarak ke Goal (Manhattan Distance) ---
        dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(
            player_pos[1] - goal_pos[1]
        )

        # Bobot dikurangi (20) agar Player tidak terlalu 'kaku' ingin garis lurus
        score -= dist_to_goal * 20

        # --- 3. Gradasi Bahaya Musuh (Smooth Repulsion) ---
        dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(
            player_pos[1] - enemy_pos[1]
        )

        # Hindari pembagian dengan nol (jika enemy tepat di sebelah/sama)
        safe_dist_enemy = max(dist_to_enemy, 0.5)

        # Logika Medan Tolak-Menolak:
        # Semakin dekat musuh, nilai minusnya semakin besar secara eksponensial/drastis.
        # Ini membuat AI merasa 'panas' jika musuh mendekat, meski belum jarak 1.
        score -= 2000.0 / safe_dist_enemy

        # --- 4. Bonus Mobilitas (Agar tidak terjebak di pojok) ---
        # AI akan lebih suka posisi yang punya banyak opsi langkah (ruang terbuka)
        num_legal_moves = len(self._get_legal_actions())
        score += num_legal_moves * 15

        return score

    def _get_legal_actions(self):
        """
        Get all legal actions for the current player.

        Returns:
            set: Legal move positions for current turn
        """
        # Check whose turn it is
        if self.state.turn == "player":
            return self.state.get_move_range(self.state.player_pos)
        elif self.state.turn == "enemy":
            return self.state.get_move_range(self.state.enemy_pos, move_range=3)
        else:
            return set()

    def is_terminal(self):
        is_term, _ = self.state.is_terminal()
        return is_term
