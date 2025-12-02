from environment.environment import TacticalEnvironment

class AlphaBetaNode:
    """
    SIMPLE AlphaBeta Node - FIXED VERSION
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

        # 1. TERMINAL STATES
        if player_pos == goal_pos:
            return self.WIN_SCORE
        if player_pos == enemy_pos or player_pos in self.state.traps:
            return self.LOSE_SCORE

        score = 0.0
        
        # 2. JARAK KE GOAL (selalu negatif, makin dekat makin baik)
        dist_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        score -= dist_to_goal * 10  # PASTI ADA NILAI!
        
        # 3. JAUH DARI MUSUH (sangat penting!)
        dist_to_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
        
        if dist_to_enemy <= 2:
            score -= 1000  # Sangat dekat = buruk
        elif dist_to_enemy <= 4:
            score -= 300   # Dekat = agak buruk
        else:
            score += 50    # Jauh = baik
            
        # 4. BONUS untuk mobilitas
        legal_moves = self.get_legal_actions()
        num_moves = len(legal_moves)
        score += num_moves * 5
        
        # 5. HINDARI PERANGKAP
        for trap in self.state.traps:
            trap_dist = abs(player_pos[0] - trap[0]) + abs(player_pos[1] - trap[1])
            if trap_dist == 0:
                score -= 5000
            elif trap_dist == 1:
                score -= 1000
            elif trap_dist == 2:
                score -= 200
        
        # 6. HINDARI POJOK
        if (player_pos[0] == 0 or player_pos[0] == self.state.width - 1 or
            player_pos[1] == 0 or player_pos[1] == self.state.height - 1):
            score -= 100
            
        # PASTIKAN SCORE TIDAK 0!
        if score == 0:
            score = 1.0  # Minimal 1
        
        # 7. BONUS untuk posisi tengah
        center_x, center_y = self.state.width // 2, self.state.height // 2
        dist_to_center = abs(player_pos[0] - center_x) + abs(player_pos[1] - center_y)
        score -= dist_to_center * 3

        # 8. HINDARI GERAKAN BOLAK-BALIK (jika ada parent)
        if self.parent:
            grandparent = self.parent.parent
            if grandparent and player_pos == tuple(grandparent.state.player_pos):
                score -= 500  # Penalty untuk bolak-balik

        # 9. PASTIKAN SCORE TIDAK 0!
       
        # PASTIKAN SCORE TIDAK 0!
        # === TAMBAH INI ===
        if -0.1 < score < 0.1:  # Jika score hampir 0
            score += 0.3  # <-- PASTI 0.3  Tambah nilai kecil positif
    
            
        return score

    def get_legal_actions(self):
        return self.state.get_valid_actions(unit='current')

    def is_terminal(self):
        is_term, _ = self.state.is_terminal()
        return is_term