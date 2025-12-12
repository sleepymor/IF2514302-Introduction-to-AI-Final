from algorithm.alphabeta.alphabetanode import AlphaBetaNode
from utils.logger import Logger


class AlphaBetaSearch:
    def __init__(self, max_depth=4):  # Depth kecil dulu
        self.max_depth = max_depth
        self.log = Logger("AlphaBeta")

    def search(self, state):
        """Simple AlphaBeta dengan depth 2"""
        legal_actions = list(state.get_valid_actions(unit="current"))

        if not legal_actions:
            return None

        self.log.info(f"Available moves: {len(legal_actions)}")

        best_score = float("-inf")
        best_action = legal_actions[0]

        # Coba semua action
        for action in legal_actions:
            # Player bergerak
            next_state = state.clone()
            next_state.step(action)

        # Hitung score setelah player bergerak
        score = self._evaluate_state(next_state, depth=1)

        self.log.info(f"Action {action} -> score: {score}")

        if score > best_score:
            best_score = score
            best_action = action

        self.log.info(f"Chosen: {best_action} (score: {best_score})")
        return best_action

    def _evaluate_state(self, state, depth):
        """Evaluasi state dengan mempertimbangkan enemy move"""

        # === PERBAIKAN: Buat node dulu ===
        node = AlphaBetaNode(state)

        if depth >= 2:  # Depth 2: player -> enemy -> evaluasi
            node = AlphaBetaNode(state)
        return node.evaluate()

        # Giliran enemy (minimizer)
        worst_score = 999999

        # Simulasikan semua kemungkinan enemy moves
        enemy_pos = tuple(state.enemy_pos)
        player_pos = tuple(state.player_pos)

        # 4 arah musuh
        enemy_moves = [
            (enemy_pos[0] + 1, enemy_pos[1]),
            (enemy_pos[0] - 1, enemy_pos[1]),
            (enemy_pos[0], enemy_pos[1] + 1),
            (enemy_pos[0], enemy_pos[1] - 1),
        ]

        # Filter yang valid
        valid_enemy_moves = []
        for move in enemy_moves:
            if 0 <= move[0] < state.width and 0 <= move[1] < state.height:
                valid_enemy_moves.append(move)

        if not valid_enemy_moves:
            node = AlphaBetaNode(state)
        return node.evaluate()

        # Enemy pilih yang terburuk untuk player
        for enemy_move in valid_enemy_moves:
            next_state = state.clone()
            next_state.enemy_pos = list(enemy_move)

            node = AlphaBetaNode(next_state)
            score = node.evaluate()

            worst_score = min(worst_score, score)

        return worst_score

    def _simple_heuristic(self, state, action):
        """TAMBAH FUNGSI INI di class AlphaBetaSearch"""
        # Posisi setelah action
        next_pos = action

        # Hitung jarak ke musuh (semakin jauh semakin baik)
        enemy_pos = tuple(state.enemy_pos)
        dist_to_enemy = abs(next_pos[0] - enemy_pos[0]) + abs(
            next_pos[1] - enemy_pos[1]
        )

        # Hitung jarak ke goal (semakin dekat semakin baik)
        goal_pos = state.goal
        dist_to_goal = abs(next_pos[0] - goal_pos[0]) + abs(next_pos[1] - goal_pos[1])

        # Kombinasi: utamakan menjauh dari musuh, lalu mendekati goal
        return dist_to_enemy * 10 - dist_to_goal * 5

    def max_value(self, state, alpha, beta, depth):
        node = AlphaBetaNode(state)

        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        v = -float("inf")
        legal_actions = list(state.get_valid_actions(unit="current"))

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action)

            v = max(v, self.min_value(next_state, alpha, beta, depth + 1))

            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    def min_value(self, state, alpha, beta, depth):
        node = AlphaBetaNode(state)

        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        v = float("inf")
        legal_actions = list(state.get_valid_actions(unit="current"))

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action)

            v = min(v, self.max_value(next_state, alpha, beta, depth + 1))

            if v <= alpha:
                return v
            beta = min(beta, v)

        return v
