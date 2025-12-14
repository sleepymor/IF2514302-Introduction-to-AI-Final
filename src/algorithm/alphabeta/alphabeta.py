from algorithm.alphabeta.alphabetanode import AlphaBetaNode
from utils.logger import Logger


class AlphaBetaSearch:
    """Alpha-Beta pruning search algorithm implementation."""

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.log = Logger("AlphaBeta")

    def search(self, state):
        """
        Memulai pencarian Alpha-Beta untuk menemukan gerakan terbaik.
        """
        # Kita mulai sebagai Maximizer (Player)
        alpha = -float('inf')
        beta = float('inf')

        best_val = -float('inf')
        best_action = None

        legal_actions = list(state.get_valid_actions(unit='current'))

        if not legal_actions:
            return None

        # Iterasi langkah pertama (Root)
        for action in legal_actions:
            # Clone state dan terapkan langkah
            next_state = state.clone()
            next_state.step(action)

            # Panggil min_value
            val = self.min_value(next_state, alpha, beta, 1)

            # --- TAMBAHAN 3: Tampilkan Log Skor ---
            self.log.info(f"Action {action} evaluated -> score: {val}")
            # --------------------------------------

            if val > best_val:
                best_val = val
                best_action = action

            # Update Alpha (untuk pruning di level root, meski jarang terjadi)
            alpha = max(alpha, best_val)

        # Log hasil akhir
        self.log.info(f"Best action found: {best_action} with score: {best_val}")

        return best_action

    def max_value(self, state, alpha, beta, depth):
        """Max value function for alpha-beta pruning."""
        node = AlphaBetaNode(state)

        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        v = -float('inf')
        legal_actions = list(state.get_valid_actions(unit='current'))

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action)

            v = max(v, self.min_value(next_state, alpha, beta, depth + 1))

            if v >= beta:
                # Optional: Log jika terjadi pruning (bisa membuat terminal penuh)
                # self.log.info(f"Pruning at depth {depth} (Beta Cutoff)")
                return v
            alpha = max(alpha, v)

        return v

    def min_value(self, state, alpha, beta, depth):
        """Min value function for alpha-beta pruning."""
        node = AlphaBetaNode(state)

        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        v = float('inf')
        legal_actions = list(state.get_valid_actions(unit='current'))

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action)

            v = min(v, self.max_value(next_state, alpha, beta, depth + 1))

            if v <= alpha:
                # Optional: Log jika terjadi pruning
                # self.log.info(f"Pruning at depth {depth} (Alpha Cutoff)")
                return v
            beta = min(beta, v)

        return v