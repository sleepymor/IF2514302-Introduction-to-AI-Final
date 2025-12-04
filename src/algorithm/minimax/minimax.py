import random

from algorithm.minimax.minimaxnode import MinimaxNode
from utils.logger import Logger


class MinimaxSearch:
    """Minimax search algorithm implementation."""

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.log = Logger("Minimax")

    def search(self, state):
        """
        Memulai pencarian Minimax untuk menemukan gerakan terbaik.
        """
        best_val = -float('inf')
        best_action = None

        legal_actions = list(state.get_valid_actions(unit='current'))

        if not legal_actions:
            return None

        self.log.info("Minimax is thinking...")

        # Level Akar (Root)
        for action in legal_actions:
            # Clone state dan terapkan langkah
            next_state = state.clone()
            next_state.step(action)

            # Panggil min_value (karena setelah kita gerak, giliran musuh)
            val = self.min_value(next_state, 1)

            # Log skor untuk setiap aksi
            self.log.info(f"Action {action} evaluated -> score: {val}")

            if val > best_val:
                best_val = val
                best_action = action

        self.log.info(f"Best action found: {best_action} with score: {best_val}")
        return best_action

    def max_value(self, state, depth):
        """Max value function for minimax."""
        # 1. Buat Node untuk cek terminal/evaluasi
        node = MinimaxNode(state)  # <--- Buat Objek Node

        # 2. Cek apakah sudah terminal atau mencapai kedalaman maksimum
        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()  # <--- Panggil method evaluate() dari class

        v = -float('inf')
        legal_actions = list(state.get_valid_actions(unit='current'))

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action)

            # Rekursi ke min_value
            v = max(v, self.min_value(next_state, depth + 1))

        return v

    def min_value(self, state, depth):
        """Min value function for minimax."""
        # 1. Buat Node untuk cek terminal/evaluasi
        node = MinimaxNode(state)  # <--- Buat Objek Node

        # 2. Cek terminal/depth
        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()  # <--- Panggil method evaluate() dari class

        v = float('inf')
        legal_actions = list(state.get_valid_actions(unit='current'))

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action)

            # Rekursi ke max_value
            v = min(v, self.max_value(next_state, depth + 1))

        return v