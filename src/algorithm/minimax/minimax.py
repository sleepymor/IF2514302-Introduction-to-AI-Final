from algorithm.minimax.minimaxnode import MinimaxNode
from utils.logger import Logger
import random


class MinimaxSearch:
    def __init__(
        self, max_depth=2
    ):  # <--- Ubah default depth jadi 3 atau 4 agar lebih pintar
        self.max_depth = max_depth
        self.log = Logger("Minimax")

    def search(self, state):
        """
        Memulai pencarian Minimax.
        PERBAIKAN: Menambahkan 'Random Tie-Breaking' agar tidak stuck (looping).
        """
        best_val = -float("inf")

        # List untuk menampung semua gerakan terbaik (jika ada yang nilainya sama)
        best_actions = []

        legal_actions = list(state.get_valid_actions(unit="current"))

        if not legal_actions:
            return None

        # Acak urutan pengecekan agar AI tidak bias ke satu arah (misal: selalu cek atas dulu)
        random.shuffle(legal_actions)

        # Iterasi level pertama (Root)
        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action, simulate=True)

            # Masuk ke rekursi (Depth 1)
            val = self.min_value(next_state, 1)

            # self.log.info(f"Action {action} -> Score: {val}") # Uncomment untuk debug

            # LOGIKA PEMILIHAN TERBAIK (Update)
            if val > best_val:
                best_val = val
                best_actions = [action]  # Reset list, isi dengan juara baru
            elif val == best_val:
                best_actions.append(action)  # Jika seri, tambahkan ke list kandidat

        # Jika ada banyak gerakan dengan nilai terbaik yang sama, pilih acak
        if best_actions:
            chosen_action = random.choice(best_actions)
            # self.log.info(f"Chosen {chosen_action} from candidates: {best_actions} with score {best_val}")
            return chosen_action

        return None

    def max_value(self, state, depth):
        # Giliran Maximizer (Player)
        node = MinimaxNode(state)
        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        v = -float("inf")
        legal_actions = list(state.get_valid_actions(unit="current"))

        # Jika tidak ada gerakan legal (terjepit), evaluasi posisi saat ini
        if not legal_actions:
            return node.evaluate()

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action, simulate=True)
            v = max(v, self.min_value(next_state, depth + 1))

        return v

    def min_value(self, state, depth):
        # Giliran Minimizer (Enemy)
        node = MinimaxNode(state)
        if depth == self.max_depth or node.is_terminal():
            return node.evaluate()

        v = float("inf")
        legal_actions = list(state.get_valid_actions(unit="current"))

        if not legal_actions:
            return node.evaluate()

        for action in legal_actions:
            next_state = state.clone()
            next_state.step(action, simulate=True)
            v = min(v, self.max_value(next_state, depth + 1))

        return v
