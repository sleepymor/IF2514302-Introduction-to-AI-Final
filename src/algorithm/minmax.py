# algorithm/minimax/minimax.py
import math
from environment.environment import TacticalEnvironment

class MinimaxAgent:
    def __init__(self, depth=3):
        self.depth = depth

    def evaluate(self, env: TacticalEnvironment):
        """
        Evaluation function (semakin besar semakin baik untuk player)
        """
        px, py = env.player_pos
        ex, ey = env.enemy_pos
        gx, gy = env.goal

        # Jarak ke goal (semakin dekat semakin bagus)
        dist_goal = abs(px - gx) + abs(py - gy)

        # Jarak dari musuh (semakin jauh semakin bagus)
        dist_enemy = abs(px - ex) + abs(py - ey)

        # Trap penalty
        trap_penalty = -1000 if (px, py) in env.traps else 0

        # Terminal states
        is_terminal, reason = env.is_terminal()
        if is_terminal:
            if reason == "goal": return 99999
            if reason == "caught": return -99999
            if reason == "trap": return -99999

        # Skor keseluruhan
        return -dist_goal * 5 + dist_enemy * 3 + trap_penalty


    def minimax(self, env: TacticalEnvironment, depth, maximizing):
        is_terminal, reason = env.is_terminal()
        if depth == 0 or is_terminal:
            return self.evaluate(env), None

        actions = list(env.get_valid_actions())
        if not actions:
            return self.evaluate(env), None

        best_action = None

        # MAX = player
        if maximizing:
            best_value = -math.inf
            for action in actions:
                new_env = env.clone()
                new_env.step(action, simulate=True)

                value, _ = self.minimax(new_env, depth - 1, False)

                if value > best_value:
                    best_value = value
                    best_action = action

            return best_value, best_action

        # MIN = enemy
        else:
            best_value = math.inf
            for action in actions:
                new_env = env.clone()
                new_env.step(action, simulate=True)

                value, _ = self.minimax(new_env, depth - 1, True)

                if value < best_value:
                    best_value = value
                    best_action = action

            return best_value, best_action


    def action(self, env: TacticalEnvironment):
        _, best_move = self.minimax(env.clone(), self.depth, maximizing=True)
        return best_move
