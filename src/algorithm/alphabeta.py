# algorithm/alphabeta/alphabeta.py
import math
from environment.environment import TacticalEnvironment

class AlphaBetaAgent:
    def __init__(self, depth=4):
        self.depth = depth

    def evaluate(self, env: TacticalEnvironment):
        px, py = env.player_pos
        ex, ey = env.enemy_pos
        gx, gy = env.goal

        dist_goal = abs(px - gx) + abs(py - gy)
        dist_enemy = abs(px - ex) + abs(py - ey)
        trap_penalty = -1000 if (px, py) in env.traps else 0

        is_terminal, reason = env.is_terminal()
        if is_terminal:
            if reason == "goal": return 99999
            if reason == "caught": return -99999
            if reason == "trap": return -99999

        return -dist_goal * 5 + dist_enemy * 3 + trap_penalty


    def alphabeta(self, env: TacticalEnvironment, depth, alpha, beta, maximizing):
        is_terminal, _ = env.is_terminal()
        if depth == 0 or is_terminal:
            return self.evaluate(env), None

        actions = list(env.get_valid_actions())
        if not actions:
            return self.evaluate(env), None

        best_action = None

        # MAX node (player)
        if maximizing:
            value = -math.inf
            for action in actions:
                new_env = env.clone()
                new_env.step(action, simulate=True)

                eval_value, _ = self.alphabeta(new_env, depth - 1, alpha, beta, False)

                if eval_value > value:
                    value = eval_value
                    best_action = action

                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # prune

            return value, best_action

        # MIN node (enemy)
        else:
            value = math.inf
            for action in actions:
                new_env = env.clone()
                new_env.step(action, simulate=True)

                eval_value, _ = self.alphabeta(new_env, depth - 1, alpha, beta, True)

                if eval_value < value:
                    value = eval_value
                    best_action = action

                beta = min(beta, value)
                if beta <= alpha:
                    break  # prune

            return value, best_action


    def action(self, env: TacticalEnvironment):
        _, best_move = self.alphabeta(env.clone(), self.depth, -math.inf, math.inf, True)
        return best_move
