import math
import random
from environment.environment import TacticalEnvironment
from utils.logger import Logger

log = Logger("Minimax")

class MinimaxSearch:
    
    def __init__(self, max_depth=4, use_improvements=True):
        self.max_depth = max_depth
        self.use_improvements = use_improvements
        self.move_history = []
        log.info(f"MinimaxSearch initialized with max_depth={self.max_depth}")

    def search(self, initial_state: TacticalEnvironment):
        best_action = None
        best_score = -math.inf
        legal_actions = list(initial_state.get_valid_actions(unit='current'))
        
        if not legal_actions:
            return tuple(initial_state.player_pos)

        if self.use_improvements:
            random.shuffle(legal_actions)

        if (self.use_improvements and 
            len(self.move_history) >= 4 and 
            self.move_history[-2:] == self.move_history[-4:-2]):
            log.info("Pattern detected - using diversified move")
            recent_actions = set(self.move_history[-2:])
            diverse_actions = [a for a in legal_actions if a not in recent_actions]
            if diverse_actions:
                action = random.choice(diverse_actions)
                self.move_history.append(action)
                return action

        for action in legal_actions:
            new_state = initial_state.clone()
            new_state.step(action, simulate=True)
            score = self._minimax(new_state, self.max_depth - 1, False)
            log.info(f"Action {action} evaluated -> score: {score}")

            if score > best_score:
                best_score = score
                best_action = action
        
        if self.use_improvements and best_action:
            self.move_history.append(best_action)
            if len(self.move_history) > 8:
                self.move_history.pop(0)
                
        log.info(f"Best action found: {best_action} with score: {best_score}")
        return best_action

    def _minimax(self, current_state: TacticalEnvironment, depth: int, is_maximizing_player: bool):
        is_term, term_reason = current_state.is_terminal()
        
        if depth == 0 or is_term:
            return self._enhanced_evaluate(current_state, term_reason, depth)

        legal_actions = list(current_state.get_valid_actions(unit='current'))

        if is_maximizing_player:
            max_eval = -math.inf
            for action in legal_actions:
                new_state = current_state.clone()
                new_state.step(action, simulate=True)
                
                eval_val = self._minimax(new_state, depth - 1, False)
                max_eval = max(max_eval, eval_val)
            return max_eval
            
        else:
            min_eval = math.inf
            for action in legal_actions:
                new_state = current_state.clone()
                new_state.step(action, simulate=True)
                
                eval_val = self._minimax(new_state, depth - 1, True)
                min_eval = min(min_eval, eval_val)
            return min_eval

    def _enhanced_evaluate(self, env, terminal_reason, depth):
        player_pos = tuple(env.player_pos)
        enemy_pos = tuple(env.enemy_pos)
        goal_pos = env.goal


        if terminal_reason == "goal":
            return 10000.0 - depth * 10  
        if terminal_reason == "trap" or terminal_reason == "caught":
            return -10000.0 + depth * 10 

        dist_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        goal_score = -dist_goal * 15.0  
        dist_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
        enemy_score = 0.0

        if dist_enemy <= 2:  
            enemy_score = -100.0 * (3 - dist_enemy)
        elif dist_enemy >= 4: 
            enemy_score = 20.0

        trap_score = -300.0 if player_pos in env.traps else 0.0
        strategic_bonus = 0.0
        player_to_goal = dist_goal
        enemy_to_goal = abs(enemy_pos[0] - goal_pos[0]) + abs(enemy_pos[1] - goal_pos[1])
        
        if player_to_goal < enemy_to_goal:
            strategic_bonus = 25.0 

        progress_bonus = -depth * 2.0  

        total_score = (
            goal_score +
            enemy_score + 
            trap_score +
            strategic_bonus +
            progress_bonus
        )

        return total_score