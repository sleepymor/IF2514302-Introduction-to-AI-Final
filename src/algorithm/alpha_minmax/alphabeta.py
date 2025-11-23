# algorithm/alpha_minmax/alphabeta.py
import math
import random
from environment.environment import TacticalEnvironment

class AlphaBeta:
    def __init__(self, max_depth=5, use_improvements=True):  
        self.max_depth = max_depth
        self.use_improvements = use_improvements
        self.move_history = []
    
    def search(self, env: TacticalEnvironment):
        valid_actions = list(env.get_valid_actions('player'))
        
        if not valid_actions:
            return tuple(env.player_pos)
        
        if self.use_improvements:
            random.shuffle(valid_actions)
        
        if (self.use_improvements and 
            len(self.move_history) >= 6 and 
            self.move_history[-3:] == self.move_history[-6:-3]):
            print("🔄 Pattern detected - using diversified move")
            recent_actions = set(self.move_history[-3:])
            diverse_actions = [a for a in valid_actions if a not in recent_actions]
            if diverse_actions:
                action = random.choice(diverse_actions)
                self.move_history.append(action)
                return action
        
        best_value = -math.inf
        best_action = None
        alpha = -math.inf
        beta = math.inf
        
        for action in valid_actions:
            new_env = env.clone()
            new_env.step(action, simulate=True)
            
            value = self._min_value(new_env, depth=1, alpha=alpha, beta=beta)
            
            if value > best_value:
                best_value = value
                best_action = action
                alpha = max(alpha, best_value)
        
        if best_action and self.use_improvements:
            self.move_history.append(best_action)
            if len(self.move_history) > 10:
                self.move_history.pop(0)
        
        return best_action if best_action else valid_actions[0]
    
    def _max_value(self, env, depth, alpha, beta):
        terminal, reason = env.is_terminal()
        if terminal or depth >= self.max_depth:
            return self._evaluate(env, reason, depth)
        
        best_value = -math.inf
        valid_actions = list(env.get_valid_actions('player'))
        
        if not valid_actions:
            return self._evaluate(env, None, depth)
        
        
        if self.use_improvements:
            random.shuffle(valid_actions)
        
        for action in valid_actions:
            new_env = env.clone()
            result = new_env.step(action, simulate=True)
            
            if result and result[0]:
                value = self._evaluate(new_env, result[1], depth)
            else:
                value = self._min_value(new_env, depth + 1, alpha, beta)
            
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)
            
            if beta <= alpha:
                break
        
        return best_value
    
    def _min_value(self, env, depth, alpha, beta):
        terminal, reason = env.is_terminal()
        if terminal or depth >= self.max_depth:
            return self._evaluate(env, reason, depth)
        
        best_value = math.inf
        valid_actions = list(env.get_valid_actions('enemy'))
        
        if not valid_actions:
            return self._evaluate(env, None, depth)
        
        for action in valid_actions:
            new_env = env.clone()
            result = new_env.step(action, simulate=True)
            
            if result and result[0]:
                value = self._evaluate(new_env, result[1], depth)
            else:
                value = self._max_value(new_env, depth + 1, alpha, beta)
            
            best_value = min(best_value, value)
            beta = min(beta, best_value)
            
            if beta <= alpha:
                break
        
        return best_value
    
    
    def _evaluate(self, env, terminal_reason, depth):
        """
        IMPROVED evaluation function - lebih fokus ke goal dan hindari enemy
        """
        player_pos = tuple(env.player_pos)
        enemy_pos = tuple(env.enemy_pos)
        goal_pos = env.goal

        # 1. TERMINAL STATES 
        if terminal_reason == "goal":
            return 100000.0 - depth  
        if terminal_reason == "trap" or terminal_reason == "caught":
            return -100000.0 + depth  

        # 2. DISTANCE TO GOAL 
        dist_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        goal_score = -dist_goal * 10.0  

        # 3. ENEMY AVOIDANCE 
        dist_enemy = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
        enemy_score = 0.0
        
        if dist_enemy <= 2: 
            enemy_score = -100.0 * (3 - dist_enemy)
        elif dist_enemy >= 4: 
            enemy_score = 20.0
        else:  
            enemy_score = -10.0

        # 4. TRAP AVOIDANCE
        trap_score = -300.0 if player_pos in env.traps else 0.0

        # 5. PROGRESS BONUS
        progress_bonus = -depth * 2.0  

        # 6. EXPLORATION BONUS 
        exploration = random.uniform(-1.0, 1.0) if self.use_improvements else 0.0

        total_score = (
            goal_score +
            enemy_score + 
            trap_score +
            progress_bonus +
            exploration
        )

        return total_score