
import math
import random
from environment.environment import TacticalEnvironment

class Minimax:
    def __init__(self, max_depth=4, use_improvements=True):
        self.max_depth = max_depth
        self.use_improvements = use_improvements
        self.move_history = []
    
    def search(self, env: TacticalEnvironment):
        """
        Minimax search untuk PLAYER - cari move terbaik
        """
        valid_actions = list(env.get_valid_actions('player'))
        
        if not valid_actions:
            return tuple(env.player_pos)
        
        if self.use_improvements:
            random.shuffle(valid_actions)
        
        if (self.use_improvements and 
            len(self.move_history) >= 4 and 
            self.move_history[-2:] == self.move_history[-4:-2]):
            print("🔄 Pattern detected - using diversified move")
            recent_actions = set(self.move_history[-2:])
            diverse_actions = [a for a in valid_actions if a not in recent_actions]
            if diverse_actions:
                action = random.choice(diverse_actions)
                self.move_history.append(action)
                return action
        
        best_value = -math.inf
        best_action = valid_actions[0]  
        
        for action in valid_actions:
            new_env = env.clone()
            new_env.step(action, simulate=True)
            value = self._min_value(new_env, depth=1)

            if value > best_value:
                best_value = value
                best_action = action
        
        if self.use_improvements:
            self.move_history.append(best_action)
            if len(self.move_history) > 8:
                self.move_history.pop(0)
        
        return best_action
    
    def _max_value(self, env, depth):
        """
        MAX player - player's turn
        """
        terminal, reason = env.is_terminal()
        if terminal or depth >= self.max_depth:
            return self._evaluate(env, reason, depth)
        
        best_value = -math.inf
        valid_actions = list(env.get_valid_actions('player'))
        
        if not valid_actions:
            return self._evaluate(env, None, depth)
        
        for action in valid_actions:
            new_env = env.clone()
            result = new_env.step(action, simulate=True)
            
            if result and result[0]:
                value = self._evaluate(new_env, result[1], depth)
            else:
                value = self._min_value(new_env, depth + 1)
            
            best_value = max(best_value, value)
        
        return best_value
    
    def _min_value(self, env, depth):
        """
        MIN player - enemy's turn (simulate dengan A* strategy)
        """
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
                value = self._max_value(new_env, depth + 1)
            
            best_value = min(best_value, value)
        
        return best_value
    
    def _evaluate(self, env, terminal_reason, depth):
        """
        Evaluation function untuk PLAYER perspective
        """
        player_pos = tuple(env.player_pos)
        enemy_pos = tuple(env.enemy_pos)
        goal_pos = env.goal

        # TERMINAL STATES
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