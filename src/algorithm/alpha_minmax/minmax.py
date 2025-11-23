# algorithm/alpha_minmax/minmax.py
import math
import random
from environment.environment import TacticalEnvironment

class Minimax:
    def __init__(self, max_depth=5, use_improvements=True):
        self.max_depth = max_depth
        self.use_improvements = use_improvements
        self.move_history = []
    
    def search(self, env: TacticalEnvironment):
        valid_actions = list(env.get_valid_actions('player'))
        
        if not valid_actions:
            return tuple(env.player_pos)
        
        # IMPROVEMENTS
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
        
        for action in valid_actions:
            new_env = env.clone()
            new_env.step(action, simulate=True)
            
            value = self._min_value(new_env, depth=1)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        if best_action and self.use_improvements:
            self.move_history.append(best_action)
            if len(self.move_history) > 10:
                self.move_history.pop(0)
        
        return best_action if best_action else valid_actions[0]
    
    def _max_value(self, env, depth):
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
                value = self._min_value(new_env, depth + 1)
            
            best_value = max(best_value, value)
        
        return best_value
    
    def _min_value(self, env, depth):
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
        Evaluation function untuk ENEMY - fokus kejar player
        """
        player_pos = tuple(env.player_pos)
        enemy_pos = tuple(env.enemy_pos)
        goal_pos = env.goal

        if terminal_reason == "caught":
            return 100000.0 - depth
        if terminal_reason == "goal":
            return -100000.0
        if terminal_reason == "trap": 
            return 50000.0 - depth

        dist_to_player = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
        chase_score = -dist_to_player * 15.0  

        dist_player_to_goal = abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])
        block_score = -dist_player_to_goal * 5.0 

        cutting_position_bonus = 0.0
        if (abs(enemy_pos[0] - goal_pos[0]) + abs(enemy_pos[1] - goal_pos[1]) < 
            abs(player_pos[0] - goal_pos[0]) + abs(player_pos[1] - goal_pos[1])):
            cutting_position_bonus = 50.0 

        depth_penalty = -depth * 1.0

        total_score = (
            chase_score +
            block_score + 
            cutting_position_bonus +
            depth_penalty
        )

        return total_score