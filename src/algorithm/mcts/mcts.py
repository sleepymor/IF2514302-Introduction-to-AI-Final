import numpy as np
import random
import math
from environment.environment import TacticalEnvironment
from algorithm.mcts.mctsnode import MCTSNode
from algorithm.astar.astar import AStar
from utils.logger import Logger


class MCTS:
    """
    Monte Carlo Tree Search algorithm for tactical decision-making.

    This class implements the four main phases of MCTS:
    1. Selection - Traverse tree using UCB1 policy
    2. Expansion - Add new child node to tree
    3. Simulation - Run rollout to terminal state
    4. Backpropagation - Update node statistics

    Additionally includes safety mechanisms to override dangerous moves
    that would put the player too close to enemies.

    Attributes:
        iterations (int): Number of MCTS iterations to perform
        exploration_constant (float): UCB1 exploration parameter (C)
        max_sim_depth (int): Maximum simulation depth before timeout
        log (Logger): Logger instance for debug/info messages
        verbose (bool): Whether to print detailed logs
    """

    def __init__(
        self,
        iterations: int = 2500,
        exploration_constant: float = 2.0,
        max_sim_depth: int = 150,
        verbose: bool = True,
    ):
        """
        Initialize MCTS algorithm with specified parameters.

        Args:
            iterations: Number of MCTS iterations (default: 2500)
            exploration_constant: UCB1 exploration constant (default: 2.0)
            max_sim_depth: Maximum rollout depth (default: 150)
            verbose: Enable detailed logging (default: True)
        """
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_sim_depth = max_sim_depth
        self.log = Logger("MCTS")
        self.verbose = verbose

    def search(self, node: TacticalEnvironment) -> tuple:
        """
        Execute MCTS search to find the best action.

        This method performs the complete MCTS search with the following steps:
        1. Check for instant win opportunities at root
        2. Run MCTS iterations (selection, expansion, simulation, backpropagation)
        3. Select best child based on visit count and win rate
        4. Apply safety override if selected move is too risky

        Args:
            node: Initial game state (TacticalEnvironment)

        Returns:
            tuple: Best action as (x, y) coordinates
        """
        # ========== 1. INSTANT WIN CHECK ==========
        # Check if we can win immediately without running full MCTS
        root_state = MCTSNode(state=node).state
        valid_actions = list(root_state.get_valid_actions(unit="current"))

        for action in valid_actions:
            temp_state = root_state.clone()
            temp_state.step(action, simulate=True)

            if tuple(temp_state.player_pos) == temp_state.goal:
                self.log.info(f"Instant Win found at {action}!")
                return action

        # ========== 2. MCTS MAIN LOOP ==========
        # Initialize root node and run MCTS iterations
        root = MCTSNode(state=node)

        for _ in range(self.iterations):
            selected = self.selection(root)
            expanded = self.expansion(selected)
            reward = self.simulation(expanded)
            self.backpropagation(expanded, reward)

        # ========== 3. BEST CHILD SELECTION ==========
        # Select child with highest win rate, requiring minimum visits for stability
        if not root.children:
            return random.choice(valid_actions)

        # Require at least 25% of fair share visits for consideration
        min_visits = max(1, root.visits // len(root.children) // 2)
        candidate_children = [
            child for child in root.children if child.visits >= min_visits
        ]

        # Fallback to all children if none meet minimum visits
        if not candidate_children:
            candidate_children = root.children

        # Select child with highest win rate
        best_child = max(
            candidate_children,
            key=lambda c: c.wins / (c.visits + 1) if c.visits > 0 else 0,
        )
        mcts_action = best_child.action

        if self.verbose:
            win_rate = (
                best_child.wins / best_child.visits if best_child.visits > 0 else 0
            )
            self.log.info(
                f"MCTS suggests: {mcts_action}, "
                f"visits: {best_child.visits}, "
                f"win_rate: {win_rate:.2f}"
            )

        # ========== 4. SAFETY OVERRIDE ==========
        # Prevent suicide moves that enter enemy attack range
        enemy_pos = tuple(root_state.enemy_pos)

        try:
            dist_to_enemy = abs(mcts_action[0] - enemy_pos[0]) + abs(
                mcts_action[1] - enemy_pos[1]
            )
        except Exception:
            dist_to_enemy = float("inf")

        # Override if move is within enemy attack range (distance <= 2)
        if dist_to_enemy <= 2:
            if self.verbose:
                self.log.warning(
                    f"DANGER! MCTS chose a risky move {mcts_action} "
                    f"(Dist {dist_to_enemy}). Overriding..."
                )

            # Find safest alternative action (maximize distance from enemy)
            best_safe_action = None
            max_dist = -1

            for action in valid_actions:
                d = abs(action[0] - enemy_pos[0]) + abs(action[1] - enemy_pos[1])

                if d > dist_to_enemy and d > max_dist:
                    max_dist = d
                    best_safe_action = action

            # Use safe action if found
            if best_safe_action:
                if self.verbose:
                    self.log.info(
                        f"Safety Override: Switched to {best_safe_action} "
                        f"(Dist {max_dist})"
                    )
                return best_safe_action
            else:
                # Fallback: choose action furthest from enemy
                fallback = max(
                    valid_actions,
                    key=lambda a: abs(a[0] - enemy_pos[0]) + abs(a[1] - enemy_pos[1]),
                )
                if self.verbose:
                    self.log.warning(
                        f"No strictly safer move found; fallback to {fallback}"
                    )
                return fallback

        return mcts_action

    def selection(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Traverse tree using UCB1 until reaching expandable node.

        Traverses the tree by repeatedly selecting the best child (according to
        UCB1 policy) until reaching a node that is either terminal, not fully
        expanded, or a leaf node.

        Args:
            node: Starting node for selection

        Returns:
            MCTSNode: Selected node for expansion
        """
        while not node.is_terminal():
            if not node.children:
                return node
            if not node.is_fully_expanded():
                return node
            node = node.best_child(c=self.exploration_constant)

        return node

    def expansion(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add a new child node to the tree.

        Selects a random untried action from the node and creates a new child
        node by applying that action to a cloned state.

        Args:
            node: Node to expand

        Returns:
            MCTSNode: Newly created child node, or original node if terminal
        """
        if node.is_terminal() or not node.untried_actions:
            return node

        # Select random untried action
        action = random.choice(node.untried_actions)

        # Create new state by applying action
        new_state = node.state.clone()
        new_state.step(action, simulate=True)

        # Create and add child node
        child = MCTSNode(new_state, parent=node, action=action)
        node.add_child(child)

        return child

    def simulation(self, node: MCTSNode) -> float:
        """
        Simulation phase: Run rollout from node to terminal state.

        Simulates gameplay using lightweight policies for both player and enemy
        until reaching a terminal state or maximum depth. Returns normalized
        reward based on outcome.

        Args:
            node: Starting node for simulation

        Returns:
            float: Normalized reward in range [0.0, 1.0]
        """
        state = node.state.clone()

        # Check if already terminal
        is_term, reason = state.is_terminal()
        if is_term:
            return self.rollout_reward(state, reason)

        # Run simulation up to maximum depth
        for _ in range(self.max_sim_depth):
            if state.turn == "player":
                player_action = self._player_policy(state)
                state.step(player_action, simulate=True)

            elif state.turn == "enemy":
                enemy_action = self._enemy_policy(state)
                state.step(enemy_action, simulate=True)

            # Check for terminal state
            is_term, reason = state.is_terminal()
            if is_term:
                return self.rollout_reward(state, reason)

        # Timeout: return partial reward
        return self.rollout_reward(state, reason=None)

    def _player_policy(self, state: TacticalEnvironment) -> tuple:
        """
        Fast player policy for simulation rollouts.

        Uses epsilon-greedy strategy with A* guidance:
        - 85% of time: Follow A* path to goal (greedy)
        - 15% of time: Random exploration

        Includes safety checks to avoid traps and enemy proximity.

        Args:
            state: Current game state

        Returns:
            tuple: Selected action as (x, y) coordinates, or None if no moves
        """
        goal = tuple(state.goal)
        player = tuple(state.player_pos)
        enemy = tuple(state.enemy_pos)

        legal_moves = list(state.get_valid_actions(unit="current"))
        if not legal_moves:
            return None

        # Direct finish if goal is reachable
        if goal in legal_moves:
            return goal

        def manhattan(a: tuple, b: tuple) -> int:
            """Calculate Manhattan distance between two points."""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # ========== EPSILON-GREEDY EXPLORATION ==========
        # 15% random exploration, 85% greedy exploitation
        eps = 0.15
        if random.random() < eps:
            return random.choice(legal_moves)

        # ========== A* GUIDED GREEDY POLICY ==========
        # Try to follow A* path if available and safe
        try:
            astar = AStar(state)
            path = astar.search(player, goal)

            if path and len(path) > 1:
                next_step = path[1]

                # Verify next step is safe
                if next_step in legal_moves:
                    dist_to_enemy_next = manhattan(next_step, enemy)

                    # Check: not on trap and not too close to enemy
                    if next_step not in state.traps and dist_to_enemy_next > 1:
                        return next_step
        except Exception:
            pass

        # ========== MANHATTAN DISTANCE GREEDY FALLBACK ==========
        # Score each move based on goal proximity and enemy distance
        best_score = float("-inf")
        best_move = None

        for move in legal_moves:
            # Strongly avoid traps
            if move in state.traps:
                continue

            dist_to_goal = manhattan(move, goal)
            dist_to_enemy = manhattan(move, enemy)

            # Multi-factor scoring:
            # - Primary: minimize distance to goal (weight: 10.0)
            # - Secondary: maximize distance from enemy (weight: 1.0)
            score = -dist_to_goal * 10.0 + dist_to_enemy * 1.0

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is not None:
            return best_move

        # ========== LAST RESORT ==========
        # Pick safest remaining move (furthest from enemy)
        return max(legal_moves, key=lambda m: manhattan(m, enemy))

    def _enemy_policy(self, state: TacticalEnvironment) -> tuple:
        """
        Fast enemy policy for simulation rollouts.

        Simple greedy policy: move towards player to minimize distance.
        If player is in range, attack immediately.

        Args:
            state: Current game state

        Returns:
            tuple: Selected action as (x, y) coordinates
        """
        enemy_pos = tuple(state.enemy_pos)
        player_pos = tuple(state.player_pos)

        legal_moves = list(state.get_move_range(state.enemy_pos, move_range=2))

        # Attack if player is in range
        if player_pos in legal_moves:
            return player_pos

        # Move towards player (minimize Manhattan distance)
        best_move = enemy_pos
        min_dist = float("inf")

        for move in legal_moves:
            dist = abs(move[0] - player_pos[0]) + abs(move[1] - player_pos[1])
            if dist < min_dist:
                min_dist = dist
                best_move = move

        return best_move

    def rollout_reward(self, state: TacticalEnvironment, reason: str) -> float:
        """
        Calculate normalized reward for rollout outcome.

        Rewards are normalized to [0.0, 1.0] range:
        - Win (goal reached): 1.0
        - Loss (trap/caught): 0.0
        - Timeout/Survival: [0.25, 0.95] based on goal proximity and enemy distance

        Args:
            state: Final state after rollout
            reason: Terminal reason ('goal', 'trap', 'caught', or None for timeout)

        Returns:
            float: Normalized reward in [0.0, 1.0]
        """
        # ========== WIN CONDITION ==========
        if reason == "goal":
            return 1.0

        # ========== LOSS CONDITIONS ==========
        if reason == "trap" or reason == "caught":
            return -1.0

        # ========== TIMEOUT / SURVIVAL SCORING ==========
        # Compute partial reward based on goal proximity and enemy distance
        player_pos = tuple(state.player_pos)
        enemy_pos = tuple(state.enemy_pos)
        goal_pos = tuple(state.goal)

        def manhattan(a: tuple, b: tuple) -> int:
            """Calculate Manhattan distance."""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Calculate goal proximity ratio
        dist_goal = manhattan(player_pos, goal_pos)
        max_dist = float(state.width + state.height)  # Max possible distance

        closeness = 1.0 - (dist_goal / max_dist)
        closeness = max(0.0, min(1.0, closeness))

        base_score = 0.25 + 0.7 * closeness

        # ========== ENEMY PROXIMITY PENALTY ==========
        # Apply penalty if too close to enemy
        dist_enemy = manhattan(player_pos, enemy_pos)

        if dist_enemy <= 1:
            penalty_factor = 0.5  # Severe penalty (adjacent)
        elif dist_enemy == 2:
            penalty_factor = 0.7  # Moderate penalty (attack range)
        elif dist_enemy == 3:
            penalty_factor = 0.85  # Light penalty (nearby)
        else:
            penalty_factor = 1.0  # No penalty (safe distance)

        score = base_score * penalty_factor

        # Clamp to [0.0, 0.99] to keep < 1.0 (only true win = 1.0)
        return max(0.0, min(0.99, score))

    def backpropagation(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase: Update statistics up the tree.

        Propagates simulation result (reward) back up the tree from the given
        node to the root, updating visit counts and win totals at each node.

        Args:
            node: Starting node (typically expanded/simulated node)
            reward: Reward value to propagate (normalized to [0.0, 1.0])
        """
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
