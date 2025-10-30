# IF2514302-Introduction-to-AI-Final-Project ( Tactical Evade-and-Reach Problem)
# Concept and Idea

### 1. Problem Overview

This project focused  on designing AI agent capable of pathfinding in Adversarial Environment . AI goals it find the fastest and safest path to exit the Sequence while minimizing total cost and avoiding obstacle.
Unlike standard deterministic pathfinding problems (such as A* or Dijkstra), this environments operates in a stochastic and adversarial setting, where environmental conditions and agent interactions change over time.  
At every step, both the player (AI) and the enemy take turns moving within the set Environment. The player must plan optimal paths while considering not only the static layout (walls and traps) but also the unpredictable movement of enemies who actively pursue the player.
### 2. Core Concept

The core idea of this project is to create a turn-based tactical environment where an AI controlled agent (the player) must intelligently navigate a **grid-based map** to reach a goal while avoiding enemies and hazardous tiles.  
Each tile in the grid has an associated movement cost or risk value, representing different terrain types.

What differentiates this environment from traditional pathfinding problems is the adversarial and stochastic nature of the system:

- Enemies move each turn, attempting to chase or intercept the player.
- The player AI must adapt its route dynamically based on enemy positions, trap locations, and changing costs.
- Decisions are made in a turn-based sequence: the player acts, then enemies respond, and the state of the grid updates before the next turn.

This setting transforms the problem from simple route optimization into a strategic decision-making challenge where the AI must consider future consequences of each move, similar to adversarial games like chess or tactical RPGs.
### 3. Research Goal

The primary goal of this project is to evaluate and compare adversarial AI algorithms within a dynamic, turn-based environment that involves both pathfinding and strategic evasion.
Specific objectives include:
1. Algorithm Comparison
    Evaluate the effectiveness of Minimax, Alpha-Beta Pruning, and Monte Carlo Search in navigating adversarial environments.
2. Performance Analysis
	Measure each algorithm’s performance in terms of:
	- Total movement cost 
	- Time complexity / computational efficiency   
	- Survival rate (reaching goal vs being caught)
	- Decision optimality under uncertainty     
3. Adversarial Behaviour Modelling
    Study how the presence of an active pursuing enemy changes the decision-making process compared to standard pathfinding.
4. Dynamic Path Optimization
    Explore how AI agents can adapt to changing risks, such as moving enemies or newly triggered traps.
5. Visualization and Simulation 
    Create an interpretable simulation that visualizes each turn showing how different algorithms behave under identical conditions.
### 4. Inspiration
This project draws inspiration from both academic AI problems and tactical game design:
- Fire Emblem
	Inspired the grid movement mechanic, where each decision can drastically alter future outcomes.
- Pac-Man
	Provided the concept of **active pursuit**, where the player must constantly adapt to avoid moving enemies rather than static barriers.
- Traveling Salesman Problem (TSP)
	Provided the foundation for route optimization minimizing total cost and distance while planning ahead.
- N-Queen Problem
	Contributed the idea of positional reasoning and conflict avoidance, which parallels the player avoiding enemy attack zones.
# Algorithm Implementation
### 1. Minimax Algorithm
### 2. Alpha-Beta Pruning
### 3. Monte Carlo Search
# Libraries and Tools
### 1. PyGame
### 2. Pandas
### 3. MathPlotLib
### 4. NumPy
### 5. Google Collabs
# Team Role
| No | Name             | Role / Task Description                                                                                                                                                                   |
| -- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **Rizal** | Create the environment grid (walls, traps, goal, adjustable size). Develop the main loop for all algorithms. Work on MCTS, Alpha-Beta, and Minimax. Handle final integration and testing. |
| 2  | **Jayen**        | Develop the main loop for all algorithms. Implement the **MCTS** algorithm. Integrate the implemented algorithm into the main system.                                                     |
| 3  | **Ibnu**         | Develop parts of **MCTS** (tree structure and simulation). Integrate the implemented algorithm into the main system.                                                                      |
| 4  | **Wisnu**        | Implement the **Alpha-Beta Pruning** algorithm. Integrate the implemented algorithm into the main system.                                                                                 |
| 5  | **Dylan**        | Implement the **Minimax** algorithm. Integrate the implemented algorithm into the main system.                                                                                            |


