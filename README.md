# Multi-Agent Search 

This repo  contains the implementation of various multi-agent search agents for the Pacman game. The search agents include Reflex Agent, Minimax Agent, Alpha-Beta Agent, and Monte Carlo Tree Search Agent. The code is based on the UC Berkeley Pacman AI project, and the original repository can be found at [UC Berkeley Pacman AI Project](http://ai.berkeley.edu/multiagent.html).

## Table of Contents
- [File Structure](#file-structure)
- [How to Run](#how-to-run)
- [Layouts](#layouts)
- [Testing](#testing)
- [Reflex Agent](#reflex-agent)
- [Minimax Agent](#minimax-agent)
- [Alpha-Beta Agent](#alpha-beta-agent)


## File Structure

This  repository has the following directory structure:
```
pacman_multi_agent/
    source/
        multiAgents.py
        pacman.py
        util.py
        ...
    README.md
    ...
```

The relevant files and directories for this project are:
-   `source/multiAgents.py`: Contains the implementation of the search agents.
-   `source/pacman.py`: The main Pacman game file.
-   `source/util.py`: Contains utility functions and classes.
- `source/results/`: Contains the collected performance results of the agents

## How to Run

To run the different search agents, follow these steps:
1. Download or clone the repository.
2. Open a terminal and navigate to the `source` directory:
   ```
   cd path/to/pacman_multi_agent/source/
   ```
3. Run the desired search agent using the appropriate command:
   - Reflex Agent:
     ```
     python pacman.py -p ReflexAgent -l testClassic
     ```
   - Minimax Agent:
     ```
     python pacman.py -p MinimaxAgent -a depth=3 -l smallClassic
     ```
   - Alpha-Beta Agent:
     ```
     python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
     ```
   - Monte Carlo Tree Search Agent:
     ```
     python pacman.py -p MonteCarloTreeSearchAgent -l mediumClassic
     ```

Here's a list of all the command-line arguments available for the `pacman.py` script:

- `-n` or `--numGames`: Specify the number of games to play (default: 1).
- `-l` or `--layout`: Specify the layout file to load the map layout from (default: 'mediumClassic').
- `-p` or `--pacman`: Specify the Pacman agent to use (default: 'KeyboardAgent').
- `-t` or `--textGraphics`: Display output as text only (default: False).
- `-q` or `--quietTextGraphics`: Generate minimal output and no graphics (default: False).
- `-g` or `--ghosts`: Specify the ghost agent to use (default: 'RandomGhost').
- `-k` or `--numghosts`: Specify the maximum number of ghosts to use (default: 4).
- `-z` or `--zoom`: Zoom the size of the graphics window (default: 1.0).
- `-f` or `--fixRandomSeed`: Fix the random seed to always play the same game (default: False).
- `-r` or `--recordActions`: Write game histories to a file (default: False).
- `--replay`: Specify a recorded game file (pickle) to replay (default: None).
- `-a` or `--agentArgs`: Pass comma-separated arguments to the Pacman agent (default: None).
- `-x` or `--numTraining`: Specify the number of episodes for training (default: 0).
- `--frameTime`: Set the delay time between frames; < 0 means keyboard control (default: 0.1).
- `-c` or `--catchExceptions`: Turn on exception handling and timeouts during games (default: False).
- `--timeout`: Set the maximum length of time an agent can spend computing in a single game (default: 30).

These arguments allow customization of the game settings, choosing different agents, modifying the display options, recording and replaying games, and more.

## Layouts

The Pacman game includes several layouts that define the game environment. The layouts are stored in the `layouts` directory of this repository. Some of the notable layouts are:

- `testClassic`: A small classic layout for testing purposes.
- `smallClassic`: A small classic layout suitable for running the agents.
- `mediumClassic`: A medium-sized classic layout.

These layouts provide different configurations of walls, food pellets, and ghost positions, allowing for various scenarios to test and evaluate the performance of the search agents.

## Testing

The `testing.py` script was written test and evaluate the performance of each search agent. It runs each agent on multiple layouts for a specified number of times and collects performance metrics.

To run the tests, use the following command:

```
python testing.py
```

This command will run all the search agents on various layouts and display the results, including the average score, win rate, and execution time per run. 


## Reflex Agent

The `ReflexAgent` class implements a reflex agent that chooses actions based on a state evaluation function. The evaluation function considers the following factors:
- Current game score
- Distance to the nearest food
- Proximity to ghosts (avoiding close ghosts and chasing scared ghosts)

The agent selects the action that maximizes the evaluation function.

### Key Logic
- The reflex agent evaluates the desirability of each action based on the current game state and the resulting successor state.
- It considers the game score, the distance to the nearest food, and the proximity to ghosts (both avoiding close ghosts and chasing scared ghosts).
- The evaluation function assigns weights to these factors to determine the overall desirability of each action.
- The agent selects the action that maximizes the evaluation function, leading to the most desirable successor state.

## Minimax Agent

The `MinimaxAgent` class implements an adversarial search agent using the minimax algorithm. It considers each agent's turn at each depth level, where Pacman is the maximizing agent and ghosts are minimizing agents.

The `getAction` method recursively explores the game tree to a specified depth, alternating between maximizing Pacman's score and minimizing the ghosts' scores. It returns the action that leads to the best score for Pacman.

### Key Logic
- The minimax agent uses the minimax algorithm to explore the game tree and make decisions.
- It recursively evaluates game states by alternating between maximizing Pacman's score and minimizing the ghosts' scores.
- At each depth level, the agent considers the legal actions available to the current agent (Pacman or ghost) and generates successor states.
- The algorithm recursively computes the minimax value for each successor state, propagating the values back up the tree.
- For Pacman (the maximizing agent), the algorithm selects the action that leads to the maximum minimax value.
- For ghosts (the minimizing agents), the algorithm selects the action that leads to the minimum minimax value.
- The agent continues the recursive exploration until it reaches the specified depth limit or a terminal state (win or lose).
- Finally, the agent returns the action that leads to the best score for Pacman based on the minimax values.

## Alpha-Beta Agent

The `AlphaBetaAgent` class extends the minimax agent by incorporating alpha-beta pruning to efficiently explore the game tree. Alpha-beta pruning allows the agent to discard certain branches of the tree that are guaranteed to be worse than the current best option, reducing the number of states explored.

The agent maintains alpha and beta values to keep track of the best scores for Pacman and the ghosts, respectively. It updates these values as it explores the tree and prunes branches accordingly.

### Key Logic
- The alpha-beta agent extends the minimax algorithm by incorporating alpha-beta pruning to improve efficiency.
- It maintains alpha and beta values to keep track of the best scores for Pacman and the ghosts, respectively.
- The agent explores the game tree recursively, similar to the minimax agent, but with additional pruning steps.
- For Pacman (the maximizing agent), the algorithm updates the alpha value whenever a better score is found. If the current score exceeds the beta value, the agent can safely prune the remaining branches, as the minimizing agent (ghost) will never choose this path.
- For ghosts (the minimizing agents), the algorithm updates the beta value whenever a worse score is found. If the current score is less than the alpha value, the agent can safely prune the remaining branches, as the maximizing agent (Pacman) will never choose this path.
- The agent continues the recursive exploration until it reaches the specified depth limit or a terminal state (win or lose), or until pruning occurs.
- Finally, the agent returns the action that leads to the best score for Pacman based on the alpha-beta pruned minimax values.

## Monte Carlo Tree Search Agent

The `MonteCarloTreeSearchAgent` class implements an agent that uses Monte Carlo Tree Search (MCTS) to make decisions. MCTS is a heuristic search algorithm that balances exploration and exploitation to find the best action.

The agent builds a tree of game states, where each node represents a state and each edge represents an action. It iteratively selects promising nodes, expands the tree, simulates games to estimate the utility of states, and propagates the results back up the tree.

The `MonteCarloNode` class represents a node in the MCTS tree. It maintains statistics such as the average utility and visit count, and provides methods for selection, expansion, simulation, and backpropagation.

The `getAction` method of the `MonteCarloTreeSearchAgent` performs a specified number of MCTS iterations, starting from the current game state. It then selects the action that leads to the most promising child node.

### Key Logic
- The MCTS agent builds a tree of game states, where each node represents a state and each edge represents an action.
- The agent iteratively performs the following steps:
  1. Selection: Starting from the root node, the agent traverses down the tree, selecting the most promising child node based on the UCB1 (Upper Confidence Bound) formula. The UCB1 formula balances exploration and exploitation by considering both the average utility and the visit count of each node.
  2. Expansion: When the agent reaches a node that has not been fully expanded (i.e., there are untried actions), it selects an untried action and creates a new child node corresponding to the resulting state.
  3. Simulation: From the newly expanded node or a fully expanded node, the agent simulates a random game playout until a terminal state is reached. The simulation follows a random policy, selecting actions uniformly at random.
  4. Backpropagation: After the simulation, the agent propagates the obtained utility value back up the tree, updating the average utility and visit count of each node along the path.
- The agent repeats these steps for a specified number of iterations to build and refine the MCTS tree.
- Finally, the agent selects the action that leads to the most promising child node based on the collected statistics (e.g., the node with the highest average utility or the most visits).

---


