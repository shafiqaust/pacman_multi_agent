
from util import manhattanDistance
from game import Directions
import random, util
import math
from featureExtractors import *
from pacman import GameState
from game import Agent

def scoreEvaluationFunction(currentGameState):
    utility = currentGameState.getScore()
    return utility



class ReflexAgent(Agent):
    """
    A reflex agent makes decisions at each decision point by evaluating its possible actions 
    based on a predefined state evaluation function. This function assesses the desirability of
    future states resulting from possible actions, without planning ahead further than one move.
    
    This template is provided as a starting point, and can be modified as needed except for the 
    method headers.
    """

    def getAction(self, gameState: GameState):
        """
        Selects the best action for Pacman based on the evaluation function.

        Parameters:
            gameState (GameState): The current state of the game from which Pacman must decide on an action.

        Returns:
            Directions.X: The direction (as part of the Directions enum) that represents the best move for Pacman.

        Methodology:
            1. Gather all legal moves Pacman can make from the current state.
            2. Evaluate each move using the evaluation function.
            3. Choose the move with the highest score. If multiple moves have the same high score, one is chosen randomly.
        """
        # Collect all possible actions Pacman can take in the current game state
        legalMoves = gameState.getLegalActions()
        # print(f"Legal moves available: {legalMoves}")

        # Evaluate each action using the evaluation function and store the scores
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # print(f"Scores for each move: {list(zip(legalMoves, scores))}")

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # print(f"Best score: {bestScore}, Indices with best score: {bestIndices}")

        # Randomly choose among the best actions
        chosenIndex = random.choice(bestIndices)  # This provides some variety in Pacman's behavior
        # print(f"Chosen move: {legalMoves[chosenIndex]} based on index {chosenIndex}")
        
        return legalMoves[chosenIndex]  # Return the best action
    
    def evaluationFunction(self, currentGameState, action):
        """
        Computes a numeric value representing the desirability of the resulting state after taking a given action.

        Parameters:
            currentGameState (GameState): The current state of the game.
            action (str): A legal action Pacman can take from the currentGameState.

        Returns:
            float: A score that represents the desirability of the resulting state from performing the action.

        Evaluation:
            - The function computes the successor state resulting from the action.
            - The score increases based on proximity to food and the state's game score.
            - The score adjusts based on proximity to non-scared ghosts and the benefit of scared ghosts.
            - Close encounters with non-scared ghosts heavily penalize the score to avoid losing the game.
        """
        # Generate the next game state after taking the proposed action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        # print(f"New position after action '{action}': {newPos}")
        # print(f"Action '{action}', New ghost states: {[ghost.getPosition() for ghost in newGhostStates]}")

        # Start with the game score of the successor state
        score = successorGameState.getScore()
        # print(f"Initial score from game state for action '{action}': {score}")

        # Encourage getting closer to the nearest piece of food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            score += 10.0 / min(foodDistances)  # Closer food increases score more significantly
            # print(f"Food distances after action '{action}': {foodDistances}, Score adjustment: +{10.0 / min(foodDistances)}")

        # Adjust score based on ghost proximity
        for ghostState in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            if ghostState.scaredTimer > 0 and ghostDistance > 0:
                # Bonus for being near scared ghosts
                score += 10.0 / ghostDistance
                print(f"Near scared ghost at distance {ghostDistance}: +{10.0 / ghostDistance}")
            elif ghostDistance < 2:
                # Heavy penalty for being too close to a non-scared ghost
                score -= 20.0
                print(f"Too close to a non-scared ghost at distance {ghostDistance}: -20")

        return score




def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()





class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)





class MinimaxAgent(MultiAgentSearchAgent):
    """
    A Minimax agent implements the Minimax algorithm for decision making in a game environment with multiple agents.
    The Minimax algorithm considers each agent's turn at each depth level:
    - Pacman is the maximizing agent trying to increase its score.
    - Ghosts are minimizing agents aiming to reduce Pacman's score.
    """

    def getAction(self, gameState):
        """
        Computes the best action for the current game state using the Minimax algorithm.
        
        Args:
            gameState (GameState): The current state of the game.
            
        Returns:
            The best action for Pacman as determined by the Minimax algorithm.
        """

        def minimax(agentIndex, depth, gameState):
            """
            Recursively computes the minimax value of game states.
            
            Args:
                agentIndex (int): The current agent's index.
                depth (int): The current depth in the game tree.
                gameState (GameState): The current game state being evaluated.
                
            Returns:
                The minimax value of the state.
            """
            # Check for terminal state or if maximum depth is reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                # print(f"Evaluating game state at depth {depth} with result {self.evaluationFunction(gameState)}")
                return self.evaluationFunction(gameState)

            # Determine the next agent and increment depth if it wraps around to Pacman
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth

            # Collect all legal moves for the current agent
            actions = gameState.getLegalActions(agentIndex)
            # print(f"Agent {agentIndex} actions at depth {depth}: {actions}")

            # If no legal moves, evaluate the current state
            if not actions:
                return self.evaluationFunction(gameState)

            # Recursively call minimax on all legal successor states and gather results
            results = [minimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in actions]

            # Select the appropriate action: maximize for Pacman, minimize for Ghosts
            if agentIndex == 0:  # Pacman's turn, maximize
                bestResult = max(results)
            else:  # Ghosts' turn, minimize
                bestResult = min(results)

            # print(f"Best result for agent {agentIndex} at depth {depth}: {bestResult}")
            return bestResult

        # Start minimax from Pacman (agent 0) at the current game state at depth 0
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):  # Pacman is agent 0
            score = minimax(1, 0, gameState.generateSuccessor(0, action))
            # print(f"Action {action} score: {score}")
            if score > bestScore:
                bestScore = score
                bestAction = action
                # print(f"New best action found: {action} with score: {score}")

        return bestAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    An agent that uses the alpha-beta pruning algorithm within the Minimax framework.
    This algorithm is designed to efficiently decide the best possible action in a multi-agent
    environment by reducing the number of nodes evaluated in the Minimax decision tree.
    - Pacman is the maximizing agent trying to maximize his score.
    - Ghosts are minimizing agents trying to minimize Pacman's score.
    """

    def getAction(self, gameState: GameState):
        """
        Determines the best action for Pacman to take from the current game state
        using the Minimax strategy with alpha-beta pruning to enhance performance.

        Args:
            gameState (GameState): The current state from which the agent will determine its action.

        Returns:
            The optimal action for the current game state as determined by the alpha-beta algorithm.
        """

        def maxValue(gameState, depth, alpha, beta):
            """
            Compute the maximum possible score Pacman can achieve from the given game state.

            Args:
                gameState (GameState): The current game state being evaluated.
                depth (int): The current depth in the game tree.
                alpha (float): The best already explored option along the path to the root for the maximizer.
                beta (float): The best already explored option along the path to the root for the minimizer.

            Returns:
                The maximum score achievable from this state (float).
            """
            if gameState.isWin() or gameState.isLose() or depth == 0:
                # Terminal state or maximum depth reached, evaluate using the evaluation function
                return self.evaluationFunction(gameState)

            value = float('-inf')
            for action in gameState.getLegalActions(0):  # Pacman's actions
                successor = gameState.generateSuccessor(0, action)
                # Recursively find the minimal score the opponent can achieve
                value = max(value, minValue(successor, depth, alpha, beta, 1))
                if value > beta:
                    # Alpha-beta pruning: if max's value is greater than beta, prune the branch
                    print(f"Pruning at maxValue with value {value} > beta {beta}")
                    return value
                alpha = max(alpha, value)  # Update alpha
                print(f"Updated alpha in maxValue to {alpha}")

            return value

        def minValue(gameState, depth, alpha, beta, agentIndex):
            """
            Compute the minimum possible score Pacman's opponents can force from the given game state.

            Args:
                gameState (GameState): The current game state being evaluated.
                depth (int): The current depth in the game tree.
                alpha (float): The alpha cutoff value for pruning.
                beta (float): The beta cutoff value for pruning.
                agentIndex (int): The index of the current agent (ghost).

            Returns:
                The minimum score achievable from this state (float).
            """
            if gameState.isWin() or gameState.isLose() or depth == 0:
                # Terminal state or depth limit reached, evaluate using the evaluation function
                return self.evaluationFunction(gameState)

            value = float('inf')
            for action in gameState.getLegalActions(agentIndex):  # Actions available to the ghost
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    # If last ghost, next is Pacman's turn at a deeper level
                    value = min(value, maxValue(successor, depth - 1, alpha, beta))
                else:
                    # Next is another ghost's turn at the same depth
                    value = min(value, minValue(successor, depth, alpha, beta, agentIndex + 1))
                if value < alpha:
                    # Alpha-beta pruning: if min's value is less than alpha, prune the branch
                    print(f"Pruning at minValue with value {value} < alpha {alpha}")
                    return value
                beta = min(beta, value)  # Update beta
                print(f"Updated beta in minValue to {beta}")

            return value
        
        alpha = float('-inf')  # Worst possible score for maximizer
        beta = float('inf')    # Best possible score for minimizer
        bestAction = None
        bestValue = float('-inf')
        
        for action in gameState.getLegalActions(0):  # Pacman's possible actions
            successor = gameState.generateSuccessor(0, action)
            value = minValue(successor, self.depth, alpha, beta, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
                print(f"New best action {action} with value {value}")
            alpha = max(alpha, bestValue)
            print(f"Global alpha updated to: {alpha}")

        return bestAction





class MonteCarloNode:
    def __init__(self, game_state, parent_move=None, exploration_factor=2, max_steps=50):
        '''
        Initializes a Monte Carlo Tree Search node.
        
        Parameters:
            game_state (GameState): Current state of the game.
            parent_move (str): Move that led to this state.
            exploration_factor (float): Coefficient for exploration in UCB1.
            max_steps (int): Maximum steps allowed in simulation to prevent infinite loops.

        Attributes:
            average_utility (float): Running average of the node's utility value.
            visit_count (int): Number of times the node has been visited.
            child_nodes (list): Child nodes generated from this node.
        '''
        self.average_utility = 0
        self.visit_count = 0
        self.child_nodes = []
        self.exploration_factor = exploration_factor
        self.current_state = game_state
        self.parent_move = parent_move
        self.max_steps = max_steps
    
    def process(self):
        
        """
        Processes the node by simulation, expansion, or selecting and processing a child node.
        Updates utility and visit count based on the result.
        """
  
        if self.is_terminal():
            # If the node has no children, check if it has been visited to decide between simulation or expansion.
            if self.visit_count == 0:
                self.average_utility = self.simulate()
                self.visit_count = 1
            else:
                self.expand()
        
        # If no children exist even after attempting expansion, simulate from this node.
        if not self.child_nodes:
            result = self.simulate()
        else:
            # Otherwise, recursively process the selected child to continue the search.
            result = self.child_nodes[self.choose()].process()

        # Backpropagation step: update the average utility and visit count.
        self.average_utility += result
        self.visit_count += 1

        return result     

    def simulate(self):
        """
        Simulates the outcome from the current game state by playing out random moves until a terminal state.
        Returns the utility of the simulated outcome.
        """

        simulation_state = self.current_state
        num_players = simulation_state.getNumAgents()
        reached_end = False
        step_count = 0
        # Continue simulation until a terminal state or the step limit is reached.
        while not reached_end:
            for player_id in range(num_players):
                step_count += 1
                reached_end = simulation_state.isWin() or simulation_state.isLose() or step_count > self.max_steps
                if reached_end:
                    break
                possible_moves = simulation_state.getLegalActions(player_id)
                # Randomly select a move and advance the game state.
                simulation_state = simulation_state.generateSuccessor(player_id, random.choice(possible_moves))

        # Evaluate the resulting state using a utility function specific to the game.
        evaluation = self.evaluate(simulation_state, 'food') / (simulation_state.getWalls().width + simulation_state.getWalls().height)
        if simulation_state.isWin():
            evaluation += 1
        elif simulation_state.isLose():
            evaluation -= 1
        return evaluation

    def expand(self):
        
        """
        Expands the node by creating new child nodes for each possible action from this state.
        """

        for move in self.current_state.getLegalActions(0):
            new_state = self.current_state.generateSuccessor(0, move)
            self.child_nodes.append(MonteCarloNode(new_state, move, self.exploration_factor, self.max_steps))

    def choose(self):
        # Selects the best child to explore next, based on the Upper Confidence Bound (UCB1) algorithm.
        ucb_scores = []
        index = 0
        # Calculate the UCB1 score for each child; higher scores are more promising for exploration.
        for node in self.child_nodes:
            if node.visit_count > 0:
                ucb_scores.append(node.average_utility + self.exploration_factor * math.sqrt(math.log(self.visit_count) / node.visit_count))
            else:
                # Immediately select any unvisited child.
                return index
            index += 1
        return max(range(len(ucb_scores)), key=lambda i: ucb_scores[i])

    def is_terminal(self):
        # Checks if this node is terminal, which is true if there are no children.
        return len(self.child_nodes) == 0

    def evaluate(self, state, arg='ghost'):
        # Evaluates the game state based on specified criteria ('ghost' proximity or 'food' distance).
        if arg == 'ghost':
            ghost_info = state.getGhostStates()
            pacman_position = state.getPacmanPosition()
            ghost_distances = [manhattanDistance(ghost.getPosition(), pacman_position) * (1 + 3 * int(ghost.scaredTimer > 0)) for ghost in ghost_info]
            return min(ghost_distances)
        elif arg == 'food':
            foods = state.getFood()
            pacman_position = state.getPacmanPosition()
            food_distances = [1/manhattanDistance(food, pacman_position) for food in foods.asList()]
            closest_food_distance = max(food_distances) if food_distances else 0
            return closest_food_distance

    def optimal_move(self):
        # Determines the best move to make from this node, based on the utility of its children.
        best_node = self.child_nodes[self.choose()]
        move_quality = {}
        proximity_to_ghosts = self.evaluate(self.current_state, 'ghost')
        # Compare utilities of actions and choose the best one.
        for node in self.child_nodes:
            if abs(node.average_utility - best_node.average_utility) < 1:
                if proximity_to_ghosts < 3:
                    move_quality[node.parent_move] = self.evaluate(node.current_state, 'ghost')
                else:
                    path, _ = closestLoc(self.current_state.getPacmanPosition(), self.current_state.getFood().asList(), self.current_state.getWalls())
                    move_quality[node.parent_move] = int(node.parent_move == path[0]) if path else 0
                    if Directions.STOP in move_quality and len(move_quality) > 1:
                        del move_quality[Directions.STOP]
        chosen_move = max(move_quality, key=move_quality.get)
        return chosen_move

class MonteCarloTreeSearchAgent(MultiAgentSearchAgent):
    """
    Implements a reflex agent that uses Monte Carlo Tree Search to decide actions.
    """
    def __init__(self, feature_extractor='SimpleExtractor'):
        self.feature_extractor = util.lookup(feature_extractor, globals())()
        MultiAgentSearchAgent.__init__(self)


    def MCTS_decision(self, gameState):
        # Performs the Monte Carlo Tree Search starting from the initial game state.
        root_node = MonteCarloNode(game_state=gameState)
        iterations = 50
        while iterations:
            root_node.process()
            iterations -= 1
        return root_node.optimal_move()

    def getAction(self, gameState):
        # Returns the action determined by the MCTS.
        return self.MCTS_decision(gameState)


