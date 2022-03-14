# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Set up food lists
        listofFood = newFood.asList()
        seenFood = []
        foodLeft = []

        # Set current position and cost
        xy1 = newPos
        actionCost = successorGameState.getScore()

        # Set positions of ghosts and capsules
        ghostPosition = successorGameState.getGhostPositions()
        ghostDirection = newGhostStates[0].getDirection()
        capsPosition = successorGameState.getCapsules()

        # Check to see what food is left to eat
        for i in listofFood:
            if i not in seenFood:
                foodLeft.append(i)


        # Check distance from nearest food
        # For current food
        for j in foodLeft:
            # For every other piece of food get the distances between current state and each piece of food
                
            # Take the minimum distance between current state and a piece of food
            distance, foood = min([((((xy1[0] - food[0]) ** 2 + (xy1[1] - food[1]) ** 2 ) ** 0.5), food) for food in foodLeft])

            # Calculate a cost for that food to get there
            actionCost = actionCost + (0.31/distance)

            # Move to next piece of food   
            xy1 = foood

            # Remove piece of food after it has been eaten 
            foodLeft.remove(foood)

        # Check the distance from ghost
        for k in ghostPosition:
            ghostDistance = min([((util.manhattanDistance(newPos, ghost))) for ghost in ghostPosition])
            # Make sure to stay away from the ghost
            if ghostDistance < 3 and max(newScaredTimes) == 0:
                actionCost = actionCost + (ghostDistance * 0.4)
        
        # Check see if current position has a piece of food, calculate cost if it is there
        for l in capsPosition:
            if l and newPos:
                actionCost = actionCost + 100

        "*** YOUR CODE HERE ***"
        return actionCost

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
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Choosing the min or max value needed at the current state
        def chooseValue(gameState, agent, depth):

            # Check to see if all agents for current depth have been proccessed
            if agent == gameState.getNumAgents():
                depth = depth + 1
                agent = agent % agent

            # Check if at lowest depth
            if depth == self.depth:
                return ("", self.evaluationFunction(gameState))

            # Check for a losing or winning state
            if gameState.isWin():
                return ('', self.evaluationFunction(gameState))

            if gameState.isLose():
                return ('', self.evaluationFunction(gameState))
            
            # If agent is pacman get max value
            # If agent is a ghost get min value
            else: 
                if agent == 0:
                    return getMax(gameState, agent, depth)
                else:
                    return getMin(gameState, agent, depth)

            

        def getMax(gameState, agent, depth):

            # Set minimum value, pair (action, value), and all legal actions
            value = float('-Inf')
            pair = ("", value)
            legalActions = gameState.getLegalActions(agent)
            
            # For each actiction
            for actions in legalActions:

                # Create successors for current agent and action
                successors = gameState.generateSuccessor(agent, actions)
                # Get minimax value for current action
                successValue = chooseValue(successors, agent + 1, depth)
                # If new value is greater than current update date it and the pair with current action
                if successValue[1] > value:
                    
                    pair = (actions, successValue[1])
                    value = successValue[1]
            
            return pair


        def getMin(gameState, agent, depth):

            # Set minimum value, pair (action, value), and all legal actions
            value = float('Inf')
            pair = ("", value)
            legalActions = gameState.getLegalActions(agent)

            # For each actiction
            for actions in legalActions:

                # Create successors for current agent and action
                successors = gameState.generateSuccessor(agent, actions)
                # Get minimax value for current action
                successValue = chooseValue(successors, agent + 1 , depth)
                # If new value is less than current update date it and the pair with current action
                if successValue[1] < value:

                    pair = (actions, successValue[1])
                    value = successValue[1]
            
            return pair

        legalActions = gameState.getLegalActions(0)

        return chooseValue(gameState, 0, 0)[0]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def chooseValue(gameState, agent, depth, alpha, beta):

            # Check to see if all agents have been proccessed
            if agent == gameState.getNumAgents():
                depth = depth + 1
                agent = agent % agent

            # Check if at lowest depth
            if depth == self.depth:
                return ('', self.evaluationFunction(gameState))

            # Check for a losing or winning state
            if gameState.isWin():
                return ('', self.evaluationFunction(gameState))

            if gameState.isLose():
                return ('', self.evaluationFunction(gameState))
            
            # If agent is pacman get max value
            # If agent is a ghost get min value
            else: 
                if agent == 0:
                    return getMax(gameState, agent, depth, alpha, beta)
                else:
                    return getMin(gameState, agent, depth, alpha, beta)

            

        def getMax(gameState, agent, depth, alpha, beta):

            # Set minimum value, pair (action, value), and all legal actions
            value = float('-Inf')
            pair = ('', value)
            legalActions = gameState.getLegalActions(agent)
            
            # For each actiction
            for actions in legalActions:

                # Create successors for current agent and action
                successors = gameState.generateSuccessor(agent, actions)
                # Get minimax value for current action
                successValue = chooseValue(successors, agent + 1, depth, alpha, beta)
                # If new value is greater than current update date it and the pair
                if successValue[1] > value:
                    pair = (actions, successValue[1])
                    value = successValue[1]

                # Checking to see if current value is greater than beta
                # If true no need to continue checking
                if value > beta:
                    return (actions, value)

                # Update alpha if the current value is greater than alpha
                alpha = max(alpha, value)
            
            return pair


        def getMin(gameState, agent, depth, alpha, beta):

            # Set minimum value, pair (action, value), and all legal actions
            value = float('Inf')
            pair = ('', value)
            legalActions = gameState.getLegalActions(agent)

            # For each actiction
            for actions in legalActions:

                # Create successors for current agent and action
                successors = gameState.generateSuccessor(agent, actions)
                # Get minimax value for current action
                successValue = chooseValue(successors, agent + 1, depth, alpha, beta)
                # If new value is less than current update date it and the pair
                if successValue[1] < value:

                    pair = (actions, successValue[1])
                    value = successValue[1]

                # Checking to see if current value is less than alpha
                # If true no need to continue checking
                if value < alpha:
                    return (actions, value)

                # Update beta if the current value is less than beta
                beta = min(beta, value)
            
            return pair

        legalActions = gameState.getLegalActions(0)
        alpha = float('-Inf')
        beta  = float('Inf')
        return chooseValue(gameState, 0, 0, alpha, beta)[0]

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"


        # Choosing the min or max value needed at the current state
        def chooseValue(gameState, agent, depth):

            # Check to see if all agents have been proccessed
            #print('agents ', agent)
            if agent == gameState.getNumAgents():
                depth = depth + 1
                agent = agent % agent

            # Check if at lowest depth
            if depth == self.depth:
                #print('evalulation function', self.evaluationFunction(gameState))
                return ("", self.evaluationFunction(gameState))

            # Check for a losing or winning state
            if gameState.isWin():
                return ('', self.evaluationFunction(gameState))

            if gameState.isLose():
                return ('', self.evaluationFunction(gameState))
            
            # If agent is pacman get max value
            # If agent is a ghost get min value
            else: 
                if agent == 0:
                    return getMax(gameState, agent, depth)
                else:
                    return getMin(gameState, agent, depth)

            

        def getMax(gameState, agent, depth):

            # Set minimum value, pair (action, value), and all legal actions
            value = float('-Inf')
            pair = ("", value)
            legalActions = gameState.getLegalActions(agent)
            expectiMax = 0
            
            # For each actiction
            for actions in legalActions:

                # Create successors for current agent and action
                successors = gameState.generateSuccessor(agent, actions)
                # Get minimax value for current action
                successValue = chooseValue(successors, agent + 1, depth)
                # If new value is greater than current update date it and the pair
                if successValue[1] > value:

                    pair = (actions, successValue[1])
                    value = successValue[1]
            
            return pair


        def getMin(gameState, agent, depth):

            # Set minimum value, pair (action, value), and all legal actions
            value = 0
            pair = ("", value)
            legalActions = gameState.getLegalActions(agent)
            expectiMin = 1/len(legalActions)

            # For each actiction
            for actions in legalActions:

                # Create successors for current agent and action
                successors = gameState.generateSuccessor(agent, actions)
                # Get minimax value for current action
                successValue = chooseValue(successors, agent + 1 , depth)
                # Get expectiMin value with paired action
                value = value + (successValue[1] * expectiMin)
                pair = (actions, value)
                    
            return pair

        legalActions = gameState.getLegalActions(0)

        return chooseValue(gameState, 0, 0)[0]



        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    # Evaluation Score and set of visited elements0
    value = 0
    closestThings= []

    # Check for each ghost the distance to pacman
    for ghost in ghostStates:
        distance = manhattanDistance(position, ghost.getPosition())
        closestThings.append(distance)
    closestGhost = min(closestThings)
    # Decrese value based on how close the ghost, the closer the ghost the greater decrease
    if closestGhost:
        value = value - 100 * (1/closestGhost)
    # Pacman has hit a ghost
    else:
        value = value - 1000
    
    # Rest visited list and grab positions of all the pieces of food
    closestThings = []
    foodList = food.asList()

    # Maker sure there is food and calculate the distance to pacman
    if foodList:

        for food in foodList:
            distance = manhattanDistance(position, food)
            closestThings.append(distance)
        # Decrease value based on how far the food is, the closer the better the score is
        cloestPieceofFood = min(closestThings)
        value = value - cloestPieceofFood

    else:
        value = value
    
    # Reset list
    closestThings = []

    # Check if any capsules exist and grab their distance to pacman
    if len(capsules) != 0:

        for cap in capsules:
            distance = manhattanDistance(position, cap)
            closestThings.append(distance)
        #if len(closestThings) != 0:
        # Update value based on how close the capsule is, the closer the better
        closestCapsule = min(closestThings)
        value = value - 40 * (1/closestCapsule)

    else:
        value = value

    # If a capsule is eatten then try and eat the ghost
    if scaredTimes[0] != 0:
    	value = value + (3000/closestGhost)

    # Decrease value if food is leftß
    value = value - 1000 * (len(foodList))

    return value
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
