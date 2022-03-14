# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class Node:

    """
    This class defines a node structure used to store data
    of a state, its parent, its actions, and path-cost
    """
    def __init__(self, state, parent = None, action = None, pathCost = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.pathCost = pathCost

    def getState(self):
        """
        Returns the state of current node
        """
        return self.state

    def getParent(self):
        """
        Returns the parent node of the current node
        """
        return self.parent

    def getAction(self):
        """
        Returns the action required to move to the current node
        """
        return self.action

    def getPathCost(self):
        """
        Returns the path-cost required to move to the current node
        """
        return self.pathCost


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    curr = problem.getStartState()

    closed = list() # Initialize closed set
    frontier = util.Stack() # Initialize frontier
    frontier.push(Node(curr)) # Add start state to frontier

    while True:
        if frontier.isEmpty():
            util.raiseNotDefined()

        currNode = frontier.pop() # Pop node off of frontier

        # Check if node is goal state
        if problem.isGoalState(currNode.getState()):
            soln = [] # Intitialize solution list

            # Iterate over solution path nodes
            while currNode.getParent() is not None:
                soln.insert(0, currNode.getAction())
                currNode = currNode.getParent()
            return soln

        # Check if state is in closed list
        if currNode.getState() not in closed:
            closed.append((currNode.getState())) # Add current state to the closed list
            # Iterate over successor nodes to current node and add to the frontier
            for i in problem.getSuccessors(currNode.getState()):
                frontier.push(Node(i[0], currNode, i[1], currNode.getPathCost() + i[2]))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    curr = problem.getStartState()

    closed = list() # Initialize closed set
    frontier = util.Queue() # Initialize frontier
    frontier.push(Node(curr)) # Add start state to frontier

    while True:
        if frontier.isEmpty():
            util.raiseNotDefined()

        currNode = frontier.pop() # Pop node off of frontier

        # Check if node is goal state
        if problem.isGoalState(currNode.getState()):
            soln = []

            # Iterate over solution path nodes
            while currNode.getParent() is not None:
                soln.insert(0, currNode.getAction())
                currNode = currNode.getParent()
            print(soln)
            return soln

        # Check if state is in closed list
        if currNode.getState() not in closed:
            closed.append((currNode.getState())) # Add current state to the closed list
            # Iterate over successor nodes to current node and add to the frontier
            for i in problem.getSuccessors(currNode.getState()):
                frontier.push(Node(i[0], currNode, i[1], currNode.getPathCost() + i[2]))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    curr = problem.getStartState()

    closed = list() # Initialize closed set
    frontier = util.PriorityQueue() # Initialize frontier
    frontier.push(Node(curr), 0) # Add start state to frontier

    while True:
        if frontier.isEmpty():
            util.raiseNotDefined()

        currNode = frontier.pop() # Pop node off of frontier

        # Check if node is goal state
        if problem.isGoalState(currNode.getState()):
            soln = []

            # Iterate over solution path nodes
            while currNode.getParent() is not None:
                soln.insert(0, currNode.getAction())
                currNode = currNode.getParent()
            return soln

        # Check if state is in closed list
        if currNode.getState() not in closed:
            closed.append((currNode.getState())) # Add current state to the closed list
            # Iterate over successor nodes to current node and add to the frontier
            for i in problem.getSuccessors(currNode.getState()):
                frontier.push(Node(i[0], currNode, i[1], currNode.getPathCost() + i[2]), currNode.getPathCost() + i[2])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start = problem.getStartState()

    closed = list() # Initialize closed set
    frontier = util.PriorityQueue() # Initialize frontier
    frontier.push(Node(start), heuristic(start, problem)) # Add start state to frontier

    while True:
        if frontier.isEmpty():
            util.raiseNotDefined()

        currNode = frontier.pop() # Pop node off of frontier

        # Check if node is goal state
        if problem.isGoalState(currNode.getState()):
            soln = []

            # Iterate over solution path nodes
            while currNode.getParent() is not None:
                soln.insert(0, currNode.getAction())
                currNode = currNode.getParent()
            return soln

        # Check if state is in closed list
        if currNode.getState() not in closed:
            closed.append((currNode.getState())) # Add current state to the closed list

            # Iterate over successor nodes to current node and add to the frontier
            for i in problem.getSuccessors(currNode.getState()):
                frontier.push(Node(i[0], currNode, i[1], currNode.getPathCost() + i[2]), currNode.getPathCost() + i[2] + heuristic(i[0], problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
