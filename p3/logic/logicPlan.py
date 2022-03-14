# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


# **************
#  Question 1 
# **************
def sentence1():
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** BEGIN YOUR CODE HERE ***"
    expressions = []
    A = Expr('A')
    B = Expr('B')
    C = Expr('C')
    expr1 = A | B
    expr2 = ~A % (~B | C)
    expr3 = disjoin(~A, ~B, C)

    return conjoin(expr1, expr2, expr3)
    "*** END YOUR CODE HERE ***"


# **************
#  Question 1 
# **************
def sentence2():
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** BEGIN YOUR CODE HERE ***"
    expressions = []
    A = Expr('A')
    B = Expr('B')
    C = Expr('C')
    D = Expr('D')
    expr1 = C % (B | D)
    expr2 = A >> (~B & ~D)
    expr3 = ~(B & ~C) >> A
    expr4 = ~D >> C

    return conjoin(expr1, expr2, expr3, expr4)
    "*** END YOUR CODE HERE ***"


# **************
#  Question 1 
# **************
def sentence3():
    """Using the symbols PacmanAlive[1], PacmanAlive[0], PacmanBorn[0], and PacmanKilled[0],
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    """
    "*** BEGIN YOUR CODE HERE ***"
    PacmanAlive_1 = PropSymbolExpr('PacmanAlive', 1)
    PacmanAlive_0 = PropSymbolExpr('PacmanAlive', 0)
    PacmanBorn_0 = PropSymbolExpr('PacmanBorn', 0)
    PacmanKilled_0 = PropSymbolExpr('PacmanKilled', 0)

    expr1 = PacmanAlive_1 % ((PacmanAlive_0 & ~PacmanKilled_0) | (~PacmanAlive_0 & PacmanBorn_0))
    expr2 = ~(PacmanAlive_0 & PacmanBorn_0)
    expr3 = PacmanBorn_0
    return conjoin(expr1, expr2, expr3)
    "*** END YOUR CODE HERE ***"


# **************
#  Question 1 
# **************
def findModel(sentence):
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** BEGIN YOUR CODE HERE ***"
    sentence_cnf = to_cnf(sentence)
    models = pycoSAT(sentence_cnf)
    return models

    "*** END YOUR CODE HERE ***"


def modelToString(model):
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


# **************
#  Question 2 
# **************
def atLeastOne(literals):
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals in the list is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"
    return disjoin(literals)
    "*** END YOUR CODE HERE ***"


# **************
#  Question 2 
# **************
def atMostOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    disjunctions = []

    for i, j in itertools.combinations(literals, 2):
        disjunctions.append(disjoin(~i, ~j))

    return conjoin(disjunctions)  # Returns a conjunction of disjunctions of the negated literals
    "*** END YOUR CODE HERE ***"


# **************
#  Question 2 
# **************
def exactlyOne(literals):
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    return conjoin(atLeastOne(literals), atMostOne(literals))
    "*** END YOUR CODE HERE ***"


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, time = parsed
            plan[int(time)] = action
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


def pacmanSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    possibilities = []
    if not walls_grid[x][y+1]:
        possibilities.append( PropSymbolExpr(var_str, x, y+1, t-1)
                            & PropSymbolExpr('South', t-1))
    if not walls_grid[x][y-1]:
        possibilities.append( PropSymbolExpr(var_str, x, y-1, t-1) 
                            & PropSymbolExpr('North', t-1))
    if not walls_grid[x+1][y]:
        possibilities.append( PropSymbolExpr(var_str, x+1, y, t-1) 
                            & PropSymbolExpr('West', t-1))
    if not walls_grid[x-1][y]:
        possibilities.append( PropSymbolExpr(var_str, x-1, y, t-1) 
                            & PropSymbolExpr('East', t-1))

    if not possibilities:
        return None
    
    return PropSymbolExpr(var_str, x, y, t) % disjoin(possibilities)


def pacmanSLAMSuccessorStateAxioms(x, y, t, walls_grid, var_str=pacman_str):
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    moved_tm1_possibilities = []
    if not walls_grid[x][y+1]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x, y+1, t-1)
                            & PropSymbolExpr('South', t-1))
    if not walls_grid[x][y-1]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x, y-1, t-1) 
                            & PropSymbolExpr('North', t-1))
    if not walls_grid[x+1][y]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x+1, y, t-1) 
                            & PropSymbolExpr('West', t-1))
    if not walls_grid[x-1][y]:
        moved_tm1_possibilities.append( PropSymbolExpr(var_str, x-1, y, t-1) 
                            & PropSymbolExpr('East', t-1))

    if not moved_tm1_possibilities:
        return None

    moved_tm1_sent = conjoin([~PropSymbolExpr(var_str, x, y, t-1) , ~PropSymbolExpr(wall_str, x, y), disjoin(moved_tm1_possibilities)])

    unmoved_tm1_possibilities_aux_exprs = [] # merged variables
    aux_expr_defs = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, t - 1)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, t - 1)
        unmoved_tm1_possibilities_aux_exprs.append(wall_dir_combined_literal)
        aux_expr_defs.append(wall_dir_combined_literal % wall_dir_clause)

    unmoved_tm1_sent = conjoin([
        PropSymbolExpr(var_str, x, y, t-1),
        disjoin(unmoved_tm1_possibilities_aux_exprs)])

    return conjoin([PropSymbolExpr(var_str, x, y, t) % disjoin([moved_tm1_sent, unmoved_tm1_sent])] + aux_expr_defs)


# **************
#  Question 3 
# **************
def pacphysics_axioms(t, all_coords, non_outer_wall_coords):
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
    """
    pacphysics_sentences = []

    "*** BEGIN YOUR CODE HERE ***"
    actions = ['North', 'South', 'East', 'West']

    act_literals = []
    p_coords = []

    for wall in all_coords:
        p_coord = PropSymbolExpr(pacman_str, wall[0], wall[1], t)
        w_coord = PropSymbolExpr(wall_str, wall[0], wall[1])
        pacphysics_sentences.append(w_coord >> ~p_coord)

    for act in actions:
        action = PropSymbolExpr(act, t)
        act_literals.append(action)

    for coord in non_outer_wall_coords:
        p_coord = PropSymbolExpr(pacman_str, coord[0], coord[1], t)
        p_coords.append(p_coord)

    pacphysics_sentences.append(exactlyOne(p_coords))
    pacphysics_sentences.append(exactlyOne(act_literals))

    "*** END YOUR CODE HERE ***"

    return conjoin(pacphysics_sentences)


# **************
#  Question 3 
# **************
def check_location_satisfiability(x1_y1, x0_y0, action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - problem = An instance of logicAgents.LocMapProblem
    Return:
        - a model proving whether Pacman is at (x1, y1) at time t = 1
        - a model proving whether Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    "*** BEGIN YOUR CODE HERE ***"
    pacman_t0 = PropSymbolExpr(pacman_str, x0, y0, 0)
    pacman_t1 = PropSymbolExpr(pacman_str, x1, y1, 1)

    # At t = 0:
    # Add to KB: Pacman’s current location (x0, y0)
    KB.append(pacman_t0)

    # Add to KB: pacphysics_axioms(...)
    KB.append(pacphysics_axioms(0, all_coords, non_outer_wall_coords))

    # Add to KB: Pacman takes action0
    KB.append(PropSymbolExpr(action0, 0))

    # Add to KB: allLegalSuccessorAxioms(t+1, ...)
    KB.append(allLegalSuccessorAxioms(1, walls_grid, non_outer_wall_coords))

    # At t = 1
    # Add to KB: pacphysics_axioms(...)
    KB.append(pacphysics_axioms(1, all_coords, non_outer_wall_coords))

    # Add to KB: Pacman takes action1
    KB.append(PropSymbolExpr(action1, 1))


    KB_conjoin = conjoin(KB)
    # Check satisfiability
    model1 = findModel(conjoin(KB_conjoin, ~pacman_t1))
    model2 = findModel(conjoin(KB_conjoin, pacman_t1))
    models = (model1, model2)
    return models
    "*** END YOUR CODE HERE ***"


# **************
#  Question 4 
# **************
def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal
    
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), 
            range(height + 2)))
    
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    # Add to KB: Initial knowledge: Pacman’s initial location at timestep 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, 0))
    for t in range(50):
        pacman_g = PropSymbolExpr(pacman_str, xg, yg, t)

        # Add to KB: Initial knowledge: Pacman can only be at exactlyOne of
        # the locations in non_wall_coords at timestep t.
        p_coords = []
        for coord in non_wall_coords:
            p_coords.append(PropSymbolExpr(pacman_str, coord[0], coord[1], t))
        KB.append(exactlyOne(p_coords))

        # Is there a satisfying assignment for the variables given the knowledge base so far?
        # Use findModel and pass in the Goal Assertion and KB.
        pacman_g = conjoin(pacman_g, pacmanSuccessorStateAxioms(xg, yg, t, walls))
        model_assert = findModel(conjoin(conjoin(KB), pacman_g))

        # If there is, return a sequence of actions from start to goal using extractActionSequence.
        # Here, Goal Assertion is the expression asserting that Pacman is at the goal at timestep t.
        if(model_assert is not False):
            return extractActionSequence(model_assert, actions)

        # Add to KB: Pacman takes exactly one action per timestep.
        act_literals = []
        for act in actions:
            act_literals.append(PropSymbolExpr(act, t))
        KB.append(exactlyOne(act_literals))

        # Add to KB: Transition Model sentences: call pacmanSuccessorStateAxioms(...) for all possible
        # pacman positions in non_wall_coords.
        for pos in non_wall_coords:
            KB.append(pacmanSuccessorStateAxioms(pos[0], pos[1], t, walls))

    "*** END YOUR CODE HERE ***"



# Helpful Debug Method
def visualize_coords(coords_list, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualize_bool_array(bool_arr, problem):
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)


def sensorAxioms(t, non_outer_wall_coords):
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, t, x + dx, y + dy)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def four_bit_percept_rules(t, percepts):
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def num_adj_walls_percept_rules(t, percepts):
    """
    SLAM uses a weaker num_adj_walls sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    num_adj_walls = sum(percepts)
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def allLegalSuccessorAxioms(t, walls_grid, non_outer_wall_coords): 
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorStateAxioms(
            x, y, t, walls_grid, var_str=pacman_str)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)





# Abbreviations
plp = positionLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
