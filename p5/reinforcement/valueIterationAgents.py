# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    # *********************
    #    Question 1
    # *********************
    def runValueIteration(self):
        # Write value iteration code here
        """ 
        Question 1: runValueIteration method 
        """
        "*** YOUR CODE HERE ***"

        mdp = self.mdp
        gamma = self.discount
        states = mdp.getStates()
        iterations = self.iterations 
        for x in range(0, iterations):
            values = util.Counter()
            for state in states:
                if mdp.isTerminal(state):
                    values[state] = 0
                else:
                    actions = mdp.getPossibleActions(state)
                    maxvalue = float("-inf")
                    for action in actions:
                        v = 0
                        trans_probs = mdp.getTransitionStatesAndProbs(state, action)
                        for trans, prob in trans_probs:
                            v += prob * (mdp.getReward(state, action, trans) + gamma * self.values[trans])
                        maxvalue = max(v, maxvalue)
                        values[state] = maxvalue

            self.values = values  

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    # *********************
    #    Question 1
    # *********************
    def computeQValueFromValues(self, state, action):
        """
        Question 1 

          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        gamma = self.discount
        values = self.values
        # print('values ', self.values)
        # print('iterations ', self.iterations)
        # print('value keys ', self.values.sortedKeys())
        # print('mdp ', mdp.getStates())
        q = 0
        trans_probs = mdp.getTransitionStatesAndProbs(state, action)
        for temp in trans_probs:
            q += temp[1] * (mdp.getReward(state, action, temp[0]) + (gamma * values[temp[0]]))
        return q

        # util.raiseNotDefined()

    # *********************
    #    Question 1 
    # *********************
    def computeActionFromValues(self, state):
        """
        Question 1 

          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        actions = mdp.getPossibleActions(state)
        if mdp.isTerminal(state):
            return None
        v = float("-inf")
        move = None
        for action in actions:
            temp = self.computeQValueFromValues(state, action)
            if temp >= v:
                v = temp
                move = action
        return move
        #     transition = mdp.getTransitionStatesAndProbs(state, action)[0]
        #     prob = transition[1]
            # reward = mdp.getReward(state, action, transition[0])
            # print('reward ', reward)
        #     print('trans ', transition)
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    # *********************
    #    Question 4
    # *********************
    def runValueIteration(self):
        """
        Question 4
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        gamma = self.discount
        states = mdp.getStates()
        iterations = self.iterations
        values = self.values
        i = 0
        terminalState = mdp.isTerminal(states)

        for x in range(0, iterations):
            
            i = states[x % len(states)]
            terminalState = mdp.isTerminal(i)
 
            if not terminalState:
                action = self.getAction(i)
                values[i] = self.computeQValueFromValues(i, action)
    
        self.values = values



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    # *********************
    #    Question 5 
    # *********************
    def runValueIteration(self):
        """
        Question 5 - Extra Credit
        """
        "*** YOUR CODE HERE ***"

