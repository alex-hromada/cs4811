# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    # *********************
    #    Question 6 
    # *********************
    def __init__(self, **args):
        """ 
        Question 6
        """
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvals = util.Counter()

    # *********************
    #    Question 6 
    # *********************
    def getQValue(self, state, action):
        """
        Question 6 

          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        qValues = self.qvals
        if (state, action) not in qValues:
            qValues[(state, action)] = 0.0
        self.qvals = qValues
        return self.qvals[(state, action)]
        util.raiseNotDefined()

    # *********************
    #    Question 6 
    # *********************
    def computeValueFromQValues(self, state):
        """
        Question 6 

          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        alpha = self.alpha
        gamma = self.discount
        epsilon = self.epsilon
        temp = util.Counter()

        if len(actions) == 0:
          return 0.0
        for action in actions:
          temp[action] = self.getQValue(state, action)
        return temp[temp.argMax()]
        
        
        
        util.raiseNotDefined()

    # *********************
    #    Question 6 
    # *********************
    def computeActionFromQValues(self, state):
        """
        Question 6 

          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        alpha = self.alpha
        gamma = self.discount
        epsilon = self.epsilon
        temp = util.Counter()
        if len(actions) == 0:
          return None
        for action in actions:
          temp[action] = self.getQValue(state, action)
        return temp.argMax()
        #random.choice()
        util.raiseNotDefined()

    # *********************
    #    Question 7 
    # *********************
    def getAction(self, state):
        """
        Question 7 

          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        alpha = self.alpha
        gamma = self.discount
        epsilon = self.epsilon
        coin = util.flipCoin(epsilon)
        lenActions = len(legalActions)
        
        if lenActions != 0:
          if coin:
            action = random.choice(legalActions)
          else:
            action = self.computeActionFromQValues(state)

        if lenActions == 0:
          return action

        return action

    # *********************
    #    Question 6 
    # *********************
    def update(self, state, action, nextState, reward):
        """
        Question 6 

          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        alpha = self.alpha
        gamma = self.discount
        epsilon = self.epsilon
        self.qvals[(state,action)] =  ((1-alpha) * self.getQValue(state,action)) + alpha * (reward + gamma * self.computeValueFromQValues(nextState))
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    # *********************
    #    Question 10
    # *********************
    def getQValue(self, state, action):
        """
        Question 10 

          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        featureExtractor = self.featExtractor
        q = 0
        features = featureExtractor.getFeatures(state, action)
        weights = self.getWeights()
        for feature in features:
          q = q + features[feature] * weights[feature]
        return q
        util.raiseNotDefined()

    # *********************
    #    Question 10 
    # *********************
    def update(self, state, action, nextState, reward):
        """
        Question 10 
        
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        alpha = self.alpha
        gamma = self.discount
        epsilon = self.epsilon
        featureExtractor = self.featExtractor
        q = 0
        features = featureExtractor.getFeatures(state, action)
        weights = self.getWeights()
        qCurrent = self.getQValue(state,action)
        qNext = self.computeValueFromQValues(nextState)

        for feature in features:
          weights[feature] = weights[feature] + alpha * features[feature] * ((reward + gamma * (qNext)) - qCurrent)

        self.weights = weights


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
