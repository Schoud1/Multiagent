#Sukhmani Choudhry
#Schoud9@emory.edu
# /*THIS  CODE  WAS MY OWN WORK , IT WAS  WRITTEN  WITHOUT  CONSULTING  ANY SOURCES  OUTSIDE  OF  THOSE  APPROVED  BY THE  INSTRUCTOR. _Sukhmani_Choudhry_*/

# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor)
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Prioritize staying away from ghosts
        score = successorGameState.getScore()

        # The farther we are from ghosts, the better.
        minGhostDist = 9999999
        for ghost in newGhostStates:
            dist = util.manhattanDistance(newPos, ghost.getPosition()) #the score will get higher as pacman gets farther from the ghost

            if dist < minGhostDist:
                minGhostDist = dist

        score += minGhostDist

        # The farther we are from food, the worse
        if currentFood[newPos[0]][newPos[1]]:
            score += 100 # it gives score boost as pacman eats
        else:
            minFoodDist = 9999999

            for food in newFood.asList():
                dist = util.manhattanDistance(food, newPos) #find distance from ghost

                if dist < minFoodDist:
                    minFoodDist = dist

            score -= minFoodDist #the farther the pacman is from food, the more it gets penalized

        # don't let pacman stand still!
        if action == Directions.STOP:
            score -= 100 #score decreases

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
        """
        bestScore = -(float("inf"))
        bestAction = None

        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            score = self.getScore(newState, self.depth, 0) #best score

            if score > bestScore:
                bestScore = score # if the score is bigger than best score, then bestscore is assigned to the score
                bestAction = action

        return bestAction

    def getScore(self, gameState, depth, agent):
        agent += 1 #calculate the next agent and increase depth

        if agent == gameState.getNumAgents():
            agent = 0
            depth -= 1 #depth decreases

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agent == 0: #maximize for pacman
            bestScore = -(float("inf")) #maximize score

            for action in gameState.getLegalActions(agent):
                newState = gameState.generateSuccessor(agent, action)
                score = self.getScore(newState, depth, agent)

                if score > bestScore:
                    bestScore = score #updates bestscore if condition is met

            return bestScore
        else:
            bestScore = float("inf")

            for action in gameState.getLegalActions(agent):
                newState = gameState.generateSuccessor(agent, action)
                score = self.getScore(newState, depth, agent)

                if score < bestScore:
                    bestScore = score #updates bestscore if condition is met

            return bestScore


class AlphaBetaAgent(MultiAgentSearchAgent): #best possible move that pacman can make
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def val(gameState, alpha, beta, depth, agent):
            if agent == gameState.getNumAgents():
                agent = 0
                depth -= 1 #depth decreases

            if depth == 0 or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState)

            if agent == 0: #maximize for pacman
                return maxVal(gameState, alpha, beta, depth, agent)
            else: #minimize for ghosts
                return minVal(gameState, alpha, beta, depth, agent)

        def minVal(gameState, alpha, beta, depth, agent): #minimizer function
            v = float("inf")
            for action in gameState.getLegalActions(agent):
                newState = gameState.generateSuccessor(agent, action)
                v = min(v, val(newState, alpha, beta, depth, agent + 1))

                if v < alpha:
                    return v

                beta = min(v, beta)

            return v

        def maxVal(gameState, alpha, beta, depth, agent): #maximizer function
            v = -(float("inf"))
            for action in gameState.getLegalActions(agent):
                newState = gameState.generateSuccessor(agent, action)
                v = max(v, val(newState, alpha, beta, depth, agent + 1))

                if v > beta:
                    return v

                alpha = max(v, alpha)

            return v

        v = -(float("inf"))
        alpha = -(float("inf"))
        beta = float("inf")
        best = None

        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action) #newstate is updated
            s = val(newState, alpha, beta, self.depth, 1)

            if s > v:
                v = s
                best = action #if conditions are met then best is equal to action
            if s > beta:
                return s

            alpha = max(alpha, s)

        return best

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
        def val(gameState, depth, agent):
            if agent == gameState.getNumAgents():
                agent = 0
                depth -= 1

            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agent == 0:
                return maxVal(gameState, depth, agent)
            else:
                return randVal(gameState, depth, agent)

        def maxVal(gameState, depth, agent): #best possible move for pacman
            bestScore = -(float("inf"))

            for action in gameState.getLegalActions(agent):
                newState = gameState.generateSuccessor(agent, action)
                bestScore = max(bestScore, val(newState, depth, agent + 1))

            return bestScore

        def randVal(gameState, depth, agent): #checking for ghosts
            legalActions = gameState.getLegalActions(agent)
            weight = 1.0 / len(legalActions)
            score = 0

            for action in legalActions:
                newState = gameState.generateSuccessor(agent, action)
                score += weight * val(newState, depth, agent + 1)

            return score #balances

        bestScore = -(float("inf"))
        bestMove = None

        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            score = val(newState, self.depth, 1)

            if score > bestScore:
                bestScore = score
                bestMove = action

        return bestMove

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Make food super important. Getting closer and closer to food yields greater
                   and greater rewards
    """
    pacPos = currentGameState.getPacmanPosition()

    foods = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    closestFood = float("inf")
    for food in foods:
        dist = util.manhattanDistance(food, pacPos)

        if dist < closestFood:
            closestFood = dist

    score += 2 ** (-1 * closestFood) #closer the food, the better

    return score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()



