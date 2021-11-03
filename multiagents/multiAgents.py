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
import random
import util
import sys


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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        #print(successorGameState)
        #print(newPos)
        #print(newFood)
        #print('---------------------')
        #for ghostState in newGhostStates:
        #    print(ghostState)
        #print(newScaredTimes)

        "*** YOUR CODE HERE ***"
        #Nos interesa lograr la puntuación más elevada de cada movimiento para elegir siempre la que más aporte
        #por lo que cuanto más comamos, más alejados de los fantasmas estemos mejor

        #Variable que vamos a devolver
        resultado = 0

        #Comprobar que hemos comido comida
        listaComidaSucesor = newFood.asList() #lista de comidas para un estado sucesor 
        listaComidaActual = currentGameState.getFood().asList() #lista de comida del estado actual

        if(len(listaComidaActual) > len(listaComidaSucesor)): #significa que hemos comido 1 comida en la transicion del estado actual al sucesor, añadimos 100 puntos
            resultado = resultado + 10

        #No comer también restará
        if(len(listaComidaActual) == len(listaComidaSucesor)): #significa que hemos comido 1 comida en la transicion del estado actual al sucesor, añadimos 100 puntos
            resultado = resultado - 5

        #Haremos lo mismo para los puntos grandes
        listaPuntosSucesor = successorGameState.getCapsules()
        listaPuntosActual = currentGameState.getCapsules()

        if(len(listaPuntosActual) > len(listaPuntosSucesor)): #significa que hemos comido punto
            resultado = resultado + 10

         #Tambien nos interesa estar alejados de los fantasmas, por lo que una distancia cercana a un fantasma nos bajará puntos
        #conseguimos las distancias a los fantasmas
        distanciaFantasmas = []
        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0: #solo nos queremos alejar si el fantasma no está asustado
                distancia = manhattanDistance(ghostState.getPosition(), newPos)
                distanciaFantasmas.append(distancia)

        for distancia in distanciaFantasmas: #penalizaremos mucho el estar a menos de 4 (distancia manhattan) de un fantasma
            if distancia < 4:
                resultado = resultado - 10

        #Muchas veces el pacman se queda quieto así que restaremos puntos por ello
        if action == 'Stop':
            resultado = resultado - 5

        #Como nos interesa ganar daremos muchos puntos por ello
        if successorGameState.isWin():
           resultado = resultado + sys.maxsize

        #Haremos lo contrario por perder
        if successorGameState.isLose():
            resultado = resultado - sys.maxsize

        #Probando hasta aquí me he dado cuenta que no se premia el acercarse a la comida y eso hace que el pacman no se acerce a por ella
        comidaMasCercana = sys.maxsize
        for comida in newFood.asList():
            comidaMasCercana = min(comidaMasCercana, manhattanDistance(comida, newPos))
        resultado = resultado + 10 - comidaMasCercana*2
        print(successorGameState)
        return successorGameState.getScore() + resultado
        


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def maximizar(self, gameState, depth):
        profundidadActual = depth + 1
        index = 0 #sabemos que siempre que maximicemos será con el pacman, que tiene índice 0
        if(gameState.isLose() or gameState.isWin() or profundidadActual == self.depth): #si se cumple alguna de las condiciones se devuelve el valor
            return self.evaluationFunction(gameState)
        v = -sys.maxsize #como dice la teoria, inicializamos v a menos infinito
        accionesPosibles = gameState.getLegalActions(index) #vemos que movimientos posibles tiene el pacman
        for accion in accionesPosibles:
            sucesor = gameState.generateSuccessor(index, accion) #generamos los estados sucesores con las acciones
            v = max(v, self.minimizar(sucesor, profundidadActual, index+1))
        return v


    def minimizar(self, gameState, depth, index):
        if(gameState.isLose() or gameState.isWin()): #no comprobamos la profundidad porque ya se ha dado la maximización
            return self.evaluationFunction(gameState)
        v = sys.maxsize
        accionesPosibles = gameState.getLegalActions(index)
        for accion in accionesPosibles:
            sucesor = gameState.generateSuccessor(index, accion)
            numeroAgentes = gameState.getNumAgents()-1
            if(index == numeroAgentes): #el fantasma que está jugando es el último, y el siguiente es el pacman
                v = min(v, self.maximizar(sucesor, depth))
            else: #todavía quedan más fantasmas por jugar
                v = min(v, self.minimizar(sucesor, depth, index+1))
        return v


    def accion_raiz(self, gameState):
        index = 0 #indice del pacman, que es el que toma la accion
        accionesLegales = gameState.getLegalActions(index)
        accionQueEjecuta = None
        v = -sys.maxsize
        for accion in accionesLegales:
            sucesor = gameState.generateSuccessor(index,accion)
            puntuacion = self.minimizar(sucesor,0,1) #la siguente acción al root es un fantasma que minimiza
            if(v < puntuacion): #buscamos cual de los sucesores nos devuelve un mayor valor
                v = puntuacion
                accionQueEjecuta = accion
        return accionQueEjecuta



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
        
        return self.accion_raiz(gameState)
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maximizar(self, gameState, depth, alfa, beta):
        profundidadActual = depth + 1
        index = 0 #sabemos que siempre que maximicemos será con el pacman, que tiene índice 0
        if(gameState.isLose() or gameState.isWin() or profundidadActual == self.depth): #si se cumple alguna de las condiciones se devuelve el valor
            return self.evaluationFunction(gameState)
        v = -sys.maxsize #como dice la teoria, inicializamos v a menos infinito
        accionesPosibles = gameState.getLegalActions(index) #vemos que movimientos posibles tiene el pacman
        for accion in accionesPosibles:
            sucesor = gameState.generateSuccessor(index, accion) #generamos los estados sucesores con las acciones
            v = max(v, self.minimizar(sucesor, profundidadActual, index+1, alfa, beta))
            if(v > beta): #no podemos poner 'v >= beta' porque el enunciado dice que no hay que podar en caso de igualdad
                return v
            alfa = max(alfa, v)
        return v


    def minimizar(self, gameState, depth, index, alfa, beta):
        if(gameState.isLose() or gameState.isWin()): #no comprobamos la profundidad porque ya se ha dado la maximización
            return self.evaluationFunction(gameState)
        v = sys.maxsize
        accionesPosibles = gameState.getLegalActions(index)
        for accion in accionesPosibles:
            sucesor = gameState.generateSuccessor(index, accion)
            numeroAgentes = gameState.getNumAgents()-1
            if(index == numeroAgentes): #el fantasma que está jugando es el último, y el siguiente es el pacman
                v = min(v, self.maximizar(sucesor, depth, alfa, beta))
                if(v < alfa): #no podemos poner 'v <= alfa' porque el enunciado dice que no hay que podar en caso de igualdad
                    return v
                beta = min(beta, v)
            else: #todavía quedan más fantasmas por jugar
                v = min(v, self.minimizar(sucesor, depth, index+1, alfa, beta))
                if(v < alfa): #no podemos poner 'v <= alfa' porque el enunciado dice que no hay que podar en caso de igualdad
                    return v
                beta = min(beta, v)
        return v


    def accion_raiz(self, gameState):
        index = 0 #indice del pacman, que es el que toma la accion
        accionesLegales = gameState.getLegalActions(index)
        accionQueEjecuta = None
        v = -sys.maxsize
        alfa = -sys.maxsize #como nos dice la teoria inicializamos los valore alfa
        beta = sys.maxsize # y beta que irán propagandose a sus nodos hijos
        for accion in accionesLegales:
            sucesor = gameState.generateSuccessor(index,accion)
            puntuacion = self.minimizar(sucesor, 0, 1, alfa, beta) #la siguente acción al root es un fantasma que minimiza
            if(v < puntuacion): #buscamos cual de los sucesores nos devuelve un mayor valor
                v = puntuacion
                accionQueEjecuta = accion
            if(v > beta): #no podemos poner 'v >= beta' porque el enunciado dice que no hay que podar en caso de igualdad
                return accionQueEjecuta
            alfa = max(alfa, v)
        return accionQueEjecuta 

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.accion_raiz(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maximizar(self, gameState, depth):
        profundidadActual = depth + 1
        index = 0 #sabemos que siempre que maximicemos será con el pacman, que tiene índice 0
        if(gameState.isLose() or gameState.isWin() or profundidadActual == self.depth): #si se cumple alguna de las condiciones se devuelve el valor
            return self.evaluationFunction(gameState)
        v = -sys.maxsize #como dice la teoria, inicializamos v a menos infinito
        accionesPosibles = gameState.getLegalActions(index) #vemos que movimientos posibles tiene el pacman
        for accion in accionesPosibles:
            sucesor = gameState.generateSuccessor(index, accion) #generamos los estados sucesores con las acciones
            v = max(v, self.value(sucesor, profundidadActual, index+1))
        return v


    def value(self, gameState, depth, index):
        if(gameState.isLose() or gameState.isWin()): #no comprobamos la profundidad porque ya se ha dado la maximización
            return self.evaluationFunction(gameState)
        v = 0 #para expectimax se inicializa v=0 en la parte probabilística
        accionesPosibles = gameState.getLegalActions(index)
        numeroSucesores = len(accionesPosibles)
        probabilidad = 1/numeroSucesores
        for accion in accionesPosibles:
            sucesor = gameState.generateSuccessor(index, accion)
            numeroAgentes = gameState.getNumAgents()-1
            if(index == numeroAgentes): #el fantasma que está jugando es el último, y el siguiente es el pacman
                v += self.maximizar(sucesor, depth) #si soy un fantasma voy a coger el mayor valor que dejen por debajo, porque no soy experto
            else: #todavía quedan más fantasmas por jugar
                v += self.value(sucesor, depth, index+1)
        return(v * probabilidad)


    def accion_raiz(self, gameState):
        index = 0 #indice del pacman, que es el que toma la accion
        accionesLegales = gameState.getLegalActions(index)
        accionQueEjecuta = None
        v = -sys.maxsize
        for accion in accionesLegales:
            sucesor = gameState.generateSuccessor(index,accion)
            puntuacion = self.value(sucesor, 0, 1) #la siguente acción al root es un fantasma que minimiza
            if(v < puntuacion): #buscamos cual de los sucesores nos devuelve un mayor valor
                v = puntuacion
                accionQueEjecuta = accion
        return accionQueEjecuta 

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.accion_raiz(gameState)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
