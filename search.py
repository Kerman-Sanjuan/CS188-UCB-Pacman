# search.py
from game import Directions
from util import Queue
from util import Stack
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
        Devuelve la posicion inicial del pacman.
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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    
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
    Problem es de tipo -> SearchProblem
    """
    
    "*** YOUR CODE HERE ***"
    
    #Álvaro version 1
    s = Directions.SOUTH
    n = Directions.NORTH
    e = Directions.NORTH
    w = Directions.WEST


    inicio = problem.getStartState()
    no_observed = util.Stack()
    caminoPila = util.Stack()
    observed = []
    camino = []
    no_observed.push(inicio)
    pos = inicio
    
    while not problem.isGoalState(pos):
        observed.append(pos)
        listaVecinos = problem.getSuccessors(pos)
        for coord, accion, coste in listaVecinos:
            if coord not in observed:
                no_observed.push(coord)
                paso = camino + [accion]
                caminoPila.push(paso)
        camino = caminoPila.pop()
        pos = no_observed.pop()
    return camino
    '''
    #Álvaro version 2

    no_observed = util.Stack()
    observed = set()
    inicio = problem.getStartState()
    no_observed.push((inicio,[]))

    while not no_observed.isEmpty():
        actual = no_observed.pop()
        coord_actual = actual[0]
        path_actual = actual[1]
        if problem.isGoalState(coord_actual):
            return path_actual
        hijos = problem.getSuccessors(coord_actual)
        for coord_hijo, action_hijo, cost_hijo in hijos:
            if coord_hijo not in observed:
                path = path_actual + [action_hijo]
                no_observed.push((coord_hijo,path))
        observed.add(coord_actual)
    return []
    '''
            
        




def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    s = Directions.SOUTH
    n = Directions.NORTH
    e = Directions.NORTH
    w = Directions.WEST

    '''
    #Álvaro version punteros->

    pos = {'coord' : problem.getStartState()}
    no_observed = util.Queue()
    no_observed.push(pos)
    observed = set()
    while not no_observed.isEmpty():
        pos = no_observed.pop()
        observed.add(pos['coord'])
        successors = problem.getSuccessors(pos['coord'])
        for successor in successors:
            next_casilla = {'anterior' : pos, 'coord' : successor[0], 'action' : successor[1], 'cost' : successor[2]}
            if next_casilla['coord'] not in observed:
                no_observed.push(next_casilla)
                observed.add(next_casilla['coord'])
                if problem.isGoalState(next_casilla['coord']):
                        camino = []
                        pos = next_casilla
                        while 'anterior' in pos:
                            camino.append(pos['action'])
                            pos = pos['anterior']
                        camino.reverse()
                        return camino


    Kerman ->

    no_observed = util.Queue()
    observed = set()

    no_observed.push((problem.getStartState(), []))

    while not no_observed.isEmpty():
        actual = no_observed.pop()
        pos_actual = actual[0]
        camino_hasta_ahora = actual[1]

        if problem.isGoalState(pos_actual):
            return camino_hasta_ahora
        
        hijos = problem.getSuccessors(pos_actual)

        for coord, action, cost in hijos:
            coord_hijo = coord
            accion_hijo = action
            if coord_hijo not in observed:
                no_observed.push((coord_hijo, camino_hasta_ahora+[accion_hijo]))
                observed.add(coord_hijo)
        observed.add(pos_actual) 
    
    return []'''


    no_observed = util.Queue() #cola en la que guardaremos cada posicion y el camino que se haya realizado hasta llegar a esa posicion. Guardaremos en una tupla
    observed = set() #será un set en el que guardamos las coordenadas ya visitadas para no repetir su busqueda
    inicio = problem.getStartState() #conoceremos las coordenadas del estado inicial
    no_observed.push((inicio,[])) #guardaremos antes del loop la posicion inicial y una lista con las acciones hasta llegar a esa posicion (es decir, vacia)

    while not no_observed.isEmpty():
        actual = no_observed.pop() #cogemos el primer elemento de la cola
        coord_actual = actual[0] #en esta variable guardamos las coordenadas de la posicion actual
        path_actual = actual[1] #en esta varibale guardamos la lista de acciones que se han dado para llegar a la posicion actual
        if problem.isGoalState(coord_actual): #si la coordenada actual coincide con la coordenada objetivo devolvemos el camino hecho hasta el momento
            return path_actual
        hijos = problem.getSuccessors(coord_actual) #en esta variable guardamos una lista de truplas (coord, action, cost) de las casillas vecinas a la actual
        for coord_hijo, action_hijo, cost_hijo in hijos:
            if coord_hijo not in observed: #si el hijo nunca ha sido visitado habrá que añadirlo a la cola para visitarlo después, con sus coordenadas y el camino hasta él
                path = path_actual + [action_hijo] #el camino hasta esta posición sera el camino hasta la casilla anterior más el movimiento a esta última
                no_observed.push((coord_hijo,path))
                observed.add(coord_hijo)
        observed.add(coord_actual)
    return [] #solo se da en caso de error

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
