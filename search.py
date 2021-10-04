# search.py
from game import Directions
from util import Queue
from util import Stack
from util import PriorityQueue
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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):

    no_observed = util.Stack()
    observed = set()
    inicio = problem.getStartState()
    no_observed.push((inicio, []))

    while not no_observed.isEmpty():
        actual = no_observed.pop()
        coord_actual = actual[0]
        path_actual = actual[1]

        if problem.isGoalState(coord_actual):
            return path_actual

        for coord_hijo, action_hijo, cost_hijo in problem.getSuccessors(coord_actual):
            if coord_hijo not in observed:
                path = path_actual + [action_hijo]
                no_observed.push((coord_hijo, path))
        observed.add(coord_actual)

    return []


def breadthFirstSearch(problem):

    no_observed = util.Queue()
    observed = set()
    inicio = problem.getStartState()
    no_observed.push((inicio, []))

    while not no_observed.isEmpty():
        actual = no_observed.pop()
        coord_actual = actual[0]
        path_actual = actual[1]

        if problem.isGoalState(coord_actual):
            return path_actual

        for coord_hijo, action_hijo, cost_hijo in problem.getSuccessors(coord_actual):
            if coord_hijo not in observed:
                path = path_actual + [action_hijo]
                no_observed.push((coord_hijo, path))
                observed.add(coord_hijo)
        observed.add(coord_actual)

    return []


def uniformCostSearch(problem):
    no_observed = PriorityQueue()
    observed = set()
    no_observed.push((problem.getStartState(), [], 0), 0)

    while not no_observed.isEmpty():
        actual = no_observed.pop()
        cord_actual = actual[0]
        camino_actual = actual[1]
        coste_actual = actual[2]
        if problem.isGoalState(cord_actual):
            return camino_actual

        if cord_actual not in observed:
            observed.add(cord_actual)
            for child in problem.getSuccessors(cord_actual):
                cord_hijo = child[0]
                movimiento_hijo = child[1]
                coste_hijo = child[2]

                if cord_hijo not in observed:
                    no_observed.push((cord_hijo, camino_actual+[movimiento_hijo], coste_actual+coste_hijo),
                                     coste_actual+coste_hijo)

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    no_observed = PriorityQueue()
    observed = set()

    # ((x,y),[camino],coste_sin_heuristico],coste_con_heuristico)
    no_observed.push((problem.getStartState(), [], 0), 0)

    while not no_observed.isEmpty():
        actual = no_observed.pop()
        cord_actual = actual[0]
        camino_actual = actual[1]
        coste_actual = actual[2]

        if problem.isGoalState(cord_actual):
            return camino_actual

        if cord_actual not in observed:
            observed.add(cord_actual)
            for child in problem.getSuccessors(cord_actual):
                cord_hijo = child[0]
                movimiento_hijo = child[1]
                coste_hijo = child[2]
                if cord_hijo not in observed:
                    coste_total_sin_heuristico = coste_actual + coste_hijo
                    no_observed.push((cord_hijo, camino_actual+[movimiento_hijo], coste_total_sin_heuristico),
                                     coste_total_sin_heuristico+heuristic(cord_hijo, problem))

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
