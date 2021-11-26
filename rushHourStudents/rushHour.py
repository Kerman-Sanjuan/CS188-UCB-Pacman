# rushhour.py
# --------------
# Licensing Information: idea taken from http://www.theiling.de/projects/rushhour.html
# 

import search
import random
import argparse

# Module Classes

class RushHourState:
    """
    This class defines the mechanics of the game itself.  The
    task of recasting this problem as a search problem is left to
    the RushHourSearchProblem class.
    """

    def __init__( self, board=[], cells=[], cars={} ):
        """
          Constructs a new board from an ordering of strings.

        board: a list of strings of length 6 representing an
          instance of the board.  '.' represents the blank
          space. Each letter represents a car or truck.
          Thus, the list

        # Example: board = ["aaobcc", "..ob..", "xxo...", "deeffp", "d..k.p", "hh.k.p"]
          represents the board:
            aaobcc
            ..ob..
            xxo...
            deeffp
            d..k.p
            hh.k.p

        The configuration of the board is stored in a 2-dimensional
        list (a list of lists) 'cells'.
        The elements xx represent the red car.
        Additionally, a hash table contains the coordinates of each car (attribute "cars")
        """
        self.cells = []
        self.blank = '.'
        self.cars = {} # For each car: Key (letter representing the car -> 
                   # Value: ('x', 2, 'h', (x,y)): (letter, length, direction(h|v), coordinates of the first position, starting from the upper left position)
        if board == [] :
            self.cells = cells
            self.cars = cars
            exit_row = round((len(self.cells)-1)/2)
            exit_column = len(self.cells[0]) - 1
            self.exit = (exit_row, exit_column) # Goal
            return
            
        board = board[:] # Make a copy so as not to cause side-effects.
        for row in range( len(board) ):
           self.cells.append( [] )
           for col in range( len(board[0]) ):
              #print(board[row][col])
              self.cells[row].append( board[row][col] )
        #print(self.cells)
        print("Dimensions: ", len(self.cells), len(self.cells[0]))

        exit_row = round((len(self.cells)-1)/2)
        exit_column = len(self.cells[0]) - 1
        self.exit = (exit_row, exit_column) # Goal
        
        # Mark the cars
        for row in range( len(board) ):
           for col in range( len(board[0]) ):
               self.add(row, col, board[row][col])
        #print(self.cars)
        
    def add(self, row, col , v):
        """
          adds the car to the dictionary in case it was not there previously
        """
        if v not in self.cars and not v == self.blank:
            startx, starty = self.find(v)
            #print(startx, starty)
            horizontal, length = self.findDirection(startx, starty, v)
            self.cars[v] = (v, horizontal, length, (startx, starty))

    def find(self, v): # examine from left to right, top to bottom, this way it will find the upper leftmost cell of each car
        for x in range(len(self.cells)):
            for y in range(len(self.cells[0])):
                if self.cells[x][y] == v:
                    return (x,y)
        return (-1,-1) # can not happen

    def findDirection(self, x, y, value): # x, y is the start position of the car. value is the name (letter) of the car
        # Returns the direction ('h'(orizontal) or 'v'(ertical)) and length of the element
        #print("VALUES: ", x, y, value, "LIMITS: ", len(self.cells), len(self.cells[0]))
        if self.valid(x, y+1) and self.cells[x][y+1] == value:
            if self.valid(x, y+2) and self.cells[x][y+2] == value:
                return ('h', 3)
            else:
                return ('h', 2)
        if self.valid(x+1, y) and self.cells[x+1][y] == value:
            if self.valid(x+2, y) and self.cells[x+2][y] == value:
                return ('v', 3)
            else:
                return ('v', 2)
        return ('X', 0)

    def valid(self, x, y):
        return (x >= 0) and (x < len(self.cells)) and (y >= 0) and (y < len(self.cells[0]))

    
    def isGoal( self ):
        """
          Checks to see if the puzzle is in its goal state:

            -------------------------
            |   |   |   |   |   |   |
            -------------------------
            |   |   |   |   |   |   |
            -------------------------
            |   |   |   |   | x | x 
            -------------------------
            |   |   |   |   |   |   |
            -------------------------
            |   |   |   |   |   |   |
            -------------------------
            |   |   |   |   |   |   |
            -------------------------

        >>> RushHourState(["aaobcc", "..ob..", "..o.xx", "deeffp", "d..k.p", "hh.k.p"]).isGoal()
        True

        >>> RushHourState(["aaobcc", "..ob..", "xxo...", "deeffp", "d..k.p", "hh.k.p"]).isGoal()
        False
        """
        # TODO

        #primero debemos buscar el número de líneas que tiene el tablero 
        lineas = len(self.cells)
        #ahora conseguimos la posición del coche rojo en el tablero
        posicion_coche = self.cars['x'][3]
        posicion_coche_x, posicion_coche_y = posicion_coche


        if lineas == 6: #para un tablero de 6x6 si la posicion del coche son x=2 y=4 estará en el state goal (X son las filas e Y son las columnas)
            if (posicion_coche_x == 2 and posicion_coche_y == 4):
                return True
            else:
                return False
        else:   
            if (posicion_coche_x == 4 and posicion_coche_y == 7):
                return True
            else: 
                return False

    def legalMoves( self ):
        """
        Returns a list of legal moves from the current state.

        Moves consist of moving each car to the right, left, up or down
        These are encoded as 'up', 'down', 'left', 'right'

        >>> RushHourState(["aaobcc", "..ob..", "..o.xx", "deeffp", "d..k.p", "hh.k.p"]).legalMoves()
          representing the board:
            aaobcc
            ..ob..
            xxo...
            deeffp
            d..k.p
            hh.k.p
        [('b', 'down'), ('p', 'up'), ('h', 'right')]
        """
        moves = []
        for key in self.cars: #recorremos el diccionario de coches es el tablero
            coche = self.cars[key] #obtenemos todos los valores del coche con clave=key en el diccionario. 
            #Devolverá algo así -> ('x', 2, 'h', (x,y)): (letter, length, direction(h|v), coordinates of the first position, starting from the upper left position)
            carCoordinates = coche[3] #obtenemos las coordenadas del coche. Es el 3er valor
            x,y = carCoordinates #asignamos a x la coordenada de la fila y a y la de la columna
            orientacion = coche[1] #nos interesa saber si el coche está colocado en vertical o en horizontal
            nombre = coche[0]
            longitudCoche = coche[2]
            if orientacion == 'v': #el coche está colocado en vertical y solo puede o subir o bajar. Trataremos casos críticos
                if(x == 0): #la posición más elevada del coche está en la primera fila y por lo tanto no puede subir
                    if(self.cells[x+longitudCoche][y] == self.blank): #en este caso la posición debajo del coche está libre
                        moves.append((nombre,'down'))
                elif((x+longitudCoche) == len(self.cells)): #la posición de la parte inferior del coche está en la última fila y por ello no puede bajar
                    if(self.cells[x-1][y] == self.blank): #la casilla de encima del coche está vacía
                        moves.append((nombre,'up'))
                else: #puede tanto subir como bajar
                    if(self.cells[x+longitudCoche][y] == self.blank): #en este caso la posición de debajo del coche está libre
                        moves.append((nombre,'down'))
                    if(self.cells[x-1][y] == self.blank): #la casilla de encima del coche está vacía
                        moves.append((nombre,'up'))
            else: #en este caso el coche está colocado en horizontal y el coche puede moverse de izquierda a derecha. 
                if(y == 0): #esta en la primera columna (izquierda) y no puede ir a la izquiera
                    if(self.cells[x][y+longitudCoche] == self.blank): #existe hueco a la derecha
                        moves.append((nombre,'right'))
                elif((y+longitudCoche) == len(self.cells)): #el final del coche está en la columna de más a la derecha
                    if(self.cells[x][y-1] == self.blank): #el hueco a la izquiera del coche está vacío
                        moves.append((nombre,'left'))
                else: #puede tanto ir a la izquierda como a la derecha
                        if(self.cells[x][y+longitudCoche] == self.blank): #existe hueco a la derecha
                            moves.append((nombre,'right'))
                        if(self.cells[x][y-1] == self.blank): #el hueco a la izquiera del coche está vacío
                            moves.append((nombre,'left'))
        return moves

    def result(self, movePair):
        """
          Returns a new rushHour with the current state
        updated based on the provided move.

        The move should be drawn from a list returned by legalMoves.
        Illegal moves will raise an exception, which may be an array bounds
        exception.

        NOTE: This function *does not* change the current object.  Instead,
        it returns a new object.
        """
        newCells = self.copyCells()
        newCars = self.cars.copy()
        (car, move) = movePair
        (c, dir, len, (cx, cy)) = self.cars[car]
        if(move == 'up'):
            newCells[cx+len-1][cy] = self.blank #el hueco final que ocupaba el coche que se ha movido hacia arriba queda vacío
            newCells[cx-1][cy] = c #la posición de encima del coche ahora pertenecerá al coche de nombre 'c'
            newCars[c] = (c, dir, len, (cx-1, cy)) #en el diccionario actualizamos la posición del coche una posición más arriba
        elif(move == 'down'):
            newCells[cx][cy] = self.blank #el hueco primero que ocupa el coche queda ahora vacío
            newCells[cx+len][cy] = c
            newCars[c] = (c, dir, len, (cx+1, cy))
        elif(move == 'left'):
            newCells[cx][cy+len-1] = self.blank
            newCells[cx][cy-1] = c
            newCars[c] = (c, dir, len, (cx, cy-1))
        elif(move == 'right'):
            newCells[cx][cy] = self.blank
            newCells[cx][cy+len] = c
            newCars[c] = (c, dir, len, (cx, cy+1))
        else:
            raise "Illegal Move"

        newState = RushHourState([], newCells, newCars)
        return newState

    def copyCells(self):
        # Returns a new copy of the cell matrix
        result = []
        for i in range(len(self.cells)):
            result.append(self.cells[i].copy())
        return result

    # Utilities for comparison and display
    def __eq__(self, other):
        """
            Overloads '==' such that two states with the same configuration
          are equal.
        """
        for row in range( len(self.cells) ):
            for col in range( len(self.cells[0]) ):
                if self.cells[row][col] != other.cells[row][col]:
                    return False
        return True

    def __hash__(self):
        return hash(str(self.cells))

    def __getAsciiString(self):
        """
          Returns a display string for the maze
        """
        lines = []
        horizontalLine = ('-' * (1 + 4 * len(self.cells)))
        lines.append(horizontalLine)
        for row in self.cells:
            rowLine = '|'
            for col in row:
                if col == 0:
                    col = ' '
                rowLine = rowLine + ' ' + col.__str__() + ' |'
            lines.append(rowLine)
            lines.append(horizontalLine)
        return '\n'.join(lines)

    def __str__(self):
        return self.__getAsciiString()



class RushHourSearchProblem(search.SearchProblem):
    """
      Implementation of a SearchProblem for the Rush Hour domain

      Each state is represented by an instance of an RushHourState.
    """
    
    def __init__(self, board): # board is of type RushHourState
        "Creates a new RushHourSearchProblem which stores search information."
        self.board = board
        self.expansions = 0

    def getStartState(self):
        return self.board

    def isGoalState(self, state):
        return state.isGoal()

    def getSuccessors(self, state):
        """
          Returns list of (successor, action, stepCost) pairs where
          each succesor is either left, right, up, or down
          from the original state and the cost is 1.0 for each
        """
        self.expansions = self.expansions + 1 # to account the number of expanded nodes
        succ = []
        for a in state.legalMoves():
            succ.append((state.result(a), a, 1))
        return succ

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to takeimport argparse

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)

RUSH_HOUR_DATA = [["....a.", "....a.", "..xxa.", "...r..", "...rmm", "......"],
                  ["aaobc.", "..obc.", "xxo...", "deeffp", "d..k.p", "hh.k.p"],
                  ["aaabcd", "effbcd", "e.xxcd", "ggh...", ".ih.jj", ".ikkll"],
                  # Hardest initial Rush Hour 6x6 configuration
                  # On the Symbolic Computation of the Hardest Configurations of the RUSH HOUR Game 
                  # Sebastien Collette, Jean-Francois Raskin and Frederic Serva
                  # https://di.ulb.ac.be/algo/secollet/papers/crs06.pdf
                  # The hardest configuration of the RUSH HOUR game is given
                  # in Figure 1c and it requires 93 steps to reach a winning configuration. From that ini-
                  # tial configuration 24132 configurations can be reached
                  ["aabbc..d.", "efffcggd.", "e.hhij.kk", "..lmijnnn", "xxlm.....", ".o.mppqqr", "sottuvvvr", "s.wwuyy.r", "szzzu...."], # bfs: > 300,000 expanded states, Astar: 140,000 states
                  ["abbb.c...", "a..d.ceee", "...d.c..f", "gg.d.hhhf", "ixxj....f", "i.kj.lllm", "nnkopp..m", "q.kor...m", "qsssrttuu"],
                  # 9x9 configuration
                  ["..abbc", "..a..c", "..axxc", "...dee", "fggd..", "f..dhh"] # 6x6 configuration: 8693 expanded states with bfs, 8392 with A* and heuristic
                  ]


def loadRushHour(boardNumber):
    """
      boardNumber: The number of the board to load.

      Returns a rush hour object generated from one of the
      provided puzzles in RUSH_HOUR_DATA.

      boardNumber can range from 0 to n.
    """
    return RushHourState(RUSH_HOUR_DATA[boardNumber])

def rushHourHeuristic(state, problem=None):
    lineas = len(state.cells)
    
    x,y = state.cars['x'][3]
    if lineas == 6:
        posGoal = (2,4)
    else:
        posGoal = (4,7)
    
    heuristico = posGoal[1]-y
    print(heuristico)
    return heuristico
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='A program to calculate a solution to the Rush Hour problem.')
    parser.add_argument("--board_number", help="The board from the initial boards available", type=int, default=0)
    parser.add_argument("--search_algorithm", help="The algorithm: bfs or aStar", default="bfs")
    args = parser.parse_args()
    board = loadRushHour(args.board_number)
    print('The board:')
    print(board)
    problem = RushHourSearchProblem(board)
    if args.search_algorithm == "bfs":
        path = search.breadthFirstSearch(problem)
    else:
        if args.search_algorithm == "dfs":
            path = search.depthFirstSearch(problem)   
        else: 
            #path = search.aStarSearch(problem)
            path = search.aStarSearch(problem, rushHourHeuristic)
    print("Number of expanded nodes: ", problem.expansions)
    print('%s found a path of %d moves: %s' % (args.search_algorithm, len(path), str(path)))
    curr = board
    i = 1
    for a in path:
        curr = curr.result(a)
        print('After %d move%s: %s' % (i, ("", "s")[i>1], a))
        print(curr)

        input("Press return for the next state...")   # wait for key stroke
        i += 1
