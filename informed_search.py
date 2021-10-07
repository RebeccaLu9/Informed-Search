############################################################
# CIS 521: Informed Search Homework
############################################################

student_name = "Jingyi Lu"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import random
import queue
import math


############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    pass

class TilePuzzle(object):
    
    # Required
    def __init__(self, board):
        self.board = board
        self.row = len(board)
        self.col = len(board[0])
        self.r = self.row
        self.c = self.col
        self.moves = ['up', 'down', 'left', 'right']
        self.cost = 0
        self.sol = self.goal()

    def goal(self):
        goal =  []
        for i in range(self.row):
            goal.append([x+1 for x in range(self.col*i,self.col*(i+1))])
        goal[-1][-1] = 0
        return goal

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        #first find the location of 0
        loc = [(i, elem.index(0)) for i, elem in enumerate(self.board) if 0 in elem]
        row0 = loc[0][0]
        col0 = loc[0][1]
        if(direction == 'up' and row0 == 0):
            return False
        elif(direction == 'up' and row0 != 0):
            self.board[row0][col0] = self.board[row0-1][col0]
            self.board[row0-1][col0] = 0
            
        elif(direction == 'down' and row0 == self.row-1):
            return False
        elif(direction == 'down' and row0 != self.row-1):
            self.board[row0][col0] = self.board[row0+1][col0]
            self.board[row0+1][col0] = 0
        
        elif(direction == 'left' and col0 == 0):
            return False
        elif(direction == 'left' and col0 != 0):
            self.board[row0][col0] = self.board[row0][col0-1] 
            self.board[row0][col0-1] =0
            
        elif(direction == 'right' and col0 == self.col-1):
            return False
        elif(direction == 'right' and col0 != self.col-1):
            self.board[row0][col0] = self.board[row0][col0+1] 
            self.board[row0][col0+1]  = 0

        return True

    def scramble(self, num_moves):
        seq = ['up','down','left','right']
        for elem in range(num_moves):
            random.choice(seq)

    def is_solved(self):
        result = []
        for i in range(self.row):
            result.append([x+1 for x in range(self.col*i,self.col*(i+1))])
        result[-1][-1] = 0
        if(self.board == result): 
            return True
        else: return False

    def copy(self):
        copy= []
        for i in range(self.row):
            copy.append([x+1 for x in range(self.col*i,self.col*(i+1))])
        for i in range(self.row):
            for j in range(self.col):
                if(copy[i][j] != self.board[i][j]): copy[i][j] = self.board[i][j]
        new = TilePuzzle(copy)
        return new

    def successors(self):
        #up, down, left, right
        newPuzz = self.copy()
        if newPuzz.perform_move('up'):
            yield('up', newPuzz)
        newPuzz = self.copy()
        if newPuzz.perform_move('down'):
            yield('down', newPuzz)
        newPuzz = self.copy()
        if newPuzz.perform_move('left'):
            yield('left', newPuzz)
        newPuzz = self.copy()
        if newPuzz.perform_move('right'):
            yield('right', newPuzz)
        return

    # Required
    def find_solutions_iddfs(self):
        found = False
        limit = 0
        while not found:
            for move in self.iddfs_helper(limit, []):
                yield move
                found = True
            limit += 1

    def iddfs_helper(self, limit, moves):
        if self.is_solved():
            yield moves
        elif len(moves) < limit:
            for move, puzzle in self.successors():
                for elem in puzzle.iddfs_helper(limit, moves + [move]):
                    yield elem

    def manhattan(self, goal):
        result = 0
        pos = {}

        for r in range(self.r):
            for c in range(self.c):
                pos[goal[r][c]] = (r, c)

        for r in range(self.r):
            for c in range(self.c):
                a = self.board[r][c]
                pos2 = pos[a]
                result += abs(r - pos2[0]) + abs(c - pos2[1])
        return result

    # Required
    def find_solution_a_star(self):
        frontier = set()
        frontier.add(self)
        come_from = {}
        come_from[self] = None
        cost_so_far = {}
        cost_so_far[self] = 0
        self.h = self.manhattan(self.sol)
        self.route = []

        while frontier:
            curr = min(frontier, key = lambda x: x.cost)
            
            if curr.is_solved():
                return curr.route
            frontier.remove(curr)

            for move, puzzle in curr.successors():
                if puzzle.is_solved():
                    puzzle.route = curr.route + [move]
                    return puzzle.route

                new_cost = cost_so_far[curr] + curr.manhattan(puzzle.board)
                puzzle.cost = new_cost + puzzle.manhattan(self.sol)

                check = True
                for board in come_from:
                    if board.board == puzzle.board and board.cost < puzzle.cost:
                        check = False
                        continue
                        
                if check:
                    frontier.add(puzzle)
                    puzzle.route = curr.route + [move]
                    come_from[puzzle] = curr
                    cost_so_far[puzzle] = new_cost



def create_tile_puzzle(rows, cols):
    result = []
    for i in range(rows):
        result.append([x+1 for x in range(cols*i,cols*(i+1))])
    result[-1][-1] = 0
    return TilePuzzle(result)

############################################################
# Section 2: Grid Navigation
############################################################
class Grid:
    def __init__(self, start, goal, scene):
        self.start = start
        self.goal = goal
        self.scene = scene
        self.r = len(scene)
        self.c = len(scene[0])
        
    def no_obstacle(self, point):
        r = point[0]
        c = point[1]
        if(self.scene[r][c]): return False
        return True
    
    def is_valid(self, point):
        r = point[0]
        c = point[1]
        return ((0<=r<self.r) and (0<=c<self.c))
    
    def get_successor(self, current):
        x = current[0]
        y = current[1]
        results = [(x+1,y), (x+1,y-1), (x,y-1), (x-1,y-1),(x-1,y), (x-1,y+1), (x,y+1), (x+1,y+1)]

        if (x+y)%2 == 0:
            results.reverse()

        results = filter(self.is_valid, results)
        results = filter(self.no_obstacle, results)
        return results
    
    def back_moves(self,come_from):
        current = self.goal[:]
        start = self.start[:]
        moves = []
        
        while(current != start):
            moves.append(current)
            current = come_from[current]
            
        moves.append(self.start)
        moves.reverse()
        result = []
        result = moves[:]
        return result
    

    def solve(self):
        frontier = queue.PriorityQueue()
        frontier.put((0, self.start))
        come_from = {}
        cost_so_far = {}
        come_from[self.start] = None
        cost_so_far[self.start] = 0

        if self.start == self.goal:return self.start

        if not self.no_obstacle(self.start) or not self.no_obstacle(self.goal):
            return None

        if not self.is_valid(self.start):return None
        if not self.is_valid(self.goal):return None

        while not frontier.empty():
            _, current = frontier.get()

            if current == self.goal:
                return self.back_moves(come_from)

            for next_pos in self.get_successor(current):
                new_cost = cost_so_far[current]+h_euclidean(current, next_pos)

                if (next_pos not in cost_so_far) or (new_cost < cost_so_far[next_pos]):if (next_pos not in cost_so_far) or (new_cost < cost_so_far[next_pos]):
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + h_euclidean(self.goal, next_pos)
                    frontier.put((priority, next_pos))
                    come_from[next_pos] = current
        return None
 
def h_euclidean(current, next_step):
    return (math.sqrt( (next_step[0]-current[0])**2 + (next_step[1]-current[1])**2  ) )

def find_path(start, goal, scene):
    grid = Grid(start, goal, scene)
    move = grid.solve()
    return move
############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

def getsuccessors(grid):
    for i in range(len(grid)):
        new_grid = grid[:]
        if grid[i] == 0:
            continue
            
        if ((i+1) < len(grid) and grid[i+1] == 0):#right
            new_grid[i+1] = new_grid[i]
            new_grid[i] = 0
            yield ((i, i+1), new_grid)
            
        elif ((i+2) < len(grid) and grid[i+2] == 0 and grid[i+1] != 0): #right 2
            new_grid[i+2] = new_grid[i]
            new_grid[i] = 0
            yield ((i, i+2), new_grid)
            
        elif ((i-1) >= 0 and grid[i-1] == 0): #left
            new_grid[i-1] = new_grid[i]
            new_grid[i] = 0
            yield ((i, i-1), new_grid)
            
        elif ((i-2) >= 0 and grid[i-2] == 0 and grid[i-1] != 0): #left 2
            new_grid[i-2] = new_grid[i]
            new_grid[i] = 0
            yield ((i, i-2), new_grid)
            
            
#def backtrack(grid, visited, goal, start):
def backtrack( visited, goal, start):
        current = [goal, None] #goal is where we at, current[1] is the move we take from previous to current
        path = []
        while current[0] != start:
            current = visited[tuple(current[0])]
            path.append(current[1])#because we track moves from end to start

        path.reverse()
        return path


def solve_distinct_disks(length, n):
    grid = [0 for x in range(length)]
    grid[:n] = [x for x in range(1, n+1)]

    frontier = queue.PriorityQueue()
    frontier.put((grid,0))
    visited = {}
    start1 = grid[:]
    final = grid[:]
    final.reverse()

    if grid == final or length == 0 or n == 0: #solved
        return []
    
    cost_so_far = {}
    come_from= {}
    cost_so_far[tuple(start1)] = 0
    come_from[tuple(start1)] = None

    def cost(current, new, n = n):
        length = len(new)
        result = 0
        for i in range(n):
            current_ind = current.index(i+1)
            goal_ind = length - (i+1)
            result += abs(current_ind - goal_ind)
        return result
   
    while not frontier.empty():
        current = frontier.get()[0]

        for move, new_grid in getsuccessors(current):
            t_new_grid = tuple(new_grid)
            #print(cost_so_far[tuple(current)])
            new_cost = cost_so_far[tuple(current)] + cost(current, new_grid)
            
            if (t_new_grid not in visited) or (new_cost < cost_so_far[tuple(new_grid)]):
            #if (t_new_grid not in visited):
                if new_grid  == final: #solved
                    visited[t_new_grid] = (current, move)
                    #print(visited)
                    #moves = backtrack(grid, visited, t_new_grid, start = start1)
                    moves = backtrack(visited, t_new_grid, start1)
                    return moves
        
                cost_so_far[tuple(new_grid)] = new_cost
                come_from[tuple(current)] = new_grid
                priority = new_cost + cost(new_grid, final)
                frontier.put((new_grid, priority))
                #frontier.put(new_grid)
                visited[t_new_grid] = [current, move]

    return None




############################################################
# test
############################################################
if __name__ == "__main__":
    # b = [[4,1,2], [0,5,3], [7,8,6]]
    # p = TilePuzzle(b)
    # p.find_solution_a_star()
    solve_distinct_disks(5, 2)
# b = [[1,2,3], [4,0,5], [6,7,8]]
# p = TilePuzzle(b)
# p.find_solution_a_star()


#Q3
# solve_distinct_disks(4, 2)



############################################################
# Section 4: Feedback
############################################################

# Just an approximation is fine.
feedback_question_1 = 12

feedback_question_2 = """
I hope the TAs can talk more about how to solve the specific problems
"""

feedback_question_3 = """
It may be helpful for interviews
"""
