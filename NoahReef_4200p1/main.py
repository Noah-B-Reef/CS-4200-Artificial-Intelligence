import numpy as np
import pandas as pd
import time

def check_is_valid(mat):

    '''
    Summary: Checks if a given 8-puzzle is valid
    Input: mat - matrix representing the 8-puzzle
    Output: True if matrix has even number of inversions, False otherwise
    '''

    num_inv = 0

    size_mat = np.shape(mat)
    num_rows = size_mat[0]
    num_cols = size_mat[1]

    # convert matrix to 1D array
    mat_1d = mat.reshape(1,num_rows*num_cols)
    
    for i in range(num_rows*num_cols):
        for j in range(i+1,num_rows*num_cols):
            if mat_1d[0,i] > mat_1d[0,j] and mat_1d[0,i] != 0 and mat_1d[0,j] != 0:
                num_inv += 1


    if num_inv % 2 == 0:
        return True
    else:
        return False

def manhattan_distances(mat):

    '''
    Summary: Calculates the manhattan distance of a given 8-puzzle
    Input: mat - matrix representing the 8-puzzle
    Output: sum of manhattan distances
    '''

    true_pos = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}
    size_mat = np.shape(mat)
    num_rows = size_mat[0]
    num_cols = size_mat[1]

    sum = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if mat[i,j] != 0:
                true_row, true_col = true_pos[mat[i,j]]
                sum += abs(i - true_row) + abs(j - true_col)
    return sum

def displaced_distances(mat):
    
        '''
        Summary: Calculates the displaced tiles of a given 8-puzzle
        Input: mat - matrix representing the 8-puzzle
        Output: sum of displaced tiles
        '''
    
        true_pos = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}
        size_mat = np.shape(mat)
        num_rows = size_mat[0]
        num_cols = size_mat[1]
    
        sum = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if mat[i,j] != 0:
                    true_row, true_col = true_pos[mat[i,j]]
                    if i != true_row or j != true_col:
                        sum += 1
        return sum

def get_neighbors(mat):

    '''
    Summary: Gets the neighbors of a given 8-puzzle
    Input: mat - matrix representing the 8-puzzle
    Output: list of neighbors
    '''

    size_mat = np.shape(mat)
    num_rows = size_mat[0]
    num_cols = size_mat[1]

    # find the position of the 0
    for i in range(num_rows):
        for j in range(num_cols):
            if mat[i,j] == 0:
                row = i
                col = j

    # find the neighbors of the 0
    neighbors = []
    if row > 0:
        B = mat.copy()
        B[row,col] = B[row-1,col]
        B[row-1,col] = 0
        neighbors.append(B)

    if row < num_rows - 1:
        B = mat.copy()
        B[row,col] = B[row+1,col]
        B[row+1,col] = 0
        neighbors.append(B)

    if col > 0:
        B = mat.copy()
        B[row,col] = B[row,col-1]
        B[row,col-1] = 0
        neighbors.append(B)

    if col < num_cols - 1:
        B = mat.copy()
        B[row,col] = B[row,col+1]
        B[row,col+1] = 0
        neighbors.append(B)

    return neighbors


def A_star(mat, heuristic):
    
    '''
    Summary: Performs A* search on a given 8-puzzle
    Input: mat - matrix representing the 8-puzzle
    Output: path - list of states visited
            states_visited - list of states visited
            num_iters - number of iterations
    '''

    # initialize the path, states_visited, and num_iters
    path = []
    states_visited = []
    num_iters = 0

    # initialize the frontier and explored sets
    frontier = []
    explored = []

    # add the initial state to the frontier
    frontier.append((mat, heuristic(mat)))

    # while the frontier is not empty
    while frontier:

        # sort the frontier by the heuristic
        frontier.sort(key = lambda x: x[1])

        # pop the state with the lowest heuristic from the frontier
        state = frontier.pop(0)

        # add the state to the explored set
        explored.append(state[0])

        # add the state to the path
        path.append(state[0])

        # add the state to the states_visited
        states_visited.append(state[0])

        # if the state is the goal state
        if np.array_equal(state[0], np.matrix([[0,1,2],[3,4,5],[6,7,8]])):
            return path, states_visited, num_iters

        # get the neighbors of the state
        neighbors = get_neighbors(state[0])

        # for each neighbor
        for neighbor in neighbors:

            # if the neighbor is not in the frontier or explored set
            if  not any(np.array_equal(neighbor, state) for state in frontier) and not any(np.array_equal(neighbor, state) for state in explored):

                # add the neighbor to the frontier
                frontier.append((neighbor, heuristic(neighbor)))

        # increment the number of iterations
        num_iters += 1

    return path, states_visited, num_iters

def read_file(filename):
    '''
    Summary: Reads a file and returns a list of 8-puzzles
    Input: filename - name of file to read
    Output: list of 8-puzzles
    '''

    # open the file
    f = open(filename, "r")

    # initialize the list of 8-puzzles
    puzzles = []

    # read the file line by line
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [lines for lines in lines if lines != '/////////////////////////////////////////////////////']
    lines = [[int(num) for num in line.split(" ")] for line in lines]

    # convert the lines to 8-puzzles
    for i in range(0,len(lines),3):
        puzzle = np.matrix(lines[i:i+3])
        puzzles.append(puzzle)
    
    return puzzles

def Experiment():
    for i in range(4,21,4):
        df = pd.DataFrame(columns = ["Length", "Manhattan Distance", "Manhattan Distance Time", "Displaced Tiles", "Displaced Tiles Time"])
        puzzles = read_file("Length" + str(i) + ".txt")
        for puzzle in puzzles:
            if check_is_valid(puzzle):
                start = time.time()
                path_mahat, states_visited, num_iters = A_star(puzzle, manhattan_distances)
                end = time.time()
                manhattan_time = end - start

                start = time.time()
                path_displaced, states_visited, num_iters = A_star(puzzle, displaced_distances)
                end = time.time()
                displaced_time = end - start

                df = pd.concat([df, pd.DataFrame([[len(path_mahat), manhattan_distances(puzzle), manhattan_time, displaced_distances(puzzle), displaced_time]], columns = ["Length", "Manhattan Distance", "Manhattan Distance Time", "Displaced Tiles", "Displaced Tiles Time"])])        
                df.to_csv("Length" + str(i) + ".csv", index = False)


def main():
   Experiment()
main()