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

class Node:
    def __init__(self, state, parent, g, h):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        else:
            return np.array_equal(self.state, other.state)
        
        return np.array_equal(self.state, other.state)

    def __hash__(self):
        return hash(str(self.state))

def A_star(mat, heuristic):
    
    '''
    Summary: Performs A* search on a given 8-puzzle
    Input: mat - matrix representing the 8-puzzle
    Output: path - list of states visited
            states_visited - list of states visited
            num_iters - number of iterations
    '''

    # intialize frontier and explored
    frontier = []
    explored = set()

    # initialize the root node
    root = Node(mat, None, 0, heuristic(mat))

    # add the root node to the frontier
    frontier.append(root)

    # initialize the path
    path = []

    # initialize the number of iterations
    num_iters = 0

    # while the frontier is not empty
    while len(frontier) > 0:
            
            # sort the frontier
            frontier.sort()
    
            # get the node with the lowest f value
            node = frontier.pop(0)
    
            # check if node is the goal
            if heuristic(node.state) == 0:
                # backtrack to get the path
                while node.parent != None:
                    path.append(node.state)
                    node = node.parent
                path.append(node.state)
                path.reverse()
                return path, explored, num_iters
    
            # add the node to the explored set
            explored.add(node)
    
            # get the neighbors of the node
            neighbors = get_neighbors(node.state)
    
            # for each neighbor
            for neighbor in neighbors:
                # create a node
                neighbor_node = Node(neighbor, node, node.g + 1, heuristic(neighbor))
    
                # check if the neighbor is in the frontier or explored
                if neighbor_node not in frontier and neighbor_node not in explored:
                    # add the neighbor to the frontier
                    frontier.append(neighbor_node)
                    num_iters += 1
    
                # check if the neighbor is in the frontier
                elif neighbor_node in frontier:
                    # get the node in the frontier
                    frontier_node = frontier[frontier.index(neighbor_node)]
    
                    # check if the neighbor has a lower f value
                    if neighbor_node.f < frontier_node.f:
                        # replace the frontier node with the neighbor node
                        frontier[frontier.index(neighbor_node)] = neighbor_node
                        num_iters += 1

                
  

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

def generate_puzzles(num, depth = 100):
    '''
    Summary: Generates a list of 8-puzzles
    Input: depth - depth of 8-puzzles to generate
    Output: list of 8-puzzles
    '''

    # initialize the list of 8-puzzles
    puzzles = []

    # generate the 8-puzzles
    for i in range(num):
        puzzle = np.matrix([[0,1,2],[3,4,5],[6,7,8]])
        for j in range(int(depth)):
            neighbors = get_neighbors(puzzle)
            puzzle = neighbors[np.random.randint(0,len(neighbors))]
        puzzles.append(puzzle)

    return puzzles

def Experiment():

    '''
    Summary: Performs the experiment for the 8-puzzle. Saves results to csv files
    Input: None
    Output: None
    '''

    avg_search_mahattan = []
    avg_time_mahattan = []

    avg_search_displaced = []
    avg_time_displaced = []

    for i in range(4,21,4):

        # create csv file
        df = pd.DataFrame(columns = ["d", "Manhattan Distance", "Manhattan Distance Time", "Displaced Tiles", "Displaced Tiles Time"])

        # averages the results for each length
        mahattan_searches = []
        displaced_searches = []

        mahattan_times = []
        displaced_times = []


        puzzles = read_file("Length" + str(i) + ".txt")
        for puzzle in puzzles:
            if check_is_valid(puzzle):
                start = time.time()
                path_mahat, states_visited_mahat, num_iters = A_star(puzzle, manhattan_distances)
                end = time.time()
                manhattan_time = end - start

                start = time.time()
                path_displaced, states_visited_displaced, num_iters = A_star(puzzle, displaced_distances)
                end = time.time()
                displaced_time = end - start

                mahattan_searches.append(len(states_visited_mahat))
                displaced_searches.append(len(states_visited_displaced))

                mahattan_times.append(manhattan_time)
                displaced_times.append(displaced_time)

                df = pd.concat([df, pd.DataFrame([[i,len(states_visited_mahat), manhattan_time, len(states_visited_mahat), displaced_time]], columns = ["d", "Manhattan Distance", "Manhattan Distance Time", "Displaced Tiles", "Displaced Tiles Time"])])        
                df.to_csv("Length" + str(i) + ".csv", index = False)
        
        avg_search_mahattan.append(np.mean(mahattan_searches))
        avg_time_mahattan.append(np.mean(mahattan_times))

        avg_search_displaced.append(np.mean(displaced_searches))
        avg_time_displaced.append(np.mean(displaced_times))

    df = pd.DataFrame(columns = ["Length", "Avg Manhattan Distance", " Avg Manhattan Distance Time", "Avg Displaced Tiles", "Avg Displaced Tiles Time"])
    df = pd.concat([df, pd.DataFrame([[4, avg_search_mahattan[0], avg_time_mahattan[0], avg_search_displaced[0], avg_time_displaced[0]]], columns = ["Length", "Avg Manhattan Distance", " Avg Manhattan Distance Time", "Avg Displaced Tiles", "Avg Displaced Tiles Time"])])
    df = pd.concat([df, pd.DataFrame([[8, avg_search_mahattan[1], avg_time_mahattan[1], avg_search_displaced[1], avg_time_displaced[1]]], columns = ["Length", "Avg Manhattan Distance", " Avg Manhattan Distance Time", "Avg Displaced Tiles", "Avg Displaced Tiles Time"])])
    df = pd.concat([df, pd.DataFrame([[12, avg_search_mahattan[2], avg_time_mahattan[2], avg_search_displaced[2], avg_time_displaced[2]]], columns = ["Length", "Avg Manhattan Distance", " Avg Manhattan Distance Time", "Avg Displaced Tiles", "Avg Displaced Tiles Time"])])
    df = pd.concat([df, pd.DataFrame([[16, avg_search_mahattan[3], avg_time_mahattan[3], avg_search_displaced[3], avg_time_displaced[3]]], columns = ["Length", "Avg Manhattan Distance", " Avg Manhattan Distance Time", "Avg Displaced Tiles", "Avg Displaced Tiles Time"])])
    df = pd.concat([df, pd.DataFrame([[20, avg_search_mahattan[4], avg_time_mahattan[4], avg_search_displaced[4], avg_time_displaced[4]]], columns = ["Length", "Avg Manhattan Distance", " Avg Manhattan Distance Time", "Avg Displaced Tiles", "Avg Displaced Tiles Time"])])
    df.to_csv("Summary.csv", index = False)


def main():
    print("CS 4200 Project 1")

    # Prompt user for input
    print("Select:")
    print("1. Single Test Puzzle")
    print("2. Exit")

    print()
    choice = input("Enter Choice: ")
    print()

    if choice == '3':
        exit()

    num_of_puzzles = 1

    print("Select Input Method:")
    print("1. Manual")
    print("2. Random")
    print()
    input_method = input("Enter Choice: ")
    print()

    if input_method == '1':
        row1 = input("Enter Row 1: ")
        row2 = input("Enter Row 2: ")
        row3 = input("Enter Row 3: ")
        
    
    print("Select Heuristic:")
    print("1. Manhattan Distance")
    print("2. Displaced Tiles")
    print()
    heuristic = input("Enter Choice: ")
    print()

    if input_method == '1':
        puzzle = np.matrix([[int(num) for num in row1.split(" ")] ,[int(num) for num in row2.split(" ")] ,[int(num) for num in row3.split(" ")]])
    else:
        puzzle = generate_puzzles(num_of_puzzles)[0]

    print("Puzzle:")
    print(puzzle)

    
    if heuristic == '1':
        start = time.time()
        path, states_visited, num_iters = A_star(puzzle, manhattan_distances)
        end = time.time()
    else:
        start = time.time()
        path, states_visited, num_iters = A_star(puzzle, displaced_distances)
        end = time.time()

    print("Path:")
    count = 1
    for state in path:
        print("Step " + str(count))
        count += 1

        # pretty print state
        print(state[0,0], state[0,1], state[0,2])
        print(state[1,0], state[1,1], state[1,2])
        print(state[2,0], state[2,1], state[2,2])
        print()
        print()

    print("Search Cost: " + str(len(states_visited)))
    print("Time: " + str(end - start))

main()