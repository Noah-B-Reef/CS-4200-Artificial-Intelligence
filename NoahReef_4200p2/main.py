import numpy as np
import pandas as pd
import time
import sys

class Node:

    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.fitness = measure_fitness(state)


def uphill_ascent(initial):
    '''
    Summary: Finds greedy optimal solution to the n-queens problem using hill climbing algorithm
    Input: initial state
    Output: optimal solution
    '''
    
    # Initialize current state
    current = Node(initial, None)

    # Initialize neighbors
    neighbors = actions(current.state)


    # Get best neighbor
    neighbors.sort(key=lambda x: measure_fitness(x))
    best_neighbor = neighbors[0]
 
    count = 0
    # While the best neighbor is not the current state
    while not (best_neighbor == current.state).all():

        # Set current state to best neighbor
        current = Node(best_neighbor, current)

        # Get neighbors of current state
        neighbors = actions(current.state)

        
        # Get best neighbor
        neighbors.sort(key=lambda x: measure_fitness(x))
        best_neighbor = neighbors[0]
        count += 1

    return best_neighbor,count




def measure_fitness(state):
    '''
    Summary: Measures the fitness of a given state (number of queens attacking each other)
    Input: state
    Output: fitness
    '''
    
    if len(queen_locations(state)) != 8:
        return 50000
    
    # Initialize fitness
    fitness = 0

    # For each row
    for i in range(8):
        # For each column
        for j in range(8):
            # If there is a queen
            if state[i,j] == 1:
                # Add the number of queens attacking each other
                fitness += check_diagonal(state, i, j)
                fitness += check_row(state, i, j)
                fitness += check_column(state, i, j)

    return fitness

def check_row(state, row, col):
    '''
    Summary: Checks the rows of a given state
    Input: state, row, column
    Output: number of queens attacking each other
    '''

    # Initialize counter
    count = 0

    # Check left
    for i in range(col-1, -1, -1):
        if state[row,i] == 1:
            count += 1

    # Check right
    for i in range(col+1, 8):
        if state[row,i] == 1:
            count += 1

    return count

def check_column(state, row, col):
    '''
    Summary: Checks the columns of a given state
    Input: state, row, column
    Output: number of queens attacking each other
    '''

    # Initialize counter
    count = 0

    # Check up
    for i in range(row-1, -1, -1):
        if state[i,col] == 1:
            count += 1

    # Check down
    for i in range(row+1, 8):
        if state[i,col] == 1:
            count += 1

    return count

def check_diagonal(state, row, col):
    '''
    Summary: Checks the diagonals of a given state
    Input: state, row, column
    Output: number of queens attacking each other
    '''

    # Initialize counter
    count = 0

    # Check upper left diagonal
    for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if state[i,j] == 1:
            count += 1

    # Check upper right diagonal
    for i, j in zip(range(row-1, -1, -1), range(col+1, 8)):
        if state[i,j] == 1:
            count += 1

    # Check lower left diagonal
    for i, j in zip(range(row+1, 8), range(col-1, -1, -1)):
        if state[i,j] == 1:
            count += 1

    # Check lower right diagonal
    for i, j in zip(range(row+1, 8), range(col+1, 8)):
        if state[i,j] == 1:
            count += 1

    return count

def actions(state):
    '''
    Summary: Finds all possible actions of where a queen can be moved
    Input: state
    Output: list of actions
    '''
    
    # Initialize actions
    actions = [state]

    queens = queen_locations(state)

    for queen in queens:
        A = np.copy(state)
        A[queen] = 0
        # All possible actions for a given queen

        # Check rows
        for i in range(8):
            if i != queen[0] and A[i, queen[1]] != 1:
                A[i, queen[1]] = 1
                actions.append(A)
                A = np.copy(state)
                A[queen] = 0
        
        # Check columns
        for i in range(8):
            if i != queen[1] and A[queen[0], i] != 1:
                A[queen[0], i] = 1
                actions.append(A)
                A = np.copy(state)
                A[queen] = 0
        
    
        # Check diagonals
        for i in range(8):
            if i != queen[0] and i != queen[1] and A[i,i] != 1:
                    A[i,i] = 1
                    actions.append(A)
                    A = np.copy(state)
                    A[queen] = 0

    
    return actions

def queen_locations(state):
    '''
    Summary: Finds the locations of all queens on the board
    Input: state
    Output: list of queen locations
    '''

    # Initialize queen locations
    locations = []

    # For each row
    for i in range(8):
        # For each column
        for j in range(8):
            # If there is a queen
            if state[i,j] == 1:
                # Add the location to the list
                locations.append((i,j))

    return locations

def generate_random_state():
    '''
    Summary: Generates a random state for the n-queens problem
    Input: None
    Output: random state
    '''
    
    state = np.zeros(64)
    state[:8] =  1
    np.random.shuffle(state)
    state = state.reshape(8,8)

    return state

def print_state(state):
    '''
    Summary: Prints a given state
    Input: state
    Output: None
    '''
    for row in state:
        for i in row:
            print(int(i), end=' ')
        print()
    print()


def genetic_algorithm():
    '''
    Summary: Finds optimal solution to the n-queens problem using genetic algorithm
    Input: initial state
    Output: optimal solution
    '''

    # Initialize population with fitness scores
    population = []
    for i in range(1000):
        state = generate_random_state()
        population.append((state, 1/np.exp(measure_fitness(state))))

    best_state = population[0]
    while best_state[1] != 1:
        # Choose parents
        parents = choose_parents(population)

        # Crossover
        children = []
        
        # sort parents by fitness
        parents.sort(key=lambda x: x[1], reverse=True)

        for i in range(len(parents) // 2):
            children.append(crossover(parents[i][0], parents[i+1][0]))
        
        # Mutate
        mut_prob = 0.2

        for i in range(len(children)):
            if np.random.random() < mut_prob:
                children[i] = mutate(children[i])
        
        # upate population
        for i in range(len(children)):
            population[i] = (children[i], 1/np.exp(measure_fitness(children[i])))
            population[i+1] = parents[i]

        population.sort(key=lambda x: x[1], reverse=True) 

        best_state = population[0]
        print("Best state: ")
        print_state(population[0][0])
        print("Best fitness: ", population[0][1])
        
    population.sort(key=lambda x: x[1],reverse=True)
    return population[0][0]

def crossover(parent1, parent2):
    '''
    Summary: Performs crossover on two parents
    Input: parent1, parent2
    Output: child
    '''

    # Initialize child
    child = np.zeros((8,8))

    child[:4,:] = parent1[:4,:]
    child[4:,:] = parent2[4:,:]


    
    return child

def mutate(child):
    '''
    Summary: Mutates a child
    Input: child
    Output: mutated child
    '''
    
    # flip queen bit in a random position
    row = np.random.randint(0,8)
    col = np.random.randint(0,8)

    child[row,col] = 1

    # find random queen and remove it
    queens = queen_locations(child)
    queen = np.random.choice(len(queens))
    child[queens[queen]] = 0


    return child

def choose_parents(population):
    '''
    Summary: Chooses a parent from the population
    Input: population
    Output: parent
    '''

    # Initialize probabilities
    probs = []

    # For each individual in the population
    for individual in population:
        # Add the probability of being chosen to the list
        probs.append(individual[1]/sum_fitness(population))

    # Choose a parent
    idx = np.random.choice(len(population),size=int(0.5 * len(population)), p=probs)
    parents = [population[i] for i in idx]



    return parents

def sum_fitness(population):
    '''
    Summary: Sums the fitness of the population
    Input: population
    Output: sum of fitness
    '''

    # Initialize sum
    sum = 0

    # For each individual in the population
    for individual in population:
        # Add the fitness to the sum
        sum += individual[1]

    return sum

    
    


def Experiment():
    '''
    Summary: Runs the experiment for the n-queens problem
    Input: None
    Output: None
    '''
    
    # Initialize dataframe
    df = pd.DataFrame(columns=['problem', 'search cost', 'avg time', 'solved'])

    for i in range(100):
        # Generate random state
        state = generate_random_state()

        start = time.time()
        # Run hill climbing algorithm
        sol, cost = uphill_ascent(state)
        end = time.time()

        if measure_fitness(sol) != 0:
            solved = 0
        else:
            solved = 1

        # Add data to dataframe
        df = pd.concat([df, pd.DataFrame([[i, cost, end-start, solved]], columns=['problem', 'search cost', 'avg time', 'solved'])])
    
    # print precentage of solved problems
    print("Percentage of solved problems: ", df['solved'].sum() / len(df))

    # Save dataframe to csv
    df.to_csv('hill_climbing.csv', index=False)

def main():
    if sys.argv[1] == 'hill_climbing':
       state = generate_random_state()
       sol, cost = uphill_ascent(state)
       print("Initital State: ")
       print_state(state)
       print("Solution: ")
       print_state(sol)
       print("Search Cost: ", cost)

    elif sys.argv[1] == 'genetic':
        sol = genetic_algorithm()
        print("Solution: ")
        print_state(sol)
        print("Fitness: ", 1/np.exp(measure_fitness(sol)))
    
    elif sys.argv[1] == 'experiment':
        Experiment()
main()