import numpy as np
import time 
def create_board():

    '''
    Summary: Create an 8x8 board with all values set to -1
    Input: None
    Output: 8x8 numpy array
    '''

    return np.full((8, 8), -1)

def print_board(board):
    
        '''
        Summary: Print the board to the console
        Input: 8x8 numpy array
        Output: None
        '''

        # Column header
        col = ["A", "B", "C", "D", "E", "F", "G", "H"]
        # print row header
        for i in [" ", "1", "2", "3", "4", "5", "6", "7", "8"]:
            print(i, end=" ")
        print()

        for i in range(8):
            print(col[i], end=" ")
            for j in range(8):
                if board[i][j] == -1:
                    print("-", end=" ")
                elif board[i][j] == 0:
                    print("O", end=" ")
                else:
                    print("X", end=" ")
            print()
        print()

def x_moves(board):
     
     '''
     Summary: Return a list of all moves made by X so far
     Input: 8x8 numpy array
     Output: List of tuples
     '''
     moves = []
     for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                moves.append((i, j))
    
     return moves

def o_moves(board):
        
        '''
        Summary: Return a list of all moves made by O so far
        Input: 8x8 numpy array
        Output: List of tuples
        '''
        
        moves = []
        for i in range(8):
            for j in range(8):
                if board[i][j] == 0:
                    moves.append((i, j))
        
        return moves

def valid_move(board, move):
    
    '''
    Summary: Check if a move is valid
    Input: 8x8 numpy array, tuple
    Output: Boolean
    '''
    
    if move in x_moves(board) or move in o_moves(board) or move[0] < 0 or move[0] > 7 or move[1] < 0 or move[1] > 7:
        return False
    else:
        return True
    
def heuristic(board):

    '''
    Summary: Calculate the heuristic value of a board
    Input: 8x8 numpy array
    Output: Integer
    '''

    # Calculate the number of moves made by X and O
    x = 0
    o = 0

    # Check if X has 3 in a row horizontally
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                if j < 6 and board[i][j + 1] == 1 and board[i][j + 2] == 1:
                    x += 100

    # Check if X has 3 in a row vertically
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                if i < 6 and board[i + 1][j] == 1 and board[i + 2][j] == 1:
                    x += 100
    
    # Check if X has 2 in a row hoirzontally with an empty space on either side
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                if j < 6 and board[i][j + 1] == 1 and board[i][j + 2] == -1:
                    x += 10
                if j > 1 and board[i][j - 1] == 1 and board[i][j - 2] == -1:
                    x += 10
    
    # Check if X has 2 in a row vertically with an empty space on either side
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                if i < 6 and board[i + 1][j] == 1 and board[i + 2][j] == -1:
                    x += 10
                if i > 1 and board[i - 1][j] == 1 and board[i - 2][j] == -1:
                    x += 10
    
    # Check if X has 1 in a row horizontally with an empty space on either side
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                if j < 7 and board[i][j + 1] == -1:
                    x += 1
                if j > 0 and board[i][j - 1] == -1:
                    x += 1
    
    # Check if X has 1 in a row vertically with an empty space on either side
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                if i < 7 and board[i + 1][j] == -1:
                    x += 1
                if i > 0 and board[i - 1][j] == -1:
                    x += 1
    
    # Check if O has 3 in a row horizontally
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                if j < 6 and board[i][j + 1] == 0 and board[i][j + 2] == 0:
                    o += 100
    
    # Check if O has 3 in a row vertically
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                if i < 6 and board[i + 1][j] == 0 and board[i + 2][j] == 0:
                    o += 100
    
    # Check if O has 2 in a row horizontally with an empty space on either side
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                if j < 6 and board[i][j + 1] == 0 and board[i][j + 2] == -1:
                    o += 10
                if j > 1 and board[i][j - 1] == 0 and board[i][j - 2] == -1:
                    o += 10
    
    # Check if O has 2 in a row vertically with an empty space on either side
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                if i < 6 and board[i + 1][j] == 0 and board[i + 2][j] == -1:
                    o += 10
                if i > 1 and board[i - 1][j] == 0 and board[i - 2][j] == -1:
                    o += 10
    
    # Check if O has 1 in a row horizontally with an empty space on either side
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                if j < 7 and board[i][j + 1] == -1:
                    o += 1
                if j > 0 and board[i][j - 1] == -1:
                    o += 1
    
    # Check if O has 1 in a row vertically with an empty space on either side
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                if i < 7 and board[i + 1][j] == -1:
                    o += 1
                if i > 0 and board[i - 1][j] == -1:
                    o += 1
    
    return x - o

def minimax(board, depth, alpha, beta, maximizing_player):
    
    '''
    Summary: Use the minimax algorithm to find the best move for a player
    Input: 8x8 numpy array, integer, integer, integer, boolean
    Output: Tuple
    '''

    # Check if the game is over or the maximum depth has been reached
    if game_over(board) or depth == 2:
        return (heuristic(board), None)
    
    else:
        # If the player is maximizing
        if maximizing_player:
            max_eval = float("-inf")
            best_move = None

            # Iterate through all possible moves
            for move in possible_moves(board):
                board[move[0]][move[1]] = 1
                eval = minimax(board, depth + 1, alpha, beta, False)[0]
                board[move[0]][move[1]] = -1
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            
            return (max_eval, best_move)
        
        # If the player is minimizing
        else:
            min_eval = float("inf")
            best_move = None

            # Iterate through all possible moves
            for move in possible_moves(board):
                board[move[0]][move[1]] = 0
                eval = minimax(board, depth + 1, alpha, beta, True)[0]
                board[move[0]][move[1]] = -1
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            
            return (min_eval, best_move)
    
def possible_moves(board):
    
        '''
        Summary: Return a list of all possible moves
        Input: 8x8 numpy array
        Output: List of tuples
        '''
    
        moves = []
        for i in range(8):
            for j in range(8):
                if board[i][j] == -1:
                    moves.append((i, j))
        
        return moves

def game_over(board):

    '''
    Summary: Check if the game is over
    Input: 8x8 numpy array
    Output: Boolean
    '''

    # Check if X has won by getting 4 in a row
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:
                if j < 5 and board[i][j + 1] == 1 and board[i][j + 2] == 1 and board[i][j + 3] == 1:
                    return True
                if i < 5 and board[i + 1][j] == 1 and board[i + 2][j] == 1 and board[i + 3][j] == 1:
                    return True
                
    
    # Check if O has won by getting 4 in a row verically or horizontally
    for i in range(8):
        for j in range(8):
            if board[i][j] == 0:
                if j < 5 and board[i][j + 1] == 0 and board[i][j + 2] == 0 and board[i][j + 3] == 0:
                    return True
                if i < 5 and board[i + 1][j] == 0 and board[i + 2][j] == 0 and board[i + 3][j] == 0:
                    return True
                
    
    # If the board is full, return True
    for i in range(8):
        for j in range(8):
            if board[i][j] == -1:
                return False
    
    return True

def main():

    move_dict = {"a1": (0, 0), "a2": (0, 1), "a3": (0, 2), "a4": (0, 3), "a5": (0, 4), "a6": (0, 5), "a7": (0, 6), "a8": (0, 7),
                    "b1": (1, 0), "b2": (1, 1), "b3": (1, 2), "b4": (1, 3), "b5": (1, 4), "b6": (1, 5), "b7": (1, 6), "b8": (1, 7),
                    "c1": (2, 0), "c2": (2, 1), "c3": (2, 2), "c4": (2, 3), "c5": (2, 4), "c6": (2, 5), "c7": (2, 6), "c8": (2, 7),
                    "d1": (3, 0), "d2": (3, 1), "d3": (3, 2), "d4": (3, 3), "d5": (3, 4), "d6": (3, 5), "d7": (3, 6), "d8": (3, 7),
                    "e1": (4, 0), "e2": (4, 1), "e3": (4, 2), "e4": (4, 3), "e5": (4, 4), "e6": (4, 5), "e7": (4, 6), "e8": (4, 7),
                    "f1": (5, 0), "f2": (5, 1), "f3": (5, 2), "f4": (5, 3), "f5": (5, 4), "f6": (5, 5), "f7": (5, 6), "f8": (5, 7),
                    "g1": (6, 0), "g2": (6, 1), "g3": (6, 2), "g4": (6, 3), "g5": (6, 4), "g6": (6, 5), "g7": (6, 6), "g8": (6, 7),
                    "h1": (7, 0), "h2": (7, 1), "h3": (7, 2), "h4": (7, 3), "h5": (7, 4), "h6": (7, 5), "h7": (7, 6), "h8": (7, 7)
                 }
    

    move_history = []

    first = input("Would you like to go first? (y/n): ")

    if first == "y":
        turn = True
    else:
        turn = False
    
    board = create_board()
    print_board(board)

    while not game_over(board):
      if turn:
        move = input("Enter your move: ")
        move = move.lower()
        move = move_dict[move]
        while not valid_move(board, move):
            print("Invalid move")
            move = input("Enter your move: ")
            move = move.lower()
            move_history.append(move)
            move = move_dict[move]
        board[move[0]][move[1]] = 0

        print_board(board)
        turn = False

      else:
            start = time.time()
            move = minimax(board, 0, float("-inf"), float("inf"), False)[1]
            end = time.time()

            board[move[0]][move[1]] = 1
            key_list = list(move_dict.keys())
            val_list = list(move_dict.values())
            index = val_list.index(move)
            move = key_list[index]
            move_history.append(move)

            print("Computer move: " + str(move) + " in " + str(end - start) + " seconds")
            print_board(board)


            turn = True
        

      if game_over(board):
            break

    
    if heuristic(board) > 0:
        print("X wins")
    elif heuristic(board) < 0:
        print("O wins")
    else:
        print("Draw")

    print("Move history: ")
    print_move_history(move_history, first)

def print_move_history(move_history, first):
    
    '''
    Summary: Print the move history to the console
    Input: List of tuples, boolean
    Output: None
    '''

    if first == "y":
        for i in range(len(move_history)):
            if i % 2 == 0:
                print(str(i+1) + ". X: " + str(move_history[i]))
            else:
                print(str(i+1) + ". O: " + str(move_history[i]))
    else:
        for i in range(len(move_history)):
            if i % 2 == 0:
                print(str(i+1) + ". O: " + str(move_history[i]))
            else:
                print(str(i+1) + ". X: " + str(move_history[i]))


if __name__ == "__main__":
    main()


    