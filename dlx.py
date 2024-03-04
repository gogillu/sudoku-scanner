def is_valid(board, row, col, num):
    # Check if the number is already in the row
    if num in board[row]:
        return False
    
    # Check if the number is already in the column
    if num in [board[i][col] for i in range(9)]:
        return False
    
    # Check if the number is already in the 3x3 box
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    
    return True

def find_empty_location(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return -1, -1

def solve_sudoku(board):
    row, col = find_empty_location(board)
    if row == -1 and col == -1:
        return True  # If no empty location left, Sudoku is solved
    
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0  # Backtrack if the solution is not valid
    return False

def solve(input_board):
    board = [list(row) for row in input_board]
    if solve_sudoku(board):
        return board
    else:
        return "No solution exists."

# Example usage:
input_board = [
    [0, 0, 0, 1, 0, 4, 0, 0, 0],
    [0, 9, 1, 2, 7, 0, 0, 8, 0],
    [0, 0, 8, 0, 6, 0, 0, 1, 0],
    [3, 0, 0, 0, 0, 0, 8, 6, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 9],
    [0, 2, 6, 0, 0, 0, 0, 0, 4],
    [0, 6, 0, 0, 5, 0, 2, 0, 0],
    [0, 1, 0, 0, 4, 3, 7, 5, 0],
    [0, 0, 0, 6, 0, 7, 0, 0, 0]
]

solved_board = solve(input_board)
for row in solved_board:
    print(row)
