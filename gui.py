import sys
import random
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from tenserflow_machine_digit_predict_model import *
import threading

# from dlx import DLX
import numpy as np
class Worker(QThread):
    update_signal = pyqtSignal(object)  # Signal to update the UI


    def run(self):
        # This is the code that will run in a separate thread
        for i in range(1):  # Let's say we want to update 10 times
            time.sleep(0.05)  # Sleep for 1 second
            new_matrix = generate_random_matrix()
            self.update_signal.emit(new_matrix)  # Emit the signal with the new matrix

    def register_ui(self, main_window):
        self.main_window = main_window

    def is_valid_move(self, row, col, num, matrix):
        # Check row
        for j in range(9):
            if matrix[row][j] == num:
                return False

        # Check column
        for i in range(9):
            if matrix[i][col] == num:
                return False

        # Check subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if matrix[i][j] == num:
                    return False

        return True

    def solve_sudoku(self, matrix,responsive=True, ui_resp = 0.002):
        for i in range(9):
            for j in range(9):
                if matrix[i][j] == 0:
                    for num in range(1, 10):
                        if self.is_valid_move(i, j, num, matrix):
                            matrix[i][j] = num
                            if responsive:
                                time.sleep(ui_resp)
                            # time.sleep(0.00000000011)
                            self.update_signal.emit(self.transform_1_9_or_empty_sudoku(matrix))
                            # time.sleep(6)                            
                            if self.solve_sudoku(matrix):
                                # time.sleep(0.00000000011)
                                # self.update_signal.emit(self.transform_1_9_or_empty_sudoku(matrix))
                                # print(matrix)
                                return True
                            matrix[i][j] = 0  # Undo the choice if it doesn't lead to a solution
                    return False
        return True
    
    def transform_1_9_or_empty_sudoku(self, matrix):
        # nM = [["" for _ in range(9)] for _ in range(9)]  # Initialize a 9x9 matrix with all zeros
        nM = []
        i=0
        for _ in range(9):
            row = []
            j=0
            for _ in range(9):
                if matrix[i][j] > 0:
                    row.append(str(matrix[i][j]))
                else:
                    row.append("")
        
                j += 1
            i += 1
            nM.append(row)

        return nM


    # for i in range(0,9):
    #     for j in range(0,9):
    #         print(i,j,matrix[i][j],type(matrix[i][j]),matrix[i][j] == 0)
    #         if int(matrix[i][j]) == 0:
    #             nM[i][j] == ""
    #             print("*. ",nM)
    #         else:
    #             nM[i][j] == str(matrix[i][j])
    #             print("**. ",nM)        
    #         print("__. ",nM)        
    
    # print("==>",nM)
    # return nM

class MainWindow(QWidget):
    def __init__(self, matrix, worker):
        super().__init__()
        self.matrix = matrix
        self.prefilled_matrix = [[False] * 9 for _ in range(9)]
        self.worker = worker
        self.model, self.class_names = loadModel()
        self.edit_boxes = []
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        for i in range(9):
            row = []
            for j in range(9):
                edit_box = QLineEdit(str(self.matrix[i][j]))
                edit_box.setAlignment(Qt.AlignCenter)
                edit_box.setStyleSheet("background-color: white; color: black; font-size: 24px;")
                edit_box.setFixedSize(40, 40)
                layout.addWidget(edit_box, i, j)
                row.append(edit_box)
            self.edit_boxes.append(row)

        button_layout = QVBoxLayout()

        button1 = QPushButton('Load sudoku from scanned image')
        button1.clicked.connect(self.load)
        button_layout.addWidget(button1)

        button2 = QPushButton('Solve slowly')
        button2.clicked.connect(self.solve_old)
        button_layout.addWidget(button2)

        button3 = QPushButton('Solve instantly')
        button3.clicked.connect(self.solve_instantly)
        button_layout.addWidget(button3)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.setWindowTitle('9x9 Edit Boxes')
        self.setGeometry(100, 100, 500, 500)
        self.show()

    def update_values(self, matrix):
        for i in range(9):
            for j in range(9):
                self.edit_boxes[i][j].setText(str(matrix[i][j]))
                if (i // 3 + j // 3) % 2 == 0:
                    self.edit_boxes[i][j].setStyleSheet("background-color: lightgray; color: black; font-size: 24px;")
                else:
                    self.edit_boxes[i][j].setStyleSheet("background-color: gray; color: black; font-size: 24px;")

    def update_values_prefilled(self, matrix):
        for i in range(9):
            for j in range(9):
                self.edit_boxes[i][j].setText(str(matrix[i][j]))
                if self.prefilled_matrix[i][j]:
                    if (i // 3 + j // 3) % 2 == 0:
                        self.edit_boxes[i][j].setStyleSheet("background-color: lightgray; color: black; font-size: 24px;")
                    else:
                        self.edit_boxes[i][j].setStyleSheet("background-color: gray; color: black; font-size: 24px;")
                    # self.edit_boxes[i][j].setStyleSheet("background-color: white; color: black; font-size: 32px; border: 2px solid yellow;")
                elif (i // 3 + j // 3) % 2 == 0:
                    self.edit_boxes[i][j].setStyleSheet("background-color: lightgray; color: blue; font-family: Lucida Console; font-size: 36px;")
                else:
                    self.edit_boxes[i][j].setStyleSheet("background-color: gray; color: blue; font-family: Lucida Console; font-size: 36px;")

    def load(self):
        img_matrix = [[""] * 9 for _ in range(9)]
        # new_matrix = generate_random_matrix()
        # self.worker.update_signal.emit(new_matrix)
        print("Option 1 clicked")
    
        z = 0
        for i in range(0,9):
            print('\n')
            for j in range(0,9):

            # non_hidden_files = [file for file in os.listdir('individual_grids') if not file.startswith('.')]
            # for entry in non_hidden_files:
            # full_path = '/Users/govind/projects/self/sudoku/individual_grids/'+entry #os.path.join('individual_grids', entry)

                if True:
                    print(' ')
                    z += 1

                    # img_matrix[i][j] = 'individual_grids/grid_'+str(z)+'.jpg'
                    image_path = 'individual_grids/grid_'+str(z)+'.jpg'  # Specify the path to an image of a digit
                    v = predict_with_teachable_ml_optimized(image_path,self.model,self.class_names)
                    if v > 0:
                        print(i,j,v)
                        img_matrix[i][j] = str(v)
                        self.prefilled_matrix[i][j] = True
                        self.worker.update_signal.connect(main_window.update_values_prefilled)

                    else:
                        img_matrix[i][j] = ""
                    self.matrix[i][j] = v

        self.worker.update_signal.emit(img_matrix)


    def solve_old(self):
        for i in range(9):
            for j in range(9):
                text = self.edit_boxes[i][j].text()
                if text.isdigit():
                    self.matrix[i][j] = int(text)
                else:
                    self.matrix[i][j] = 0  # Treat empty fields as 0s
    
        # Read the matrix
        print(self.matrix)

        if is_valid_sudoku(self.matrix):
            print("valid sudoku")
        else:
            print("invalid sudoku")

        # self.worker.solve_sudoku(self.matrix)
        thread = threading.Thread(target=run_method, args=(self.worker, "solve_sudoku", self.matrix))
        thread.start()

        # new_matrix = generate_random_matrix()
        # self.worker.update_signal.emit(new_matrix)
        print("Sudoku solved ...")

    def solve_instantly(self):
        print("x")
        sol, done = self.solve(self.matrix)
        # R = self.worker.solve_sudoku(self.matrix,False,0)
        if done:
            print("solved")
            print(sol)
            self.worker.update_signal.emit(sol)
        else:
            print(self.matrix)
            print("sol")


    def is_valid(self, board, row, col, num):
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

    def find_empty_location(self, board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return i, j
        return -1, -1

    def solve_sudoku_f(self, board):
        row, col = self.find_empty_location(board)
        if row == -1 and col == -1:
            return True  # If no empty location left, Sudoku is solved
        
        for num in range(1, 10):
            if self.is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_sudoku_f(board):
                    return True
                board[row][col] = 0  # Backtrack if the solution is not valid
        return False

    def solve(self, input_board):
        board = [list(row) for row in input_board]
        if self.solve_sudoku_f(board):
            return board, True
        else:
            return "No solution exists.", False



def run_method(obj_instance, method_name, parameter):
    method_to_run = getattr(obj_instance, method_name)
    method_to_run(parameter)

def generate_random_matrix():
    matrix = []
    for _ in range(9):
        row = []
        for _ in range(9):
            row.append(str(""))
        matrix.append(row)
    return matrix

def is_valid_sudoku(board):
    def is_valid_row(row):
        seen = set()
        for num in row:
            if num != 0:
                if num in seen:
                    return False
                seen.add(num)
        return True

    def is_valid_col(col):
        seen = set()
        for num in col:
            if num != 0:
                if num in seen:
                    return False
                seen.add(num)
        return True

    def is_valid_box(box):
        print(box)
        seen = set()
        for row in box:
            for num in row:
                if num != 0:
                    if num in seen:
                        return False
                    seen.add(num)
        return True

    for i in range(9):
        if not is_valid_row(board[i]):
            return False

    for j in range(9):
        if not is_valid_col([board[i][j] for i in range(9)]):
            return False

    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            if not is_valid_col([board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]):
                return False

    return True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    matrix = [["" for _ in range(9)] for _ in range(9)]  # Initialize a 9x9 matrix with all zeros
    worker = Worker()
    main_window = MainWindow(matrix,worker)
    worker.register_ui(main_window)
    worker.update_signal.connect(main_window.update_values)
    worker.start()
    sys.exit(app.exec_())
