#!/usr/bin/env python3
"""
Sudoku Scanner GUI

This module provides a PyQt5-based graphical user interface for the Sudoku Scanner
application. It allows users to manually input Sudoku puzzles, load puzzles from
camera-processed images, and solve them with optional visualization.

Features:
- 9x9 interactive grid for manual Sudoku input
- Load puzzles from camera-processed individual grid images
- Two solving modes: instant solving and step-by-step visualization
- Color-coded display to distinguish original vs. solved digits
- Real-time backtracking algorithm visualization

Usage:
    python3 gui.py

Author: Sudoku Scanner Team
"""

import sys
import random
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from tenserflow_machine_digit_predict_model import *
import threading
import numpy as np


class Worker(QThread):
    """
    Worker thread for handling Sudoku solving operations.
    
    This class runs the computationally intensive Sudoku solving algorithms
    in a separate thread to prevent the GUI from freezing. It uses Qt signals
    to communicate with the main UI thread for real-time updates.
    
    Signals:
        update_signal: Emitted when the grid should be updated with new values
    """
    
    update_signal = pyqtSignal(object)  # Signal to update the UI

    def run(self):
        """
        Default run method for the worker thread.
        
        This method is called when the thread starts. Currently used for
        testing purposes with a random matrix generation.
        """
        for i in range(1):
            time.sleep(0.05)
            new_matrix = generate_random_matrix()
            self.update_signal.emit(new_matrix)

    def register_ui(self, main_window):
        """
        Register the main window UI for interaction.
        
        Args:
            main_window (MainWindow): The main GUI window instance
        """
        self.main_window = main_window

    def is_valid_move(self, row, col, num, matrix):
        """
        Check if placing a number at a specific position is valid.
        
        Validates the move according to Sudoku rules:
        - Number must not exist in the same row
        - Number must not exist in the same column
        - Number must not exist in the same 3x3 subgrid
        
        Args:
            row (int): Row index (0-8)
            col (int): Column index (0-8)
            num (int): Number to place (1-9)
            matrix (list): 9x9 Sudoku grid
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        # Check row
        for j in range(9):
            if matrix[row][j] == num:
                return False

        # Check column
        for i in range(9):
            if matrix[i][col] == num:
                return False

        # Check 3x3 subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if matrix[i][j] == num:
                    return False

        return True

    def solve_sudoku(self, matrix, responsive=True, ui_resp=0.002):
        """
        Solve Sudoku using backtracking algorithm with optional visualization.
        
        This method implements a recursive backtracking algorithm to solve
        the Sudoku puzzle. It can optionally provide real-time visualization
        of the solving process by emitting update signals.
        
        Args:
            matrix (list): 9x9 Sudoku grid to solve
            responsive (bool): Whether to emit real-time updates
            ui_resp (float): Delay between updates for visualization
            
        Returns:
            bool: True if puzzle is solved, False if no solution exists
        """
        for i in range(9):
            for j in range(9):
                if matrix[i][j] == 0:
                    for num in range(1, 10):
                        if self.is_valid_move(i, j, num, matrix):
                            matrix[i][j] = num
                            if responsive:
                                time.sleep(ui_resp)
                            self.update_signal.emit(self.transform_1_9_or_empty_sudoku(matrix))
                            
                            if self.solve_sudoku(matrix):
                                return True
                            matrix[i][j] = 0  # Backtrack
                    return False
        return True
    
    def transform_1_9_or_empty_sudoku(self, matrix):
        """
        Transform numerical matrix to string matrix for GUI display.
        
        Converts the internal numerical representation (0 for empty cells)
        to string representation ("" for empty cells) used by the GUI.
        
        Args:
            matrix (list): 9x9 numerical matrix
            
        Returns:
            list: 9x9 string matrix suitable for GUI display
        """
        nM = []
        for i in range(9):
            row = []
            for j in range(9):
                if matrix[i][j] > 0:
                    row.append(str(matrix[i][j]))
                else:
                    row.append("")
            nM.append(row)
        return nM


class MainWindow(QWidget):
    """
    Main GUI window for the Sudoku Scanner application.
    
    Provides an interactive 9x9 grid for Sudoku input and solving, along with
    control buttons for different operations. The interface supports both
    manual input and loading puzzles from camera-processed images.
    
    Attributes:
        matrix (list): Current 9x9 Sudoku grid state
        prefilled_matrix (list): Boolean matrix indicating pre-filled cells
        worker (Worker): Worker thread for solving operations
        model: Trained machine learning model for digit recognition
        class_names: Class names for the ML model
        edit_boxes (list): 2D list of QLineEdit widgets for the grid
    """
    
    def __init__(self, matrix, worker):
        """
        Initialize the main window.
        
        Args:
            matrix (list): Initial 9x9 grid state
            worker (Worker): Worker thread for background operations
        """
        super().__init__()
        self.matrix = matrix
        self.prefilled_matrix = [[False] * 9 for _ in range(9)]
        self.worker = worker
        self.model, self.class_names = loadModel()
        self.edit_boxes = []
        self.initUI()

    def initUI(self):
        """
        Initialize the user interface.
        
        Creates the 9x9 grid of input boxes and control buttons,
        sets up the layout and styling for the application window.
        """
        # Create main grid layout
        layout = QGridLayout()
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create 9x9 grid of input boxes
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

        # Create control buttons
        button_layout = QVBoxLayout()

        button1 = QPushButton('Load sudoku from scanned image')
        button1.clicked.connect(self.load)
        button1.setToolTip('Load a Sudoku puzzle from camera-processed individual grid images')
        button_layout.addWidget(button1)

        button2 = QPushButton('Solve slowly')
        button2.clicked.connect(self.solve_old)
        button2.setToolTip('Solve the puzzle with step-by-step visualization')
        button_layout.addWidget(button2)

        button3 = QPushButton('Solve instantly')
        button3.clicked.connect(self.solve_instantly)
        button3.setToolTip('Solve the puzzle immediately without visualization')
        button_layout.addWidget(button3)

        # Combine layouts
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(button_layout)

        # Set window properties
        self.setLayout(main_layout)
        self.setWindowTitle('Sudoku Scanner - Interactive Solver')
        self.setGeometry(100, 100, 500, 500)
        self.show()

    def update_values(self, matrix):
        """
        Update the grid display with new values.
        
        Updates all grid cells with values from the provided matrix
        and applies alternating background colors for visual clarity.
        
        Args:
            matrix (list): 9x9 matrix of values to display
        """
        for i in range(9):
            for j in range(9):
                self.edit_boxes[i][j].setText(str(matrix[i][j]))
                if (i // 3 + j // 3) % 2 == 0:
                    self.edit_boxes[i][j].setStyleSheet("background-color: lightgray; color: black; font-size: 24px;")
                else:
                    self.edit_boxes[i][j].setStyleSheet("background-color: gray; color: black; font-size: 24px;")

    def update_values_prefilled(self, matrix):
        """
        Update grid display with distinction between prefilled and solved values.
        
        Updates the grid while maintaining visual distinction between
        originally filled cells and newly solved cells.
        
        Args:
            matrix (list): 9x9 matrix of values to display
        """
        for i in range(9):
            for j in range(9):
                self.edit_boxes[i][j].setText(str(matrix[i][j]))
                if self.prefilled_matrix[i][j]:
                    # Style for prefilled cells
                    if (i // 3 + j // 3) % 2 == 0:
                        self.edit_boxes[i][j].setStyleSheet("background-color: lightgray; color: black; font-size: 24px;")
                    else:
                        self.edit_boxes[i][j].setStyleSheet("background-color: gray; color: black; font-size: 24px;")
                else:
                    # Style for solved cells
                    if (i // 3 + j // 3) % 2 == 0:
                        self.edit_boxes[i][j].setStyleSheet("background-color: lightgray; color: blue; font-family: Lucida Console; font-size: 36px;")
                    else:
                        self.edit_boxes[i][j].setStyleSheet("background-color: gray; color: blue; font-family: Lucida Console; font-size: 36px;")

    def load(self):
        """
        Load Sudoku puzzle from camera-processed individual grid images.
        
        Reads digit images from the 'individual_grids' directory,
        uses the trained ML model to recognize digits, and populates
        the grid with the recognized values.
        
        The method processes 81 images (grid_1.jpg to grid_81.jpg)
        corresponding to the 9x9 Sudoku grid positions.
        """
        print("Loading Sudoku from scanned images...")
        img_matrix = [[""] * 9 for _ in range(9)]
        
        z = 0
        for i in range(9):
            for j in range(9):
                z += 1
                
                # Construct image path for current grid cell
                image_path = f'individual_grids/grid_{z}.jpg'
                
                # Use ML model to predict digit
                v = predict_with_teachable_ml_optimized(image_path, self.model, self.class_names)
                
                if v is not None and v > 0:
                    print(f"Position ({i},{j}): Detected digit {v}")
                    img_matrix[i][j] = str(v)
                    self.prefilled_matrix[i][j] = True
                    self.worker.update_signal.connect(main_window.update_values_prefilled)
                else:
                    img_matrix[i][j] = ""
                    
                self.matrix[i][j] = v if v is not None else 0

        # Update the display
        self.worker.update_signal.emit(img_matrix)
        print("Sudoku loaded successfully!")

    def solve_old(self):
        """
        Solve the Sudoku puzzle with step-by-step visualization.
        
        Reads the current grid state, validates it, and starts the
        solving process in a separate thread with real-time updates
        showing the backtracking algorithm in action.
        """
        print("Starting step-by-step solving...")
        
        # Read current grid state
        for i in range(9):
            for j in range(9):
                text = self.edit_boxes[i][j].text()
                if text.isdigit():
                    self.matrix[i][j] = int(text)
                else:
                    self.matrix[i][j] = 0

        print("Grid state:", self.matrix)

        # Validate Sudoku
        if is_valid_sudoku(self.matrix):
            print("Valid Sudoku - starting to solve...")
        else:
            print("Warning: Invalid Sudoku detected")

        # Start solving in separate thread
        thread = threading.Thread(target=run_method, args=(self.worker, "solve_sudoku", self.matrix))
        thread.start()
        print("Sudoku solving started with visualization...")

    def solve_instantly(self):
        """
        Solve the Sudoku puzzle instantly without visualization.
        
        Uses a fast solving algorithm to provide immediate results
        without the step-by-step visualization delay.
        """
        print("Solving instantly...")
        
        # Read current grid state
        for i in range(9):
            for j in range(9):
                text = self.edit_boxes[i][j].text()
                if text.isdigit():
                    self.matrix[i][j] = int(text)
                else:
                    self.matrix[i][j] = 0

        # Solve using fast algorithm
        sol, done = self.solve(self.matrix)
        
        if done:
            print("Puzzle solved successfully!")
            self.worker.update_signal.emit(sol)
        else:
            print("No solution found for this puzzle")

    def is_valid(self, board, row, col, num):
        """
        Check if placing a number is valid according to Sudoku rules.
        
        Args:
            board (list): Current board state
            row (int): Row position
            col (int): Column position
            num (int): Number to place
            
        Returns:
            bool: True if placement is valid
        """
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False
        
        # Check 3x3 subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num:
                    return False
        
        return True

    def find_empty_location(self, board):
        """
        Find the next empty cell in the board.
        
        Args:
            board (list): Current board state
            
        Returns:
            tuple: (row, col) of empty cell, or (-1, -1) if none found
        """
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return i, j
        return -1, -1

    def solve_sudoku_f(self, board):
        """
        Fast Sudoku solving algorithm using backtracking.
        
        Args:
            board (list): Board to solve
            
        Returns:
            bool: True if solved, False if no solution
        """
        row, col = self.find_empty_location(board)
        if row == -1 and col == -1:
            return True  # Puzzle solved
        
        for num in range(1, 10):
            if self.is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_sudoku_f(board):
                    return True
                board[row][col] = 0  # Backtrack
        return False

    def solve(self, input_board):
        """
        Solve the Sudoku puzzle and return the result.
        
        Args:
            input_board (list): Input puzzle to solve
            
        Returns:
            tuple: (solution_board, success_flag)
        """
        board = [list(row) for row in input_board]
        if self.solve_sudoku_f(board):
            return board, True
        else:
            return "No solution exists.", False


def run_method(obj_instance, method_name, parameter):
    """
    Utility function to run a method on an object instance.
    
    Args:
        obj_instance: Object to call method on
        method_name (str): Name of method to call
        parameter: Parameter to pass to the method
    """
    method_to_run = getattr(obj_instance, method_name)
    method_to_run(parameter)


def generate_random_matrix():
    """
    Generate an empty 9x9 matrix for testing purposes.
    
    Returns:
        list: 9x9 matrix filled with empty strings
    """
    matrix = []
    for _ in range(9):
        row = []
        for _ in range(9):
            row.append("")
        matrix.append(row)
    return matrix


def is_valid_sudoku(board):
    """
    Validate if a Sudoku board configuration is valid.
    
    Checks that no number appears twice in any row, column, or 3x3 subgrid.
    
    Args:
        board (list): 9x9 Sudoku board to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
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
        seen = set()
        for row in box:
            for num in row:
                if num != 0:
                    if num in seen:
                        return False
                    seen.add(num)
        return True

    # Check all rows
    for i in range(9):
        if not is_valid_row(board[i]):
            return False

    # Check all columns
    for j in range(9):
        if not is_valid_col([board[i][j] for i in range(9)]):
            return False

    # Check all 3x3 boxes
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            if not is_valid_col([board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]):
                return False

    return True


if __name__ == '__main__':
    """
    Main entry point for the GUI application.
    
    Creates the PyQt5 application, initializes the main window and worker thread,
    and starts the event loop.
    """
    print("Starting Sudoku Scanner GUI...")
    
    app = QApplication(sys.argv)
    
    # Initialize empty matrix and worker thread
    matrix = [["" for _ in range(9)] for _ in range(9)]
    worker = Worker()
    
    # Create main window and connect signals
    main_window = MainWindow(matrix, worker)
    worker.register_ui(main_window)
    worker.update_signal.connect(main_window.update_values)
    worker.start()
    
    print("GUI initialized. You can now:")
    print("1. Manually input a Sudoku puzzle")
    print("2. Load a puzzle from camera-processed images")
    print("3. Solve with or without visualization")
    
    # Start the application
    sys.exit(app.exec_())
