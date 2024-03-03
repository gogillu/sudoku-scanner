import sys
import random
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from tenserflow_machine_digit_predict_model import *

class Worker(QThread):
    update_signal = pyqtSignal(object)  # Signal to update the UI

    def run(self):
        # This is the code that will run in a separate thread
        for i in range(1):  # Let's say we want to update 10 times
            time.sleep(0.05)  # Sleep for 1 second
            new_matrix = generate_random_matrix()
            self.update_signal.emit(new_matrix)  # Emit the signal with the new matrix

class MainWindow(QWidget):
    def __init__(self, matrix, worker):
        super().__init__()
        self.matrix = matrix
        self.worker = worker
        self.model, self.class_names = loadModel()
        self.initUI()

    def initUI(self):
        layout = QGridLayout()
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.edit_boxes = []

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

        button1 = QPushButton('Load')
        button1.clicked.connect(self.load)
        button_layout.addWidget(button1)

        button2 = QPushButton('Solve')
        button2.clicked.connect(self.solve)
        button_layout.addWidget(button2)

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
                    else:
                        img_matrix[i][j] = ""
        
        self.worker.update_signal.emit(img_matrix)


    def solve(self):
        new_matrix = generate_random_matrix()
        self.worker.update_signal.emit(new_matrix)
        print("Option 2 clicked")

def generate_random_matrix():
    matrix = []
    for _ in range(9):
        row = []
        for _ in range(9):
            row.append(str(random.randint(1, 9)))
        matrix.append(row)
    return matrix

if __name__ == '__main__':
    app = QApplication(sys.argv)
    matrix = [["" for _ in range(9)] for _ in range(9)]  # Initialize a 9x9 matrix with all zeros
    worker = Worker()
    main_window = MainWindow(matrix,worker)
    worker.update_signal.connect(main_window.update_values)
    worker.start()
    sys.exit(app.exec_())
