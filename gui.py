import sys
import random
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class Worker(QThread):
    update_signal = pyqtSignal(object)  # Signal to update the UI

    def run(self):
        # This is the code that will run in a separate thread
        for i in range(10000000000):  # Let's say we want to update 10 times
            time.sleep(0.2)  # Sleep for 1 second
            new_matrix = generate_random_matrix()
            self.update_signal.emit(new_matrix)  # Emit the signal with the new matrix

class MainWindow(QWidget):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix
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

        self.setLayout(layout)
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

def generate_random_matrix():
    matrix = []
    for _ in range(9):
        row = []
        for _ in range(9):
            row.append(random.randint(1, 9))
        matrix.append(row)
    return matrix

if __name__ == '__main__':
    app = QApplication(sys.argv)
    matrix = [[0 for _ in range(9)] for _ in range(9)]  # Initialize a 9x9 matrix with all zeros
    main_window = MainWindow(matrix)
    worker = Worker()
    worker.update_signal.connect(main_window.update_values)
    worker.start()
    sys.exit(app.exec_())
