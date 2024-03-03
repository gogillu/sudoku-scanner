import sys
import random
import time
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QCoreApplication
import cv2
from PyQt5.QtGui import QPixmap, QImage, QGuiApplication
from sudoku_user_feedback_handler import SudokuUserFeedbackHandler

class FrameFetcher(QThread):
    update_signal = pyqtSignal(list)  # Signal to update the UI with four images
    sfh = SudokuUserFeedbackHandler()

    def run(self):
        while True:
            frames = self.get_ongoing_frames()  # Assuming get_ongoing_frames() returns four images as a list
            # time.sleep(0.1)  # Adjust the sleep time as per your requirement

            self.update_signal.emit(frames)

    def get_ongoing_frames(self):
        f1,f2,f3,f4 = self.sfh.get_ongoing_frame()
        return [f1,f2,f3,f4]

class MainWindow(QWidget):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Qt grid UI
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(0)
        grid_layout.setVerticalSpacing(0)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        self.edit_boxes = []

        for i in range(9):
            row = []
            for j in range(9):
                edit_box = QLineEdit(str(self.matrix[i][j]))
                edit_box.setAlignment(Qt.AlignCenter)
                edit_box.setStyleSheet("background-color: white; color: black; font-size: 24px;")
                edit_box.setFixedSize(40, 40)
                grid_layout.addWidget(edit_box, i, j)
                row.append(edit_box)
            self.edit_boxes.append(row)

        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)

        layout.addWidget(grid_widget)

        # Image display
        image_layout = QHBoxLayout()

        self.video_labels = [QLabel() for _ in range(4)]  # Create labels for four images
        for label in self.video_labels:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid black;")
            image_layout.addWidget(label)

        layout.addLayout(image_layout)

        self.setLayout(layout)
        self.setWindowTitle('Video and 9x9 Edit Boxes')
        self.resize_to_screen()  # Resize window to fit screen
        self.show()

    def resize_to_screen(self):
        # Resize window to fit screen size
        screen = QGuiApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        self.resize(screen_rect.size())

    def update_values(self, matrix):
        for i in range(9):
            for j in range(9):
                self.edit_boxes[i][j].setText(str(matrix[i][j]))
                if (i // 3 + j // 3) % 2 == 0:
                    self.edit_boxes[i][j].setStyleSheet("background-color: lightgray; color: black; font-size: 24px;")
                else:
                    self.edit_boxes[i][j].setStyleSheet("background-color: gray; color: black; font-size: 24px;")

    def update_video_frames(self, frames):
        for i, frame in enumerate(frames):
            frame = self.resize_frame(frame)  # Resize frame to fit screen
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_labels[i].setPixmap(pixmap)

    def resize_frame(self, frame):
        # Resize frame to fit screen
        screen = QGuiApplication.primaryScreen()
        screen_rect = screen.availableGeometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        if frame_width > screen_width or frame_height > screen_height:
            frame_ratio = frame_width / frame_height
            screen_ratio = screen_width / screen_height

            if frame_ratio > screen_ratio:
                new_width = screen_width
                new_height = int(new_width / frame_ratio)
            else:
                new_height = screen_height
                new_width = int(new_height * frame_ratio)

            frame = cv2.resize(frame, (new_width, new_height))

        return frame

def generate_random_matrix():
    matrix = []
    for _ in range(9):
        row = []
        for _ in range(9):
            row.append(random.randint(1, 9))
        matrix.append(row)
    return matrix

def get_ongoing_frames():
    # Replace this function with your implementation to get four ongoing frames
    frames = [cv2.imread(f"frame{i}.jpg") for i in range(1, 5)]  # Load sample images
    return frames

if __name__ == '__main__':
    app = QApplication(sys.argv)
    matrix = [[0 for _ in range(9)] for _ in range(9)]  # Initialize a 9x9 matrix with all zeros
    main_window = MainWindow(matrix)
    frame_fetcher = FrameFetcher()
    frame_fetcher.update_signal.connect(main_window.update_video_frames)
    frame_fetcher.start()
    sys.exit(app.exec_())
