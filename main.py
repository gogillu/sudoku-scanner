# Add imports at the top
from camera_handler import CameraHandler
from big_contour_processor import BigContourProcessor
from sudoku_user_feedback_handler import SudokuUserFeedbackHandler

class MainController:
    def __init__(self):
        # self.camera_handler = CameraHandler()
        # self.big_contour_processor = BigContourProcessor("./input")
        self.sudoku_user_feedback_handler = SudokuUserFeedbackHandler()

    def run(self):
        # self.camera_handler.run()
        # self.big_contour_processor.run()
        self.sudoku_user_feedback_handler.run()

# Main execution
if __name__ == "__main__":
    controller = MainController()
    controller.run()
