from contour_processor import ContourProcessor

class DetectSmallGrid:
    def __init__(self):
        self.hello = "hello"
        self.contour_processor = ContourProcessor()

    def get_separate_small_grids(self, frame):
        return self.contour_processor.show_small_contours(frame, 2500)