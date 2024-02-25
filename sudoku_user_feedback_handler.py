import cv2
import os
from camera_feed import CameraFeed
from contour_processor import ContourProcessor
from crop_big_grid import CropBigGrid
import numpy as np

class SudokuUserFeedbackHandler:
    def __init__(self):
        self.camera_feed = CameraFeed()
        self.crop_big_grid = CropBigGrid()

    def run(self):
        while True:
            frame = self.camera_feed.read_frame()
            if frame is None:
                break

            full_frame, cropped_grid, combined_frame = self.crop_big_grid.get_combined_video_capture_and_cropped_full_grid(frame)

            # Show the combined frame
            cv2.imshow("Real-time Feed vs. Filtered Feed", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
            
        self.camera_feed.release()
        cv2.destroyAllWindows()
