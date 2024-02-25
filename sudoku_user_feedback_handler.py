import cv2
import os
from camera_feed import CameraFeed
from contour_processor import ContourProcessor
from crop_big_grid import CropBigGrid
import numpy as np
from detect_small_grid import DetectSmallGrid
from image_grid_processor import ImageGridProcessor

class SudokuUserFeedbackHandler:
    def __init__(self):
        self.camera_feed = CameraFeed()
        self.crop_big_grid = CropBigGrid()
        self.detect_small_grid = DetectSmallGrid()
        self.image_grid_processor = ImageGridProcessor()

    def run(self):
        while True:
            frame = self.camera_feed.read_frame()
            if frame is None:
                break

            full_frame, cropped_grid, combined_frame = self.crop_big_grid.get_combined_video_capture_and_cropped_full_grid(frame)

            # Show the combined frame
            # cv2.imshow("Real-time Feed vs. Filtered Feed", combined_frame)
            # cv2.imshow("Small grid", contour_img)

            contour_img = self.detect_small_grid.get_separate_small_grids(cropped_grid)

            cropped_grid_copy = cropped_grid.copy()
            combined_frame_81 = self.image_grid_processor.divide_and_combine_frame(cropped_grid_copy)
            # cv2.imshow('combined grid', combined_frame_81)

            row2 = self.crop_big_grid.adjust_and_concatenate_images(contour_img, combined_frame_81)
            self.image_grid_processor.save_grid_parts(combined_frame_81)

            combine_feed_grid_small_contour = self.crop_big_grid.adjust_and_concatenate_images_vertically(combined_frame,row2)
            cv2.imshow('small contours', combine_feed_grid_small_contour)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
            
        self.camera_feed.release()
        cv2.destroyAllWindows()
