import cv2
import os
from camera_feed import CameraFeed
from contour_processor import ContourProcessor
import numpy as np

class CropBigGrid:
    def __init__(self):
        self.camera_feed = CameraFeed()
        self.capture = False
        self.frame_count = 0
        self.frames = []
        self.contour_processor = ContourProcessor()

    def adjust_and_concatenate_images(self, img1, img2):
        """
        Adjusts img1 and img2 to have the same height by padding the smaller image with a black background.
        Then concatenates the images side by side.

        Args:
        - img1: First image (numpy array).
        - img2: Second image (numpy array).

        Returns:
        - Concatenated image.
        """
        # Get the height and width of both images
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # Determine the maximum height
        max_height = max(height1, height2)

        # If img1 is shorter, pad it
        if height1 < max_height:
            # Calculate the padding size
            diff = max_height - height1
            # Create padding
            pad = np.zeros((diff, width1, 3), dtype=np.uint8)
            # Stack the padding on img1
            img1_padded = np.vstack((img1, pad))
        else:
            img1_padded = img1

        # If img2 is shorter, pad it
        if height2 < max_height:
            # Calculate the padding size
            diff = max_height - height2
            # Create padding
            pad = np.zeros((diff, width2, 3), dtype=np.uint8)
            # Stack the padding on img2
            img2_padded = np.vstack((img2, pad))
        else:
            img2_padded = img2

        # Concatenate the adjusted images side by side
        combined_frame = np.hstack((img1_padded, img2_padded))

        return combined_frame

    def get_combined_video_capture_and_cropped_full_grid(self, frame):
        full_frame, cropped_grid = self.contour_processor.process_image(frame)

        # Combine the original frame (left) and the processed/last valid frame (right)
        combined_frame = self.adjust_and_concatenate_images(full_frame, cropped_grid)

        return full_frame, cropped_grid, combined_frame