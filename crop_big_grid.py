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
    
    def adjust_and_concatenate_images_color_with_gray(self, img1, img2):
        """
        Adjusts img1 (colorful) and img2 (black and white) to have the same height by padding the smaller image with a black background.
        Converts img2 to a 3-channel BGR image if it is grayscale.
        Then concatenates the images side by side.

        Args:
        - img1: First image (color, numpy array).
        - img2: Second image (grayscale, numpy array).

        Returns:
        - Concatenated image.
        """
        # Convert img2 to BGR if it is grayscale
        if len(img2.shape) == 2:  # img2 is grayscale
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Get the height and width of both images
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # Determine the maximum height
        max_height = max(height1, height2)

        # If img1 is shorter, pad it
        if height1 < max_height:
            diff = max_height - height1
            pad = np.zeros((diff, width1, 3), dtype=np.uint8)
            img1_padded = np.vstack((img1, pad))
        else:
            img1_padded = img1

        # If img2 is shorter, pad it
        if height2 < max_height:
            diff = max_height - height2
            pad = np.zeros((diff, width2, 3), dtype=np.uint8)
            img2_padded = np.vstack((img2, pad))
        else:
            img2_padded = img2

        # Concatenate the adjusted images side by side
        combined_image = np.hstack((img1_padded, img2_padded))

        return combined_image

    def adjust_and_concatenate_images_vertically(self, img1, img2):
        """
        Adjusts img1 and img2 to have the same width by padding the smaller image with a black background.
        Then concatenates the images vertically (one on top of the other).

        Args:
        - img1: First image (numpy array).
        - img2: Second image (numpy array).

        Returns:
        - Concatenated image.
        """
        # Get the height and width of both images
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # Determine the maximum width
        max_width = max(width1, width2)

        # If img1 is narrower, pad it
        if width1 < max_width:
            # Calculate the padding size
            diff = max_width - width1
            # Create padding
            pad = np.zeros((height1, diff, 3), dtype=np.uint8)
            # Stack the padding on the side of img1
            img1_padded = np.hstack((img1, pad))
        else:
            img1_padded = img1

        # If img2 is narrower, pad it
        if width2 < max_width:
            # Calculate the padding size
            diff = max_width - width2
            # Create padding
            pad = np.zeros((height2, diff, 3), dtype=np.uint8)
            # Stack the padding on the side of img2
            img2_padded = np.hstack((img2, pad))
        else:
            img2_padded = img2

        # Concatenate the adjusted images vertically
        combined_frame = np.vstack((img1_padded, img2_padded))

        return combined_frame

    def get_combined_video_capture_and_cropped_full_grid(self, frame):
        full_frame, cropped_grid = self.contour_processor.process_image(frame)

        return full_frame, cropped_grid

        # # Combine the original frame (left) and the processed/last valid frame (right)
        # combined_frame = self.adjust_and_concatenate_images(full_frame, cropped_grid)

        # return full_frame, cropped_grid, combined_frame