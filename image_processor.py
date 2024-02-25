import cv2
import os
import numpy as np

class ImageProcessor:
    def __init__(self, save_path='processed/initial_contours'):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def process_image(self, frame, frame_count):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find edges for contour detection
        edged = cv2.Canny(gray, 30, 200)
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert grayscale image back to BGR for drawing contours in color
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw contours in green on the grayscale image
        cv2.drawContours(gray_bgr, contours, -1, (0, 255, 0), 2)  # Use a thickness of 2 for better visibility

        # Save the grayscale image with green contours
        save_path = os.path.join(self.save_path, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(save_path, gray_bgr)

        return gray_bgr
