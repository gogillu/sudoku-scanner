import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def process_image(frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find edges for contour detection
        edged = cv2.Canny(gray, 30, 200)
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the original frame
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        return frame
