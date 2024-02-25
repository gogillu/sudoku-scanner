import cv2
import os

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
        # Draw contours on the original frame
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        
        # Save the processed image with contours
        cv2.imwrite(os.path.join(self.save_path, f'frame_{frame_count:04d}.jpg'), frame)

        return frame
