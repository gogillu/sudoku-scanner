import cv2
import numpy as np
import os

class ImageGridProcessor:
    def __init__(self, grid_size=9, padding=10):
        self.grid_size = grid_size  # Number of divisions per row and column
        self.padding = padding  # Padding between images in pixels

    def divide_and_combine_frame(self, frame):
        # Calculate the size of each grid cell
        height, width = frame.shape[:2]
        cell_height = height // self.grid_size
        cell_width = width // self.grid_size

        # Initialize a list to hold the grid parts
        grid_parts = []

        # Divide the frame into grid parts
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                start_y = row * cell_height
                end_y = start_y + cell_height
                start_x = col * cell_width
                end_x = start_x + cell_width
                grid_part = frame[start_y:end_y, start_x:end_x]
                grid_parts.append(grid_part)

        # Combine the grid parts with padding
        combined_image = self._combine_with_padding(grid_parts)

        return combined_image

    def _combine_with_padding(self, grid_parts):
        # Create rows with padding
        rows = []
        for i in range(0, len(grid_parts), self.grid_size):
            row = np.hstack([np.pad(img, ((0, 0), (0, self.padding), (0, 0)), 'constant') for img in grid_parts[i:i+self.grid_size]])
            rows.append(row)

        # Combine rows with vertical padding
        combined_image = np.vstack([np.pad(row, ((0, self.padding), (0, 0), (0, 0)), 'constant') for row in rows])

        # Remove padding from the right and bottom edges
        combined_image = combined_image[:-self.padding, :-self.padding]

        return combined_image
    
    def save_grid_parts(self, frame, output_dir="individual_grids"):
        """
        Saves each part of the 9x9 grid of the given frame as individual images.

        Args:
        - frame: The input image frame.
        - output_dir: Directory to save the grid images.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        height, width = frame.shape[:2]
        cell_height = height // self.grid_size
        cell_width = width // self.grid_size

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                start_y = row * cell_height
                end_y = start_y + cell_height
                start_x = col * cell_width
                end_x = start_x + cell_width
                grid_part = frame[start_y:end_y, start_x:end_x]

                # Convert to grayscale
                gray_part = cv2.cvtColor(grid_part, cv2.COLOR_BGR2GRAY)
                # Apply binary thresholding to make it purely black and white
                _, bw_part = cv2.threshold(gray_part, 127, 255, cv2.THRESH_BINARY)

                # Construct filename for the grid part
                filename = f"grid_{row * self.grid_size + col + 1}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Save the grid part
                cv2.imwrite(filepath, bw_part)
