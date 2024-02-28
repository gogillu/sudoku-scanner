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

    def divide_and_combine_small_cropped_frame_black_n_white(self, frame):
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
                BnW_part = self.crop_BnW_image_with_10_percent(grid_part)
                grid_parts.append(BnW_part)

        # Combine the grid parts with padding
        combined_image = self._combine_without_padding(grid_parts)
        # combined_image = self.combine_images(grid_parts)

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

    def _combine_without_padding(self, grid_parts):
        """
        Combine single-channel images (e.g., grayscale or binary thresholded images) 
        into a single image without padding between the images.

        Parameters:
        - grid_parts: List of single-channel (grayscale or binary) images to be combined.
        - grid_size: The number of images per row in the final combined image.
        - padding: The number of pixels to pad between images.

        Returns:
        - A single combined image.
        """
        # Create rows with padding
        rows = []
        for i in range(0, len(grid_parts), self.grid_size):
            # Adjust padding based on the image being single-channel
            row = np.hstack([np.pad(img, ((0, 0), (0, self.padding)), 'constant') for img in grid_parts[i:i+self.grid_size]])
            rows.append(row)

        # Combine rows with vertical padding, adjusting for single-channel images
        combined_image = np.vstack([np.pad(row, ((0, self.padding), (0, 0)), 'constant') for row in rows])

        # Remove padding from the right and bottom edges
        combined_image = combined_image[:-self.padding, :-self.padding]

        return combined_image 
   
    def combine_images(self, matrix):
        """
        Combine images (cv2 frames) from a matrix (list of lists) into a single image.
        Each inner list represents a row, and each element is an image (cv2 frame).
        
        Parameters:
        - matrix: List of lists of cv2 frames.
        
        Returns:
        - Combined image as a single cv2 frame.
        """
        # First, we need to normalize the sizes within each row and column
        max_heights = []
        max_widths = [0] * max(len(row) for row in matrix)  # Initialize with zeros

        # Calculate max heights for each row and max widths for each column
        for row in matrix:
            max_height = max(image.shape[0] for image in row)
            max_heights.append(max_height)
            
            for i, image in enumerate(row):
                max_widths[i] = max(max_widths[i], image.shape[1])

        # Resize images in the matrix to fit the max dimensions
        for rowIndex, row in enumerate(matrix):
            for colIndex, image in enumerate(row):
                desired_size = (max_widths[colIndex], max_heights[rowIndex])
                matrix[rowIndex][colIndex] = cv2.resize(image, desired_size, interpolation=cv2.INTER_CUBIC)
        
        # Concatenate images in each row
        rows = [cv2.hconcat(row) for row in matrix]
        
        # Concatenate rows to get the final image
        final_image = cv2.vconcat(rows)
        
        return final_image
    
    def crop_BnW_image_with_10_percent(self, frame):
        # Read the image using OpenCV
        img = frame #cv2.imread(image_path)

        # # Get dimensions of the original image
        # height, width = img.shape[:2]

        # # Calculate the cropping dimensions
        # left = 10
        # top = 10
        # right = int(width - 10)
        # bottom = int(height - 10)

        # # Crop the image using array slicing
        # cropped_img = img[top:bottom, left:right]

        # Convert to grayscale
        gray_part = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding to make it purely black and white
        _, bw_part = cv2.threshold(gray_part, 127, 255, cv2.THRESH_BINARY)

        # Return the cropped image
        return bw_part


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
