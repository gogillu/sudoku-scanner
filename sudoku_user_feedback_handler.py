#!/usr/bin/env python3
"""
Sudoku User Feedback Handler

This module handles the real-time camera feed processing for Sudoku detection
and grid extraction. It provides the main interface between the camera input
and the image processing pipeline.

The handler manages:
- Real-time camera feed capture
- Sudoku grid detection and extraction
- Individual cell segmentation and processing
- Visual feedback and combined frame display
- Grid cell saving for digit recognition

Author: Sudoku Scanner Team
"""

import cv2
import os
from camera_feed import CameraFeed
from contour_processor import ContourProcessor
from crop_big_grid import CropBigGrid
import numpy as np
from detect_small_grid import DetectSmallGrid
from image_grid_processor import ImageGridProcessor
from datetime import datetime


class SudokuUserFeedbackHandler:
    """
    Handles real-time camera processing and user feedback for Sudoku detection.
    
    This class orchestrates the entire pipeline from camera input to processed
    Sudoku grid extraction. It manages multiple image processing components
    and provides real-time visual feedback to the user.
    
    Attributes:
        camera_feed (CameraFeed): Manages camera input and frame capture
        crop_big_grid (CropBigGrid): Handles grid detection and cropping
        detect_small_grid (DetectSmallGrid): Processes individual cells
        image_grid_processor (ImageGridProcessor): Manages grid segmentation
    """
    
    def __init__(self):
        """
        Initialize the SudokuUserFeedbackHandler.
        
        Sets up all the required image processing components:
        - Camera feed handler for video capture
        - Grid cropping processor for Sudoku detection
        - Small grid detector for individual cells
        - Image grid processor for cell extraction and saving
        """
        self.camera_feed = CameraFeed()
        self.crop_big_grid = CropBigGrid()
        self.detect_small_grid = DetectSmallGrid()
        self.image_grid_processor = ImageGridProcessor()

    def run(self):
        """
        Start the main real-time processing loop.
        
        Continuously captures frames from the camera, processes them to detect
        Sudoku grids, extracts individual cells, and displays the results.
        The loop continues until the user presses 'q' to quit.
        
        Processing steps for each frame:
        1. Capture frame from camera
        2. Detect and crop Sudoku grid
        3. Extract individual cells
        4. Process and save cell images
        5. Display combined visualization
        
        Controls:
            'q': Quit the application
        """
        print("Starting real-time Sudoku processing...")
        print("Controls: Press 'q' to quit")
        
        while True:
            # Capture frame from camera
            frame = self.camera_feed.read_frame()
            if frame is None:
                print("Error: Could not read frame from camera")
                break

            # Process the frame to detect and crop Sudoku grid
            full_frame, cropped_grid = self.crop_big_grid.get_combined_video_capture_and_cropped_full_grid(frame)

            # Combine the original frame (left) and the processed frame (right)
            combined_frame = self.crop_big_grid.adjust_and_concatenate_images(full_frame, cropped_grid)

            # Detect individual cells within the cropped grid
            contour_img = self.detect_small_grid.get_separate_small_grids(cropped_grid)

            # Process the grid for cell extraction
            cropped_grid_copy = cropped_grid.copy()
            combined_frame_81 = self.image_grid_processor.divide_and_combine_frame(cropped_grid_copy)
            combined_frame_81_BnW_smaller = self.image_grid_processor.divide_and_combine_small_cropped_frame_black_n_white(cropped_grid_copy)

            # Create combined visualization
            row2 = self.crop_big_grid.adjust_and_concatenate_images_color_with_gray(
                self.crop_big_grid.adjust_and_concatenate_images(contour_img, combined_frame_81),
                combined_frame_81_BnW_smaller
            )
            
            # Save individual grid cells for digit recognition
            self.image_grid_processor.save_grid_parts(cropped_grid_copy)

            # Combine all visualizations vertically
            combine_feed_grid_small_contour = self.crop_big_grid.adjust_and_concatenate_images_vertically(combined_frame, row2)
            cv2.imshow('Sudoku Scanner - Real-time Processing', combine_feed_grid_small_contour)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Cleanup
        self.camera_feed.release()
        cv2.destroyAllWindows()
        print("Real-time processing stopped.")

    def get_ongoing_frame(self):
        """
        Get a single processed frame without starting the continuous loop.
        
        This method performs the same processing as run() but for a single frame,
        useful for integration with other components or testing.
        
        Returns:
            tuple: A tuple containing:
                - full_frame: Original camera frame
                - cropped_grid: Detected and cropped Sudoku grid
                - combined_frame_81: Combined grid visualization
                - combined_frame_81_BnW_smaller: Black and white cell visualization
                
        Returns None for each element if frame capture fails.
        """
        # Capture frame from camera
        frame = self.camera_feed.read_frame()
        if frame is None:
            return None, None, None, None

        # Process the frame
        full_frame, cropped_grid = self.crop_big_grid.get_combined_video_capture_and_cropped_full_grid(frame)
        combined_frame = self.crop_big_grid.adjust_and_concatenate_images(full_frame, cropped_grid)

        # Process individual cells (reduced processing for single frame)
        cropped_grid_copy = cropped_grid.copy()
        combined_frame_81 = self.image_grid_processor.divide_and_combine_frame(cropped_grid_copy)
        combined_frame_81_BnW_smaller = self.image_grid_processor.divide_and_combine_small_cropped_frame_black_n_white(cropped_grid_copy)
        
        # Save grid parts for digit recognition
        self.image_grid_processor.save_grid_parts(cropped_grid_copy)

        return full_frame, cropped_grid, combined_frame_81, combined_frame_81_BnW_smaller
            
    def __del__(self):
        """
        Cleanup method called when the object is destroyed.
        
        Ensures proper cleanup of camera resources and OpenCV windows.
        """
        try:
            self.camera_feed.release()
            cv2.destroyAllWindows()
        except:
            pass  # Ignore cleanup errors

    def resize_half(self, frame):
        """
        Resize a frame to half its original dimensions and save it.
        
        This utility method is used for creating smaller versions of frames,
        useful for debugging or creating thumbnails.
        
        Args:
            frame (numpy.ndarray): Input frame to resize
            
        Returns:
            numpy.ndarray: Resized frame with half dimensions
            
        Note:
            The resized frame is automatically saved to the tmp/ directory
            with a timestamp filename.
        """
        height, width = frame.shape[:2]

        # Calculate new dimensions (half size)
        new_width = int(width / 2)
        new_height = int(height / 2)
        
        # Resize the frame
        newFrame = cv2.resize(frame, (new_width, new_height))
        
        # Save the resized frame with timestamp
        os.makedirs("tmp", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        cv2.imwrite(f"tmp/{timestamp}.png", newFrame)
        
        return newFrame
