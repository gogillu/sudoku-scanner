#!/usr/bin/env python3
"""
Sudoku Scanner - Main Entry Point

This module serves as the main entry point for the Sudoku Scanner application.
It initializes the camera-based real-time processing system that detects
Sudoku grids from camera feed and processes them for digit recognition.

Usage:
    python3 main.py

The application will start the camera feed and begin processing Sudoku grids
in real-time. Press 'q' to quit the application.

Author: Sudoku Scanner Team
"""

from sudoku_user_feedback_handler import SudokuUserFeedbackHandler


class MainController:
    """
    Main controller class that orchestrates the Sudoku Scanner application.
    
    This class manages the initialization and execution of the real-time
    Sudoku processing system, including camera feed handling, grid detection,
    and digit recognition.
    
    Attributes:
        sudoku_user_feedback_handler (SudokuUserFeedbackHandler): Handles the
            real-time camera feed processing and user interactions.
    """
    
    def __init__(self):
        """
        Initialize the MainController.
        
        Sets up the sudoku user feedback handler for real-time processing.
        Camera handler and big contour processor are currently commented out
        but can be enabled for additional functionality.
        """
        # Alternative processing options (currently disabled)
        # self.camera_handler = CameraHandler()
        # self.big_contour_processor = BigContourProcessor("./input")
        
        # Main processing handler for real-time camera feed
        self.sudoku_user_feedback_handler = SudokuUserFeedbackHandler()

    def run(self):
        """
        Start the main application loop.
        
        Begins the real-time camera processing for Sudoku detection and
        grid extraction. The application will continue running until the
        user presses 'q' to quit.
        
        The processing includes:
        - Camera feed capture
        - Sudoku grid detection and extraction
        - Individual cell segmentation
        - Real-time visualization of processing steps
        """
        print("Starting Sudoku Scanner...")
        print("Position a Sudoku puzzle in front of your camera.")
        print("Press 'q' to quit the application.")
        
        # Start the main processing loop
        self.sudoku_user_feedback_handler.run()
        
        print("Sudoku Scanner stopped.")


# Main execution
if __name__ == "__main__":
    """
    Main entry point of the application.
    
    Creates a MainController instance and starts the application.
    This will begin the real-time camera processing for Sudoku detection.
    """
    try:
        controller = MainController()
        controller.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your camera connection and try again.")