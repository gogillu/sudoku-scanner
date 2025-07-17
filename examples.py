#!/usr/bin/env python3
"""
Sudoku Scanner - Example Usage

This script demonstrates how to use the Sudoku Scanner components
programmatically for various tasks.

Usage:
    python3 examples.py
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def example_camera_processing():
    """
    Example: Basic camera processing and grid detection.
    
    This example shows how to use the camera processing components
    to capture and process Sudoku grids from camera feed.
    """
    print("Example 1: Camera Processing")
    print("-" * 30)
    
    try:
        from sudoku_user_feedback_handler import SudokuUserFeedbackHandler
        
        # Initialize the camera processing handler
        handler = SudokuUserFeedbackHandler()
        
        print("Camera processing handler initialized successfully")
        print("Note: This would normally start camera processing.")
        print("To actually run it, use: python3 main.py")
        
        # Example of getting a single frame (would require camera)
        # full_frame, cropped_grid, grid_81, grid_bw = handler.get_ongoing_frame()
        
        print("✓ Camera processing example completed")
        
    except ImportError as e:
        print(f"✗ Required modules not available: {e}")
        print("Install dependencies with: pip install -r requirements.txt")
    except Exception as e:
        print(f"✗ Error in camera processing: {e}")
    
    print()


def example_sudoku_solving():
    """
    Example: Programmatic Sudoku solving.
    
    This example shows how to solve a Sudoku puzzle programmatically
    without using the GUI.
    """
    print("Example 2: Sudoku Solving")
    print("-" * 25)
    
    # Example Sudoku puzzle (0 represents empty cells)
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    print("Original puzzle:")
    print_sudoku(puzzle)
    
    try:
        # Use the solving algorithm from the GUI module
        from gui import MainWindow, Worker
        
        # Create a dummy worker and main window for access to solving methods
        worker = Worker()
        main_window = MainWindow([[""] * 9 for _ in range(9)], worker)
        
        # Solve the puzzle
        solution, solved = main_window.solve(puzzle)
        
        if solved:
            print("\nSolved puzzle:")
            print_sudoku(solution)
            print("✓ Sudoku solved successfully!")
        else:
            print("✗ Could not solve the puzzle")
            
    except ImportError as e:
        print(f"✗ Required modules not available: {e}")
        print("Install dependencies with: pip install -r requirements.txt")
    except Exception as e:
        print(f"✗ Error in solving: {e}")
    
    print()


def example_digit_recognition():
    """
    Example: Digit recognition from images.
    
    This example shows how to use the digit recognition functionality
    to predict digits from individual cell images.
    """
    print("Example 3: Digit Recognition")
    print("-" * 28)
    
    try:
        from tenserflow_machine_digit_predict_model import loadModel, predict_with_teachable_ml_optimized
        
        # Load the trained model
        print("Loading digit recognition model...")
        model, class_names = loadModel()
        print("✓ Model loaded successfully")
        
        # Example of predicting from an image file
        # This would work if you have images in the individual_grids folder
        sample_image_path = "individual_grids/grid_1.jpg"
        
        if os.path.exists(sample_image_path):
            prediction = predict_with_teachable_ml_optimized(
                sample_image_path, model, class_names
            )
            print(f"Prediction for {sample_image_path}: {prediction}")
        else:
            print(f"Sample image {sample_image_path} not found")
            print("To test digit recognition:")
            print("1. Run camera processing: python3 main.py")
            print("2. Capture some grid images")
            print("3. Run this example again")
        
        print("✓ Digit recognition example completed")
        
    except ImportError as e:
        print(f"✗ Required modules not available: {e}")
        print("Install dependencies with: pip install -r requirements.txt")
    except Exception as e:
        print(f"✗ Error in digit recognition: {e}")
    
    print()


def example_validation():
    """
    Example: Sudoku validation.
    
    This example shows how to validate Sudoku puzzles.
    """
    print("Example 4: Sudoku Validation")
    print("-" * 28)
    
    try:
        from gui import is_valid_sudoku
        
        # Valid puzzle (partially filled)
        valid_puzzle = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        
        # Invalid puzzle (duplicate 5 in first row)
        invalid_puzzle = [
            [5, 3, 5, 0, 7, 0, 0, 0, 0],  # Two 5s in this row
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        
        print("Validating puzzles:")
        
        if is_valid_sudoku(valid_puzzle):
            print("✓ Valid puzzle: PASSED")
        else:
            print("✗ Valid puzzle: FAILED (unexpected)")
        
        if not is_valid_sudoku(invalid_puzzle):
            print("✓ Invalid puzzle: CORRECTLY REJECTED")
        else:
            print("✗ Invalid puzzle: INCORRECTLY ACCEPTED")
        
        print("✓ Validation example completed")
        
    except ImportError as e:
        print(f"✗ Required modules not available: {e}")
    except Exception as e:
        print(f"✗ Error in validation: {e}")
    
    print()


def print_sudoku(grid):
    """
    Pretty print a Sudoku grid.
    
    Args:
        grid (list): 9x9 Sudoku grid
    """
    print("+" + "-" * 21 + "+")
    for i, row in enumerate(grid):
        if i == 3 or i == 6:
            print("+" + "-" * 21 + "+")
        
        row_str = "|"
        for j, cell in enumerate(row):
            if j == 3 or j == 6:
                row_str += "|"
            
            if cell == 0:
                row_str += " ."
            else:
                row_str += f" {cell}"
        
        row_str += "|"
        print(row_str)
    print("+" + "-" * 21 + "+")


def main():
    """
    Run all examples.
    """
    print("Sudoku Scanner - Example Usage")
    print("=" * 40)
    print()
    
    # Run examples
    example_camera_processing()
    example_sudoku_solving()
    example_digit_recognition()
    example_validation()
    
    print("All examples completed!")
    print()
    print("Next steps:")
    print("1. Try running the full applications:")
    print("   - python3 main.py (camera processing)")
    print("   - python3 gui.py (interactive GUI)")
    print("2. Check the documentation:")
    print("   - README.md (main documentation)")
    print("   - API.md (API reference)")
    print("   - CONTRIBUTING.md (development guide)")


if __name__ == "__main__":
    main()