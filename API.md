# API Documentation

This document provides detailed API documentation for the Sudoku Scanner application's main components and classes.

## Core Classes

### MainController

**File:** `main.py`

The main entry point controller that orchestrates the Sudoku Scanner application.

#### Methods

##### `__init__()`
Initializes the MainController with the sudoku user feedback handler.

##### `run()`
Starts the main application loop for real-time camera processing.

**Usage:**
```python
controller = MainController()
controller.run()
```

---

### SudokuUserFeedbackHandler

**File:** `sudoku_user_feedback_handler.py`

Handles real-time camera feed processing and Sudoku grid detection.

#### Attributes
- `camera_feed`: CameraFeed instance for video capture
- `crop_big_grid`: CropBigGrid instance for grid detection
- `detect_small_grid`: DetectSmallGrid instance for cell detection
- `image_grid_processor`: ImageGridProcessor instance for cell processing

#### Methods

##### `__init__()`
Initializes all image processing components.

##### `run()`
Starts the main real-time processing loop.
- Captures frames from camera
- Detects and crops Sudoku grids
- Extracts individual cells
- Displays combined visualization
- Saves grid parts for recognition

**Controls:**
- Press 'q' to quit

##### `get_ongoing_frame()`
Returns a single processed frame without starting continuous loop.

**Returns:**
- `tuple`: (full_frame, cropped_grid, combined_frame_81, combined_frame_81_BnW_smaller)

##### `resize_half(frame)`
Utility method to resize frame to half dimensions and save it.

**Parameters:**
- `frame` (numpy.ndarray): Input frame to resize

**Returns:**
- `numpy.ndarray`: Resized frame

---

### MainWindow (GUI)

**File:** `gui.py`

PyQt5-based graphical interface for interactive Sudoku solving.

#### Attributes
- `matrix`: 9x9 current grid state
- `prefilled_matrix`: Boolean matrix indicating pre-filled cells
- `worker`: Worker thread for background operations
- `model`: ML model for digit recognition
- `edit_boxes`: 2D list of QLineEdit widgets

#### Methods

##### `__init__(matrix, worker)`
Initializes the GUI window.

**Parameters:**
- `matrix` (list): Initial 9x9 grid state
- `worker` (Worker): Worker thread instance

##### `initUI()`
Creates the user interface with 9x9 grid and control buttons.

##### `update_values(matrix)`
Updates grid display with new values.

**Parameters:**
- `matrix` (list): 9x9 matrix of values to display

##### `update_values_prefilled(matrix)`
Updates grid with distinction between prefilled and solved values.

**Parameters:**
- `matrix` (list): 9x9 matrix of values to display

##### `load()`
Loads Sudoku puzzle from camera-processed images in `individual_grids/` directory.

##### `solve_old()`
Solves puzzle with step-by-step visualization using backtracking algorithm.

##### `solve_instantly()`
Solves puzzle immediately without visualization.

##### `solve(input_board)`
Core solving method using fast backtracking algorithm.

**Parameters:**
- `input_board` (list): 9x9 input puzzle

**Returns:**
- `tuple`: (solution_board, success_flag)

---

### Worker Thread

**File:** `gui.py`

QThread subclass for handling computationally intensive operations.

#### Signals
- `update_signal`: Emitted when GUI should be updated with new values

#### Methods

##### `register_ui(main_window)`
Registers the main window for interaction.

##### `is_valid_move(row, col, num, matrix)`
Validates if placing a number at position is legal.

**Parameters:**
- `row` (int): Row index (0-8)
- `col` (int): Column index (0-8)
- `num` (int): Number to place (1-9)
- `matrix` (list): Current 9x9 grid

**Returns:**
- `bool`: True if move is valid

##### `solve_sudoku(matrix, responsive=True, ui_resp=0.002)`
Solves Sudoku with optional real-time visualization.

**Parameters:**
- `matrix` (list): 9x9 grid to solve
- `responsive` (bool): Whether to emit real-time updates
- `ui_resp` (float): Delay between updates

**Returns:**
- `bool`: True if solved successfully

##### `transform_1_9_or_empty_sudoku(matrix)`
Converts numerical matrix to string matrix for GUI display.

**Parameters:**
- `matrix` (list): 9x9 numerical matrix

**Returns:**
- `list`: 9x9 string matrix

---

## Image Processing Components

### CameraFeed

**File:** `camera_feed.py`

Manages camera input and frame capture.

#### Methods
- `read_frame()`: Captures and returns current frame
- `release()`: Releases camera resources

### CropBigGrid

**File:** `crop_big_grid.py`

Handles Sudoku grid detection and cropping from camera frames.

#### Methods
- `get_combined_video_capture_and_cropped_full_grid(frame)`: Detects and crops grid
- `adjust_and_concatenate_images(img1, img2)`: Combines images horizontally
- `adjust_and_concatenate_images_vertically(img1, img2)`: Combines images vertically

### DetectSmallGrid

**File:** `detect_small_grid.py`

Processes individual cells within detected Sudoku grids.

#### Methods
- `get_separate_small_grids(cropped_grid)`: Extracts individual cell contours

### ImageGridProcessor

**File:** `image_grid_processor.py`

Manages grid segmentation and individual cell extraction.

#### Methods
- `divide_and_combine_frame(frame)`: Divides grid into 81 cells
- `save_grid_parts(frame)`: Saves individual cells to `individual_grids/`
- `divide_and_combine_small_cropped_frame_black_n_white(frame)`: Creates B&W cell visualization

---

## Machine Learning Components

### DigitRecognizer

**File:** `digit_recognizer.py`

Handles digit recognition using trained neural networks.

#### Methods
- `train_model()`: Trains new digit recognition model
- `predict_digit(image)`: Predicts digit from image
- `preprocess_image(image)`: Preprocesses image for prediction

### Model Interface

**File:** `tenserflow_machine_digit_predict_model.py`

Interface for TensorFlow/Keras model operations.

#### Functions
- `loadModel()`: Loads pre-trained model and class names
- `predict_with_teachable_ml_optimized(image_path, model, class_names)`: Predicts digit from image file

---

## Utility Functions

### Validation Functions

**File:** `gui.py`

#### `is_valid_sudoku(board)`
Validates if Sudoku board configuration is valid.

**Parameters:**
- `board` (list): 9x9 Sudoku board

**Returns:**
- `bool`: True if valid

#### `generate_random_matrix()`
Generates empty 9x9 matrix for testing.

**Returns:**
- `list`: 9x9 matrix with empty strings

#### `run_method(obj_instance, method_name, parameter)`
Utility to run method on object instance.

**Parameters:**
- `obj_instance`: Object to call method on
- `method_name` (str): Method name
- `parameter`: Parameter to pass

---

## Data Structures

### Grid Representation

Sudoku grids are represented as 9x9 lists:

```python
# Numerical representation (0 = empty)
matrix = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    # ... 7 more rows
]

# String representation ("" = empty)
string_matrix = [
    ["5", "3", "", "", "7", "", "", "", ""],
    ["6", "", "", "1", "9", "5", "", "", ""],
    # ... 7 more rows
]
```

### Image Processing Pipeline

1. **Frame Capture** → `camera_feed.read_frame()`
2. **Grid Detection** → `crop_big_grid.get_combined_video_capture_and_cropped_full_grid()`
3. **Cell Extraction** → `image_grid_processor.divide_and_combine_frame()`
4. **Cell Saving** → `image_grid_processor.save_grid_parts()`
5. **Digit Recognition** → `predict_with_teachable_ml_optimized()`

---

## Error Handling

### Common Exceptions

- **Camera Not Found**: Check camera availability and permissions
- **Model Loading Error**: Ensure model files exist and are valid
- **Image Processing Error**: Verify image format and quality
- **GUI Thread Error**: Ensure proper Qt signal/slot connections

### Debug Mode

Enable verbose output by setting debug flags in respective modules:

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Configuration

### Model Paths
- `mnist.h5`: MNIST-based digit classifier
- `keras_Model.h5`: Custom trained model
- `individual_grids/`: Directory for extracted cell images

### Camera Settings
- Default camera index: 0
- Frame resolution: Auto-detected
- Processing FPS: ~30 (depends on hardware)

### GUI Settings
- Grid cell size: 40x40 pixels
- Font sizes: 24px (normal), 36px (solved)
- Color scheme: Gray/light gray alternating backgrounds