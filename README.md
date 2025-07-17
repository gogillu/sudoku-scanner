# Sudoku Scanner

Sudoku Scanner is an advanced computer vision application that uses machine learning to detect and solve Sudoku puzzles in real-time from camera feeds or images. The application combines sophisticated image processing techniques with deep learning models to accurately recognize digits and provide instant solutions with optional visualization of the solving process.

## Features

- **Real-time Camera Processing:** Live detection and processing of Sudoku grids from camera feed
- **Intelligent Grid Detection:** Automatic extraction and preprocessing of Sudoku grids from images
- **Neural Network Digit Recognition:** Uses trained TensorFlow/Keras models for accurate digit identification
- **Multiple Solving Modes:** Choose between instant solving or step-by-step visualization
- **Interactive GUI:** User-friendly PyQt5 interface for manual input and solving
- **Backtracking Visualization:** Watch the solving algorithm work in real-time
- **Grid Cell Processing:** Automatic extraction and processing of individual Sudoku cells
- **Model Training Support:** Built-in support for training custom digit recognition models

## Demo Videos

### Real-time Sudoku Detection and Solving
https://www.youtube.com/watch?v=9uN24XRxSF0
[![Watch the video](https://img.youtube.com/vi/9uN24XRxSF0/hqdefault.jpg)](https://www.youtube.com/watch?v=9uN24XRxSF0)

### Backtracking Algorithm Visualization
https://www.youtube.com/watch?v=7ta76SZ3ZFY
[![Watch the video](https://img.youtube.com/vi/7ta76SZ3ZFY/hqdefault.jpg)](https://www.youtube.com/watch?v=7ta76SZ3ZFY)

## Installation

### Prerequisites

- Python 3.7 or higher
- A webcam (for real-time processing)
- Tesseract OCR (optional, for OCR fallback)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gogillu/sudoku-scanner.git
   cd sudoku-scanner
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR (optional):**
   - **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
   - **macOS:** `brew install tesseract`
   - **Windows:** Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Verify installation:**
   ```bash
   python3 -c "import cv2, tensorflow, PyQt5; print('All dependencies installed successfully')"
   ```

## Usage

The application provides multiple ways to interact with Sudoku puzzles:

### 1. Real-time Camera Processing
Process Sudoku puzzles from your camera feed in real-time:

```bash
python3 main.py
```

**Features:**
- Live camera feed with grid detection
- Automatic grid extraction and processing
- Real-time digit recognition
- Press 'q' to quit

### 2. Interactive GUI Mode
Use the graphical interface for manual input and solving:

```bash
python3 gui.py
```

**GUI Features:**
- **Load from Camera:** Capture and process a Sudoku from camera
- **Manual Input:** Enter digits manually in the 9x9 grid
- **Solve Slowly:** Watch the backtracking algorithm work step-by-step
- **Solve Instantly:** Get immediate solution
- Color-coded display showing original vs. solved digits

### 3. Recommended Workflow

For best results, follow this sequence:

1. **Start the camera processor:**
   ```bash
   python3 main.py
   ```
   - Position your Sudoku puzzle in front of the camera
   - Ensure good lighting and clear grid visibility
   - Wait for the system to detect and process the grid
   - Individual grid cells will be saved to `individual_grids/` folder

2. **Launch the GUI solver:**
   ```bash
   python3 gui.py
   ```
   - Click "Load sudoku from scanned image" to use the processed cells
   - Choose solving mode:
     - "Solve slowly" for educational visualization
     - "Solve instantly" for quick results

## Architecture Overview

### Core Components

1. **Camera Processing (`camera_feed.py`, `camera_handler.py`)**
   - Real-time video capture and frame processing
   - Camera interface management

2. **Image Processing Pipeline**
   - `contour_processor.py`: Detects and processes contours in images
   - `crop_big_grid.py`: Extracts and crops Sudoku grids from images
   - `detect_small_grid.py`: Processes individual cells within the grid
   - `image_grid_processor.py`: Handles grid segmentation and cell extraction

3. **Digit Recognition (`digit_recognizer.py`, `tenserflow_machine_digit_predict_model.py`)**
   - TensorFlow/Keras-based neural networks for digit classification
   - Support for multiple model architectures (MNIST-based, custom models)
   - Preprocessing and prediction pipeline

4. **Sudoku Solving**
   - `dlx.py`: Dancing Links Algorithm (DLX) implementation
   - `gui.py`: Backtracking algorithm with visualization
   - Multiple solving strategies and validation

5. **User Interface**
   - `gui.py`: PyQt5-based graphical interface
   - `ui_manager.py`: UI state management
   - Real-time visualization and interaction

6. **Utilities**
   - `logger.py`: Logging functionality
   - `frame_saver.py`: Image saving and management
   - `video_creator.py`, `video_player.py`: Video processing utilities

### Data Flow

1. **Image Acquisition** → Camera feed or static image
2. **Grid Detection** → Contour detection and perspective correction
3. **Cell Extraction** → Individual digit cell segmentation
4. **Digit Recognition** → Neural network prediction
5. **Grid Validation** → Sudoku rule checking
6. **Solving** → Algorithm execution with optional visualization
7. **Display** → Results presentation in GUI

## Model Information

The application uses pre-trained models for digit recognition:

- **mnist.h5**: MNIST-based digit classifier
- **keras_Model.h5**: Custom trained model for better accuracy on Sudoku digits
- Models are automatically loaded or trained if not present

### Training Custom Models

To train your own digit recognition model:

1. Collect training images in the `individual_grids/` folder
2. Run the digit recognizer training mode
3. The trained model will be saved automatically

## Troubleshooting

### Common Issues

1. **Camera not detected:**
   ```bash
   # Check available cameras
   python3 -c "import cv2; print('Cameras:', [i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
   ```

2. **Dependencies missing:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Qt platform plugin error:**
   ```bash
   export QT_QPA_PLATFORM=xcb  # Linux
   # or
   pip install --upgrade PyQt5
   ```

4. **TensorFlow GPU issues:**
   ```bash
   # For CPU-only installation
   pip install tensorflow-cpu
   ```

5. **Grid detection problems:**
   - Ensure good lighting conditions
   - Make sure the Sudoku grid is clearly visible and unobstructed
   - Check that the grid edges are well-defined

### Performance Tips

- Use good lighting for better digit recognition
- Ensure the camera is stable and the grid is clearly visible
- Close other applications to free up system resources
- Use the instant solve mode for faster results

## Documentation

This project includes comprehensive documentation:

- **[README.md](README.md)** - Main documentation (this file)
- **[API.md](API.md)** - Detailed API documentation for all classes and methods
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guide for contributors
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[check_installation.py](check_installation.py)** - Installation verification script
- **[examples.py](examples.py)** - Example usage and API demonstrations

## File Structure

```
sudoku-scanner/
├── README.md                           # Main documentation
├── API.md                             # API reference documentation
├── CONTRIBUTING.md                    # Development and contribution guide
├── requirements.txt                   # Python dependencies
├── check_installation.py             # Installation verification
├── examples.py                       # Usage examples
│
├── main.py                           # Main entry point for camera processing
├── gui.py                           # PyQt5 GUI interface
├── sudoku_user_feedback_handler.py  # User interaction handler
│
├── camera_feed.py                   # Camera interface
├── camera_handler.py               # Camera management
├── contour_processor.py            # Image contour detection
├── crop_big_grid.py                # Grid extraction and cropping
├── detect_small_grid.py            # Individual cell detection
├── image_grid_processor.py         # Grid processing utilities
│
├── digit_recognizer.py             # Digit recognition using ML
├── tenserflow_machine_digit_predict_model.py  # TensorFlow model interface
├── dlx.py                          # Dancing Links Algorithm
│
├── individual_grids/               # Extracted digit images (created at runtime)
├── keras_Model.h5                 # Pre-trained Keras model
└── mnist.h5                       # MNIST-based model
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Follow the installation instructions above
2. Install development dependencies:
   ```bash
   pip install jupyter matplotlib scipy
   ```
3. Run tests and ensure everything works before submitting changes

## References and Acknowledgments

1. [Deep Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/)
2. [Teachable Machine](https://teachablemachine.withgoogle.com/train/image)
3. OpenCV for computer vision functionality
4. TensorFlow/Keras for machine learning models
5. PyQt5 for the graphical user interface

## License

This project is open source. Please check the repository for license details.
