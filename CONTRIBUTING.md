# Contributing to Sudoku Scanner

Thank you for your interest in contributing to the Sudoku Scanner project! This document provides guidelines and information for developers who want to contribute to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Structure](#code-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Known Issues](#known-issues)

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Git
- A webcam (for testing camera functionality)
- Basic knowledge of:
  - Computer Vision (OpenCV)
  - Machine Learning (TensorFlow/Keras)
  - GUI development (PyQt5)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/sudoku-scanner.git
   cd sudoku-scanner
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# For development, also install optional packages
pip install jupyter matplotlib scipy
```

### 3. Verify Installation

```bash
python3 check_installation.py
```

### 4. Test Basic Functionality

```bash
# Test camera processing (will fail gracefully if no camera)
python3 main.py

# Test GUI interface
python3 gui.py
```

## Code Structure

### Core Components

```
sudoku-scanner/
├── main.py                              # Main entry point
├── gui.py                              # PyQt5 GUI interface
├── sudoku_user_feedback_handler.py    # Camera processing handler
│
├── camera_feed.py                      # Camera interface
├── camera_handler.py                  # Camera management
│
├── contour_processor.py               # Image contour detection
├── crop_big_grid.py                   # Grid extraction
├── detect_small_grid.py               # Cell detection
├── image_grid_processor.py            # Grid processing
│
├── digit_recognizer.py                # ML digit recognition
├── tenserflow_machine_digit_predict_model.py  # TensorFlow interface
├── dlx.py                             # Dancing Links Algorithm
│
├── ui_manager.py                      # UI utilities
├── logger.py                          # Logging
├── frame_saver.py                     # Image saving
└── video_*.py                         # Video processing utilities
```

### Key Modules

1. **Camera Processing**: `camera_feed.py`, `camera_handler.py`
2. **Image Processing**: `contour_processor.py`, `crop_big_grid.py`, `detect_small_grid.py`
3. **Machine Learning**: `digit_recognizer.py`, `tenserflow_machine_digit_predict_model.py`
4. **GUI**: `gui.py`, `ui_manager.py`
5. **Algorithms**: `dlx.py` (Sudoku solving)

## Contributing Guidelines

### Types of Contributions

We welcome contributions in the following areas:

1. **Bug Fixes**: Fix existing issues or improve stability
2. **Feature Enhancements**: Improve existing functionality
3. **New Features**: Add new capabilities
4. **Documentation**: Improve or add documentation
5. **Performance**: Optimize algorithms or processing
6. **Testing**: Add or improve test coverage

### Areas for Improvement

#### High Priority
- **Error Handling**: Better error handling and user feedback
- **Performance**: Optimize image processing and ML inference
- **Model Accuracy**: Improve digit recognition accuracy
- **Cross-platform**: Better Windows/macOS compatibility

#### Medium Priority
- **UI/UX**: Improve GUI design and usability
- **Configuration**: Add configuration file support
- **Logging**: Improve logging and debugging capabilities
- **Video Processing**: Better video handling and export

#### Low Priority
- **Additional Models**: Support for different ML architectures
- **OCR Integration**: Better OCR fallback options
- **Grid Templates**: Support for different Sudoku variants

### Before You Start

1. **Check existing issues**: Look for related issues or feature requests
2. **Create an issue**: If no issue exists, create one to discuss your idea
3. **Get feedback**: Wait for maintainer feedback before starting large changes

## Testing

### Manual Testing

1. **Camera Processing**:
   ```bash
   python3 main.py
   # Test with different lighting conditions
   # Test with different Sudoku puzzles
   # Verify grid detection accuracy
   ```

2. **GUI Interface**:
   ```bash
   python3 gui.py
   # Test manual input
   # Test loading from images
   # Test both solving modes
   ```

3. **Installation**:
   ```bash
   python3 check_installation.py
   # Verify all dependencies
   # Test in clean environment
   ```

### Automated Testing

Currently, the project lacks comprehensive automated tests. Contributing test cases would be highly valuable:

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Image Processing Tests**: Test with known image inputs
4. **ML Model Tests**: Validate model predictions

### Test Data

When contributing tests, please:
- Use synthetic or clearly licensed test images
- Include both positive and negative test cases
- Document expected behavior
- Avoid large binary files in the repository

## Submitting Changes

### Pull Request Process

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the code style guidelines
   - Add appropriate documentation
   - Test your changes thoroughly

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**:
   - Provide clear description of changes
   - Reference related issues
   - Include screenshots for UI changes
   - List any breaking changes

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Changes are tested manually
- [ ] Documentation is updated if needed
- [ ] No new warnings or errors introduced
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all classes and methods
- Use type hints where appropriate
- Keep functions focused and reasonably sized

### Documentation Style

- Use clear, concise language
- Include code examples where helpful
- Document parameters and return values
- Explain complex algorithms or logic
- Update README.md for significant changes

### Example Code Style

```python
def detect_sudoku_grid(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Detect and extract a Sudoku grid from an image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        tuple: (extracted_grid, detection_success)
            - extracted_grid: Cropped grid image
            - detection_success: True if grid was found
            
    Raises:
        ValueError: If image is invalid or empty
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is invalid")
    
    # Process image...
    processed_image = preprocess_image(image)
    
    # Detect contours...
    contours = find_grid_contours(processed_image)
    
    return extracted_grid, success
```

## Known Issues

### Current Limitations

1. **Camera Compatibility**: Some cameras may not work properly
2. **Lighting Sensitivity**: Performance varies with lighting conditions
3. **Grid Detection**: May fail with partially visible or tilted grids
4. **Model Accuracy**: Digit recognition accuracy could be improved

### Areas Needing Work

1. **Error Recovery**: Better handling of processing failures
2. **Resource Management**: Memory and CPU usage optimization
3. **Cross-platform**: Testing and compatibility improvements
4. **Configuration**: User-configurable settings

## Getting Help

- **Issues**: Check existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Refer to README.md and API.md

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file (when created)
- Release notes for significant contributions
- Git commit history

Thank you for contributing to Sudoku Scanner!