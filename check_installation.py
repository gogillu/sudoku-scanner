#!/usr/bin/env python3
"""
Installation Verification Script for Sudoku Scanner

This script verifies that all required dependencies are properly installed
and the application can be initialized without errors.

Usage:
    python3 check_installation.py
"""

import sys
import importlib


def check_module(module_name, friendly_name=None):
    """
    Check if a module can be imported.
    
    Args:
        module_name (str): Name of the module to import
        friendly_name (str): Human-friendly name for display
        
    Returns:
        bool: True if module is available, False otherwise
    """
    if friendly_name is None:
        friendly_name = module_name
        
    try:
        importlib.import_module(module_name)
        print(f"✓ {friendly_name} is available")
        return True
    except ImportError:
        print(f"✗ {friendly_name} is NOT available")
        return False


def check_camera():
    """
    Check if camera is available (requires OpenCV).
    
    Returns:
        bool: True if camera is accessible, False otherwise
    """
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera is accessible")
            cap.release()
            return True
        else:
            print("⚠ Camera is not accessible (may be in use or not connected)")
            return False
    except:
        print("✗ Cannot check camera (OpenCV not available)")
        return False


def check_directories():
    """
    Check and create required directories.
    
    Returns:
        bool: True if directories can be created, False otherwise
    """
    import os
    
    required_dirs = ['individual_grids', 'tmp']
    
    try:
        for dir_name in required_dirs:
            os.makedirs(dir_name, exist_ok=True)
            if os.path.exists(dir_name):
                print(f"✓ Directory '{dir_name}' is ready")
            else:
                print(f"✗ Cannot create directory '{dir_name}'")
                return False
        return True
    except Exception as e:
        print(f"✗ Error creating directories: {e}")
        return False


def main():
    """
    Main verification function.
    """
    print("Sudoku Scanner - Installation Verification")
    print("=" * 45)
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 7):
        print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro} is supported")
    else:
        print(f"✗ Python {python_version.major}.{python_version.minor} is too old. Python 3.7+ required")
        return False
    
    print("\nChecking required modules:")
    print("-" * 25)
    
    # Check core dependencies
    modules_ok = True
    modules_ok &= check_module('cv2', 'OpenCV (cv2)')
    modules_ok &= check_module('numpy', 'NumPy')
    modules_ok &= check_module('tensorflow', 'TensorFlow')
    modules_ok &= check_module('keras', 'Keras')
    
    # Check GUI dependencies
    print("\nChecking GUI dependencies:")
    print("-" * 25)
    modules_ok &= check_module('PyQt5', 'PyQt5')
    modules_ok &= check_module('PyQt5.QtWidgets', 'PyQt5 Widgets')
    modules_ok &= check_module('PyQt5.QtCore', 'PyQt5 Core')
    
    # Check image processing dependencies
    print("\nChecking image processing dependencies:")
    print("-" * 35)
    modules_ok &= check_module('PIL', 'Pillow (PIL)')
    
    # Optional dependencies
    print("\nChecking optional dependencies:")
    print("-" * 28)
    check_module('pytesseract', 'Tesseract OCR (optional)')
    check_module('keras_ocr', 'Keras OCR (optional)')
    check_module('matplotlib', 'Matplotlib (optional)')
    
    # Check directories
    print("\nChecking directories:")
    print("-" * 18)
    dirs_ok = check_directories()
    
    # Check camera
    print("\nChecking hardware:")
    print("-" * 16)
    camera_ok = check_camera()
    
    # Summary
    print("\nSummary:")
    print("=" * 45)
    
    if modules_ok:
        print("✓ All required modules are available")
    else:
        print("✗ Some required modules are missing")
        print("  Run: pip install -r requirements.txt")
    
    if dirs_ok:
        print("✓ All required directories are ready")
    else:
        print("✗ Could not create required directories")
    
    if camera_ok:
        print("✓ Camera is ready for use")
    elif modules_ok:  # Only warn about camera if OpenCV is available
        print("⚠ Camera check failed (may work when actually running the app)")
    
    print("\nNext steps:")
    if modules_ok and dirs_ok:
        print("1. Run: python3 main.py (for camera processing)")
        print("2. Run: python3 gui.py (for interactive GUI)")
    else:
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Re-run this verification script")
        print("3. If successful, run: python3 main.py or python3 gui.py")
    
    return modules_ok and dirs_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)