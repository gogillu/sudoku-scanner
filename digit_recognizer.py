import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
# import pytesseract

# Update this line with the correct path to your Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

class DigitRecognizer:
    def __init__(self, model_path='mnist_model.h5'):
        self.model_path = model_path
        # Check if the model file exists
        if os.path.exists(self.model_path):
            print("Loading the pre-trained model...")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("Model file not found, training the model...")
            self.model = self.train_model()
            # Save the trained model
            self.model.save(self.model_path)
            print("Model trained and saved.")

    def train_model(self):
        # Load and prepare the MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        # Normalize the input images
        train_images, test_images = train_images / 255.0, test_images / 255.0

        # Add a channels dimension
        train_images = train_images[..., tf.newaxis]
        test_images = test_images[..., tf.newaxis]

        # Build the model
        model = self.build_model()

        # Train the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

        return model

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        return model

    def predict(self, frame):
        # Preprocess the frame
        frame_resized = cv2.resize(frame, (28, 28))
        cv2.imshow('resized',frame_resized)
        print('==')
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_normalized = frame_gray / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=[0, -1])

        # Make a prediction
        predictions = self.model.predict(frame_expanded)
        predicted_digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return predicted_digit, confidence

from PIL import Image

def crop_image(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Get dimensions of the original image
    height, width = img.shape[:2]
    
    # Calculate the cropping dimensions
    left = int(width * 0.1)
    top = int(height * 0.1)
    right = int(width * 0.9)
    bottom = int(height * 0.9)
    
    # Crop the image using array slicing
    cropped_img = img[top:bottom, left:right]
    
    # Return the cropped image
    return cropped_img

# On macOS, if you've installed Tesseract via Homebrew, 
# you usually don't need to set 'tesseract_cmd' as it should be in the PATH.
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract' # Uncomment if necessary

def extract_digits(img_path):
    img = cv2.imread(img_path)
    # Convert to grayscale for better OCR accuracy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use Tesseract to extract text. Note the correction in the custom config
    custom_config = r'--oem 3 --psm 6 digits'  # Focus on digit recognition
    text = pytesseract.image_to_string(gray, config=custom_config)
    print("Extracted Text:", text)
    return text

# Example usage
if __name__ == "__main__":
    recognizer = DigitRecognizer()

    matrix = [[0] * 9 for _ in range(9)]
    
    z = 0
    for i in range(0,9):
        print('\n')
        for j in range(0,9):
            print(' ')
            z += 1

            # Load an image
            image_path = 'individual_grids/grid_'+str(z)+'.jpg'  # Specify the path to an image of a digit
            frame = cv2.imread(image_path)
            # cropped_image = crop_image(image_path)
            # cv2.imshow("cropped",cropped_image)
            
            if frame is not None:
                digit, confidence = recognizer.predict(frame)
                if confidence > 0.75:
                    matrix[i][j] =  digit#extract_digits(image_path)
                else:
                    matrix[i][j] = ' '
                print(f"Predicted Digit: {digit}, Confidence: {confidence}")
            else:
                print("Error loading the image.")

    for i in range(0,9):
        print('\n')
        for j in range(0,9):
            print(matrix[i][j],end=' ')

