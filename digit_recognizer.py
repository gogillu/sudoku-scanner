import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from PIL import Image
import pytesseract

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
        print(predictions)
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
    left = int(width * 0.15)
    top = int(height * 0.15)
    right = int(width * 0.85)
    bottom = int(height * 0.85)
    
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

from tensorflow.keras.models import load_model
model = load_model('mnist_model.h5')

def preprocess_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to the size expected by your model, e.g., 28x28 for MNIST
    img = cv2.resize(img, (28, 28))
    
    # Invert image colors so that the digit is white and the background is black
    img = 255 - img
    
    # Normalize pixel values to be between 0 and 1
    img = img / 255.0
    
    # Expand dimensions to match the model's input shape, e.g., (1, 28, 28, 1) for MNIST
    img = np.expand_dims(img, axis=[0, -1])
    
    return img

def is_image_empty(image_path, threshold=0.75):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate the percentage of white pixels
    white_pixels = np.sum(img > 200) / (img.shape[0] * img.shape[1])
    
    # If the percentage of white pixels is above the threshold, consider it as empty
    return white_pixels > threshold

def predict_digit(image_path):
    if is_image_empty(image_path):
        return " "
    else:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        return digit

def image_contains_digit(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image is empty
    if image is None or np.all(image == 255):
        return " "
    
    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Use pytesseract to do OCR on the image
    custom_config = r'--oem 3 --psm 10 outputbase digits'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    # Remove any non-digit characters
    text = ''.join(filter(str.isdigit, text))
    
    if text:
        # Check if the text is a digit
        if text.isdigit() and 1 <= int(text) <= 9:
            return text
        else:
            return "X"
    else:
        return " "

import keras_ocr

# Function to detect and recognize text in an image using keras-ocr
def get_text_from_image(image_path):
    # keras-ocr will automatically download pre-trained weights for the detector and recognizer.
    pipeline = keras_ocr.pipeline.Pipeline()

    # Load the image
    image = keras_ocr.tools.read(image_path)

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([image])

    # We will take the first (and only) prediction group
    predictions = prediction_groups[0]

    # Extracting text and boxes from the predictions
    text_blocks = [(text, box) for text, box in predictions]

    # Returning the text blocks
    return text_blocks

# Example usage:
# text_blocks = get_text_from_image('/path/to/your/image.jpg')
# for text, box in text_blocks:
#     print(f'Detected text: {text}, with bounding box: {box}')


from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

def predict_digit_new(image_path):
    # Load the pre-trained model
    model = load_model('mnist.h5')
    
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to grayscale, resize it to 28x28 (the size expected by the model), and invert colors
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    
    # Convert the image to a numpy array, normalize it, and add a batch dimension
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)  # Assuming the model expects a 4D input: [batch_size, height, width, channels]
    
    # Predict the digit
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    
    return predicted_digit

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

def get_mnist_model(model_path='mnist_model.h5'):
    # Check if the model file exists
    if os.path.exists(model_path):
        print("Loading model from file.")
        # Load the model from the file
        model = load_model(model_path)
    else:
        print("Training model for the first time.")
        # Load data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Preprocess the data
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # Build the model
        model = Sequential([
            Flatten(input_shape=(28, 28, 1)),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

        # Save the model to the specified path
        model.save(model_path)
        print("Model trained and saved to {}".format(model_path))
    
    return model

# Example usage
if __name__ == "__main__":
    recognizer = DigitRecognizer()
    get_mnist_model()

    matrix = [[0] * 9 for _ in range(9)]
    
    z = 0
    for i in range(0,9):
        print('\n')
        for j in range(0,9):

        # non_hidden_files = [file for file in os.listdir('individual_grids') if not file.startswith('.')]
        # for entry in non_hidden_files:
        # full_path = '/Users/govind/projects/self/sudoku/individual_grids/'+entry #os.path.join('individual_grids', entry)

            if True:
                print(' ')
                z += 1

                # Load an image
                image_path = 'individual_grids/grid_'+str(z)+'.jpg'  # Specify the path to an image of a digit
                frame = cv2.imread(image_path)
                # print(full_path)
                cropped_image = crop_image(image_path)

                # cv2.imshow("cropped",cropped_image)
                frame_resized = cv2.resize(cropped_image, (28, 28))
                cv2.imwrite('temp_'+image_path, frame_resized)
                
                if frame is not None:
                    # matrix[i][j] = get_text_from_image(image_path) #predict_digit("temp_"+image_path)
                    print(image_path)
                    digit, confidence = recognizer.predict(frame_resized)
                    # digit = predict_digit_new(full_path)
                    print(digit,confidence)
                    if confidence > 0.41:
                        matrix[i][j] =  digit #extract_digits(image_path)
                    else:
                        matrix[i][j] = ' '
                    print(f"Predicted Digit: {digit}, Confidence: {confidence}")
                else:
                    print("Error loading the image.")

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    break

    for i in range(0,9):
        print('\n')
        for j in range(0,9):
            print(matrix[i][j],end=' ')

