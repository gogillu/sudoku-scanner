import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image

def get_or_load_mnist_model(model_path='mnist_model.h5'):
    if os.path.exists(model_path):
        print("Loading model from file.")
        model = load_model(model_path)
    else:
        print("Downloading and training model for the first time.")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        model = Sequential([
            Flatten(input_shape=(28, 28, 1)),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        model.save(model_path)
        print("Model trained and saved to {}".format(model_path))
    
    return model

def predict_digit(image_path, model_path='mnist_model.h5'):
    model = get_or_load_mnist_model(model_path)
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_data = np.array(img)
    img_data = img_data.reshape(1, 28, 28, 1).astype('float32') / 255
    
    prediction = model.predict(img_data)
    return np.argmax(prediction)

# Example usage:
# Replace 'path_to_your_image.jpg' with the actual path to an image
predicted_digit = predict_digit('/Users/govind/projects/self/sudoku/individual_grids/'+'grid_5.jpg')
print(f'The predicted digit is: {predicted_digit}')
