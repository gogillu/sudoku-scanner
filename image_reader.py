import os
import cv2

class ImageReader:
    def __init__(self, directory):
        self.directory = directory

    def read_images(self):
        return [os.path.join(self.directory, f) for f in os.listdir(self.directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
