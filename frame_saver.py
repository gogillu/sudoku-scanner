# frame_saver.py
import cv2
import os

class FrameSaver:
    def __init__(self, directory="input"):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_frame(self, frame, count):
        filename = f"frame_{count:04d}.jpg"
        cv2.imwrite(os.path.join(self.directory, filename), frame)
