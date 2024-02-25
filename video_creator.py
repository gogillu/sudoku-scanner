# video_creator.py
import cv2

class VideoCreator:
    def __init__(self, filename='output.mp4', fps=20.0, frame_size=(640, 480)):
        # Use 'XVID' for compatibility, adjust for MP4 as needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fourcc, frame_size, True)

    def add_frame(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()
