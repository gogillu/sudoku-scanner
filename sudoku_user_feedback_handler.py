import cv2
import os
from image_processor import ImageProcessor
from video_projector import VideoProjector
from ui_manager import UIManager
from frame_saver import FrameSaver
from camera_feed import CameraFeed
from video_creator import VideoCreator
from video_player import VideoPlayer
from contour_processor import ContourProcessor
import numpy as np

class SudokuUserFeedbackHandler:
    def __init__(self):
        self.camera_feed = CameraFeed()
        self.capture = False
        self.frame_count = 0
        self.frames = []
        self.contour_processor = ContourProcessor()

    def adjust_and_concatenate_images(self, img1, img2):
        """
        Adjusts img1 and img2 to have the same height by padding the smaller image with a black background.
        Then concatenates the images side by side.

        Args:
        - img1: First image (numpy array).
        - img2: Second image (numpy array).

        Returns:
        - Concatenated image.
        """
        # Get the height and width of both images
        height1, width1 = img1.shape[:2]
        height2, width2 = img2.shape[:2]

        # Determine the maximum height
        max_height = max(height1, height2)

        # If img1 is shorter, pad it
        if height1 < max_height:
            # Calculate the padding size
            diff = max_height - height1
            # Create padding
            pad = np.zeros((diff, width1, 3), dtype=np.uint8)
            # Stack the padding on img1
            img1_padded = np.vstack((img1, pad))
        else:
            img1_padded = img1

        # If img2 is shorter, pad it
        if height2 < max_height:
            # Calculate the padding size
            diff = max_height - height2
            # Create padding
            pad = np.zeros((diff, width2, 3), dtype=np.uint8)
            # Stack the padding on img2
            img2_padded = np.vstack((img2, pad))
        else:
            img2_padded = img2

        # Concatenate the adjusted images side by side
        combined_frame = np.hstack((img1_padded, img2_padded))

        return combined_frame

    def run(self):
        while True:
            frame = self.camera_feed.read_frame()
            if frame is None:
                break

            # cv2.imshow("Video Feed", frame)

            full_frame, cropped_grid = self.contour_processor.process_image(frame)

            # Combine the original frame (left) and the processed/last valid frame (right)
            combined_frame = self.adjust_and_concatenate_images(full_frame, cropped_grid)

            # Show the combined frame
            cv2.imshow("Real-time Feed vs. Filtered Feed", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
            
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('s'):
            #     self.capture = not self.capture
            # elif key == ord('e'):
            #     # processed_frames = [frame for frame in self.frames]
            #     # VideoProjector.project_video(self.frames)
                
            #     self.convert_frames_to_video(self.frames)
            #     break

        self.camera_feed.release()
        cv2.destroyAllWindows()

    # def convert_frames_to_video(self,frames, output_file="output_video.mp4", fps=20.0):
    #     if not frames:
    #         raise ValueError("The list of frames is empty.")

    #     frame_height, frame_width = frames[0].shape[:2]
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        
    #     for frame in frames:
    #         video_writer.write(frame)
        
    #     video_writer.release()
