import cv2
import os
from image_processor import ImageProcessor
from video_projector import VideoProjector
from ui_manager import UIManager
from frame_saver import FrameSaver
from camera_feed import CameraFeed
from video_creator import VideoCreator
from video_player import VideoPlayer

class CameraHandler:
    def __init__(self):
        self.camera_feed = CameraFeed()
        self.ui_manager = UIManager()
        self.frame_saver = FrameSaver()
        self.image_processor = ImageProcessor()
        self.capture = False
        self.frame_count = 0
        self.frames = []
        # self.video_creator = VideoCreator('output.mp4', 20.0, (640, 480))

    def run(self):
        while True:
            frame = self.camera_feed.read_frame()
            if frame is None:
                break

            if self.capture:
                self.ui_manager.draw_blinking_dot(frame)
                processed_frame = self.image_processor.process_image(frame, self.frame_count)
                self.frames.append(processed_frame)
                self.frame_saver.save_frame(processed_frame, self.frame_count)
                self.frame_count += 1
                # self.video_creator.add_frame(processed_frame)

            self.ui_manager.draw_instructions(frame, "Press 'S' to start, 'E' to exit")
            cv2.imshow("Video Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.capture = not self.capture
            elif key == ord('e'):
                processed_frames = [frame for frame in self.frames]
                VideoProjector.project_video(processed_frames)

                # After breaking out of the loop
                # self.video_creator.release()  # Finalize the video file
                # VideoPlayer.play_video('output.mp4')  # Play the compiled video

                self.convert_frames_to_video(self.frames)

                break

        self.camera_feed.release()
        cv2.destroyAllWindows()

    def convert_frames_to_video(self,frames, output_file="output_video.mp4", fps=20.0):
        if not frames:
            raise ValueError("The list of frames is empty.")

        print('--++--')
        print(len(frames))
        print('------')
        frame_height, frame_width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        
        for frame in frames:
            video_writer.write(frame)
        
        video_writer.release()
