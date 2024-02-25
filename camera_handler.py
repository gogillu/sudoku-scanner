import cv2
import os
from image_processor import ImageProcessor
from video_projector import VideoProjector

class CameraHandler:
    def __init__(self):
        self.capture = False
        self.frames = []
        self.blink_status = False
        self.blink_counter = 0
        # Initialize ImageProcessor with the path to save contour images
        self.image_processor = ImageProcessor()

    def draw_instructions(self, frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Calculate text size to center the text
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        # Position the text at the bottom of the frame with a black background
        cv2.rectangle(frame, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, frame.shape[0] - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_blinking_dot(self, frame):
        if self.blink_counter % 30 == 0:  # Adjust this value for slower blinking
            self.blink_status = not self.blink_status
        if self.blink_status:
            cv2.circle(frame, (50, 50), 10, (0, 0, 255), -1)
        self.blink_counter += 1

    def run_camera(self):
        cap = cv2.VideoCapture(0)  # Open the default camera
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # If the frame is not captured correctly, exit

            if self.capture:
                self.draw_blinking_dot(frame)
                # Save the original frame in the 'input' directory
                self.save_frame(frame, frame_count)
                frame_count += 1

            instruction_text = "Press 'S' to start capturing, 'E' to stop and process" if not self.capture else "Recording - Press 'E' to stop"
            self.draw_instructions(frame, instruction_text)
            cv2.imshow('Video Feed', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S') and not self.capture:
                self.capture = True
                self.frames = []
                self.blink_counter = 0
                frame_count = 0
                print("Capture started.")

            elif key == ord('e') or key == ord('E'):
                if self.capture:
                    self.capture = False
                    processed_frames = []
                    for i, frame in enumerate(self.frames):
                        # Process each frame to draw contours and save it
                        processed_frame = self.image_processor.process_image(frame, i)
                        processed_frames.append(processed_frame)
                    print("Processing and projecting captured frames...")
                    VideoProjector.project_video(processed_frames)
                else:
                    break  # Exit if 'E' is pressed without capturing

            if self.capture:
                self.frames.append(frame)

        cap.release()
        cv2.destroyAllWindows()

    def save_frame(self, frame, count):
        input_dir = 'input'
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        frame_path = os.path.join(input_dir, f'frame_{count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
