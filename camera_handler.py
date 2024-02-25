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

    def draw_instructions(self, frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 10
        # Create a black background rectangle for the text
        cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_blinking_dot(self, frame):
        if self.blink_counter % 30 == 0:  # Adjust this value for slower blinking
            self.blink_status = not self.blink_status
        if self.blink_status:
            cv2.circle(frame, (50, 50), 10, (0, 0, 255), -1)
        self.blink_counter += 1

    def save_frame(self, frame, count):
        if not os.path.exists('input'):
            os.makedirs('input')
        frame_path = os.path.join('input', f'frame_{count:04d}.jpg')
        cv2.imwrite(frame_path, frame)

    def run_camera(self):
        cap = cv2.VideoCapture(0)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.capture:
                self.draw_blinking_dot(frame)
                self.save_frame(frame, frame_count)
                frame_count += 1

            instruction_text = "Press 'S' to start capturing, 'E' to stop and process" if not self.capture else "Recording - Press 'E' to stop"
            self.draw_instructions(frame, instruction_text)

            cv2.imshow('Video Feed', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                self.capture = True
                self.frames = []
                self.blink_counter = 0
                frame_count = 0
                print("Capture started.")

            elif key == ord('e') or key == ord('E'):
                if self.capture:
                    self.capture = False
                    print("Processing and projecting captured frames...")
                    processed_frames = [ImageProcessor.process_image(frame) for frame in self.frames]
                    VideoProjector.project_video(processed_frames)
                else:
                    break

            if self.capture:
                self.frames.append(frame)

        cap.release()
        cv2.destroyAllWindows()
