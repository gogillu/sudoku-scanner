import cv2
import os

class CameraHandler:
    def __init__(self):
        self.capture = False
        self.blink_status = False
        self.blink_counter = 0  # Counter to control the blink speed

    def draw_instructions(self, frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_background_height = 50  # Height of the black background for the text
        frame_height, frame_width = frame.shape[:2]
        # Extend frame to include a black background for the text below the video feed
        extended_frame = cv2.copyMakeBorder(frame, 0, text_background_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(extended_frame, text, (10, frame_height + 35), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return extended_frame

    def draw_blinking_dot(self, frame):
        # Only toggle the blink status after a certain number of frames have passed
        if self.blink_counter % 20 == 0:  # Adjust this value to make the blinking slower or faster
            self.blink_status = not self.blink_status
        self.blink_counter += 1

        # Adjusting the position of the blinking dot to be in the top left corner
        dot_position = (30, 30)
        dot_color = (0, 0, 255)  # Red color in BGR
        dot_radius = 10

        if self.blink_status:
            cv2.circle(frame, dot_position, dot_radius, dot_color, -1)  # Filled circle

    def run_camera(self):
        cap = cv2.VideoCapture(0)  # Open the default camera
        frame_rate = 24
        count = 0
        frames = []

        if not os.path.exists('input'):
            os.makedirs('input')

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # If the frame is not captured correctly, exit

            # Draw the blinking dot first to ensure it appears on the video feed
            if self.capture:
                self.draw_blinking_dot(frame)

            if not self.capture:
                extended_frame = self.draw_instructions(frame, "Press 'S' to start capturing")
            else:
                extended_frame = self.draw_instructions(frame, "Recording - Press 'E' to stop")

            cv2.imshow('Video Feed', extended_frame)

            if self.capture:
                # Capture frames at the rate of 24 frames per second
                if count % (30 // frame_rate) == 0:
                    frame_path = os.path.join('input', f'frame_{count}.jpg')
                    cv2.imwrite(frame_path, frame)
                    frames.append(frame)
                count += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                self.capture = True
                count = 0  # Reset count when starting capture
                self.blink_counter = 0  # Also reset the blink counter

            if key == ord('e') or key == ord('E'):
                self.capture = False
                print(f"Captured {len(frames)} frames.")
                break  # Exit loop

        cap.release()
        cv2.destroyAllWindows()

        # Here you might want to handle `frames` list, e.g., saving or processing
