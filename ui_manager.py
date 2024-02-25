# ui_manager.py
import cv2

class UIManager:
    def __init__(self):
        self.blink_status = False
        self.blink_counter = 0

    def draw_instructions(self, frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.rectangle(frame, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, frame.shape[0] - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_blinking_dot(self, frame):
        if self.blink_counter % 30 == 0:
            self.blink_status = not self.blink_status
        if self.blink_status:
            cv2.circle(frame, (50, 50), 10, (0, 0, 255), -1)
        self.blink_counter += 1
