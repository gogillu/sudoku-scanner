# Add imports at the top
from camera_handler import CameraHandler

class MainController:
    def __init__(self):
        self.camera_handler = CameraHandler()

    def run(self):
        self.camera_handler.run()

# Main execution
if __name__ == "__main__":
    controller = MainController()
    controller.run()
