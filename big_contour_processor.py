from image_reader import ImageReader
from contour_processor import ContourProcessor
from logger import Logger

class BigContourProcessor:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.image_reader = ImageReader(input_dir)
        self.contour_processor = ContourProcessor()
        
    def run(self):
        image_paths = self.image_reader.read_images()
        for image_path in image_paths:
            found = self.contour_processor.process_image(image_path)
            if found:
                Logger.log(f"Large contour found and processed for {image_path}")
            else:
                Logger.log(f"Sufficiently large contour not found for {image_path}")

