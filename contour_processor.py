import cv2
import os
import numpy as np
from datetime import datetime

class ContourProcessor:
    def __init__(self, output_dir="big_contour_crop"):
        self.output_dir = output_dir
        self.grid_dir = output_dir + "_only_biggest_grid"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.last_big_contour_img_cropped = None

    def process_image(self, frame):
        # image = cv2.imread(image_path)
        processed_img, is_large_contour_present = self.process_frame_for_large_contours(frame)

        if is_large_contour_present:
            filename = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")+".jpg"
            cv2.imwrite(os.path.join(self.output_dir, filename), processed_img)
            self.last_big_contour_img_cropped = self.crop_to_biggest_contour(frame)
            cv2.imwrite(os.path.join(self.grid_dir, filename), self.last_big_contour_img_cropped)

        if self.last_big_contour_img_cropped is not None:
            return processed_img,self.last_big_contour_img_cropped
        
        return processed_img,processed_img

    def is_square_or_rectangle(self, contour, epsilon_factor=0.04):
        """
        Determines if a contour is a square or rectangle.

        Args:
        - contour: The contour to check.
        - epsilon_factor: Factor for approxPolyDP accuracy.

        Returns:
        - True if the contour is square or rectangle, False otherwise.
        """
        # Approximate the contour to a simpler shape
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)

        # Squares and rectangles have 4 vertices
        if len(approx) == 4:
            # Optionally, check if the shape is approximately a square (all sides are equal)
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                return True  # More likely a square due to aspect ratio
            return True  # Treat as rectangle/square without strict aspect ratio check
        return False

    def process_frame_for_large_contours(self, frame, specific_area=150000):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the minimum and maximum pixel values in the grayscale image
        min_val = np.min(gray)
        max_val = np.max(gray)

        # Perform contrast adjustment
        alpha = 255.0 / (max_val - min_val)
        beta = -min_val * alpha

        adjusted_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        img_blur = cv2.GaussianBlur(adjusted_image,(5,5),3)
        img_canny = cv2.Canny(img_blur,3,3)
        # cv2.imshow("blur",img_canny)
        img_copy = frame.copy()

        # Convert to black and white
        # _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        big_square_contour_found = False

        contour_img = img_copy #np.copy(frame)
        for contour in contours:
            area = cv2.contourArea(contour)
            if specific_area < 1000:
                print(specific_area, area)
            
            if True and (area > specific_area and self.is_square_or_rectangle(contour)):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)
                    cv2.putText(contour_img, f"Area: {area}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(contour_img, "Square/Rect", (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    big_square_contour_found = True

        return contour_img,big_square_contour_found

    def show_small_contours(self, frame, specific_area=150000):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Calculate the minimum and maximum pixel values in the grayscale image
        # min_val = np.min(gray)
        # max_val = np.max(gray)

        # # Perform contrast adjustment
        # alpha = 255.0 / (max_val - min_val)
        # beta = -min_val * (0.1 + alpha)

        # adjusted_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        img_blur = cv2.GaussianBlur(gray,(5,5),3)
        img_canny = cv2.Canny(img_blur,3,3)
        # cv2.imshow("blur",img_canny)
        img_copy = frame.copy()

        # Convert to black and white
        # _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour_img = img_copy #np.copy(frame)

        big_square_contour_found = False
        # print(len(contours))
        useful_contours = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            # print(area)
            if area < 5000:
                useful_contours += 1

            # if specific_area < 1000:
            #     print(specific_area, area)
            
            if True and (area > specific_area and self.is_square_or_rectangle(contour)):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # big_square_contour_found = True

                    cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)
                    cv2.putText(contour_img, f"Area: {area}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.putText(contour_img, "Square/Rect", (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        print('useful contours', useful_contours)

        return contour_img #,big_square_contour_found

    def crop_to_biggest_contour(self, frame):
        """
        Processes the given frame, finds contours, identifies the biggest contour,
        and returns a cropped image around this contour.

        Args:
        - frame: The input image frame.

        Returns:
        - cropped_image: The cropped image around the biggest contour. Returns None if no contours are found.
        """
        # Convert to grayscale for better contour detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate the minimum and maximum pixel values in the grayscale image
        min_val = np.min(gray)
        max_val = np.max(gray)

        # Perform contrast adjustment
        alpha = 255.0 / (max_val - min_val)
        beta = -min_val * alpha

        adjusted_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        img_blur = cv2.GaussianBlur(adjusted_image,(5,5),3)
        img_canny = cv2.Canny(img_blur,3,3)
        # cv2.imshow("blur",img_canny)
        # img_copy = frame.copy()

        # Convert to black and white
        # _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No contours were found
            return None

        # Find the biggest contour based on area
        biggest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the biggest contour
        x, y, w, h = cv2.boundingRect(biggest_contour)
        
        # Crop the image to this bounding box
        cropped_image = frame[y:y+h, x:x+w]

        return cropped_image
