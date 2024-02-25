import cv2
import numpy as np

def is_square_or_rectangle(contour, epsilon_factor=0.04):
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

def process_frame_for_large_contours(frame, specific_area=150000):
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
    cv2.imshow("blur",img_canny)
    img_copy = frame.copy()

    # Convert to black and white
    # _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    big_square_contour_found = False

    contour_img = img_copy #np.copy(frame)
    for contour in contours:
        area = cv2.contourArea(contour)
        if True and (area > specific_area and is_square_or_rectangle(contour)):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)
                cv2.putText(contour_img, f"Area: {area}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(contour_img, "Square/Rect", (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                big_square_contour_found = True

    return contour_img,big_square_contour_found

def capture_from_video_feed():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video feed.")
        return
    
    last_valid_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame,big_square_contour_found = process_frame_for_large_contours(frame, specific_area=50000)

        # Update last_valid_frame if criteria met
        if big_square_contour_found:
            last_valid_frame = processed_frame
        # If no valid frame yet, use a black placeholder
        if last_valid_frame is None:
            last_valid_frame = processed_frame

        if big_square_contour_found:
            # cv2.imshow("Processed Video Feed", processed_frame)
            last_valid_frame = processed_frame


        # Combine the original frame (left) and the processed/last valid frame (right)
        combined_frame = np.hstack((frame, last_valid_frame))

        # Show the combined frame
        cv2.imshow("Real-time Feed vs. Filtered Feed", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_from_video_feed()
