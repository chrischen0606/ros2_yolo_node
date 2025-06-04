import cv2
import numpy as np
import os

def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_bgr = image[y, x]
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Clicked at ({x}, {y})")
        print(f"BGR: {pixel_bgr}")
        print(f"HSV: {pixel_hsv}")
        
        # Draw a circle on the selected point
        cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", image_display)

# Load your image here
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = 'full_frame_20250601_065816_173.png'
image = cv2.imread(image_path)  # Replace with your image file
if image is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

image_display = image.copy()

cv2.imshow("Image", image_display)
cv2.setMouseCallback("Image", pick_color)

print("Click on the image to pick a pixel color. Press any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
