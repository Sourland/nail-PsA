import os
import cv2
import numpy as np
from object_detection.landmarks import landmarks_to_pixel_coordinates
import segmentation.bg
from object_detection.utils import locate_hand_landmarks, get_segmentation_mask
from landmarks_constants import *
import mediapipe.tasks.python.vision as vision
from mediapipe.tasks.python.vision import HandLandmarkerResult


class RegionExtractor:
    def __init__(self, detector_path="hand_landmarker.task", image_output_dir="./output"):
        self.detector_path = detector_path
        self.image_output_dir = image_output_dir
        self.image = None

    def load_image(self, path):
        self.image, self.landmarks = locate_hand_landmarks(path, self.detector_path)
        if not self.landmarks.hand_landmarks:
            print(f"Warning: No landmarks detected for {os.path.basename(path)}")
            return
        self.landmark_pixels = landmarks_to_pixel_coordinates(self.image, self.landmarks)

    def find_bounding_box(self, image, landmarks, landmark_name, which_hand):
        assert which_hand in ["Right", "Left"]
        x, y = landmark_pixels[landmark_name]
        height, width = image.shape
        left = right = 0

        if which_hand == "Right":
            neighbours = joint_neighbours_right_hand[landmark_name]
        else:
            neighbours = joint_neighbours_left_hand[landmark_name]

        # Iterate from the landmark's x-coordinate towards the left of the image
        for i in range(x, -1, -1):
            if i < 0 or i >= width or y < 0 or y >= height:
                # Skip this iteration if index out of bounds
                continue

            if image[y, i] == 0:
                left = np.abs(x - i)  # Return the coordinates of the nearest black pixel
                break

        # Iterate from the landmark's x-coordinate towards the right of the image
        for i in range(x, width):
            if i < 0 or i >= width or y < 0 or y >= height:
                # Skip this iteration if index out of bounds
                continue

            if image[y, i] == 0:
                right = np.abs(x - i)  # Return the coordinates of the nearest black pixel
                break

        # left, right = has_overstepped_boundaries(left, right, landmarks, neighbours, x, image)
        rect_width = rect_height = left + right
        top_left = (np.clip(x - rect_width // 2, 0, width), np.clip(y - rect_height // 2, 0, height))
        bottom_right = (np.clip(x + rect_width // 2, 0, width), np.clip(y + rect_height // 2, 0, height))

        return top_left, bottom_right


    def crop_image(self, top_left, bottom_right):
        """
        Crops the image to the specified region.
        """
        x_top_left, y_top_left = top_left
        x_bottom_right, y_bottom_right = bottom_right
        return self.image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]


def temp_func(path):
    image, landmarks = locate_hand_landmarks(path, "hand_landmarker.task")
    if not landmarks.hand_landmarks:
        print(f"Warning: No landmarks detected for {os.path.basename(path)}")
        return [0, 0, 0, 0], [0, 0, 0, 0]
    landmark_pixels = landmarks_to_pixel_coordinates(image, landmarks)
    enhanced_image = image

    try:
        result = segmentation.bg.remove(data=enhanced_image)
    except ValueError as e:
        print(f"Caught a value error: {e} on image {os.path.basename(path)}")
        return [0, 0, 0, 0], [0, 0, 0, 0]
    
    seg_mask = get_segmentation_mask(result)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return seg_mask, image, landmark_pixels


image_path = "dataset/hands/swolen/hand2.png"
seg_mask, image, landmark_pixels = temp_func(image_path)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.namedWindow("Segmentation Mask", cv2.WINDOW_NORMAL)
cv2.imshow("Segmentation Mask", seg_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
