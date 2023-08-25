import cv2
import numpy as np
from mediapipe.tasks.python import vision

from landmarks_constants import joint_neighbours_left_hand, joint_neighbours_right_hand


def find_bounding_box(image, landmarks, landmark_name, which_hand):
    assert which_hand in ["Right", "Left"]
    x, y = landmark_to_pixels(image, landmarks, landmark_name)
    height, width = image.shape
    left = right = 0

    if which_hand == "Right":
        neighbours = joint_neighbours_right_hand[landmark_name]
    else:
        neighbours = joint_neighbours_left_hand[landmark_name]

    # Iterate from the landmark's x-coordinate towards the left of the image
    for i in range(x, -1, -1):
        if image[y, i] == 0:
            left = np.abs(x - i)  # Return the coordinates of the nearest black pixel
            break

    # Iterate from the landmark's x-coordinate towards the right of the image
    for i in range(x, width):
        if image[y, i] == 0:
            right = np.abs(x - i)  # Return the coordinates of the nearest black pixel
            break
    left, right = has_overstepped_boundaries(left, right, landmarks, neighbours, x, image)
    rect_width = rect_height = left + right
    top_left = (np.clip(x - rect_width // 2, 0, width), np.clip(y - rect_height // 2, 0, height))
    bottom_right = (np.clip(x + rect_width // 2, 0, width), np.clip(y + rect_height // 2, 0, height))

    return top_left, bottom_right


def has_overstepped_boundaries(left: int,
                               right: int,
                               landmarks: list[vision.HandLandmarkerResult],
                               neighbors: dict,
                               current_landmark,
                               image: np.ndarray
                               ):
    if isinstance(neighbors, list):
        landmark_left_x, _ = landmark_to_pixels(image, landmarks, neighbors[0])
        landmark_right_x, _ = landmark_to_pixels(image, landmarks, neighbors[1])

        if (current_landmark - left) < landmark_left_x:
            left = np.abs(current_landmark - landmark_left_x) // 2

        if (current_landmark + right) > landmark_right_x:
            right = np.abs(current_landmark - landmark_right_x) // 2

        return left, right
    else:
        return left, right


def landmark_to_pixels(image, landmarks, landmark_name):
    (height, width) = image.shape
    landmark_x = round(width * landmarks[landmark_name].x)  # X coordinate of the Mediapipe landmark # col
    landmark_y = round(height * landmarks[landmark_name].y)  # Y coordinate of the Mediapipe landmark # row
    return landmark_x, landmark_y


def crop_image(image, top_left, bottom_right):
    x_top_left, y_top_left = top_left
    x_bottom_right, y_bottom_right = bottom_right

    return image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]

