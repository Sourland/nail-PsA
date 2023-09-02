import cv2
import numpy as np
from hand_landmarks import landmarks
from pixel_finder import landmark_to_pixels
from landmarks_constants import *

output_image = cv2.imread('../seg_mask.jpg')

landmarks_pixel = [landmark_to_pixels(cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY), landmarks, idx) for idx, landmark
                   in enumerate(landmarks)]
landmark_depth = [landmark.z for landmark in landmarks]
# Load image and find contour
image = output_image


def get_joint_thickness_euclidian(image, landmark_point):
    """
    Given an image and a landmark point, this function computes the thickness of the finger at that point.
    Args:
        image: The image containing the hand
        landmark_point: The landmark point of the finger
    Returns:
        thickness: The thickness of the finger at the given point
    """
    height, width, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour corresponds to the desired object
    contour = max(contours, key=cv2.contourArea)

    # Given point
    point = np.array(landmark_point)

    # Split the contour points into left and right based on the x-coordinate
    left_contour = contour[contour[:, 0, 0] < point[0]]
    right_contour = contour[contour[:, 0, 0] > point[0]]

    # Compute distances from the given point to each side of the contour
    distances_left = np.sqrt(np.sum((left_contour - point) ** 2, axis=2))
    distances_right = np.sqrt(np.sum((right_contour - point) ** 2, axis=2))

    # Get the closest point from the left and the right
    closest_left = left_contour[np.argmin(distances_left)]
    closest_right = right_contour[np.argmin(distances_right)]

    # Compute thickness
    thickness = np.linalg.norm((closest_left - closest_right))

    return thickness


landmarks_to_process = {
    "INDEX_FINGER_PIP": 6,
    "INDEX_FINGER_DIP": 7,
    "MIDDLE_FINGER_PIP": 10,
    "MIDDLE_FINGER_DIP": 11,
    "RING_FINGER_PIP": 14,
    "RING_FINGER_DIP": 15,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19
}

results = {}


def get_mean_pip_dip_distance(landmarks):
    landmarks = np.array(landmarks)
    index_pip_dip_distance = np.linalg.norm(landmarks[INDEX_FINGER_PIP] - landmarks[INDEX_FINGER_DIP])
    middle_finger_pip_dip_distance = np.linalg.norm(landmarks[MIDDLE_FINGER_PIP] - landmarks[MIDDLE_FINGER_DIP])
    ring_pip_dip_distance = np.linalg.norm(landmarks[RING_FINGER_PIP] - landmarks[RING_FINGER_DIP])
    pinky_pip_dip_distance = np.linalg.norm(landmarks[PINKY_PIP] - landmarks[PINKY_DIP])
    return np.mean([index_pip_dip_distance, middle_finger_pip_dip_distance, ring_pip_dip_distance, pinky_pip_dip_distance])


for landmark_name, landmark_index in landmarks_to_process.items():
    landmark_point = landmarks_pixel[landmark_index]
    thickness = get_joint_thickness_euclidian(output_image, landmark_point)

    results[landmark_name] = {
        "thickness": thickness
    }
mean_distance = get_mean_pip_dip_distance(landmarks_pixel)
for key in results.keys():
    print(f"{key} EFFECTIVE WIDTH: {results[key]['thickness'] / mean_distance}")

print()