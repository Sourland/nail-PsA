import cv2
import numpy as np
from landmark_extraction.utils.landmarks_constants import *


def get_largest_contour(image: np.ndarray) -> np.ndarray:
    """
    Extracts the largest contour from a binary image.

    Args:
        image (np.ndarray): A binary image from which contours are to be extracted.

    Returns:
        np.ndarray: The largest contour found in the image.

    Raises:
        None

    Test Case:
        >>> image = np.zeros((100, 100), dtype=np.uint8)
        >>> cv2.rectangle(image, (30, 30), (70, 70), 255, -1)
        >>> contour = extract_contour(image)
        >>> type(contour)
        <class 'numpy.ndarray'>
        >>> contour.shape[0] > 0  # The contour should have more than 0 points
        True
    """

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if not contours:
        return

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Return the largest contour by area
    return np.squeeze(contours[0])

def closest_contour_point(landmarks: list, contour: np.ndarray) -> list:
    """
    Finds the closest points on a contour to each given landmark.

    Args:
        landmarks (list): A list of landmarks, where each landmark is a tuple (x, y).
        contour (np.ndarray): The contour to which the closest points are to be found.

    Returns:
        list: A list of tuples, where each tuple contains the closest points on the left and right of each landmark.

    Test Case:
        >>> landmarks = [(50, 50), (70, 70)]
        >>> contour = np.array([[40, 40], [60, 60], [80, 80]])
        >>> closest_points = closest_contour_point(landmarks, contour)
        >>> len(closest_points)
        2
        >>> all(len(point) == 2 for point in closest_points)
        True
    """

    closest_points = []

    for ctr, landmark in enumerate(landmarks):
        if ctr in [THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]:
            left_contour = contour[(contour[:, 0] <= landmark[0]) & (contour[:, 1] >= landmark[1])]
            right_contour = contour[(contour[:, 0] > landmark[0]) & (contour[:, 1] >= landmark[1])]
        else:
            left_contour = contour[(contour[:, 0] <= landmark[0])]
            right_contour = contour[(contour[:, 0] > landmark[0])]


        # For empty contours (when landmarks are on extreme left or right), use the entire contour
        if len(left_contour) == 0:
            left_contour = contour
        if len(right_contour) == 0:
            right_contour = contour

        # Find closest point on the left of the landmark
        left_distances = np.sqrt(np.sum((left_contour - landmark)**2, axis=1))
        left_min_index = np.argmin(left_distances)
        left_closest_point = tuple(left_contour[left_min_index])

        # Find closest point on the right of the landmark
        right_distances = np.sqrt(np.sum((right_contour - landmark)**2, axis=1))
        right_min_index = np.argmin(right_distances)
        right_closest_point = tuple(right_contour[right_min_index])

        closest_points.append((left_closest_point, right_closest_point))

    return closest_points