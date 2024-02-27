import math
import os

import cv2
import numpy as np
from utils.contours import get_largest_contour, closest_contour_point
from hand_landmarker import adjust_point_to_roi, find_object_width_at_row, transform_point
from utils.roi_helpers import extract_roi, get_bounding_box_from_points
from object_detection.landmarks_constants import *

def is_inside_rotated_rect(rotated_point, rect):
    """
    Checks if a point is inside a rotated rectangle.

    Args:
        rotated_point (tuple): The point to check (x, y).
        rect (tuple): The rotated rectangle (center, size, angle).

    Returns:
        bool: True if the point is inside the rectangle, False otherwise.

    Test Case:
        >>> rotated_point = (55, 55)
        >>> rect = ((50, 50), (40, 20), 0)
        >>> is_inside_rotated_rect(rotated_point, rect)
        True
    """
    (center, (width, height), _) = rect
    x_min = center[0] - width / 2
    y_min = center[1] - height / 2
    x_max = center[0] + width / 2
    y_max = center[1] + height / 2

    return x_min <= rotated_point[0] <= x_max and y_min <= rotated_point[1] <= y_max


def calculate_transformed_image_shape(image_shape, rotation_matrix):
    """
    Processes a finger to compute measurements and adjust images.

    Args:
        finger_key (str): The key identifying the finger.
        landmarks_per_finger (dict): A dictionary mapping fingers to their respective landmarks.
        closest_points (list): A list of closest contour points for each landmark.
        landmark_pixels (list): Pixel coordinates of the landmarks.
        rgb_mask (np.ndarray): The RGB mask of the image.
        PATH (str): The file path of the input image.
        FINGER_OUTPUT_DIR (str): The directory where output images are saved.

    Returns:
        None

    Test Case:
        # Due to the complexity and dependency on external files and data, specific test cases should be created based on the actual scenario.
    """
    height, width = image_shape
    # Corners of the original image
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    # Add a column of 1s for affine transformation
    ones = np.ones(shape=(len(corners), 1))
    corners_ones = np.hstack([corners, ones])

    # Transform the corners
    transformed_corners = rotation_matrix.dot(corners_ones.T).T

    # Calculate new bounding box
    x_coords, y_coords = zip(*transformed_corners[:, :2])
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # New dimensions
    new_width, new_height = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))

    return new_height, new_width

def add_padding(image, landmark_pixels, padding_size=250):
    image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
    landmark_pixels = [(x + padding_size, y + padding_size) for x, y in landmark_pixels]
    return image, landmark_pixels
    
