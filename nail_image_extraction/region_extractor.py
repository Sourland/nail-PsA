import numpy as np
from mediapipe.tasks.python import vision
from landmark_extraction.utils.landmarks_constants import * 


def find_bounding_box(image, landmarks, landmark_name, which_hand):
    """
    Finds the bounding box around a specified landmark in the hand.

    Args:
        image (np.ndarray): The image containing the hand.
        landmarks (dict): A dictionary of landmarks.
        landmark_name (str): The name of the landmark for which the bounding box is to be found.
        which_hand (str): Indicates whether the hand is 'Right' or 'Left'.

    Returns:
        tuple: A tuple containing the coordinates of the top-left and bottom-right corners of the bounding box.

    Raises:
        AssertionError: If 'which_hand' is not 'Right' or 'Left'.

    Test Case:
        >>> image = np.zeros((100, 100), dtype=np.uint8)
        >>> landmarks = {'index': {'x': 0.5, 'y': 0.5}}
        >>> find_bounding_box(image, landmarks, 'index', 'Right')
        ((50, 50), (50, 50))
    """
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


    left, right = has_overstepped_boundaries(left, right, landmarks, neighbours, x, image)
    rect_width = rect_height = left + right
    top_left = (np.clip(x - rect_width // 2, 0, width), np.clip(y - rect_height // 2, 0, height))
    bottom_right = (np.clip(x + rect_width // 2, 0, width), np.clip(y + rect_height // 2, 0, height))

    return top_left, bottom_right


def has_overstepped_boundaries(left: int, right: int, landmarks: list[vision.HandLandmarkerResult], neighbors: dict, current_landmark: int, image: np.ndarray) -> tuple:
    """
    Adjusts the left and right boundaries based on neighboring landmarks.

    Args:
        left (int): Distance to the left boundary from the current landmark.
        right (int): Distance to the right boundary from the current landmark.
        landmarks (list[vision.HandLandmarkerResult]): List of hand landmarks.
        neighbors (dict): Dictionary of neighboring landmarks.
        current_landmark (int): The current landmark's X-coordinate.
        image (np.ndarray): The image containing the hand.

    Returns:
        tuple: A tuple of adjusted left and right boundary distances.

    Test Case:
        >>> image = np.zeros((100, 100), dtype=np.uint8)
        >>> landmarks = [{'index': vision.HandLandmarkerResult(x=0.5, y=0.5)}]
        >>> neighbors = {'left_neighbor': 'thumb', 'right_neighbor': 'middle'}
        >>> has_overstepped_boundaries(10, 10, landmarks, neighbors, 50, image)
        (10, 10)  # May vary based on actual landmark positions and neighbors
    """
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


def landmark_to_pixels(image: np.ndarray, landmarks: dict, landmark_name: str) -> tuple:
    """
    Converts a landmark's relative position to pixel coordinates.

    Args:
        image (np.ndarray): The image containing the hand.
        landmarks (dict): A dictionary of landmarks.
        landmark_name (str): The name of the landmark.

    Returns:
        tuple: The pixel coordinates (X, Y) of the landmark.

    Test Case:
        >>> image = np.zeros((100, 100), dtype=np.uint8)
        >>> landmarks = {'index': {'x': 0.5, 'y': 0.5}}
        >>> landmark_to_pixels(image, landmarks, 'index')
        (50, 50)
    """

    (height, width) = image.shape
    landmark_x = round(width * landmarks[landmark_name].x)  # X coordinate of the Mediapipe landmark # col
    landmark_y = round(height * landmarks[landmark_name].y)  # Y coordinate of the Mediapipe landmark # row
    return landmark_x, landmark_y


def crop_image(image: np.ndarray, top_left: tuple, bottom_right: tuple) -> np.ndarray:
    """
    Crops the image to the specified region.

    Args:
        image (np.ndarray): The image to be cropped.
        top_left (tuple): The top-left corner coordinates of the crop region.
        bottom_right (tuple): The bottom-right corner coordinates of the crop region.

    Returns:
        np.ndarray: The cropped region of the image.

    Test Case:
        >>> image = np.zeros((100, 100), dtype=np.uint8)
        >>> top_left = (20, 20)
        >>> bottom_right = (80, 80)
        >>> cropped_image = crop_image(image, top_left, bottom_right)
        >>> cropped_image.shape
        (60, 60)
    """
    x_top_left, y_top_left = top_left
    x_bottom_right, y_bottom_right = bottom_right

    return image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]

