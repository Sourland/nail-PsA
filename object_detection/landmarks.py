import cv2
import numpy as np

from object_detection.roi_extraction import extract_roi, get_bounding_box
from .pixel_finder import landmark_to_pixels
from .landmarks_constants import landmarks_per_finger


def landmarks_to_pixel_coordinates(image: np.ndarray, landmarks: object) -> list:
    """
    Converts hand landmarks into pixel coordinates on the given image.

    Args:
        image (np.ndarray): The image on which the hand landmarks are detected. Should be in BGR format.
        landmarks (object): An object containing hand landmarks data, typically obtained from a hand tracking model.

    Returns:
        list: A list of tuples, each representing the (x, y) pixel coordinates of a hand landmark.

    Test Case:
        Assume `fake_image` is a numpy array representing an image and `fake_landmarks` is a mock object of landmarks.
        >>> fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> fake_landmarks = MockLandmarks()  # a mock landmarks object
        >>> pixels = landmarks_to_pixel_coordinates(fake_image, fake_landmarks)
        >>> type(pixels)
        <class 'list'>
        >>> type(pixels[0])
        <class 'tuple'>
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [landmark_to_pixels(gray, landmarks.hand_landmarks[0], idx) for idx in
            range(len(landmarks.hand_landmarks[0]))]


def transform_point(point: tuple, matrix: np.ndarray) -> tuple:
    """
    Applies an affine transformation to a point using the given transformation matrix.

    Args:
        point (tuple): The (x, y) coordinates of the point to be transformed.
        matrix (np.ndarray): The 3x3 affine transformation matrix.

    Returns:
        tuple: The transformed (x, y) coordinates.

    Test Case:
        >>> point = (10, 20)
        >>> matrix = np.array([[1, 0, 5], [0, 1, 10], [0, 0, 1]])
        >>> transform_point(point, matrix)
        (15, 30)
    """
    # Convert point to homogeneous coordinates
    homogeneous_point = np.array([[point[0]], [point[1]], [1]])
    
    # Apply affine transformation
    transformed_point = np.squeeze(np.dot(matrix, homogeneous_point))
    
    # Return the transformed point in (x, y) format
    return (int(transformed_point[0]), int(transformed_point[1]))


def adjust_for_roi_crop(point: tuple, roi_center: tuple, roi_size: tuple) -> np.ndarray:
    """
    Adjusts a point's coordinates based on the region of interest (ROI) cropping.

    Args:
        point (tuple): The (x, y) coordinates of the point.
        roi_center (tuple): The center (x, y) of the ROI.
        roi_size (tuple): The size (width, height) of the ROI.

    Returns:
        np.ndarray: The adjusted coordinates of the point as a numpy array.

    Test Case:
        >>> point = (150, 200)
        >>> roi_center = (100, 100)
        >>> roi_size = (50, 50)
        >>> adjust_for_roi_crop(point, roi_center, roi_size)
        array([100, 150])
    """
    x_offset = int(roi_center[0] - roi_size[0] // 2)
    y_offset = int(roi_center[1] - roi_size[1] // 2)
    
    adjusted_x = point[0] - x_offset
    adjusted_y = point[1] - y_offset
    return np.array([adjusted_x, adjusted_y])


def transform_landmarks(finger_key: str, landmarks_per_finger: dict, closest_points: list, landmark_pixels: list, rgb_mask: np.ndarray) -> tuple:
    """
    Transforms finger landmarks for a specified finger based on ROI and rotation matrix.

    Args:
        finger_key (str): The key identifying the finger.
        landmarks_per_finger (dict): A dictionary mapping fingers to their respective landmarks.
        closest_points (list): A list of points close to the finger landmarks.
        landmark_pixels (list): Pixel coordinates of the landmarks.
        rgb_mask (np.ndarray): The RGB mask of the image.

    Returns:
        tuple: A tuple containing the bounding rectangle, new positions of PIP and DIP joints, 
               and the ROI of the finger.
    """
    finger_roi_points = [item for idx in landmarks_per_finger[finger_key][1:] for item in closest_points[idx]]
    finger_roi_points.append(landmark_pixels[landmarks_per_finger[finger_key][0]])

    rect = get_bounding_box(rgb_mask, finger_roi_points)
    roi, rotation_matrix = extract_roi(rgb_mask, rect)

    dip = np.array(landmark_pixels[landmarks_per_finger[finger_key][1]])
    pip = np.array(landmark_pixels[landmarks_per_finger[finger_key][2]])

    # Rotate the landmarks
    rotated_pip = transform_point(pip, rotation_matrix)
    rotated_dip = transform_point(dip, rotation_matrix)

    # Map the landmarks to the resized image
    new_pip = adjust_for_roi_crop(rotated_pip, rect[0], rect[1])
    new_dip = adjust_for_roi_crop(rotated_dip, rect[0], rect[1])

    return rect, new_pip, new_dip, roi


def find_object_width_at_row(image: np.ndarray, row: int, col: int) -> int:
    """
    Computes the width of an object in the image at a specified row.

    Args:
        image (np.ndarray): The image containing the object.
        row (int): The row index at which to compute the width.
        col (int): The column index from where to start the scan.

    Returns:
        int: The computed width of the object at the specified row.

    Test Case:
        >>> image = np.array([[0, 0, 0], [255, 255, 255], [0, 0, 0]], dtype=np.uint8)
        >>> find_object_width_at_row(image, 1, 0)
        3
    """
    left = col
    right = col
    while left > 0 and np.all(image[row, left] != [0, 0, 0]):
        left -= 1
    while right < image.shape[1] - 1 and np.all(image[row, right] != [0, 0, 0]):
        right += 1
    return (right - left)


def calculate_widths_and_distance(new_pip: tuple, new_dip: tuple, roi: np.ndarray) -> tuple:
    """
    Calculates the widths at the PIP and DIP joints and the distance between them in an ROI.

    Args:
        new_pip (tuple): The (x, y) coordinates of the PIP joint in the ROI.
        new_dip (tuple): The (x, y) coordinates of the DIP joint in the ROI.
        roi (np.ndarray): The region of interest in the image.

    Returns:
        tuple: A tuple containing the widths at the PIP and DIP joints and the vertical distance between them.
    """
    # Compute pixel width of object at the row of new_pip
    pip_width = find_object_width_at_row(roi, new_pip[1], new_pip[0])
    
    # Compute pixel width of object at the row of new_dip
    dip_width = find_object_width_at_row(roi, new_dip[1], new_dip[0])

    # Compute vertical pixel distance between new_pip and new_dip
    vertical_distance = abs(new_dip[1] - new_pip[1])

    return pip_width, dip_width, vertical_distance