import cv2
import numpy as np

from object_detection.landmarks_constants import *
def extract_contour(image: np.ndarray) -> np.ndarray:
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


def get_left_and_right_contour_points(landmark: tuple, contour: np.ndarray) -> tuple:
    """
    Splits a contour into left and right parts based on a given landmark.

    Args:
        landmark (tuple): The landmark (x, y) used as the splitting point.
        contour (np.ndarray): The contour to be split.

    Returns:
        tuple: A tuple containing two numpy arrays, the left and right parts of the contour.

    Test Case:
        >>> landmark = (50, 50)
        >>> contour = np.array([[40, 40], [60, 60], [80, 80]])
        >>> left_contour, right_contour = get_left_and_right_contour_points(landmark, contour)
        >>> type(left_contour), type(right_contour)
        (<class 'numpy.ndarray'>, <class 'numpy.ndarray'>)
    """

    left_contour = contour[(contour[:, 0] <= landmark[0])]
    right_contour = contour[(contour[:, 0] > landmark[0])]

    return left_contour, right_contour


def reorient_contour(contour: np.ndarray, orientation: str = 'clockwise') -> np.ndarray:
    """
    Reorients a contour to a specified orientation (clockwise or counterclockwise).

    Args:
        contour (np.ndarray): The contour to be reoriented.
        orientation (str, optional): The desired orientation ('clockwise' or 'counterclockwise'). Default is 'clockwise'.

    Returns:
        np.ndarray: The reoriented contour.

    Test Case:
        >>> contour = np.array([[10, 10], [20, 20], [10, 20]])
        >>> reoriented_contour = reorient_contour(contour)
        >>> type(reoriented_contour)
        <class 'numpy.ndarray'>
    """

    # Calculate the contour area, and consider the orientation
    area = cv2.contourArea(contour, oriented=True)
    
    # Check if the contour is already in the desired orientation
    if (area < 0 and orientation == 'clockwise') or (area > 0 and orientation == 'counterclockwise'):
        # The contour is already in the desired orientation, no change needed
        return contour
    else:
        # The contour is in the opposite orientation, so reverse it
        return np.flipud(contour)


def create_finger_contour(tip_proxy: tuple, left_contour: np.ndarray, right_contour: np.ndarray, closest_mcp_left: tuple, closest_mcp_right: tuple) -> np.ndarray:
    """
    Creates a contour for a finger by combining segments from the left and right contours and a connecting line.

    Args:
        tip_proxy (tuple): The proxy point for the fingertip.
        left_contour (np.ndarray): The left part of the contour.
        right_contour (np.ndarray): The right part of the contour.
        closest_mcp_left (tuple): The closest point on the left contour to the MCP joint.
        closest_mcp_right (tuple): The closest point on the right contour to the MCP joint.

    Returns:
        np.ndarray: The contour of the finger.

    Test Case:
        >>> tip_proxy = (50, 50)
        >>> left_contour = np.array([[40, 40], [45, 45], [50, 50]])
        >>> right_contour = np.array([[50, 50], [55, 55], [60, 60]])
        >>> closest_mcp_left, closest_mcp_right = (40, 40), (60, 60)
        >>> finger_contour = create_finger_contour(tip_proxy, left_contour, right_contour, closest_mcp_left, closest_mcp_right)
        >>> type(finger_contour)
        <class 'numpy.ndarray'>
    """
    # Get the indices of the closest points on the left and right contours
    idx_mcp_left = np.where(np.all(left_contour == closest_mcp_left, axis=1))[0][0]
    idx_mcp_right = np.where(np.all(right_contour == closest_mcp_right, axis=1))[0][0]
    idx_tip_proxy = np.where(np.all(np.vstack((left_contour, right_contour)) == tip_proxy, axis=1))[0][0]
    
    # Get the segments of the left and right contours
    left_segment = left_contour[:idx_mcp_left]
    right_segment = right_contour[idx_mcp_right:idx_tip_proxy+1]
    
    # Ensure segments are in correct order
    left_segment = left_segment if (left_segment[-1] == tip_proxy).all() else left_segment[::-1]
    right_segment = right_segment if (right_segment[0] == tip_proxy).all() else right_segment[::-1]
    
    # Create the connecting line between MCP points
    connecting_line = np.array([closest_mcp_left, closest_mcp_right])
    
    # Concatenate the segments and the connecting line to form the full contour
    full_contour = np.concatenate((left_segment, right_segment, connecting_line))

    return full_contour
