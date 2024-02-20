import numpy as np
from object_detection.landmarks import adjust_for_roi_crop, transform_point
from object_detection.roi_extraction import extract_roi


def is_point_inside_rect(image: np.ndarray, point: tuple, rect: tuple) -> bool:
    """
    Determines if a point is inside a given rectangle after applying rotation and cropping adjustments.

    Args:
        image (np.ndarray): The image containing the rectangle.
        point (tuple): The (x, y) coordinates of the point.
        rect (tuple): A tuple describing the rectangle, containing the center, size, and rotation angle.

    Returns:
        bool: True if the point is inside the rectangle, False otherwise.

    Test Case:
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> point = (10, 10)
        >>> rect = ((50, 50), (20, 20), 0)
        >>> is_point_inside_rect(image, point, rect)
        False
    """

    (center, (width, height), theta) = rect
    roi, rotation_matrix = extract_roi(image, rect)
    rotated_point = transform_point(point, rotation_matrix)
    adjusted_point = adjust_for_roi_crop(rotated_point, rect[0], rect[1])

    # Check if the adjusted point is within the rectangle's boundaries
    half_width, half_height = width / 2, height / 2
    return (-half_width <= adjusted_point[0] <= half_width) and (-half_height <= adjusted_point[1] <= half_height)


def process_neighbor_finger(this_pip: tuple, this_dip: tuple, neighbor_key: str, landmarks_per_finger: dict, landmark_pixels: list, rect: tuple, rgb_mask: np.ndarray, roi: np.ndarray, used_fingers: list) -> np.ndarray:
    """
    Processes the neighboring finger and adjusts the ROI if necessary.

    Args:
        this_pip (tuple): The (x, y) coordinates of the PIP joint of the current finger.
        this_dip (tuple): The (x, y) coordinates of the DIP joint of the current finger.
        neighbor_key (str): The key identifying the neighboring finger.
        landmarks_per_finger (dict): A dictionary mapping fingers to their respective landmarks.
        landmark_pixels (list): Pixel coordinates of the landmarks.
        rect (tuple): A tuple describing the rectangle, containing the center, size, and rotation angle.
        rgb_mask (np.ndarray): The RGB mask of the image.
        roi (np.ndarray): The region of interest in the image.
        used_fingers (list): A list of fingers already processed.

    Returns:
        np.ndarray: The adjusted ROI if the neighboring finger is relevant, or the original ROI.
    """
    neighbor_pip = np.array(landmark_pixels[landmarks_per_finger[neighbor_key][1]])
    neighbor_dip = np.array(landmark_pixels[landmarks_per_finger[neighbor_key][2]])

    if is_point_inside_rect(rgb_mask, neighbor_dip, rect) or is_point_inside_rect(rgb_mask, neighbor_pip, rect):
        used_fingers.append(neighbor_key)
        return adjust_roi_for_neighbor(this_pip, this_dip, neighbor_pip, neighbor_dip, rect, roi)
    return roi


def adjust_roi_for_neighbor(this_pip: tuple, this_dip: tuple, neighbor_pip: tuple, neighbor_dip: tuple, rect: tuple, roi: np.ndarray) -> np.ndarray:
    """
    Adjusts the region of interest (ROI) based on the current and neighboring finger positions.

    Args:
        this_pip (tuple): The (x, y) coordinates of the PIP joint of the current finger.
        this_dip (tuple): The (x, y) coordinates of the DIP joint of the current finger.
        neighbor_pip (tuple): The (x, y) coordinates of the PIP joint of the neighboring finger.
        neighbor_dip (tuple): The (x, y) coordinates of the DIP joint of the neighboring finger.
        rect (tuple): A tuple describing the rectangle, containing the center, size, and rotation angle.
        roi (np.ndarray): The region of interest in the image.

    Returns:
        np.ndarray: The adjusted ROI based on the positions of the PIP and DIP joints.
    """
    # Transform the neighbor landmarks
    transformed_neighbor_pip = transform_point(neighbor_pip, rect[1])
    transformed_neighbor_dip = transform_point(neighbor_dip, rect[2])

    # Compute middle points and adjust ROI
    pip_middle = (this_pip + transformed_neighbor_pip) // 2
    dip_middle = (this_dip + transformed_neighbor_dip) // 2

    # Calculate line slope and intercept
    slope, intercept = calculate_line_slope_intercept(pip_middle, dip_middle)

    # Adjust the ROI based on pip and dip positions
    return adjust_roi_based_on_line(roi, pip_middle, slope, intercept)


def calculate_line_slope_intercept(pip_middle: tuple, dip_middle: tuple) -> tuple:
    """
    Calculates the slope and intercept of the line formed by the middle points of PIP and DIP joints.

    Args:
        pip_middle (tuple): The middle point of the PIP joint.
        dip_middle (tuple): The middle point of the DIP joint.

    Returns:
        tuple: A tuple containing the slope and intercept of the line.

    Test Case:
        >>> pip_middle = (10, 10)
        >>> dip_middle = (20, 20)
        >>> calculate_line_slope_intercept(pip_middle, dip_middle)
        (1.0, 0.0)
    """
    slope = (pip_middle[1] - dip_middle[1]) / (pip_middle[0] - dip_middle[0])
    intercept = pip_middle[1] - slope * pip_middle[0]
    return slope, intercept


def adjust_roi_based_on_line(roi: np.ndarray, pip_middle: tuple, slope: float, intercept: float) -> np.ndarray:
    """
    Adjusts the ROI based on the position of the PIP joint and the slope of the line.

    Args:
        roi (np.ndarray): The region of interest in the image.
        pip_middle (tuple): The middle point of the PIP joint.
        slope (float): The slope of the line connecting PIP and DIP joints.
        intercept (float): The y-intercept of the line.

    Returns:
        np.ndarray: The adjusted ROI.

    Test Case:
        >>> roi = np.ones((100, 100), dtype=np.uint8)
        >>> pip_middle = (50, 50)
        >>> slope = 1.0
        >>> intercept = 0
        >>> adjust_roi_based_on_line(roi, pip_middle, slope, intercept)
        # This will zero out either the left or right half of the ROI based on pip_middle's position
    """
    pip_left = pip_middle[0] < (slope * pip_middle[1] + intercept)
    if pip_left:
        roi[:, int(pip_middle[0]):] = 0
    else:
        roi[:, :int(pip_middle[0])] = 0
    return roi
