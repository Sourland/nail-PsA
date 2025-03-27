import numpy as np


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


def adjust_point_to_roi(point: tuple, roi_center: tuple, roi_size: tuple) -> np.ndarray:
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