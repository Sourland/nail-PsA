import cv2
import numpy as np

def get_bounding_box(image: np.ndarray, points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the minimum area rotated bounding box for a set of points.

    Args:
        image (np.ndarray): The image on which the points are located.
        points (list[np.ndarray]): A list of points, each point being a numpy array of x, y coordinates.

    Returns:
        tuple: A tuple containing the center, size (width, height), and rotation angle (theta) of the bounding box.

    Test Case:
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> points = [np.array([30, 40]), np.array([50, 60]), np.array([70, 80])]
        >>> center, size, theta = get_bounding_box(image, points)
        >>> center, size, theta  # Outputs may vary based on input points
        ((50.0, 60.0), (40.0, 28.2842712474619), -45.0)
    """
   # Convert the list of points to a suitable format for cv2.minAreaRect
    rect_points = np.array(points).reshape(-1, 1, 2).astype(int)

    # Compute the rotated bounding box
    rect = cv2.minAreaRect(rect_points)
    (center, (width, height), theta) = rect

    # Draw the rectangle onto the blank image
    # cv2.polylines(image, [np.int0(cv2.boxPoints(rect))], True, (0, 255, 0), 2)
    # cv2.imwrite(OUTPUT_PATH, image)
    # Ensure the rectangle is in portrait orientation
    if width > height:
        width, height = height, width
        theta -= 90  # Adjust the rotation

    return (center, (width, height), theta)


def extract_roi(image: np.ndarray, rect: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts a region of interest (ROI) from an image based on a given rotated rectangle.

    Args:
        image (np.ndarray): The image from which the ROI is to be extracted.
        rect (tuple): A tuple describing the rotated rectangle (center, (width, height), angle).

    Returns:
        tuple: A tuple containing the ROI as an image and the rotation matrix used for the extraction.

    Test Case:
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> rect = ((50, 50), (40, 20), 45)
        >>> roi, rotation_matrix = extract_roi(image, rect)
        >>> roi.shape  # Outputs may vary based on rect dimensions
        (23, 43, 3)
    """
    # Scale up the width and height by 15% for a margin.
    center, size, theta = rect
    width, height = size
    width += width * 0.15
    height += height * 0.15
    # Obtain the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1)

    # Perform the affine transformation on the padded image
    warped_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), borderValue=(0, 0, 0))

    # Extract the ROI
    x, y = int(center[0] - width // 2), int(center[1] - height // 2)
    roi = warped_image[y:y + int(height), x:x + int(width)]
    return roi, rotation_matrix
