import cv2
import numpy as np

def get_bounding_box(image: np.ndarray, points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
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


def extract_roi(image, rect):
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