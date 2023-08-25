import cv2
import numpy as np

from hand_landmarks import landmarks, output_image, which_hand
from pixel_finder import landmark_to_pixels
from landmarks_constants import *


def calculate_vector(landmark1, landmark2):
    """
    Calculate the vector from landmark1 to landmark2.
    """
    return np.array([landmark2.x - landmark1.x, landmark2.y - landmark1.y, landmark2.z - landmark1.z])


def average_vector(vectors):
    """
    Calculate the average vector from a list of vectors.
    """
    avg = [sum(component) / len(vectors) for component in zip(*vectors)]
    return tuple(avg)


def get_finger_vectors(landmarks, finger_name):
    """
    Compute the vectors for a given finger.
    Args:
        landmarks:
        finger_name:

    Returns:

    """
    assert finger_name in ["INDEX", "MIDDLE", "RING", "PINKY", "THUMB"]
    if finger_name == "INDEX":
        return average_vector([
            calculate_vector(landmarks[INDEX_FINGER_MCP], landmarks[INDEX_FINGER_PIP]),
            calculate_vector(landmarks[INDEX_FINGER_PIP], landmarks[INDEX_FINGER_DIP]),
            calculate_vector(landmarks[INDEX_FINGER_DIP], landmarks[INDEX_FINGER_TIP])
        ])
    elif finger_name == "MIDDLE":
        return average_vector([
            calculate_vector(landmarks[MIDDLE_FINGER_MCP], landmarks[MIDDLE_FINGER_PIP]),
            calculate_vector(landmarks[MIDDLE_FINGER_PIP], landmarks[MIDDLE_FINGER_DIP]),
            calculate_vector(landmarks[MIDDLE_FINGER_DIP], landmarks[MIDDLE_FINGER_TIP])
        ])
    elif finger_name == "RING":
        return average_vector([
            calculate_vector(landmarks[RING_FINGER_MCP], landmarks[RING_FINGER_PIP]),
            calculate_vector(landmarks[RING_FINGER_PIP], landmarks[RING_FINGER_DIP]),
            calculate_vector(landmarks[RING_FINGER_DIP], landmarks[RING_FINGER_TIP])
        ])

    elif finger_name == "PINKY":
        return average_vector([
            calculate_vector(landmarks[PINKY_MCP], landmarks[PINKY_PIP]),
            calculate_vector(landmarks[PINKY_PIP], landmarks[PINKY_DIP]),
            calculate_vector(landmarks[PINKY_DIP], landmarks[PINKY_TIP])
        ])
    elif finger_name == "THUMB":
        return average_vector([calculate_vector(landmarks[THUMB_CMC], landmarks[THUMB_MCP]),
                               calculate_vector(landmarks[THUMB_MCP], landmarks[THUMB_IP]),
                               calculate_vector(landmarks[THUMB_IP], landmarks[THUMB_TIP])])


def normalize(vector):
    """
    Normalize a 3D vector.
    """
    norm = np.linalg.norm(vector)
    return vector[0] / norm, vector[1] / norm, vector[2] / norm


def follow_direction(start_point, direction_vector, image_shape):
    """
    Starts from 'start_point' and follows 'direction_vector' until the boundary of 'image_shape' is hit.
    Returns the last valid point inside the image.

    Args:
        start_point (tuple): Starting point (x, y).
        direction_vector (tuple): Direction vector (dx, dy, dz). Note: dz is ignored for 2D images.
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        tuple: Final point (x, y) before hitting the image boundary.
    """
    prev_x = prev_y = 0
    height, width = image_shape

    # Normalize the direction vector
    normalized_vector = normalize(direction_vector)
    dx, dy, _ = normalized_vector  # We're ignoring the z dimension for 2D images

    # Starting coordinates
    x, y = start_point

    # Move in the direction until hitting the boundary
    while 0 <= x < width and 0 <= y < height:
        prev_x, prev_y = x, y
        x += dx
        y += dy

    return int(prev_x), int(prev_y)


def crop_skewed_rectangle(img, pts):
    """
    Extracts and corrects a skewed rectangle from an image.

    Args:
    - img (np.ndarray): The source image.
    - pts (np.ndarray): The 4 corner points of the skewed rectangle.

    Returns:
    - np.ndarray: The cropped and corrected image.
    """

    # Sort the points in counter-clockwise order starting from the top-left corner
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Determine the width and height of the new rectangle
    widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
    widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
    heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Get the perspective transformation matrix and warp the image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped


vectors = []
for finger in ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]:
    vectors.append(get_finger_vectors(landmarks, finger))


def estimate_joint_thickness(landmarks, output_image):
    """

    Args:
        landmarks:
        output_image:

    Returns:

    """
    landmarks_pixel = [landmark_to_pixels(cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY), landmarks, idx) for
                       idx, landmark in enumerate(landmarks)]
    return landmarks_pixel


landmarks_pixel = [landmark_to_pixels(cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY), landmarks, idx) for idx, landmark
                   in enumerate(landmarks)]

if which_hand == "Right":
    # Index Finger
    index_finger_bottom_right = (np.array(landmarks_pixel[THUMB_IP]) + np.array(landmarks_pixel[INDEX_FINGER_MCP])) // 2
    index_finger_bottom_left = (np.array(landmarks_pixel[INDEX_FINGER_MCP]) + np.array(
        landmarks_pixel[MIDDLE_FINGER_MCP])) // 2
    index_finger_top_right = follow_direction(index_finger_bottom_right, vectors[1], output_image.shape[:2])
    index_finger_top_left = follow_direction(index_finger_bottom_left, vectors[2], output_image.shape[:2])
    points = np.array(
        [index_finger_top_left, index_finger_bottom_left, index_finger_bottom_right, index_finger_top_right]
    )
    cropped_image = crop_skewed_rectangle(output_image, points)
    cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cropped", 300, 700)
    cv2.imshow("Cropped", cropped_image)
    cv2.waitKey(0)

    # Middle Finger
    middle_finger_bottom_right = (np.array(landmarks_pixel[INDEX_FINGER_MCP]) + np.array(
        landmarks_pixel[MIDDLE_FINGER_MCP])) // 2
    middle_finger_bottom_left = (np.array(landmarks_pixel[MIDDLE_FINGER_MCP]) + np.array(
        landmarks_pixel[RING_FINGER_MCP])) * (1 / 3)
    middle_finger_top_right = follow_direction(middle_finger_bottom_right, vectors[2], output_image.shape[:2])
    middle_finger_top_left = follow_direction(middle_finger_bottom_left, vectors[2], output_image.shape[:2])
    points = np.array(
        [middle_finger_top_left, middle_finger_bottom_left, middle_finger_bottom_right, middle_finger_top_right]
    )
    cropped_image = crop_skewed_rectangle(output_image, points)
    cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cropped", 300, 700)
    cv2.imshow("Cropped", cropped_image)
    cv2.waitKey(0)

    # Ring Finger
    ring_finger_bottom_right = (np.array(landmarks_pixel[MIDDLE_FINGER_MCP]) + np.array(
        landmarks_pixel[RING_FINGER_MCP])) // 2
    ring_finger_bottom_left = (np.array(landmarks_pixel[RING_FINGER_MCP]) + np.array(landmarks_pixel[PINKY_MCP])) // 2
    ring_finger_top_right = follow_direction(ring_finger_bottom_right, vectors[3], output_image.shape[:2])
    ring_finger_top_left = follow_direction(ring_finger_bottom_left, vectors[3], output_image.shape[:2])
    points = np.array(
        [ring_finger_top_left, ring_finger_bottom_left, ring_finger_bottom_right, ring_finger_top_right]
    )
    cropped_image = crop_skewed_rectangle(output_image, points)
    cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cropped", 300, 700)
    cv2.imshow("Cropped", cropped_image)
    cv2.waitKey(0)

    # Pinky Finger
    pinky_finger_limit_right = landmarks_pixel[RING_FINGER_MCP][0] - landmarks_pixel[PINKY_MCP][0] // 2
    pinky_finger_limit_left = 0
    pinky_finger = output_image[:landmarks_pixel[PINKY_MCP][1], pinky_finger_limit_left:pinky_finger_limit_right]

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 300, 700)
    cv2.imshow('image', pinky_finger)
    cv2.waitKey(0)
