import cv2
import numpy as np

from object_detection.roi_extraction import extract_roi, get_bounding_box
from .pixel_finder import landmark_to_pixels
from .landmarks_constants import landmarks_per_finger


def landmarks_to_pixel_coordinates(image, landmarks) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [landmark_to_pixels(gray, landmarks.hand_landmarks[0], idx) for idx in
            range(len(landmarks.hand_landmarks[0]))]

def transform_point(point, matrix):
    # Convert point to homogeneous coordinates
    homogeneous_point = np.array([[point[0]], [point[1]], [1]])
    
    # Apply affine transformation
    transformed_point = np.squeeze(np.dot(matrix, homogeneous_point))
    
    # Return the transformed point in (x, y) format
    return (int(transformed_point[0]), int(transformed_point[1]))


def adjust_for_roi_crop(point, roi_center, roi_size):
    x_offset = int(roi_center[0] - roi_size[0] // 2)
    y_offset = int(roi_center[1] - roi_size[1] // 2)
    
    adjusted_x = point[0] - x_offset
    adjusted_y = point[1] - y_offset
    return np.array([adjusted_x, adjusted_y])


def transform_landmarks(finger_key, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask):
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


def find_object_width_at_row(image, row, col):
    """Helper function to compute the width of the object at a specific row."""
    left = col
    right = col
    while left > 0 and np.all(image[row, left] != [0, 0, 0]):
        left -= 1
    while right < image.shape[1] - 1 and np.all(image[row, right] != [0, 0, 0]):
        right += 1
    return (right - left)


def calculate_widths_and_distance(new_pip, new_dip, roi):
    # Compute pixel width of object at the row of new_pip
    pip_width = find_object_width_at_row(roi, new_pip[1], new_pip[0])
    
    # Compute pixel width of object at the row of new_dip
    dip_width = find_object_width_at_row(roi, new_dip[1], new_dip[0])

    # Compute vertical pixel distance between new_pip and new_dip
    vertical_distance = abs(new_dip[1] - new_pip[1])

    return pip_width, dip_width, vertical_distance