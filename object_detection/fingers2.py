import os

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from .landmarks_constants import *
from .pixel_finder import landmark_to_pixels
from .utils import locate_hand_landmarks, draw_landmarks_on_image, resize_image
import subprocess
import segmentation


def get_segmentation_mask(image: np.ndarray, threshold: int = 11) -> np.ndarray:
    # Check if the image has three channels
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels. Received image with shape {}.".format(image.shape))

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Generate the mask
    mask = np.where(grayscale_image > threshold, 255, 0)

    return mask.astype(np.uint8)
    

def extract_contour(image: np.ndarray) -> np.ndarray:
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if not contours:
        return

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Return the largest contour by area
    return np.squeeze(contours[0])



def landmarks_to_pixel_coordinates(image, landmarks) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [landmark_to_pixels(gray, landmarks.hand_landmarks[0], idx) for idx in
            range(len(landmarks.hand_landmarks[0]))]


def closest_contour_point(landmarks, contour):
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


def draw_landmarks_and_connections(image, landmarks, closest_points):
    # Draw landmarks as red circles
    for landmark in landmarks:
        cv2.circle(image, tuple(map(int, landmark)), 3, (0, 0, 255), -1)

    # Draw closest contour points as blue circles and lines connecting them in green
    for landmark, (left_closest_point, right_closest_point) in zip(landmarks, closest_points):
        cv2.circle(image, tuple(map(int, left_closest_point)), 3, (255, 0, 0), -1)
        cv2.circle(image, tuple(map(int, right_closest_point)), 3, (255, 0, 0), -1)
        cv2.line(image, tuple(map(int, landmark)), tuple(map(int, left_closest_point)), (0, 255, 0), 1)
        cv2.line(image, tuple(map(int, landmark)), tuple(map(int, right_closest_point)), (0, 255, 0), 1)

    return image


def get_bounding_box(image: np.ndarray, points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    # Convert the list of points to a suitable format for cv2.minAreaRect
    rect_points = np.array(points).reshape(-1, 1, 2).astype(int)

    # Compute the rotated bounding box
    rect = cv2.minAreaRect(rect_points)
    (center, (width, height), theta) = rect

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


def save_roi_image(roi, path):
    if roi.size > 0 and roi is not None:
        cv2.imwrite(path, roi)
    else:
        print(f"Warning: The ROI image is empty or None. Skipping save operation for finger {os.path.basename(path)}.")

def process_finger(finger_key, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask, PATH, OUTPUT_DIR):
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

    # Compute pixel width of object at the row of new_pip
    pip_width = find_object_width_at_row(roi, new_pip[1], new_pip[0])
    
    # Compute pixel width of object at the row of new_dip
    dip_width = find_object_width_at_row(roi, new_dip[1], new_dip[0])

    # Compute vertical pixel distance between new_pip and new_dip
    vertical_distance = abs(new_dip[1] - new_pip[1])

    effective_pip_width = pip_width / vertical_distance
    effective_dip_width = dip_width / vertical_distance

    # Return pip_width, dip_width, and vertical_distance
    return effective_pip_width, effective_dip_width


def find_object_width_at_row(image, row, col):
    """Helper function to compute the width of the object at a specific row."""
    left = col
    right = col
    while left > 0 and np.all(image[row, left] != [0, 0, 0]):
        left -= 1
    while right < image.shape[1] - 1 and np.all(image[row, right] != [0, 0, 0]):
        right += 1
    return (right - left)

    

def process_image(PATH, MASKS_OUTPUT_DIR, FINGER_OUTPUT_DIR, NAIL_OUTPUT_DIR):
    image, landmarks = locate_hand_landmarks(PATH, "hand_landmarker.task")
    
    if not landmarks.hand_landmarks:
        print(f"Warning: No landmarks detected for {os.path.basename(PATH)}")
        return

    landmark_pixels = landmarks_to_pixel_coordinates(image, landmarks)
    enhanced_image = image
    padding = 250
    
    try:
        result = segmentation.bg.remove(data=enhanced_image)
    except ValueError as e:
        print(f"Caught a value error: {e} on image {os.path.basename(PATH)}")
        return
        
    result = cv2.copyMakeBorder(result, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    landmark_pixels = [(x+padding, y+padding) for x, y in landmark_pixels]
    
    seg_mask = get_segmentation_mask(result)
    OUTPUT_PATH_MASK = os.path.join(MASKS_OUTPUT_DIR, "seg_" + os.path.basename(PATH))
    cv2.imwrite(OUTPUT_PATH_MASK, seg_mask)

    contour = extract_contour(seg_mask)
    if contour is None or len(contour.shape) == 1:
        print(f"Warning: The contour is empty. Skipping {os.path.basename(PATH)}.")
        return

    rgb_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
    closest_points = closest_contour_point(landmark_pixels, contour)
    effective_pip_widths, effective_dip_widths = [], []
    for key in ['INDEX', 'MIDDLE', 'RING', 'PINKY']:
        pip_width, dip_width = process_finger(key, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask, PATH, FINGER_OUTPUT_DIR)
        effective_pip_widths.append(pip_width)
        effective_dip_widths.append(dip_width)

    return effective_pip_widths, effective_dip_widths
