import math
import os

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from object_detection.contour_extraction import closest_contour_point, create_finger_contour, extract_contour, get_left_and_right_contour_points, reorient_contour
from object_detection.landmarks import adjust_for_roi_crop, calculate_widths_and_distance, find_object_width_at_row, landmarks_to_pixel_coordinates, transform_landmarks, transform_point
from object_detection.roi_extraction import extract_roi, get_bounding_box
from object_detection.segmentation import get_segmentation_mask
from .landmarks_constants import *
from .pixel_finder import landmark_to_pixels
from .utils import locate_hand_landmarks, draw_landmarks_on_image, resize_image, save_roi_image
import subprocess
import segmentation
from shapely.geometry import Polygon
from random import randint

def rect_to_polygon(rect):
    center, size, theta = rect
    cx, cy = center
    width, height = size
    angle = math.radians(theta)

    # Calculate the corner points
    dx = width / 2
    dy = height / 2
    corners = []
    for x in [-dx, dx]:
        for y in [-dy, dy]:
            nx = x * math.cos(angle) - y * math.sin(angle)
            ny = x * math.sin(angle) + y * math.cos(angle)
            corners.append((cx + nx, cy + ny))

    return Polygon(corners)

def calculate_iou(rect1, rect2):
    poly1 = rect_to_polygon(rect1)
    poly2 = rect_to_polygon(rect2)

    # Calculate intersection and union
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection

    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    return iou


def process_finger(finger_key, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask, PATH, FINGER_OUTPUT_DIR):
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

    # Draw the landmarks on the image
    cv2.circle(roi, tuple(new_pip), 5, (255, 0, 0), -1)
    cv2.circle(roi, tuple(new_dip), 5, (255, 0, 0), -1)

    # # Save the ROI image
    # output_path = os.path.join(FINGER_OUTPUT_DIR, finger_key + "_" + os.path.basename(PATH))
    # save_roi_image(roi, output_path)

    # Compute pixel width of object at the row of new_pip
    pip_width = find_object_width_at_row(roi, new_pip[1], new_pip[0])
    
    # Compute pixel width of object at the row of new_dip
    dip_width = find_object_width_at_row(roi, new_dip[1], new_dip[0])

    # Compute vertical pixel distance between new_pip and new_dip
    vertical_distance = abs(new_dip[1] - new_pip[1])

    # Return pip_width, dip_width, and vertical_distance
    return pip_width, dip_width, vertical_distance, rect, roi


def is_point_inside_rect(image, point, rect):
    (center, (width, height), theta) = rect
    roi, rotation_matrix = extract_roi(image, rect)
    rotated_point = transform_point(point, rotation_matrix)
    adjusted_point = adjust_for_roi_crop(rotated_point, rect[0], rect[1])

    # Check if the adjusted point is within the rectangle's boundaries
    half_width, half_height = width / 2, height / 2
    return (-half_width <= adjusted_point[0] <= half_width) and (-half_height <= adjusted_point[1] <= half_height)

    
def process_image(PATH, MASKS_OUTPUT_DIR, FINGER_OUTPUT_DIR, NAIL_OUTPUT_DIR):
    image, landmarks = locate_hand_landmarks(PATH, "hand_landmarker.task")
    
    if not landmarks.hand_landmarks:
        print(f"Warning: No landmarks detected for {os.path.basename(PATH)}")
        return [0, 0, 0, 0], [0, 0, 0, 0]

    landmark_pixels = landmarks_to_pixel_coordinates(image, landmarks)
    enhanced_image = image
    padding = 250
    
    try:
        result = segmentation.bg.remove(data=enhanced_image)
    except ValueError as e:
        print(f"Caught a value error: {e} on image {os.path.basename(PATH)}")
        return [0, 0, 0, 0], [0, 0, 0, 0]
        
    result = cv2.copyMakeBorder(result, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    landmark_pixels = [(x+padding, y+padding) for x, y in landmark_pixels]
    
    seg_mask = get_segmentation_mask(result)
    OUTPUT_PATH_MASK = os.path.join(MASKS_OUTPUT_DIR, "seg_" + os.path.basename(PATH))
    cv2.imwrite(OUTPUT_PATH_MASK, seg_mask)

    contour = extract_contour(seg_mask)
    if contour is None or len(contour.shape) == 1:
        print(f"Warning: The contour is empty. Skipping {os.path.basename(PATH)}.")
        return [0, 0, 0, 0], [0, 0, 0, 0]

    rgb_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
    closest_points = closest_contour_point(landmark_pixels, contour)
    vertical_distances = []
    pip_widths, dip_widths = [], []
    used_fingers = []
    for key in ['INDEX', 'MIDDLE', 'RING', 'PINKY']:
        if key in used_fingers:
            continue
        pip_width, dip_width, vertical_distance, rect, roi = process_finger(key, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask, PATH, FINGER_OUTPUT_DIR)
        pip_widths.append(pip_width)
        dip_widths.append(dip_width)
        vertical_distances.append(vertical_distance)

        # Get the neighboring fingers for the current finger
        neighbors = finger_neighbors[key]

        # Transform and check each neighboring finger's landmarks
        for neighbor_key in neighbors:
            neighbor_pip = np.array(landmark_pixels[landmarks_per_finger[neighbor_key][1]])
            neighbor_dip = np.array(landmark_pixels[landmarks_per_finger[neighbor_key][2]])
            transformed_neighbor_pip = transform_point(neighbor_pip, rect[1])
            transformed_neighbor_dip = transform_point(neighbor_dip, rect[2])
            if is_point_inside_rect(rgb_mask, neighbor_dip, rect) or is_point_inside_rect(rgb_mask, neighbor_pip, rect):
                this_pip = np.array(landmark_pixels[landmarks_per_finger[key][1]])
                this_dip = np.array(landmark_pixels[landmarks_per_finger[key][2]])

                # Middle point of pip and dip
                pip_middle = (this_pip + transformed_neighbor_pip) // 2
                dip_middle = (this_dip + transformed_neighbor_dip) // 2

                # Line slope and intercept
                slope = (pip_middle[1] - dip_middle[1]) / (pip_middle[0] - dip_middle[0])
                intercept = pip_middle[1] - slope * pip_middle[0]

                # Check if this pip and dip are left or right of the line defined by slope and intercept
                pip_left = pip_middle[0] < (slope * pip_middle[1] + intercept)
                dip_left = dip_middle[0] < (slope * dip_middle[1] + intercept)

                if pip_left and dip_left:
                    # Make every pixel to the rigth of the roi black
                    roi[:, int(pip_middle[0]):] = 0
                    neighbor_roi = roi[:, :int(pip_middle[0])]
                elif not pip_left and not dip_left:
                    # Make every pixel to the left of the roi black
                    roi[:, :int(pip_middle[0])] = 0
                    neighbor_roi = roi[:, int(pip_middle[0]):]


                used_fingers.append(neighbor_key)
                print(f"Warning: Finger {key} is overlapping with finger {neighbor_key} in image {os.path.basename(PATH)}")

                    mean_vertical_distance = np.mean(vertical_distances)

    return np.array(pip_widths) / mean_vertical_distance, np.array(dip_widths) / mean_vertical_distance
