import math
import os

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from object_detection.contour_extraction import closest_contour_point, create_finger_contour, extract_contour, get_left_and_right_contour_points, reorient_contour
from object_detection.landmarks import calculate_widths_and_distance, landmarks_to_pixel_coordinates, transform_landmarks
from object_detection.segmentation import get_segmentation_mask
from .landmarks_constants import *
from .pixel_finder import landmark_to_pixels
from .utils import locate_hand_landmarks, draw_landmarks_on_image, resize_image
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


def process_image(PATH, MASKS_OUTPUT_DIR, FINGER_OUTPUT_DIR, NAIL_OUTPUT_DIR):
    image, landmarks = locate_hand_landmarks(PATH, "hand_landmarker.task")
    
    if not landmarks.hand_landmarks:
        print(f"Warning: No landmarks detected for {os.path.basename(PATH)}")
        return [0, 0, 0, 0], [0, 0, 0, 0]

    landmark_pixel_positions = landmarks_to_pixel_coordinates(image, landmarks)
    enhanced_image = image
    padding = 250
    
    try:
        result = segmentation.bg.remove(data=enhanced_image)
    except ValueError as e:
        print(f"Caught a value error: {e} on image {os.path.basename(PATH)}")
        return [0, 0, 0, 0], [0, 0, 0, 0]
        
    result = cv2.copyMakeBorder(result, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    landmark_pixel_positions = [(x+padding, y+padding) for x, y in landmark_pixel_positions]
    
    seg_mask = get_segmentation_mask(result)
    OUTPUT_PATH_MASK = os.path.join(MASKS_OUTPUT_DIR, "seg_" + os.path.basename(PATH))
    # cv2.imwrite(OUTPUT_PATH_MASK, seg_mask)

    contour = extract_contour(seg_mask)
    # contour = reorient_contour(contour)

    if contour is None or len(contour.shape) == 1:
        print(f"Warning: The contour is empty. Skipping {os.path.basename(PATH)}.")
        return [0, 0, 0, 0], [0, 0, 0, 0]

    rgb_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
    hull = cv2.convexHull(contour)
    # cv2.drawContours(rgb_mask, [contour], -1, (0, 255, 0), 1)
    # cv2.drawContours(rgb_mask, [hull], -1, (0, 0, 255), 1)
    # cv2.imwrite(os.path.join(MASKS_OUTPUT_DIR, "contour_" + os.path.basename(PATH)), rgb_mask)


    vertical_distances = []
    pip_widths, dip_widths = [], []
    lol = ['INDEX', 'MIDDLE', 'RING', 'PINKY']
    for idx, finger in enumerate(["MIDDLE"]):
        tip = landmarks_per_finger[finger][-1]
        tip_proxy = closest_contour_point(landmark_pixel_positions[tip], np.squeeze(hull))
        left_contour, right_contour = get_left_and_right_contour_points(tip_proxy, contour)
        closest_mcp_left = closest_contour_point(landmark_pixel_positions[landmarks_per_finger[finger][0]], left_contour)
        closest_mcp_right = closest_contour_point(landmark_pixel_positions[landmarks_per_finger[finger][0]], right_contour)
        # finger_contour = create_finger_contour(tip_proxy, left_contour, right_contour, closest_mcp_left, closest_mcp_right)
        cv2.polylines(rgb_mask, [right_contour], isClosed=False, color=(randint(128, 255), randint(128, 255), 0), thickness=2)
        cv2.polylines(rgb_mask, [left_contour], isClosed=False, color=(randint(128, 255), 0, randint(128, 255)), thickness=2)

        # rect, new_pip, new_dip, roi = transform_landmarks(finger, landmarks_per_finger, finger_contour, landmark_pixel_positions, rgb_mask)
        # pip_width, dip_width, vertical_distance = calculate_widths_and_distance(new_pip, new_dip, roi)
        # pip_widths.append(pip_width)
        # dip_widths.append(dip_width)
        # vertical_distances.append(vertical_distance)


        # cv2.imwrite(os.path.join(FINGER_OUTPUT_DIR, finger + "_" + os.path.basename(PATH)), roi)
        # # Get the neighboring fingers for the current finger
        # neighbors = finger_neighbors[key]

        # # Transform and check each neighboring finger's landmarks
        # for neighbor_key in neighbors:
        #     # Use transform_landmarks for each neighbor finger
        #     neighbor_rect, _, _, neighbor_roi = transform_landmarks(neighbor_key, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask)

        #     # Check if the neighbor finger's ROI overlaps with the current finger's ROI
        #     if calculate_iou(rect, neighbor_rect) > 0.5:
        #         print(f"The ROI of {neighbor_key} finger overlaps with the ROI of {key} finger.")

    mean_vertical_distance = np.mean(vertical_distances)
    cv2.imwrite(os.path.join(MASKS_OUTPUT_DIR, "contour_" + os.path.basename(PATH)), rgb_mask)
    return np.array(pip_widths) / mean_vertical_distance, np.array(dip_widths) / mean_vertical_distance
