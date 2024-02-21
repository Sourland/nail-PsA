import math
import os

import cv2
import numpy as np
from object_detection.contour_extraction import closest_contour_point, extract_contour
from object_detection.landmarks import adjust_for_roi_crop, find_object_width_at_row, landmarks_to_pixel_coordinates, transform_point
from object_detection.roi_extraction import extract_roi, get_bounding_box_from_points
from object_detection.segmentation import get_segmentation_mask
from .landmarks_constants import *
from .utils import locate_hand_landmarks, save_roi_image
import segmentation


def is_inside_rotated_rect(rotated_point, rect):
    """
    Checks if a point is inside a rotated rectangle.

    Args:
        rotated_point (tuple): The point to check (x, y).
        rect (tuple): The rotated rectangle (center, size, angle).

    Returns:
        bool: True if the point is inside the rectangle, False otherwise.

    Test Case:
        >>> rotated_point = (55, 55)
        >>> rect = ((50, 50), (40, 20), 0)
        >>> is_inside_rotated_rect(rotated_point, rect)
        True
    """
    (center, (width, height), _) = rect
    x_min = center[0] - width / 2
    y_min = center[1] - height / 2
    x_max = center[0] + width / 2
    y_max = center[1] + height / 2

    return x_min <= rotated_point[0] <= x_max and y_min <= rotated_point[1] <= y_max


def get_rotated_image_shape(image_shape, rotation_matrix):
    """
    Processes a finger to compute measurements and adjust images.

    Args:
        finger_key (str): The key identifying the finger.
        landmarks_per_finger (dict): A dictionary mapping fingers to their respective landmarks.
        closest_points (list): A list of closest contour points for each landmark.
        landmark_pixels (list): Pixel coordinates of the landmarks.
        rgb_mask (np.ndarray): The RGB mask of the image.
        PATH (str): The file path of the input image.
        FINGER_OUTPUT_DIR (str): The directory where output images are saved.

    Returns:
        None

    Test Case:
        # Due to the complexity and dependency on external files and data, specific test cases should be created based on the actual scenario.
    """
    height, width = image_shape
    # Corners of the original image
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    # Add a column of 1s for affine transformation
    ones = np.ones(shape=(len(corners), 1))
    corners_ones = np.hstack([corners, ones])

    # Transform the corners
    transformed_corners = rotation_matrix.dot(corners_ones.T).T

    # Calculate new bounding box
    x_coords, y_coords = zip(*transformed_corners[:, :2])
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # New dimensions
    new_width, new_height = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))

    return new_height, new_width


def process_finger(finger_key, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask, PATH, FINGER_OUTPUT_DIR):
    """
    Processes a finger to compute measurements and adjust images.

    Args:
        finger_key (str): The key identifying the finger.
        landmarks_per_finger (dict): A dictionary mapping fingers to their respective landmarks.
        closest_points (list): A list of closest contour points for each landmark.
        landmark_pixels (list): Pixel coordinates of the landmarks.
        rgb_mask (np.ndarray): The RGB mask of the image.
        PATH (str): The file path of the input image.
        FINGER_OUTPUT_DIR (str): The directory where output images are saved.

    Returns:
        None

    Test Case:
        # Due to the complexity and dependency on external files and data, specific test cases should be created based on the actual scenario.
    """
    def process_neighbors(roi, neighbors, landmarks_per_finger, landmark_pixels, rotation_matrix, rect, pip_width, dip_width, vertical_distance):
        """
        Processes neighboring fingers and adjusts the ROI image accordingly.

        Args:
            roi (np.ndarray): The region of interest of the finger.
            neighbors (list): List of keys identifying neighboring fingers.
            landmarks_per_finger (dict): A dictionary mapping fingers to their respective landmarks.
            landmark_pixels (list): Pixel coordinates of the landmarks.
            rotation_matrix (np.ndarray): The rotation matrix used for the initial ROI extraction.
            rect (tuple): The bounding box of the ROI in the original image.

        Returns:
            np.ndarray: The modified ROI after processing neighbors.
        """
        for neighbor_key in neighbors:
            neighbor_pip = np.array(landmark_pixels[landmarks_per_finger[neighbor_key][1]])
            neighbor_dip = np.array(landmark_pixels[landmarks_per_finger[neighbor_key][2]])
            
            # Rotate the landmarks
            rotated_neighbor_pip = transform_point(neighbor_pip, rotation_matrix)
            rotated_neighbor_dip = transform_point(neighbor_dip, rotation_matrix)

            if is_inside_rotated_rect(rotated_neighbor_dip, rect) and is_inside_rotated_rect(rotated_neighbor_pip, rect):
                
                # Map the landmarks to the resized image
                transformed_neighbor_pip = adjust_for_roi_crop(rotated_neighbor_pip, rect[0], rect[1])
                transformed_neighbor_dip = adjust_for_roi_crop(rotated_neighbor_dip, rect[0], rect[1])
                
                # Draw neighbor landmarks on the image, cyan for pip and magenta for dip
                cv2.circle(roi, tuple(transformed_neighbor_pip), 5, (255, 255, 0), -1)
                cv2.circle(roi, tuple(transformed_neighbor_dip), 5, (255, 0, 255), -1)
                # Middle point of pip and dip
                pip_middle = ((new_pip[0] + transformed_neighbor_pip[0]) / 2, (new_pip[1] + transformed_neighbor_pip[1]) / 2)
                dip_middle = ((new_dip[0] + transformed_neighbor_dip[0]) / 2, (new_dip[1] + transformed_neighbor_dip[1]) / 2)

                # Transform pip middle and dip middle to have the center of roi as origin
                pip_middle = (pip_middle[0] - roi.shape[1] / 2, pip_middle[1] - roi.shape[0] / 2)
                dip_middle = (dip_middle[0] - roi.shape[1] / 2, dip_middle[1] - roi.shape[0] / 2)

                angle = math.degrees(math.atan2(dip_middle[1] - pip_middle[1], dip_middle[0] - pip_middle[0]))
                angle = 90 + angle  # Adjusting to make the line vertical

                # Calculate the center of roi
                center = (roi.shape[1] / 2, roi.shape[0] / 2)
                new_rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Calculate size of rotated image
                new_height, new_width = get_rotated_image_shape(roi.shape[:2], new_rotation_matrix)

                # Pad the image to make sure it doesn't get cropped
                roi = cv2.copyMakeBorder(roi, 0, np.abs(new_height - roi.shape[0]), 0, 
                                        np.abs(new_width - roi.shape[1]), cv2.BORDER_CONSTANT, value=0)
                # Rotate the image
                rotated_roi = cv2.warpAffine(roi, new_rotation_matrix, (new_width, new_height))

                # Rotate the transformed landmarks
                rotated_pip = transform_point(new_pip, new_rotation_matrix)
                rotated_dip = transform_point(new_dip, new_rotation_matrix)
                rotated_neighbor_pip = transform_point(transformed_neighbor_pip, new_rotation_matrix)
                rotated_neighbor_dip = transform_point(transformed_neighbor_dip, new_rotation_matrix)

                # Calculate new mid points
                pip_middle = ((rotated_pip[0] + rotated_neighbor_pip[0]) // 2, (rotated_pip[1] + rotated_neighbor_pip[1]) //2)
                dip_middle = ((rotated_dip[0] + rotated_neighbor_dip[0]) // 2, (rotated_dip[1] + rotated_neighbor_dip[1]) // 2)

                # Check if landmarks are on the left or right side of the neighbor
                left = rotated_neighbor_pip[0] < pip_middle[0]
                if not left:
                    # Black the image to the right of the middle point
                    rotated_roi[:, int(pip_middle[0]):] = 0
                else:
                    # Black the image to the left of the middle point
                    rotated_roi[:, :int(pip_middle[0])] = 0
                
                # Compute pixel width of object at the row of new_pip
                pip_width = find_object_width_at_row(rotated_roi, rotated_pip[1], rotated_pip[0])
                # Compute pixel width of object at the row of new_dip
                dip_width = find_object_width_at_row(rotated_roi, rotated_dip[1], rotated_dip[0])
                vertical_distance = abs(new_dip[1] - new_pip[1])
                output_path = os.path.join(FINGER_OUTPUT_DIR, "rotated_" + finger_key + "_" + neighbor_key + "_" + os.path.basename(PATH))
                save_roi_image(rotated_roi, output_path)
            return pip_width, dip_width, vertical_distance
        return pip_width, dip_width, vertical_distance
        


    finger_roi_points = [item for idx in landmarks_per_finger[finger_key][1:] for item in closest_points[idx]]
    finger_roi_points.append(landmark_pixels[landmarks_per_finger[finger_key][0]])

    rect = get_bounding_box_from_points(rgb_mask, finger_roi_points)
    roi, rotation_matrix = extract_roi(rgb_mask, rect)
    pip = np.array(landmark_pixels[landmarks_per_finger[finger_key][1]])
    dip = np.array(landmark_pixels[landmarks_per_finger[finger_key][2]])

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

    vertical_distance = abs(new_dip[1] - new_pip[1])
    # Draw the landmarks on the image, blue for pip, red for dip
    cv2.circle(roi, tuple(new_pip), 5, (255, 0, 0), -1)
    cv2.circle(roi, tuple(new_dip), 5, (0, 0, 255), -1)

    neighbors = finger_neighbors[finger_key]

    pip_width, dip_width, vertical_distance = process_neighbors(roi, neighbors, landmarks_per_finger, landmark_pixels, rotation_matrix, rect, pip_width, dip_width, vertical_distance)

    # Compute vertical pixel distance between new_pip and new_dip
    vertical_distance = abs(new_dip[1] - new_pip[1])

    # Return pip_width, dip_width, and vertical_distance
    return pip_width, dip_width, vertical_distance


def process_image(PATH, MASKS_OUTPUT_DIR, FINGER_OUTPUT_DIR):
    """
    Processes an image to detect hand landmarks and compute related metrics.

    Args:
        PATH (str): The file path of the input image.
        MASKS_OUTPUT_DIR (str): The directory where mask output images are saved.
        FINGER_OUTPUT_DIR (str): The directory where finger-related output images are saved.
        NAIL_OUTPUT_DIR (str): The directory where nail-related output images are saved.

    Returns:
        tuple: Arrays of ratios of pip widths and dip widths to the mean vertical distance.

    Test Case:
        # Requires specific image files and directory setup for a practical test case.
    """
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
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
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
    pip_widths, dip_widths, vertical_distances = [], [], []
    for key in ['INDEX', 'MIDDLE', 'RING', 'PINKY']:
        pip_width, dip_width, vertical_distance = process_finger(key, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask, PATH, FINGER_OUTPUT_DIR)
        pip_widths.append(pip_width)
        dip_widths.append(dip_width)

    mean_vertical_distance = np.mean(vertical_distances)

    return np.array(pip_widths) / mean_vertical_distance, np.array(dip_widths) / mean_vertical_distance
