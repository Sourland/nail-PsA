import math
import os

import cv2
import numpy as np
from biometric_extraction.utils.contours import closest_contour_point, get_largest_contour
from biometric_extraction.utils.biomarker_helpers import add_padding, calculate_transformed_image_shape, is_inside_rotated_rect
from landmark_extraction.utils.landmarks_constants import *
from biometric_extraction.utils.roi_helpers import extract_roi, get_bounding_box_from_points
from segmentation.bg import BackgroundRemover
from biometric_extraction.utils.transform import transform_point, adjust_point_to_roi, find_object_width_at_row
from landmark_extraction.hand_landmarker import HandLandmarks

class HandBiometricAnalyzer:
    def __init__(self, segmentor : BackgroundRemover, hand_landmarker : HandLandmarks, masks_output_dir, finger_output_dir):
        self.masks_output_dir = masks_output_dir
        self.finger_output_dir = finger_output_dir
        self.padding = 250
        self.segmentor = segmentor
        self.hand_landmarker = hand_landmarker

    def process(self, path):
        """
        Processes an image to detect hand landmarks and compute related metrics.

        Args:
            path (str): The file path of the input image.

        Returns:
            tuple: Arrays of ratios of pip widths and dip widths to the mean vertical distance.
        """

        landmarks, image = self.hand_landmarker.locate_hand_landmarks(path)

        if not landmarks:
            print(f"Warning: No landmarks detected for {os.path.basename(path)}")
            return [0, 0, 0, 0], [0, 0, 0, 0]
        
        try:
            seg_mask = self.segmentor.get_segmentation_mask(image)
        except ValueError as e:
            print(f"Caught a value error: {e} on image {os.path.basename(path)}")
            return [0, 0, 0, 0], [0, 0, 0, 0]

        seg_mask, landmark_pixels = add_padding(seg_mask, landmarks)

        contour = get_largest_contour(seg_mask)
        
        if contour is None or len(contour.shape) == 1:
            print(f"Warning: The contour is empty. Skipping {os.path.basename(path)}.")
            return [0, 0, 0, 0], [0, 0, 0, 0]
        
        rgb_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
        closest_points = closest_contour_point(landmark_pixels, contour)
        pip_widths, dip_widths, vertical_distances = [], [], []
        
        for finger in ['INDEX', 'MIDDLE', 'RING', 'PINKY']:
            pip_width, dip_width, vertical_distance = self.compute_biometrics(finger, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask)
            pip_widths.append(pip_width)
            dip_widths.append(dip_width)
            vertical_distances.append(vertical_distance)
        mean_vertical_distance = np.mean(vertical_distances)
        
        return np.array(pip_widths) / mean_vertical_distance, np.array(dip_widths) / mean_vertical_distance
    
    @staticmethod
    def compute_biometrics(finger_key, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask):
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
                    transformed_neighbor_pip = adjust_point_to_roi(rotated_neighbor_pip, rect[0], rect[1])
                    transformed_neighbor_dip = adjust_point_to_roi(rotated_neighbor_dip, rect[0], rect[1])
                    
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
                    new_height, new_width = calculate_transformed_image_shape(roi.shape[:2], new_rotation_matrix)

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
                return pip_width, dip_width, vertical_distance
            return pip_width, dip_width, vertical_distance
            


        finger_roi_points = [item for idx in landmarks_per_finger[finger_key][1:] for item in closest_points[idx]]
        finger_roi_points.append(landmark_pixels[landmarks_per_finger[finger_key][0]])

        rect = get_bounding_box_from_points(finger_roi_points)
        roi, rotation_matrix = extract_roi(rgb_mask, rect)
        pip = np.array(landmark_pixels[landmarks_per_finger[finger_key][1]])
        dip = np.array(landmark_pixels[landmarks_per_finger[finger_key][2]])

        # Rotate the landmarks
        rotated_pip = transform_point(pip, rotation_matrix)
        rotated_dip = transform_point(dip, rotation_matrix)

        # Map the landmarks to the resized image
        new_pip = adjust_point_to_roi(rotated_pip, rect[0], rect[1])
        new_dip = adjust_point_to_roi(rotated_dip, rect[0], rect[1])

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