import os

import cv2
import numpy as np
import segmentation.bg

from object_detection.contour_extraction import closest_contour_point, get_largest_contour
from object_detection.finger_width import compute_biometrics
from object_detection.landmarks import landmarks_to_pixel_coordinates
from object_detection.utils import locate_hand_landmarks, get_segmentation_mask
from object_detection.landmarks_constants import landmarks_per_finger


class HandBiometricAnalyzer:
    def __init__(self, masks_output_dir, finger_output_dir):
        self.masks_output_dir = masks_output_dir
        self.finger_output_dir = finger_output_dir
        self.padding = 250

    def process(self, path):
        """
        Processes an image to detect hand landmarks and compute related metrics.

        Args:
            path (str): The file path of the input image.

        Returns:
            tuple: Arrays of ratios of pip widths and dip widths to the mean vertical distance.
        """
        image, landmarks = locate_hand_landmarks(path, "hand_landmarker.task")
        if not landmarks.hand_landmarks:
            print(f"Warning: No landmarks detected for {os.path.basename(path)}")
            return [0, 0, 0, 0], [0, 0, 0, 0]
        landmark_pixels = landmarks_to_pixel_coordinates(image, landmarks)
        enhanced_image = image

        try:
            result = segmentation.bg.remove(data=enhanced_image)
        except ValueError as e:
            print(f"Caught a value error: {e} on image {os.path.basename(path)}")
            return [0, 0, 0, 0], [0, 0, 0, 0]

        result = cv2.copyMakeBorder(result, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT, value=0)
        image = cv2.copyMakeBorder(image, self.padding, self.padding, self.padding, self.padding, cv2.BORDER_CONSTANT, value=0)
        landmark_pixels = [(x + self.padding, y + self.padding) for x, y in landmark_pixels]
        
        seg_mask = get_segmentation_mask(result)
        output_path_mask = os.path.join(self.masks_output_dir, "seg_" + os.path.basename(path))
        cv2.imwrite(output_path_mask, seg_mask)
        contour = get_largest_contour(seg_mask)
        
        if contour is None or len(contour.shape) == 1:
            print(f"Warning: The contour is empty. Skipping {os.path.basename(path)}.")
            return [0, 0, 0, 0], [0, 0, 0, 0]
        
        rgb_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
        closest_points = closest_contour_point(landmark_pixels, contour)
        pip_widths, dip_widths, vertical_distances = [], [], []
        
        for finger in ['INDEX', 'MIDDLE', 'RING', 'PINKY']:
            pip_width, dip_width, vertical_distance = compute_biometrics(
                finger, landmarks_per_finger, closest_points, landmark_pixels, rgb_mask, path, self.finger_output_dir)
            pip_widths.append(pip_width)
            dip_widths.append(dip_width)
            vertical_distances.append(vertical_distance)
        mean_vertical_distance = np.mean(vertical_distances)
        
        return np.array(pip_widths) / mean_vertical_distance, np.array(dip_widths) / mean_vertical_distance