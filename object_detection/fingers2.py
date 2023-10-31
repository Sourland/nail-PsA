import os

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from landmarks_constants import landmarks_per_finger, joint_neighbours_right_hand, joint_neighbours_left_hand
from object_detection.pixel_finder import landmark_to_pixels
from utils import load_hand_landmarker, locate_hand_landmarks, draw_landmarks_on_image
import subprocess

def get_landmarks(image_path, detector_path):
    img = cv2.imread(image_path, cv2.COLOR_RGBA2RGB)
    detector = load_hand_landmarker(detector_path)
    landmarks = locate_hand_landmarks(image_path, detector)
    return landmarks


def load_image_and_locate_landmarks(hand_path, detector_path):
    img = cv2.imread(hand_path, cv2.COLOR_RGBA2RGB)
    detector = load_hand_landmarker(detector_path)
    landmarks = locate_hand_landmarks(hand_path, detector)

    return img, landmarks

def get_segmentation_mask(image: np.ndarray, threshold: int = 11) -> np.ndarray:
    """
    Generate a binary segmentation mask based on pixel intensity.

    Parameters:
    - image: an RGB image in numpy array format.
    - threshold: pixel values above this threshold will be set to 1, below will be set to 0.

    Returns:
    - Binary segmentation mask as numpy array.
    """

    # Check if the image has three channels
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels. Received image with shape {}.".format(image.shape))

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Generate the mask
    mask = np.where(grayscale_image > threshold, 255, 0)

    return mask.astype(np.uint8)
    

def extract_contour(image: np.ndarray) -> np.ndarray:
    print(image.dtype)
    # Threshold the grayscale image
    # _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Return the largest contour by area
    return max(contours, key=cv2.contourArea)




def landmarks_to_pixel_coordinates(image, landmarks) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [landmark_to_pixels(gray, landmarks.hand_landmarks[0], idx) for idx in
            range(len(landmarks.hand_landmarks[0]))]



def process_image(PATH, OUTPUT_DIR):
    image = cv2.imread(PATH, cv2.COLOR_RGBA2RGB)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "seg_" + os.path.basename(PATH))
    # result = backgroundremover.bg.remove(image)
    # cv2.imwrite(OUTPUT_PATH, result)
    cmd = ["backgroundremover", "-i", PATH, "-m", "u2net", "-o", OUTPUT_PATH]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(os.path.basename(PATH))
    # _,  landmarks = load_image_and_locate_landmarks(OUTPUT_PATH, "hand_landmarker.task")
    image = cv2.imread(OUTPUT_PATH, cv2.COLOR_RGBA2RGB)
    

        
    seg_mask = get_segmentation_mask(image)
    contour = extract_contour(seg_mask)
    cv2.drawContours(seg_mask, [contour], 0, 128, 2)
    cv2.namedWindow('Image with Contour', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Contour', seg_mask)
    cv2.waitKey(0)
    # # if not landmarks.hand_landmarks:
    # #     return
    # # else:
    # #     landmarks = landmarks.hand_landmarks[0]
    # cv2.imwrite(OUTPUT_PATH, output)
    


if __name__ == "__main__":
    DIR_PATH = "dataset/hands/swolen/"
    OUTPUT_DIR = "results/SegMasks/"

    # Make sure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for image_name in os.listdir(DIR_PATH):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):  # checking file extension
            image_path = os.path.join(DIR_PATH, image_name)
            process_image(image_path, OUTPUT_DIR)

