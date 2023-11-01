import os

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from .landmarks_constants import *
from .pixel_finder import landmark_to_pixels
from .utils import load_hand_landmarker, locate_hand_landmarks, draw_landmarks_on_image
import subprocess
import segmentation
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

    # Retur n the largest contour by area
    return np.squeeze(contours[0])


def landmarks_to_pixel_coordinates(image, landmarks) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [landmark_to_pixels(gray, landmarks.hand_landmarks[0], idx) for idx in
            range(len(landmarks.hand_landmarks[0]))]


def closest_contour_point(landmarks, contour):

    """
    For each landmark, find the closest left and right points on the contour.

    Args:
    - landmarks (list of tuples): List of landmarks as (x, y) coordinates.
    - contour (numpy array): Contour returned by cv2.findContours.

    Returns:
    - List of tuples, each tuple containing two (x, y) coordinates - the closest left and right points on the contour for each landmark.
    """
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


def draw_landmarks_and_connections(image, landmarks, contour):
    """
    Draw landmarks, closest left and right points on the contour, and lines connecting them on the image.

    Args:
    - image (numpy array): The image on which to draw.
    - landmarks (list of tuples): List of landmarks as (x, y) coordinates.
    - contour (numpy array): Contour returned by cv2.findContours.
    """
    closest_points = closest_contour_point(landmarks, contour)

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


def process_image(PATH, OUTPUT_DIR):
    image = cv2.imread(PATH, cv2.COLOR_RGBA2RGB)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "seg_" + os.path.basename(PATH))
    # result = backgroundremover.bg.remove(image)
    # cv2.imwrite(OUTPUT_PATH, result)
    # cmd = ["backgroundremover", "-i", PATH, "-m", "u2net", "-o", OUTPUT_PATH]
    result = segmentation.bg.remove(image)
    cv2.namedWindow('Image with Contour', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Contour', result)
    cv2.waitKey(0)
    # print(os.path.basename(PATH))
    _,  landmarks = load_image_and_locate_landmarks(OUTPUT_PATH, "hand_landmarker.task")
    if not landmarks.hand_landmarks:
        return
    else:
        landmarks = landmarks
    image = cv2.imread(OUTPUT_PATH, cv2.COLOR_RGBA2RGB)    
    landmark_pixels = landmarks_to_pixel_coordinates(image, landmarks)
    seg_mask = get_segmentation_mask(image)
    contour = extract_contour(seg_mask)

    rgb_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
    output = draw_landmarks_and_connections(rgb_mask, landmark_pixels, contour)
    
    # cv2.imwrite(OUTPUT_PATH, output)
    cv2.namedWindow('Image with Contour', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Contour', output)
    cv2.waitKey(0)

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

