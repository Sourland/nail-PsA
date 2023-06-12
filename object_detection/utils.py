import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandRegionExtractor:
    def __int__(self, landmark_detector, hand_extractor):
        ...


def resize_image(img: np.ndarray, new_size: int) -> np.ndarray:
    """
    Resizes the image by resizing the smaller axis to the desired size in order to maintain aspect ratio.

    Args:
        img: The image to be resized
        new_size: the new size of the smaller axis

    Returns: Resized image

    Raises: None
    """
    scale_percent = new_size / min((img.shape[0], img.shape[1]))
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)

    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def load_hand_landmarker(path: str) -> vision.HandLandmarker:
    """
    Loads a hand landmark detection model from the specified path and returns a HandLandmarker object.

    Args:
        path (str): The path to the model asset file.

    Returns:
        vision.HandLandmarker: A HandLandmarker object for detecting landmarks on hands in images.

    Raises:
        None
    """
    # Define the base options for loading the model
    base_options = python.BaseOptions(model_asset_path=path)

    # Set options for the HandLandmarker object
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.1)

    # Create and return the HandLandmarker object
    return vision.HandLandmarker.create_from_options(options)


def locate_hand_landmarks(image: str, detector: vision.HandLandmarker) -> vision.HandLandmarkerResult:
    """
    Detects hand landmarks in the input image using the specified HandLandmarker object.

    Args:
        image (np.ndarray): The input image to detect hand landmarks on.
        detector (vision.HandLandmarker): A HandLandmarker object for detecting landmarks on hands in images.

    Returns:
        vision.HandLandmarkerResult: The result of the hand landmark detection, which includes the detected landmarks and their confidence scores.

    Raises:
        None
    """
    # Convert the input image to a mediapipe image
    mediapipe_image = mp.Image.create_from_file(image)

    # Use the HandLandmarker object to detect hand landmarks in the mediapipe image
    return detector.detect(mediapipe_image)
