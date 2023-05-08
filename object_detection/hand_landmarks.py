import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from draw_landmarks_on_image import draw_landmarks_on_image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.13)

    # Create and return the HandLandmarker object
    return vision.HandLandmarker.create_from_options(options)


def locate_hand_landmarks(image: np.ndarray, detector: vision.HandLandmarker) -> vision.HandLandmarkerResult:
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
    mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Use the HandLandmarker object to detect hand landmarks in the mediapipe image
    return detector.detect(mediapipe_image)


img = cv2.imread('../hand3.jpg', cv2.IMREAD_UNCHANGED)
resized_img = resize_image(img, 300)
detector = load_hand_landmarker('../hand_landmarker.task')
landmarks = locate_hand_landmarks(resized_img, detector)
annotated_image = draw_landmarks_on_image(
    mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_img).numpy_view(),
    landmarks
)
# cv2.imshow('image', annotated_image)
# cv2.waitKey(0)

sam = sam_model_registry["vit_h"](checkpoint="../sam_vit_h_4b8939.pth")
predictor = SamAutomaticMaskGenerator(sam)
masks = predictor.generate(resized_img)
i = 0



