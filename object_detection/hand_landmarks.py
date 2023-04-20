import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from draw_landmarks_on_image import draw_landmarks_on_image

def resize_image(img: np.ndarray, new_size:int) -> np.ndarray:
    """
    Resizes the image by resizing the smaller axis to the desired size in order to maintain aspect ratio.

    Args:
        img: The image to be resized
        new_size: the new size of the smaller axis

    Returns: Resized image
    """
    scale_percent = new_size / min((img.shape[0], img.shape[1]))
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)

    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


base_options = python.BaseOptions(model_asset_path='../hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.13)
# detector
detector = vision.HandLandmarker.create_from_options(options)
img = cv2.imread('../hand.jpg', cv2.IMREAD_UNCHANGED)
resized_img = resize_image(img, 300)
resized_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized_img)
detection_result = detector.detect(resized_img)
annotated_image = draw_landmarks_on_image(resized_img.numpy_view(), detection_result)
cv2.imshow('image', annotated_image)
cv2.waitKey(0)





