import cv2
from object_detection.hand_landmarks import load_hand_landmarker, locate_hand_landmarks
from object_detection.utils import resize_image


class ImageProcessor:
    BG_COLOR = (0, 0, 0)  # gray
    MASK_COLOR = (255, 255, 255)  # white
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480

    def __init__(self, image_path, model_path):
        self.image_path = image_path
        self.model_path = model_path
        self.detector = self.load_hand_landmarker(self.model_path)

    def load_hand_landmarker(self, model_path):
        return load_hand_landmarker(model_path)

    def locate_hand_landmarks(self):
        return locate_hand_landmarks(self.image_path, self.detector)

    def resize_and_show(self, image):
        resized_image = resize_image(image, min(self.DESIRED_HEIGHT, self.DESIRED_WIDTH))
        cv2.imwrite('../seg_mask.jpg', resized_image)
