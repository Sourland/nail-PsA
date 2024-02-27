import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandLandmarks:
    def __init__(self, hand_landmarker_path: str):
        self.base_options = python.BaseOptions(model_asset_path=hand_landmarker_path)
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options,
                                           num_hands = 2,
                                           min_hand_detection_confidence=0.01)
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    @staticmethod
    def landmark_to_pixels(image: np.ndarray, landmarks: dict, landmark_name: str) -> tuple:
        """
        Converts a landmark's relative position to pixel coordinates.

        Args:
            image (np.ndarray): The image containing the hand.
            landmarks (dict): A dictionary of landmarks.
            landmark_name (str): The name of the landmark.

        Returns:
            tuple: The pixel coordinates (X, Y) of the landmark.

        Test Case:
            >>> image = np.zeros((100, 100), dtype=np.uint8)
            >>> landmarks = {'index': {'x': 0.5, 'y': 0.5}}
            >>> landmark_to_pixels(image, landmarks, 'index')
            (50, 50)
        """

        (height, width) = image.shape
        landmark_x = round(width * landmarks[landmark_name].x)  # X coordinate of the Mediapipe landmark # col
        landmark_y = round(height * landmarks[landmark_name].y)  # Y coordinate of the Mediapipe landmark # row
        return landmark_x, landmark_y
    
    def locate_hand_landmarks(self, image_path: str):
        """
        Detects hand landmarks in the input image using the specified HandLandmarker object.

        Args:
            image_path (str): The file path of the input image.

        Returns:
            vision.HandLandmarkerResult: The result of the hand landmark detection, which includes the detected landmarks and their confidence scores.

        Raises:
            None

        Test Case:
            # This function requires specific input files and a HandLandmarker object, hence a practical test would involve using actual files.
        """
        def landmarks_batch_to_pixel_coords(image: np.ndarray, landmarks: object) -> list:
            """
            Converts hand landmarks into pixel coordinates on the given image.

            Args:
                image (np.ndarray): The image on which the hand landmarks are detected. Should be in BGR format.
                landmarks (object): An object containing hand landmarks data, typically obtained from a hand tracking model.

            Returns:
                list: A list of tuples, each representing the (x, y) pixel coordinates of a hand landmark.

            Test Case:
                Assume `fake_image` is a numpy array representing an image and `fake_landmarks` is a mock object of landmarks.
                >>> fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
                >>> fake_landmarks = MockLandmarks()  # a mock landmarks object
                >>> pixels = landmarks_to_pixel_coordinates(fake_image, fake_landmarks)
                >>> type(pixels)
                <class 'list'>
                >>> type(pixels[0])
                <class 'tuple'>
            """
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return [self.landmark_to_pixels(gray, landmarks.hand_landmarks[0], idx) for idx in
                    range(len(landmarks.hand_landmarks[0]))]

        mediapipe_image = mp.Image.create_from_file(image_path)
        landmarks = self.detector.detect(mediapipe_image)
        if not landmarks.hand_landmarks:
            Warning(f"Warning: No landmarks detected for {image_path}")
            return None
        image = mediapipe_image.numpy_view().astype(np.uint8)
        return landmarks_batch_to_pixel_coords(image, landmarks), image
