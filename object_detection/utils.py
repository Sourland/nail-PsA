import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def resize_image(img: np.ndarray, new_size: int) -> np.ndarray:
    """
    Resizes an image by resizing the smaller axis to the desired size while maintaining the aspect ratio.

    Args:
        img (np.ndarray): The image to be resized.
        new_size (int): The new size of the smaller axis of the image.

    Returns:
        np.ndarray: The resized image.

    Raises:
        None

    Test Case:
        >>> img = np.zeros((50, 100, 3), dtype=np.uint8)
        >>> new_size = 25
        >>> resized_img = resize_image(img, new_size)
        >>> resized_img.shape
        (25, 50, 3)
    """
    scale_percent = new_size / min((img.shape[0], img.shape[1]))
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)

    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def locate_hand_landmarks(image_path: str, detector_path: str) -> vision.HandLandmarkerResult:
    """
    Detects hand landmarks in the input image using the specified HandLandmarker object.

    Args:
        image_path (str): The file path of the input image.
        detector_path (str): The file path of the HandLandmarker model.

    Returns:
        vision.HandLandmarkerResult: The result of the hand landmark detection, which includes the detected landmarks and their confidence scores.

    Raises:
        None

    Test Case:
        # This function requires specific input files and a HandLandmarker object, hence a practical test would involve using actual files.
    """
    base_options = python.BaseOptions(model_asset_path=detector_path)

    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands = 2,
                                           min_hand_detection_confidence=0.01)
    detector = vision.HandLandmarker.create_from_options(options)
    # Convert the input image to a mediapipe image
    mediapipe_image = mp.Image.create_from_file(image_path)

    # Use the HandLandmarker object to detect hand landmarks in the mediapipe image
    return mediapipe_image.numpy_view().astype(np.uint8), detector.detect(mediapipe_image)


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result: vision.HandLandmarkerResult) -> np.ndarray:
    """
    Draws hand landmarks and handedness on an input RGB image.

    Args:
        rgb_image (np.ndarray): The input RGB image.
        detection_result (vision.HandLandmarkerResult): The detection result containing hand landmarks and handedness.

    Returns:
        np.ndarray: The annotated image with hand landmarks and handedness visualized.

    Raises:
        None

    Test Case:
        # This function requires a specific detection_result object, hence a practical test would involve using an actual detection result.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def draw_landmarks_and_connections(image, landmarks, closest_points):
    """
    Draws landmarks as red circles and closest points as blue circles with green lines connecting them.

    Args:
        image (np.ndarray): The image on which landmarks and connections will be drawn.
        landmarks (list): A list of landmarks to be drawn on the image.
        closest_points (list): A list of tuples containing closest points for each landmark.

    Returns:
        np.ndarray: The image with landmarks and connections drawn.

    Test Case:
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> landmarks = [(30, 30), (70, 70)]
        >>> closest_points = [((25, 25), (35, 35)), ((65, 65), (75, 75))]
        >>> result_image = draw_landmarks_and_connections(image, landmarks, closest_points)
        # result_image will have red circles at landmarks, blue circles at closest points, and green lines connecting them
    """
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


def save_roi_image(roi, path):
    """
    Saves the region of interest (ROI) image to the specified path.

    Args:
        roi (np.ndarray): The ROI image to be saved.
        path (str): The file path where the image will be saved.

    Returns:
        None
    """
    if roi.size > 0 and roi is not None:
        cv2.imwrite(path, roi)
    else:
        print(f"Warning: The ROI image is empty or None. Skipping save operation for finger {os.path.basename(path)}.")

        
def get_segmentation_mask(image: np.ndarray, threshold: int = 11) -> np.ndarray:
    """
    Generates a segmentation mask for an image based on a grayscale threshold.

    This function converts an RGB image to grayscale and then creates a binary mask
    where pixels above a certain threshold are marked as foreground (255) and the rest as background (0).

    Args:
        image (np.ndarray): The RGB image from which to generate the mask.
        threshold (int, optional): The grayscale threshold for foreground-background segmentation. Defaults to 11.

    Returns:
        np.ndarray: A binary mask of the same size as the input image.

    Raises:
        ValueError: If the input image is not a 3-channel RGB image.

    Test Case:
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = get_segmentation_mask(image)
        >>> mask.shape
        (100, 100)
        >>> np.unique(mask)
        array([  0, 255], dtype=uint8)  # Only two values should be present in the mask: 0 and 255
    """
# Check if the image has three channels
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels. Received image with shape {}.".format(image.shape))

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Generate the mask
    mask = np.where(grayscale_image > threshold, 255, 0)

    return mask.astype(np.uint8)