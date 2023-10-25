import json
import pickle

import cv2
import numpy as np
from numpy import ndarray
import mediapipe as mp
from pixel_finder import landmark_to_pixels
from landmarks_constants import *
from utils import draw_landmarks_on_image, load_hand_landmarker, locate_hand_landmarks

# Update the LANDMARKS_TO_PROCESS dictionary
LANDMARKS_TO_PROCESS = {
    "INDEX": {
        "MCP": INDEX_FINGER_MCP,
        "PIP": INDEX_FINGER_PIP,
        "DIP": INDEX_FINGER_DIP,
        "TIP": INDEX_FINGER_TIP,
    },
    "MIDDLE": {
        "MCP": MIDDLE_FINGER_MCP,
        "PIP": MIDDLE_FINGER_PIP,
        "DIP": MIDDLE_FINGER_DIP,
        "TIP": MIDDLE_FINGER_TIP,
    },
    "RING": {
        "MCP": RING_FINGER_MCP,
        "PIP": RING_FINGER_PIP,
        "DIP": RING_FINGER_DIP,
        "TIP": RING_FINGER_TIP,
    },
    "PINKY": {
        "MCP": PINKY_MCP,
        "PIP": PINKY_PIP,
        "DIP": PINKY_DIP,
        "TIP": PINKY_TIP,
    },
}


def landmarks_to_pixel_coordinates(image_path: str) -> tuple[list, np.ndarray]:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return [landmark_to_pixels(gray, landmarks.hand_landmarks[0], idx) for idx in
            range(len(landmarks.hand_landmarks[0]))], image


def extract_contour(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def find_closest_points(point, contour, case=None):
    if case == "TIP":
        contour = contour[contour[:, 0, 1] > point[1]]

    left_contour = contour[contour[:, 0, 0] < point[0]]
    right_contour = contour[contour[:, 0, 0] > point[0]]

    distances_left = np.sqrt(np.sum((left_contour - point) ** 2, axis=2))
    distances_right = np.sqrt(np.sum((right_contour - point) ** 2, axis=2))

    return left_contour[np.argmin(distances_left)], right_contour[np.argmin(distances_right)]


def get_closest_contour_points(image_contour: np.ndarray, landmark_point: list[int]) -> tuple[
    np.ndarray, np.ndarray, float]:
    closest_left, closest_right = find_closest_points(landmark_point, image_contour)
    thickness = np.linalg.norm((closest_left - closest_right))
    return closest_left, closest_right, thickness


def get_mean_pip_dip_distance(landmarks: np.ndarray) -> ndarray:
    return np.mean([np.linalg.norm(landmarks[i] - landmarks[j]) for i, j in [
        (INDEX_FINGER_PIP, INDEX_FINGER_DIP),
        (MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP),
        (RING_FINGER_PIP, RING_FINGER_DIP),
        (PINKY_PIP, PINKY_DIP)
    ]])


def draw_closest_points(image: np.ndarray, landmark_point: list[int], contour: np.ndarray, case=None) -> None:
    closest_left, closest_right = find_closest_points(landmark_point, contour, case)
    cv2.circle(image, tuple(closest_left[0]), 3, (0, 0, 255), -1)
    cv2.circle(image, tuple(closest_right[0]), 3, (0, 0, 255), -1)
    cv2.line(image, tuple(landmark_point), tuple(closest_left[0]), (0, 0, 255), 1)
    cv2.line(image, tuple(landmark_point), tuple(closest_right[0]), (0, 0, 255), 1)


def plot_contour(image: np.ndarray, contour: np.ndarray) -> None:
    image_copy = image.copy()
    cv2.drawContours(image_copy, [contour], 0, (0, 255, 0), 2)
    cv2.namedWindow('Image with Contour', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Contour', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_landmarks(image_contour: np.ndarray, landmarks_pixel: list[list[int]]) -> dict[str, dict[str, any]]:
    results = {}
    for landmark_name, landmark_index in LANDMARKS_TO_PROCESS.items():
        landmark_point = landmarks_pixel[landmark_index]
        closest_left, closest_right, thickness = get_closest_contour_points(image_contour, landmark_point)
        results[landmark_name] = {
            "id": landmark_name,
            "thickness": thickness,
            "closest_left": closest_left,
            "closest_right": closest_right
        }
    return results


def draw_bounding_box(image: np.ndarray, points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Draw a rotated bounding box around the given points on the image."""
    # Convert the list of points to a suitable format for cv2.minAreaRect
    rect_points = np.array(points).reshape(-1, 1, 2).astype(int)

    # Compute the rotated bounding box
    rect = cv2.minAreaRect(rect_points)
    (center, (width, height), theta) = rect

    # Ensure the rectangle is in portrait orientation
    if width > height:
        width, height = height, width
        theta -= 90  # Adjust the rotation

    box = cv2.boxPoints(((center[0], center[1]), (width, height), theta)).astype(int)

    # Draw the rotated bounding box on the image
    cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=1)
    return (center, (width, height), theta)


def extract_roi(image, rect, rotation=False):
    """
    Extracts and returns the region of interest inside the rectangle.

    :param image: The original image.
    :param rect: A tuple that contains the center (x, y), size (width, height), and angle of the rectangle.
    :return: The extracted region of interest.
    """

    # Scale up the width and height by 10% for a margin.
    center, size, theta = rect
    width, height = size
    width += width * 0.1
    height += height * 0.1

    # Obtain the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1)

    # Perform the affine transformation
    warped_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    # Extract the ROI
    x = np.clip(int(center[0] - width // 2), 0, width - 1).astype(int)
    y = np.clip(int(center[1] - height // 2), 0, height - 1).astype(int)
    roi = warped_image[y:y + int(height), x:x + int(width)]

    return roi, rotation_matrix


def plot_roi(roi, landmarks, finger_ctr):
    """
    Plots the given region of interest with landmarks and connects them with a line.

    :param roi: The region of interest to be plotted.
    :param landmarks: List of landmarks to be drawn.
    :param finger_ctr: Counter for naming the saved image.
    """

    # Draw the landmarks
    for i, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        cv2.circle(roi, (x, y), 2, (0, 255, 0), -1)

        # Connect landmarks with a line
        if i > 0:
            prev_x, prev_y = int(landmarks[i - 1][0]), int(landmarks[i - 1][1])
            cv2.line(roi, (prev_x, prev_y), (x, y), (255, 0, 0), 1)

    # Save the image with landmarks
    cv2.imwrite(f'finger_{finger_ctr}.jpg', roi)

    # Display the image
    cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
    cv2.imshow('ROI', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transform_landmarks(landmarks, center, wh, theta):
    """
    Adjusts the landmarks to a rotated bounding box ROI.

    Parameters:
    - landmarks: List of (x, y) tuples for each landmark.
    - center: (x, y) tuple for the center of the bounding box.
    - wh: (width, height) tuple for the bounding box.
    - theta: Rotation angle in radians.

    Returns:
    List of adjusted (x, y) tuples for each landmark.
    """
    # Convert landmarks to numpy array for easy calculations
    landmarks = np.array(landmarks, dtype=np.float32)

    # 1. Translate landmarks so that center of bounding box is at (0, 0)
    landmarks -= center

    # 2. Rotate landmarks by -theta to "unrotate" them
    rotation_matrix = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta), np.cos(-theta)]
    ])
    landmarks = np.dot(landmarks, rotation_matrix.T)  # .T because we want to transform columns

    # 3. Scale landmarks according to size of ROI
    landmarks[:, 0] /= wh[0]  # x-coordinate
    landmarks[:, 1] /= wh[1]  # y-coordinate

    return np.array(landmarks)


def load_landmarks(load_path):
    """
    Load HandLandmarkResult from a pickle file.

    Parameters:
        load_path: str, path to the pickle file

    Returns:
        HandLandmarkResult: The loaded landmark instance
    """
    with open(load_path, 'rb') as f:
        landmarks = pickle.load(f)
    return landmarks


def load_and_annotate_image(hand_path, detector_path):
    img = cv2.imread(hand_path, cv2.COLOR_RGBA2RGB)
    detector = load_hand_landmarker(detector_path)
    landmarks = locate_hand_landmarks(hand_path, detector)
    annotated_image = draw_landmarks_on_image(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=img).numpy_view(),
        landmarks
    )
    cv2.imwrite('../landmarksHand.jpg', annotated_image)
    return img, landmarks


HAND_PATH = '../dataset/hands/swolen/hand40.jpg'


# Function to compute the midpoint
def compute_midpoint(point1, point2):
    return (point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2


if __name__ == "__main__":
    FINGERS = ["INDEX", "MIDDLE", "RING", "PINKY"]
    rectangles = {"INDEX": [], "MIDDLE": [], "RING": [], "PINKY": []}
    image, landmarks = load_and_annotate_image(HAND_PATH, '../hand_landmarker.task')
    landmarks_pixel, mask = landmarks_to_pixel_coordinates(HAND_PATH)
    new_landmarks = dict()
    for finger in FINGERS:
        finger_landmarks = LANDMARKS_TO_PROCESS[finger]
        for idx, landmark in enumerate(finger_landmarks.values()):
            # print(joint_neighbours_right_hand[landmark])
            if finger == "PINKY":

                midpoint_left = compute_midpoint(landmarks_pixel[landmark],
                                                 landmarks_pixel[joint_neighbours_right_hand[landmark]])
                midpoint_right = (1, landmarks_pixel[landmark][1])

                rectangles[finger].append(np.array(midpoint_left))
                rectangles[finger].append(np.array(midpoint_right))
                # cv2.circle(image, midpoint_right, 3, (0, 0, 255), -1)
                # cv2.putText(image, str(idx), midpoint_right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # cv2.circle(image, midpoint_left, 3, (0, 0, 255), -1)
                # cv2.putText(image, str(idx), midpoint_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                midpoint_left = compute_midpoint(landmarks_pixel[landmark],
                                                 landmarks_pixel[joint_neighbours_right_hand[landmark][0]])
                midpoint_right = compute_midpoint(landmarks_pixel[landmark],
                                                  landmarks_pixel[joint_neighbours_right_hand[landmark][1]])
                rectangles[finger].append(np.array(midpoint_left))
                rectangles[finger].append(np.array(midpoint_right))
                # rectangles[finger].append(np.array((landmarks_pixel[landmark][0], 1)))
        rect = draw_bounding_box(image, rectangles[finger])
        roi, rotation_matrix = extract_roi(image, rect)
        center, (width, height), theta = rect
        if width > height:
            width, height = height, width
            theta += 90  # Adjust the rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1)

        roi_origin = (int(center[0] - width // 2), int(center[1] - height // 2))
        finger_landmarks = [landmarks_pixel[finger_landmarks["MCP"]],
                            landmarks_pixel[finger_landmarks["PIP"]],
                            landmarks_pixel[finger_landmarks["DIP"]],
                            landmarks_pixel[finger_landmarks["TIP"]]]
        new_landmarks[finger] = transform_landmarks(finger_landmarks, center, (width, height), theta)
        plot_roi(roi, new_landmarks[finger], finger)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
