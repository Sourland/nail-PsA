import cv2
import numpy as np
from hand_landmarks import landmarks
from pixel_finder import landmark_to_pixels
from landmarks_constants import *
from utils import draw_landmarks_on_image

landmarks_components = landmarks.hand_landmarks[0]


def landmarks_to_pixel_coordinates(image_path: str) -> tuple[list, np.ndarray]:
    """
       Converts landmarks to pixel coordinates based on the given image path.

       Args:
           image_path (str): Path to the image file.

       Returns:
           tuple[list, np.ndarray]: A tuple containing landmarks in pixel format and the read image.
   """
    output_image = cv2.imread(image_path)
    landmarks_pixel = [landmark_to_pixels(cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY), landmarks_components, idx)
                       for idx, landmark in enumerate(landmarks_components)]
    return landmarks_pixel, output_image


def extract_contour(image: np.ndarray) -> np.ndarray:
    """
    Extract the largest contour from the given image.

    Args:
        image (np.ndarray): The image from which to extract the contour.

    Returns:
        np.ndarray: The largest contour found in the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def plot_contour(image: np.ndarray, contour: np.ndarray) -> None:
    """
    Plots the given contour on the provided image.

    Args:
        image (np.ndarray): The image on which to plot the contour.
        contour (np.ndarray): The contour to plot on the image.
    """
    # Copy the image to avoid modifying the original
    image_copy = image.copy()
    # Draw the contour on the copied image
    cv2.drawContours(image_copy, [contour], 0, (0, 255, 0), 2)  # Drawing the contour in green color
    # Display the image with contour
    cv2.imshow('Image with Contour', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_joint_thickness_euclidian(image_contour: np.ndarray, landmark_point: list[int]) \
        -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calculates the joint thickness based on the euclidian distance between the closest points
    in the contour to the given landmark point.

    Args:
        image_contour (np.ndarray): The contour of the image.
        landmark_point (list[int]): The pixel coordinates of the landmark point.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: The leftmost and rightmost closest points and their distance (thickness).
    """
    point = np.array(landmark_point)
    left_contour = image_contour[image_contour[:, 0, 0] < point[0]]
    right_contour = image_contour[image_contour[:, 0, 0] > point[0]]

    distances_left = np.sqrt(np.sum((left_contour - point) ** 2, axis=2))
    distances_right = np.sqrt(np.sum((right_contour - point) ** 2, axis=2))

    closest_left = left_contour[np.argmin(distances_left)]
    closest_right = right_contour[np.argmin(distances_right)]
    thickness = np.linalg.norm((closest_left - closest_right))
    return closest_left, closest_right, thickness


def get_mean_pip_dip_distance(landmarks: list[list[int]]) -> float:
    """
     Calculates the mean distance between PIP and DIP landmarks.

     Args:
         landmarks (list[list[int]]): A list of landmarks pixel coordinates.

     Returns:
         float: The mean PIP-DIP distance.
     """
    landmarks = np.array(landmarks)
    distances = [
        np.linalg.norm(landmarks[INDEX_FINGER_PIP] - landmarks[INDEX_FINGER_DIP]),
        np.linalg.norm(landmarks[MIDDLE_FINGER_PIP] - landmarks[MIDDLE_FINGER_DIP]),
        np.linalg.norm(landmarks[RING_FINGER_PIP] - landmarks[RING_FINGER_DIP]),
        np.linalg.norm(landmarks[PINKY_PIP] - landmarks[PINKY_DIP])
    ]
    return np.mean(distances)


def process_landmarks(image_contour: np.ndarray, landmarks_pixel: list[list[int]]) -> dict[str, dict[str, any]]:
    """
    Processes the landmarks to retrieve the closest contour points and their thickness.

    Args:
        image_contour (np.ndarray): The contour of the image.
        landmarks_pixel (list[list[int]]): List of landmark pixel coordinates.

    Returns:
        dict[str, dict[str, any]]: Processed landmarks information including id, thickness, and closest contour points.
    """
    landmarks_to_process = {
        "INDEX_FINGER_PIP": 6,
        "INDEX_FINGER_DIP": 7,
        "MIDDLE_FINGER_PIP": 10,
        "MIDDLE_FINGER_DIP": 11,
        "RING_FINGER_PIP": 14,
        "RING_FINGER_DIP": 15,
        "PINKY_PIP": 18,
        "PINKY_DIP": 19
    }
    results = {}
    for landmark_name, landmark_index in landmarks_to_process.items():
        landmark_point = landmarks_pixel[landmark_index]
        closest_left, closest_right, thickness = get_joint_thickness_euclidian(image_contour, landmark_point)
        results[landmark_name] = {
            "id": landmark_name,
            "thickness": thickness,
            "closest_left": closest_left,
            "closest_right": closest_right
        }
    return results


def draw_closest_points(image: np.ndarray, landmark_point: tuple, contour: np.ndarray) -> np.ndarray:
    """
    Draw the closest points of the contour to the landmark on the image.

    Args:
        image (np.ndarray): The image to be modified.
        landmark_point (tuple): The coordinates of the landmark point.
        contour (np.ndarray): The contour of the image.

    Returns:
        np.ndarray: The modified image with the closest points drawn.
    """
    point = np.array(landmark_point)

    left_contour = contour[contour[:, 0, 0] < point[0]]
    right_contour = contour[contour[:, 0, 0] > point[0]]

    distances_left = np.sqrt(np.sum((left_contour - point) ** 2, axis=2))
    distances_right = np.sqrt(np.sum((right_contour - point) ** 2, axis=2))

    closest_left = left_contour[np.argmin(distances_left)]
    closest_right = right_contour[np.argmin(distances_right)]

    # cv2.circle(image, tuple(point), 3, (0, 255, 0), -1)  # Green circle for landmark
    cv2.circle(image, tuple(closest_left[0]), 3, (0, 0, 255), -1)  # Red circle for left point
    cv2.circle(image, tuple(closest_right[0]), 3, (0, 0, 255), -1)  # Red circle for right point

    cv2.line(image, tuple(point), tuple(closest_left[0]), (0, 0, 255), 1)  # Line to left point
    cv2.line(image, tuple(point), tuple(closest_right[0]), (0, 0, 255), 1)  # Line to right point

    return image


if __name__ == "__main__":
    landmarks_to_process = {
        "INDEX_FINGER_PIP": 6,
        "INDEX_FINGER_DIP": 7,
        "MIDDLE_FINGER_PIP": 10,
        "MIDDLE_FINGER_DIP": 11,
        "RING_FINGER_PIP": 14,
        "RING_FINGER_DIP": 15,
        "PINKY_PIP": 18,
        "PINKY_DIP": 19
    }
    landmarks_pixel, output_image = landmarks_to_pixel_coordinates('../seg_mask.jpg')
    image_contour = extract_contour(output_image)
    plot_contour(output_image, image_contour)
    results = process_landmarks(image_contour, landmarks_pixel)
    mean_distance = get_mean_pip_dip_distance(landmarks_pixel)

    for key in results.keys():
        print(f"{key} EFFECTIVE WIDTH: {results[key]['thickness'] / mean_distance}")

        # Draw closest points for each landmark on the output_image
        output_image = draw_closest_points(output_image, landmarks_pixel[landmarks_to_process[key]], image_contour)

    output_image = draw_landmarks_on_image(output_image, landmarks)
    # Save or show the modified image with drawn points
    cv2.imwrite('output_with_points.jpg', output_image)
    cv2.imshow('Modified Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
