import numpy as np
from numpy import ndarray

from hand_landmarks import landmarks
from pixel_finder import landmark_to_pixels
from landmarks_constants import *
from utils import draw_landmarks_on_image

# Update the LANDMARKS_TO_PROCESS dictionary
LANDMARKS_TO_PROCESS = {
    "INDEX_FINGER_MCP": INDEX_FINGER_MCP,
    "INDEX_FINGER_PIP": INDEX_FINGER_PIP,
    "INDEX_FINGER_DIP": INDEX_FINGER_DIP,
    "INDEX_FINGER_TIP": INDEX_FINGER_TIP,

    "MIDDLE_FINGER_MCP": MIDDLE_FINGER_MCP,
    "MIDDLE_FINGER_PIP": MIDDLE_FINGER_PIP,
    "MIDDLE_FINGER_DIP": MIDDLE_FINGER_DIP,
    "MIDDLE_FINGER_TIP": MIDDLE_FINGER_TIP,

    "RING_FINGER_MCP": RING_FINGER_MCP,
    "RING_FINGER_PIP": RING_FINGER_PIP,
    "RING_FINGER_DIP": RING_FINGER_DIP,
    "RING_FINGER_TIP": RING_FINGER_TIP,

    "PINKY_MCP": PINKY_MCP,
    "PINKY_PIP": PINKY_PIP,
    "PINKY_DIP": PINKY_DIP,
    "PINKY_TIP": PINKY_TIP
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


def get_joint_thickness_euclidian(image_contour: np.ndarray, landmark_point: list[int]) -> tuple[
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
        closest_left, closest_right, thickness = get_joint_thickness_euclidian(image_contour, landmark_point)
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
    cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
    return (center, (width, height), theta)


def extract_roi(image, rect):
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
    x, y = int(center[0] - width // 2), int(center[1] - height // 2)
    roi = warped_image[y:y + int(height), x:x + int(width)]

    return roi, rotation_matrix


import cv2
import numpy as np


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


def transform_landmarks(landmarks, rotation_matrix, roi_origin):
    """
    Transforms the landmarks' positions based on the rotated cropped image.

    :param landmarks: List of landmarks from the original image in (x, y) format.
    :param rotation_matrix: 2x3 Affine rotation matrix used for cropping.
    :param roi_origin: The top-left corner (x, y) of the cropped ROI in the rotated image.
    :return: Transformed landmarks for the cropped image.
    """
    transformed_landmarks = []

    for landmark in landmarks:
        # Convert landmark to homogeneous coordinates
        original_coord = np.array([landmark[0], landmark[1], 1])

        # Apply the affine transformation (rotation + translation)
        rotated_coord = np.dot(rotation_matrix, original_coord)

        # Adjust for the cropping (subtracting the roi_origin)
        transformed_coord = (rotated_coord[0] - roi_origin[0], rotated_coord[1] - roi_origin[1])
        transformed_landmarks.append(transformed_coord)

    return np.array(transformed_landmarks)


if __name__ == "__main__":
    landmarks_pixel, mask = landmarks_to_pixel_coordinates('../seg_mask.jpg')
    image_contour = extract_contour(mask)
    plot_contour(mask, image_contour)

    results = process_landmarks(image_contour, landmarks_pixel)
    mean_distance = get_mean_pip_dip_distance(np.array(landmarks_pixel))

    mask_roi = mask.copy()
    # For demonstration: print the effective width for both MCP and TIP landmarks.
    for key, idx in LANDMARKS_TO_PROCESS.items():
        print(f"{key} EFFECTIVE WIDTH: {results[key]['thickness'] / mean_distance}")
        if key.endswith("TIP"):
            draw_closest_points(mask_roi, landmarks_pixel[idx], image_contour, case="TIP")
        else:
            draw_closest_points(mask_roi, landmarks_pixel[idx], image_contour)

    # Extract and store all the ROIs
    rois = []

    # Draw bounding boxes using MCP and TIP landmarks for each finger.
    finger_names = ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY"]
    new_landmarks = dict()
    for ctr, finger in enumerate(finger_names):
        mcp = finger + "_MCP"
        pip = finger + "_PIP"
        dip = finger + "_DIP"
        tip = finger + "_TIP"

        mcp_left = results[mcp]["closest_left"]
        mcp_right = results[mcp]["closest_right"]
        pip_left = results[pip]["closest_left"]
        pip_right = results[pip]["closest_right"]
        dip_left = results[dip]["closest_left"]
        dip_right = results[dip]["closest_right"]
        tip_left = results[tip]["closest_left"]
        tip_right = results[tip]["closest_right"]

        rect = draw_bounding_box(mask_roi,
                                 [mcp_left, mcp_right, pip_right, pip_left, dip_left, dip_right, tip_right, tip_left])

        roi, rotation_matrix = extract_roi(mask, rect)
        center, (width, height), theta = rect

        if width > height:
            width, height = height, width
            theta += 90  # Adjust the rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1)

        roi_origin = (int(center[0] - width // 2), int(center[1] - height // 2))

        new_landmarks[finger] = transform_landmarks(
            [landmarks_pixel[LANDMARKS_TO_PROCESS[mcp]],
             landmarks_pixel[LANDMARKS_TO_PROCESS[pip]],
             landmarks_pixel[LANDMARKS_TO_PROCESS[dip]],
             landmarks_pixel[LANDMARKS_TO_PROCESS[tip]]],
            rotation_matrix, roi_origin
        )
        print(new_landmarks[finger])
        plot_roi(roi, new_landmarks[finger], ctr)
        i = 0
    # Draw landmarks on the image and display it.
    mask_roi = draw_landmarks_on_image(mask_roi, landmarks)
    cv2.imwrite('output_with_points.jpg', mask_roi)
    cv2.namedWindow('Modified Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Modified Image', mask_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
