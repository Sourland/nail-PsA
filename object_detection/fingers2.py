import os

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from landmarks_constants import landmarks_per_finger, joint_neighbours_right_hand, joint_neighbours_left_hand
from object_detection.utils import load_hand_landmarker, locate_hand_landmarks
import subprocess


def get_landmarks(image_path, detector_path):
    img = cv2.imread(image_path, cv2.COLOR_RGBA2RGB)
    detector = load_hand_landmarker(detector_path)
    landmarks = locate_hand_landmarks(image_path, detector)
    return landmarks


def get_landmarks_pixel(landmarks, image_shape):
    h, w = image_shape
    landmarks_pixel = []
    for landmark in landmarks:
        if isinstance(landmark, tuple):
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
        else:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
        landmarks_pixel.append((x, y))
    return landmarks_pixel


def get_finger_roi(landmark_data, finger: str, lambda_val=2 / 3):
    # Extract indices of the landmarks of the index finger
    finger_indices = [landmark for landmark in landmarks_per_finger[finger]]

    # Use these indices to extract the landmarks from the provided data
    landmarks = [landmark_data[i] for i in finger_indices]
    neighbors = []
    for index in finger_indices:
        neighbors.append(joint_neighbours_left_hand[index])
    neighbors = np.array(neighbors)

    finger_tip = landmarks[-1]
    left_middle_tip = ((1 - lambda_val) * finger_tip.x + lambda_val * landmark_data[neighbors[-1, 0]].x,
                       (1 - lambda_val) * finger_tip.y + lambda_val * landmark_data[neighbors[-1, 0]].y)

    right_middle_tip = ((1 - lambda_val) * finger_tip.x + lambda_val * landmark_data[neighbors[-1, 1]].x,
                        (1 - lambda_val) * finger_tip.y + lambda_val * landmark_data[neighbors[-1, 1]].y)
    finger_mcp = landmarks[0]
    left_middle_mcp = ((1 - lambda_val) * finger_mcp.x + lambda_val * landmark_data[neighbors[0, 0]].x,
                       (1 - lambda_val) * finger_mcp.y + lambda_val * landmark_data[neighbors[0, 0]].y)

    right_middle_mcp = ((1 - lambda_val) * finger_mcp.x + lambda_val * landmark_data[neighbors[0, 1]].x,
                        (1 - lambda_val) * finger_mcp.y + lambda_val * landmark_data[neighbors[0, 1]].y)

    return [right_middle_tip, left_middle_tip, left_middle_mcp, right_middle_mcp]


def get_bounding_box(points):
    # Convert the points into a numpy array of shape Nx2
    points_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    # Compute the bounding box around the points
    x, y, w, h = cv2.boundingRect(points_np)
    return x, y, w, h


# if __name__ == "__main__":
#     PATH = "../dataset/hands/swolen/hand81.jpg"
#     image = cv2.imread(PATH, cv2.COLOR_RGBA2RGB)
#     OUTPUT_PATH = "../results/SegMasks/seg_" + os.path.basename(PATH)
#     cmd = ["backgroundremover", "-i", PATH, "-m", "u2net", "-o",  OUTPUT_PATH]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     output = cv2.imread("output.png", cv2.COLOR_RGBA2RGB)
#
#     landmarks = get_landmarks("../dataset/hands/healthy/hand1.jpg", "../hand_landmarker.task").hand_landmarks[0]
#     roi = get_finger_roi(landmarks, "INDEX")
#     roi_pixel = get_landmarks_pixel(roi, image.shape[:2])
#     x, y, w, h = get_bounding_box(roi_pixel)
#
#     roi_image = image[y:y + h, x:x + w].astype(np.uint8)
#
#     # roi_image = segment_image(roi_image, "../selfie_multiclass_256x256.tflite")
#
#     cv2.imwrite("roi.jpg", output)
#     # Display the ROI
#     # cv2.namedWindow("ROI Image", cv2.WINDOW_NORMAL)
#     # cv2.imshow("ROI Image", roi_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

def process_image(PATH, OUTPUT_DIR):
    image = cv2.imread(PATH, cv2.COLOR_RGBA2RGB)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "seg_" + os.path.basename(PATH))
    cmd = ["backgroundremover", "-i", PATH, "-m", "u2net", "-o", OUTPUT_PATH]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = cv2.imread(OUTPUT_PATH, cv2.COLOR_RGBA2RGB)
    print(os.path.basename(PATH))
    # Assuming the landmarks are same for all images (which may not be the case, adjust if needed)
    landmarks = get_landmarks(PATH, "../hand_landmarker.task")
    if not landmarks.hand_landmarks:
        return
    else:
        landmarks = landmarks.hand_landmarks[0]
    roi = get_finger_roi(landmarks, "INDEX")
    roi_pixel = get_landmarks_pixel(roi, image.shape[:2])
    x, y, w, h = get_bounding_box(roi_pixel)

    roi_image = image[y:y + h, x:x + w].astype(np.uint8)
    # roi_image = segment_image(roi_image, "../selfie_multiclass_256x256.tflite")

    roi_output_path = os.path.join(OUTPUT_DIR, "roi_" + os.path.basename(PATH))
    # cv2.imwrite(roi_output_path, output)


if __name__ == "__main__":
    DIR_PATH = "../dataset/hands/swolen/"
    OUTPUT_DIR = "../results/SegMasks/"

    # Make sure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for image_name in os.listdir(DIR_PATH):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):  # checking file extension
            image_path = os.path.join(DIR_PATH, image_name)
            process_image(image_path, OUTPUT_DIR)
