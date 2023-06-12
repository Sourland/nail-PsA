import cv2
import numpy as np


def find_bounding_box(image, landmarks):
    x, y = landmarks
    height, width = image.shape
    left = right = 0

    # Iterate from the landmark's x-coordinate towards the left of the image
    for i in range(x, -1, -1):
        if image[y, i] == 0:
            left = np.abs(x - i)  # Return the coordinates of the nearest black pixel
            break

    # Iterate from the landmark's x-coordinate towards the right of the image
    for i in range(x, width):
        if image[y, i] == 0:
            right = np.abs(x - i)  # Return the coordinates of the nearest black pixel
            break

    rect_width = rect_height = left + right
    top_left = (np.clip(x - rect_width // 2, 0, width), np.clip(y - rect_height // 2, 0, height))
    bottom_right = (np.clip(x + rect_width // 2, 0, width), np.clip(y + rect_height // 2, 0, height))

    return top_left, bottom_right


def check_traveling_boundaries(image, landmarks, landmark):
    ...


def crop_image(image, top_left, bottom_right):
    x_top_left, y_top_left = top_left
    x_bottom_right, y_bottom_right = bottom_right

    return image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
