import cv2
import numpy as np


def find_bounding_box(image, landmark):
    def find_nearest_black_pixel_above(image, landmark):
        x, y = landmark

        # Iterate from the landmark's y-coordinate towards the top of the image
        for i in range(y, -1, -1):
            if image[i, x] == 0:
                return np.abs(y - i)  # Return the coordinates of the nearest black pixel

        return None

    def find_nearest_black_pixel_left(image, landmark):
        x, y = landmark

        # Iterate from the landmark's x-coordinate towards the left of the image
        for i in range(x, -1, -1):
            if image[y, i] == 0:
                return np.abs(x - i)  # Return the coordinates of the nearest black pixel

        return None  # If no black pixel is found to the left of the landmark

    def find_nearest_black_pixel_right(image, landmark):
        x, y = landmark
        image_width = image.shape[1]

        # Iterate from the landmark's x-coordinate towards the right of the image
        for i in range(x, image_width):
            if image[y, i] == 0:
                return np.abs(x - i)  # Return the coordinates of the nearest black pixel

        return None  # If no black pixel is found to the right of the landmark

    above = find_nearest_black_pixel_above(image, landmark)
    left = find_nearest_black_pixel_left(image, landmark)
    right = find_nearest_black_pixel_right(image, landmark)

    return above, left, right
