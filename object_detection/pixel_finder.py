import cv2
import numpy as np


def find_furthest_black_pixel(landmark, segmented_image):
    # Find the contour of the hand object in the segmented image
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Get the first (largest) contour
    contour = max(contours, key=cv2.contourArea)

    furthest_distance = 0
    furthest_point = None

    # Iterate over each point in the contour
    for point in contour[:, 0, :]:
        x, y = point

        # Calculate the distance between the landmark and the contour point
        distance = np.linalg.norm(landmark - (x, y))

        if distance > furthest_distance:
            furthest_distance = distance
            furthest_point = (x, y)

    if furthest_point is not None:
        # Get the corresponding black pixel coordinates
        furthest_black_pixel = furthest_point
        return furthest_black_pixel

    return None

