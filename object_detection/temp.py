import numpy as np

from object_detection.landmarks import adjust_for_roi_crop, transform_point
from object_detection.roi_extraction import extract_roi

def is_point_inside_rect(image, point, rect):
    (center, (width, height), theta) = rect
    roi, rotation_matrix = extract_roi(image, rect)
    rotated_point = transform_point(point, rotation_matrix)
    adjusted_point = adjust_for_roi_crop(rotated_point, rect[0], rect[1])

    # Check if the adjusted point is within the rectangle's boundaries
    half_width, half_height = width / 2, height / 2
    return (-half_width <= adjusted_point[0] <= half_width) and (-half_height <= adjusted_point[1] <= half_height)


def process_neighbor_finger(this_pip, this_dip, neighbor_key, landmarks_per_finger, landmark_pixels, rect, rgb_mask, roi, used_fingers):
    neighbor_pip = np.array(landmark_pixels[landmarks_per_finger[neighbor_key][1]])
    neighbor_dip = np.array(landmark_pixels[landmarks_per_finger[neighbor_key][2]])

    if is_point_inside_rect(rgb_mask, neighbor_dip, rect) or is_point_inside_rect(rgb_mask, neighbor_pip, rect):
        used_fingers.append(neighbor_key)
        return adjust_roi_for_neighbor(this_pip, this_dip, neighbor_pip, neighbor_dip, rect, roi)
    return roi


def adjust_roi_for_neighbor(this_pip, this_dip, neighbor_pip, neighbor_dip, rect, roi):
    # Transform the neighbor landmarks
    transformed_neighbor_pip = transform_point(neighbor_pip, rect[1])
    transformed_neighbor_dip = transform_point(neighbor_dip, rect[2])

    # Compute middle points and adjust ROI
    pip_middle = (this_pip + transformed_neighbor_pip) // 2
    dip_middle = (this_dip + transformed_neighbor_dip) // 2

    # Calculate line slope and intercept
    slope, intercept = calculate_line_slope_intercept(pip_middle, dip_middle)

    # Adjust the ROI based on pip and dip positions
    return adjust_roi_based_on_line(roi, pip_middle, slope, intercept)



def calculate_line_slope_intercept(pip_middle, dip_middle):
    slope = (pip_middle[1] - dip_middle[1]) / (pip_middle[0] - dip_middle[0])
    intercept = pip_middle[1] - slope * pip_middle[0]
    return slope, intercept

def adjust_roi_based_on_line(roi, pip_middle, slope, intercept):
    pip_left = pip_middle[0] < (slope * pip_middle[1] + intercept)
    if pip_left:
        roi[:, int(pip_middle[0]):] = 0
    else:
        roi[:, :int(pip_middle[0])] = 0
    return roi


