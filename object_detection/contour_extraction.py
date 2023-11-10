import cv2
import numpy as np

def extract_contour(image: np.ndarray) -> np.ndarray:
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if not contours:
        return

    # Sort the contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Return the largest contour by area
    return np.squeeze(contours[0])

def closest_contour_point(point, contour):
    idx = np.argmin(np.linalg.norm(contour - point, axis=1))
    closest_point = contour[idx]
    return closest_point

def get_left_and_right_contour_points(landmark, contour):
    left_contour = contour[(contour[:, 0] <= landmark[0])]
    right_contour = contour[(contour[:, 0] > landmark[0])]

    return left_contour, right_contour

def reorient_contour(contour, orientation='clockwise'):
    # Calculate the contour area, and consider the orientation
    area = cv2.contourArea(contour, oriented=True)
    
    # Check if the contour is already in the desired orientation
    if (area < 0 and orientation == 'clockwise') or (area > 0 and orientation == 'counterclockwise'):
        # The contour is already in the desired orientation, no change needed
        return contour
    else:
        # The contour is in the opposite orientation, so reverse it
        return np.flipud(contour)


def create_finger_contour(tip_proxy, left_contour, right_contour, closest_mcp_left, closest_mcp_right):
    # Get the indices of the closest points on the left and right contours
    idx_mcp_left = np.where(np.all(left_contour == closest_mcp_left, axis=1))[0][0]
    idx_mcp_right = np.where(np.all(right_contour == closest_mcp_right, axis=1))[0][0]
    idx_tip_proxy = np.where(np.all(np.vstack((left_contour, right_contour)) == tip_proxy, axis=1))[0][0]
    
    # Get the segments of the left and right contours
    left_segment = left_contour[:idx_mcp_left]
    right_segment = right_contour[idx_mcp_right:idx_tip_proxy+1]
    
    # Ensure segments are in correct order
    left_segment = left_segment if (left_segment[-1] == tip_proxy).all() else left_segment[::-1]
    right_segment = right_segment if (right_segment[0] == tip_proxy).all() else right_segment[::-1]
    
    # Create the connecting line between MCP points
    connecting_line = np.array([closest_mcp_left, closest_mcp_right])
    
    # Concatenate the segments and the connecting line to form the full contour
    full_contour = np.concatenate((left_segment, right_segment, connecting_line))

    return full_contour
