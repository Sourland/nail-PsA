import cv2
import numpy as np

# Load the hand segmentation mask image
image = cv2.imread('results/SegMasks/seg_hand7.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find contours in the hand mask image
contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the hand, we'll find the convex hull around it
hand_contour = max(contours, key=cv2.contourArea)
hull = cv2.convexHull(hand_contour, returnPoints=False)

# Find convexity defects
defects = cv2.convexityDefects(hand_contour, hull)

# Check if defects were found
if defects is not None:
    # Filter out small defects to focus on the areas between fingers
    threshold = 10  # Example threshold, you may need to adjust this
    finger_defects = [defect for defect in defects if defect[0][3] > threshold]

    # Process each defect to isolate fingers
    finger_contours = []
    for i in range(len(finger_defects) - 1):
        start_idx, _, _, _ = finger_defects[i][0]
        end_idx, _, _, _ = finger_defects[i+1][0]

        # Segment the hand contour into individual fingers based on defect points
        finger_contour = hand_contour[start_idx: end_idx]
        finger_contours.append(finger_contour)

    # Now finger_contours contains individual finger segments
    # You can draw them on a new mask or the original image to verify
    finger_mask = np.zeros_like(gray_image)
    for i, finger in enumerate(finger_contours):
        if len(finger) > 0:  # Check if the contour is not empty
            cv2.drawContours(finger_mask, [finger], -1, (255), 2)
            cv2.putText(finger_mask, str(i), tuple(finger[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

    cv2.imshow('Finger Contours', finger_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
