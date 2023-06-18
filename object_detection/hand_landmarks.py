import pickle
from utils import *
import cv2
from draw_landmarks_on_image import draw_landmarks_on_image
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# import matplotlib.pyplot as plt
from pixel_finder import find_bounding_box, crop_image
from landmarks_constants import *



"""
Open Image and extract landmarks
"""
img = cv2.imread('../hand4.jpg', cv2.IMREAD_UNCHANGED)
detector = load_hand_landmarker('../hand_landmarker.task')
landmarks = locate_hand_landmarks('../hand4.jpg', detector)
which_hand = landmarks.handedness[0][0].category_name
annotated_image = draw_landmarks_on_image(
    mp.Image(image_format=mp.ImageFormat.SRGB, data=img).numpy_view(),
    landmarks
)
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("image", 300, 700)
#
# cv2.imshow('image', annotated_image)
# cv2.waitKey(0)
landmarks = landmarks.hand_landmarks[0]
"""
Segment Image and extract segmentation mask as a binary image
"""
file = open('../masks4.p', 'rb')
masks = pickle.load(file)
segmented_image = img.copy()
segmented_image[~masks[0]["segmentation"], :] = [0, 0, 0]
segmented_image[masks[0]["segmentation"], :] = [255, 255, 255]
segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image', segmented_image)
# cv2.waitKey(0)
# Load the segmented image and landmark coordinates
"""
Mediapipe landmarks are normalized in [0,1]. Use width and height to extract the landmark position in pixel coordinates
"""


"""
From the landmark 
"""

for area in areas_of_interest:
    top_left, bottom_right = find_bounding_box(segmented_image, landmarks, area, which_hand)
    extracted_image = crop_image(img, top_left, bottom_right)
    extracted_image = resize_image(extracted_image, 144)
    cv2.imwrite(f"../results/area{area}.jpg", extracted_image)


# top_left, bottom_right = find_bounding_box(segmented_image, landmarks, PINKY_TIP, which_hand)
# extracted_image = crop_image(img, top_left, bottom_right)
# cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), thickness=2)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.imwrite("../nail.jpg", extracted_image)
