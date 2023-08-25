import math
from utils import *
import cv2
from pixel_finder import find_bounding_box, crop_image
from landmarks_constants import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from joint_thickness import get_finger_vectors

BG_COLOR = (0, 0, 0)  # gray
MASK_COLOR = (255, 255, 255)  # white
HAND_PATH = '../hands/hand4.jpg'

img = cv2.imread(HAND_PATH, cv2.IMREAD_UNCHANGED)
detector = load_hand_landmarker('../hand_landmarker.task')
landmarks = locate_hand_landmarks(HAND_PATH, detector)
which_hand = landmarks.handedness[0][0].category_name
annotated_image = draw_landmarks_on_image(
    mp.Image(image_format=mp.ImageFormat.SRGB, data=img).numpy_view(),
    landmarks
)
cv2.imwrite('../landmarksHand.jpg', annotated_image)


# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


# Performs resizing and showing the image
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imwrite('../seg_mask.jpg', img)


# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='../selfie_multiclass_256x256.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options)

# Create the image segmenter
with vision.ImageSegmenter.create_from_options(options) as segmenter:
    # Loop through demo image(s):
    # Create the MediaPipe image file that will be segmented
    image = mp.Image.create_from_file(HAND_PATH)

    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(image)[0].numpy_view()

    # Generate solid color images for showing the output segmentation mask.
    image_data = image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((segmentation_result,) * 3, axis=-1) > 0.2
    output_image = np.where(condition, fg_image, bg_image)

    annotated_image = draw_landmarks_on_image(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=output_image).numpy_view(),
        landmarks
    )
    cv2.imwrite('../landmarksHandMask.jpg', annotated_image)


    resize_and_show(output_image)


landmarks = landmarks.hand_landmarks[0]
"""
Mediapipe landmarks are normalized in [0,1]. Use width and height to extract the landmark position in pixel coordinates
"""

"""
From the landmark 
"""

for area in areas_of_interest:
    top_left, bottom_right = find_bounding_box(segmentation_result, landmarks, area, which_hand)
    extracted_image = crop_image(img, top_left, bottom_right)
    extracted_image = resize_image(extracted_image, 244)
    cv2.imwrite(f"../results/area{area}.jpg", extracted_image)
    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), thickness=3)
    cv2.imwrite("../boxes.jpg", img)


