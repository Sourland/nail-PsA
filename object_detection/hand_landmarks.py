import cv2
import math
from utils import *
from landmarks_constants import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pixel_finder import find_bounding_box, crop_image

# from joint_thickness import get_finger_vectors

BG_COLOR = (0, 0, 0)  # gray
MASK_COLOR = (255, 255, 255)  # white
HAND_PATH = '../hands/hand_swolo2.png'
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480


def load_and_annotate_image(hand_path, detector_path):
    img = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
    detector = load_hand_landmarker(detector_path)
    landmarks = locate_hand_landmarks(hand_path, detector)
    annotated_image = draw_landmarks_on_image(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=img).numpy_view(),
        landmarks
    )
    cv2.imwrite('../landmarksHand.jpg', annotated_image)
    return img, landmarks


def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imwrite('../seg_mask.jpg', img)
    return img


def segment_image(image, model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(base_options=base_options)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        segmentation_result = segmenter.segment(image)
        segmentation_mask = segmentation_result[0].numpy_view()
    return segmentation_mask


def process_mask(segmentation_mask, image_data):
    thresholded_mask = np.where(segmentation_mask > 0.1, 1, 0)
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((thresholded_mask,) * 3, axis=-1)
    output_image = np.where(condition, fg_image, bg_image)
    return output_image, thresholded_mask


def extract_regions(img, thresholded_mask, landmarks):
    landmarks = landmarks.hand_landmarks[0]
    for area in areas_of_interest:
        top_left, bottom_right = find_bounding_box(thresholded_mask, landmarks, area, which_hand)
        extracted_image = crop_image(img, top_left, bottom_right)
        extracted_image = resize_image(extracted_image, 244)
        cv2.imwrite(f"../results/area{area}.jpg", extracted_image)
        cv2.imwrite("../boxes.jpg", img)


image, landmarks = load_and_annotate_image(HAND_PATH, '../hand_landmarker.task')
which_hand = landmarks.handedness[0][0].category_name

mp_image = mp.Image.create_from_file(HAND_PATH)
segmentation_mask = segment_image(mp_image, '../selfie_multiclass_256x256.tflite')
output_image, thresholded_mask = process_mask(segmentation_mask, mp_image.numpy_view())

annotated_image = draw_landmarks_on_image(
    mp.Image(image_format=mp.ImageFormat.SRGB, data=output_image).numpy_view(),
    landmarks
)
cv2.imwrite('../landmarksHandMask.jpg', annotated_image)

resize_and_show(output_image)
extract_regions(image, thresholded_mask, landmarks)
