import io
import json
import os
import math
import pickle
import warnings

from PIL.Image import Image
from rembg import remove, new_session
from utils import *
from landmarks_constants import *
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pixel_finder import find_bounding_box, crop_image

my_session = new_session("u2net_human_seg")


def load_and_annotate_image(hand_path, detector_path):
    img = cv2.imread(hand_path, cv2.COLOR_RGBA2RGB)
    detector = load_hand_landmarker(detector_path)
    landmarks = locate_hand_landmarks(hand_path, detector)
    annotated_image = draw_landmarks_on_image(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=img).numpy_view(),
        landmarks
    )
    cv2.imwrite('../landmarksHand.jpg', annotated_image)
    return img, landmarks


def resize_and_show(image, result_path):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imwrite(result_path, img)
    return img


def remove_bg_and_whiten(image_path):
    # 1. Open the RGBA image using OpenCV
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)  # Read with alpha channel
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
    equalized_img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)

    # 2. Remove image background (assuming you've a function 'remove_background' to do this)
    img_without_bg = remove(equalized_img, session=my_session)
    if img_without_bg.shape[2] == 4:
        img_without_bg = img_without_bg[:, :, :3]
    # 3. Iterate through each pixel in the image, and if it's not black, turn it white
    mask = np.all(img_without_bg != [0, 0, 0], axis=-1)

    output_image = img_without_bg.copy()
    # output_image[mask] = [255, 255, 255]

    # 4. Return the modified MxNx3 image
    return output_image, mask


def process_mask(segmentation_mask, image_data):
    thresholded_mask = np.where(segmentation_mask > 0.1, 1, 0)
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((thresholded_mask,) * 3, axis=-1)
    output_image = np.where(condition, fg_image, bg_image)
    return output_image, thresholded_mask


def extract_regions(img, thresholded_mask, landmarks, original_filename, which_hand):
    landmarks = landmarks.hand_landmarks[0]
    for idx, area in enumerate(areas_of_interest):
        top_left, bottom_right = find_bounding_box(thresholded_mask, landmarks, area, which_hand)
        if top_left[0] == bottom_right[0] or top_left[1] == bottom_right[1]:
            return

        if original_filename == "hand75.jpg":
            lol = 1
        extracted_image = crop_image(img, top_left, bottom_right)
        extracted_image = resize_image(extracted_image, 244)

        # Use original filename and area index to generate a unique result filename
        result_filename = f"{original_filename.rsplit('.', 1)[0]}_area{idx}.jpg"
        cv2.imwrite(os.path.join("../results", result_filename), extracted_image)


def preprocess_image(image):
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Histogram Equalization
    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    image = cv2.merge((y_eq, cr, cb))
    image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)

    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def save_landmarks(landmarks, save_path):
    """
    Save HandLandmarkResult to a pickle file.

    Parameters:
        landmarks: HandLandmarkResult, containing landmark points data
        save_path: str, path to where the pickle file will be saved
    """
    with open(save_path, 'wb') as f:
        pickle.dump(landmarks, f)


BG_COLOR = (0, 0, 0)  # gray
MASK_COLOR = (255, 255, 255)  # white
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Specify the path where the images are stored
HANDS_FOLDER_PATH = '../dataset/hands/swolen'
SKIN_CLASS = 2
# Get a list of all image filenames in the folder
image_filenames = os.listdir(HANDS_FOLDER_PATH)

# Loop through each image filename
for image_filename in image_filenames:
    print(image_filename)
    # Construct the full path to the image
    HAND_PATH = os.path.join(HANDS_FOLDER_PATH, image_filename)

    # Execute your code
    image, landmarks = load_and_annotate_image(HAND_PATH, '../hand_landmarker.task')
    # image = preprocess_image(image)
    if not landmarks.hand_landmarks:
        wstr = f"HANDMARKS NOT LOCATED FOR IMAGE " + image_filename
        warnings.warn(wstr)
        continue

    which_hand = landmarks.handedness[0][0].category_name

    mp_image = mp.Image.create_from_file(HAND_PATH)
    output_image, thresholded_mask = remove_bg_and_whiten(HAND_PATH)
    # output_image, thresholded_mask = process_mask(segmentation_mask, image)

    annotated_image = draw_landmarks_on_image(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=output_image).numpy_view(),
        landmarks
    )

    # Specify a path to save the result, use the original image_filename to generate a result filename
    result_path = os.path.join('../results/SegMasks', f'segmask_{image_filename}')

    resize_and_show(output_image, result_path)

    save_landmarks(landmarks, f'../results/Landmarks/landmarks_{image_filename[:5]}.pkl')

    extract_regions(image, thresholded_mask, landmarks, image_filename, which_hand)
