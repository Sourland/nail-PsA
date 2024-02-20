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


def load_and_annotate_image(hand_path: str, detector_path: str) -> tuple:
    """
    Loads an image from a given path and annotates it with hand landmarks.

    Args:
        hand_path (str): Path to the hand image file.
        detector_path (str): Path to the hand landmarker model.

    Returns:
        tuple: A tuple containing the loaded image and the hand landmarks.

    Test Case:
        # Assuming 'hand.jpg' and 'hand_landmarker.model' are valid paths.
        >>> img, landmarks = load_and_annotate_image('hand.jpg', 'hand_landmarker.model')
        >>> type(img), type(landmarks)
        (<class 'numpy.ndarray'>, <class 'mediapipe.tasks.python.vision.HandLandmarkerResult'>)
    """

    img = cv2.imread(hand_path, cv2.COLOR_RGBA2RGB)
    detector = load_hand_landmarker(detector_path)
    landmarks = locate_hand_landmarks(hand_path, detector)
    annotated_image = draw_landmarks_on_image(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=img).numpy_view(),
        landmarks
    )
    cv2.imwrite('../landmarksHand.jpg', annotated_image)
    return img, landmarks


def resize_and_show(image: np.ndarray, result_path: str) -> np.ndarray:
    """
    Resizes an image to a predetermined width or height while maintaining aspect ratio and saves it.

    Args:
        image (np.ndarray): The image to be resized.
        result_path (str): Path to save the resized image.

    Returns:
        np.ndarray: The resized image.

    Test Case:
        >>> image = np.zeros((100, 200, 3), dtype=np.uint8)
        >>> resized_img = resize_and_show(image, 'resized.jpg')
        >>> resized_img.shape
        (240, 480, 3)  # Shape will vary based on DESIRED_WIDTH and DESIRED_HEIGHT
    """
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imwrite(result_path, img)
    return img


def remove_bg_and_whiten(image_path: str) -> np.ndarray:
    """
    Removes the background from an image and whitens the non-black pixels.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The image with the background removed and non-black pixels whitened.

    Test Case:
        # Assuming 'image_with_bg.jpg' is a valid image path.
        >>> img_without_bg = remove_bg_and_whiten('image_with_bg.jpg')
        >>> type(img_without_bg)
        <class 'numpy.ndarray'>
        >>> img_without_bg.shape[2]
        3  # The returned image should have 3 channels
    """

    # 1. Open the RGBA image using OpenCV
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)  # Read with alpha channel
    # img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
    # equalized_img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)

    # 2. Remove image background (assuming you've a function 'remove_background' to do this)
    img_without_bg = remove(img, session=my_session)
    if img_without_bg.shape[2] == 4:
        img_without_bg = img_without_bg[:, :, :3]
    # 3. Iterate through each pixel in the image, and if it's not black, turn it white

    # output_image[mask] = [255, 255, 255]

    # 4. Return the modified MxNx3 image
    return img_without_bg


def segment_image(image: np.ndarray, model_path: str) -> np.ndarray:
    """
    Segments an image using a specified model.

    Args:
        image (np.ndarray): The image to be segmented.
        model_path (str): Path to the segmentation model.

    Returns:
        np.ndarray: The segmentation mask of the image.

    Test Case:
        # Assuming 'image.jpg' and 'segmentation_model.tflite' are valid paths.
        >>> img = cv2.imread('image.jpg')
        >>> mask = segment_image(img, 'segmentation_model.tflite')
        >>> type(mask)
        <class 'numpy.ndarray'>
        >>> mask.shape
        (height, width)  # Shape will vary based on the input image size
    """

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_confidence_masks=True)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        segmentation_result = segmenter.segment(image)
        segmentation_mask = segmentation_result.confidence_masks[SKIN_CLASS].numpy_view()
    return segmentation_mask


def process_mask(segmentation_mask: np.ndarray, image_data: np.ndarray) -> tuple:
    """
    Processes a segmentation mask to create foreground and background images.

    Args:
        segmentation_mask (np.ndarray): The segmentation mask.
        image_data (np.ndarray): The original image data.

    Returns:
        tuple: A tuple containing the processed image and the thresholded mask.

    Test Case:
        >>> segmentation_mask = np.random.rand(100, 100)
        >>> image_data = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> processed_img, thresholded_mask = process_mask(segmentation_mask, image_data)
        >>> processed_img.shape, thresholded_mask.shape
        ((100, 100, 3), (100, 100))
    """

    thresholded_mask = np.where(segmentation_mask > 0.1, 1, 0)
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((thresholded_mask,) * 3, axis=-1)
    output_image = np.where(condition, fg_image, bg_image)
    return output_image, thresholded_mask


def extract_regions(img: np.ndarray, thresholded_mask: np.ndarray, landmarks: object, original_filename: str, which_hand: str):
    """
    Extracts specific regions from an image based on thresholded mask and landmarks.

    Args:
        img (np.ndarray): The original image.
        thresholded_mask (np.ndarray): The thresholded mask of the image.
        landmarks (object): The landmarks object.
        original_filename (str): The filename of the original image.
        which_hand (str): Specifies the hand ('Right' or 'Left').

    Returns:
        None

    Test Case:
        # Requires a specific setup with image, mask, landmarks, filename, and hand orientation for a practical test.
    """
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


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Applies preprocessing steps to an image including bilateral filtering and histogram equalization.

    Args:
        image (np.ndarray): The image to be preprocessed.

    Returns:
        np.ndarray: The preprocessed image.

    Test Case:
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> preprocessed_img = preprocess_image(image)
        >>> preprocessed_img.shape
        (100, 100, 3)
    """
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


def save_landmarks(landmarks: object, save_path: str):
    """
    Saves HandLandmarkResult to a pickle file.

    Args:
        landmarks (object): HandLandmarkResult, containing landmark points data.
        save_path (str): Path to where the pickle file will be saved.

    Returns:
        None

    Test Case:
        # Requires a HandLandmarkResult object and a valid save path for a practical test.
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

    output_image = remove_bg_and_whiten(HAND_PATH)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=output_image.astype(np.uint8))
    segmentation_mask = segment_image(mp_image, '../selfie_multiclass_256x256.tflite')
    output_image, thresholded_mask = process_mask(segmentation_mask, image)

    annotated_image = draw_landmarks_on_image(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=output_image).numpy_view(),
        landmarks
    )

    # Specify a path to save the result, use the original image_filename to generate a result filename
    result_path = os.path.join('../results/SegMasks', f'segmask_{image_filename}')

    resize_and_show(output_image, result_path)

    extract_regions(image, thresholded_mask, landmarks, image_filename, which_hand)
