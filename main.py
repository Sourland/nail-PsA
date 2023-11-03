from object_detection.fingers2 import process_image
import os
if __name__ == "__main__":
    DIR_PATH = "dataset/hands/swolen/"
    MASK_OUTPUT_DIR = "results/LandmarkPics/"
    FINGER_OUTPUT_DIR = "results/FingerPics/"
    NAIL_OUTPUT_DIR = "results/nails/"

    # Make sure the output directory exists
    if not os.path.exists(MASK_OUTPUT_DIR):
        os.makedirs(MASK_OUTPUT_DIR)

    if not os.path.exists(FINGER_OUTPUT_DIR):
        os.makedirs(FINGER_OUTPUT_DIR)
    
    if not os.path.exists(NAIL_OUTPUT_DIR):
        os.makedirs(NAIL_OUTPUT_DIR)

    for image_name in os.listdir(DIR_PATH):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):  # checking file extension
            image_path = os.path.join(DIR_PATH, image_name)
            process_image(image_path, MASK_OUTPUT_DIR, FINGER_OUTPUT_DIR, NAIL_OUTPUT_DIR)