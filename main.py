from object_detection.fingers2 import process_image
import os
if __name__ == "__main__":
    DIR_PATH = "dataset/hands/swolen/"
    OUTPUT_DIR = "results/SegMasks/"

    # Make sure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for image_name in os.listdir(DIR_PATH):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):  # checking file extension
            image_path = os.path.join(DIR_PATH, image_name)
            process_image(image_path, OUTPUT_DIR)