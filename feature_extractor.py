from object_detection.fingers2 import process_image
import os
from tqdm import tqdm  # import tqdm

if __name__ == "__main__":
    DIR_PATH = "dataset/hands/swolen/"
    MASK_OUTPUT_DIR = "results/SegMasks/"
    FINGER_OUTPUT_DIR = "results/FingerPics/"
    NAIL_OUTPUT_DIR = "results/nails/"

    # Make sure the output directory exists
    if not os.path.exists(MASK_OUTPUT_DIR):
        os.makedirs(MASK_OUTPUT_DIR)

    if not os.path.exists(FINGER_OUTPUT_DIR):
        os.makedirs(FINGER_OUTPUT_DIR)
    
    if not os.path.exists(NAIL_OUTPUT_DIR):
        os.makedirs(NAIL_OUTPUT_DIR)

    # Filter image names and wrap with tqdm for a progress bar
    image_names = [img for img in os.listdir(DIR_PATH) if img.endswith(('.jpg', '.jpeg', '.png'))]
    pip_features = []
    dip_features = []
    for image_name in tqdm(image_names, desc="Processing images"):
        image_path = os.path.join(DIR_PATH, image_name)
        pip_feature, dip_feature = process_image(image_path, MASK_OUTPUT_DIR, FINGER_OUTPUT_DIR, NAIL_OUTPUT_DIR)
        pip_features.append(pip_feature)
        dip_features.append(dip_feature)
    
import csv

output_csv_path = "results/features_pip.csv"
with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write header
    csvwriter.writerow(["Image", "PIP_Effective_Width_Index", "PIP_Effective_Width_Middle", "PIP_Effective_Width_Ring", "PIP_Effective_Width_Pinky"])
    
    # Write features
    for image_name, pip_feature, dip_feature in zip(image_names, pip_features, dip_features):
        csvwriter.writerow([image_name, pip_feature[0], pip_feature[1], dip_feature[0], dip_feature[1]])


output_csv_path = "results/features_dip.csv"
with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Write header
    csvwriter.writerow(["Image", "DIP_Effective_Width_Index", "DIP_Effective_Width_Middle", "DIP_Effective_Width_Ring", "DIP_Effective_Width_Pinky"])
    
    # Write features
    for image_name, pip_feature, dip_feature in zip(image_names, pip_features, dip_features):
        csvwriter.writerow([image_name, pip_feature[0], pip_feature[1], dip_feature[0], dip_feature[1]])
