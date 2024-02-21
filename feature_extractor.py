import random
from object_detection.finger_width import process_hand_image
import os
from tqdm import tqdm  # import tqdm
import csv
from object_detection.hand_biometric_analyzer import HandBiometricAnalyzer
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

    
    image_names = [img for img in os.listdir(DIR_PATH) if img.endswith(('.jpg', '.jpeg', '.png'))]
    pip_features = []
    dip_features = []

    analyzer = HandBiometricAnalyzer(MASK_OUTPUT_DIR, FINGER_OUTPUT_DIR)
    for image_name in tqdm(image_names, desc="Processing images"):
        image_path = os.path.join(DIR_PATH, image_name)
        pip_feature, dip_feature = analyzer.process(image_path)
        pip_features.append(pip_feature)
        dip_features.append(dip_feature)


    output_csv_path = "results/features_pip_swollen.csv"
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header
        csvwriter.writerow(["Image", "PIP_Effective_Width_Index", "PIP_Effective_Width_Middle", "PIP_Effective_Width_Ring", "PIP_Effective_Width_Pinky"])
            
        # Write features
        for image_name, pip_feature in zip(image_names, pip_features):
            csvwriter.writerow([image_name, pip_feature[0], pip_feature[1], pip_feature[2], pip_feature[3]])


    output_csv_path = "results/features_dip_swollen.csv"
    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header
        csvwriter.writerow(["Image", "DIP_Effective_Width_Index", "DIP_Effective_Width_Middle", "DIP_Effective_Width_Ring", "DIP_Effective_Width_Pinky"])
        
        # Write features
        for image_name,dip_feature in zip(image_names, dip_features):
            csvwriter.writerow([image_name, dip_feature[0], dip_feature[1], dip_feature[2], dip_feature[3]])


    # # New directory with healthy hand images
    # HEALTHY_DIR_PATH = "dataset/hands/healthy/"
    # # Number of random images to select

    # # List all images in the new directory
    # all_image_names = [img for img in os.listdir(HEALTHY_DIR_PATH) if img.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
    # # Randomly select 60 images
    # # random_image_names = random.sample(all_image_names, NUM_IMAGES_TO_SELECT)

    # # Process each randomly selected image
    # pip_features = []
    # dip_features = []
    # for image_name in tqdm(all_image_names, desc="Processing images"):
    #     image_path = os.path.join(HEALTHY_DIR_PATH, image_name)
    #     pip_feature, dip_feature = process_image(image_path, MASK_OUTPUT_DIR, FINGER_OUTPUT_DIR, NAIL_OUTPUT_DIR)
    #     pip_features.append(pip_feature)
    #     dip_features.append(dip_feature)

    # # Assuming `process_image` returns two tuples, one for PIP and one for DIP, each containing features
    # # Save the features to CSV files
    # pip_output_csv_path = "results/features_pip_healthy.csv"
    # dip_output_csv_path = "results/features_dip_healthy.csv"

    # # Save PIP features
    # with open(pip_output_csv_path, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     # Write header
    #     csvwriter.writerow(["Image", "PIP_Effective_Width_Index", "PIP_Effective_Width_Middle", "PIP_Effective_Width_Ring", "PIP_Effective_Width_Pinky"])  # Adjust based on actual features returned
    #     # Write features
    #     for image_name, pip_feature in zip(all_image_names, pip_features):
    #         csvwriter.writerow([image_name] + list(pip_feature))

    # # Save DIP features
    # with open(dip_output_csv_path, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     # Write header
    #     csvwriter.writerow(["Image", "DIP_Effective_Width_Index", "DIP_Effective_Width_Middle", "DIP_Effective_Width_Ring", "DIP_Effective_Width_Pinky"])  # Adjust based on actual features returned
    #     # Write features
    #     for image_name, dip_feature in zip(all_image_names, dip_features):
    #         csvwriter.writerow([image_name] + list(dip_feature))

