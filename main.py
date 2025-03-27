import argparse

import cv2
from classification import model
from landmark_extraction import hand_landmarker as handL
from landmark_extraction.utils.landmarks_constants import *
from nail_image_extraction.region_extractor import crop_image, find_bounding_box
from segmentation import bg  
from biometric_extraction import hand_biometric_extractor as hbe

def nailpsa(image_path):
    segmentor = bg.BackgroundRemover()
    hand_landmarker = handL.HandLandmarks("hand_landmarker.task")
    analyzer = hbe.HandBiometricExtractor(segmentor=segmentor, hand_landmarker=hand_landmarker)
    model = model.NailPsoriasisPredictor()

    landmark_result, image = hand_landmarker.locate_hand_landmarks(image_path)
    seg_mask = segmentor.get_segmentation_mask(image)
    pip_feature, dip_feature = analyzer.process(seg_mask, landmark_result)

    handness = hand_landmarker.get_handness()
    nail_images = []
    for landmark in [INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]:
        top_left, bottom_right = find_bounding_box(image=seg_mask, landmarks=landmark_result, landmark_name=landmark, which_hand=handness)
        nail_image = crop_image(image=image, top_left=top_left, bottom_right=bottom_right)
        nail_image = cv2.cvtColor(nail_image, cv2.COLOR_BGR2RGB)
        nail_images.append(nail_image)

    

    print(f"Processing image: {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="nailpsa", description="Process an image")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    nailpsa(args.image_path)
