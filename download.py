from roboflow import Roboflow
import mediapipe as mp
import cv2 as cv
import json

rf = Roboflow(api_key="pO2frbDKGssTk8wmtQwy")
project = rf.workspace("knm").project("nail-disease-detection-mxoqy")
dataset = project.version(4).download("coco")


f = open("dataset/Nail-Disease-Detection-4/train/_annotations.coco.json")
nail_disease_detection_4 = json.load(f)


project = rf.workspace("rajarata-university-of-sri-lanka").project("nail-disease-detection-system")
nail_disease_detection_system_2 = project.version(2).download("coco")


project = rf.workspace("tugas-akhir-kx2jl").project("nail-disease-jmapt")
nail_disease_detection_dataset = project.version(1).download("coco")
i = 2