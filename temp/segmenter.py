import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class Segmenter:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(base_options=base_options)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def segment_image(self, image_path):
        image = mp.Image.create_from_file(image_path)
        return self.segmenter.segment(image)[0].numpy_view()
