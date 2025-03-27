import typing
from PIL import Image
import cv2
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion
import moviepy.editor as mpy
import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional
from segmentation.u2net import detect
from segmentation.net import Net

# closes https://github.com/nadermx/backgroundremover/issues/18
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


class BackgroundRemover:
    def __init__(self, threshold: int = 11, model_name="u2net", device="cpu"):
        self.threshold = threshold
        self.model = self.get_model(model_name)
        self.device = device

    def alpha_matting_cutout(self, img, mask, foreground_threshold, background_threshold, erode_structure_size, base_size):
        size = img.size

        img.thumbnail((base_size, base_size), Image.LANCZOS)
        mask = mask.resize(img.size, Image.LANCZOS)

        img = np.asarray(img)
        mask = np.asarray(mask)

        is_foreground = mask > foreground_threshold
        is_background = mask < background_threshold

        structure = None
        if erode_structure_size > 0:
            structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int64)

        is_foreground = binary_erosion(is_foreground, structure=structure)
        is_background = binary_erosion(is_background, structure=structure, border_value=1)

        trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
        trimap[is_foreground] = 255
        trimap[is_background] = 0

        img_normalized = img / 255.0
        trimap_normalized = trimap / 255.0

        alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
        foreground = estimate_foreground_ml(img_normalized, alpha)
        cutout = stack_images(foreground, alpha)

        cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
        cutout = Image.fromarray(cutout)
        cutout = cutout.resize(size, Image.LANCZOS)

        return cutout

    def naive_cutout(self, img, mask):
        empty = Image.new("RGB", img.size, 0)
        cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
        return cutout

    def get_model(self, model_name):
        if model_name == "u2netp":
            return detect.load_model(model_name="u2netp")
        if model_name == "u2net_human_seg":
            return detect.load_model(model_name="u2net_human_seg")
        else:
            return detect.load_model(model_name="u2net")

    def remove(self, data, alpha_matting=False, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_structure_size=10, alpha_matting_base_size=1000):
        img = Image.fromarray(data.astype('uint8'), 'RGB')
        mask = detect.predict(self.model, np.array(img)).convert("L")

        if alpha_matting:
            cutout = self.alpha_matting_cutout(
                img,
                mask,
                alpha_matting_foreground_threshold,
                alpha_matting_background_threshold,
                alpha_matting_erode_structure_size,
                alpha_matting_base_size,
            )
        else:
            cutout = self.naive_cutout(img, mask)

        return np.array(cutout)

    def iter_frames(self, path):
        return mpy.VideoFileClip(path).resize(height=320).iter_frames(dtype="uint8")

    @torch.no_grad()
    def remove_many(self, image_data: typing.List[np.array], net: Net):
        image_data = np.stack(image_data)
        image_data = torch.as_tensor(image_data, dtype=torch.float32, device=self.device)
        return net(image_data).numpy()


    def get_segmentation_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generates a segmentation mask for an image based on a grayscale threshold.

        This function converts an RGB image to grayscale and then creates a binary mask
        where pixels above a certain threshold are marked as foreground (255) and the rest as background (0).

        Args:
            image (np.ndarray): The RGB image from which to generate the mask.
            threshold (int, optional): The grayscale threshold for foreground-background segmentation. Defaults to 11.

        Returns:
            np.ndarray: A binary mask of the same size as the input image.

        Raises:
            ValueError: If the input image is not a 3-channel RGB image.

        Test Case:
            >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            >>> mask = get_segmentation_mask(image)
            >>> mask.shape
            (100, 100)
            >>> np.unique(mask)
            array([  0, 255], dtype=uint8)  # Only two values should be present in the mask: 0 and 255
        """

        result = self.remove(image)

        # Check if the image has three channels
        if len(result.shape) != 3 or result.shape[2] != 3:
            raise ValueError("Expected an RGB image with 3 channels. Received image with shape {}.".format(image.shape))

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        # Generate the mask
        mask = np.where(grayscale_image > self.threshold, 255, 0)

        return mask.astype(np.uint8)