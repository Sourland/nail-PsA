import cv2
import numpy as np

def get_segmentation_mask(image: np.ndarray, threshold: int = 11) -> np.ndarray:
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
# Check if the image has three channels
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels. Received image with shape {}.".format(image.shape))

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Generate the mask
    mask = np.where(grayscale_image > threshold, 255, 0)

    return mask.astype(np.uint8)