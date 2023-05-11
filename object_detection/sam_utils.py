import matplotlib.pyplot as plt
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def show_annotations(anns, axes=None):
    """
    Displays annotations on an axes.

    Args:
        anns: A list of annotations.
        axes: The axes object to display the annotations on.
            If not provided, the current axes will be used.

    Returns:
        None
    """

    if len(anns) == 0:
        # If there are no annotations, return early
        return

    if axes:
        ax = axes
    else:
        # If axes is not provided, use the current axes
        ax = plt.gca()
        ax.set_autoscale_on(False)

    # Sort the annotations by area in descending order
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    for ann in sorted_anns:
        # Retrieve the segmentation mask
        m = ann['segmentation']

        # Create an image array of ones with dimensions matching the mask
        img = np.ones((m.shape[0], m.shape[1], 3))

        # Generate a random color mask
        color_mask = np.random.random((1, 3)).tolist()[0]

        # Apply the color to the image array
        for i in range(3):
            img[:, :, i] = color_mask[i]

        # Display the image array with the mask applied
        ax.imshow(np.dstack((img, m * 0.5)))


def show_mask(mask, ax, random_color=False):
    """
    Displays a mask on an axes.

    Args:
        mask (numpy.ndarray): The mask to display.
        ax (matplotlib.axes.Axes): The axes object to display the mask on.
        random_color (bool, optional): If True, a random color will be used for the mask.
            If False, a default color will be used. Default is False.

    Returns:
        None
    """

    if random_color:
        # Generate a random color with an alpha channel
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Use a default color
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]

    # Reshape the mask and color arrays
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Display the mask image on the provided axes
    cv2.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """
    Displays points with labels on an axes.

    Args:
        coords (numpy.ndarray): The coordinates of the points.
        labels (numpy.ndarray): The labels associated with each point.
        ax (matplotlib.axes.Axes): The axes object to display the points on.
        marker_size (int, optional): The size of the markers. Default is 375.

    Returns:
        None
    """

    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    # Display positive points as green markers
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color='green',
        marker='*',
        s=marker_size,
        edgecolor='white',
        linewidth=1.25
    )

    # Display negative points as red markers
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color='red',
        marker='*',
        s=marker_size,
        edgecolor='white',
        linewidth=1.25
    )


def show_box(box, ax):
    """
    Displays a bounding box on an axes.

    Args:
        box: The bounding box coordinates [x_min, y_min, x_max, y_max].
        ax: The axes object to display the bounding box on.

    Returns:
        None
    """

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]

    # Add a rectangle patch to the axes representing the bounding box
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            w,
            h,
            edgecolor='green',
            facecolor=(0, 0, 0, 0),
            lw=2
        )
    )

img = cv2.imread('../hand7.jpg', cv2.IMREAD_UNCHANGED)
# resized_img = resize_image(img, 720)
sam = sam_model_registry["vit_h"](checkpoint="../sam_vit_h_4b8939.pth")
predictor = SamAutomaticMaskGenerator(sam)
masks = predictor.generate(img)
img[~masks[0]["segmentation"], :] = [0, 0, 0]
img[masks[0]["segmentation"], :] = [255, 255, 255]

import pickle

file = open('../masks7.p', 'wb')
pickle.dump(masks, file)
file.close()