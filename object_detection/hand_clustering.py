import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data, segmentation, color

# Load a sample image from scikit-image
image = cv2.imread("results/SegMasks/seg_hand64.jpg", cv2.COLOR_RGBA2RGB)
# 1. Felzenszwalb's efficient graph-based segmentation
segments_fz = segmentation.felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")

# 2. SLIC superpixels
segments_slic = segmentation.slic(image, n_segments=6, compactness=10, sigma=1)
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")

# 3. Quickshift image segmentation
segments_quick = segmentation.quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

# 4. Watershed algorithm
gradient = segmentation.mark_boundaries(image, segments_fz)
segments_watershed = segmentation.watershed(gradient, markers=250, compactness=0.001)
print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")

# Display the results
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

# Display the results using cv2
cv2.imshow("Felzenszwalbs's method", cv2.cvtColor(color.label2rgb(segments_fz, image, kind='avg'), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.imshow('SLIC', cv2.cvtColor(color.label2rgb(segments_slic, image, kind='avg'), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.imshow('Quickshift', cv2.cvtColor(color.label2rgb(segments_quick, image, kind='avg'), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.imshow('Compact Watershed', cv2.cvtColor(color.label2rgb(segments_watershed, image, kind='avg'), cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()
